#!/usr/bin/env python3
import heapq
import math
import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Header, Empty
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')

        self.declare_parameter('goal_reach_tolerance', 0.35)
        self.declare_parameter('waypoint_stride_cells', 5)
        self.declare_parameter('obstacle_inflation_cells', 3)
        self.goal_reach_tolerance = float(self.get_parameter('goal_reach_tolerance').value)
        self.waypoint_stride_cells = int(self.get_parameter('waypoint_stride_cells').value)
        self.obstacle_inflation_cells = int(self.get_parameter('obstacle_inflation_cells').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_msg: Optional[OccupancyGrid] = None
        self.occ_grid: Optional[np.ndarray] = None
        self.dist_map: Optional[np.ndarray] = None  # metres to nearest obstacle, per free cell
        self.robot_map_xy: Optional[Tuple[float, float]] = None
        self.all_goals: List[Tuple[float, float]] = []
        self.remaining_goals: List[Tuple[float, float]] = []
        self.failed_goals = set()

        self.active_goal: Optional[Tuple[float, float]] = None
        self.pending_waypoints: List[Tuple[float, float]] = []
        self.active_waypoint_idx = -1
        self._last_replan_time: float = 0.0

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self.pose_sub = self.create_subscription(TFMessage, '/model/vehicle_blue/pose', self.true_pose_cb, 10)
        self.goals_sub = self.create_subscription(MarkerArray, '/goal_points', self.goals_cb, 10)

        self.goal_pub = self.create_publisher(PoseStamped, '/planner_goal_pose', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.skip_wp_sub = self.create_subscription(Empty, '/skip_waypoint', self.skip_waypoint_cb, 10)

        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info('AStar planner ready: waiting for /map, /odom, and /goal_points')

    def skip_waypoint_cb(self, _msg: Empty):
        """Advance to the next waypoint when the controller signals it is stuck."""
        if not self.pending_waypoints or self.active_waypoint_idx < 0:
            return
        next_idx = self.active_waypoint_idx + 1
        if next_idx < len(self.pending_waypoints):
            self.active_waypoint_idx = next_idx
            wp = self.pending_waypoints[self.active_waypoint_idx]
            self.get_logger().warn(
                f'Skipping to waypoint {next_idx}/{len(self.pending_waypoints)-1}: {wp}')
            self.publish_goal_pose(wp)
        else:
            # Already at last waypoint; force goal reached so planner moves on
            self.get_logger().warn('Skip requested at last waypoint — marking goal reached')
            reached = self.active_goal
            self.active_goal = None
            self.pending_waypoints = []
            self.active_waypoint_idx = -1
            if reached and reached in self.remaining_goals:
                self.remaining_goals.remove(reached)

    def map_cb(self, msg: OccupancyGrid):
        self.map_msg = msg
        h = int(msg.info.height)
        w = int(msg.info.width)
        if h == 0 or w == 0:
            return
        grid = np.array(msg.data, dtype=np.int16).reshape((h, w))
        self.occ_grid = self.inflate_obstacles(grid, self.obstacle_inflation_cells)
        # Distance transform: for every free cell, compute metres to nearest obstacle.
        # Used by A* to penalise paths that pass close to walls.
        free_mask = (self.occ_grid < 50)
        self.dist_map = distance_transform_edt(free_mask) * float(msg.info.resolution)

    def true_pose_cb(self, msg: TFMessage):
        """Ground-truth world position. Uses maze_world→vehicle_blue transform."""
        for tf in msg.transforms:
            if tf.child_frame_id == 'vehicle_blue' and tf.header.frame_id == 'maze_world':
                self.robot_map_xy = (
                    float(tf.transform.translation.x),
                    float(tf.transform.translation.y),
                )
                return

    def odom_cb(self, msg: Odometry):
        # Only used as fallback if ground-truth topic not yet received.
        if self.robot_map_xy is None:
            self.robot_map_xy = (
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
            )

    def goals_cb(self, msg: MarkerArray):
        goals = []
        for marker in msg.markers:
            goals.append((float(marker.pose.position.x), float(marker.pose.position.y)))

        if not goals:
            return

        if sorted(goals) != sorted(self.all_goals):
            self.all_goals = goals
            self.remaining_goals = goals.copy()
            self.failed_goals.clear()
            self.active_goal = None
            self.pending_waypoints = []
            self.active_waypoint_idx = -1
            self.get_logger().info(f'Received {len(goals)} goal points')

    def tick(self):
        if self.map_msg is None or self.occ_grid is None or self.robot_map_xy is None:
            return
        if not self.remaining_goals and self.active_goal is None:
            return

        # If we currently follow waypoints, check progress and continue publishing.
        if self.pending_waypoints and self.active_waypoint_idx >= 0:
            wp = self.pending_waypoints[self.active_waypoint_idx]
            dist_to_wp = self.distance(self.robot_map_xy, wp)

            # Off-path detection: robot drifted too far from current waypoint.
            # This happens after gap-nav maneuvers push the robot off the A* path
            # and the remaining waypoints end up behind/beside the robot.
            # Force a fresh replan to the same goal from the current position.
            _now = time.monotonic()
            if (dist_to_wp > 0.8
                    and self.active_goal is not None
                    and _now - self._last_replan_time > 2.0):
                self.get_logger().warn(
                    f'Off-path: {dist_to_wp:.2f} m from waypoint — replanning to {self.active_goal}')
                self.pending_waypoints = []
                self.active_waypoint_idx = -1
                self._last_replan_time = _now
                return

            if dist_to_wp <= self.goal_reach_tolerance:
                if self.active_waypoint_idx < len(self.pending_waypoints) - 1:
                    self.active_waypoint_idx += 1
                    self.publish_goal_pose(self.pending_waypoints[self.active_waypoint_idx])
                    return

                # Last waypoint reached -> goal reached.
                reached = self.active_goal
                self.active_goal = None
                self.pending_waypoints = []
                self.active_waypoint_idx = -1
                if reached in self.remaining_goals:
                    self.remaining_goals.remove(reached)
                    self.get_logger().info(f'Reached goal {reached}; {len(self.remaining_goals)} goals remaining')
                return

            # Keep current waypoint active.
            self.publish_goal_pose(wp)
            return

        # Need a new plan to the next goal.
        next_goal = self.select_next_goal()
        if next_goal is None:
            return

        start_rc = self.world_to_grid(self.robot_map_xy[0], self.robot_map_xy[1])
        goal_rc = self.world_to_grid(next_goal[0], next_goal[1])
        if start_rc is None or goal_rc is None:
            self.get_logger().warn('Start or goal is outside map bounds; skipping goal')
            self.failed_goals.add(next_goal)
            return

        start_rc = self.find_nearest_free(start_rc)
        goal_rc = self.find_nearest_free(goal_rc)
        if start_rc is None or goal_rc is None:
            self.get_logger().warn(f'No nearby free cell for start/goal toward {next_goal}; trying others')
            self.failed_goals.add(next_goal)
            return

        path_rc = self.a_star(start_rc, goal_rc, self.dist_map)
        if not path_rc:
            self.get_logger().warn(f'No path found to goal {next_goal}; trying others')
            self.failed_goals.add(next_goal)
            return

        waypoints = self.path_to_waypoints(path_rc)
        if not waypoints:
            self.failed_goals.add(next_goal)
            return

        self.active_goal = next_goal
        self.pending_waypoints = waypoints
        self.active_waypoint_idx = 0
        self.publish_path(path_rc)
        self.publish_goal_pose(self.pending_waypoints[self.active_waypoint_idx])
        self.get_logger().info(
            f'Planned to goal {next_goal} with {len(path_rc)} cells and {len(waypoints)} waypoints'
        )

    def inflate_obstacles(self, grid: np.ndarray, radius_cells: int) -> np.ndarray:
        h, w = grid.shape
        occ = (grid >= 50) | (grid < 0)
        inflated = occ.copy()
        if radius_cells <= 0:
            return np.where(inflated, 100, 0).astype(np.int16)

        occupied_idx = np.argwhere(occ)
        for r, c in occupied_idx:
            r0 = max(0, r - radius_cells)
            r1 = min(h, r + radius_cells + 1)
            c0 = max(0, c - radius_cells)
            c1 = min(w, c + radius_cells + 1)
            inflated[r0:r1, c0:c1] = True

        return np.where(inflated, 100, 0).astype(np.int16)

    def select_next_goal(self) -> Optional[Tuple[float, float]]:
        candidates = [g for g in self.remaining_goals if g not in self.failed_goals]
        if not candidates:
            return None
        return min(candidates, key=lambda g: self.distance(self.robot_map_xy, g))

    def world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        assert self.map_msg is not None
        info = self.map_msg.info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        col = int((x - ox) / res)
        row = int((y - oy) / res)
        if row < 0 or col < 0 or row >= int(info.height) or col >= int(info.width):
            return None
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        assert self.map_msg is not None
        info = self.map_msg.info
        x = info.origin.position.x + (col + 0.5) * info.resolution
        y = info.origin.position.y + (row + 0.5) * info.resolution
        return x, y

    def path_to_waypoints(self, path_rc: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        if not path_rc:
            return []
        stride = max(1, self.waypoint_stride_cells)
        sampled = path_rc[::stride]
        # Do not command the current start cell as a waypoint; it can produce near-zero cmd_vel.
        if sampled and sampled[0] == path_rc[0] and len(path_rc) > 1:
            sampled = sampled[1:]
        if not sampled:
            sampled = [path_rc[-1]]
        if sampled[-1] != path_rc[-1]:
            sampled.append(path_rc[-1])

        raw_count = len(sampled)

        # Greedy string-pulling: skip any intermediate waypoint that is visible
        # in a straight line from the previous kept waypoint (using the inflated
        # occupancy grid so the robot's width is respected).  Always keep the
        # final goal cell.
        pruned = self._string_pull(sampled)

        self.get_logger().info(
            f'Path pruning: {raw_count} raw waypoints → {len(pruned)} after string-pull')

        return [self.grid_to_world(r, c) for r, c in pruned]

    def _string_pull(self, cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Greedy visibility string-pull over the inflated occupancy grid."""
        if len(cells) <= 2:
            return cells
        result = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            # Scan backward from the end: find the farthest cell reachable in
            # a straight unobstructed line from the last kept cell.
            j = len(cells) - 1
            while j > i + 1:
                if self._bresenham_clear(result[-1], cells[j]):
                    break
                j -= 1
            result.append(cells[j])
            i = j
        return result

    def _bresenham_clear(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Return True if every cell on the Bresenham line a→b is free (< 50)."""
        assert self.occ_grid is not None
        grid = self.occ_grid
        h, w = grid.shape
        r0, c0 = a
        r1, c1 = b
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            if r < 0 or r >= h or c < 0 or c >= w:
                return False
            if grid[r, c] >= 50:
                return False
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return True

    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int],
               dist_map: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        assert self.occ_grid is not None
        grid = self.occ_grid
        h, w = grid.shape
        # Exponential wall-proximity penalty: cells near obstacles cost more so A*
        # routes through corridor centres.  Unlike the old 1/dist cap, the exponential
        # preserves a gradient even in narrow corridors (where capped values were
        # identical for edge vs centre cells, letting A* hug walls freely).
        WALL_WEIGHT = 2.0   # penalty magnitude at the inflated-wall boundary
        WALL_SIGMA  = 0.25  # falloff distance in metres (≈5 cells at 0.05 m/cell)

        if grid[start[0], start[1]] >= 50 or grid[goal[0], goal[1]] >= 50:
            return []

        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            cr, cc = current
            for dr, dc, move_cost in neighbors:
                nr = cr + dr
                nc = cc + dc
                if nr < 0 or nc < 0 or nr >= h or nc >= w:
                    continue
                if grid[nr, nc] >= 50:
                    continue

                # Prevent corner-cutting through obstacles on diagonal moves.
                if dr != 0 and dc != 0:
                    if grid[cr + dr, cc] >= 50 or grid[cr, cc + dc] >= 50:
                        continue

                nxt = (nr, nc)
                # Exponential wall-proximity penalty.  At the inflated-wall boundary
                # (dist=0) penalty = WALL_WEIGHT; decays smoothly toward 0 in open
                # space.  This creates a consistent gradient toward corridor centres
                # in passages of any width, unlike the old capped 1/dist formula.
                wall_penalty = (WALL_WEIGHT * math.exp(-dist_map[nr, nc] / WALL_SIGMA)
                                if dist_map is not None else 0.0)
                tentative = g_score[current] + move_cost + wall_penalty
                if tentative < g_score.get(nxt, float('inf')):
                    came_from[nxt] = current
                    g_score[nxt] = tentative
                    f = tentative + self.heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f, nxt))

        return []

    def find_nearest_free(self, rc: Tuple[int, int], max_radius: int = 12) -> Optional[Tuple[int, int]]:
        assert self.occ_grid is not None
        r0, c0 = rc
        h, w = self.occ_grid.shape
        if 0 <= r0 < h and 0 <= c0 < w and self.occ_grid[r0, c0] < 50:
            return (r0, c0)

        for radius in range(1, max_radius + 1):
            r_min = max(0, r0 - radius)
            r_max = min(h - 1, r0 + radius)
            c_min = max(0, c0 - radius)
            c_max = min(w - 1, c0 + radius)
            best = None
            best_d = float('inf')
            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    if self.occ_grid[r, c] >= 50:
                        continue
                    d = (r - r0) * (r - r0) + (c - c0) * (c - c0)
                    if d < best_d:
                        best_d = d
                        best = (r, c)
            if best is not None:
                return best

        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def distance(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
        return math.hypot(p0[0] - p1[0], p0[1] - p1[1])

    def publish_goal_pose(self, xy: Tuple[float, float]):
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(xy[0])
        msg.pose.position.y = float(xy[1])
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def publish_path(self, path_rc: List[Tuple[int, int]]):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'

        for r, c in path_rc:
            x, y = self.grid_to_world(r, c)
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.orientation.w = 1.0
            path.poses.append(p)

        self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()