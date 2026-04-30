#astar_planner.py
#!/usr/bin/env python3
"""
A* Path Planner
===============
Subscribes:
    /map           (nav_msgs/OccupancyGrid)    — occupancy grid map (stable)
    /planner_goal  (geometry_msgs/PoseStamped) — mission goal from navigator

Publishes:
  /waypoints     (nav_msgs/Path)  — pruned A* path (waypoints for navigator)
  /planned_path  (nav_msgs/Path)  — raw A* path    (for RViz)

All coordinates are in the 'map' frame, which equals the Gazebo world frame.
No scipy — obstacle inflation uses pure numpy roll-based dilation.
"""

import heapq
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import tf2_ros


# ---------------------------------------------------------------------------
# Numpy-only obstacle inflation (no scipy)
# ---------------------------------------------------------------------------

def inflate_obstacles(grid: np.ndarray, radius: int) -> np.ndarray:
    """
    Dilate True (obstacle) cells outward by `radius` cells using iterative
    np.roll. Equivalent to a square structuring element.
    Wrap-around artefacts from np.roll are corrected at the borders.
    """
    if radius <= 0:
        return grid.copy()
    out = grid.copy()
    for shift in range(1, radius + 1):
        out |= np.roll(grid,  shift, axis=1)
        out |= np.roll(grid, -shift, axis=1)
    tmp = out.copy()
    for shift in range(1, radius + 1):
        out |= np.roll(tmp,  shift, axis=0)
        out |= np.roll(tmp, -shift, axis=0)
    # Fix wrap-around edges
    out[:radius,  :] = grid[:radius,  :]
    out[-radius:, :] = grid[-radius:, :]
    out[:,  :radius] = grid[:,  :radius]
    out[:, -radius:] = grid[:, -radius:]
    return out


# ---------------------------------------------------------------------------
# A* (8-connected, Euclidean heuristic)
# ---------------------------------------------------------------------------

def astar(grid: np.ndarray, start: tuple, goal: tuple) -> list:
    rows, cols = grid.shape
    h = lambda r, c: np.hypot(goal[0] - r, goal[1] - c)

    heap = [(h(*start), 0.0, start[0], start[1])]
    came = {}
    g    = {start: 0.0}

    MOVES = [(-1,0,1.),( 1,0,1.),(0,-1,1.),(0,1,1.),
             (-1,-1,1.414),(-1,1,1.414),(1,-1,1.414),(1,1,1.414)]

    while heap:
        _, gc, r, c = heapq.heappop(heap)
        if (r, c) == goal:
            path = []
            cur  = goal
            while cur in came:
                path.append(cur); cur = came[cur]
            path.append(start); path.reverse()
            return path
        for dr, dc, cost in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols): continue
            if grid[nr, nc]: continue

            # Prevent diagonal corner-cutting between two blocked orthogonal cells.
            if dr != 0 and dc != 0:
                if grid[r + dr, c] or grid[r, c + dc]:
                    continue

            ng = gc + cost
            if ng < g.get((nr, nc), float('inf')):
                g[(nr, nc)] = ng
                came[(nr, nc)] = (r, c)
                heapq.heappush(heap, (ng + h(nr, nc), ng, nr, nc))
    return []


# ---------------------------------------------------------------------------
# Bresenham line-of-sight
# ---------------------------------------------------------------------------

def los(grid: np.ndarray, r0, c0, r1, c1) -> bool:
    dr = abs(r1-r0); sr = 1 if r1>r0 else -1
    dc = abs(c1-c0); sc = 1 if c1>c0 else -1
    err = dr - dc; r, c = r0, c0
    while (r, c) != (r1, c1):
        if grid[r, c]: return False
        e2 = 2*err
        if e2 > -dc: err -= dc; r += sr
        if e2 <  dr: err += dr; c += sc
    return True


# ---------------------------------------------------------------------------
# String-pull (greedy LOS pruning)
# ---------------------------------------------------------------------------

def string_pull(path: list, grid: np.ndarray) -> list:
    if len(path) <= 2: return path
    pruned = [path[0]]; anchor = 0; i = 2
    while i < len(path):
        if not los(grid, *path[anchor], *path[i]):
            pruned.append(path[i-1]); anchor = i-1
        i += 1
    pruned.append(path[-1])
    return pruned


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')

        self.declare_parameter('inflation_radius',   4)
        self.declare_parameter('occupied_threshold', 65)
        self.declare_parameter('map_frame',         'map')
        self.declare_parameter('base_frame',   'base_link')

        self.inflation_r = self.get_parameter('inflation_radius').value
        self.occ_thresh  = self.get_parameter('occupied_threshold').value
        self.map_frame   = self.get_parameter('map_frame').value
        self.base_frame  = self.get_parameter('base_frame').value

        self.current_map   = None
        self.inflated_grid = None
        self.current_goal  = None

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(OccupancyGrid, '/map',          self._map_cb, 10)
        self.create_subscription(PoseStamped,   '/planner_goal', self._goal_cb, 10)

        self.path_pub = self.create_publisher(Path, '/waypoints',    10)
        self.raw_pub  = self.create_publisher(Path, '/planned_path', 10)

        self.get_logger().info(
            f'AStarPlanner ready  inflation={self.inflation_r} cells  '
            f'thresh={self.occ_thresh}  frame={self.map_frame}')

    # ---- Callbacks ----

    def _map_cb(self, msg: OccupancyGrid):
        self.current_map   = msg
        h = msg.info.height; w = msg.info.width
        flat = np.array(msg.data, dtype=np.int8).reshape(h, w)
        bool_grid = flat >= self.occ_thresh
        self.inflated_grid = inflate_obstacles(bool_grid, self.inflation_r)
        self.get_logger().info(
            f'Map ready: {w}x{h} cells  res={msg.info.resolution}m  '
            f'origin=({msg.info.origin.position.x:.2f},'
            f'{msg.info.origin.position.y:.2f})')
        if self.current_goal:
            self._plan()

    def _goal_cb(self, msg: PoseStamped):
        self.current_goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(
            f'Goal received: ({self.current_goal[0]:.2f}, {self.current_goal[1]:.2f})')
        self._plan()

    # ---- Helpers ----

    def _w2c(self, wx, wy, info):
        """World → (row, col)"""
        col = int((wx - info.origin.position.x) / info.resolution)
        row = int((wy - info.origin.position.y) / info.resolution)
        return row, col

    def _c2w(self, row, col, info):
        """Cell centre → world (x, y)"""
        wx = info.origin.position.x + (col + 0.5) * info.resolution
        wy = info.origin.position.y + (row + 0.5) * info.resolution
        return wx, wy

    def _clamp(self, row, col, rows, cols):
        return max(0, min(rows-1, row)), max(0, min(cols-1, col))

    def _nearest_free(self, grid, cell):
        visited = set(); q = deque([cell])
        while q:
            r, c = q.popleft()
            if (r, c) in visited: continue
            visited.add((r, c))
            if not grid[r, c]: return (r, c)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<grid.shape[0] and 0<=nc<grid.shape[1] and (nr,nc) not in visited:
                    q.append((nr, nc))
        return None

    def _get_robot_pose(self):
        """Look up robot pose in map frame (map = Gazebo world frame)."""
        # Try the configured frame first, then the namespaced frame used by Gazebo.
        frame_candidates = [self.base_frame, 'vehicle_blue/base_link']
        try:
            for frame in frame_candidates:
                try:
                    t = self._tf_buffer.lookup_transform(
                        self.map_frame, frame, rclpy.time.Time(),
                        timeout=Duration(seconds=0.2))
                    return t.transform.translation.x, t.transform.translation.y
                except Exception:
                    continue
        except Exception:
            pass

        self.get_logger().warn(
            f'TF lookup failed for {self.map_frame}->({", ".join(frame_candidates)})')
        return None

    # ---- Planning ----

    def _plan(self):
        if self.current_map is None or self.inflated_grid is None:
            self.get_logger().warn('No map yet'); return
        if self.current_goal is None: return

        robot = self._get_robot_pose()
        if robot is None: return

        info  = self.current_map.info
        rows, cols = info.height, info.width
        ig    = self.inflated_grid
        rx, ry = robot
        gx, gy = self.current_goal

        sc = self._clamp(*self._w2c(rx, ry, info), rows, cols)
        gc = self._clamp(*self._w2c(gx, gy, info), rows, cols)

        if ig[sc]:
            self.get_logger().warn('Start in obstacle after inflation — searching nearest free cell')
            sc = self._nearest_free(ig, sc)
            if sc is None:
                self.get_logger().error('No free cell near start')
                return
        if ig[gc]:
            self.get_logger().warn('Goal in obstacle — searching nearest free cell')
            gc = self._nearest_free(ig, gc)
            if gc is None:
                self.get_logger().error('No free cell near goal'); return

        cells = astar(ig, sc, gc)
        if not cells:
            self.get_logger().warn(f'A* no path {sc} → {gc}'); return

        pruned = string_pull(cells, ig)
        self.get_logger().info(
            f'Path: {len(cells)} cells → {len(pruned)} waypoints')

        self.path_pub.publish(self._to_path(pruned, info))
        self.raw_pub.publish( self._to_path(cells,  info))

    def _to_path(self, cells, info) -> Path:
        p = Path()
        p.header = Header()
        p.header.stamp    = self.get_clock().now().to_msg()
        p.header.frame_id = self.map_frame
        for row, col in cells:
            wx, wy = self._c2w(row, col, info)
            ps = PoseStamped()
            ps.header = p.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            p.poses.append(ps)
        return p


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
