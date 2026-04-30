#navigator.py

#!/usr/bin/env python3
"""
Autonomous Navigator
====================
Subscribes:
  /goal_points  (visualization_msgs/MarkerArray) — static goal spheres
  /waypoints    (nav_msgs/Path)                  — A* path from planner

Publishes:
  /planner_goal    (geometry_msgs/PoseStamped) → astar_planner
  /controller_goal (geometry_msgs/PoseStamped) → diffdrive_pid
  /navigator_status (std_msgs/String)          — human-readable progress

All poses use frame_id='map' (= Gazebo world frame).
Robot pose is read via TF map → vehicle_blue/base_link.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import tf2_ros


class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')

        self.declare_parameter('goal_tolerance',     0.40)
        self.declare_parameter('waypoint_tolerance', 0.35)
        self.declare_parameter('stuck_timeout',     10.0)
        self.declare_parameter('loop_rate',         10.0)
        self.declare_parameter('send_goal_interval', 0.3)
        self.declare_parameter('status_interval',    1.0)
        self.declare_parameter('map_frame',         'map')
        self.declare_parameter('base_frame',   'base_link')

        self.goal_tol  = self.get_parameter('goal_tolerance').value
        self.wp_tol    = self.get_parameter('waypoint_tolerance').value
        self.stuck_to  = self.get_parameter('stuck_timeout').value
        self.send_int  = self.get_parameter('send_goal_interval').value
        self.status_int = self.get_parameter('status_interval').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        rate           = self.get_parameter('loop_rate').value

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(MarkerArray, '/goal_points', self._goals_cb, 10)
        self.create_subscription(Path,        '/waypoints',   self._waypts_cb, 10)

        self.planner_pub    = self.create_publisher(PoseStamped, '/planner_goal',    10)
        self.controller_pub = self.create_publisher(PoseStamped, '/controller_goal', 10)
        self.status_pub     = self.create_publisher(String,      '/navigator_status', 10)

        self.all_goals      = []
        self.remaining      = []
        self.current        = None    # (x, y, name)
        self.waypoints      = []      # [(x, y)]
        self.wp_idx         = 0
        self.goals_ready    = False

        self.last_pos       = None
        self.last_prog_t    = None
        self.last_send_t    = 0.0
        self.last_status_t  = 0.0

        self.create_timer(1.0 / rate, self._loop)
        self.get_logger().info('Navigator ready — waiting for /goal_points')

    # ---- helpers ----

    def _robot_pose(self):
        frame_candidates = [self.base_frame, 'vehicle_blue/base_link']
        for frame in frame_candidates:
            try:
                t = self._tf_buffer.lookup_transform(
                    self.map_frame, frame, rclpy.time.Time(),
                    timeout=Duration(seconds=0.15))
                return t.transform.translation.x, t.transform.translation.y
            except Exception:
                continue
        return None

    @staticmethod
    def _d(ax, ay, bx, by): return float(np.hypot(ax-bx, ay-by))

    def _pub_planner(self, x, y):
        ps = PoseStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = 'map'
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.w = 1.0
        self.planner_pub.publish(ps)

    def _pub_ctrl(self, x, y):
        ps = PoseStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = 'map'
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.w = 1.0
        self.controller_pub.publish(ps)

    def _log(self, text):
        msg = String(); msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    # ---- callbacks ----

    def _goals_cb(self, msg: MarkerArray):
        if self.goals_ready: return
        goals = [(m.pose.position.x, m.pose.position.y,
                  f'{m.ns}_{m.id}') for m in msg.markers]
        if not goals: return
        self.all_goals   = goals
        self.remaining   = list(goals)
        self.goals_ready = True
        names = ', '.join(f'{n}({x:.2f},{y:.2f})' for x,y,n in goals)
        self._log(f'Goals received: {names}')
        self._next_goal()

    def _waypts_cb(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn('Empty waypoint path'); return
        self.waypoints = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.wp_idx    = 0
        self.get_logger().info(f'New path: {len(self.waypoints)} waypoints')

    # ---- state machine ----

    def _next_goal(self):
        if not self.remaining:
            self._log('ALL GOALS REACHED'); self.current = None; return
        pose = self._robot_pose()
        if pose is None:
            self.get_logger().warn('Robot pose unavailable, retrying goal selection')
            return
        rx, ry = pose
        self.remaining.sort(key=lambda g: self._d(rx, ry, g[0], g[1]))
        self.current      = self.remaining[0]
        self.waypoints    = []
        self.wp_idx       = 0
        self.last_prog_t  = self.get_clock().now().nanoseconds * 1e-9
        self.last_pos     = (rx, ry)
        gx, gy, gn = self.current
        self._log(f'Targeting {gn} ({gx:.2f}, {gy:.2f})')
        self._pub_planner(gx, gy)

    def _loop(self):
        if not self.goals_ready or self.current is None: return
        pose = self._robot_pose()
        if pose is None: return
        rx, ry = pose
        gx, gy, gn = self.current
        now = self.get_clock().now().nanoseconds * 1e-9

        # Goal reached?
        if self._d(rx, ry, gx, gy) < self.goal_tol:
            self._log(f'REACHED {gn}')
            self.remaining = [g for g in self.remaining if g[2] != gn]
            self._next_goal(); return

        # Waiting for plan
        if not self.waypoints:
            if now - self.last_send_t > self.send_int:
                self._pub_planner(gx, gy); self.last_send_t = now
            return

        # Advance waypoint index
        while self.wp_idx < len(self.waypoints) - 1:
            wx, wy = self.waypoints[self.wp_idx]
            if self._d(rx, ry, wx, wy) < self.wp_tol:
                self.wp_idx += 1
            else:
                break

        tx, ty = self.waypoints[self.wp_idx]
        if now - self.last_send_t > self.send_int:
            self._pub_ctrl(tx, ty); self.last_send_t = now

        # Stuck detection
        if self.last_pos and self._d(rx, ry, *self.last_pos) > 0.05:
            self.last_prog_t = now; self.last_pos = (rx, ry)
        if self.last_prog_t and now - self.last_prog_t > self.stuck_to:
            self.get_logger().warn('Stuck — replanning')
            self.waypoints = []; self.wp_idx = 0
            self.last_prog_t = now
            self._pub_planner(gx, gy); return

        if now - self.last_status_t > self.status_int:
            self._log(
                f'{gn}: dist={self._d(rx,ry,gx,gy):.2f}m  '
                f'wp={self.wp_idx+1}/{len(self.waypoints)}  '
                f'wp_dist={self._d(rx,ry,tx,ty):.2f}m')
            self.last_status_t = now


def main(args=None):
    rclpy.init(args=args)
    node = Navigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
