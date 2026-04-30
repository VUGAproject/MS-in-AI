#lidar_mapper.py

#!/usr/bin/env python3
"""
LiDAR Mapper Node
Builds a live occupancy grid map from 2D LiDAR scans using ray-casting.
Publishes /live_map (nav_msgs/OccupancyGrid) at 2 Hz.

The map uses log-odds updates so cells accumulate evidence over time:
  - A LiDAR ray end-point marks the cell as OCCUPIED (log-odds += HIT)
  - Every cell along the ray before the hit is marked FREE (log-odds += MISS)
  - Unknown cells stay at 0 log-odds
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import tf2_ros
from rclpy.duration import Duration


def euler_from_quaternion(q):
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


class LidarMapper(Node):
    """
    Incremental occupancy grid mapper using log-odds updates.

    Parameters
    ----------
    resolution   : meters per cell (default 0.10 m)
    map_width_m  : total map width  in meters (default 30.0 m)
    map_height_m : total map height in meters (default 30.0 m)
    origin_x     : world x of map bottom-left corner (default -15.0)
    origin_y     : world y of map bottom-left corner (default -15.0)
    log_odds_hit : log-odds increment for an occupied hit  (default  1.5)
    log_odds_miss: log-odds increment for a free ray step  (default -0.5)
    log_odds_max : clamp ceiling for log-odds              (default  10.0)
    log_odds_min : clamp floor  for log-odds               (default -10.0)
    publish_rate : map publish frequency in Hz             (default  2.0)
    """

    def __init__(self):
        super().__init__('lidar_mapper')

        # --- parameters ---
        self.declare_parameter('resolution',    0.10)
        self.declare_parameter('map_width_m',  30.0)
        self.declare_parameter('map_height_m', 30.0)
        self.declare_parameter('origin_x',    -15.0)
        self.declare_parameter('origin_y',    -15.0)
        self.declare_parameter('log_odds_hit',   1.5)
        self.declare_parameter('log_odds_miss', -0.5)
        self.declare_parameter('log_odds_max',  10.0)
        self.declare_parameter('log_odds_min', -10.0)
        self.declare_parameter('publish_rate',   2.0)

        self.res      = self.get_parameter('resolution').value
        self.w_m      = self.get_parameter('map_width_m').value
        self.h_m      = self.get_parameter('map_height_m').value
        self.orig_x   = self.get_parameter('origin_x').value
        self.orig_y   = self.get_parameter('origin_y').value
        self.lo_hit   = self.get_parameter('log_odds_hit').value
        self.lo_miss  = self.get_parameter('log_odds_miss').value
        self.lo_max   = self.get_parameter('log_odds_max').value
        self.lo_min   = self.get_parameter('log_odds_min').value
        rate          = self.get_parameter('publish_rate').value

        self.cols = int(self.w_m / self.res)
        self.rows = int(self.h_m / self.res)

        # log-odds grid — float32, initialised to 0 (unknown)
        self.log_odds = np.zeros((self.rows, self.cols), dtype=np.float32)

        # TF
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Subscriber — LiDAR with BEST_EFFORT matching Gazebo bridge QoS
        lidar_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/lidar', self._scan_cb, lidar_qos)

        # Publisher
        self.map_pub = self.create_publisher(OccupancyGrid, '/live_map', 10)
        self.timer   = self.create_timer(1.0 / rate, self._publish_map)

        self.get_logger().info(
            f'LidarMapper ready  grid={self.cols}x{self.rows}  '
            f'res={self.res}m  origin=({self.orig_x},{self.orig_y})'
        )

    # ------------------------------------------------------------------
    # helpers: world ↔ grid coordinate conversion
    # ------------------------------------------------------------------
    def _world_to_grid(self, wx, wy):
        col = int((wx - self.orig_x) / self.res)
        row = int((wy - self.orig_y) / self.res)
        return col, row

    def _in_bounds(self, col, row):
        return 0 <= col < self.cols and 0 <= row < self.rows

    # ------------------------------------------------------------------
    # Bresenham ray-casting
    # ------------------------------------------------------------------
    def _bresenham(self, c0, r0, c1, r1):
        """Yield (col, row) cells from (c0,r0) exclusive to (c1,r1) inclusive."""
        dc = abs(c1 - c0);  sc = 1 if c1 > c0 else -1
        dr = -abs(r1 - r0); sr = 1 if r1 > r0 else -1
        err = dc + dr
        c, r = c0, r0
        while True:
            if c == c1 and r == r1:
                yield c, r
                break
            e2 = 2 * err
            if e2 >= dr:
                err += dr; c += sc
            if e2 <= dc:
                err += dc; r += sr
            yield c, r

    # ------------------------------------------------------------------
    # LiDAR callback
    # ------------------------------------------------------------------
    def _scan_cb(self, msg: LaserScan):
        # Get robot pose from TF
        try:
            t = self._tf_buffer.lookup_transform(
                'odom', 'base_link',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
        except Exception:
            return

        tx = t.transform.translation.x
        ty = t.transform.translation.y
        _, _, yaw = euler_from_quaternion(t.transform.rotation)

        robot_col, robot_row = self._world_to_grid(tx, ty)

        angle = msg.angle_min
        for r in msg.ranges:
            angle += msg.angle_increment
            if not np.isfinite(r) or r < msg.range_min or r > msg.range_max:
                continue

            # World-frame end-point of the ray
            ex = tx + r * np.cos(yaw + angle)
            ey = ty + r * np.sin(yaw + angle)
            end_col, end_row = self._world_to_grid(ex, ey)

            # Trace the ray: free cells along the ray, occupied at the endpoint
            for col, row in self._bresenham(robot_col, robot_row, end_col, end_row):
                if not self._in_bounds(col, row):
                    break
                if col == end_col and row == end_row:
                    # Mark as occupied (only if ray terminates before max range)
                    if r < msg.range_max - 0.05:
                        self.log_odds[row, col] = np.clip(
                            self.log_odds[row, col] + self.lo_hit,
                            self.lo_min, self.lo_max)
                else:
                    # Mark as free
                    self.log_odds[row, col] = np.clip(
                        self.log_odds[row, col] + self.lo_miss,
                        self.lo_min, self.lo_max)

    # ------------------------------------------------------------------
    # Convert log-odds → ROS OccupancyGrid and publish
    # ------------------------------------------------------------------
    def _publish_map(self):
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        msg.info.resolution = self.res
        msg.info.width  = self.cols
        msg.info.height = self.rows
        msg.info.origin.position.x = self.orig_x
        msg.info.origin.position.y = self.orig_y
        msg.info.origin.orientation.w = 1.0

        # Convert log-odds to {-1=unknown, 0=free, 100=occupied}
        grid = np.full((self.rows, self.cols), -1, dtype=np.int8)
        grid[self.log_odds >  0.5] = 100   # occupied
        grid[self.log_odds < -0.5] = 0     # free

        msg.data = grid.flatten().tolist()
        self.map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
