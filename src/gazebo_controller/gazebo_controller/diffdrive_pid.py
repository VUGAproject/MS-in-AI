#!/usr/bin/env python3
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
import tf2_ros

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    # Clamp sinp to avoid domain errors in arcsin
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class DiffDrivePID(Node):
    """
    PID controller for differential drive robot using trailer hitch approach.
    Based on EN613 midterm solution.
    """
    def __init__(self):
        super().__init__('diffdrive_pid')

        # PID gains
        self.declare_parameter('kp', 0.8)
        self.declare_parameter('kd', 0.2)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('lookahead', 0.5)  # trailer hitch distance
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('max_linear_vel', 1.5)
        self.declare_parameter('max_angular_vel', 1.5)
        self.declare_parameter('angular_scale', 0.7)  # Scale down angular commands
        self.declare_parameter('goal_tolerance', 0.25)
        self.declare_parameter('heading_rotate_threshold', 1.2)
        self.declare_parameter('heading_slowdown_threshold', 0.45)
        self.declare_parameter('min_turn_speed_scale', 0.35)
        self.declare_parameter('base_frame', 'vehicle_blue/base_link')

        self.k_p = float(self.get_parameter('kp').get_parameter_value().double_value)
        self.k_d = float(self.get_parameter('kd').get_parameter_value().double_value)
        self.k_i = float(self.get_parameter('ki').get_parameter_value().double_value)
        self.length = float(self.get_parameter('lookahead').get_parameter_value().double_value)
        self.publish_rate = float(self.get_parameter('publish_rate').get_parameter_value().double_value)
        self.max_linear_vel = float(self.get_parameter('max_linear_vel').get_parameter_value().double_value)
        self.max_angular_vel = float(self.get_parameter('max_angular_vel').get_parameter_value().double_value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').get_parameter_value().double_value)
        self.heading_rotate_threshold = float(self.get_parameter('heading_rotate_threshold').get_parameter_value().double_value)
        self.heading_slowdown_threshold = float(self.get_parameter('heading_slowdown_threshold').get_parameter_value().double_value)
        self.min_turn_speed_scale = float(self.get_parameter('min_turn_speed_scale').get_parameter_value().double_value)
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.dt = 1.0 / self.publish_rate

        # I/O
        qos_profile = QoSProfile(depth=10)
        self.goal_sub = self.create_subscription(PoseStamped, '/planner_goal_pose', self.goal_received, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_received, 20)
        # Ground truth pose from Gazebo PosePublisher plugin (world frame, no drift).
        self.pose_sub = self.create_subscription(TFMessage, '/model/vehicle_blue/pose', self.true_pose_cb, 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile)
        self.trail_pub = self.create_publisher(Path, '/robot_trail', 10)

        # State
        self.desired_goal = np.array([0.0, 0.0])
        self.robot_state = np.array([0.0, 0.0, 0.0])
        self.Xp_last = None
        self.has_goal = False
        self.has_odom = False

        self.trail_path = Path()
        self.trail_path.header.frame_id = 'map'
        self.last_trail_xy = None

        # LiDAR gap-navigation state
        self._lidar_ranges = None
        self._lidar_angle_min = 0.0
        self._lidar_angle_inc = 0.1745
        self.lidar_sub = self.create_subscription(LaserScan, '/lidar', self.lidar_cb, 10)

        # Stuck recovery state
        self._recovering = False
        self._recovery_frames = 0
        self._recovery_direction = 1.0
        self._stuck_last_xy = None
        self._stuck_last_time = None
        self._recovery_total = int(3.0 * self.publish_rate)  # 3 s of recovery
        self._recovery_back_frames = int(1.0 * self.publish_rate)  # first 1 s: back up straight

        # Timer loop
        self.timer = self.create_timer(self.dt, self.publish_robot_cmd)

        self.get_logger().info(f'DiffDrivePID started: kp={self.k_p}, kd={self.k_d}, ki={self.k_i}, lookahead={self.length}, rate={self.publish_rate} Hz')

    def lidar_cb(self, msg: LaserScan):
        self._lidar_ranges = np.array(msg.ranges, dtype=float)
        self._lidar_angle_min = float(msg.angle_min)
        self._lidar_angle_inc = float(msg.angle_increment)

    def true_pose_cb(self, msg: TFMessage):
        """Receive Gazebo ground-truth world position.
        The transform frame_id='maze_world', child_frame_id='vehicle_blue'
        gives the vehicle's actual world coordinates."""
        for tf in msg.transforms:
            if tf.child_frame_id == 'vehicle_blue' and tf.header.frame_id == 'maze_world':
                t = tf.transform.translation
                _, _, yaw = euler_from_quaternion(tf.transform.rotation)
                self.robot_state = np.array([float(t.x), float(t.y), yaw])
                self.has_odom = True
                return

    def odom_received(self, msg):
        self.has_odom = True  # fallback signal only; true_pose_cb provides position

    def goal_received(self, msg):
        """Callback for goal pose messages"""
        # Check if goal is in the correct frame (odom)
        goal_frame = msg.header.frame_id

        # In this project launch, map and odom are intentionally aligned.
        if goal_frame == 'map':
            self.desired_goal = np.array([msg.pose.position.x, msg.pose.position.y])
            self.get_logger().info(f'Goal set to: [{self.desired_goal[0]:.3f}, {self.desired_goal[1]:.3f}] using map=odom alignment')
        elif goal_frame != 'odom' and goal_frame != '':
            self.get_logger().warn(f'Goal received in frame "{goal_frame}" but expecting "odom". Attempting to transform...')
            try:
                # Try to transform the goal to odom frame
                when = rclpy.time.Time()
                trans = self._tf_buffer.lookup_transform(
                    'odom', goal_frame,
                    when, timeout=Duration(seconds=1.0))
                
                # Get the goal position in the source frame
                goal_x = msg.pose.position.x
                goal_y = msg.pose.position.y
                
                # Get translation
                tx = trans.transform.translation.x
                ty = trans.transform.translation.y
                
                # Get rotation (convert quaternion to yaw)
                quat = trans.transform.rotation
                _, _, yaw = euler_from_quaternion(quat)
                
                # Apply full transformation: rotate then translate
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                
                goal_x_odom = cos_yaw * goal_x - sin_yaw * goal_y + tx
                goal_y_odom = sin_yaw * goal_x + cos_yaw * goal_y + ty
                
                self.desired_goal = np.array([goal_x_odom, goal_y_odom])
                self.get_logger().info(f'Goal transformed to odom frame: [{self.desired_goal[0]:.3f}, {self.desired_goal[1]:.3f}]')
            except Exception as e:
                self.get_logger().error(f'Failed to transform goal: {str(e)}')
                return
        else:
            # Goal is already in odom frame (or no frame specified, assume odom)
            self.desired_goal = np.array([msg.pose.position.x, msg.pose.position.y])
            self.get_logger().info(f'Goal set to: [{self.desired_goal[0]:.3f}, {self.desired_goal[1]:.3f}] in odom frame')
        
        # Reset the integral term when a new goal is set
        self.Xp_last = None
        self.has_goal = True

    def compute_vel(self):
        """Go-to-goal controller with LiDAR gap navigation and stuck-recovery fallback."""
        if not self.has_goal:
            self._stuck_last_time = None
            self._recovering = False
            return np.array([0.0, 0.0])

        x_r, y_r, th_r = self.robot_state
        cur_pos = np.array([x_r, y_r])

        dx = self.desired_goal[0] - x_r
        dy = self.desired_goal[1] - y_r
        dist = float(np.hypot(dx, dy))

        if dist < self.goal_tolerance:
            self.has_goal = False
            self._recovering = False
            self._stuck_last_time = None
            return np.array([0.0, 0.0])

        # ── Stuck recovery ──────────────────────────────────────────────────
        if self._recovering:
            self._recovery_frames += 1
            if self._recovery_frames < self._recovery_total:
                if self._recovery_frames < self._recovery_back_frames:
                    # Phase 1: back up straight to clear the wall
                    return np.array([-0.3, 0.0])
                else:
                    # Phase 2: turn in place toward goal
                    return np.array([0.0, self._recovery_direction * self.max_angular_vel])
            # Recovery done — reset and resume normal driving
            self._recovering = False
            self._stuck_last_xy = cur_pos.copy()
            self._stuck_last_time = time.monotonic()

        now = time.monotonic()
        if self._stuck_last_time is None:
            self._stuck_last_xy = cur_pos.copy()
            self._stuck_last_time = now
        elif now - self._stuck_last_time > 3.0:
            moved = float(np.linalg.norm(cur_pos - self._stuck_last_xy))
            if moved < 0.15:  # less than 15 cm in 3 s — stuck
                goal_angle = np.arctan2(dy, dx)
                err = normalize_angle(goal_angle - th_r)
                self._recovery_direction = 1.0 if err >= 0 else -1.0
                self._recovering = True
                self._recovery_frames = 0
                return np.array([-0.2, self._recovery_direction * self.max_angular_vel])
            else:
                self._stuck_last_xy = cur_pos.copy()
                self._stuck_last_time = now

        # ── LiDAR gap navigation ───────────────────────────────────────────
        goal_angle = np.arctan2(dy, dx)
        # Default: steer straight toward goal (angle is relative to robot heading)
        heading_to_goal = normalize_angle(goal_angle - th_r)
        steer = heading_to_goal

        if self._lidar_ranges is not None and len(self._lidar_ranges) > 0:
            n = len(self._lidar_ranges)
            # Replace self-detections (< 0.28 m) and invalid values with max range
            rng = np.where(
                np.isfinite(self._lidar_ranges) & (self._lidar_ranges > 0.28),
                self._lidar_ranges,
                30.0
            )
            # Count blocked rays in ±25° cone directly toward goal
            blocked = 0
            for i in range(n):
                ray_a = self._lidar_angle_min + i * self._lidar_angle_inc
                if abs(normalize_angle(ray_a - heading_to_goal)) < 0.44:  # ±25°
                    if rng[i] < 0.55:
                        blocked += 1

            if blocked > 0:
                # Path to goal is obstructed — find the best open gap.
                # Score each ray by free space weighted by closeness to goal direction.
                best_score = -1.0
                best_ray = heading_to_goal
                for i in range(n):
                    ray_a = self._lidar_angle_min + i * self._lidar_angle_inc
                    # Only look in the forward ±120° sector to avoid U-turns
                    if abs(ray_a) > 2.09:
                        continue
                    angular_dist = abs(normalize_angle(ray_a - heading_to_goal))
                    score = rng[i] * np.exp(-1.5 * angular_dist)
                    if score > best_score:
                        best_score = score
                        best_ray = ray_a
                steer = best_ray

        # ── Heading and velocity control ────────────────────────────────────
        heading_err = float(steer)   # already robot-frame relative angle
        abs_err = abs(heading_err)

        if abs_err > self.heading_rotate_threshold:
            az = float(np.clip(self.k_p * 2.0 * heading_err,
                               -self.max_angular_vel, self.max_angular_vel))
            return np.array([0.0, az])

        lx = float(np.clip(self.k_p * dist, 0.0, self.max_linear_vel))
        if abs_err > self.heading_slowdown_threshold:
            lx *= max(self.min_turn_speed_scale, 1.0 - abs_err)

        az = float(np.clip(self.k_p * 2.0 * heading_err,
                           -self.max_angular_vel, self.max_angular_vel))
        return np.array([lx, az])

    def publish_robot_cmd(self):
        """Main control loop callback"""
        if not self.has_odom:
            return  # wait for first ground-truth pose

        # Append world position to trail for RViz visualization.
        cur_xy = (self.robot_state[0], self.robot_state[1])
        if self.last_trail_xy is None or np.linalg.norm(
                np.array(cur_xy) - np.array(self.last_trail_xy)) > 0.05:
            p = PoseStamped()
            p.header.stamp = self.get_clock().now().to_msg()
            p.header.frame_id = 'map'
            p.pose.position.x = float(cur_xy[0])
            p.pose.position.y = float(cur_xy[1])
            p.pose.position.z = 0.0
            p.pose.orientation.w = 1.0
            self.trail_path.header.stamp = p.header.stamp
            self.trail_path.poses.append(p)
            if len(self.trail_path.poses) > 2000:
                self.trail_path.poses = self.trail_path.poses[-2000:]
            self.trail_pub.publish(self.trail_path)
            self.last_trail_xy = cur_xy

        desired_vel = self.compute_vel()

        msg = Twist()
        msg.linear.x = float(desired_vel[0])
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(desired_vel[1])

        self.vel_pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = DiffDrivePID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
