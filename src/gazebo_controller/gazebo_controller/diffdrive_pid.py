#!/usr/bin/env python3
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Empty
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
        self.declare_parameter('heading_rotate_threshold', 2.3)
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
        self.skip_wp_pub = self.create_publisher(Empty, '/skip_waypoint', 10)

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
        self._stuck_last_heading = 0.0
        self._stuck_count = 0          # consecutive stuck events; triggers waypoint skip
        self._struggling_since = None  # tracks how long robot is making poor progress
        self._reverse_until_clear = False  # immediate escape: back up until forward is open
        self._recovery_total = int(1.5 * self.publish_rate)  # 1.5 s total recovery
        self._recovery_back_frames = int(0.5 * self.publish_rate)  # first 0.5 s: back up straight

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
            self._reverse_until_clear = False
            self._stuck_last_time = None
            self._stuck_count = 0
            self._struggling_since = None
            return np.array([0.0, 0.0])

        # ── Immediate wall-hit reversal ────────────────────────────────────────
        # If any ray in the forward ±60° arc is closer than 0.22 m the robot is
        # essentially touching a wall.  Back up until the whole forward arc is
        # clear (> 0.50 m) before resuming normal navigation.
        if self._lidar_ranges is not None and len(self._lidar_ranges) > 0:
            n_w = len(self._lidar_ranges)
            fwd_min = 30.0
            for i in range(n_w):
                ray_a = self._lidar_angle_min + i * self._lidar_angle_inc
                if abs(ray_a) < 1.047:  # ±60° forward
                    r = float(self._lidar_ranges[i])
                    if np.isfinite(r) and r > 0.28:
                        fwd_min = min(fwd_min, r)
            if fwd_min < 0.22:
                self._reverse_until_clear = True
            if self._reverse_until_clear:
                if fwd_min > 0.50:
                    self._reverse_until_clear = False  # enough space — resume
                else:
                    return np.array([-0.3, 0.0])  # keep reversing

        # ── Stuck recovery ──────────────────────────────────────────────────
        if self._recovering:
            self._recovery_frames += 1
            if self._recovery_frames < self._recovery_total:
                if self._recovery_frames < self._recovery_back_frames:
                    # Phase 1: back up straight to create separation from the wall
                    return np.array([-0.3, 0.0])
                else:
                    # Phase 2: gentle forward arc toward goal
                    goal_angle_rec = np.arctan2(
                        self.desired_goal[1] - self.robot_state[1],
                        self.desired_goal[0] - self.robot_state[0])
                    err_rec = normalize_angle(goal_angle_rec - self.robot_state[2])
                    az_rec = float(np.clip(self.k_p * 2.0 * err_rec,
                                          -self.max_angular_vel, self.max_angular_vel))
                    return np.array([0.3, az_rec])
            # Recovery done — reset and resume normal driving
            self._recovering = False
            self._stuck_last_xy = cur_pos.copy()
            self._stuck_last_time = time.monotonic()

        now = time.monotonic()
        if self._stuck_last_time is None:
            self._stuck_last_xy = cur_pos.copy()
            self._stuck_last_time = now
            self._stuck_last_heading = th_r
        elif now - self._stuck_last_time > 3.0:
            moved = float(np.linalg.norm(cur_pos - self._stuck_last_xy))
            turned = abs(normalize_angle(th_r - self._stuck_last_heading))
            # Only declare stuck if robot barely moved AND barely turned.
            # A robot arcing through a corner has large heading change — not stuck.
            if moved < 0.15 and turned < 0.5:
                # Recovery direction: prefer the side with more LiDAR clearance.
                recovery_dir = 1.0  # default: turn left
                if self._lidar_ranges is not None:
                    nl = len(self._lidar_ranges)
                    left_sum, right_sum = 0.0, 0.0
                    for i in range(nl):
                        ray_a = self._lidar_angle_min + i * self._lidar_angle_inc
                        r = float(self._lidar_ranges[i]) if np.isfinite(self._lidar_ranges[i]) else 0.0
                        if 0.17 < ray_a < 1.57:   # left arc  ~10°–90°
                            left_sum += r
                        elif -1.57 < ray_a < -0.17:  # right arc ~10°–90°
                            right_sum += r
                    if right_sum > left_sum:
                        recovery_dir = -1.0
                else:
                    # Fall back to goal-heading direction if no LiDAR data yet
                    goal_angle = np.arctan2(dy, dx)
                    err = normalize_angle(goal_angle - th_r)
                    recovery_dir = 1.0 if err >= 0 else -1.0
                self._recovery_direction = recovery_dir
                self._stuck_count += 1
                # After 2 consecutive stuck events (~6+ s), skip to next waypoint
                if self._stuck_count >= 2:
                    self.get_logger().warn(
                        f'Stuck {self._stuck_count} times consecutively — requesting waypoint skip')
                    self.skip_wp_pub.publish(Empty())
                    self._stuck_count = 0
                self._recovering = True
                self._recovery_frames = 0
                return np.array([-0.2, self._recovery_direction * self.max_angular_vel])
            else:
                self._stuck_last_xy = cur_pos.copy()
                self._stuck_last_time = now
                self._stuck_last_heading = th_r
                self._stuck_count = 0          # real progress — reset counter
                self._struggling_since = None  # made real progress, reset escalation

        # ── Track struggling (slow movement for > 5 s → widen perception) ────
        speed_proxy = float(np.linalg.norm(cur_pos - (self._stuck_last_xy if self._stuck_last_xy is not None else cur_pos)))
        if speed_proxy < 0.05:  # essentially stationary right now
            if self._struggling_since is None:
                self._struggling_since = time.monotonic()
        else:
            self._struggling_since = None
        struggling = (self._struggling_since is not None and
                      time.monotonic() - self._struggling_since > 5.0)

        # ── LiDAR gap navigation ───────────────────────────────────────────
        goal_angle = np.arctan2(dy, dx)
        heading_to_goal = normalize_angle(goal_angle - th_r)  # robot-frame angle to goal
        steer = heading_to_goal

        if self._lidar_ranges is not None and len(self._lidar_ranges) > 0:
            n = len(self._lidar_ranges)
            # Blanket self-filter: suppress anything < 0.28 m (own chassis/wheels)
            rng = np.where(
                np.isfinite(self._lidar_ranges) & (self._lidar_ranges > 0.28),
                self._lidar_ranges,
                30.0
            )

            # When struggling > 5 s: lower clearance bar and widen cone so the
            # robot "sees" tight or diagonal paths it was previously ignoring.
            MIN_CLEAR = 0.35 if struggling else 0.55
            blocked_cone = 1.047 if struggling else 0.785  # ±60° escalated, ±45° normal
            open_mask = rng > MIN_CLEAR

            # Check whether the direct path to goal is blocked
            direct_blocked = any(
                rng[i] < MIN_CLEAR
                for i in range(n)
                if abs(normalize_angle(
                    (self._lidar_angle_min + i * self._lidar_angle_inc) - heading_to_goal
                )) < blocked_cone
            )

            if direct_blocked:
                # Find contiguous open-ray segments in the forward ±150° sector
                segments = []
                j = 0
                while j < n:
                    ray_a = self._lidar_angle_min + j * self._lidar_angle_inc
                    if open_mask[j] and abs(ray_a) <= 2.618:
                        start = j
                        while j < n and open_mask[j] and abs(
                                self._lidar_angle_min + j * self._lidar_angle_inc) <= 2.618:
                            j += 1
                        segments.append((start, j - 1))
                    else:
                        j += 1

                if segments:
                    # Pick the best segment and steer toward its CENTER,
                    # then BLEND 60% gap-center + 40% goal to give a natural
                    # margin — robot doesn't need to be perfectly centered.
                    best_score = -1.0
                    best_center = heading_to_goal
                    for start, end in segments:
                        mid_i = (start + end) / 2.0
                        center_angle = self._lidar_angle_min + mid_i * self._lidar_angle_inc
                        angular_dist = abs(normalize_angle(center_angle - heading_to_goal))
                        gap_width = end - start + 1
                        score = float(gap_width) * np.exp(-0.7 * angular_dist)
                        if score > best_score:
                            best_score = score
                            best_center = center_angle
                    # Blend: 95% gap center, 5% goal — strongly follow corridor geometry
                    blended = 0.95 * best_center + 0.05 * heading_to_goal
                    steer = normalize_angle(blended)

        # ── Heading and velocity control ────────────────────────────────────
        heading_err = float(normalize_angle(steer))
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
