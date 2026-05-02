#!/usr/bin/env python3
import numpy as np
from time import sleep

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
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

        # TF - used to look up map→base_link which gives true world position
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # I/O
        qos_profile = QoSProfile(depth=10)
        self.goal_sub = self.create_subscription(PoseStamped, '/planner_goal_pose', self.goal_received, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_received, 20)
        self.scan_sub = self.create_subscription(LaserScan, '/lidar', self.scan_received, 10)
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

        # LiDAR reactive avoidance
        self.scan_ranges = None
        self.scan_angle_min = 0.0
        self.scan_angle_increment = 0.0
        self.LIDAR_DANGER_DIST = 0.30   # metres — obstacle closer than this triggers avoidance
        self.LIDAR_FRONT_HALF_ANGLE = 0.52  # radians (~30 deg each side of forward)

        # Timer loop
        self.timer = self.create_timer(self.dt, self.publish_robot_cmd)

        self.get_logger().info(f'DiffDrivePID started: kp={self.k_p}, kd={self.k_d}, ki={self.k_i}, lookahead={self.length}, rate={self.publish_rate} Hz')

    def scan_received(self, msg: LaserScan):
        self.scan_ranges = np.array(msg.ranges, dtype=float)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_increment = msg.angle_increment

    def lidar_obstacle_ahead(self):
        """Returns (blocked, turn_sign) where turn_sign=+1 means turn left, -1 means turn right."""
        if self.scan_ranges is None:
            return False, 0
        r = self.scan_ranges
        n = len(r)
        angle_min = self.scan_angle_min
        inc = self.scan_angle_increment
        half = self.LIDAR_FRONT_HALF_ANGLE
        danger = self.LIDAR_DANGER_DIST

        # Split front sector into left half and right half
        left_blocked = False
        right_blocked = False
        for i, dist in enumerate(r):
            if not np.isfinite(dist) or dist <= 0.01:
                continue
            angle = angle_min + i * inc
            if abs(angle) > half:
                continue
            if dist < danger:
                if angle >= 0:
                    left_blocked = True
                else:
                    right_blocked = True

        blocked = left_blocked or right_blocked
        if blocked:
            # Turn away from the more blocked side
            turn_sign = -1 if left_blocked else 1
            if left_blocked and right_blocked:
                turn_sign = -1  # default: turn right
        else:
            turn_sign = 0
        return blocked, turn_sign

    def odom_received(self, msg):
        pose = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(orientation)
        self.robot_state = np.array([pose.x, pose.y, yaw])
        self.has_odom = True

        # Append position to trail for RViz visualization.
        cur_xy = (pose.x, pose.y)
        if self.last_trail_xy is None or np.linalg.norm(np.array(cur_xy) - np.array(self.last_trail_xy)) > 0.05:
            p = PoseStamped()
            p.header.stamp = self.get_clock().now().to_msg()
            p.header.frame_id = 'map'
            p.pose.position.x = float(pose.x)
            p.pose.position.y = float(pose.y)
            p.pose.position.z = 0.0
            p.pose.orientation.w = 1.0
            self.trail_path.header.stamp = p.header.stamp
            self.trail_path.poses.append(p)
            if len(self.trail_path.poses) > 2000:
                self.trail_path.poses = self.trail_path.poses[-2000:]
            self.trail_pub.publish(self.trail_path)
            self.last_trail_xy = cur_xy

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
        """Go-to-goal proportional controller.

        Computes (linear_x, angular_z) to steer the robot toward self.desired_goal.
        Uses proportional control on bearing and distance. Never commands reverse.
        Turns in place when heading error is large before moving forward.
        """
        if not self.has_goal:
            return np.array([0.0, 0.0])

        x_r = self.robot_state[0]
        y_r = self.robot_state[1]
        th_r = self.robot_state[2]

        dx = self.desired_goal[0] - x_r
        dy = self.desired_goal[1] - y_r
        dist = float(np.sqrt(dx * dx + dy * dy))

        if dist < self.goal_tolerance:
            self.has_goal = False
            self.Xp_last = None
            return np.array([0.0, 0.0])

        goal_angle = np.arctan2(dy, dx)
        heading_err = normalize_angle(goal_angle - th_r)
        abs_err = abs(heading_err)

        # Turn in place when significantly misaligned before driving forward.
        if abs_err > self.heading_rotate_threshold:
            az = float(np.clip(self.k_p * 2.0 * heading_err,
                               -self.max_angular_vel, self.max_angular_vel))
            return np.array([0.0, az])

        # Drive forward; slow down when still correcting heading.
        lx = float(np.clip(self.k_p * dist, 0.0, self.max_linear_vel))
        if abs_err > self.heading_slowdown_threshold:
            lx *= max(self.min_turn_speed_scale, 1.0 - abs_err)

        az = float(np.clip(self.k_p * 2.0 * heading_err,
                           -self.max_angular_vel, self.max_angular_vel))

        self.Xp_last = None  # not used by this controller
        return np.array([lx, az])

    def publish_robot_cmd(self):
        """Main control loop callback"""
        # Prefer TF map→base_link which, with the spawn-encoded map→odom static TF,
        # gives the robot's true world position. Fall back to raw /odom if TF not ready.
        try:
            trans = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), timeout=Duration(seconds=0.05))
            pose = trans.transform.translation
            _, _, yaw = euler_from_quaternion(trans.transform.rotation)
            self.robot_state = np.array([float(pose.x), float(pose.y), yaw])
            self.has_odom = True
        except Exception:
            if not self.has_odom:
                return  # nothing yet

        # ── LiDAR reactive avoidance ───────────────────────────────────────────
        # If the LiDAR sees an obstacle closer than LIDAR_DANGER_DIST directly
        # ahead, override normal control: stop forward motion and turn away.
        blocked, turn_sign = self.lidar_obstacle_ahead()
        if blocked and self.has_goal:
            desired_vel = np.array([0.0, turn_sign * self.max_angular_vel * 0.6])
        else:
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
