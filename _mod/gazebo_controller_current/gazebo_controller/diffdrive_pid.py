#diffdrive_pid.py
#!/usr/bin/env python3
"""
DiffDrive PID Controller
========================
Trailer-hitch proportional-derivative controller for a differential-drive robot.

Subscribes: /controller_goal (geometry_msgs/PoseStamped) — waypoint from navigator
Publishes:  /cmd_vel         (geometry_msgs/Twist)        — wheel velocity command

Robot pose is obtained via TF lookup: map -> base_link.
The map frame equals the Gazebo world frame (via static TF map->world in launch).
Goal poses are published in the map frame, so the error is computed entirely
in one consistent world-frame coordinate system.

TRAILER HITCH APPROACH:
The hitch point X_p is projected 'lookahead' metres in front of the robot.
The PID error is (goal - X_p). Rotating that error into the robot body frame
gives:
  V[0] = forward component  → linear velocity
  V[1] = lateral component  → angular velocity (divided by lookahead)

BACKWARD DRIVING:
Linear velocity is clamped to [0, max_linear_vel].
When the goal is behind the robot, angular velocity turns the robot toward
the goal before it drives forward. This is correct for narrow maze corridors.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Twist
import tf2_ros


def euler_from_quaternion(q):
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    sinp = np.clip(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0)
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(sinr, cosr), np.arcsin(sinp), np.arctan2(siny, cosy)


class DiffDrivePID(Node):
    def __init__(self):
        super().__init__('diffdrive_pid')

        # Defaults match the values set in full_simulation.launch.py.
        # The launch file parameters override these at runtime.
        self.declare_parameter('kp',              2.0)
        self.declare_parameter('kd',              0.1)
        self.declare_parameter('ki',              0.0)
        self.declare_parameter('lookahead',       0.4)
        self.declare_parameter('publish_rate',   40.0)
        self.declare_parameter('max_linear_vel',  1.8)
        self.declare_parameter('max_angular_vel', 3.0)
        self.declare_parameter('goal_tolerance', 0.18)
        self.declare_parameter('turn_in_place_angle_deg', 70.0)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')

        self.k_p      = self.get_parameter('kp').value
        self.k_d      = self.get_parameter('kd').value
        self.k_i      = self.get_parameter('ki').value
        self.length   = self.get_parameter('lookahead').value
        rate          = self.get_parameter('publish_rate').value
        self.max_lin  = self.get_parameter('max_linear_vel').value
        self.max_ang  = self.get_parameter('max_angular_vel').value
        self.goal_tol = self.get_parameter('goal_tolerance').value
        self.turn_in_place_angle = np.deg2rad(
            self.get_parameter('turn_in_place_angle_deg').value)
        self.base_frame = self.get_parameter('base_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.dt       = 1.0 / rate

        # TF — reads robot pose in map frame (map = Gazebo world via static TF)
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.goal_sub = self.create_subscription(
            PoseStamped, '/controller_goal', self._goal_cb, 10)
        self.vel_pub  = self.create_publisher(
            Twist, '/cmd_vel', QoSProfile(depth=10))

        # State
        self.desired_goal  = np.array([0.0, 0.0])
        self.robot_state   = np.array([0.0, 0.0, 0.0])   # [x, y, yaw] in map frame
        self.Xp_last       = None
        self.goal_received = False

        self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(
            f'DiffDrivePID ready  kp={self.k_p} kd={self.k_d}  '
            f'lookahead={self.length}m  rate={rate}Hz  '
            f'TF: {self.map_frame} -> {self.base_frame}'
        )

    # ------------------------------------------------------------------
    def _goal_cb(self, msg: PoseStamped):
        """Accept waypoint goal — expected in map frame."""
        frame = msg.header.frame_id
        gx = msg.pose.position.x
        gy = msg.pose.position.y

        if frame not in (self.map_frame, 'world', ''):
            # Transform to map frame if goal arrives in a different frame
            try:
                t = self._tf_buffer.lookup_transform(
                    self.map_frame, frame, rclpy.time.Time(),
                    timeout=Duration(seconds=0.5))
                tx = t.transform.translation.x
                ty = t.transform.translation.y
                _, _, yaw = euler_from_quaternion(t.transform.rotation)
                # Apply 2D rigid transform: rotate then translate
                gx_new = np.cos(yaw) * gx - np.sin(yaw) * gy + tx
                gy_new = np.sin(yaw) * gx + np.cos(yaw) * gy + ty
                gx, gy = gx_new, gy_new
            except Exception as e:
                self.get_logger().warn(f'Goal transform failed: {e}')
                return

        self.desired_goal  = np.array([gx, gy])
        self.Xp_last       = None   # reset derivative on new goal
        self.goal_received = True
        self.get_logger().info(f'New goal: ({gx:.3f}, {gy:.3f})')

    # ------------------------------------------------------------------
    def _control_loop(self):
        """Read robot pose from TF and publish cmd_vel."""
        try:
            frame_candidates = [self.base_frame, 'vehicle_blue/base_link']
            t = None
            for frame in frame_candidates:
                try:
                    t = self._tf_buffer.lookup_transform(
                        self.map_frame, frame,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=0.2))
                    break
                except Exception:
                    continue
            if t is None:
                raise RuntimeError('No matching base frame in TF tree')
        except Exception as e:
            self.get_logger().warn(
                f'TF {self.map_frame}->{self.base_frame} failed: {e}',
                throttle_duration_sec=2.0)
            return

        pos = t.transform.translation
        _, _, yaw = euler_from_quaternion(t.transform.rotation)
        self.robot_state = np.array([pos.x, pos.y, yaw])

        if not self.goal_received:
            return

        vel = self._compute_vel()

        msg = Twist()
        msg.linear.x  = float(vel[0])
        msg.angular.z = float(vel[1])
        self.vel_pub.publish(msg)

    # ------------------------------------------------------------------
    def _compute_vel(self) -> np.ndarray:
        x_r, y_r, th_r = self.robot_state

        goal_dx = self.desired_goal[0] - x_r
        goal_dy = self.desired_goal[1] - y_r
        goal_dist = float(np.hypot(goal_dx, goal_dy))
        if goal_dist < self.goal_tol:
            self.Xp_last = None
            return np.array([0.0, 0.0])

        goal_heading = np.arctan2(goal_dy, goal_dx)
        heading_err = np.arctan2(np.sin(goal_heading - th_r), np.cos(goal_heading - th_r))

        # Trailer hitch point projected in front of the robot
        X_p = np.array([x_r + self.length * np.cos(th_r),
                         y_r + self.length * np.sin(th_r)])

        p_err = (self.desired_goal - X_p)[:, None]   # shape (2,1)

        if self.Xp_last is None:
            d_err = np.zeros((2, 1))
        else:
            # Derivative of the error w.r.t. the hitch position (not the goal)
            prev_p_err = (self.desired_goal - self.Xp_last)[:, None]
            d_err = (p_err - prev_p_err) / self.dt

        # Rotate world-frame error into robot body frame
        inv_rot = np.array([[ np.cos(th_r), np.sin(th_r)],
                             [-np.sin(th_r), np.cos(th_r)]])

        V = inv_rot @ (self.k_p * p_err - self.k_d * d_err)

        linear_x  = float(V[0, 0])
        angular_z = float(V[1, 0]) / self.length

        # If the goal is mostly behind the robot, rotate in place first.
        if abs(heading_err) > self.turn_in_place_angle:
            linear_x = 0.0

        # Reduce forward speed when heading error is large to avoid wall strikes.
        heading_scale = max(0.0, np.cos(heading_err))
        linear_x *= heading_scale

        # Clamp: no backward driving in maze corridors.
        # angular_z is NOT clamped before the linear clamp so the robot
        # can spin in place to face a behind-goal target.
        linear_x  = np.clip(linear_x,  0.0,          self.max_lin)
        angular_z = np.clip(angular_z, -self.max_ang, self.max_ang)

        self.Xp_last = X_p
        return np.array([linear_x, angular_z])


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