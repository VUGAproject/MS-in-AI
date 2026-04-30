#ros_gz_bridge.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
            '/tf_static@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
            '/lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/model/vehicle_blue/pose@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
        ],
        remappings=[
            ('/tf_static', '/tf_static'),
            ('/model/vehicle_blue/pose', '/tf'),
        ]
    )

    return LaunchDescription([bridge])