#spawn_entities.launch.py

import os
import csv

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('gazebo_controller')

    maze_arg = DeclareLaunchArgument(
        'maze',
        default_value='basic_maze',
        description='Maze folder name'
    )

    def launch_setup(context):
        maze_folder = context.perform_substitution(LaunchConfiguration('maze'))

        vehicle_model_file = os.path.join(pkg_share, 'sdf', 'vehicle_blue_model.sdf')
        goal_sphere_file   = os.path.join(pkg_share, 'sdf', 'goal_sphere.sdf')
        poses_file         = os.path.join(pkg_share, 'sdf', maze_folder, 'poses.csv')

        poses = {}
        with open(poses_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                poses[row['name']] = {k: row[k] for k in ('x', 'y', 'z', 'yaw')}

        # ---- Spawn robot ----
        robot_pose = poses['robot']
        spawn_vehicle = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-file', vehicle_model_file,
                '-name', 'vehicle_blue',
                '-x', robot_pose['x'],
                '-y', robot_pose['y'],
                '-z', robot_pose['z'],
                '-Y', robot_pose['yaw'],
            ],
            output='screen'
        )

        # ---- Spawn goal spheres ----
        spawn_goals = []
        for key in ('goal_1', 'goal_2', 'goal_3'):
            gp = poses[key]
            spawn_goals.append(Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-file', goal_sphere_file,
                    '-name', key,
                    '-x', gp['x'], '-y', gp['y'], '-z', gp['z'],
                ],
                output='screen'
            ))

        # ---- TF tree design ----
        #
        # What Gazebo publishes via bridge:
        #   /tf from DiffDrive plugin:     odom -> base_link  (wheel odometry)
        #   /tf from PosePublisher+bridge: world -> vehicle_blue/base_link
        #                                  world -> vehicle_blue/left_wheel  etc.
        #
        # What our navigation nodes need:
        #   TF lookup: map -> base_link
        #
        # THE SIMPLEST CORRECT SOLUTION:
        #   Publish map -> odom as identity (map and odom share same origin).
        #   Then the chain is: map -> odom -> base_link  ✓
        #   The DiffDrive plugin keeps odom -> base_link accurate as the robot moves.
        #
        # We do NOT alias vehicle_blue/base_link -> base_link because
        # base_link already has a parent (odom) from the DiffDrive plugin.
        # Two parents for one TF frame causes TF errors and warnings.
        #
        # map -> world is kept for completeness (so RViz can show Gazebo world objects)
        # but navigation uses odom -> base_link exclusively.

        # map == odom (identity) — THE critical TF for navigation
        # All navigation nodes look up map->base_link.
        # With this, the chain map->odom->base_link resolves correctly.
        static_tf_map_odom = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_map_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen'
        )

        # map == world (identity) — lets world-frame Gazebo objects appear in map frame
        static_tf_map_world = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_map_world',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'world'],
            output='screen'
        )

        # Lidar sensor frame
        lidar_tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_lidar',
            arguments=['0.16', '0', '0.08', '0', '0', '0',
                       'base_link', 'vehicle_blue/lidar/lidar_sensor'],
            output='screen'
        )

        # ---- Map publisher (static occupancy grid from map.yaml) ----
        map_publisher = Node(
            package='gazebo_controller',
            executable='map_publisher',
            name='map_publisher',
            output='screen',
            parameters=[{'maze': maze_folder}]
        )

        # ---- Goal points publisher ----
        goal_points_publisher = Node(
            package='gazebo_controller',
            executable='goal_points_publisher',
            name='goal_points_publisher',
            output='screen',
            parameters=[{'maze': maze_folder}]
        )

        return [
            spawn_vehicle,
            *spawn_goals,
            static_tf_map_odom,
            static_tf_map_world,
            lidar_tf,
            map_publisher,
            goal_points_publisher,
        ]

    return LaunchDescription([
        maze_arg,
        OpaqueFunction(function=launch_setup)
    ])