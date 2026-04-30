#full_simulation.launch.py

import os

from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, DeclareLaunchArgument,
                             OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('gazebo_controller')

    # Accept both maze:= and map_folder:= (assignment uses map_folder:=)
    maze_arg = DeclareLaunchArgument(
        'maze', default_value='basic_maze',
        description='Maze folder: basic_maze | Maze_hr | Maze_ng | Maze_ql_1')
    map_folder_arg = DeclareLaunchArgument(
        'map_folder', default_value='',
        description='Alias for maze (map_folder:=Maze_hr)')

    def launch_setup(context):
        mf = context.perform_substitution(LaunchConfiguration('map_folder'))
        mz = context.perform_substitution(LaunchConfiguration('maze'))
        maze_folder = mf if mf else mz

        world_file = os.path.join(pkg_share, 'sdf', maze_folder, 'maze_world.sdf')

        gazebo = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('ros_gz_sim'),
                             'launch', 'gz_sim.launch.py')
            ]),
            launch_arguments={'gz_args': f'-r {world_file}'}.items(),
        )

        spawn_entities = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'spawn_entities.launch.py')),
            launch_arguments={'maze': maze_folder}.items()
        )

        bridge = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'ros_gz_bridge.launch.py'))
        )

        # ---------------------------------------------------------------
        # PID controller
        # Gains tuned for world-frame (map-frame) pose.
        # kp=2.0: strong proportional response
        # kd=0.1: light damping to prevent overshoot at corners
        # lookahead=0.4: hitch distance — longer = smoother, faster straight runs
        # max_linear=1.8, max_angular=3.0: fast enough for maze navigation
        # ---------------------------------------------------------------
        pid_controller = Node(
            package='gazebo_controller',
            executable='diffdrive_pid',
            name='diffdrive_pid',
            output='screen',
            parameters=[
                {'kp': 1.5},
                {'kd': 0.20},
                {'ki': 0.0},
                {'lookahead': 0.35},
                {'publish_rate': 40.0},
                {'max_linear_vel': 0.90},
                {'max_angular_vel': 2.4},
                {'goal_tolerance': 0.18},
                {'turn_in_place_angle_deg': 70.0},
                {'map_frame': 'map'},
                {'base_frame': 'base_link'},
            ]
        )

        # A* planner — subscribes to /map and /planner_goal
        # inflation_radius=4 cells x 0.05m = 0.20m clearance around walls
        astar_planner = Node(
            package='gazebo_controller',
            executable='astar_planner',
            name='astar_planner',
            output='screen',
            parameters=[
                {'inflation_radius':   4},
                {'occupied_threshold': 65},
                {'map_frame':         'map'},
                {'base_frame':   'base_link'},
            ]
        )

        # Navigator — orchestrates goal ordering, waypoint stepping, stuck recovery
        navigator = Node(
            package='gazebo_controller',
            executable='navigator',
            name='navigator',
            output='screen',
            parameters=[
                {'goal_tolerance':     0.40},
                {'waypoint_tolerance': 0.35},
                {'stuck_timeout':      10.0},
                {'loop_rate':          10.0},
                {'send_goal_interval':  0.3},
                {'status_interval':     1.0},
                {'map_frame':         'map'},
                {'base_frame':   'base_link'},
            ]
        )

        rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(pkg_share, 'rviz', 'rviz_view.rviz')]
        )

        return [gazebo, spawn_entities, bridge,
                pid_controller, astar_planner, navigator, rviz]

    return LaunchDescription([
        maze_arg,
        map_folder_arg,
        OpaqueFunction(function=launch_setup)
    ])