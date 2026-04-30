# EN605.613 Final Project — Autonomous Maze Navigation

## Architecture

```
Gazebo world
  ↓ /tf (odom→base_link, world→vehicle_blue/base_link)
  ↓ /odom
Static TF: map → world (identity)
Static TF: vehicle_blue/base_link → base_link (alias)
  → map frame = Gazebo world frame

map_publisher  → /map (OccupancyGrid, static)
goal_points_publisher → /goal_points (MarkerArray)
astar_planner  ← /map, /planner_goal  → /waypoints, /planned_path
navigator      ← /goal_points, /waypoints  → /planner_goal, /controller_goal
diffdrive_pid  ← /controller_goal  → /cmd_vel
```

**Data flow:**
`Static map → A* planner → waypoint path → navigator → PID controller → cmd_vel`

## Build and run

```bash
colcon build --symlink-install
source install/setup.bash

ros2 launch gazebo_controller full_simulation.launch.py map_folder:=Maze_hr
ros2 launch gazebo_controller full_simulation.launch.py map_folder:=Maze_ng
ros2 launch gazebo_controller full_simulation.launch.py map_folder:=Maze_ql_1
```

## Key design decisions

### Coordinate frames
- `map` = Gazebo world frame (connected via `map → world` identity TF).
- The DiffDrive plugin publishes `odom → base_link` (relative odometry) **and**
  the PosePublisher plugin publishes `world → vehicle_blue/base_link` (absolute).
- A static TF `vehicle_blue/base_link → base_link` aliases the namespaced link
  name so that all code can use the simple name `base_link`.
- All goal coordinates (poses.csv), map origin (map.yaml), and waypoints are in
  the Gazebo world frame = map frame. No coordinate offset needed.

### Why NOT odom-based localization
The DiffDrive odometry origin is set by Gazebo at plugin initialization, which
does not equal the robot's spawn position. Using odom as the reference frame
would require a per-maze `map → odom` offset. Instead we use the absolute
world-frame pose from the PosePublisher, which matches poses.csv exactly.

### Backward driving fix
The trailer-hitch PID clamps linear velocity to `[0, max_linear_vel]`. When the
goal is behind the robot, the angular velocity turns it first; linear speed is
applied only once the robot is broadly facing the goal. This prevents the robot
from reversing into walls in narrow corridors.

### Inflation radius
Set to 4 cells × 0.05 m/cell = 0.20 m clearance. The robot body radius is
~0.12 m, so 0.20 m gives adequate wall clearance while leaving navigable
corridors in the 1-metre-wide maze passages.

## Debugging topics

```bash
ros2 topic echo /navigator_status
ros2 topic echo /cmd_vel
ros2 topic echo /waypoints
ros2 run tf2_tools view_frames    # visualise TF tree
ros2 topic echo /tf               # check odom→base_link is present
```
