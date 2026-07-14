# dexcontrol_navigation

SLAM and Nav2 bringup for Dexmate Vega robots running through
`dexcontrol_ros`.

This package is the navigation package in the unified `ros_ws/src` runtime
workspace. The ROS package name is `dexcontrol_navigation`.

## What It Launches

Mapping:

1. Starts `dexcontrol_bridge` unless `use_bridge:=false`.
2. Converts `/lidar_3d_front/points` and `/lidar_3d_back/points` from
   `PointCloud2` into planar `LaserScan` messages in frame `base`.
3. Filters front-lidar self-returns near the robot body and arms.
4. Merges front/back scans into `/scan`.
5. Runs `slam_toolbox` online mapping.

Navigation:

1. Starts the same bridge and scan pipeline.
2. Starts Nav2 from `nav2_bringup` with AMCL and the configured map.
3. Optionally starts RViz with the checked-in Nav2 config.

The expected TF chain is:

```text
map -> odom -> base -> front_lidar
                 \-> back_lidar
```

The bridge publishes `odom -> base`; `robot_state_publisher` publishes the robot
description frames; `slam_toolbox` or AMCL publishes `map -> odom`.

## Build

Build from the unified workspace inside the ROS Humble Docker container:

```bash
export PROJECT_NAME=${PROJECT_NAME:-dexcontrol-ros}
cd "/workspaces/$PROJECT_NAME"
source /opt/ros/humble/setup.bash
python3 -m pip install -e .

cd ros_ws
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
colcon build --symlink-install --packages-up-to dexcontrol_navigation
source install/setup.bash
```

## Mapping

```bash
ros2 launch dexcontrol_navigation mapping.launch.py use_rviz:=true
```

If the bridge is already running:

```bash
ros2 launch dexcontrol_navigation mapping.launch.py use_bridge:=false use_rviz:=true
```

Useful checks:

```bash
ros2 topic hz /lidar_3d_front/scan
ros2 topic hz /lidar_3d_back/scan
ros2 topic hz /scan
ros2 topic hz /map
ros2 run tf2_ros tf2_echo map base
```

Save a map:

```bash
cd "/workspaces/$PROJECT_NAME/ros_ws"
mkdir -p src/dexcontrol_navigation/maps
ros2 run nav2_map_server map_saver_cli -f src/dexcontrol_navigation/maps/thor_initial
```

This creates:

```text
src/dexcontrol_navigation/maps/thor_initial.yaml
src/dexcontrol_navigation/maps/thor_initial.pgm
```

## Nav2

Launch against a saved map:

```bash
ros2 launch dexcontrol_navigation nav.launch.py \
  map:="/workspaces/$PROJECT_NAME/ros_ws/src/dexcontrol_navigation/maps/lab_test.yaml" \
  use_rviz:=true
```

Launch with the checked-in starter map:

```bash
ros2 launch dexcontrol_navigation nav.launch.py use_rviz:=true
```

In RViz, use `2D Pose Estimate` before sending a `Nav2 Goal`. Keep the hardware
e-stop ready and verify `/cmd_vel` before longer navigation runs.

## Tuning Notes

- If `/scan` is empty, verify TF and tune `config/cloud_to_scan.yaml`
  `min_height` / `max_height` in SI meters.
- If scan points include the robot body or arms, tune
  `config/scan_self_filter.yaml` exclusion boxes in frame `base`.
- If maps bend through long aisles, slow down and verify odometry yaw drift
  before tuning loop-closure thresholds.
- If Nav2 commands are too aggressive, reduce velocity and acceleration limits
  in `config/nav2_params.yaml`.
