# Dexcontrol ROS 2 Workspace

Unified ROS 2 Humble workspace for running the Dexmate Vega 1P bridge, robot
description, MoveIt config, SLAM, and Nav2 on Jetson Thor inside Docker.

## Workspace Packages

This repository keeps the runtime ROS packages together under `ros_ws/src`:

| Package | Purpose |
| --- | --- |
| `dexcontrol_ros` | Hardware bridge from the `dexcontrol` Python API to ROS topics, TF, odom, point clouds, commands, and e-stop. |
| `dexmate_vega_description` | Vega 1P F5D6 robot description and URDF assets. |
| `dexmate_vega_moveit_config` | MoveIt configuration and trajectory adapter for the bridge command topics. |
| `dexcontrol_navigation` | Mapping and Nav2 bringup. This is the navigation package requested for the unified workspace. |

The bridge imports the `dexcontrol` Python package from this repository or from
PyPI, so install the Python API inside the container before launching hardware.

## Jetson Docker Runtime

Use the repository directory name as the Docker image/container/environment name
so the host repo, container, and optional Conda environment stay synchronized.

From the Jetson Thor host:

```bash
git clone git@github.com:ksinaei-chwy/dexcontrol-ros.git
cd dexcontrol-ros
PROJECT_NAME=$(basename "$PWD")

docker run --rm -it \
  --name "$PROJECT_NAME" \
  --network host \
  --ipc host \
  --privileged \
  -w "/workspaces/$PROJECT_NAME" \
  -e ROBOT_NAME="${ROBOT_NAME:-dm/vg150fef71c9-1p}" \
  -e ROBOT_IP="${ROBOT_IP:-}" \
  -e ZENOH_CONFIG="/root/.dexmate/comm/zenoh/chewy/zenoh_peer_config.json5" \
  -v "$PWD":"/workspaces/$PROJECT_NAME" \
  -v "$HOME/.dexmate":"/root/.dexmate:ro" \
  osrf/ros:humble-desktop \
  bash
```

If you use a custom Jetson/L4T ROS Humble image, keep the same `PROJECT_NAME`
pattern and replace only the final image name.

Inside the container:

```bash
export PROJECT_NAME=$(basename "$PWD")
cd "/workspaces/$PROJECT_NAME"
source /opt/ros/humble/setup.bash

apt update
apt install -y \
  python3-colcon-common-extensions \
  python3-pip \
  python3-rosdep \
  ros-humble-moveit \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-pointcloud-to-laserscan \
  ros-humble-robot-state-publisher \
  ros-humble-slam-toolbox \
  ros-humble-tf2-ros \
  ros-humble-xacro

python3 -m pip install --upgrade pip
python3 -m pip install -e .

cd ros_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
colcon build --symlink-install
source install/setup.bash
```

If `rosdep init` has not been run in the image, run `rosdep init` once as root
or `sudo rosdep init` as a non-root user, then repeat `rosdep update`.
If your custom image uses a non-root user, prepend `sudo` to the `apt` commands.

## Hardware Bridge

Default launch values match Vega 1P F5D6:

- `robot_name`: `dm/vg150fef71c9-1p`
- `zenoh_config`: `$HOME/.dexmate/comm/zenoh/chewy/zenoh_peer_config.json5`
- frames: `map -> odom -> base -> front_lidar/back_lidar`

Run the bridge:

```bash
source /opt/ros/humble/setup.bash
source "/workspaces/$PROJECT_NAME/ros_ws/install/setup.bash"
ros2 launch dexcontrol_ros dexcontrol_bridge.launch.py
```

For direct robot endpoint testing:

```bash
ros2 launch dexcontrol_ros dexcontrol_bridge.launch.py robot_ip:=192.168.5.20:7447
```

Check core topics:

```bash
ros2 topic hz /joint_states
ros2 topic hz /odom
ros2 topic hz /lidar_3d_front/points
ros2 topic hz /lidar_3d_back/points
ros2 run tf2_ros tf2_echo odom base
```

## Mapping

`mapping.launch.py` starts the bridge, converts front/back 3D lidar point clouds
to planar scans, filters robot self-returns, merges scans into `/scan`, and runs
`slam_toolbox`.

```bash
ros2 launch dexcontrol_navigation mapping.launch.py use_rviz:=true
```

If the bridge is already running in another terminal:

```bash
ros2 launch dexcontrol_navigation mapping.launch.py use_bridge:=false use_rviz:=true
```

Expected mapping topics:

```bash
ros2 topic hz /lidar_3d_front/scan
ros2 topic hz /lidar_3d_back/scan
ros2 topic hz /scan
ros2 topic hz /map
ros2 run tf2_ros tf2_echo map base
```

Save the map after slow, repeatable loops through the area:

```bash
cd "/workspaces/$PROJECT_NAME/ros_ws"
mkdir -p src/dexcontrol_navigation/maps
ros2 run nav2_map_server map_saver_cli -f src/dexcontrol_navigation/maps/thor_initial
```

Record a reproducibility bag using the local log standard:

```bash
mkdir -p ../logs/current_run/artifacts
ros2 bag record \
  -o ../logs/current_run/artifacts/thor_mapping \
  /tf /tf_static /odom /scan /map \
  /lidar_3d_front/points /lidar_3d_back/points
```

## Nav2

Use the saved map or the checked-in office-area starter map:

```bash
ros2 launch dexcontrol_navigation nav.launch.py \
  map:="/workspaces/$PROJECT_NAME/ros_ws/src/dexcontrol_navigation/maps/thor_initial.yaml" \
  use_rviz:=true
```

In RViz, set the initial pose with `2D Pose Estimate`, then send a `Nav2 Goal`.
Keep the hardware e-stop ready and verify `/cmd_vel` before longer runs:

```bash
ros2 topic echo /cmd_vel
ros2 topic hz /scan
ros2 topic echo /amcl_pose
```

For CLI testing after AMCL is localized:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}"
```

## MoveIt

Build and source the workspace, then launch the hardware MoveIt stack:

```bash
ros2 launch dexmate_vega_moveit_config hardware_moveit.launch.py
```

Use `dry_run:=true` first if you want MoveIt to accept trajectories without
publishing hardware commands:

```bash
ros2 launch dexmate_vega_moveit_config hardware_moveit.launch.py dry_run:=true
```

## Manual Safety Checklist

- Confirm the Zenoh config exists inside the container.
- Confirm `ros2 topic hz /odom` is stable before mapping or Nav2.
- Confirm `ros2 topic hz /scan` is stable and the scan lies flat in frame `base`.
- Confirm `/cmd_vel` is zero before enabling Nav2 goals.
- Keep the hardware e-stop reachable during mapping, localization, and navigation.
