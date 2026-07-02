# dexcontrol_ros

ROS 2 wrapper for the Dexmate `dexcontrol` Python API. The main bridge node keeps
local command buffers and sends joint/base commands to dexcontrol from a fixed-rate
timer, defaulting to 250 Hz.

## Topics and Service

- Publishes `/joint_states` with arms, torso, head, hands, and optionally chassis joints.
- Publishes `/dexcontrol/joint_feedback` with dexcontrol-specific feedback: positions,
  velocities, currents, torques, error codes, driver timestamps, and fingertip force
  when exposed by the hand API.
- Subscribes to `/joint_commands` plus per-component topics:
  `/left_arm/joint_commands`, `/right_arm/joint_commands`,
  `/left_hand/joint_commands`, `/right_hand/joint_commands`, and
  `/torso/joint_commands`, and `/head/joint_commands`.
- Publishes `/left_arm/ft_sensor/wrench` and `/right_arm/ft_sensor/wrench`.
- Publishes `/<lidar_3d_sensor>/points` for the Vega 3D lidars.
- Subscribes to `/cmd_vel` and publishes `/odom`.
- Publishes `odom -> base` TF. The launch file starts `robot_state_publisher`
  with `dexmate_vega_description` by default, so robot, lidar, camera, hand, and
  end-effector frames come from the Vega URDF.
- Provides `/soft_estop` as `std_srvs/SetBool`: `true` activates software e-stop,
  `false` releases it.

## Build and Run

Run this inside the ROS/Docker/Conda environment that already has `dexcontrol`
installed and the Dexmate communication config available. Do not install ROS
packages into the host Python environment.

Start the robot-side services after every robot boot:

```bash
dextop node start
dexsensor launch --sensor lidar
```

```bash
cd dexcontrol/ros_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch dexcontrol_ros dexcontrol_bridge.launch.py robot_ip:=192.168.50.20:7447
```

The launch file defaults to `robot_name:=dm/vg150fef71c9-1p` and
`zenoh_config:=$HOME/.dexmate/comm/zenoh/chewy/zenoh_peer_config.json5`.
You can still override either launch argument if needed.

Tune frames, rates, enabled sensors, and watchdog settings in
`config/vega_bridge.yaml`.

## Notes

The bridge frame defaults match the `vega_1p_f5d6` URDF: `base`, `front_lidar`,
`back_lidar`, `L_ee`, and `R_ee`. If another robot description is launched
separately, pass `publish_robot_description:=false` to avoid duplicate TF
publishers.

MoveIt configuration is intentionally deferred until MoveIt and the setup GUI are
available on this machine.
