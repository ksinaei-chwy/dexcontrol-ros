# dex_pico_teleop

Standalone ROS 2 teleoperation package for Dexmate Vega 1 Pro using Pico XR
headset, controllers, and ankle trackers. The node receives compact JSON over
TCP or UDP, retargets motion locally, and commands the existing
`dexcontrol_ros` bridge topics.

## Runtime Shape

The intended deployment is the robot-side Docker container running on the
onboard Jetson Thor computer. Start `dexcontrol_ros` first, then launch this
package inside the same ROS environment.

Install the preferred IK stack in the container:

```bash
python3 -m pip install pin pin-pink "scipy>=1.14" "qpsolvers[open_source_solvers]"
```

The node defaults to `kinematics_backend:=auto`, which tries Pinocchio/Pink
first and falls back to the local numeric backend if optional dependencies are
missing. Use `kinematics_backend:=pink` to fail fast when Pink is unavailable.

```bash
cd /workspaces/dexcontrol-ros/ros_ws
colcon build --symlink-install --packages-select dex_pico_teleop
source install/setup.bash
ros2 launch dex_pico_teleop pico_teleop.launch.py
```

For dry-run testing without publishing robot commands:

```bash
ros2 launch dex_pico_teleop pico_teleop.launch.py publish_commands:=false
```

If the Pico sender app is configured for TCP instead of UDP, launch the node
in TCP mode and point the headset at the robot/Jetson Wi-Fi IP and port:

```bash
ros2 launch dex_pico_teleop pico_teleop.launch.py \
  publish_commands:=false \
  network_transport:=tcp \
  network_host:=0.0.0.0 \
  network_port:=63901
```

## Pico JSON Packet

The default receiver transport is TCP on port `63901` to match XRoboToolkit's
PC-service pose-sync connection. Plain TCP JSON is also accepted as one JSON
object per line, so every packet must end with `\n`. UDP is still supported
with `network_transport:=udp`, using one JSON object per datagram. Pose order
is `[x, y, z, qx, qy, qz, qw]`.

In XRoboToolkit, the Network panel's `IP` field is the headset IP. Put the
robot/Jetson address in `PC Service IP` or `Enter Manually`, then reconnect.

```json
{
  "timestamp_ns": 123,
  "frame": "openxr_y_up",
  "head": {"pose": [0, 1.6, 0, 0, 0, 0, 1]},
  "controllers": {
    "left": {
      "pose": [0, 1.2, -0.3, 0, 0, 0, 1],
      "trigger": 0.0,
      "grip": 0.0,
      "joystick": [0.0, 0.0],
      "buttons": {"stick": false}
    },
    "right": {
      "pose": [0, 1.2, -0.3, 0, 0, 0, 1],
      "trigger": 0.0,
      "grip": 0.0,
      "joystick": [0.0, 0.0],
      "buttons": {"stick": false}
    }
  },
  "trackers": {
    "left_ankle": {"pose": [0, 0.1, 0, 0, 0, 0, 1], "confidence": 1.0},
    "right_ankle": {"pose": [0, 0.1, 0, 0, 0, 0, 1], "confidence": 1.0}
  }
}
```

`frame` may be `openxr_y_up` or `robot_z_up`. OpenXR packets are converted to
Vega/ROS convention: x forward, y left, z up.

## MeshCat Dry-Run Visualization

Install MeshCat in the same Python environment as ROS:

```bash
python3 -m pip install meshcat
```

Start teleop in dry-run mode, calibrate, and enable it:

```bash
ros2 launch dex_pico_teleop pico_teleop.launch.py publish_commands:=false
ros2 service call /dex_pico_teleop/calibrate std_srvs/srv/Trigger {}
ros2 service call /dex_pico_teleop/enabled std_srvs/srv/SetBool "{data: true}"
```

In another terminal, launch the MeshCat visualizer:

```bash
ros2 launch dex_pico_teleop meshcat_visualizer.launch.py
```

Open the MeshCat URL printed by the visualizer. The node subscribes to
`/dex_pico_teleop/log_frame`, so it only moves the simulated model and never
publishes robot commands.

## Calibration And Safety

1. Start `dexcontrol_bridge` and verify `/joint_states`.
2. Start this node with `publish_commands:=false` first.
3. Stand in a neutral pose with both ankle trackers visible.
4. Call `ros2 service call /dex_pico_teleop/calibrate std_srvs/srv/Trigger {}`.
5. Enable teleop with `ros2 service call /dex_pico_teleop/enabled std_srvs/srv/SetBool "{data: true}"`.
6. Re-launch with `publish_commands:=true` only after status looks correct.

Arm motion is hold-to-enable with the corresponding grip. Base motion is gated
by the configured deadman button and publishes zero velocity whenever input is
stale, held, disabled, or uncalibrated.

Hand commands are disabled until `*_hand_close_offsets` match the configured or
discovered hand joint count.
