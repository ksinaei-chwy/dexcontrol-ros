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

The node defaults to `kinematics_backend:=pink`, so missing Pink/Pinocchio
dependencies fail fast. Use `kinematics_backend:=auto` only for local debug
sessions where falling back to the numeric backend is acceptable.

```bash
cd /workspaces/dexcontrol-ros/ros_ws
colcon build --symlink-install --packages-select dex_pico_teleop
source install/setup.bash
ros2 launch dex_pico_teleop pico_teleop.launch.py
```

## Pink Self-Collision Barrier

When the Pink backend is active, arm IK uses Pink's `SelfCollisionBarrier` by
default. The kinematic model still comes from `robot_urdf_path`, while the
barrier uses the Vega collision-sphere URDF and removes disabled collision pairs
from the MoveIt SRDF:

```yaml
pink_self_collision_enabled: true
pink_self_collision_components: [left_arm, right_arm]
pink_self_collision_srdf_path: ""
pink_self_collision_urdf_path: ""
pink_self_collision_max_pairs: 24
pink_self_collision_min_distance: 0.04
pink_self_collision_gain: 1.0
pink_self_collision_safe_displacement_gain: 0.0
```

Leaving the SRDF and collision URDF paths empty selects the installed
`dexmate_vega_moveit_config/config/vega_1p_f5d6.srdf` and
`dexmate_vega_description/robots/humanoid/vega_1p/vega_1p_f5d6_collision_spheres.collision.urdf`.
The default component list intentionally protects the two arm solvers. Torso
and head barriers can be enabled for tuning, but the current SRDF/sphere set has
nominal close torso/head pairs that should be evaluated in MeshCat before
hardware use.

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

Use this test before sending anything to the real robot. Teleop runs with
`publish_commands:=false`, publishes only `/dex_pico_teleop/log_frame`, and the
MeshCat node mirrors those computed joint targets in a browser.

Install MeshCat in the same Python environment as ROS:

```bash
python3 -m pip install meshcat
```

Confirm the Jetson/robot computer IP on the Wi-Fi network. This is the address
that XRoboToolkit should connect to, and it is also the address to use from a
desktop or laptop browser:

```bash
ip -4 addr
```

For example, if the Jetson Wi-Fi IP is `10.233.169.139`, the desktop MeshCat
URL will be:

```text
http://10.233.169.139:7000/static/
```

The local URL printed by MeshCat, such as `http://127.0.0.1:7000/static/`, only
works inside the Jetson/container session. Use `http://<jetson-ip>:7000/static/`
from another machine on the same network.

Terminal 1: start the bridge or any source that publishes `/joint_states`, then
verify joint states are visible:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic echo --once /joint_states
```

Terminal 2: start the Pico teleop node in dry-run TCP mode:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 launch dex_pico_teleop pico_teleop.launch.py \
  publish_commands:=false \
  network_transport:=tcp \
  network_host:=0.0.0.0 \
  network_port:=63901
```

In XRoboToolkit on the Pico headset:

1. Set the connection mode to TCP.
2. Set the PC/service address to the Jetson IP, for example `10.233.169.139`.
3. Set the port to `63901`.
4. Keep the headset IP separate from the Jetson IP; both devices must have
   different addresses.
5. Enable tracking send for the headset and controllers, then reconnect.

Terminal 3: start the MeshCat visualizer:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 launch dex_pico_teleop meshcat_visualizer.launch.py
```

Open the MeshCat URL in the desktop browser. If you are browsing from the
desktop/laptop, use `http://<jetson-ip>:7000/static/`, not `127.0.0.1`.

After Pico data is connected, stand in a neutral pose and call:

```bash
ros2 service call /dex_pico_teleop/calibrate std_srvs/srv/Trigger {}
ros2 service call /dex_pico_teleop/enabled std_srvs/srv/SetBool "{data: true}"
```

Expected behavior:

1. Torso follows headset/operator height after calibration.
2. Each arm follows only while holding that controller's grip button.
3. Hand closing uses trigger values only if hand joint offsets are configured.
4. Base motion remains gated by the configured deadman button.
5. When an arm is driven toward the body or the other neutral arm, IK should
   slow or stop before the collision-sphere margin is crossed.

Useful sanity checks:

```bash
ros2 topic echo --once --full-length /dex_pico_teleop/status
ros2 topic hz /dex_pico_teleop/log_frame
ros2 topic echo --once --full-length /dex_pico_teleop/log_frame
```

`/dex_pico_teleop/status` should report `calibrated: true`,
`enabled: true`, and `stale_input: false`. `/dex_pico_teleop/log_frame` should
publish near the control rate and contain changing `torso`, `head`, `left_arm`,
and `right_arm` arrays when the relevant inputs are active. The MeshCat node
subscribes only to `/dex_pico_teleop/log_frame`, so it moves the simulated model
and never publishes robot commands.

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
