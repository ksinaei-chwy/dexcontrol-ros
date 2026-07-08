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

When the Pink backend is active, arm IK always enforces position limits. Pink's
`SelfCollisionBarrier` is available but defaults off because timing probes on
the current setup showed it cannot meet the 50 Hz teleop loop: even one
collision-barrier iteration for both arms took about 47 ms, and normal
multi-iteration solves were much slower. Torso height is solved with closed-form
planar geometry, and the head defaults to a fixed pitch command, so normal
teleop does not use Pink for torso/head. The arm kinematic model still comes
from `robot_urdf_path`, while the optional barrier uses the Vega
collision-sphere URDF and removes disabled collision pairs from the MoveIt SRDF:

```yaml
pink_self_collision_enabled: false
pink_self_collision_components: [left_arm, right_arm]
pink_self_collision_srdf_path: ""
pink_self_collision_urdf_path: ""
pink_self_collision_max_pairs: 24
pink_self_collision_min_distance: 0.04
pink_self_collision_gain: 1.0
pink_self_collision_safe_displacement_gain: 0.0
pink_velocity_limit_enabled: false
```

Leaving the SRDF and collision URDF paths empty selects the installed
`dexmate_vega_moveit_config/config/vega_1p_f5d6.srdf` and
`dexmate_vega_description/robots/humanoid/vega_1p/vega_1p_f5d6_collision_spheres.collision.urdf`.
The default component list targets the two arm solvers when the barrier is
explicitly enabled. Torso and head self-collision components are ignored because
those groups are not solved through Pink during normal teleop.

The default teleop loop runs at 50 Hz. For IK responsiveness, arm commands use
a separate slew limit from torso/head commands:

```yaml
pink_arm_max_iterations: 6
pink_arm_position_cost: 1.0
pink_arm_orientation_cost: 0.1
max_arm_joint_delta_per_tick: 0.08
```

Watch `/dex_pico_teleop/status` for `loop_ms`, `left_arm_iterations`,
`right_arm_iterations`, and arm error values. If `loop_ms` is consistently over
20 ms with `pink_self_collision_enabled:=false`, reduce `pink_arm_max_iterations`.
If loop timing is good but motion still lags, increase
`max_arm_joint_delta_per_tick` gradually. If `pink_self_collision_enabled:=true`,
expect the loop to miss 50 Hz on the current setup.

For dry-run testing without publishing robot commands:

```bash
ros2 launch dex_pico_teleop pico_teleop_dry_run.launch.py
```

If the Pico sender app is configured for TCP instead of UDP, launch the node
in TCP mode and point the headset at the robot/Jetson Wi-Fi IP and port:

```bash
ros2 launch dex_pico_teleop pico_teleop_dry_run.launch.py \
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
MeshCat node mirrors those computed joint targets in a browser. This launch does
not start `dexcontrol_bridge`; if `/joint_states` is already present, MeshCat
uses it only as the initial pose before the first teleop log frame.

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

Terminal 1: start the Pico teleop dry-run stack in TCP mode:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 launch dex_pico_teleop pico_teleop_dry_run.launch.py \
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

Open the MeshCat URL in the desktop browser. If you are browsing from the
desktop/laptop, use `http://<jetson-ip>:7000/static/`, not `127.0.0.1`.

After Pico data is connected, stand in a neutral pose and call:

```bash
ros2 service call /dex_pico_teleop/calibrate std_srvs/srv/Trigger {}
ros2 service call /dex_pico_teleop/calibrate_reach std_srvs/srv/Trigger {}
ros2 service call /dex_pico_teleop/enabled std_srvs/srv/SetBool "{data: true}"
```

The same actions are available from Pico controller clicks while tracking data
is fresh: right `A` neutral-calibrates, right `B` calibrates reach, left `Y`
enables teleop, and left `X` disables teleop. The CLI services above continue
to work unchanged.

Expected behavior:

1. Torso follows headset/operator height after calibration.
2. Each arm follows the corresponding controller's shoulder-relative posture
   while teleop is enabled.
3. Hand closing uses trigger values only if hand joint offsets are configured.
4. Base motion follows the controller joysticks and returns to zero when they
   are centered.
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
and `right_arm` arrays plus optional `debug` retarget/timing data when teleop is
enabled. The MeshCat node subscribes to `/dex_pico_teleop/log_frame`, so it
moves the simulated model and never publishes robot commands.

## Calibration And Safety

1. Start `dexcontrol_bridge` and verify `/joint_states`.
2. Start this node with `publish_commands:=false` first.
3. Stand in a neutral pose with both ankle trackers visible.
4. Click right `A` or call `ros2 service call /dex_pico_teleop/calibrate std_srvs/srv/Trigger {}`.
5. Hold both controllers at comfortable full reach and click right `B`, or call `ros2 service call /dex_pico_teleop/calibrate_reach std_srvs/srv/Trigger {}`.
6. Enable teleop with left `Y` or `ros2 service call /dex_pico_teleop/enabled std_srvs/srv/SetBool "{data: true}"`.
7. Re-launch with `publish_commands:=true` only after status looks correct.

Arm motion uses posture retargeting. The controller position is interpreted
relative to the estimated operator shoulder, normalized by estimated operator
arm length, and mapped onto the robot shoulder and arm reach. If reach
calibration has been completed, the calibrated per-side arm lengths are used;
otherwise the node falls back to the height-based estimate. Base motion
follows the controller joysticks and publishes zero velocity whenever input is
stale, held, disabled, uncalibrated, or inside the joystick deadzone. With the
default base mapping, the left controller joystick commands body-frame
forward/strafe velocity and the right controller horizontal joystick axis
commands yaw velocity. Head joint tracking can be disabled with
`head_tracking_enabled:=false`; in that mode the node commands the head to a
fixed forward pose pitched down by `head_disabled_pitch_deg`.

Hand commands are disabled until `*_hand_close_offsets` match the configured or
discovered hand joint count.

## XRoboToolkit Remote Vision

The head-camera vision adapter is a separate process from teleop. It creates a
single Dexmate camera stream subscriber for the ZED head camera and republishes
that RGB stream over `dexcomm.rtc.VideoPublisher` for XRoboToolkit Remote
Vision. It does not subscribe to or publish ROS image topics.

Start the robot-side Dexmate services, including the camera sensor service, then
launch the adapter:

```bash
python3 -m pip install dexcomm-video==0.4.19 av==17.1.0

dextop node start
dexsensor launch --sensor camera

cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 launch dex_pico_teleop head_camera_vision.launch.py
```

Default stream:

- Dexmate source: `head_camera.left_rgb`
- Dexmate source transport: `zenoh`
- XRoboToolkit RTC channel:
  `xrobotoolkit/remote_vision/head_camera/left_rgb_rtc`
- Output: `1280x720`, `30 FPS`, H.264 preferred with VP8 fallback, `1.5 Mbps`

Configure XRoboToolkit Remote Vision to subscribe to the RTC channel above. If
the deployed XRoboToolkit profile uses a different channel name, pass it at
launch:

```bash
ros2 launch dex_pico_teleop head_camera_vision.launch.py \
  rtc_channel:=<xrobotoolkit-remote-vision-channel>
```

### XRoboToolkit ZED Mini Listen Mode

The app's `ZEDMINI` -> `Listen` path does not subscribe to the RTC channel. It
listens for a direct TCP H.264 stream on port `12345`. In that mode, launch the
head-camera bridge with the Pico headset IP as `xrtcp_host`; do not use the
Jetson IP for this parameter.

In XRoboToolkit on the Pico headset:

1. Open the Network panel and note the headset `IP`.
2. Select `ZEDMINI`.
3. Click `Listen`.
4. Keep the streaming port at `12345`.

On the Jetson/ROS host:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 launch dex_pico_teleop head_camera_vision.launch.py \
  rtc_enabled:=false \
  xrtcp_enabled:=true \
  xrtcp_host:=<pico-headset-ip> \
  xrtcp_port:=12345 \
  fps:=30.0 \
  xrtcp_bitrate:=3000000 \
  xrtcp_write_timeout_s:=2.0
```

The node duplicates the mono left RGB stream into a side-by-side stereo H.264
frame because XRoboToolkit's ZED Mini path expects stereo video. If the headset
connects successfully, the status topic reports `xrtcp_connected: true` and
`xrtcp_output_frames` increasing. The RTC-only fields `connected` and
`subscriber_count` may remain false/zero when `rtc_enabled:=false`.

Useful status and runtime controls:

```bash
ros2 topic echo --once --full-length /dex_pico_teleop/head_camera_vision/status
ros2 service call /dex_pico_teleop/head_camera_vision/enabled std_srvs/srv/SetBool "{data: false}"
ros2 service call /dex_pico_teleop/head_camera_vision/enabled std_srvs/srv/SetBool "{data: true}"
```

The status topic reports frame counts, output dimensions, publish failures,
connection state, and subscriber count. During teleop validation, compare this
with `/dex_pico_teleop/log_frame` and `/joint_states`; video should drop or
stall independently without making Pico input stale or changing command timing.
