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
python3 -m pip install pin pin-pink proxsuite "scipy>=1.14" "qpsolvers[open_source_solvers]"
```

The node defaults to `kinematics_backend:=pink`, so missing Pink/Pinocchio
dependencies fail fast. Use `kinematics_backend:=auto` only for local debug
sessions where falling back to the numeric backend is acceptable.

```bash
cd /workspaces/dexcontrol-ros/ros_ws
colcon build --symlink-install --packages-select dex_pico_teleop
```

## Runtime Commands

Run the common setup once in every terminal that launches a ROS process:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### Real-Robot Pico Teleop

Start `dexcontrol_bridge` first and confirm that `/joint_states` is healthy.
This command publishes arm position targets to the real robot:

```bash
ros2 launch dex_pico_teleop pico_teleop.launch.py \
  network_transport:=tcp \
  network_host:=0.0.0.0 \
  network_port:=63901 \
  control_rate_hz:=50.0
```

The default is Pink with the 18-sphere, 107-pair collision model, 4 cm minimum
surface distance, and CBF gain 6.0. If `loop_p99_ms` approaches the 20 ms
budget, retry at `control_rate_hz:=40.0`, then `30.0`.

### Teleop-Only Dry Run

This runs the real teleop node but suppresses command publication. Use it when
the bridge is providing joint feedback and a MeshCat simulator is not needed:

```bash
ros2 launch dex_pico_teleop pico_teleop.launch.py \
  publish_commands:=false \
  network_transport:=tcp \
  network_host:=0.0.0.0 \
  network_port:=63901 \
  control_rate_hz:=50.0
```

### Combined MeshCat Dry Run

This is the preferred no-hardware test. It starts Pico teleop, a delayed
position-servo feedback model, and MeshCat together; it never publishes robot
commands:

```bash
ros2 launch dex_pico_teleop pico_teleop_dry_run.launch.py \
  network_transport:=tcp \
  network_host:=0.0.0.0 \
  network_port:=63901 \
  pink_self_collision_enabled:=true \
  control_rate_hz:=50.0
```

To compare against Pink's legacy closest-pair selector, append
`pink_collision_pipeline:=closest_pairs`.

### MeshCat-Only Visualizer

Use this only when another teleop node is already publishing
`/dex_pico_teleop/log_frame`. It visualizes that stream but does not run Pico
retargeting or publish robot commands:

```bash
ros2 launch dex_pico_teleop meshcat_visualizer.launch.py \
  meshcat_show_collisions:=true \
  open_browser:=false
```

### Head-Camera Stream

Start Dexmate's camera sensor service, then publish the default Remote Vision
RTC stream:

```bash
dextop node start
dexsensor launch --sensor camera
ros2 launch dex_pico_teleop head_camera_vision.launch.py
```

The default RTC channel is
`xrobotoolkit/remote_vision/head_camera/left_rgb_rtc`. To use a different
XRoboToolkit channel, append
`rtc_channel:=<xrobotoolkit-remote-vision-channel>`.

For XRoboToolkit `ZEDMINI` -> `Listen` mode, use the Pico headset IP (not the
Jetson IP):

```bash
ros2 launch dex_pico_teleop head_camera_vision.launch.py \
  rtc_enabled:=false \
  xrtcp_enabled:=true \
  xrtcp_host:=<pico-headset-ip> \
  xrtcp_port:=12345 \
  fps:=30.0 \
  xrtcp_bitrate:=3000000 \
  xrtcp_write_timeout_s:=2.0
```

## Pink Self-Collision Barrier

The Pink arm controller is one bimanual QP. It refreshes both arms and the
torso from feedback every tick, constrains torso velocity to zero, solves once,
integrates once using the measured timer period, and publishes only arm
position targets. This keeps the solver's feedback state separate from its last
command and avoids the old two-arm, multiple-full-`dt` integration behavior.

The default collision pipeline filters the 182-sphere reference URDF to an
18-sphere real-time profile: both elbows, forearms, wrists, and palms, plus
six torso regions. It is deliberately biased toward the practical risk cases:
elbow/palm against the torso and palm/arm against the opposite arm. Fingers,
head, and base remain outside this arm IK collision model. The fixed pair set
avoids Pink's runtime closest-pair switching; larger 30/40/50-sphere profiles
and the full sphere model with Pink's closest-pair selection remain available
when broader coverage is required.

The arm model comes from `robot_urdf_path`; the barrier uses the Vega
collision-sphere URDF and removes disabled pairs from the MoveIt SRDF:

```yaml
pink_self_collision_enabled: true
pink_self_collision_components: [left_arm, right_arm]
pink_self_collision_srdf_path: ""
pink_self_collision_urdf_path: ""
pink_self_collision_max_pairs: 24
pink_self_collision_min_distance: 0.04
# Higher permits a faster approach to d_min; it does not reduce d_min itself.
pink_self_collision_gain: 6.0
pink_self_collision_safe_displacement_gain: 0.0
pink_collision_pipeline: reduced_all_pairs # or closest_pairs fallback
pink_collision_sphere_count: 18            # supported: 18, 30, 40, 50
pink_collision_sphere_inflation: 1.0
pink_velocity_limit_enabled: true
```

Leaving the SRDF and collision URDF paths empty selects the installed
`dexmate_vega_moveit_config/config/vega_1p_f5d6.srdf` and
`dexmate_vega_description/robots/humanoid/vega_1p/vega_1p_f5d6_collision_spheres.collision.urdf`.
The torso is present in the collision configuration at its live posture but is
not permitted to move by the arm QP. Head and base posture are currently fixed
from the arm solver's perspective.

The default teleop loop runs at 50 Hz. `pink_arm_max_iterations` and
`pink_self_collision_arm_max_iterations` remain compatibility parameters for
the numeric/legacy APIs; bimanual Pink always performs one solve and one
integration per timer callback.

```yaml
pink_arm_max_iterations: 6
# Compatibility setting; unified bimanual Pink still solves/integrates once.
pink_self_collision_arm_max_iterations: 2
pink_arm_position_cost: 1.0
pink_arm_orientation_cost: 0.1
max_arm_joint_delta_per_tick: 0.08
```

Finite integrated arm steps are published even when their Cartesian error is
still large. Only non-finite or infeasible results hold the previous arm
command, so a robot that is physically lagging can continue catching up.

Watch `/dex_pico_teleop/status` for `loop_ms`, `loop_p50_ms`, `loop_p95_ms`,
`loop_p99_ms`, `control_dt_ms`, `collision_min_distance`,
`collision_closest_pair`, and collision geometry/pair counts. The compact
profile leaves 107 fixed collision pairs in the present model, versus 233 for
the previous 30-sphere profile. Benchmark on the deployment computer. If the
observed `loop_p99_ms` is close to the 20 ms budget, lower the launch rate to
40 Hz, then 30 Hz if needed; a slower rate is safer than missed timer periods.
If it is still too costly, use `pink_collision_pipeline: closest_pairs` to
return to Pink's original 24-pair selector.

Compare the reduced profiles with the full sphere reference before changing
their contents or inflation:

```bash
PYTHONPATH=src/dex_pico_teleop python3 src/dex_pico_teleop/tools/evaluate_collision_profiles.py \
  --urdf src/dexmate_vega_description/urdf/vega_1p_f5d6.package.urdf \
  --srdf src/dexmate_vega_moveit_config/config/vega_1p_f5d6.srdf \
  --collision-urdf src/dexmate_vega_description/robots/humanoid/vega_1p/vega_1p_f5d6_collision_spheres.collision.urdf \
  --package-dir src
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

## F5D6 Hand Control

Each Pico controller drives its corresponding hand independently. The analog
trigger proportionally curls the thumb, index, middle, ring, and little-finger
flexion drivers (`th_j1`, `ff_j1`, `mf_j1`, `rf_j1`, and `lf_j1`). The analog
grip controls only thumb opposition (`th_j0`). The configured six-command order
and Dexmate vendor endpoints are:

```yaml
left_hand_joint_names: [L_th_j1, L_ff_j1, L_mf_j1, L_rf_j1, L_lf_j1, L_th_j0]
right_hand_joint_names: [R_th_j1, R_ff_j1, R_mf_j1, R_rf_j1, R_lf_j1, R_th_j0]
left_hand_open_positions: [0.1834, 0.2891, 0.2801, 0.2840, 0.2811, -0.0158]
left_hand_closed_positions: [-0.1, -1.0946, -1.0844, -1.0154, -1.0118, 0.84]
right_hand_open_positions: [0.1834, 0.2891, 0.2801, 0.2840, 0.2811, -0.0158]
right_hand_closed_positions: [-0.1, -1.0946, -1.0844, -1.0154, -1.0118, 0.84]
```

These are absolute positions; neutral calibration does not capture or change
the hand endpoints. Each hand configuration must contain the exact six joint
names in this order, six open values, six closed values, and only finite
numbers. If one side is invalid, that side publishes no hand command and its
`*_hand_config_valid`/`*_hand_config_error` status fields explain why. Status
also reports each side's current `*_hand_trigger` and `*_hand_grip` values.

Hand targets publish directly to `/left_hand/joint_commands` and
`/right_hand/joint_commands`. They intentionally bypass Pink, numeric arm IK,
arm collision barriers, and joint slew limiting; there is no finger collision
planner. The explicit vendor endpoints are the command bounds, and normal
teleop enable, hold, calibration, and stale-input gates still apply. Verify the
full motion in dry-run before enabling real command publication.

## MeshCat Dry-Run Visualization

Use this test before sending anything to the real robot. Teleop runs with
`publish_commands:=false`, publishes only `/dex_pico_teleop/log_frame`, and
uses a rate- and acceleration-limited simulated position servo as the next
tick's IK feedback. MeshCat renders that simulated posture and the reduced
collision spheres in a browser. This launch does not start `dexcontrol_bridge`;
if `/joint_states` is already present, MeshCat uses it only as the initial pose
before the first teleop log frame.

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
from another machine on the same network. Start the stack with the **Combined
MeshCat Dry Run** command in [Runtime Commands](#runtime-commands).

In XRoboToolkit on the Pico headset:

1. Set the connection mode to TCP.
2. Set the PC/service address to the Jetson IP, for example `10.233.169.139`.
3. Set the port to `63901`.
4. Keep the headset IP separate from the Jetson IP; both devices must have
   different addresses.
5. Enable tracking send for the headset and controllers, then reconnect.

Open the MeshCat URL in the desktop browser. If you are browsing from the
desktop/laptop, use `http://<jetson-ip>:7000/static/`, not `127.0.0.1`.

After Pico data is connected, hold each calibration posture still for at least
`0.4 s` before calling its service:

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
3. Each trigger proportionally curls all five digits on that side, while each
   grip independently rotates only that side's thumb into opposition.
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
`right_arm`, `left_hand`, and `right_hand` arrays plus optional `debug`
retarget/timing data when teleop is enabled. The MeshCat node subscribes to
`/dex_pico_teleop/log_frame`, expands the hand drivers to the URDF mimic joints,
moves the simulated model, and never publishes robot commands. With both
controllers released, both hands should show the configured open pose. Pull
only the left trigger and confirm only the left five digits curl; then hold its
trigger and vary its grip to confirm only left thumb opposition changes. Repeat
on the right and confirm neither controller moves the other hand.

## Calibration And Safety

1. Start `dexcontrol_bridge` and verify `/joint_states`.
2. Start this node with `publish_commands:=false` first.
3. Stand upright facing forward with both ankle trackers visible, elbows
   straight, and arms relaxed vertically beside the torso.
4. Hold still for at least `0.4 s`, then click right `A` or call
   `ros2 service call /dex_pico_teleop/calibrate std_srvs/srv/Trigger {}`.
5. Keep the same body orientation, extend both elbows straight forward at about
   shoulder height, hold still for `0.4 s`, and click right `B` or call
   `ros2 service call /dex_pico_teleop/calibrate_reach std_srvs/srv/Trigger {}`.
6. Enable teleop with left `Y` or `ros2 service call /dex_pico_teleop/enabled std_srvs/srv/SetBool "{data: true}"`.
7. Re-launch with `publish_commands:=true` only after status looks correct.

Arm motion uses posture retargeting. A configurable controller-local offset
first converts each tracking origin to the operator hand point. The `A`/`B`
postures then fit each shoulder's fore-aft position and arm length, and the
shoulder-to-hand vector is rotated into the robot `arm_center` frame before IK.
Set `left_controller_to_hand_point_xyz_m` and
`right_controller_to_hand_point_xyz_m` in the YAML, then use the raw-controller
and corrected-hand markers in MeshCat to tune them before hardware operation.
If reach calibration has not completed, the node falls back to the configured
shoulder geometry and height-based arm estimate. Base motion
follows the controller joysticks and publishes zero velocity whenever input is
stale, held, disabled, uncalibrated, or inside the joystick deadzone. With the
default base mapping, the left controller joystick commands body-frame
forward/strafe velocity and the right controller horizontal joystick axis
commands yaw velocity. Head joint tracking can be disabled with
`head_tracking_enabled:=false`; in that mode the node commands the head to a
fixed forward pose pitched down by `head_disabled_pitch_deg`.

Hand motion uses the absolute F5D6 endpoints documented above and is independent
of the neutral calibration pose. An invalid configuration disables only the
affected hand rather than publishing a partial or non-finite command.

## XRoboToolkit Remote Vision

The head-camera vision adapter is separate from teleop. It subscribes directly
to the latest DexTop RGB and depth frames through DexComm. RGB is submitted to
a capacity-one `dexcomm.rtc.VideoPublisher` worker for XRoboToolkit Remote
Vision. Resize/encode/network work cannot block acquisition. Depth is monitored
for rate, shape, and timing but is not sent to the headset. No raw ROS image
topic is produced.

Install the video dependencies once in the robot environment:

```bash
python3 -m pip install dexcomm-video==0.4.19 av==17.1.0
```

Start the camera sensor service and adapter with the **Head-Camera Stream**
commands in [Runtime Commands](#runtime-commands).

Default stream:

- Dexmate source: `head_camera.left_rgb`
- Dexmate source transport: `zenoh`
- Direct RGB topic: `sensors/head_camera/left_rgb` (`uint8 RGB`)
- Direct depth topic: `sensors/head_camera/depth` (`float32` metres)
- XRoboToolkit RTC channel:
  `xrobotoolkit/remote_vision/head_camera/left_rgb_rtc`
- Output: `1280x720`, `30 FPS`, H.264 preferred with VP8 fallback, `1.5 Mbps`

Configure XRoboToolkit Remote Vision to subscribe to the RTC channel above. The
runtime section also shows how to pass a different `rtc_channel`.

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

On the Jetson/ROS host, use the **Head-Camera Stream** ZED Mini command in
[Runtime Commands](#runtime-commands).

The node duplicates the mono left RGB stream into a side-by-side stereo H.264
frame because XRoboToolkit's ZED Mini path expects stereo video. This
compatibility encoder/socket has its own capacity-one worker and is disabled by
default. If enabled, the nested status object `xrtcp.connected` becomes true and
`xrtcp.published_frames` increases.

Useful status and runtime controls:

```bash
ros2 topic echo --once --full-length /dex_pico_teleop/head_camera_vision/status
ros2 service call /dex_pico_teleop/head_camera_vision/enabled std_srvs/srv/SetBool "{data: false}"
ros2 service call /dex_pico_teleop/head_camera_vision/enabled std_srvs/srv/SetBool "{data: true}"
```

The status topic reports independent RGB/depth source FPS, source/receive ages,
transport delay, output queue replacements, processing time, failures,
connection state, and subscriber count. During teleop validation, compare this
with `/dex_pico_teleop/log_frame` and `/joint_states`; video may drop or stall
independently without making Pico input stale or changing command timing.
