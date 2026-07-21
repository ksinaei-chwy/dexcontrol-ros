# Dexmate Vega LeRobot recorder

`dex_vega_lerobot_recorder` records reviewed teleoperation demonstrations as a
local Hugging Face LeRobotDataset v3. The robot command loop remains unchanged:
the recorder only observes post-bridge command audit topics, measured feedback,
and a direct DexComm head-camera stream. Disk/video work runs on a bounded worker
queue outside ROS subscription callbacks, and Hub synchronization is secondary
to the finalized local dataset.

The implementation was validated against `lerobot==0.4.4`. It uses the installed
`LeRobotDataset.create`, `add_frame`, `save_episode`, `clear_episode_buffer`,
`finalize`, local constructor-based append, and `push_to_hub` APIs rather than
maintaining a private dataset format.

## Data flow

```text
Pico teleop -> ROS command topics -> dexcontrol bridge -> vendor robot API
                                         |
                                         +-> applied command audit topics --+
Dexcontrol hardware feedback ------------+-> measured feedback topics ------+-->
DexTop RGB -> direct latest-frame DexComm source ---------------------------+   fixed-rate
optional wrist camera sources -----------------------------------------------+   snapshot
                                                                                |
                                                                                v
                         IDLE -> RECORDING -> REVIEW_PENDING -> SAVING -> IDLE
                                             |                  |
                                             +---- discard -----+
                                                                                |
                                                                                v
                                      local LeRobotDataset v3 -> optional Hub sync
```

The state machine, camera sources, feature adapter, and dataset writer are
separate from ROS message acquisition so they can be unit tested without a
robot. Only `save_episode()` commits an episode. Stopping leaves it pending;
discarding clears LeRobot's pending image/frame buffer and never increments the
episode count.

## Repository interfaces discovered

The full inspection record is in
[`docs/ROS_INTERFACE_DISCOVERY.md`](docs/ROS_INTERFACE_DISCOVERY.md). The current
interfaces are:

| Purpose | Topic or vendor stream | Type / format | Rate found |
| --- | --- | --- | --- |
| Measured joint positions | `/joint_states` | `sensor_msgs/msg/JointState` | 100 Hz bridge default |
| Exact applied joint target audit | `/dexcontrol/applied_joint_commands` | `sensor_msgs/msg/JointState` | 250 Hz bridge loop default |
| Measured base velocity | `/dexcontrol/measured_base_twist` | `geometry_msgs/msg/TwistStamped` | 50 Hz odom loop default |
| Exact applied base velocity audit | `/dexcontrol/applied_base_twist` | `geometry_msgs/msg/TwistStamped` | 250 Hz bridge loop default |
| Recorder head image | `sensors/head_camera/left_rgb` | DexComm Zenoh `uint8` RGB, 960 x 600 observed | 30 Hz measured |
| Head depth diagnostics | `sensors/head_camera/depth` | DexComm Zenoh `float32` metres, 960 x 600 observed | 29.7 Hz measured |

Pico teleop publishes component `JointState` commands on
`/torso/joint_commands`, `/head/joint_commands`, `/left_arm/joint_commands`,
`/right_arm/joint_commands`, `/left_hand/joint_commands`, and
`/right_hand/joint_commands`, plus `geometry_msgs/msg/Twist` on `/cmd_vel`.
Those are intermediate: the bridge still checks finite values, clips joints and
base velocities, and applies the base watchdog. The recorder therefore uses the
new read-only audit topics. A normal audit represents successful vendor calls.
If a joint dispatch is partial, the audit contains only successful joint names;
if a chassis dispatch fails, its audit contains a non-finite invalidation
marker. The recorder clears the affected cache and drops the tick rather than
reusing an ambiguous recent command.

The recorder and Pico vision node now have independent direct, capacity-one
DexComm subscribers. A blocked headset encoder cannot reduce recorder input
FPS. The Pico node publishes RTC Remote Vision from a bounded worker; optional
ZED TCP runs on another worker. No raw ROS image topic is created. The recorder
validates DexTop source and local receive timestamps independently and resizes
RGB to the configured dataset resolution, 640 x 480 by default.

Live F5D6 validation showed that both hands provide measured joint positions but
do not provide `get_joint_vel()` feedback. `/joint_states` inserts compatibility
zeros for those missing velocities, so the recorder uses only its measured
positions and does not include any joint velocity in `observation.state`.
`/dexcontrol/measured_joint_states` remains available for configurations where
every component provides real velocity, but it is empty on the current hardware
and is not the recorder input.

## Fixed dataset schema

Every frame contains the enabled camera features plus:

- `observation.images.head`: required real `uint8` RGB HWC head image.
- `observation.images.left_wrist`: optional `uint8` RGB HWC wrist image or
  placeholder.
- `observation.images.right_wrist`: optional `uint8` RGB HWC wrist image or
  placeholder.
- `observation.state`: 27 measured/derived `float32` values.
- `action`: 27 post-bridge commanded/derived `float32` values.
- `task`: the configured natural-language task instruction.

With `use_videos: true`, every enabled image feature is a LeRobot video feature.
The task is registered through LeRobot's current per-frame task mechanism.
LeRobot also receives `robot_type: dexmate_vega_1_pro`. Camera feature presence
is immutable within a dataset: do not mix head-only, placeholder-wrist, and
real-wrist episodes in one dataset.

### Ordered source joint list

The 32-joint order is:

```text
torso_j1 torso_j2 torso_j3
head_j1 head_j2 head_j3
L_arm_j1 L_arm_j2 L_arm_j3 L_arm_j4 L_arm_j5 L_arm_j6 L_arm_j7
R_arm_j1 R_arm_j2 R_arm_j3 R_arm_j4 R_arm_j5 R_arm_j6 R_arm_j7
L_th_j1 L_ff_j1 L_mf_j1 L_rf_j1 L_lf_j1 L_th_j0
R_th_j1 R_ff_j1 R_mf_j1 R_rf_j1 R_lf_j1 R_th_j0
```

Incoming `JointState` arrays are always looked up by name and reordered; their
wire order is never trusted. The twelve hand driver positions are reduced to
the same two logical ratios used by Pico teleoperation. They do not appear as
twelve independent dataset features.

For each hand, `open_close_ratio` is `0` at the configured open endpoint and
`1` at the configured closed endpoint. `thumb_opposition_ratio` is `0` at the
unopposed endpoint and `1` at the opposed endpoint. Measured hand state is
projected from the six measured driver positions into these two coordinates.

### `observation.state` (27 values)

| Indices | Quantity | Units / frame | Semantics |
| --- | --- | --- | --- |
| 0:3 | Torso joint positions | rad, joint-local | absolute, measured |
| 3:6 | Head joint positions | rad, joint-local | absolute, measured |
| 6:13 | Left-arm joint positions | rad, joint-local | absolute, measured |
| 13:20 | Right-arm joint positions | rad, joint-local | absolute, measured |
| 20:22 | Left `open_close_ratio`, `thumb_opposition_ratio` | ratio `[0,1]` | derived from measured driver positions |
| 22:24 | Right `open_close_ratio`, `thumb_opposition_ratio` | ratio `[0,1]` | derived from measured driver positions |
| 24:26 | `base_vx`, `base_vy` | m/s, ROS `base` | measured rate |
| 26:27 | `base_wz` | rad/s, ROS `base` | measured rate |

### `action` (27 values)

| Indices | Quantity | Units / frame | Semantics |
| --- | --- | --- | --- |
| 0:3 | Torso joint targets | rad, joint-local | absolute command after bridge validation/clipping |
| 3:6 | Head joint targets | rad, joint-local | absolute command after bridge validation/clipping |
| 6:13 | Left-arm joint targets | rad, joint-local | absolute command after bridge validation/clipping |
| 13:20 | Right-arm joint targets | rad, joint-local | absolute command after bridge validation/clipping |
| 20:22 | Left `open_close_ratio`, `thumb_opposition_ratio` | ratio `[0,1]` | reconstructed from post-bridge applied targets |
| 22:24 | Right `open_close_ratio`, `thumb_opposition_ratio` | ratio `[0,1]` | reconstructed from post-bridge applied targets |
| 24:26 | `base_vx`, `base_vy` | m/s, ROS `base` | velocity handed to the chassis API |
| 26:27 | `base_wz` | rad/s, ROS `base` | velocity handed to the chassis API |

All five flexion targets must reconstruct to one applied open/close ratio within
the configured tolerance. An off-synergy or differently clipped hand command
drops the whole sample rather than being compressed lossily. The configured
open/closed endpoints must match Pico teleoperation and are copied into every
dataset's feature specification.

The ROS base convention is +x forward, +y left, +z up. No commanded value or
compatibility/fabricated joint velocity is substituted into
`observation.state`, and no measured value is substituted into `action`.
Missing, duplicate, non-finite, dimensionally invalid, future-dated, or stale
inputs drop the whole recording tick. The runtime reports requested and
achieved FPS, dropped and stale samples, queue depth, pending duration, and
pending frame count.

The effective YAML (including command-line overrides) and a machine-readable
feature specification with dimensions, ordering, units, frames, and provenance
are stored in each dataset's `meta/vega_recording_config.yaml` and
`meta/vega_feature_specification.json`.

## Wrist placeholders

Both wrist sources currently use cached all-zero RGB arrays. Their effective
height and width always match the processed head frame, even though the YAML
retains independent future wrist resolutions. The allocation is reused until
the reference shape changes.

> Black placeholder images are real model inputs, not masks. LeRobot and π₀.₅
> do not automatically infer that they are absent cameras. Do not mix test
> episodes containing black wrist frames into a production training dataset
> without an explicit data/model decision.

When hardware is installed, set each wrist `stream_name`, direct DexComm
`topic`, transport, codec, and real resolution, then set
`placeholder_enabled: false`. `DirectRgbCameraSource` will validate `uint8 RGB`,
resize, and enforce source/receive freshness without changing dataset keys or
the episode state machine. The vendor configuration mentions
`sensors/left_wrist_camera/rgb` and `sensors/right_wrist_camera/rgb`, but those
physical streams have not yet been validated.

### Head-only datasets

Set `cameras.left_wrist.enabled: false` and
`cameras.right_wrist.enabled: false` to omit those streams completely. The
resulting LeRobot dataset contains only `observation.images.head`; it does not
write black placeholder videos. Camera feature presence is part of the dataset
schema, so use a new dataset name and do not resume or mix a head-only dataset
with a three-camera dataset.

## Installation and build

Use the robot's existing ROS Humble environment. LeRobot is deliberately a
runtime Python dependency rather than a rosdep package:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
python3 -m pip install "setuptools>=71,<80" "lerobot==0.4.4"
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
colcon build --symlink-install --packages-select \
  dex_camera_transport dexcontrol_ros dex_pico_teleop dex_vega_lerobot_recorder
source install/setup.bash
```

The video-backed default needs a working FFmpeg/PyAV encoder supplied by the
LeRobot installation. Install LeRobot into the same Python environment used by
`ros2`; do not mix a different virtual environment with the ROS node process.
The Setuptools constraint above satisfies both LeRobot (`>=71`) and the
installed colcon release (`<80`). Verify the interpreter and dependency set
after installation with:

```bash
python3 -c 'import sys, lerobot, huggingface_hub; print(sys.executable, lerobot.__version__, huggingface_hub.__version__)'
python3 -m pip check
```

The recorder receives decoded NumPy RGB arrays directly from DexComm and does
not use `sensor_msgs/Image` or `cv_bridge`.

## Configure and launch

For a cautious first live-hardware acceptance run, follow
[`docs/LIVE_ROBOT_TEST_PROCEDURE.md`](docs/LIVE_ROBOT_TEST_PROCEDURE.md). The
production procedure below assumes the operator, safety observer, and clear
work area are ready for real robot motion.

### Production configuration: `dexmate_blue_bird`

[`config/dexmate_blue_bird.yaml`](config/dexmate_blue_bird.yaml) is the
production configuration for the task:

```text
put the blue bird on the meeting desk
```

It creates the private local/Hub dataset `Kasra99/dexmate_blue_bird`, records
the real head camera only at 30 FPS and 640 x 480, and uploads committed data
only when the session is finalized. It contains no wrist-camera features and no
black placeholder videos. Do not resume the old three-camera smoke dataset with
this configuration.

Use the following common setup in every terminal:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### Terminal 1: Dexmate services and ROS bridge

After each robot boot, start the camera service and bridge. The shown robot
endpoint is the current Vega Ethernet endpoint; replace it only if the robot is
configured differently.

```bash
dextop node start
dexsensor launch --sensor camera

ros2 launch dexcontrol_ros dexcontrol_bridge.launch.py \
  robot_ip:=192.168.50.20:7447
```

Before moving on, verify live feedback:

```bash
ros2 topic hz /joint_states
ros2 topic hz /dexcontrol/applied_joint_commands
ros2 topic hz /dexcontrol/measured_base_twist
ros2 topic hz /dexcontrol/applied_base_twist
```

### Terminal 2: Pico teleoperation

Launch real teleoperation only when the safety observer is ready:

```bash
ros2 launch dex_pico_teleop pico_teleop.launch.py \
  publish_commands:=true \
  network_transport:=tcp \
  network_host:=0.0.0.0 \
  network_port:=63901 \
  control_rate_hz:=50.0
```

In XRoboToolkit, point tracking/control to the ROS computer's Wi-Fi IP on port
`63901` (the computer was `192.168.0.231` on the validated network; confirm it
locally if the network changes). Calibrate and enable teleoperation before the
first demonstration:

```bash
ros2 service call /dex_pico_teleop/calibrate std_srvs/srv/Trigger '{}'
ros2 service call /dex_pico_teleop/calibrate_reach std_srvs/srv/Trigger '{}'
ros2 service call /dex_pico_teleop/enabled std_srvs/srv/SetBool "{data: true}"
```

### Terminal 3: broadcast head camera to the Pico headset

The recorder receives the head camera directly from DexComm; this launch is for
the operator's headset view and does not add a ROS image hop to the recorder.
The currently validated Pico headset address is `192.168.0.23`:

```bash
ros2 launch dex_pico_teleop head_camera_vision.launch.py \
  rtc_enabled:=false \
  xrtcp_enabled:=true \
  xrtcp_host:=192.168.0.23 \
  xrtcp_port:=12345 \
  fps:=30.0 \
  xrtcp_bitrate:=3000000 \
  xrtcp_write_timeout_s:=2.0
```

In XRoboToolkit on the Pico, select `ZEDMINI`, press `Listen`, keep port
`12345`, and leave the live-camera view open. Confirm that the status shows a
healthy direct source and active headset sender:

```bash
ros2 topic echo --once --full-length \
  /dex_pico_teleop/head_camera_vision/status
```

Expect `rgb.source_fps` near 30, `xrtcp.connected: true`, and increasing
`xrtcp.published_frames`. If the Pico receives a different DHCP address, update
only `xrtcp_host`; do not use the robot Ethernet address for that setting.

### Terminal 4: record the production dataset

The account must already be authenticated with `hf auth login`. Do not pass
`no_hf_upload:=true`, because that command-line option intentionally overrides
the production YAML and disables upload.

```bash
ros2 launch dex_vega_lerobot_recorder record_teleop_dataset.launch.py \
  config_file:=/workspaces/dexcontrol-ros/ros_ws/src/dex_vega_lerobot_recorder/config/dexmate_blue_bird.yaml \
  no_hf_upload:=false
```

Wait for `recorder ready at 30 FPS` and `terminal keyboard input active`. Use
`a` to start, `b` to stop for review, `c` to commit, and `d` to discard. At the
end of the collection session, return to `IDLE` and press Ctrl+C once. The
recorder finalizes local data first, then uploads only committed episodes to
`Kasra99/dexmate_blue_bird`.

With this configuration, **Ctrl+C starts the session-end upload; it does not
mean that the upload has already completed.** Keep the terminal open and do not
press Ctrl+C again until both of these appear:

```text
uploaded committed dataset to Kasra99/dexmate_blue_bird
process has finished cleanly
```

The recorder launch allows 30 minutes for this graceful shutdown/upload by
default, which is necessary for large video datasets on a slow connection. Set
`shutdown_upload_timeout_s:=<seconds>` if a longer window is needed. If the
upload fails or is interrupted, the finalized local dataset is still intact and
can be uploaded safely later:

```bash
ros2 run dex_vega_lerobot_recorder upload_dataset \
  --local-directory /workspaces/dexcontrol-ros/ros_ws/data/production/dexmate_blue_bird \
  --repo-id Kasra99/dexmate_blue_bird \
  --private
```

### Other configurations

Copy [`config/vega_lerobot_recording.yaml`](config/vega_lerobot_recording.yaml)
for a three-camera/placeholder experiment. Set an absolute
`dataset.local_save_directory`, a unique name, and a task instruction before
each campaign. Confirm the configured 32 joint names against one live
`/joint_states` message.

Local-only recording, with the command-line/launch override taking precedence
over YAML:

```bash
ros2 launch dex_vega_lerobot_recorder record_teleop_dataset.launch.py \
  config_file:=/absolute/path/to/vega_lerobot_recording.yaml \
  no_hf_upload:=true
```

The node prints the exact local dataset path and does not initialize a Hub
operation, require a token, or make an upload request. To run the executable
directly, the equivalent hard override is:

```bash
ros2 run dex_vega_lerobot_recorder record_teleop_dataset \
  --config-file /absolute/path/to/vega_lerobot_recording.yaml \
  --no-hf-upload
```

Existing non-empty paths are refused. Choose exactly one explicit policy when
appropriate:

```bash
# Safely append to an existing finalized local LeRobotDataset.
ros2 launch dex_vega_lerobot_recorder record_teleop_dataset.launch.py \
  config_file:=/absolute/path/to/config.yaml resume:=true no_hf_upload:=true

# Deliberately replace this configured dataset path.
ros2 launch dex_vega_lerobot_recorder record_teleop_dataset.launch.py \
  config_file:=/absolute/path/to/config.yaml overwrite:=true no_hf_upload:=true
```

## Episode controls

The defaults are:

| Key | Operation | Valid state and result |
| --- | --- | --- |
| `a` | start | `IDLE` -> `RECORDING`; resets candidate counters only |
| `b` | stop | `RECORDING` -> `REVIEW_PENDING`; does not commit |
| `c` | save | valid pending candidate -> one committed episode |
| `d` | discard | active or pending candidate -> `IDLE`; commits nothing |

Saving rejects empty episodes and candidates below the configured frame or
duration minimum. A pending candidate blocks a new start. Key presses are
debounced with a monotonic clock. Linux input events act only on key-down
(`value == 1`), so held-key repeats and release events do not trigger state
changes.

With `input_backend: terminal`, the recorder first uses interactive standard
input. ROS 2 launch commonly gives a node pipe-backed standard input, so the
backend then opens the foreground session's controlling terminal at `/dev/tty`.
Keys do not require Enter. Confirm this startup message before demonstrating:

```text
terminal keyboard input active
```

If neither input is available, the recorder reports `no controlling TTY` and
the ROS services below remain available.

The same operations are available without a TTY:

```bash
ros2 service call /dex_vega_lerobot_recorder/start std_srvs/srv/Trigger '{}'
ros2 service call /dex_vega_lerobot_recorder/stop std_srvs/srv/Trigger '{}'
ros2 service call /dex_vega_lerobot_recorder/save std_srvs/srv/Trigger '{}'
ros2 service call /dex_vega_lerobot_recorder/discard std_srvs/srv/Trigger '{}'
```

Ctrl+C never silently commits a candidate. The default is to warn, discard any
active/review-pending episode, then finalize committed episodes. Set
`autosave_on_shutdown: true` only if that explicitly desired behavior is
acceptable; even then an invalid candidate is discarded.

### Physical foot pedals

For pedals that appear as an ordinary terminal keyboard, keep
`input_backend: terminal`. The recorder must run from a foreground terminal or
have access to that session's `/dev/tty`. For a headless service, set:

```yaml
episode_control:
  input_backend: linux_input_event
  input_device: /dev/input/event7
```

Find the stable device with `ls -l /dev/input/by-id/` or `libinput list-devices`
and prefer the corresponding `/dev/input/by-id/...-event-kbd` symlink. The
process needs read permission; commonly this means adding the service user to
the host `input` group or installing a narrowly scoped udev rule for that
pedal's vendor/product ID. Do not make all input devices world-writable. The
device path can be overridden at launch with `input_backend:=linux_input_event`
and `input_device:=/dev/input/by-id/...`. No X11 session is used.

## Hugging Face synchronization

Local data is always the source of truth. To upload at recording time, set a
namespace or explicit full repo ID, enable upload, and select `manual`,
`each_episode`, or `on_session_end` (recommended/default). Authenticate with
the standard local login or environment variable; never place a token in YAML:

```bash
hf auth login
# Or export HF_TOKEN through the service's secret/environment mechanism.
```

```bash
ros2 launch dex_vega_lerobot_recorder record_teleop_dataset.launch.py \
  config_file:=/absolute/path/to/hf_enabled_config.yaml \
  no_hf_upload:=false
```

`no_hf_upload:=false` does not force upload; it allows the YAML setting to take
effect. `private: true` is supported. Only finalized committed episodes are
uploaded. Authentication/network failures are reported and leave the complete
local dataset intact.

Upload an already finalized dataset later:

```bash
ros2 run dex_vega_lerobot_recorder upload_dataset \
  --local-directory /data/lerobot/vega_teleop_demonstrations \
  --repo-id my_hf_namespace/vega_teleop_demonstrations \
  --private
```

Use `--public` instead of `--private` only after reviewing the data.

## Inspect and visualize a dataset

Reload locally through the same public dataset class:

```bash
python3 - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="my_hf_namespace/vega_teleop_demonstrations",
    root="/data/lerobot/vega_teleop_demonstrations",
)
print(dataset.num_episodes, dataset.num_frames)
print(dataset.features)
print(dataset[0]["observation.state"].shape, dataset[0]["action"].shape)
PY
```

For an uploaded dataset, use LeRobot's visualizer:

```bash
lerobot-dataset-viz \
  --repo-id my_hf_namespace/vega_teleop_demonstrations \
  --episode-index 0
```

## π₀.₅ fine-tuning

Keep the dataset's stable camera names. `dexmate_blue_bird` is intentionally a
head-only dataset, so map only its one visual feature at training time. The
training configuration must likewise expose only that camera (or explicitly
handle missing cameras); do not put nonexistent wrist keys in the rename map:

```bash
lerobot-train \
  --dataset.repo_id=Kasra99/dexmate_blue_bird \
  --policy.path=lerobot/pi05_base \
  --rename_map='{"observation.images.head":"observation.images.base_0_rgb"}' \
  --output_dir=outputs/pi05_dexmate_blue_bird \
  --batch_size=8 \
  --steps=30000 \
  --wandb.enable=false
```

The 27-D Vega vectors fit within π₀.₅'s default 32-D state/action padding, so no
dimension override is needed and the pretrained 32-D projections retain their
shape. The processor pads the final five entries and masks them as appropriate;
the 27 real values retain the exact ordering above. In LeRobot 0.4.4 π₀.₅ does
not feed `observation.state` into its core forward pass, although the 27-D
measured/derived state is retained for future policy versions and analysis.

Placeholder wrist frames are not model masks. Before production training,
either replace them with real cameras, deliberately train with the black inputs,
or add an explicit training-time camera/mask policy.

## Tests

```bash
source /opt/ros/humble/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src/dex_vega_lerobot_recorder:$PYTHONPATH \
python3 -m pytest -q src/dex_vega_lerobot_recorder/test

colcon test --packages-select \
  dexcontrol_ros dex_pico_teleop dex_vega_lerobot_recorder
colcon test-result --verbose
```

The real round-trip test is skipped if LeRobot is absent; with LeRobot present
it writes AV1-backed camera videos, finalizes one episode, reloads it through
`LeRobotDataset`, decodes a frame, and checks configured camera features,
state/action dimensions, and the task. The mocked ROS integration publishes synthetic
`JointState`, `TwistStamped`, and BGR `Image` messages into the real recorder
node and verifies exactly one committed episode.

## Safety notes

The recorder never publishes robot commands, changes safety limits, starts
motion, or modifies collision/filter logic. It does not upload pending or
discarded data. Keep generated datasets, credentials, temporary frames, videos,
and Parquet files outside this source tree; repository ignore rules cover common
accidental local paths but are not a substitute for choosing a dedicated data
volume.
