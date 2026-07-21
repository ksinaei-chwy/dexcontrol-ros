# Live robot recording acceptance procedure

This runbook validates the Dexmate Vega 1 Pro teleoperation recorder on real
hardware without uploading data. Use ROS services for the first acceptance run;
connect foot pedals only after this procedure passes.

The recorder is read-only with respect to robot control. Its start/stop/discard
controls affect dataset capture only and do **not** stop robot motion.

## 1. Safety and operator readiness

Before starting any process:

1. Clear the robot's swept volume and keep other people outside it.
2. Confirm the physical emergency stop is reachable and tested according to
   the robot's normal operating procedure.
3. Use a second person as safety observer for the first real-motion run when
   possible.
4. Begin with Pico teleoperation in `publish_commands:=false` dry-run mode.
5. Keep the first recording local-only, with
   `hugging_face.upload_enabled: false`, `no_hf_upload:=true`, and
   `autosave_on_shutdown: false`.
6. Do not use recorder keys or services as an emergency stop. If motion is
   unsafe, use the physical emergency stop first. The secondary software
   controls are:

   ```bash
   ros2 service call /dex_pico_teleop/enabled std_srvs/srv/SetBool "{data: false}"
   ros2 service call /dex_pico_teleop/hold std_srvs/srv/SetBool "{data: true}"
   ros2 service call /soft_estop std_srvs/srv/SetBool "{data: true}"
   ```

Do not release a physical or software emergency stop until the cause of the
unsafe condition is understood and the robot is ready under its normal safety
procedure.

## 2. Prepare a persistent recording configuration

Do not use `/tmp` for a real recording. Choose a persistent volume with enough
space and replace the placeholders below. The dataset path is
`<PERSISTENT_DATA_ROOT>/<DATASET_NAME>`.

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Examples only: replace these with a mounted persistent path and unique name.
export PERSISTENT_DATA_ROOT=/data/lerobot
export DATASET_NAME=vega_live_smoke_YYYYMMDD_HHMM
export LIVE_CONFIG=/data/lerobot/config/vega_live_smoke_YYYYMMDD_HHMM.yaml

mkdir -p "$PERSISTENT_DATA_ROOT/config"
cp src/dex_vega_lerobot_recorder/config/vega_lerobot_recording.yaml "$LIVE_CONFIG"
${EDITOR:-nano} "$LIVE_CONFIG"
df -h "$PERSISTENT_DATA_ROOT"
```

Set and verify these YAML values:

```yaml
dataset:
  name: "vega_live_smoke_YYYYMMDD_HHMM"
  local_save_directory: "/data/lerobot"
  recording_fps: 20
  task_instruction: "<the exact task the operator will demonstrate>"
  robot_type: "dexmate_vega_1_pro"
  use_videos: true

hugging_face:
  upload_enabled: false

episode_control:
  autosave_on_shutdown: false
```

Keep both wrist cameras in placeholder mode for this test. Their effective
resolution will match the configured 640 x 480 head image. Black wrist images
are real pixels, not model masks, so this smoke-test dataset must not be mixed
into production training data without an explicit decision.

Keep the recorder's `robot_features.hand_synergies` endpoints identical to the
active Pico teleop `left_hand_*_positions` and `right_hand_*_positions`. The
dataset stores two ratios per hand—open/close and thumb opposition—in both
state and action. A mismatch can make valid applied targets appear off-synergy
and will correctly cause dropped samples.

For the first test, do not use `overwrite` or `resume`. Confirm the target does
not already exist; if it does, choose a new dataset name:

```bash
test ! -e "$PERSISTENT_DATA_ROOT/$DATASET_NAME"
```

## 3. Common terminal setup

Run this in every terminal used below:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

The commands assume the ROS processes run in the robot-side environment where
`dexcontrol`, DexComm, and the robot communication configuration are available.

## 4. Start robot services and the ROS bridge

After every robot boot, start the required Dexmate services. The camera service
is required for the recording test; lidar is optional for this procedure.

```bash
dextop node start
dexsensor launch --sensor camera
```

In terminal 1, start the bridge. Supply the real endpoint if discovery is not
being used:

```bash
ros2 launch dexcontrol_ros dexcontrol_bridge.launch.py \
  robot_ip:=<ROBOT_IP>:7447
```

Do not continue until measured joint positions are present. Each
`ros2 topic hz` command runs until Ctrl+C:

```bash
ros2 topic hz /joint_states
ros2 topic hz /dexcontrol/measured_base_twist
ros2 topic echo /joint_states --once --field name
```

Expected defaults are approximately 100 Hz measured joints and 50 Hz measured
base twist. Confirm that `/joint_states` contains the configured 32 joint names
and measured positions. The recorder intentionally ignores joint velocity
because the live F5D6 hands do not expose it. Missing required positions still
cause the recorder to drop the whole sample.

## 5. Validate Pico teleoperation without commanding the robot

In terminal 2, launch dry-run teleoperation:

```bash
ros2 launch dex_pico_teleop pico_teleop.launch.py \
  publish_commands:=false \
  network_transport:=tcp \
  network_host:=0.0.0.0 \
  network_port:=63901 \
  control_rate_hz:=50.0
```

Configure XRoboToolkit to send tracking over TCP to the robot/Jetson address on
port `63901`. When tracking is fresh, perform both calibrations:

1. Stand facing forward with arms relaxed beside the torso. Hold still for at
   least 0.4 seconds, then click right `A` or call neutral calibration.
2. Extend both arms straight forward near shoulder height. Hold still for at
   least 0.4 seconds, then click right `B` or call reach calibration.
3. Enable the dry-run with left `Y` or the service below.

```bash
ros2 service call /dex_pico_teleop/calibrate std_srvs/srv/Trigger '{}'
ros2 service call /dex_pico_teleop/calibrate_reach std_srvs/srv/Trigger '{}'
ros2 service call /dex_pico_teleop/enabled std_srvs/srv/SetBool "{data: true}"
ros2 topic echo /dex_pico_teleop/status --once --full-length
ros2 topic hz /dex_pico_teleop/log_frame
```

Acceptance gate:

- `calibrated: true`
- `enabled: true`
- `stale_input: false`
- `log_frame` near the configured 50 Hz
- finite joint targets and plausible controller/hand behavior
- `loop_p99_ms` comfortably below the 20 ms control period

Disable teleoperation and stop this dry-run node with Ctrl+C:

```bash
ros2 service call /dex_pico_teleop/enabled std_srvs/srv/SetBool "{data: false}"
```

## 6. Start direct head-camera transport and RTC broadcasting

In terminal 3:

```bash
ros2 launch dex_pico_teleop head_camera_vision.launch.py \
  camera_topic:=sensors/head_camera/left_rgb \
  depth_enabled:=true \
  depth_topic:=sensors/head_camera/depth \
  rtc_enabled:=true
```

In XRoboToolkit, configure Remote Vision to subscribe to
`xrobotoolkit/remote_vision/head_camera/left_rgb_rtc`. Verify the bridge before
starting the recorder:

```bash
ros2 topic echo /dex_pico_teleop/head_camera_vision/status --once --full-length
```

Expected status is `streaming`, with `transport: direct_dexcomm`, RGB and depth
`source_fps` near 30, RGB shape `[600, 960, 3]`, depth shape `[600, 960]`, and
RTC `published_frames` increasing. There is intentionally no
`/head_camera/image_rgb` topic. The recorder opens its own direct RGB source
and resizes it to the configured 640 x 480 dataset resolution.

## 7. Start real teleoperation, initially disabled

In terminal 2, launch the real command publisher:

```bash
ros2 launch dex_pico_teleop pico_teleop.launch.py \
  publish_commands:=true \
  network_transport:=tcp \
  network_host:=0.0.0.0 \
  network_port:=63901 \
  control_rate_hz:=50.0
```

The node starts disabled. Relaunching resets calibration, so repeat both
calibrations exactly as in the dry-run. Keep the operator and controllers in a
safe neutral posture before enabling real motion. Enable only when the safety
observer is ready:

```bash
ros2 service call /dex_pico_teleop/calibrate std_srvs/srv/Trigger '{}'
ros2 service call /dex_pico_teleop/calibrate_reach std_srvs/srv/Trigger '{}'
ros2 topic echo /dex_pico_teleop/status --once --full-length
ros2 service call /dex_pico_teleop/enabled std_srvs/srv/SetBool "{data: true}"
```

Make a few small, slow motions. Verify that centered joysticks command zero base
motion and that disabling teleop holds/stops command generation as documented.
Then verify all four recorder action/state streams:

```bash
ros2 topic hz /joint_states
ros2 topic hz /dexcontrol/applied_joint_commands
ros2 topic hz /dexcontrol/measured_base_twist
ros2 topic hz /dexcontrol/applied_base_twist
```

Expected bridge defaults are approximately 100, 250, 50, and 250 Hz,
respectively, while valid commands are being applied.

## 8. Launch the recorder in forced local-only mode

In terminal 4, use the service-only input backend for the first test. This
prevents an accidental pedal/key press while preserving all four ROS controls:

```bash
ros2 launch dex_vega_lerobot_recorder record_teleop_dataset.launch.py \
  config_file:="$LIVE_CONFIG" \
  no_hf_upload:=true \
  input_backend:=disabled
```

The startup log must show:

- the exact expected local dataset path;
- `recorder ready ... 20 FPS`;
- `Hugging Face upload disabled; saving locally only`;
- both wrist placeholders enabled at the processed head resolution;
- state `IDLE`, zero pending frames, and no `ERROR`.

Do not press Ctrl+C immediately after launching if the dataset writer is still
initializing. Wait for the `recorder ready` line.

## 9. Discard-first rehearsal

This confirms lifecycle controls without committing an episode.

```bash
ros2 service call /dex_vega_lerobot_recorder/start std_srvs/srv/Trigger '{}'
```

Perform a small safe motion for 3–5 seconds. Watch terminal 4. At 20 FPS,
`achieved_fps` should approach 20, `writer_queue` should normally return to 0,
and the preferred acceptance result is `dropped=0 stale=0`.

Stop capture; this does **not** commit:

```bash
ros2 service call /dex_vega_lerobot_recorder/stop std_srvs/srv/Trigger '{}'
```

Confirm the response reports `REVIEW_PENDING`, frame count, duration, drop/stale
counts, and `validation=valid`. Then discard it:

```bash
ros2 service call /dex_vega_lerobot_recorder/discard std_srvs/srv/Trigger '{}'
```

Confirm the recorder returns to `IDLE` and logs that the committed episode count
is unchanged.

## 10. Record and explicitly save one candidate

Start a new candidate:

```bash
ros2 service call /dex_vega_lerobot_recorder/start std_srvs/srv/Trigger '{}'
```

Perform the configured task for 5–10 seconds, then stop capture:

```bash
ros2 service call /dex_vega_lerobot_recorder/stop std_srvs/srv/Trigger '{}'
```

Review the operator performance and terminal summary. Do not save if the task
failed, timestamps were stale, inputs were missing, the writer queue filled, or
the achieved FPS was materially below 20. Discard and repeat instead:

```bash
ros2 service call /dex_vega_lerobot_recorder/discard std_srvs/srv/Trigger '{}'
```

Only for a valid candidate, commit exactly one episode:

```bash
ros2 service call /dex_vega_lerobot_recorder/save std_srvs/srv/Trigger '{}'
```

The response must identify committed episode `0` and the exact local path. A
second save without another start/stop must be rejected as an invalid state
transition.

## 11. Safe shutdown and finalization

First disable real teleoperation:

```bash
ros2 service call /dex_pico_teleop/enabled std_srvs/srv/SetBool "{data: false}"
```

With the recorder in `IDLE`, press Ctrl+C once in terminal 4. Wait for video
encoding/finalization and the line reporting:

```text
recorder shutdown: no pending episode; finalized committed data at ...
```

Do not terminate it a second time while finalization is in progress. If Ctrl+C
is used during `RECORDING` or `REVIEW_PENDING`, the unsaved candidate is
discarded by default; it is never silently saved.

After recorder finalization, stop the teleop, camera, and bridge processes in
that order unless the robot is needed for another test.

## 12. Reload and inspect the finalized dataset

Replace the two values in this command with the YAML values used above:

```bash
python3 - /data/lerobot/vega_live_smoke_YYYYMMDD_HHMM \
  vega_live_smoke_YYYYMMDD_HHMM <<'PY'
from pathlib import Path
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(sys.argv[1]).resolve()
name = sys.argv[2]
dataset = LeRobotDataset(repo_id=f"local/{name}", root=root)

camera_keys = {
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
}
assert dataset.num_episodes == 1
assert dataset.num_frames > 0
assert camera_keys.issubset(dataset.features)
assert dataset.features["observation.state"]["shape"] == (27,)
assert dataset.features["action"]["shape"] == (27,)
assert dataset.meta.robot_type == "dexmate_vega_1_pro"

frame = dataset[0]
assert camera_keys.issubset(frame)
assert tuple(frame["observation.state"].shape) == (27,)
assert tuple(frame["action"].shape) == (27,)

print("dataset:", root)
print("episodes:", dataset.num_episodes)
print("frames:", dataset.num_frames)
print("robot_type:", dataset.meta.robot_type)
print("tasks:", list(dataset.meta.tasks.index))
print("features:", dataset.features)
PY
```

Also confirm these local source-of-truth files exist:

```bash
test -f /data/lerobot/vega_live_smoke_YYYYMMDD_HHMM/meta/info.json
test -f /data/lerobot/vega_live_smoke_YYYYMMDD_HHMM/meta/vega_recording_config.yaml
test -f /data/lerobot/vega_live_smoke_YYYYMMDD_HHMM/meta/vega_feature_specification.json
find /data/lerobot/vega_live_smoke_YYYYMMDD_HHMM/videos -type f -name '*.mp4' -print
```

Do not upload this smoke-test episode to a production dataset. If it should be
kept as a private test artifact, upload it only after reviewing the local data
and choosing an explicit test repository ID.

## 13. Troubleshooting gates

| Symptom | Action |
| --- | --- |
| Missing required `/joint_states` positions | Fix bridge/vendor feedback; do not relax recorder staleness thresholds. |
| No applied command topic | Confirm Pico input is fresh, teleop is calibrated/enabled, and vendor dispatch succeeds. |
| Missing/stale head image | Check camera sensor service and head-camera vision status before recording. |
| Repeated stale/drop counts | Discard the candidate and diagnose the specific recorder warning. |
| Writer queue grows or fills | Stop/discard; check persistent-disk space and I/O performance. |
| Dataset path already exists | Choose a unique name. Use `resume` only for an intentionally finalized matching dataset. |
| Recorder enters `ERROR` | Disable teleop, preserve logs, and shut down cleanly; do not try to save the candidate. |
| Terminal keyboard reports no controlling TTY | Run launch from a foreground terminal; otherwise keep `input_backend:=disabled` and use ROS services, or configure a permitted Linux input-event device. |

After this service-controlled procedure passes, test the physical pedals with
the robot stationary and teleop disabled. Confirm one key-down event maps to one
state transition before using pedals during motion.
