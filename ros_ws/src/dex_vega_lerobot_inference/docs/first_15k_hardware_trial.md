# First guarded hardware trial — PI0.5 step 15k

This runbook is for one deliberately short hardware trial of the blue-bird
policy. It does not authorize the trial by itself. Do not begin the arming
section until the operator has explicitly confirmed that the wireless physical
E-stop is working, in hand, and ready to press.

The validated first-trial default remains step 15k. Exact commands for using
any pinned candidate are collected in
[Section 11](#11-complete-direct-policy-server-commands-for-5k-15k-and-30k); all other
runtime and stop procedures in this runbook remain unchanged.

The default configuration pins all of the following:

| Item | Pinned value |
|---|---|
| Policy tag | `step-015000` |
| Policy commit | `be768eb6a4e32a58f66cadea7cd2159d99a16e86` |
| Local policy | `data/models/pi05-dexmate-blue-bird/step-015000` |
| Tokenizer commit | `35e4f46485b4d07967e7e9935bc3786aad50687c` |
| Task | `put the blue bird on the meeting desk` |
| Maximum armed interval | 5.0 seconds, followed by `FAULT` |

The five-second trial is a safety/interface test, not an attempt to complete
the task. The initial limits are 0.02 rad of body target change per 30 Hz
cycle, 0.03 of hand-ratio change per cycle, 0.10 m/s base linear speed, and
0.20 rad/s base yaw rate. State/image skew is limited to 0.20 seconds. On
2026-07-21 the 30k live path measured 0.144-0.150 seconds during a passive
eight-second sample and twice crossed the former 0.15-second gate at 0.151
seconds. The recorder's independent state, receive, capture, and transport age
limits remain unchanged. Every finite postprocessed hand ratio is clamped into
`[0,1]`, logged as a throttled warning, and counted; NaN or Inf still faults.
Every finite body or expanded-hand joint target is likewise clipped to the
authoritative URDF position limits, warned, and counted instead of stopping the
trial. Missing/invalid limits and NaN/Inf still fault. The bridge still applies
its own independent clipping.

## 1. Physical preparation

1. Clear people and loose objects from the robot, arms, and mobile-base sweep.
2. Put the bird and meeting desk in the recorded task arrangement, but treat
   the first trial as a five-second motion check.
3. Keep the mobile base on a level, unobstructed surface.
4. Assign one person to the wireless E-stop. That operator must watch the robot,
   not a terminal, throughout the armed interval.
5. Confirm the E-stop can be engaged before starting any execution-capable
   process. Never use `/soft_estop` with `data: false` as part of this runbook.

The bridge retains its last joint targets while stopped. Do not release an
E-stop until inference has been disarmed/ended or stopped and the retained
targets are known to be safe.

## 2. Build the validated workspace

Run this from the workspace root. Building does not affect the already-running
bridge process.

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
mkdir -p .runtime/ros-log
export ROS_LOG_DIR="$PWD/.runtime/ros-log"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

colcon build --symlink-install --packages-select \
  dexcontrol_ros dex_camera_transport dex_vega_lerobot_recorder \
  dex_vega_lerobot_inference
source install/setup.bash
```

## 3. Replace the old bridge and prove E-stop telemetry

The bridge observed during dry-run predates the new
`/dexcontrol/estop_state` publisher. It cannot satisfy the inference arming
gate and must be replaced deliberately. Do not start a second bridge beside
the old one.

1. Engage the wireless physical E-stop.
2. In the terminal that owns the current bridge, use `Ctrl-C` and wait for it
   to exit.
3. In a prepared ROS shell, start the just-built bridge:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_LOG_DIR="$PWD/.runtime/ros-log"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROBOT_IP=192.168.50.20:7447

ros2 launch dexcontrol_ros dexcontrol_bridge.launch.py robot_ip:="$ROBOT_IP"
```

The bridge initializes its cached joint targets from measured joint positions.
In another ROS shell, verify that there is one bridge and that the pressed
physical E-stop is reported as `true`:

```bash
ros2 node list | sort
ros2 topic info --verbose /dexcontrol/estop_state
ros2 topic echo --once /dexcontrol/estop_state
```

With inference absent and the scene safe, release/reset the physical E-stop and
verify a fresh `data: false` sample. Keep the wireless stop in the operator's
hand from this point onward.

```bash
ros2 topic echo --once /dexcontrol/estop_state
ros2 topic hz /dexcontrol/estop_state
```

The expected rate is approximately 10 Hz. If the topic is missing, stale,
reports the wrong physical state, or the bridge logs an E-stop read error, stop
here. Stop `ros2 topic hz` with `Ctrl-C` after observing several samples. The
inference node intentionally does not infer safety from the presence of the
`/soft_estop` service.

## 4. Audit command-topic ownership before inference

Pico is disconnected for this trial. Confirm that no Pico node and no command
publisher is present before starting inference:

```bash
ros2 node list | sort
for topic in \
  /torso/joint_commands \
  /head/joint_commands \
  /left_arm/joint_commands \
  /right_arm/joint_commands \
  /left_hand/joint_commands \
  /right_hand/joint_commands \
  /cmd_vel
do
  ros2 topic info --verbose "$topic"
done
```

There must be no `/dex_pico_teleop` node, every command topic must have
`Publisher count: 0`, and each must show the bridge as its subscriber. Stop if
any other publisher exists.

## 5. Start the pinned 15k policy server

This process has no ROS connection and cannot actuate the robot. Model loading
takes about 172 seconds on the validated Thor runtime.

```bash
cd /workspaces/dexcontrol-ros/ros_ws
src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/.venvs/lerobot/bin/policy_server \
  --project-root /workspace \
  --model-dir \
    /workspace/data/models/pi05-dexmate-blue-bird/step-015000 \
  --model-commit be768eb6a4e32a58f66cadea7cd2159d99a16e86 \
  --checkpoint-tag step-015000 \
  --tokenizer-dir \
    /workspace/data/models/paligemma-3b-pt-224/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --tokenizer-commit 35e4f46485b4d07967e7e9935bc3786aad50687c \
  --socket-path /workspace/.runtime/pi05_policy_server.sock
```

Wait for the server to report that it is listening. Leave this terminal open.

## 6. Mandatory live-camera dry-run

In a ROS shell, run the exact 15k policy against live state and head RGB for at
least 30 seconds:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_LOG_DIR="$PWD/.runtime/ros-log"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ros2 launch dex_vega_lerobot_inference dry_run.launch.py
```

From another ROS shell, inspect status and command ownership:

```bash
ros2 topic echo --once /dex_vega_lerobot_inference/status
ros2 topic info --verbose /cmd_vel
ros2 topic info --verbose /left_arm/joint_commands
```

Require all of the following before continuing:

- `state` is `OBSERVE_ONLY` and `execution_capable` is `false`;
- `model_commit` is the pinned 15k commit and `checkpoint_tag` is
  `step-015000`;
- the exact task text is present;
- successful predictions increase, while model errors and
  `actions_published` remain zero;
- status ages remain within the configured limits;
- `state_camera_skew_seconds` remains below the reported 0.20-second maximum;
- every command topic still has publisher count zero.

Stop the dry-run with `Ctrl-C` and wait for the node to disappear. Leave the
policy server running.

## 7. Start guarded mode, still disarmed

The following launch creates the seven command publishers but does not arm and
does not publish joint/hand actions. `require_teleop_disabled` is overridden
only because the Pico headset was explicitly confirmed disconnected. The
exclusive-publisher gate remains enabled.

Do this only after the E-stop operator has explicitly said they are ready:

```bash
ros2 launch dex_vega_lerobot_inference guarded_execution.launch.py \
  allow_command_publication:=true \
  require_teleop_disabled:=false
```

Before arming, require status to show `READY`, `execution_capable: true`,
`trial_active: false`, at least one successful warm-up prediction, the pinned
15k identity, and a fresh false E-stop. Confirm the hard time limit, which
cannot be changed while this node is running:

```bash
ros2 param get /dex_vega_lerobot_inference maximum_execution_duration_seconds
ros2 topic echo --once /dex_vega_lerobot_inference/status
```

Audit all seven command topics again. Each must now have exactly one publisher,
the inference node, and a bridge subscription. The node continuously enforces
this condition and faults if a competing publisher appears.

## 8. Begin and arm one five-second trial

Keep one terminal streaming status and another ready for the stop calls:

```bash
ros2 topic echo /dex_vega_lerobot_inference/status
```

The E-stop operator gives the final verbal confirmation. Then begin the trial,
which only resets queues, and arm it with a separate call:

```bash
ros2 service call /dex_vega_lerobot_inference/begin_trial \
  std_srvs/srv/Trigger '{}'
ros2 service call /dex_vega_lerobot_inference/arm \
  std_srvs/srv/SetBool '{data: true}'
```

The node waits for a fresh postprocessed action chunk and then begins guarded
publication. Press the physical E-stop immediately for unexpected direction,
speed, posture, contact, instability, loss of operator visibility, or any
uncertainty. For an ordinary early stop, call:

```bash
ros2 service call /dex_vega_lerobot_inference/arm \
  std_srvs/srv/SetBool '{data: false}'
ros2 service call /dex_vega_lerobot_inference/end_trial \
  std_srvs/srv/Trigger '{}'
```

At five seconds the node must enter `FAULT`, stop all new joint/hand
publication, clear action queues, and send zero base velocity. The bridge holds
its last cached joint targets. Verify the `FAULT` reason in status and end the
trial. Only after the robot and scene are safe and E-stop telemetry is fresh
and false, recover to `READY`:

```bash
ros2 topic echo --once /dex_vega_lerobot_inference/status
ros2 service call /dex_vega_lerobot_inference/end_trial \
  std_srvs/srv/Trigger '{}'
ros2 service call /dex_vega_lerobot_inference/recover \
  std_srvs/srv/Trigger '{}'
```

Recovery never re-arms. Another trial requires a new `begin_trial` and `arm`
pair. Do not run a second trial until the first trial's motion, diagnostics,
latency, and stop behavior have been reviewed.

## 9. Shutdown and record

Use the disarm and end-trial calls whenever the node is not already in a fault,
then stop the guarded launch with `Ctrl-C`. Stop the policy server with
`Ctrl-C`; it removes its local socket. The bridge may remain running.

Record at minimum:

- the exact model/tag/tokenizer commits and task from status;
- start/stop UTC times and whether the five-second fault occurred;
- preprocessing, GPU, postprocessing, and end-to-end latency;
- observation/action age and queue gaps;
- `actions_published`, rate-limited actions, and base-clamped actions;
- bounded hand-ratio clamps (`hand_clamped_actions`);
- bounded URDF joint-target clamps (`joint_clamped_actions` and
  `last_joint_clamp`);
- E-stop events, faults, warnings, and operator interventions;
- qualitative arm, hand, head, torso, and base motion;
- whether stop behavior matched this runbook.

Do not proceed to a longer trial or another checkpoint merely because this
five-second interface trial completed without a fault other than its time cap.

## 10. Staged duration extension

The first successful guarded trial ended at 5.025 seconds with 130 actions
published, no inference errors, fresh timing, and the expected maximum-duration
fault. After reviewing the physical motion and confirming stop behavior, the
next stage is 15 seconds. End the old trial, stop the guarded node with
`Ctrl-C`, and restart it with an explicit finite duration:

```bash
ros2 launch dex_vega_lerobot_inference guarded_execution.launch.py \
  allow_command_publication:=true \
  require_teleop_disabled:=false \
  maximum_execution_duration_seconds:=15.0
```

Verify the parameter and status before beginning and arming a new trial:

```bash
ros2 param get /dex_vega_lerobot_inference \
  maximum_execution_duration_seconds
ros2 topic echo --once /dex_vega_lerobot_inference/status
```

It must report 15.0 seconds and the pinned 15k identity. Use a new
`begin_trial` and `arm` pair with the E-stop operator ready. Review that trial
before progressing to a separately launched 30-second trial. Do not use an
unbounded duration.

## 11. Complete direct policy-server commands for 5k, 15k, and 30k

The PI0.5 policy server is deliberately not a ROS launch. Start it directly
through the project-local Jetson runtime, leave its terminal open, and start
the checkpoint-independent guarded ROS launch in a different terminal. The
three policies share one pinned tokenizer:

| Candidate | Policy tag | Immutable model commit |
|---|---|---|
| `5k` | `step-005000` | `6a511ca59438d1c7d4510dc08cecacce5b9b7014` |
| `15k` | `step-015000` | `be768eb6a4e32a58f66cadea7cd2159d99a16e86` |
| `30k` | `step-030000` | `305c4bf9067ead22c95befb810cdafbc6135cabb` |

The shared tokenizer commit is
`35e4f46485b4d07967e7e9935bc3786aad50687c`. Each command below is complete;
do not combine a model directory from one block with a commit or tag from
another.

### Step 5k policy server

```bash
cd /workspaces/dexcontrol-ros/ros_ws
src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/.venvs/lerobot/bin/policy_server \
  --project-root /workspace \
  --model-dir \
    /workspace/data/models/pi05-dexmate-blue-bird/step-005000 \
  --model-commit 6a511ca59438d1c7d4510dc08cecacce5b9b7014 \
  --checkpoint-tag step-005000 \
  --tokenizer-dir \
    /workspace/data/models/paligemma-3b-pt-224/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --tokenizer-commit 35e4f46485b4d07967e7e9935bc3786aad50687c \
  --socket-path /workspace/.runtime/pi05_policy_server.sock
```

### Step 15k policy server

```bash
cd /workspaces/dexcontrol-ros/ros_ws
src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/.venvs/lerobot/bin/policy_server \
  --project-root /workspace \
  --model-dir \
    /workspace/data/models/pi05-dexmate-blue-bird/step-015000 \
  --model-commit be768eb6a4e32a58f66cadea7cd2159d99a16e86 \
  --checkpoint-tag step-015000 \
  --tokenizer-dir \
    /workspace/data/models/paligemma-3b-pt-224/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --tokenizer-commit 35e4f46485b4d07967e7e9935bc3786aad50687c \
  --socket-path /workspace/.runtime/pi05_policy_server.sock
```

### Step 30k policy server

```bash
cd /workspaces/dexcontrol-ros/ros_ws
src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/.venvs/lerobot/bin/policy_server \
  --project-root /workspace \
  --model-dir \
    /workspace/data/models/pi05-dexmate-blue-bird/step-030000 \
  --model-commit 305c4bf9067ead22c95befb810cdafbc6135cabb \
  --checkpoint-tag step-030000 \
  --tokenizer-dir \
    /workspace/data/models/paligemma-3b-pt-224/35e4f46485b4d07967e7e9935bc3786aad50687c \
  --tokenizer-commit 35e4f46485b4d07967e7e9935bc3786aad50687c \
  --socket-path /workspace/.runtime/pi05_policy_server.sock
```

Stop the existing guarded node and policy server before changing checkpoints.
The direct server command alone selects the checkpoint; the
checkpoint-independent ROS node discovers and verifies that server identity
when it starts.

In a prepared ROS shell, start the same guarded execution command for any of
the three policy servers. This command retains the launch file's finite
five-second default; use only the reviewed finite-duration progression in
Section 10 when extending it.

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_LOG_DIR="$PWD/.runtime/ros-log"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ros2 launch dex_vega_lerobot_inference guarded_execution.launch.py \
  allow_command_publication:=true \
  require_teleop_disabled:=false
```

The status must identify the tag and commit from the policy-server block that
you selected. The guarded launch remains disarmed until the separate
`begin_trial` and `arm` calls in Section 8.
