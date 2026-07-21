# First guarded hardware trial — GR00T N1.7 step 34k

This runbook prepares one five-second physical interface trial of the pinned
GR00T blue-bird candidate. It does not authorize motion. Do not start guarded
mode until the user gives fresh authorization for this GR00T trial and a named
operator confirms that the working wireless physical E-stop is in hand.

The five-second trial is a direction, bounded-motion, and stop-behavior check;
it is not an attempt to complete the task and does not establish physical
safety or policy quality.

| Item | Pinned value |
|---|---|
| Fine-tune | `step-034000`, `7f0f318540355031f189693e5623c1c5e8a17e93` |
| N1.7 base | `2fc962b973bccdd5d8ce4f67cc63b264d6886495` |
| Cosmos processor | `9ce19a195e423419c349abfc86fd07178b230561` |
| Task | `put the blue bird on the meeting desk` |
| Cadence / horizon | 3 Hz / 21 of 40 steps |
| Maximum armed interval | 5.0 seconds, then `FAULT` |

The guarded limits remain 0.02 rad of body-target change and 0.03 hand ratio
per 30 Hz cycle, 0.10 m/s base translation, 0.20 rad/s base yaw, 0.75-second
queue age, 1.35-second observation-to-action age, and 0.20-second measured
state/image skew. The recorder's independent state, camera capture, receive,
and transport-age gates remain active.

## 1. Physical and process preparation

1. Clear people and loose objects from the torso, head, both arms, hands, and
   mobile-base sweep.
2. Use the recorded blue-bird/meeting-desk arrangement, but plan to stop before
   task completion.
3. Put the base on a level, unobstructed surface.
4. Assign one E-stop operator who watches the robot rather than a terminal.
5. Stop every PI0.5/GR00T policy server, inference node, Pico teleop node,
   navigation node, MoveIt execution process, and recorder that is not required
   by this trial. Never run another GPU policy beside GR00T.
6. Do not release an E-stop with `/soft_estop`; the runbook never publishes
   `data: false` to that service.

The bridge holds its last cached joint targets after inference stops. Do not
invent a return pose or controller-disable command.

## 2. Build and test the exact workspace

```bash
cd /workspaces/dexcontrol-ros/ros_ws
source /opt/ros/humble/setup.bash
export ROS_LOG_DIR="$PWD/.runtime/ros-log"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

colcon build --symlink-install --packages-select \
  dexcontrol_ros dex_camera_transport dex_vega_lerobot_recorder \
  dex_vega_lerobot_inference
source install/setup.bash

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
colcon test --packages-select dex_vega_lerobot_inference \
  --event-handlers console_direct+
colcon test-result --verbose \
  --test-result-base build/dex_vega_lerobot_inference
```

Stop if the build or any test fails.

## 3. Prove bridge, E-stop, and command ownership

There must be exactly one current bridge. Do not start a second bridge beside
it. If the running bridge does not publish `/dexcontrol/estop_state`, follow
the bridge replacement procedure in the PI runbook while the physical E-stop
is engaged.

Confirm the physical stop first reports `true` while pressed and then a fresh
`false` only after the operator releases it in a clear scene:

```bash
ros2 node list | sort
ros2 topic info --verbose /dexcontrol/estop_state
ros2 topic echo --once /dexcontrol/estop_state
ros2 topic hz /dexcontrol/estop_state
```

Stop `ros2 topic hz` after several samples. Audit all command topics:

```bash
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

Require publisher count zero and exactly the intended bridge subscription on
all seven topics. Stop if Pico, navigation, MoveIt, another inference node, or
any unknown publisher is present.

## 4. Start the exact local GR00T policy server

This Python 3.12/CUDA process has no ROS or robot command interface:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/.venvs/lerobot/bin/policy_server \
  --policy-type groot \
  --project-root /workspace \
  --model-dir /workspace/data/models/groot-n17-dexmate-blue-bird/7f0f318540355031f189693e5623c1c5e8a17e93 \
  --model-commit 7f0f318540355031f189693e5623c1c5e8a17e93 \
  --checkpoint-tag step-034000 \
  --base-model-dir /workspace/data/models/groot-n1.7-3b/2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --base-model-commit 2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --cosmos-processor-dir /workspace/data/models/cosmos-reason2-2b/9ce19a195e423419c349abfc86fd07178b230561 \
  --cosmos-processor-commit 9ce19a195e423419c349abfc86fd07178b230561 \
  --socket-path /workspace/.runtime/groot_n17_policy_server.sock
```

Wait for the server-ready message. A pre-existing socket or identity mismatch
must fail closed; do not delete an unexplained socket while another server may
own it.

## 5. Mandatory final live dry-run

Run at least 30 seconds immediately before guarded mode:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_LOG_DIR="$PWD/.runtime/ros-log"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch dex_vega_lerobot_inference groot_dry_run.launch.py
```

In another terminal:

```bash
ros2 topic echo --once /dex_vega_lerobot_inference/status
```

Require the exact three commits, `step-034000`, task text, 3 Hz, 21 steps,
`OBSERVE_ONLY`, `execution_capable=false`, and increasing predictions. Require
worker errors/drops, shadow queue starvation, stale queue/observation actions,
shadow action errors, `actions_published`, and all seven command-topic publisher
counts to remain zero. Review any hand/joint clamps. Rate and base clamps are
expected safety interventions, but an unusual increase still requires review.

Stop the dry-run with `Ctrl-C` and wait for its node to disappear. Leave the
policy server running.

## 6. Start guarded mode, still disarmed

Do not continue without fresh GR00T trial authorization and the E-stop
operator's explicit readiness confirmation. With Pico fully stopped and the
zero-publisher audit complete, start:

```bash
ros2 launch dex_vega_lerobot_inference \
  groot_guarded_execution.launch.py \
  allow_command_publication:=true \
  execution_readiness_acknowledged:=true \
  require_teleop_disabled:=false \
  maximum_execution_duration_seconds:=5.0
```

This creates command publishers but remains disarmed. Before any service call,
require status `READY`, `execution_capable=true`, `trial_active=false`, exact
artifact identity, at least one warm-up prediction, fresh false E-stop
telemetry, and a 5.0-second maximum. Re-audit all command topics: each must now
have exactly one inference publisher and one bridge subscriber.

## 7. Begin and arm one five-second interface trial

Keep status visible and a stop terminal ready:

```bash
ros2 topic echo /dex_vega_lerobot_inference/status
```

After the E-stop operator gives the final verbal confirmation:

```bash
ros2 service call /dex_vega_lerobot_inference/begin_trial \
  std_srvs/srv/Trigger '{}'
ros2 service call /dex_vega_lerobot_inference/arm \
  std_srvs/srv/SetBool '{data: true}'
```

Press the physical E-stop immediately for unexpected direction, speed,
posture, contact, instability, lost visibility, or uncertainty. For an ordinary
early stop:

```bash
ros2 service call /dex_vega_lerobot_inference/arm \
  std_srvs/srv/SetBool '{data: false}'
ros2 service call /dex_vega_lerobot_inference/end_trial \
  std_srvs/srv/Trigger '{}'
```

At five seconds the node must enter `FAULT`, stop new joint/hand targets, clear
its action queue, and send zero `/cmd_vel`. The bridge holds its cached joint
targets. After inspecting the robot and status, end and recover; recovery never
re-arms:

```bash
ros2 service call /dex_vega_lerobot_inference/end_trial \
  std_srvs/srv/Trigger '{}'
ros2 service call /dex_vega_lerobot_inference/recover \
  std_srvs/srv/Trigger '{}'
```

## 8. Record and stop

Record the three commits, task, UTC start/stop, duration fault, predictions,
published actions, worker and shadow counters, all clamp/rate-limit counters,
latency/ages, E-stop events, operator interventions, qualitative motion, and
whether stop behavior matched this runbook. Then disarm/end if necessary, stop
the guarded launch, and stop the policy server.

Do not repeat, extend to 15 seconds, or proceed to 30 seconds without a new
review and fresh user authorization. A successful interface trial would still
not establish task success or physical safety.
