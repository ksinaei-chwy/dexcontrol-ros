# Dex Vega LeRobot inference

This package provides fail-closed ROS 2 inference paths for the fine-tuned
LeRobot PI0.5 and GR00T N1.7 blue-bird policies. Their default `observe_only`
launches create no publishers for any robot command topic. Initial PI0.5
implementation and validation through Stage 5 used no robot motion; one later
operator-supervised five-second PI0.5 physical interface trial completed and
stopped on its configured time cap. That result does not validate GR00T.

The documented first-trial policy-server selection is the private
`step-015000` candidate at
immutable Hub commit `be768eb6a4e32a58f66cadea7cd2159d99a16e86`.
The tokenizer is pinned to
`google/paligemma-3b-pt-224` commit
`35e4f46485b4d07967e7e9935bc3786aad50687c`. Both are already stored under
`data/models/` with local manifests; production never follows mutable `main`.

The GR00T candidate is the private `step-034000` artifact at immutable commit
`7f0f318540355031f189693e5623c1c5e8a17e93`, with raw N1.7 base commit
`2fc962b973bccdd5d8ce4f67cc63b264d6886495` and Cosmos processor commit
`9ce19a195e423419c349abfc86fd07178b230561`. Its implementation and validation
record are separate from PI0.5:

- [GR00T deployment guide](docs/groot_n17_deployment.md)
- [GR00T validation report](docs/groot_n17_validation_report.md)
- [GR00T first guarded-trial runbook](docs/first_groot_n17_hardware_trial.md)

All three exact GR00T artifacts are cached under `data/models/` with immutable
manifests. The validation report records the weight SHA/size, offline network
isolation, representative recorded-input benchmark, live timing, and longer
shadow results. Those results prepare a guarded interface trial; they do not
establish physical safety, task success, or authorization to move the robot.

## Architecture

ROS 2 Humble on this Thor uses Python 3.10, while LeRobot 0.6.0 declares
Python >=3.12. The integration uses two local processes instead of forcing an
unsupported Python installation:

```text
DexComm RGB + ROS state                      Python 3.12 / CUDA
          |                                       |
          v                                       v
ROS inference node -- latest observation --> policy server (PI0.5 or GR00T)
 Python 3.10       <-- physical Tx27 chunk --- saved pre/policy/post
          |
          v (only ARMED + every gate fresh)
sensor_msgs/JointState component commands + geometry_msgs/Twist
          |
          v
dexcontrol_ros clipping, cached joint targets, cmd_vel watchdog, E-stop
```

The Unix socket contains only local state, RGB bytes, timing metadata, and the
physical action chunk. It is created mode `0600` inside `.runtime/`. The policy
server has no ROS or robot connection. The ROS process owns arming, freshness,
URDF limits, slew/acceleration limits, bridge liveness, E-stop/teleop exclusion,
and all publication decisions.

The policy server is the single source of artifact selection. On startup, the
ROS node queries the server for its model path, checkpoint tag, and dependent
artifact identities. It maps those paths through the project-local
`/workspace` bind, independently validates the local manifests, and reports
the resolved identity in status. Every prediction and reset response repeats
that identity; changing the server artifact underneath a running ROS node
faults instead of accepting actions from the replacement.
Consequently the PI0.5 observe-only, dry-run, replay, and guarded launch files
have no model/checkpoint/tokenizer arguments and are identical for 5k, 15k,
and 30k. External PI0.5 mode rejects custom ROS parameter files that still set
those duplicated selections instead of silently ignoring them.

The PI0.5 policy server is deliberately a direct, non-ROS process started
through `scripts/run_jetson_runtime.sh`. Its command selects the complete local
model artifact and tokenizer by immutable identity. The exact commands for all
three blue-bird candidates are in the
[guarded-trial runbook](docs/first_15k_hardware_trial.md#11-complete-direct-policy-server-commands-for-5k-15k-and-30k).
Future model/task integrations can provide their own runtime configuration
without adding checkpoint selection to the guarded ROS launch or safety node.

The identity-bound socket protocol is version 2. After updating this package,
restart both the policy server and the ROS inference node; a new ROS node
rejects a still-running older server immediately.

This is approach B from the implementation brief. LeRobot 0.6.0's generic
`lerobot-rollout` owns the robot connection/lifecycle and reconciles scalar
`.pos` hardware features. That lifecycle cannot faithfully represent the
mixed absolute-joint, hand-synergy, and base-velocity contract here. The policy
server nevertheless uses LeRobot's `PreTrainedConfig`, PI05 policy class,
`make_pre_post_processors`, and the serialized preprocessor/postprocessor. It
never recreates quantile statistics or manually pads state/actions.

LeRobot 0.6.0 supports RTC for PI0/PI0.5 through
`policy.predict_action_chunk`, but this checkpoint saved `rtc_config: null`.
The validated path uses an asynchronous latest-observation worker and an
18-of-50-step receding horizon. Enabling RTC would change the saved config and
move inference-delay/overlap queue ownership into LeRobot's rollout lifecycle,
so it remains an explicit observe-only experiment rather than an unvalidated
runtime switch. Live RTC can measure latency and queue stability without a
bag; deterministic replay is needed only for an apples-to-apples output
comparison. See the official
[deployment documentation](https://huggingface.co/docs/lerobot/v0.6.0/inference)
and [PI0.5 documentation](https://huggingface.co/docs/lerobot/en/pi05).

## Authoritative ROS interfaces

The live head camera is not a ROS image topic. It is the recorder's
`DirectRgbCameraSource`, backed by `dex_camera_transport.DexCommCameraSource`
with latest-frame buffer size 1. It consumes the `left_rgb` DexComm stream via
Zenoh topic `sensors/head_camera/left_rgb`, returns contiguous `uint8` RGB, and
uses the recorder's receive/capture/transport freshness checks. There is no ROS
message type or ROS QoS for this live camera path.

| Interface | Type | Direction/use |
|---|---|---|
| `/joint_states` | `sensor_msgs/msg/JointState` | bridge -> inference measured positions |
| `/dexcontrol/measured_base_twist` | `geometry_msgs/msg/TwistStamped` | bridge -> inference measured velocity |
| `/dexcontrol/applied_joint_commands` | `sensor_msgs/msg/JointState` | bridge -> liveness/audit only |
| `/dexcontrol/applied_base_twist` | `geometry_msgs/msg/TwistStamped` | bridge -> liveness/audit only |
| `/dexcontrol/estop_state` | `std_msgs/msg/Bool` | bridge -> inference E-stop gate |
| `/dex_pico_teleop/status` | `std_msgs/msg/String` JSON | teleop -> inference mutual-exclusion gate |
| `/torso/joint_commands` | `sensor_msgs/msg/JointState` | inference -> bridge absolute targets |
| `/head/joint_commands` | `sensor_msgs/msg/JointState` | inference -> bridge absolute targets |
| `/left_arm/joint_commands` | `sensor_msgs/msg/JointState` | inference -> bridge absolute targets |
| `/right_arm/joint_commands` | `sensor_msgs/msg/JointState` | inference -> bridge absolute targets |
| `/left_hand/joint_commands` | `sensor_msgs/msg/JointState` | inference -> bridge expanded six-joint targets |
| `/right_hand/joint_commands` | `sensor_msgs/msg/JointState` | inference -> bridge expanded six-joint targets |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | inference -> bridge base velocity |
| `/soft_estop` | `std_srvs/srv/SetBool` | operator/bridge E-stop; inference never releases it |
| `/dex_vega_lerobot_inference/replay/head_image` | `sensor_msgs/msg/Image` | replay only, sensor-data QoS, `rgb8` or `bgr8` |

The `applied_*` topics are not command inputs. The bridge clips joint commands
to the vendor component limits, clips base velocity to chassis limits, and
zeros base velocity after its `cmd_vel_timeout_s` (default 0.5 seconds). It has
no joint command timeout: stopping joint publications makes it retain and
dispatch the last cached targets. Consequently inference faults cease all new
joint/hand publications, publish zero base through `/cmd_vel`, clear every
queue, and require recovery plus explicit re-arm. Returning to a pose or
disabling a joint controller is intentionally not guessed.

The bridge retains its cached joint targets while E-stop is active and can
dispatch them again after E-stop release. Before resetting a physical E-stop,
disarm/end the inference trial and verify the retained targets are safe; the
physical stop is not a substitute for this software transition.

## Exact 27-value contract

State and action have the same index names, with `.position` appended to state
joint names:

| Indices | Values | State semantics | Action semantics |
|---|---|---|---|
| 0-2 | `torso_j1..j3` | measured rad | absolute target rad |
| 3-5 | `head_j1..j3` | measured rad | absolute target rad |
| 6-12 | `L_arm_j1..j7` | measured rad | absolute target rad |
| 13-19 | `R_arm_j1..j7` | measured rad | absolute target rad |
| 20-21 | left open/close, opposition | measured ratios | absolute ratios |
| 22-23 | right open/close, opposition | measured ratios | absolute ratios |
| 24-25 | `base_vx`, `base_vy` | measured m/s | commanded m/s |
| 26 | `base_wz` | measured rad/s | commanded rad/s |

The exact names and order are asserted against
`dex_vega_lerobot_recorder/config/dexmate_blue_bird.yaml` at startup. Both hand
projection and expansion import the recorder's `reconstruct_hand_synergy` and
`expand_hand_synergy`; endpoints and the 0.02 recording disagreement tolerance
remain owned by the recorder configuration. Position limits are parsed from
the installed `vega_1p_f5d6.package.urdf`. The saved postprocessor must first
return exactly 27 finite physical values. PI0.5's 32-value padding/quantile
pipeline and GR00T's 132-value padding/min-max/Cosmos pipeline each remain in
their own serialized LeRobot preprocessor/postprocessor; neither is manually
recreated in ROS.

## Model and tokenizer acquisition

Use a read-only token in the process environment. Do not put it in YAML, shell
history, a unit file, or an image:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
read -rsp 'HF read token: ' HF_TOKEN && export HF_TOKEN && echo
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$PWD/.cache"
export TORCH_HOME="$PWD/.cache/torch"
export TRITON_CACHE_DIR="$PWD/.cache/triton"

src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/.venvs/lerobot/bin/download_model \
  --project-root /workspace \
  --model-revision be768eb6a4e32a58f66cadea7cd2159d99a16e86 \
  --checkpoint-tag step-015000 \
  --tokenizer-revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
  --model-dir data/models/pi05-dexmate-blue-bird/step-015000 \
  --tokenizer-dir \
    data/models/paligemma-3b-pt-224/35e4f46485b4d07967e7e9935bc3786aad50687c

unset HF_TOKEN
```

For the first candidate download only, `--allow-tag` accepts `step-005000`,
`step-015000`, or `step-030000`, resolves it through the Hub, downloads by the
resolved commit, and records both in `dexmate_artifact_manifest.json`. Production
launches must use the full commit. The downloader requires all seven inference
files listed in the handoff and validates the four JSON contracts. It never
uses `training_state/`.

All later loads use the local model/tokenizer directories. The server sets
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before importing LeRobot. The
tokenizer is preloaded from its pinned local snapshot and injected into the
saved tokenizer processor, avoiding its unpinned name-only network lookup.

## Thor Python environment

Validated on 2026-07-17:

- aarch64, Ubuntu 22.04, Jetson Linux R38.2.1 (JetPack 7 generation)
- ROS 2 Humble, Python 3.10.12
- system LeRobot 0.4.4
- the ROS devcontainer's Python remains CPU-only and unmodified
- GPU device nodes and Jetson driver child mounts are available to a private
  project-local mount namespace
- official NVIDIA PyTorch 25.11 ARM64 rootfs, pinned digest
  `sha256:4a85d8cf6fb3a943280960b8948cf4e9b6eca77b4414c68c9b2c7bb863f79b70`
- `.venvs/lerobot`: Python 3.12.3, LeRobot 0.6.0, NVIDIA Torch
  `2.10.0a0+b558c986e8.nv25.11`, CUDA 13.0, and working bfloat16 on Thor

LeRobot 0.6.0 requires Python >=3.12 and Torch >=2.7,<2.12. NVIDIA's current
[Jetson PyTorch matrix](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)
lists iGPU framework containers, rather than framework wheels, for JetPack 7.
The checked scripts export that exact official rootfs, keep it under
`.runtime/`, and launch it in a private mount namespace. Do not install PyPI
Torch:

```bash
src/dex_vega_lerobot_inference/scripts/bootstrap_jetson_runtime.sh "$PWD"

src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/src/dex_vega_lerobot_inference/scripts/create_lerobot_env.sh \
  /workspace /usr/bin/python3.12
```

The script validates CUDA, bfloat16 support, and an NVIDIA-compatible
Torchvision before installing anything,
installs the PI0.5/GR00T dependencies from a list that deliberately excludes
Torch and torchvision and an exact validated constraints file, installs `lerobot==0.6.0`
with `--no-deps`, and installs the local
recorder/inference Python packages. It exits without creating an environment if
Python 3.12 is absent. It exits before dependency installation if NVIDIA Torch
is not already visible.

## Start and observe

Start the policy server first. It has no ROS dependency and no actuation path:

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

The server verifies both local manifests, reports their resolved immutable
identity, and refuses to replace an existing server socket. Leave this terminal
open. Complete direct commands for 5k, 15k, and 30k are in the
[hardware runbook](docs/first_15k_hardware_trial.md#11-complete-direct-policy-server-commands-for-5k-15k-and-30k).

In a ROS Humble shell:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ROS_LOG_DIR="$PWD/.runtime/ros-log" \
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
ros2 launch dex_vega_lerobot_inference observe_only.launch.py
```

Observe `/dex_vega_lerobot_inference/status`, `diagnostics`, and
`predicted_action`. No command publishers exist in this process.

`dry_run` uses the same live recorder camera and measured ROS state while
making the non-actuating intent explicit; like observe-only, its process has no
command publishers. It additionally consumes the selected action prefix at the
30 Hz control cadence and runs the exact slew, hand, URDF, base, queue-age, and
observation-age adapters without publishing. The `shadow` status object records
every intervention and simulated starvation/fault condition:

```bash
ROS_LOG_DIR="$PWD/.runtime/ros-log" \
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
ros2 launch dex_vega_lerobot_inference dry_run.launch.py
```

Replay uses `sensor_msgs/Image` only as an explicit test interface:

```bash
ros2 launch dex_vega_lerobot_inference replay.launch.py
```

Replay `/joint_states`, `/dexcontrol/measured_base_twist`, and an `rgb8` or
`bgr8` image topic with their original timestamps. The adapter rejects stale,
duplicate, out-of-order, or excessively skewed samples. The local LeRobot
dataset is available under `data/production/dexmate_blue_bird`, but a LeRobot
dataset is not itself a rosbag. Deterministic ROS replay still needs a test-only
timestamp-preserving dataset publisher or a recorded bag.

## Guarded execution

One five-second live interface trial has been performed. The complete operator
sequence for the pinned 15k candidate is in
[first_15k_hardware_trial.md](docs/first_15k_hardware_trial.md). It includes the
required bridge restart, physical E-stop telemetry test, command-graph audit,
mandatory live-camera dry-run, guarded preflight, five-second first-trial cap,
and fault recovery. Its final section also records the exact immutable policy
server and guarded-launch command tuple for the step 30k candidate. Do not
reduce it to only the launch and service calls below.

GR00T has its own prepared, still-unauthorized five-second procedure in
[first_groot_n17_hardware_trial.md](docs/first_groot_n17_hardware_trial.md).
It requires the GR00T-specific readiness acknowledgement in addition to the
normal publication, trial-begin, and arm gates.

When separately authorized, execution has five independent gates:

1. launch `guarded_execution.launch.py` with
   `allow_command_publication:=true` (default is false),
2. load local model and tokenizer artifacts whose manifests record full
   immutable Hub commits,
3. maintain fresh false `/dexcontrol/estop_state`, fresh bridge applied
   telemetry, all bridge command subscribers, and exactly one publisher (this
   inference node) on every command topic,
4. maintain fresh disabled Pico status, or explicitly set
   `require_teleop_disabled:=false` only when Pico teleop is confirmed absent,
5. call `/dex_vega_lerobot_inference/arm` with `data: true` after beginning a
   short trial.

The guarded launch alone does not move the robot. Arming resets all queues and
publishes zero base while awaiting a fresh chunk. Fault, E-stop, stale data,
queue timeout, model/CUDA exception, teleop enable, disarm, end-trial, parameter
identity change, and shutdown all invalidate in-flight generations and clear
queues. The first-trial configuration also enters `FAULT` after a maximum armed
interval of 5.0 seconds. It then requires `end_trial`, safe-gate verification,
and an explicit `recover`; recovery never re-arms.

Every finite body-joint target and expanded hand-joint target is saturated to
the authoritative URDF position limits before publication. Each affected
action increments `joint_clamped_actions`; `last_joint_clamp` retains raw and
clipped values, and the node emits a throttled warning. This includes
floating-point boundary overshoot from hand-synergy expansion. Wrong action
shape/order, NaN/Inf, and missing or invalid URDF limits remain hard faults,
and the bridge's independent clipping remains active.

```bash
# Only after fresh authorization, reduced-limit review, and E-stop operator ready:
ros2 launch dex_vega_lerobot_inference guarded_execution.launch.py \
  allow_command_publication:=true \
  require_teleop_disabled:=false  # only for the confirmed no-Pico trial

ros2 service call /dex_vega_lerobot_inference/begin_trial std_srvs/srv/Trigger '{}'
ros2 service call /dex_vega_lerobot_inference/arm std_srvs/srv/SetBool '{data: true}'

# Stop/disarm:
ros2 service call /dex_vega_lerobot_inference/arm std_srvs/srv/SetBool '{data: false}'
ros2 service call /dex_vega_lerobot_inference/end_trial std_srvs/srv/Trigger '{}'
```

The guarded launch exposes `maximum_execution_duration_seconds` but keeps its
default at 5.0. After reviewing a clean five-second trial, extend duration in
stages rather than removing the cap. For example, the next trial is 15 seconds:

```bash
ros2 launch dex_vega_lerobot_inference guarded_execution.launch.py \
  allow_command_publication:=true \
  require_teleop_disabled:=false \
  maximum_execution_duration_seconds:=15.0
```

Duration is immutable after node startup and is reported in every status
message. Stop/restart the guarded node to change it. A clean 15-second trial may
be followed by 30 seconds after reviewing motion, clamp counts, latency, and
stop behavior; retain a finite cap for every trial.

States are `UNCONFIGURED -> MODEL_LOADING -> OBSERVE_ONLY` for non-executing
modes, or `... -> READY -> ARMED -> EXECUTING` for the guarded process. Any
fault enters `FAULT`; bridge E-stop enters `ESTOP`. Recovery returns only to
`READY` and always requires a deliberate re-arm. Inference never calls the
service to release E-stop.

## Candidate comparison

The requested first guarded trial uses `step-015000` (held-out loss 0.1042).
Then compare it with `step-005000` (0.1035) and `step-030000` (0.1258); the
small loss differences do not establish robot performance. For each candidate:

| Tag | Immutable commit | Local directory |
|---|---|---|
| `step-005000` | `6a511ca59438d1c7d4510dc08cecacce5b9b7014` | `data/models/pi05-dexmate-blue-bird/step-005000` |
| `step-015000` | `be768eb6a4e32a58f66cadea7cd2159d99a16e86` | `data/models/pi05-dexmate-blue-bird/step-015000` |
| `step-030000` | `305c4bf9067ead22c95befb810cdafbc6135cabb` | `data/models/pi05-dexmate-blue-bird/step-030000` |

Select a candidate only in the direct policy-server command; the
[hardware runbook](docs/first_15k_hardware_trial.md#11-complete-direct-policy-server-commands-for-5k-15k-and-30k)
contains complete commands for all three. Stop and restart the ROS inference
node after changing policy servers, but reuse the exact same guarded ROS launch
command. The node discovers the new server identity and independently rejects
any path, commit, tag, tokenizer, or manifest disagreement.

1. resolve and record its distinct full model commit,
2. run the same recorded observations in observe-only and save status/timing,
3. confirm finite/range/rate statistics and no command topics,
4. if authorized, use identical reduced limits, initial scene, maximum trial
   duration, and operator/E-stop procedure,
5. log model commit, tag, task, latency, queue gaps, safety interventions, and
   task outcome.

Do not select the final checkpoint merely because it trained longest.

## Validation status

Static tests, artifact verification, runtime versions, the head-only forward
pass, live RGB/state dry-run, DDS publisher audit, and latency/memory results
are in [validation_report.md](docs/validation_report.md). The measured default
is 2 Hz inference with an 18-step/0.60-second horizon; a clean run completed
47/47 predictions with no drops, model errors, or command publishers. Replay
data, the rebuilt bridge's E-stop telemetry restart, a 15k live-camera
benchmark, a 30k forward benchmark, and a separately authorized guarded trial
remain outstanding. The pinned 15k artifact has passed a head-only offline
`50x27` forward pass.

GR00T has a separate evidence boundary in
[groot_n17_validation_report.md](docs/groot_n17_validation_report.md). Its
runtime dependencies and module imports pass, but the private/gated artifacts,
forward pass, offline restart, live shadow run, and Thor latency/memory
measurements remain outstanding.
