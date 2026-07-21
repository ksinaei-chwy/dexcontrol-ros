# Validation report (2026-07-17)

Initial validation through Stage 5 sent no physical command: the live bridge
and robot were connected only so measured joint/base state and head RGB could
be read, and those policy tests used `mode=dry_run`. A later operator-supervised
Stage 6 interface trial published guarded commands for 5.025 seconds and stopped
on its configured duration fault; its telemetry is recorded below.

## Pinned artifacts

| Artifact | Immutable revision | Local directory |
|---|---|---|
| Dataset `Kasra99/dexmate_blue_bird` | `72a97b1a916699c17177e311463729d757f3119c` | not downloaded |
| Base `lerobot/pi05_base` | `7de663972b7817d2c4cf2d84c821153dfea772e9` | not required at inference |
| Fine-tuned 5k candidate | `6a511ca59438d1c7d4510dc08cecacce5b9b7014` | `data/models/pi05-dexmate-blue-bird/step-005000` |
| Fine-tuned 15k candidate | `be768eb6a4e32a58f66cadea7cd2159d99a16e86` | `data/models/pi05-dexmate-blue-bird/step-015000` |
| Fine-tuned 30k candidate | `305c4bf9067ead22c95befb810cdafbc6135cabb` | `data/models/pi05-dexmate-blue-bird/step-030000` |
| PaliGemma tokenizer | `35e4f46485b4d07967e7e9935bc3786aad50687c` | `data/models/paligemma-3b-pt-224/35e4f46485b4d07967e7e9935bc3786aad50687c` |

The Hub tag object supplied for `step-005000` resolved through the API to the
snapshot commit shown above. The 15k and 30k tags resolved to the distinct
commits shown above. All three model directories and the tokenizer directory
contain `dexmate_artifact_manifest.json`. The policy validator confirmed all seven
required inference files, parsed all JSON, checked the 9,354,050,752-byte
`model.safetensors`, and checked 27-value q01/q99 state/action statistics. The
saved config's state feature is correctly 32-dimensional because that is the
model-facing padded feature; the serialized normalizer remains physical 27-D.

The 15k and 30k downloads used a browser OAuth login without Git credential
storage. After download, the active token was logged out, the token file was
absent, and the remaining token registry was zero bytes. The requested
first-trial default is now the immutable 15k commit; the measurements below
are explicitly attributed to either the 5k live benchmark or the 15k offline
fixture so checkpoint results are not conflated.

Only tokenizer/config files were cached from `google/paligemma-3b-pt-224`; its
language-model weights were not duplicated. The policy server subsequently ran
with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and local paths only.

On 2026-07-21, checkpoint selection was consolidated into the PI0.5 policy
server. The ROS launches no longer duplicate model paths, commits, tags, or
tokenizer paths. The ROS node discovers the server identity over the local
socket, maps the reported `/workspace` paths into the host workspace, and
independently validates the same immutable manifests before becoming ready.
Prediction and reset responses are bound to that startup identity so replacing
the server beneath a running ROS node fails closed.
The policy server is a direct, non-ROS process started through
`scripts/run_jetson_runtime.sh`. The hardware runbook records complete commands
for the 5k, 15k, and 30k immutable model tuples and their shared tokenizer.
The guarded ROS launch remains checkpoint-independent.

## Project-local Thor runtime

The ROS Humble process must remain on Python 3.10, whereas LeRobot 0.6.0 needs
Python 3.12. The validated deployment therefore keeps a split process and a
Unix socket. The policy side uses an official NVIDIA ARM64 framework rootfs and
the venv remains under this repository.

| Item | Result |
|---|---|
| Architecture/kernel | `aarch64`, Linux `6.8.12-tegra` |
| ROS | Humble, Python 3.10.12 |
| NVIDIA framework image | `nvcr.io/nvidia/pytorch:25.11-py3` |
| Pinned ARM64 image digest | `sha256:4a85d8cf6fb3a943280960b8948cf4e9b6eca77b4414c68c9b2c7bb863f79b70` |
| Framework rootfs | `.runtime/rootfs/nvidia-pytorch-25.11-arm64` |
| LeRobot venv | `.venvs/lerobot` |
| Runtime Python | 3.12.3 |
| LeRobot | 0.6.0 |
| NVIDIA Torch | `2.10.0a0+b558c986e8.nv25.11` |
| Torchvision | `0.25.0a0+7a13ad0f` |
| Transformers | 5.5.4 |
| CUDA build/device | 13.0 / `NVIDIA Thor` |
| bfloat16 | reported supported; finite CUDA matmul passed |

The bootstrap uses `crane` 0.21.7 ARM64, whose downloaded release archive was
checked against SHA-256
`b6ee979d9411dfb05ce35ab9e156fe5de7def11a230764a7856ffa2eb971fa88`.
`run_jetson_runtime.sh` creates a private mount namespace, recursively binds
Jetson's injected NVIDIA child mounts, mounts the workspace at `/workspace`,
and removes all mounts automatically when the command exits. No sudo, system
package, system Python, generic PyPI Torch, or external cache was used.
Venv-local dependency versions are fixed by
`requirements/lerobot-0.6-pi-no-torch.constraints.txt`; packages inherited
from NVIDIA are fixed by the image digest.

An offline, constrained `pip install --dry-run` resolves the complete PI0.5
requirements from the local environment. A global `pip check` reports one
pre-existing framework-image warning: `nvidia-resiliency-ext 0.4.1+cuda13`
declares `pynvml`, which the image does not install. The inference runtime does
not import that extension; the pinned NVIDIA framework was left unchanged
rather than adding an unrelated package solely to silence the warning.

## Validation stages

### Stage 1 — static contracts

The original direct suite completed with `45 passed in 0.51s`; its matching
`colcon test` run completed with `45 passed in 0.48s`. On 2026-07-21, the
expanded PI0.5/GR00T regression suite completed with `78 passed in 0.61s`, 0
errors, 0 failures, and 0 skipped. Coverage includes:

- exact 27-value state/action names and order against the recorder config;
- recorder-identical left/right hand reconstruction and six-joint expansion;
- RGB/BGR and padded ROS image conversion;
- missing, stale, duplicate, out-of-order, future, and skewed timestamps;
- NaN/Inf, output shape, hand range, finite URDF joint clipping, invalid URDF
  limits, slew, base speed, and base acceleration rejection/clamping;
- one-slot latest-observation replacement and reset-generation invalidation;
- local artifact completeness, processor markers, quantile tensor shapes, and
  project-path enforcement;
- the fixed-size 50x27 Unix-socket response contract;
- checkpoint-independent PI0.5 ROS launches and the absence of a ROS
  policy-server launch;
- state-machine/no-actuation guarantees;
- exclusive command-publisher gating, including the no-Pico override case;
- the immutable, fail-closed five-second first-trial execution limit;
- real bridge name mapping, clipping, base watchdog, and the fact that
  `applied_*` topics are telemetry rather than inputs.

ROS `ament_flake8`, Python byte-compilation, `git diff --check`, and the ROS
build passed. The build covered `dexcontrol_ros`, `dex_camera_transport`,
`dex_vega_lerobot_recorder`, and `dex_vega_lerobot_inference`.

### Stage 2 — artifact/offline load

The complete pinned 5k policy, saved preprocessor, saved postprocessor, and
pinned local tokenizer loaded successfully with the network disabled for model
resolution. Model construction/remapping and CUDA transfer took 172.001 s.
All checkpoint keys loaded successfully. LeRobot 0.6.0's top-level policy
factory eagerly imports its dataset module, so `datasets` and PyAV are present
in the inference venv; no dataset videos or `torchcodec` are used by the policy
server.

### Stage 3 — forward pass

A deterministic 27-state/640x480 RGB fixture using the exact task produced a
finite postprocessed `(50, 27)` physical action chunk. The preprocessor accepted
only `observation.images.head`; PI0.5 inserted and masked absent wrist views.
The first CUDA pass measured:

| Measurement | Result |
|---|---|
| Total cold forward | 0.822 s |
| Cold GPU inference | 0.777 s |
| Preprocessing | 0.040 s |
| Peak CUDA allocated | 9,525,419,008 bytes |
| Peak CUDA reserved | 9,877,585,920 bytes |

Jetson's `nvidia-smi` reports integrated-memory usage as unsupported, so the
Torch peak counters are the available authoritative process measurement.

After selecting 15k as the requested first-trial default, a second fully
offline head-only fixture loaded commit
`be768eb6a4e32a58f66cadea7cd2159d99a16e86` with all keys successful and
returned finite postprocessed physical actions shaped `(50, 27)`:

| 15k fixture measurement | Result |
|---|---|
| Model load | 172.065 s |
| Total cold forward | 0.699 s |
| Cold GPU inference | 0.657 s |
| Preprocessing | 0.038 s |
| Peak CUDA allocated | 9,525,419,008 bytes |
| Peak CUDA reserved | 9,877,585,920 bytes |

This synthetic black-RGB/zero-state fixture validates artifact, processor,
missing-camera, CUDA, and output contracts. It does not measure task quality or
replace a 15k live-camera dry-run.

### Stage 4 — live ROS dry-run

The node consumed live `/joint_states`, live
`/dexcontrol/measured_base_twist`, and the recorder's direct `left_rgb` Zenoh
source. A direct camera probe produced contiguous `uint8` RGB `(480,640,3)`;
one measured sample had capture age 0.107 s, receive age 0.008 s, and transport
delay 0.098 s. No BGR conversion, manual resize to 224, normalization, or wrist
image duplication occurs before the saved LeRobot pipeline.

The ROS graph showed only these inference publishers:

- `/dex_vega_lerobot_inference/status`
- `/dex_vega_lerobot_inference/diagnostics`
- `/dex_vega_lerobot_inference/predicted_action`
- standard `/rosout` and `/parameter_events`

All seven authoritative command topics had publisher count zero while the
bridge subscribed to them. Status continuously reported
`execution_capable=false`, `actions_published=0`, and `state=OBSERVE_ONLY`.
The exact model/tokenizer commits and task were present in every status sample.

With the original 5 Hz submit rate, steady GPU inference was about
0.42-0.44 s and the one-slot worker correctly replaced pending observations.
At the measured 2 Hz operating point, a clean restart completed 47 of 47
submissions with zero replacements and zero errors. In a longer sample:

| Measurement | Observed range/result |
|---|---|
| Steady GPU inference | 0.423-0.432 s |
| Steady model total | 0.426-0.435 s |
| Observation to result at 2 Hz | 0.427-0.437 s |
| Dry-run queue age | 0.003-0.472 s |
| Dry-run observation-to-action age | 0.433-0.899 s |
| Peak CUDA allocated/reserved | 9.525 / 9.878 GB |
| Model errors | 0 |
| ROS actions published | 0 |

Ctrl-C invalidated/reset queues, stopped the camera, removed the ROS node, and
exited cleanly. The policy RPC reset succeeded and its server then removed the
Unix socket on clean shutdown.

A later read-only graph query confirmed the running bridge still exposes
`/soft_estop` as `std_srvs/srv/SetBool`, but not the new
`/dexcontrol/estop_state` topic. The service is therefore not broken; the live
process simply predates the rebuilt telemetry publisher. The new bridge polls
the vendor E-stop's `button_pressed` and `software_estop_enabled` fields. The
Pico status topic was also absent, consistent with the headset being
disconnected.

### Stage 5 — command-interface simulation

Tests instantiate neither a ROS command publisher nor `dexcontrol.Robot`, but
exercise the inference adapter's messages against the actual bridge mapping,
clipping, and watchdog methods. They confirm absolute body targets, expanded
hand targets, and base velocity enter the bridge's pre-clipping interfaces and
never the `applied_*` telemetry topics. No live command-interface simulation was
attempted while the real bridge was connected, because publishing even a mock
command onto its subscribed topics would violate this run's no-actuation scope.

## Measured synchronous-chunk defaults

| Parameter | Default | Rationale |
|---|---:|---|
| `control_frequency_hz` | 30 | dataset/control cadence |
| `inference_frequency_hz` | 2 | below measured ~2.3 Hz policy capacity; no replacements |
| `execution_horizon` | 18 | 0.60 s, covering 0.50 s cadence plus 0.10 s margin |
| `maximum_synchronization_skew_seconds` | 0.20 | 30k live path measured 0.144-0.150 s and crossed the former 0.15 s gate at 0.151 s |
| `maximum_action_queue_age_seconds` | 0.75 | above 0.60 s horizon, below half-chunk age |
| `maximum_observation_to_action_age_seconds` | 1.35 | covers measured inference plus complete horizon with margin |
| `action_wait_timeout_seconds` | 0.75 | faults beyond one expected chunk handoff |
| `maximum_execution_duration_seconds` | 5.0 | first hardware trial automatically faults after five seconds |

These are initial guarded-trial limits, not proof of task success. They must be
rechecked under the final power mode, thermal state, and competing GPU load.

The first arm attempt correctly published no policy actions and entered
`FAULT` when one sample measured 0.133 s of state/image skew against the former
0.120 s limit. The recorder did not use a cross-stream skew gate; it separately
accepted up to 0.30 s capture age and 0.25 s transport delay. Because the normal
live skew is already 0.096-0.106 s, the former limit had less than one 30 Hz
frame of margin. The guarded default was therefore changed to 0.15 s.

On 2026-07-21, the 30k live path twice stopped without publishing an action
when state/image skew reached 0.151 s. A passive eight-second sample measured
seven observations between 0.144471 and 0.149708 s (mean 0.147348 s), with
camera capture age between 0.154036 and 0.155256 s. At the user's explicit
request, the PI0.5 blue-bird cross-stream skew gate was increased to 0.20 s.
The authoritative 0.10 s state/receive, 0.30 s capture, and 0.25 s transport
age gates, action-age limits, motion limits, and finite execution cap remain
unchanged. This change does not apply to the separate GR00T configuration.

The next arm attempt also published no actions and entered `FAULT` because the
then-strict hand validator saw a postprocessed right open/close ratio of
`-0.002207`. This is quantile-unnormalization overshoot; the other three ratios
in that prediction were `0.002121`, `0.006433`, and `0.005894`.

The subsequent first physical interface trial executed for 5.025 seconds and
then entered the configured maximum-duration `FAULT`. It published 130 guarded
actions; 54 involved body/hand slew limiting, 121 involved base velocity or
acceleration limiting, and all 130 included a bounded hand-ratio clamp. Model
errors and pending-observation replacements were zero. At the post-trial status
sample, GPU inference was 0.430 s, observation-to-result was 0.435 s, total
observation-to-action age was 0.533 s, and state/image skew was 0.102 s. The
wireless E-stop remained false. Physical motion and operator observations must
be supplied by the operator; telemetry alone does not establish task quality.

A later 120-second-capped attempt stopped after 127 actions because a hand ratio
crossed the former `[-0.02, 1.02]` diagnostic band; a subsequent prediction
showed `right_hand.open_close_ratio=-0.021181`. At the operator's explicit
request, all finite hand ratios are now saturated to the physical `[0,1]`
range, counted in `hand_clamped_actions`, recorded in `last_hand_clamp`, and
reported through a throttled warning log. Hand NaN/Inf remains a hard fault,
hand slew and expanded-joint URDF limits remain active.

A later 30k attempt stopped before publishing because synergy expansion
produced `R_ff_j1=-1.0946000000000002` against the URDF lower limit `-1.0946`.
At the operator's explicit request, all finite body and expanded-hand joint
targets are now saturated to the authoritative URDF position limits rather
than faulting. Affected actions are counted in `joint_clamped_actions`, raw and
bounded values are recorded in `last_joint_clamp`, and the node emits a
throttled warning. Action shape/order errors, NaN/Inf, and missing, non-finite,
or inverted URDF limits remain hard faults. The bridge continues to apply its
own independent clipping.

## RTC evaluation

Installed LeRobot 0.6.0 contains PI0.5 RTC support in
`predict_action_chunk`, but the saved fine-tuned config has `rtc_config: null`.
LeRobot's RTC rollout also owns its action queue, inference-delay tracking, and
overlap/inpainting inputs. Enabling RTC here would mutate the saved model
configuration and change queue semantics before any reference replay exists.
Therefore this integration deliberately uses the benchmarked asynchronous
latest-observation worker plus a conservative synchronous receding horizon.
RTC can be run live in observe-only mode without a rosbag to measure latency,
queue behavior, and memory. A rosbag or equivalent dataset-to-topic replay is
recommended only for an apples-to-apples synchronous-versus-RTC comparison,
because it supplies identical images, state, and timestamps. Replay is not a
blocker to an initial live RTC dry-run, and RTC is not silently enabled for a
robot trial.

## Remaining before a real trial

1. Restart the rebuilt bridge and verify its new read-only
   `/dexcontrol/estop_state` (`std_msgs/msg/Bool`) telemetry. The bridge process
   that was already running did not expose this newly added gate.
2. The Pico headset is confirmed disconnected. For a no-Pico trial, explicitly
   disable only the Pico-status requirement and verify no other command
   publisher is connected. Exercise E-stop behavior without arming inference.
3. Provide a rosbag or deterministic dataset replay if RTC or checkpoint output
   comparisons are required. The pinned v3 dataset itself is not a rosbag.
4. Run a 15k live observe-only benchmark and a 30k offline/live benchmark
   before comparing them with 5k; do not assume 30k is best.
5. Review the mixed action predictions and reduced safety limits with the robot
   operator, then obtain fresh authorization for a short Stage 6 trial with the
   E-stop operator ready.

No task-success claim can be made from held-out loss or dry-run inference. No
Stage 6 trial was attempted.
