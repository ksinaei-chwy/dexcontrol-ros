# Repository agent guide

This ROS 2 workspace targets the Dexmate Vega 1 Pro on Jetson Thor. Preserve
unrelated local changes: several camera, teleoperation, recorder, and bridge
files may be intentionally uncommitted.

## Safety boundary

- Never initialize `dexcontrol.Robot`, publish a robot command, release E-stop,
  or run a physical trial during automated validation.
- `dex_vega_lerobot_recorder` is authoritative for the 27-value state/action
  contract, hand synergies, camera RGB transport, and freshness rules.
- `dexcontrol_ros` is authoritative for command topic types, clipping, cached
  joint targets, base watchdog, applied telemetry, and stop behavior.
- `/dexcontrol/applied_joint_commands` and
  `/dexcontrol/applied_base_twist` are telemetry outputs, never command inputs.
- The inference default is `observe_only`; only `mode=armed` together with
  `allow_command_publication=true` constructs command publishers. A trial-begin
  service, verified immutable model/tokenizer manifests, fresh safety/data
  gates, and a separate arm service call are still required.
- A fault stops new joint/hand targets and sends zero `/cmd_vel`. The bridge
  holds its last cached joint targets; do not invent a return pose or controller
  disable operation.

One user-authorized five-second physical interface trial has completed. Every
additional Stage 6 trial still requires fresh user authorization and
confirmation that an E-stop operator is ready.

## Validated architecture

ROS Humble uses Python 3.10.12 here, while LeRobot 0.6.0 requires Python >=3.12.
Keep the split-process boundary in `dex_vega_lerobot_inference`: the ROS 3.10
node owns safety/publication and a local Python 3.12 CUDA policy server owns
LeRobot, the saved processors, and PI0.5 or GR00T N1.7. Their project-local
Unix socket uses JSON metadata plus fixed-size raw arrays, not pickle.

For PI0.5, select the 5k/15k/30k checkpoint only in the policy-server command.
All PI0.5 ROS launch files are checkpoint-agnostic. At startup the ROS node
discovers the server model/tokenizer identity, maps its `/workspace` paths into
this workspace, independently verifies the immutable local manifests, and
rejects any disagreement. Each predict/reset response is identity-bound, so a
server checkpoint change requires restarting the ROS node and cannot silently
change the actions accepted by a running guarded process.
The identity-bound PI0.5 RPC protocol is version 2; after rebuilding this
package, restart both halves rather than connecting a new ROS node to an old
policy-server process.
Start the PI0.5 policy server directly through
`scripts/run_jetson_runtime.sh`; it is not a ROS launch. Complete immutable 5k,
15k, and 30k commands are in `docs/first_15k_hardware_trial.md`. Future trained
tasks should provide their own model/task runtime configuration while reusing
the checkpoint-agnostic guarded launch and safety node.

Validated 2026-07-17: aarch64, Jetson Linux 6.8.12-tegra, ROS Python 3.10.12,
and a project-local NVIDIA PyTorch 25.11 ARM64 runtime. `.venvs/lerobot` uses
Python 3.12.3, LeRobot 0.6.0, Torch
`2.10.0a0+b558c986e8.nv25.11`, CUDA 13.0, and working bfloat16 on NVIDIA Thor.
Do not install a generic PyPI Torch wheel or bypass
`scripts/run_jetson_runtime.sh` for the LeRobot venv.

The documented first-trial policy-server selection is the 15k policy at commit
`be768eb6a4e32a58f66cadea7cd2159d99a16e86`; the PaliGemma tokenizer commit is
`35e4f46485b4d07967e7e9935bc3786aad50687c`. Complete 5k, 15k, and 30k
artifacts are under `data/models/` with `dexmate_artifact_manifest.json`.
The 5k candidate has passed a head-only forward pass and live camera/state
dry-run. The 15k candidate has passed a head-only offline forward pass. The
30k candidate is artifact-validated only; do not claim it passed a forward
benchmark until that test is recorded.

GR00T N1.7 is implemented on the same split-process boundary. The only accepted
fine-tune is
`Kasra99/groot-n17-dexmate-blue-bird` commit
`7f0f318540355031f189693e5623c1c5e8a17e93` (`step-034000`), with base
`nvidia/GR00T-N1.7-3B` commit
`2fc962b973bccdd5d8ce4f67cc63b264d6886495` and Cosmos processor commit
`9ce19a195e423419c349abfc86fd07178b230561`. The external policy server is the
sole artifact selector; the ROS node independently verifies all three local
manifests and identities. The GR00T venv dependencies, CUDA BF16 probe, and
actual policy/processor imports passed on 2026-07-21. Later that day all three
artifacts were cached and manifested; the fine-tuned weight independently
matched 9,335,183,176 bytes and SHA-256
`549616cb8e8aebab8d3fe35207f8389b18275f5e9a770fada51a9e62faeeca94`.
An uncontended five-run synthetic forward benchmark returned finite `40x27`
actions with 165.48 ms median total latency, 155.29 ms median GPU latency, and
10.032 GB peak allocated GPU memory. An uncontended live observe-only run
completed 80/80 predictions with zero errors, drops, actions, or command-topic
publishers. A separate network-namespace-isolated cold forward pass also
completed with no network interface. A real recorded frame (dataset episode 0,
frame 874) passed ten complete-pipeline predictions with 162.33 ms median total
latency, no body/hand range violation, and 112 raw base-limit exceedances across
400 predicted steps. Synthetic and recorded-input results are not rollout
evidence.

Validated later on 2026-07-21: `groot_dry_run` consumes and safety-adapts the
selected prefix at 30 Hz without constructing command publishers. A 0.15-second
skew/2 Hz/18-step five-minute baseline exposed rejected skew up to 0.200 seconds
and 131 simulated starvations; 0.20-second skew with 2 Hz/18 steps reduced that
to one late-window starvation. The selected 3 Hz/21-step, 0.20-second-skew
five-minute run completed 843 additional predictions with zero worker errors,
drops, starvation, stale actions, hard adaptation errors, published actions,
or command-topic publishers. Maximum queue age was 0.424 seconds and
observation-to-action age 0.599 seconds. Keep these values unless a new
observe-only measurement justifies a change.

The inherited `nvidia-resiliency-ext 0.4.1+cuda13` metadata still requests a
separate `pynvml` distribution, but rootfs `nvidia-ml-py` supplies the importable
module, no policy path imports the resiliency extension/NVML, and all GR00T
runtime checks pass. This discrepancy is explicitly accepted for this policy
path without installing another package. The guarded five-second GR00T runbook
is `src/dex_vega_lerobot_inference/docs/first_groot_n17_hardware_trial.md`.
Safe defaults remain `execution_readiness_acknowledged=false` and command
publication false; a physical trial still requires fresh user authorization
and an E-stop operator.

## Project-local files and caches

Keep environments, models, sockets, logs, and caches under this workspace:

- `.venvs/lerobot`
- `data/models/`
- `.runtime/`
- `.cache/`

Never store `HF_TOKEN`; use a read-only token in the environment only. Do not
upload artifacts from this workspace.

## Build and non-actuating tests

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select \
  dexcontrol_ros dex_camera_transport dex_vega_lerobot_recorder \
  dex_vega_lerobot_inference

source install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
colcon test --packages-select dex_vega_lerobot_inference \
  --event-handlers console_direct+
colcon test-result --verbose \
  --test-result-base build/dex_vega_lerobot_inference

python3 -m ament_flake8.main \
  src/dex_vega_lerobot_inference/dex_vega_lerobot_inference \
  src/dex_vega_lerobot_inference/test
git diff --check
```

The system pytest plugin set is incompatible unless plugin autoload is
disabled. Set `ROS_LOG_DIR=$PWD/.runtime/ros-log` during local ROS runs so logs
remain inside the repository.

Create or enter the policy runtime only through:

```bash
src/dex_vega_lerobot_inference/scripts/bootstrap_jetson_runtime.sh "$PWD"
src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/.venvs/lerobot/bin/validate_runtime
```

The validated PI0.5 synchronous defaults are 2 Hz inference, an 18-step/0.60-second
horizon, 0.75-second maximum queue age, and 1.35-second maximum
observation-to-action age. The clean live run completed 47/47 predictions with
zero drops/errors and every command topic at publisher count zero. The
PI0.5 state/image skew gate is 0.20 seconds. On 2026-07-21, the 30k live path
measured 0.144-0.150 seconds during a passive non-executing sample and twice
crossed the former 0.15-second gate at 0.151 seconds; the user explicitly
requested the 0.20-second setting. The separate state, receive, capture,
transport, queue-age, and observation-to-action gates remain unchanged. Do not
loosen these values again without a new observe-only measurement.

## Inference references

- Package guide: `src/dex_vega_lerobot_inference/README.md`
- Validation record: `src/dex_vega_lerobot_inference/docs/validation_report.md`
- First 15k guarded-trial runbook:
  `src/dex_vega_lerobot_inference/docs/first_15k_hardware_trial.md`
- GR00T validation record:
  `src/dex_vega_lerobot_inference/docs/groot_n17_validation_report.md`
- First GR00T guarded-trial runbook:
  `src/dex_vega_lerobot_inference/docs/first_groot_n17_hardware_trial.md`
- Runtime config: `src/dex_vega_lerobot_inference/config/pi05_blue_bird.yaml`
- GR00T deployment guide:
  `src/dex_vega_lerobot_inference/docs/groot_n17_deployment.md`
- GR00T validation record:
  `src/dex_vega_lerobot_inference/docs/groot_n17_validation_report.md`
- GR00T runtime config:
  `src/dex_vega_lerobot_inference/config/groot_n17_blue_bird.yaml`
- Exact task: `put the blue bird on the meeting desk`
- Dataset commit: `72a97b1a916699c17177e311463729d757f3119c`
- Base PI0.5 commit: `7de663972b7817d2c4cf2d84c821153dfea772e9`
- Fine-tuned 5k commit: `6a511ca59438d1c7d4510dc08cecacce5b9b7014`
- Fine-tuned 15k commit: `be768eb6a4e32a58f66cadea7cd2159d99a16e86`
- Fine-tuned 30k commit: `305c4bf9067ead22c95befb810cdafbc6135cabb`
- Tokenizer commit: `35e4f46485b4d07967e7e9935bc3786aad50687c`

LeRobot state padding, policy-specific quantile/min-max normalization, camera
mapping, image processing, tokenization, and action unnormalization must remain
in the serialized LeRobot 0.6.0 preprocessor/postprocessor. Never reproduce
those steps manually.
At the operator's explicit request, every finite postprocessed hand ratio is
clamped to `[0,1]`, counted, and emitted as a throttled warning with raw and
clamped values. Every finite body-joint and expanded hand-joint target is also
clipped to its authoritative URDF position limits, counted in
`joint_clamped_actions`, recorded in `last_joint_clamp`, and emitted as a
throttled warning. The exact `R_ff_j1=-1.0946000000000002` versus `-1.0946`
boundary case is covered by tests. Action shape/order errors, NaN/Inf, and
missing, non-finite, or inverted URDF limits remain hard faults; hand/body slew
limits and the bridge's independent clipping remain active.

The bridge process initially observed during validation predated the new
`/dexcontrol/estop_state` telemetry. It was deliberately replaced and its
physical E-stop telemetry checked before the first guarded trial. Future bridge
restarts and trials must follow the same operator runbook.

Validated later on 2026-07-17: the first physical interface trial published 130
guarded actions and stopped on its 5.025-second duration fault with no model
errors. The guarded launch now exposes an explicit duration argument, default
5.0 seconds. Extend only in reviewed stages (next 15 seconds, then 30 seconds)
and never remove the finite cap.
