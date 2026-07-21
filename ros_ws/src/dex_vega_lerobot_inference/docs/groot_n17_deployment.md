# GR00T N1.7 deployment guide

This is the deployment contract for the Dexmate blue-bird GR00T candidate. It
does not authorize robot motion. As of 2026-07-21, the exact private/gated
artifacts are cached and manifested, the supplied weight size/SHA is verified,
synthetic and representative recorded-input complete-pipeline benchmarks pass,
and live observe-only/shadow runs exercise the real head camera, measured state,
queue, and safety adapter without command publishers. The selected live timing
is 3 Hz with a 21-step/0.70-second horizon and a measured 0.20-second
state/image skew gate. The candidate has not completed a physical rollout. See
[groot_n17_validation_report.md](groot_n17_validation_report.md) for the exact
evidence and [first_groot_n17_hardware_trial.md](first_groot_n17_hardware_trial.md)
for the prepared but still-unauthorized five-second interface trial.

## Immutable artifact set

| Role | Repository | Required revision | Local content |
|---|---|---|---|
| fine-tuned LeRobot policy | `Kasra99/groot-n17-dexmate-blue-bird` | `7f0f318540355031f189693e5623c1c5e8a17e93` (`step-034000`) | complete saved policy, preprocessor, postprocessor, and state/statistics |
| raw N1.7 base | `nvidia/GR00T-N1.7-3B` | `2fc962b973bccdd5d8ce4f67cc63b264d6886495` | complete base snapshot required while LeRobot constructs the model |
| Cosmos processor | `nvidia/Cosmos-Reason2-2B` | `9ce19a195e423419c349abfc86fd07178b230561` | tokenizer/image/video processor files only; no duplicate Cosmos weights |

The fine-tuned `model.safetensors` must be exactly 9,335,183,176 bytes with
SHA-256
`549616cb8e8aebab8d3fe35207f8389b18275f5e9a770fada51a9e62faeeca94`.
The downloader resolves every repository at the full commit, validates the
fine-tuned weight size/hash, rejects dynamic `auto_map` code, and writes a
SHA-256 inventory to `dexmate_artifact_manifest.json`. Later loads reject an
added, removed, or modified manifested file.

The base snapshot is intentionally complete. LeRobot 0.6.0 constructs
`GR00TN17` from `base_model_path` before loading the complete fine-tuned
state-dict with `strict=True`. Cosmos is different: the fine-tuned policy
already carries the VLM weights, and the saved VLM processor only needs the
pinned Qwen3-VL tokenizer and image/video processor assets.

The resolver has these pins compiled into the code. Supplying another
40-character commit is rejected; `main` and `step-034000` are not accepted as
runtime revisions even if they currently resolve to the expected commit.

## Process and safety boundary

ROS Humble remains on Python 3.10. LeRobot 0.6.0 and GR00T run in the existing
project-local Python 3.12 NVIDIA runtime:

```text
DexComm head RGB + measured ROS state       Python 3.12 / CUDA 13
                 |                                  |
                 v                                  v
       ROS safety node ------ Unix socket ----> GR00T policy server
       Python 3.10          JSON + raw arrays       saved pre/model/post
                 <--------- physical 40x27 ---------
                 |
                 +-- predicted_action/status/diagnostics in every mode
                 |
                 +-- command publishers exist only when both:
                     mode=armed AND allow_command_publication=true
```

The socket is repository-local, mode `0600`, and never uses pickle. The policy
server imports no ROS package and has no robot interface. The ROS node owns
freshness, arming, E-stop and teleop exclusion, URDF limits, slew limits,
queue age, finite-duration execution, and publication. `observe_only`,
`dry_run`, and `replay` cannot construct any bridge command publisher.

The external server is the sole artifact selector. On connection, the ROS
node requires protocol version 2, policy type `groot`, a `40x27` action
contract, the exact three commits and checkpoint tag above, and empty PI0.5
tokenizer identity. It maps the server's `/workspace` paths to the host
workspace, independently verifies all three manifests, and rejects any
identity change in a later prediction/reset response.

The guarded GR00T launch has an additional immutable
`execution_readiness_acknowledged` gate. Its default is false. It must stay
false until the missing offline and live observe-only evidence is reviewed.
All node parameters are immutable after construction, so artifact selection,
freshness rules, action bounds, and safety-gate requirements cannot be loosened
through the parameter service while a process is running. The policy server
also refuses to unlink any pre-existing socket path; an operator must resolve
an existing/stale endpoint before a new server can start.

## Training-to-runtime checks

Artifact loading fails unless both `config.json` and `train_config.json`
preserve the supplied training contract:

- policy type `groot`, LeRobot 0.6.0;
- `chunk_size=40`, `n_action_steps=40`, and one observation step;
- 132-dimensional internal state/action padding;
- `embodiment_tag=new_embodiment`;
- absolute actions (`use_relative_actions=false`) and no action decode
  transform;
- BF16 compute and BF16 parameter storage (`model_params_fp32=false`);
- no PEFT/LoRA;
- projector, diffusion model, and VLLN tuned;
- language model, vision tower, and top language layers frozen;
- batch size 8, seed 1000, and 170,000 configured training steps;
- dataset `Kasra99/dexmate_blue_bird` at commit
  `72a97b1a916699c17177e311463729d757f3119c`;
- one visual input, `observation.images.head`, plus physical 27-D state and
  27-D action features.

The saved processor JSON must contain the N1.7 pack and Cosmos VLM steps in
order and the `groot_action_unpack_unnormalize_v2` postprocessor. Its pack and
unpack state files must contain 27-value state/action min/max tensors. The
runtime never reimplements padding, min/max normalization, prompt formatting,
image processing, or action unnormalization.

The model-loading sequence is exactly:

1. set all Hugging Face/Transformers dataset and model APIs to offline mode;
2. validate the local manifests and serialized contracts;
3. load `PreTrainedConfig` from the fine-tuned directory with
   `local_files_only=True`;
4. replace only `base_model_path` with the verified local base directory and
   set the validated CUDA device;
5. call `GrootPolicy.from_pretrained(..., strict=True,
   local_files_only=True)` on the complete fine-tuned directory;
6. call `make_pre_post_processors` on that same directory;
7. override only the saved VLM processor's `model_name` with the verified
   local Cosmos processor path and device steps with CUDA/CPU placement;
8. call `predict_action_chunk` under `torch.inference_mode()` and the policy's
   saved BF16 autocast behavior;
9. run the saved postprocessor and require exactly 40 rows of 27 finite
   physical values.

## Exact ROS-to-policy mapping

The recorder configuration, not the URDF's lexical order, is the schema
oracle.

| Indices | State input | Postprocessed action |
|---|---|---|
| 0-2 | measured `torso_j1..j3.position` | absolute torso targets, rad |
| 3-5 | measured `head_j1..j3.position` | absolute head targets, rad |
| 6-12 | measured `L_arm_j1..j7.position` | absolute left-arm targets, rad |
| 13-19 | measured `R_arm_j1..j7.position` | absolute right-arm targets, rad |
| 20-21 | reconstructed left hand open/close and opposition ratios | absolute left hand ratios |
| 22-23 | reconstructed right hand open/close and opposition ratios | absolute right hand ratios |
| 24-25 | measured `base_vx`, `base_vy`, m/s | base velocity, m/s |
| 26 | measured `base_wz`, rad/s | base angular velocity, rad/s |

The head input is contiguous 640x480 `uint8` RGB from the recorder's
`DirectRgbCameraSource` (`left_rgb`, Zenoh
`sensors/head_camera/left_rgb`). State, camera capture, receive, transport,
and cross-stream timestamps retain the existing recorder/inference freshness
rules. The task is exactly `put the blue bird on the meeting desk`.

Finite hand ratios outside `[0,1]` follow the existing operator-authorized
exception: clamp, count, warn with raw/clamped values, then apply hand slew and
expanded-joint URDF limits. Hand NaN/Inf and all body-joint URDF violations are
hard faults. Base velocity and acceleration remain bounded. No other action
type receives the hand-ratio exception.

## Prepare the project-local runtime

Do not install a generic PyPI Torch wheel. Build or enter the environment only
through the pinned NVIDIA wrapper:

```bash
cd /workspaces/dexcontrol-ros/ros_ws

src/dex_vega_lerobot_inference/scripts/bootstrap_jetson_runtime.sh "$PWD"
src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/src/dex_vega_lerobot_inference/scripts/create_lerobot_env.sh \
  /workspace /usr/bin/python3.12

src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/.venvs/lerobot/bin/validate_runtime --require-groot
```

The validated GR00T additions are Accelerate 1.14.0, Diffusers 0.35.2,
dm-tree 0.1.9, PEFT 0.19.1, and timm 1.0.28. The readiness check imports the
actual Groot config, policy, and VLM processor modules in addition to probing
CUDA BF16.

## Download and verify

The Hugging Face account must be able to read the private fine-tune and must
have accepted the current NVIDIA GR00T and Cosmos repository terms. Use a
read-only token only in the environment:

```bash
cd /workspaces/dexcontrol-ros/ros_ws
read -rsp 'HF read token: ' HF_TOKEN
export HF_TOKEN

src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/.venvs/lerobot/bin/download_groot_model \
  --project-root /workspace

unset HF_TOKEN
```

The fixed destinations are:

```text
data/models/groot-n17-dexmate-blue-bird/7f0f318540355031f189693e5623c1c5e8a17e93
data/models/groot-n1.7-3b/2fc962b973bccdd5d8ce4f67cc63b264d6886495
data/models/cosmos-reason2-2b/9ce19a195e423419c349abfc86fd07178b230561
```

The command does not upload anything. After it succeeds, unset the token and
run an offline smoke benchmark:

```bash
src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/.venvs/lerobot/bin/benchmark_groot \
  --project-root /workspace --synthetic \
  --warmup-runs 1 --measured-runs 5
```

Synthetic black/zero input checks only loading, complete pre/postprocessing,
shape, finiteness, latency, GPU memory, and hard body limits; it is explicitly
marked as not rollout evidence. The representative benchmark used dataset
commit `72a97b1a916699c17177e311463729d757f3119c`, episode 0 frame 874 at
29.133 seconds. LeRobot 0.6.0 decoded its real head image through PyAV with
`return_uint8=true`; the blue bird is held between the arms. The recorder's
stored `float32[27]` state and decoded `uint8[480,640,3]` RGB were passed without
manual normalization:

```bash
src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/.venvs/lerobot/bin/benchmark_groot \
  --project-root /workspace \
  --observation-npz \
    /workspace/.runtime/groot_validation/recorded_episode_000_frame_0874.npz \
  --warmup-runs 1 --measured-runs 10
```

The benchmark returned finite `40x27` chunks, no body-joint or hand-range
violation, 162.33 ms median / 164.37 ms p95 total latency, and 10.032 GB peak
allocated GPU memory. Base values exceeded the configured velocity bounds 112
times across the 400 sampled action steps; the live shadow therefore exercises
the exact base velocity and acceleration adapter and reports its interventions.

## Start in observe-only mode

Start the non-ROS server first:

```bash
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

Then run the ROS half:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ROS_LOG_DIR="$PWD/.runtime/ros-log" \
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
ros2 launch dex_vega_lerobot_inference groot_observe_only.launch.py
```

Use `groot_dry_run.launch.py` for an explicitly named live shadow run. In that
mode the node consumes the selected prefix at 30 Hz and applies the same queue
ages, body/hand slew, URDF limits, and base limits as guarded execution, but it
has no command publishers. The `shadow` status object separates simulated
interventions and faults from published-action counters. Use
`groot_replay.launch.py` for timestamp-preserving ROS replay.

In every non-actuating mode, verify the exact three commits, `40x27` chunks,
freshness/worker/shadow counters, and publisher count zero on all seven command
topics. The selected 3 Hz/21-step parameters are stored in
`config/groot_n17_blue_bird.yaml`; the normal configuration and every safe
launch still keep `mode=observe_only`, command publication false, and execution
readiness unacknowledged.

Guarded mode is prepared only through the dedicated hardware runbook. It still
requires a fresh user authorization and E-stop operator, and it remains
disarmed until separate trial-begin and arm service calls. Offline loss,
forward passes, and clean shadow runs cannot establish physical safety or task
success.
