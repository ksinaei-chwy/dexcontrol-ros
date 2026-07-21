# GR00T N1.7 validation report (2026-07-21)

## Outcome

The safety architecture, exact artifact contract, ROS/LeRobot adapters, dynamic
`40x27` socket transport, project-local Jetson dependencies, CUDA BF16 probe,
actual LeRobot GR00T module imports, full saved pipeline, and model-loaded live
observe-only process are implemented and validated without publishing a robot
command.

All three exact artifacts are now local and manifested. The fine-tuned weight
independently matches the required 9,335,183,176 bytes and SHA-256. An
uncontended synthetic complete-pipeline benchmark produced finite `40x27`
actions with 165.48 ms median total latency and 10.032 GB peak allocated GPU
memory. A real recorded task observation passed ten complete-pipeline runs at
162.33 ms median total latency with no body/hand range violation. The selected
3 Hz/21-step live shadow then completed 843 additional predictions in five
minutes with zero worker errors/drops, queue starvation, stale actions, hard
action errors, published actions, or command-topic publishers. These results
establish a working Jetson forward path and prepare a guarded interface trial;
they do not establish physical safety or task performance.

## Scope and safety

No command publisher, `dexcontrol.Robot`, E-stop release, arm request, or
physical trial was used. Runtime tests were limited to source/unit tests and
the isolated project-local Python 3.12 CUDA environment. The existing PI0.5
validation record and its prior operator-authorized physical trial are
separate; none of those results apply to this GR00T candidate.

## Inspected authorities

Before defining the integration, the following packages and their current
documentation/configuration were inspected:

- `dex_vega_lerobot_recorder`: exact feature order, hand projection,
  head-camera RGB source, timestamps, task, and freshness limits;
- `dexcontrol_ros`: command types, cached joint targets, clipping, base
  watchdog, applied telemetry, E-stop telemetry, and stop behavior;
- `dexmate_vega_description`: authoritative position limits and joint model;
- `dexmate_vega_moveit_config`: trajectory-to-bridge command adapter and
  competing publisher behavior;
- `dexcontrol_navigation`: `/cmd_vel` ownership and navigation launch paths;
- `dex_pico_teleop`: command publisher set and teleop status exclusion;
- `dex_camera_transport`: direct RGB transport, timestamps, and latest-frame
  semantics;
- the installed LeRobot 0.6.0 GR00T configuration, model, processor, pipeline,
  and serialization source in `.venvs/lerobot`.

## Implemented checks

The fine-tuned resolver requires all saved LeRobot files, the exact
9,335,183,176-byte weight and supplied SHA-256, compatible saved config and
training config, the exact dataset commit, N1.7 pack/VLM/unpack steps, and
27-value min/max processor state. It rejects relative actions, an action decode
transform, PEFT/LoRA, wrong tuning flags, wrong feature dimensions, unpinned
revisions, dynamic remote code, path escape, or any manifest inventory change.

The base resolver requires a GR00T N1.7/Cosmos configuration, a complete
single or indexed sharded safetensors set, and an immutable manifest. The
Cosmos resolver requires only the pinned tokenizer/image/video processor set
and rejects any Cosmos model weight or shard so the 4.88 GB model is not
silently duplicated. Manifest verification hashes each multi-gigabyte file
once per verifier rather than rereading it merely to compare its size.

The policy server loads only local snapshots in offline mode, calls LeRobot's
saved processors, requests the complete GR00T chunk, and accepts only a finite
`[40,27]` postprocessed result. The protocol includes the full policy/base/
processor identity on info, reset, and prediction responses. The ROS node
independently resolves that identity and rejects a server restart or artifact
change. Non-finite JSON/actions and non-integral payload lengths are rejected;
accepted connections have a finite I/O timeout, and an existing socket is
never silently unlinked.

All existing ROS execution gates remain active. GR00T adds a default-false
execution-readiness acknowledgement. The measured live configuration is 3 Hz
with a 21-step/0.70-second execution horizon, 0.75-second maximum queue age,
1.35-second maximum observation-to-action age, and 0.20-second state/image skew
gate. Every ROS node parameter is frozen after construction, so a restart and
complete gate re-evaluation are required for any change.

`dry_run` now consumes the selected prefix at 30 Hz and runs the same queue-age,
body/hand slew, URDF, and base adapters as armed execution without constructing
command publishers. Its separate `shadow` counters expose simulated starvation,
stale actions, hard adaptation errors, and every safety intervention.

## Commands and results

Repository tests:

```bash
source /opt/ros/humble/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src/dex_vega_lerobot_inference:src/dex_vega_lerobot_recorder:src/dex_camera_transport:$PYTHONPATH \
python3 -m pytest -q src/dex_vega_lerobot_inference/test
```

The package-level `colcon test` result was `81 tests, 0 errors, 0 failures, 0
skipped` (`81 passed`; the final run's pytest phase completed in 0.57 seconds).

```bash
python3 -m ament_flake8.main \
  src/dex_vega_lerobot_inference/dex_vega_lerobot_inference \
  src/dex_vega_lerobot_inference/test
```

Result: 36 files checked, no problems found.

The project-local GR00T dependencies were installed without replacing the
NVIDIA Torch/Torchvision inherited from the pinned rootfs. The strengthened
runtime probe was run through `scripts/run_jetson_runtime.sh`:

```text
architecture                 aarch64
python                       3.12.3
lerobot                      0.6.0
torch                        2.10.0a0+b558c986e8.nv25.11
torchvision                  0.25.0a0+7a13ad0f
CUDA                         13.0
device                       NVIDIA Thor
bfloat16 matmul              passed
transformers                 5.5.4
accelerate                   1.14.0
diffusers                    0.35.2
dm-tree                      0.1.9
peft                         0.19.1
timm                         1.0.28
GR00T policy/module imports  passed
```

The first module import found that NVIDIA `transformer_engine` invokes
`ldconfig`, while the private runtime namespace omitted `/usr/sbin` and `/sbin`
from `PATH`. The wrapper was corrected, and validation now imports
`GrootConfig`, `GrootPolicy`, and `GrootN17VLMEncodeStep`; a versions-only pass
can no longer mark GR00T ready.

A full `pip check` additionally reports that the inherited NVIDIA rootfs
package `nvidia-resiliency-ext 0.4.1+cuda13` declares a separate `pynvml`
distribution. Review confirmed that the importable `pynvml` module is supplied
by rootfs `nvidia-ml-py`, no workspace policy code imports the resiliency
extension or NVML, and the CUDA/BF16, actual GR00T module, model load, offline
forward, and live shadow probes all pass. This pinned-rootfs metadata mismatch
is explicitly accepted for this policy path; the environment is still not
described as globally dependency-clean, and no package was installed merely to
silence it.

Artifact acquisition command:

```bash
src/dex_vega_lerobot_inference/scripts/run_jetson_runtime.sh "$PWD" \
  /workspace/.venvs/lerobot/bin/download_groot_model \
  --project-root /workspace
```

The original no-token attempt failed closed as designed. After authentication
was supplied to the downloader process, all three pinned snapshots completed
and wrote immutable manifests. The fine-tuned `model.safetensors` independently
verified as:

```text
size    9335183176 bytes
SHA256  549616cb8e8aebab8d3fe35207f8389b18275f5e9a770fada51a9e62faeeca94
```

This network applies Cisco Umbrella TLS inspection to Hugging Face's CDN. Xet
CAS reconstruction first failed, and standard HTTP then correctly rejected the
untrusted inspection chain. TLS verification was not disabled. The official
Cisco Umbrella root was placed under `.runtime/ca/`, combined with the pinned
rootfs public bundle, and passed explicit API/CDN verification before the
download resumed with Xet disabled. No system trust store was changed and no
credential was written to this workspace.

An adapter-only ROS process was then started with `policy_type=groot`,
`mode=observe_only`, `allow_command_publication=false`, `load_model=false`,
`camera_source=ros_image`, and a unique node name. `ros2 node info
/groot_observe_validation` reported only these application publishers:

```text
/dex_vega_lerobot_inference/diagnostics
/dex_vega_lerobot_inference/predicted_action
/dex_vega_lerobot_inference/status
```

It reported no `/cmd_vel` or joint/hand command publisher. This adapter-only
check was later superseded by the model-loaded live result below.

The first model-loaded shadow sample overlapped an independently operated PI0.5
process, so its latency was not treated as the uncontended baseline. After
`nvidia-smi` showed no CUDA process, the synthetic benchmark and live shadow
were repeated. The live run used the real `left_rgb` Zenoh camera and measured
ROS state, completed 80/80 predictions with zero errors or replacements, and
reported `actions_published=0`. A whole-graph audit showed publisher count zero
and bridge subscription count one on all six component command topics and
`/cmd_vel`.

The representative input was LeRobot dataset commit
`72a97b1a916699c17177e311463729d757f3119c`, episode 0 frame 874 at 29.133333
seconds. The recorder state came directly from `observation.state`; LeRobot
0.6.0/PyAV decoded `observation.images.head` with `return_uint8=true`. The RGB
frame visibly contains the blue bird held between the two arms. The generated
repository-local validation NPZ has SHA-256
`da162ba7ec5ce8be71cfa3ef798812e6f253b7b2e496612206211692850b7f9e`.

## Offline and shadow results

| Check | Result |
|---|---|
| exact revision/manifest validator unit fixtures | pass |
| saved min/max pre/postprocessor fixture | pass |
| `40x27` RPC serialization and validation | pass |
| non-actuating launch/default source checks | pass |
| adapter-only ROS observe node publisher audit | pass; no command publishers |
| Jetson CUDA BF16 | pass |
| actual LeRobot GR00T imports | pass |
| model weight size and SHA on downloaded artifact | pass; exact supplied values |
| strict complete model load | pass |
| complete saved pre/model/post forward pass | pass; finite `40x27` |
| token-unset local/offline-mode policy-server startup | pass |
| hard network-isolated startup and forward pass | pass; finite `40x27`, exit 0 |
| representative recorded observation | pass; 10 finite `40x27` chunks, no body/hand range violation |
| live head-camera/state observe-only run | pass; uncontended 80/80 |
| five-minute 3 Hz/21-step action-adapting shadow | pass; 843 additional predictions, no worker/queue/action fault |
| live command-topic publisher audit with GR00T loaded | pass; all seven zero |
| guarded five-second runbook | prepared; still requires fresh authorization and E-stop operator |
| physical rollout | not authorized and not run |

## Jetson latency and memory

The uncontended five-run synthetic benchmark (one warm-up) recorded:

| Measurement | Result |
|---|---:|
| load | 16.570 s |
| preprocessing median | 8.297 ms |
| GPU inference median / p95 | 155.293 / 155.694 ms |
| postprocessing median | 0.402 ms |
| total median / p95 | 165.478 / 166.084 ms |
| peak allocated / reserved GPU memory | 10,032,018,944 / 10,284,433,408 bytes |

It returned finite `40x27` actions with no body-joint limit violation and no
hand value outside `[0,1]`. Eleven synthetic base values exceeded the configured
base limit and were counted; the benchmark still passed the structural action
contract. Because zero state/black RGB is not representative and GR00T sampling
is stochastic, that count is not a rollout rate.

The ten-run representative benchmark (one warm-up) recorded:

| Measurement | Result |
|---|---:|
| load | 16.612 s |
| preprocessing median | 8.155 ms |
| GPU inference median / p95 | 152.101 / 153.812 ms |
| postprocessing median | 0.403 ms |
| total median / p95 | 162.325 / 164.365 ms |
| peak allocated / reserved GPU memory | 10,032,018,944 / 10,284,433,408 bytes |

All 400 predicted steps were finite with no body-joint violation and no hand
value outside `[0,1]`. There were 112 raw base components outside the configured
velocity bounds. This is not a physical rollout rate; live `dry_run` therefore
applied the complete base velocity/acceleration adapter and counted each
intervention.

At the uncontended live status sample, total inference was 171.587 ms, GPU
inference was 159.105 ms, state/image skew was 0.129 s, and observation-to-action
age was 0.833 s. The run completed 80/80 generations without errors or drops.
Separately, `unshare --net` removed the process network interface and a cold
zero-warm-up synthetic run still loaded in 16.703 s and returned a finite
`40x27` chunk. Its 943 ms first-pass total is a cold-start/offline proof, not a
steady-state latency sample.
The first five-minute 2 Hz/18-step baseline retained the original 0.15-second
skew gate. It completed 430 additional predictions without model errors/drops
or command publishers, but rejected transient cross-stream skew up to 0.200
seconds and accumulated 131 simulated queue starvations. The measured skew gate
was changed to 0.20 seconds without changing the independent state, capture,
receive, or transport age gates. A second five-minute 2 Hz/18-step run had no
non-duplicate observation error but exposed one late-window queue starvation,
so that cadence was not selected.

The final five-minute 3 Hz/21-step qualification recorded 299 status samples
and 843 additional predictions (903/903 overall completed) with zero worker
errors, drops/replacements, queue starvation, stale queue actions, stale
observation actions, or hard shadow adaptation errors. It evaluated 9,568
shadow actions: 9,380 required body/hand slew intervention and 8,819 required
base velocity/acceleration intervention; none required a body/expanded-hand
URDF clamp or hand-ratio clamp. This high intervention rate is expected while
the robot remains stationary and stochastic predictions are simulated, but it
must be reviewed during the five-second direction check.

Final live timing was 170.771 ms median / 201.031 ms maximum total and 158.698
ms median / 180.224 ms maximum GPU inference. Maximum measured skew was 0.182
seconds, queue age 0.424 seconds, and observation-to-action age 0.599 seconds.
Twenty sampled status messages reported an expected duplicate camera timestamp
when the 30 Hz timer polled before a new 30 Hz frame; there was no non-duplicate
freshness error or starvation. Start, midpoint, and end graph audits all showed
publisher count zero and bridge subscription count one on every command topic.

## Remaining physical-rollout boundary

The non-actuating evidence, measured parameters, dependency review, and staged
five-second procedure are complete. Safe defaults remain
`execution_readiness_acknowledged=false` and command publication false. A
physical trial is still blocked until the user gives fresh GR00T-specific
authorization and a physical E-stop operator explicitly confirms readiness.
The existing PI0.5 authorization does not cover GR00T. Follow only
[first_groot_n17_hardware_trial.md](first_groot_n17_hardware_trial.md); do not
infer permission from this report.

The step-34k checkpoint is only the best offline-loss candidate. Nothing in
this report claims physical safety, task success, or successful robot behavior.
