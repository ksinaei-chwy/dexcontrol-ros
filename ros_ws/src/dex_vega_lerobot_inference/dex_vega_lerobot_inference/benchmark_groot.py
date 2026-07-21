"""Non-actuating offline forward benchmark for the pinned GR00T candidate."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .action_adapter import load_joint_limits_from_urdf
from .contracts import BODY_JOINT_NAMES, STATE_DIMENSION, TASK
from .groot_artifact import resolve_groot_artifacts
from .groot_contracts import (
    BASE_MODEL_REVISION,
    COSMOS_PROCESSOR_REVISION,
    MODEL_REVISION,
)
from .groot_policy_runtime import GrootPolicyRuntime
from .observation_adapter import ObservationSnapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load the pinned local GR00T bundle offline and benchmark complete "
            "saved preprocessor -> policy -> postprocessor inference. This tool "
            "has no ROS imports or robot command interfaces."
        )
    )
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument(
        "--model-dir",
        default=f"data/models/groot-n17-dexmate-blue-bird/{MODEL_REVISION}",
    )
    parser.add_argument(
        "--base-model-dir",
        default=f"data/models/groot-n1.7-3b/{BASE_MODEL_REVISION}",
    )
    parser.add_argument(
        "--cosmos-processor-dir",
        default=f"data/models/cosmos-reason2-2b/{COSMOS_PROCESSOR_REVISION}",
    )
    parser.add_argument(
        "--observation-npz",
        help="NPZ with state [27] float and rgb [480,640,3] uint8 arrays.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use an all-zero smoke input; results are not rollout evidence.",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--urdf",
        default="src/dexmate_vega_description/urdf/vega_1p_f5d6.package.urdf",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.observation_npz) == bool(args.synthetic):
        raise ValueError("select exactly one of --observation-npz or --synthetic")
    if args.warmup_runs < 0 or args.measured_runs < 1:
        raise ValueError("warmup-runs must be nonnegative and measured-runs positive")

    project_root = Path(args.project_root).expanduser().resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    artifacts = resolve_groot_artifacts(
        project_root=project_root,
        model_local_path=project_root / args.model_dir,
        base_model_local_path=project_root / args.base_model_dir,
        cosmos_processor_local_path=project_root / args.cosmos_processor_dir,
        allow_download=False,
        local_files_only=True,
    )
    state, rgb = _load_observation(
        project_root,
        args.observation_npz,
        synthetic=bool(args.synthetic),
    )
    joint_limits = load_joint_limits_from_urdf(project_root / args.urdf)
    runtime = GrootPolicyRuntime(
        project_root=project_root,
        artifacts=artifacts,
        device=args.device,
        require_cuda=True,
        require_bfloat16=True,
    )

    for _ in range(args.warmup_runs):
        runtime.reset()
        runtime.predict(_snapshot(state, rgb))

    predictions = []
    for _ in range(args.measured_runs):
        runtime.reset()
        predictions.append(runtime.predict(_snapshot(state, rgb)))

    actions = np.stack([prediction.actions for prediction in predictions])
    body_violations = _body_limit_violations(actions, joint_limits)
    report = {
        "policy_type": runtime.info.policy_type,
        "model_commit": runtime.info.model_commit,
        "base_model_commit": runtime.info.base_model_commit,
        "cosmos_processor_commit": runtime.info.processor_commit,
        "checkpoint_tag": runtime.info.checkpoint_tag,
        "offline_environment": True,
        "observation_source": "synthetic_zero" if args.synthetic else args.observation_npz,
        "synthetic_is_rollout_evidence": False,
        "load_seconds": runtime.info.load_seconds,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "action_shape_per_run": list(actions.shape[1:]),
        "action_finite": bool(np.all(np.isfinite(actions))),
        "body_joint_limit_violations": body_violations,
        "hand_values_outside_unit_interval": int(
            np.count_nonzero((actions[..., 20:24] < 0.0) | (actions[..., 20:24] > 1.0))
        ),
        "base_values_outside_configured_limits": int(
            np.count_nonzero(np.abs(actions[..., 24:26]) > 0.10)
            + np.count_nonzero(np.abs(actions[..., 26]) > 0.20)
        ),
        "latency_seconds": {
            "total": _summarize(
                [prediction.timings.total_seconds for prediction in predictions]
            ),
            "gpu_inference": _summarize(
                [
                    prediction.timings.gpu_inference_seconds
                    for prediction in predictions
                ]
            ),
            "preprocessing": _summarize(
                [
                    prediction.timings.preprocessing_seconds
                    for prediction in predictions
                ]
            ),
            "postprocessing": _summarize(
                [
                    prediction.timings.postprocessing_seconds
                    for prediction in predictions
                ]
            ),
        },
        "peak_gpu_allocated_bytes": max(
            prediction.peak_gpu_allocated_bytes for prediction in predictions
        ),
        "peak_gpu_reserved_bytes": max(
            prediction.peak_gpu_reserved_bytes for prediction in predictions
        ),
    }
    report["action_contract_passed"] = bool(
        report["action_finite"] and not body_violations
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["action_contract_passed"] else 3


def _load_observation(
    project_root: Path, observation_path: str | None, *, synthetic: bool
) -> tuple[np.ndarray, np.ndarray]:
    if synthetic:
        return (
            np.zeros(STATE_DIMENSION, dtype=np.float32),
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
    assert observation_path is not None
    path = Path(observation_path).expanduser().resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("observation NPZ must be inside the project") from exc
    with np.load(path, allow_pickle=False) as data:
        state = np.asarray(data["state"], dtype=np.float32)
        rgb = np.asarray(data["rgb"])
    if state.shape != (STATE_DIMENSION,) or not np.all(np.isfinite(state)):
        raise ValueError("observation state must contain 27 finite values")
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError("observation rgb must be uint8 with shape [480,640,3]")
    return state.copy(), np.ascontiguousarray(rgb)


def _snapshot(state: np.ndarray, rgb: np.ndarray) -> ObservationSnapshot:
    now_wall = time.time_ns()
    return ObservationSnapshot(
        state=state.copy(),
        rgb=rgb.copy(),
        task=TASK,
        state_stamp_ns=now_wall,
        camera_stamp_ns=now_wall,
        receive_stamp_ns=now_wall,
        created_stamp_ns=now_wall,
        created_monotonic_ns=time.monotonic_ns(),
        state_age_seconds=0.0,
        camera_capture_age_seconds=0.0,
        camera_receive_age_seconds=0.0,
        synchronization_skew_seconds=0.0,
    )


def _body_limit_violations(actions, joint_limits) -> dict[str, int]:
    violations = {}
    for index, name in enumerate(BODY_JOINT_NAMES):
        values = actions[..., index]
        limit = joint_limits[name]
        count = int(np.count_nonzero((values < limit.lower) | (values > limit.upper)))
        if count:
            violations[name] = count
    return violations


def _summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
