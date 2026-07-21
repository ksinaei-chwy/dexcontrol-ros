"""Offline LeRobot 0.6.0 GR00T N1.7 runtime for the Dexmate policy."""

from __future__ import annotations

import os
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .artifact import configure_project_caches
from .contracts import (
    ACTION_DIMENSION,
    HEAD_CAMERA_FEATURE,
    ROBOT_TYPE,
    STATE_DIMENSION,
    TASK,
)
from .groot_artifact import GrootArtifactBundle
from .groot_contracts import (
    ACTION_CHUNK_SIZE,
    BASE_MODEL_REPO_ID,
    EMBODIMENT_TAG,
    MODEL_MAX_STATE_ACTION_DIMENSION,
    POLICY_TYPE,
)
from .observation_adapter import ObservationSnapshot
from .policy_runtime import (
    PolicyPrediction,
    PolicyRuntimeError,
    PolicyRuntimeInfo,
    PolicyTimings,
)


class GrootPolicyRuntime:
    """Load the exact local fine-tune, base, and serialized processor bundle."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        artifacts: GrootArtifactBundle,
        device: str = "cuda",
        require_cuda: bool = True,
        require_bfloat16: bool = True,
    ) -> None:
        configure_project_caches(project_root)
        self._artifacts = artifacts
        self._device_name = str(device)
        self._require_cuda = bool(require_cuda)
        self._require_bfloat16 = bool(require_bfloat16)
        self._torch: Any = None
        self._policy: Any = None
        self._preprocessor: Any = None
        self._postprocessor: Any = None
        self._device: Any = None
        self._cold = True
        self.info = self._load()

    def _load(self) -> PolicyRuntimeInfo:
        load_start = time.perf_counter()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        lerobot_version = _package_version("lerobot")
        if lerobot_version != "0.6.0":
            raise PolicyRuntimeError(
                "LeRobot 0.6.0 is required by the serialized GR00T processors; "
                f"found {lerobot_version}"
            )
        try:
            import torch
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies import get_policy_class, make_pre_post_processors
        except ImportError as exc:
            raise PolicyRuntimeError(
                f"GR00T N1.7 runtime dependency is unavailable: {exc}"
            ) from exc

        for package in ("diffusers", "dm-tree", "peft", "timm", "transformers"):
            _package_version(package)

        self._torch = torch
        self._device = torch.device(self._device_name)
        if self._require_cuda:
            if self._device.type != "cuda":
                raise PolicyRuntimeError("production GR00T runtime requires device=cuda")
            if not torch.cuda.is_available():
                raise PolicyRuntimeError(
                    "CUDA is unavailable; do not fall back to CPU for an armed runtime"
                )
            if torch.version.cuda is None:
                raise PolicyRuntimeError(
                    "installed Torch is CPU-only; use the pinned NVIDIA Jetson runtime"
                )
            if self._require_bfloat16 and not torch.cuda.is_bf16_supported():
                raise PolicyRuntimeError("CUDA device does not report bfloat16 support")

        model_path = str(self._artifacts.model.local_path)
        base_model_path = str(self._artifacts.base_model.local_path)
        processor_path = str(self._artifacts.cosmos_processor.local_path)
        try:
            policy_config = PreTrainedConfig.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self._validate_config(policy_config)
            # Preserve the serialized fine-tune config while replacing only
            # remote sources with their independently pinned local snapshots.
            policy_config.base_model_path = base_model_path
            policy_config.device = self._device_name
            policy_config.use_flash_attention = False
            if hasattr(policy_config, "gradient_checkpointing"):
                policy_config.gradient_checkpointing = False

            policy_class = get_policy_class(policy_config.type)
            self._policy = policy_class.from_pretrained(
                model_path,
                config=policy_config,
                local_files_only=True,
                strict=True,
            )
            self._policy = self._policy.to(self._device)
            self._policy.eval()
            self._preprocessor, self._postprocessor = make_pre_post_processors(
                policy_cfg=policy_config,
                pretrained_path=model_path,
                preprocessor_overrides={
                    "groot_n1_7_vlm_encode_v1": {
                        "model_name": processor_path,
                        "device": self._device_name,
                    },
                    "device_processor": {"device": self._device_name},
                },
                postprocessor_overrides={
                    "device_processor": {"device": "cpu"},
                },
            )
        except Exception as exc:  # noqa: BLE001 - serialized third-party boundary
            raise PolicyRuntimeError(
                f"failed to load GR00T N1.7 artifact and processors: {exc}"
            ) from exc

        return PolicyRuntimeInfo(
            lerobot_version=lerobot_version,
            torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,
            device_name=(
                str(torch.cuda.get_device_name(self._device))
                if self._device.type == "cuda"
                else self._device_name
            ),
            model_path=model_path,
            model_commit=self._artifacts.model.resolved_commit,
            checkpoint_tag=self._artifacts.model.checkpoint_tag,
            tokenizer_path="",
            tokenizer_commit=None,
            load_seconds=time.perf_counter() - load_start,
            policy_type=POLICY_TYPE,
            action_chunk_size=ACTION_CHUNK_SIZE,
            action_dimension=ACTION_DIMENSION,
            base_model_path=base_model_path,
            base_model_commit=self._artifacts.base_model.resolved_commit,
            processor_path=processor_path,
            processor_commit=self._artifacts.cosmos_processor.resolved_commit,
        )

    def predict(self, observation: ObservationSnapshot) -> PolicyPrediction:
        if observation.task != TASK:
            raise PolicyRuntimeError("runtime task differs from the exact training task")
        if observation.state.shape != (STATE_DIMENSION,):
            raise PolicyRuntimeError("runtime received a non-27-D state")
        if observation.rgb.dtype != np.uint8 or observation.rgb.shape != (480, 640, 3):
            raise PolicyRuntimeError("runtime received an invalid head RGB image")

        torch = self._torch
        total_start = time.perf_counter()
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        try:
            from lerobot.policies.utils import prepare_observation_for_inference

            prepare_start = time.perf_counter()
            prepared = prepare_observation_for_inference(
                observation.as_policy_observation(),
                self._device,
                task=TASK,
                robot_type=ROBOT_TYPE,
            )
            preparation_seconds = time.perf_counter() - prepare_start

            preprocess_start = time.perf_counter()
            processed = self._preprocessor(prepared)
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            preprocessing_seconds = time.perf_counter() - preprocess_start

            inference_start = time.perf_counter()
            with torch.inference_mode():
                normalized_actions = self._policy.predict_action_chunk(processed)
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            gpu_inference_seconds = time.perf_counter() - inference_start

            postprocess_start = time.perf_counter()
            physical_actions = self._postprocessor(normalized_actions)
            if hasattr(physical_actions, "detach"):
                actions = physical_actions.detach().cpu().numpy()
            else:
                actions = np.asarray(physical_actions)
            postprocessing_seconds = time.perf_counter() - postprocess_start
        except Exception as exc:  # noqa: BLE001 - CUDA/model boundary
            raise PolicyRuntimeError(f"GR00T N1.7 forward pass failed: {exc}") from exc

        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (ACTION_CHUNK_SIZE, ACTION_DIMENSION):
            raise PolicyRuntimeError(
                "postprocessor returned shape "
                f"{actions.shape}, expected ({ACTION_CHUNK_SIZE}, {ACTION_DIMENSION})"
            )
        if not np.all(np.isfinite(actions)):
            raise PolicyRuntimeError("postprocessed action chunk contains NaN or Inf")

        total_seconds = time.perf_counter() - total_start
        completed_monotonic_ns = time.monotonic_ns()
        peak_allocated = 0
        peak_reserved = 0
        if self._device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(self._device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self._device))
        cold = self._cold
        self._cold = False
        observation_to_result = max(
            0.0,
            (completed_monotonic_ns - observation.created_monotonic_ns) / 1e9,
        )
        return PolicyPrediction(
            actions=actions,
            timings=PolicyTimings(
                preparation_seconds=preparation_seconds,
                preprocessing_seconds=preprocessing_seconds,
                gpu_inference_seconds=gpu_inference_seconds,
                postprocessing_seconds=postprocessing_seconds,
                total_seconds=total_seconds,
                observation_to_result_seconds=observation_to_result,
            ),
            completed_monotonic_ns=completed_monotonic_ns,
            peak_gpu_allocated_bytes=peak_allocated,
            peak_gpu_reserved_bytes=peak_reserved,
            cold_start=cold,
        )

    def reset(self) -> None:
        if self._policy is not None:
            self._policy.reset()
        if self._preprocessor is not None:
            self._preprocessor.reset()
        if self._postprocessor is not None:
            self._postprocessor.reset()

    @staticmethod
    def _validate_config(config: Any) -> None:
        failures = []
        expected_scalars = {
            "type": POLICY_TYPE,
            "chunk_size": ACTION_CHUNK_SIZE,
            "n_action_steps": ACTION_CHUNK_SIZE,
            "max_state_dim": MODEL_MAX_STATE_ACTION_DIMENSION,
            "max_action_dim": MODEL_MAX_STATE_ACTION_DIMENSION,
            "n_obs_steps": 1,
            "embodiment_tag": EMBODIMENT_TAG,
            "use_relative_actions": False,
            "use_bf16": True,
            "model_params_fp32": False,
            "action_decode_transform": None,
            "use_peft": False,
        }
        for name, expected in expected_scalars.items():
            actual = getattr(config, name, None)
            if actual != expected:
                failures.append(f"{name}={actual!r}")
        if getattr(config, "base_model_path", None) != BASE_MODEL_REPO_ID:
            failures.append(f"base_model_path={getattr(config, 'base_model_path', None)!r}")
        output = getattr(config, "output_features", {}).get("action")
        if output is None or tuple(output.shape) != (ACTION_DIMENSION,):
            failures.append("physical action feature is not 27-D")
        state = getattr(config, "input_features", {}).get("observation.state")
        if state is None or tuple(state.shape) != (STATE_DIMENSION,):
            failures.append("physical state feature is not 27-D")
        image_features = set(getattr(config, "image_features", {}).keys())
        if image_features != {HEAD_CAMERA_FEATURE}:
            failures.append(
                "image feature set is not the single recorder head camera: "
                f"{sorted(image_features)}"
            )
        if failures:
            raise PolicyRuntimeError(
                "saved GR00T N1.7 config contract mismatch: " + "; ".join(failures)
            )


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise PolicyRuntimeError(f"required package is not installed: {name}") from exc
