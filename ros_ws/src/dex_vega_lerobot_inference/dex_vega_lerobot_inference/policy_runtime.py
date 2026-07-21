"""LeRobot 0.6.0 PI0.5 policy plus serialized processor runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .artifact import ResolvedArtifact, configure_project_caches, validate_model_artifact
from .contracts import (
    ACTION_CHUNK_SIZE,
    ACTION_DIMENSION,
    HEAD_CAMERA_FEATURE,
    MODEL_HEAD_CAMERA_FEATURE,
    ROBOT_TYPE,
    STATE_DIMENSION,
    TASK,
)
from .observation_adapter import ObservationSnapshot


class PolicyRuntimeError(RuntimeError):
    """Raised for version, device, artifact, processor, or model-contract failures."""


@dataclass(frozen=True)
class PolicyTimings:
    preparation_seconds: float
    preprocessing_seconds: float
    gpu_inference_seconds: float
    postprocessing_seconds: float
    total_seconds: float
    observation_to_result_seconds: float


@dataclass(frozen=True)
class PolicyPrediction:
    actions: np.ndarray
    timings: PolicyTimings
    completed_monotonic_ns: int
    peak_gpu_allocated_bytes: int
    peak_gpu_reserved_bytes: int
    cold_start: bool


@dataclass(frozen=True)
class PolicyRuntimeInfo:
    lerobot_version: str
    torch_version: str
    cuda_version: str | None
    device_name: str
    model_path: str
    model_commit: str | None
    checkpoint_tag: str | None
    tokenizer_path: str
    tokenizer_commit: str | None
    load_seconds: float
    policy_type: str = "pi05"
    action_chunk_size: int = ACTION_CHUNK_SIZE
    action_dimension: int = ACTION_DIMENSION
    base_model_path: str = ""
    base_model_commit: str | None = None
    processor_path: str = ""
    processor_commit: str | None = None


class Pi05PolicyRuntime:
    """Loads only local artifacts and always invokes saved pre/postprocessors."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        model: ResolvedArtifact,
        tokenizer: ResolvedArtifact,
        device: str = "cuda",
        require_cuda: bool = True,
        require_bfloat16: bool = True,
    ) -> None:
        configure_project_caches(project_root)
        validate_model_artifact(model.local_path)
        self._model_artifact = model
        self._tokenizer_artifact = tokenizer
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
        lerobot_version = _package_version("lerobot")
        if lerobot_version != "0.6.0":
            raise PolicyRuntimeError(
                f"LeRobot 0.6.0 is required by the serialized processors; found {lerobot_version}"
            )
        try:
            import torch
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies import get_policy_class, make_pre_post_processors
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise PolicyRuntimeError(f"PI0.5 runtime dependency is unavailable: {exc}") from exc

        self._torch = torch
        self._device = torch.device(self._device_name)
        if self._require_cuda:
            if self._device.type != "cuda":
                raise PolicyRuntimeError("production PI0.5 runtime requires device=cuda")
            if not torch.cuda.is_available():
                raise PolicyRuntimeError(
                    "CUDA is unavailable; do not fall back to CPU for an armed runtime"
                )
            if torch.version.cuda is None:
                raise PolicyRuntimeError(
                    "installed Torch is CPU-only; use an NVIDIA Jetson-compatible PyTorch build"
                )
            if self._require_bfloat16 and not torch.cuda.is_bf16_supported():
                raise PolicyRuntimeError("CUDA device does not report bfloat16 support")

        model_path = str(self._model_artifact.local_path)
        tokenizer_path = str(self._tokenizer_artifact.local_path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
            )
            policy_config = PreTrainedConfig.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self._validate_config(policy_config)
            policy_config.device = self._device_name
            if hasattr(policy_config, "compile_model"):
                policy_config.compile_model = False
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
                    "tokenizer_processor": {"tokenizer": tokenizer},
                    "device_processor": {"device": self._device_name},
                },
            )
        except Exception as exc:  # noqa: BLE001 - serialized third-party boundary
            raise PolicyRuntimeError(
                f"failed to load PI0.5 artifact and processors: {exc}"
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
            model_commit=self._model_artifact.resolved_commit,
            checkpoint_tag=self._model_artifact.checkpoint_tag,
            tokenizer_path=tokenizer_path,
            tokenizer_commit=self._tokenizer_artifact.resolved_commit,
            load_seconds=time.perf_counter() - load_start,
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
        completed_monotonic_ns = time.monotonic_ns()
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)

        try:
            from lerobot.policies.utils import prepare_observation_for_inference

            prepare_start = time.perf_counter()
            raw = observation.as_policy_observation()
            prepared = prepare_observation_for_inference(
                raw,
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
            raise PolicyRuntimeError(f"PI0.5 forward pass failed: {exc}") from exc

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
        if getattr(config, "type", None) != "pi05":
            failures.append(f"policy type={getattr(config, 'type', None)!r}")
        if getattr(config, "chunk_size", None) != ACTION_CHUNK_SIZE:
            failures.append(f"chunk_size={getattr(config, 'chunk_size', None)!r}")
        if getattr(config, "n_action_steps", None) != ACTION_CHUNK_SIZE:
            failures.append(f"n_action_steps={getattr(config, 'n_action_steps', None)!r}")
        if getattr(config, "max_state_dim", None) != 32:
            failures.append(f"max_state_dim={getattr(config, 'max_state_dim', None)!r}")
        if getattr(config, "max_action_dim", None) != 32:
            failures.append(f"max_action_dim={getattr(config, 'max_action_dim', None)!r}")
        if getattr(config, "n_obs_steps", None) != 1:
            failures.append(f"n_obs_steps={getattr(config, 'n_obs_steps', None)!r}")
        if getattr(config, "num_inference_steps", None) != 10:
            failures.append(
                f"num_inference_steps={getattr(config, 'num_inference_steps', None)!r}"
            )
        if tuple(getattr(config, "image_resolution", ())) != (224, 224):
            failures.append(
                f"image_resolution={getattr(config, 'image_resolution', None)!r}"
            )
        if getattr(config, "dtype", None) != "bfloat16":
            failures.append(f"dtype={getattr(config, 'dtype', None)!r}")
        if getattr(config, "use_relative_actions", None) is not False:
            failures.append("relative actions are not disabled")
        output = getattr(config, "output_features", {}).get("action")
        if output is None or tuple(output.shape) != (ACTION_DIMENSION,):
            failures.append("physical action feature is not 27-D")
        state = getattr(config, "input_features", {}).get("observation.state")
        if state is None or tuple(state.shape) != (32,):
            failures.append("model-facing state feature is not 32-D")
        image_features = set(getattr(config, "image_features", {}).keys())
        if MODEL_HEAD_CAMERA_FEATURE not in image_features:
            failures.append(
                f"model image features do not contain renamed head key {MODEL_HEAD_CAMERA_FEATURE}"
            )
        if HEAD_CAMERA_FEATURE in image_features:
            failures.append("model config unexpectedly consumes unrenamed head key")
        if failures:
            message = "saved PI0.5 config contract mismatch: " + "; ".join(
                failures
            )
            raise PolicyRuntimeError(message)


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise PolicyRuntimeError(f"required package is not installed: {name}") from exc
