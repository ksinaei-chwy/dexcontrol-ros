"""Pinned, project-local GR00T N1.7 artifact acquisition and validation."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifact import (
    ARTIFACT_MANIFEST,
    COMMIT_PATTERN,
    ArtifactValidationError,
    ResolvedArtifact,
    configure_project_caches,
    require_project_local,
)
from .contracts import (
    ACTION_DIMENSION,
    DATASET_REPO_ID,
    DATASET_REVISION,
    HEAD_CAMERA_FEATURE,
    STATE_DIMENSION,
    TASK,
)
from .groot_contracts import (
    ACTION_CHUNK_SIZE,
    BASE_MODEL_REPO_ID,
    BASE_MODEL_REVISION,
    CHECKPOINT_TAG,
    COSMOS_PROCESSOR_REPO_ID,
    COSMOS_PROCESSOR_REVISION,
    EMBODIMENT_TAG,
    MODEL_MAX_STATE_ACTION_DIMENSION,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_WEIGHT_SHA256,
    MODEL_WEIGHT_SIZE,
    POLICY_TYPE,
)


GROOT_MODEL_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
COSMOS_PROCESSOR_REQUIRED_FILES = (
    "chat_template.json",
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
COSMOS_PROCESSOR_DOWNLOAD_PATTERNS = COSMOS_PROCESSOR_REQUIRED_FILES + (
    "README.md",
    "LICENSE*",
    "NOTICE*",
)


@dataclass(frozen=True)
class GrootArtifactBundle:
    """The three independently pinned local sources required for inference."""

    model: ResolvedArtifact
    base_model: ResolvedArtifact
    cosmos_processor: ResolvedArtifact


def resolve_groot_artifacts(
    *,
    project_root: str | Path,
    model_local_path: str | Path,
    base_model_local_path: str | Path,
    cosmos_processor_local_path: str | Path,
    model_revision: str = MODEL_REVISION,
    base_model_revision: str = BASE_MODEL_REVISION,
    cosmos_processor_revision: str = COSMOS_PROCESSOR_REVISION,
    checkpoint_tag: str = CHECKPOINT_TAG,
    allow_download: bool = False,
    local_files_only: bool = True,
) -> GrootArtifactBundle:
    """Resolve and verify the complete immutable GR00T inference bundle."""

    exact_values = (
        ("model", model_revision, MODEL_REVISION),
        ("base model", base_model_revision, BASE_MODEL_REVISION),
        ("Cosmos processor", cosmos_processor_revision, COSMOS_PROCESSOR_REVISION),
    )
    for label, configured, expected in exact_values:
        if configured != expected:
            raise ArtifactValidationError(
                f"{label} revision {configured!r} differs from the deployment pin {expected}"
            )
    if checkpoint_tag != CHECKPOINT_TAG:
        raise ArtifactValidationError(
            f"checkpoint tag {checkpoint_tag!r} differs from {CHECKPOINT_TAG!r}"
        )

    root = Path(project_root).expanduser().resolve()
    configure_project_caches(root)
    model = _resolve_snapshot(
        project_root=root,
        local_path=model_local_path,
        role="groot_policy",
        repo_id=MODEL_REPO_ID,
        revision=model_revision,
        checkpoint_tag=checkpoint_tag,
        allow_download=allow_download,
        local_files_only=local_files_only,
        allow_patterns=None,
        validator=validate_groot_model_artifact,
    )
    base_model = _resolve_snapshot(
        project_root=root,
        local_path=base_model_local_path,
        role="groot_base_model",
        repo_id=BASE_MODEL_REPO_ID,
        revision=base_model_revision,
        checkpoint_tag=None,
        allow_download=allow_download,
        local_files_only=local_files_only,
        allow_patterns=None,
        validator=validate_groot_base_model_artifact,
    )
    cosmos_processor = _resolve_snapshot(
        project_root=root,
        local_path=cosmos_processor_local_path,
        role="cosmos_processor",
        repo_id=COSMOS_PROCESSOR_REPO_ID,
        revision=cosmos_processor_revision,
        checkpoint_tag=None,
        allow_download=allow_download,
        local_files_only=local_files_only,
        allow_patterns=COSMOS_PROCESSOR_DOWNLOAD_PATTERNS,
        validator=validate_cosmos_processor_artifact,
    )
    return GrootArtifactBundle(model, base_model, cosmos_processor)


def validate_groot_model_artifact(
    path: str | Path,
    *,
    expected_weight_size: int = MODEL_WEIGHT_SIZE,
    expected_weight_sha256: str = MODEL_WEIGHT_SHA256,
) -> dict[str, Any]:
    """Validate the saved LeRobot policy, processors, stats, size, and SHA-256."""

    root = Path(path).expanduser().resolve()
    _require_directory_and_files(root, GROOT_MODEL_REQUIRED_FILES, "GR00T model")
    weight = root / "model.safetensors"
    actual_size = weight.stat().st_size
    if actual_size != int(expected_weight_size):
        raise ArtifactValidationError(
            f"model.safetensors size is {actual_size}, expected {expected_weight_size}"
        )
    actual_sha256 = _sha256(weight)
    if actual_sha256 != expected_weight_sha256:
        raise ArtifactValidationError(
            "model.safetensors SHA-256 mismatch: "
            f"{actual_sha256}, expected {expected_weight_sha256}"
        )

    parsed = {
        name: _read_json(root / name)
        for name in (
            "config.json",
            "train_config.json",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
    }
    _validate_groot_policy_config(parsed["config.json"], parsed["train_config.json"])
    _validate_groot_processor_configs(
        parsed["policy_preprocessor.json"], parsed["policy_postprocessor.json"]
    )
    _validate_groot_statistics(root)
    parsed["model_weight_size"] = actual_size
    parsed["model_weight_sha256"] = actual_sha256
    return parsed


def validate_groot_base_model_artifact(path: str | Path) -> dict[str, Any]:
    """Validate the raw pinned N1.7 base used by LeRobot model construction."""

    root = Path(path).expanduser().resolve()
    _require_directory_and_files(
        root, ("config.json", "embodiment_id.json"), "GR00T base model"
    )
    weights = sorted(root.glob("model*.safetensors"))
    if not weights or any(item.stat().st_size <= 0 for item in weights):
        raise ArtifactValidationError(
            "GR00T base model has no complete model*.safetensors weights"
        )
    config = _read_json(root / "config.json")
    _reject_remote_code(config, "base config")
    model_type = str(config.get("model_type", "")).lower().replace("-", "")
    architectures = [str(value).lower() for value in config.get("architectures", [])]
    if "gr00tn1d7" not in model_type and not any(
        "gr00tn1" in value and "7" in value for value in architectures
    ):
        raise ArtifactValidationError("base config is not GR00T N1.7")
    if config.get("model_name") != COSMOS_PROCESSOR_REPO_ID:
        raise ArtifactValidationError(
            "base config does not identify the pinned Cosmos-Reason2-2B backbone"
        )
    weight_names = {item.name for item in weights}
    index_path = root / "model.safetensors.index.json"
    if len(weights) > 1 and not index_path.is_file():
        raise ArtifactValidationError(
            "sharded GR00T base model is missing model.safetensors.index.json"
        )
    if len(weights) == 1 and not index_path.is_file() and weight_names != {
        "model.safetensors"
    }:
        raise ArtifactValidationError(
            "GR00T base model contains an orphaned weight shard without an index"
        )
    if index_path.is_file():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ArtifactValidationError("GR00T base weight index has no weight_map")
        if set(weight_map.values()) != weight_names:
            raise ArtifactValidationError(
                "GR00T base weight index does not exactly reference the local shards"
            )
    return {"config.json": config, "weight_files": sorted(weight_names)}


def validate_cosmos_processor_artifact(path: str | Path) -> dict[str, Any]:
    """Validate the processor-only Cosmos snapshot used for offline tokenization."""

    root = Path(path).expanduser().resolve()
    _require_directory_and_files(
        root, COSMOS_PROCESSOR_REQUIRED_FILES, "Cosmos processor"
    )
    config = _read_json(root / "config.json")
    _reject_remote_code(config, "Cosmos config")
    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(value).lower() for value in config.get("architectures", [])]
    if "qwen3" not in model_type and not any("qwen3" in value for value in architectures):
        raise ArtifactValidationError(
            "Cosmos processor config is not Qwen3-VL compatible"
        )
    for name in (
        "chat_template.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
    ):
        value = _read_json(root / name)
        _reject_remote_code(value, name)
    unexpected_weights = sorted(
        item.name for item in root.glob("model*.safetensors") if item.is_file()
    )
    if unexpected_weights or (root / "model.safetensors.index.json").exists():
        raise ArtifactValidationError(
            "Cosmos processor directory unexpectedly contains model weights; "
            "only pinned processor/tokenizer assets are used by this runtime"
        )
    return {"config.json": config}


def validate_snapshot_manifest(
    path: str | Path,
    *,
    role: str,
    repo_id: str,
    revision: str,
    checkpoint_tag: str | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Verify immutable identity plus the recorded local file inventory."""

    root = Path(path).expanduser().resolve()
    manifest_path = root / ARTIFACT_MANIFEST
    if not manifest_path.is_file():
        raise ArtifactValidationError(f"artifact manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 2:
        raise ArtifactValidationError("artifact manifest schema_version must be 2")
    expected = {
        "artifact_role": role,
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_commit": revision,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ArtifactValidationError(
                f"artifact manifest {key}={manifest.get(key)!r}, expected {value!r}"
            )
    if checkpoint_tag is not None and manifest.get("checkpoint_tag") != checkpoint_tag:
        raise ArtifactValidationError(
            "artifact manifest checkpoint tag differs from the deployment contract"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ArtifactValidationError("artifact manifest has no file inventory")
    if verify_files:
        for relative, expected_details in files.items():
            if not isinstance(relative, str) or not isinstance(expected_details, dict):
                raise ArtifactValidationError("artifact manifest file inventory is invalid")
            item = root / relative
            try:
                item.resolve().relative_to(root)
            except ValueError as exc:
                raise ArtifactValidationError(
                    f"artifact manifest path escapes its root: {relative}"
                ) from exc

        # Hash each file once. Model startup deliberately verifies the entire
        # immutable snapshot, but the multi-gigabyte policy/base files must not
        # be read a second time merely to compare their recorded sizes.
        current_inventory = _file_inventory(root)
        current_files = set(current_inventory)
        manifested_files = set(files)
        if current_files != manifested_files:
            added = sorted(current_files - manifested_files)
            missing = sorted(manifested_files - current_files)
            raise ArtifactValidationError(
                "artifact file inventory changed; "
                f"added={added}, missing={missing}"
            )
        for relative, expected_details in files.items():
            actual_details = current_inventory[relative]
            size = actual_details["size"]
            if size != expected_details.get("size"):
                raise ArtifactValidationError(
                    f"artifact file size changed for {relative}: {size}"
                )
            expected_sha = expected_details.get("sha256")
            if (
                not isinstance(expected_sha, str)
                or actual_details["sha256"] != expected_sha
            ):
                raise ArtifactValidationError(
                    f"artifact file SHA-256 changed for {relative}"
                )
    return manifest


def _resolve_snapshot(
    *,
    project_root: Path,
    local_path: str | Path,
    role: str,
    repo_id: str,
    revision: str,
    checkpoint_tag: str | None,
    allow_download: bool,
    local_files_only: bool,
    allow_patterns: tuple[str, ...] | None,
    validator: Callable[[str | Path], dict[str, Any]],
) -> ResolvedArtifact:
    if not COMMIT_PATTERN.fullmatch(revision):
        raise ArtifactValidationError(
            f"{role} revision must be a full 40-character commit SHA"
        )
    target = require_project_local(local_path, project_root)
    if target.is_dir() and (target / ARTIFACT_MANIFEST).is_file():
        validator(target)
        validate_snapshot_manifest(
            target,
            role=role,
            repo_id=repo_id,
            revision=revision,
            checkpoint_tag=checkpoint_tag,
        )
        return ResolvedArtifact(target, repo_id, revision, revision, checkpoint_tag)

    if not allow_download:
        raise ArtifactValidationError(
            f"verified local {role} is unavailable at {target}; download is disabled"
        )
    if local_files_only:
        raise ArtifactValidationError(
            f"verified local {role} is unavailable and local_files_only=true"
        )
    if os.environ.get("HF_TOKEN", "").strip() == "":
        raise ArtifactValidationError(
            "HF_TOKEN is not set; use an environment-only read token accepted by "
            "the private/gated repositories"
        )
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise ArtifactValidationError(
            "huggingface_hub is required to download GR00T artifacts"
        ) from exc

    try:
        info = HfApi().model_info(repo_id, revision=revision, token=True)
    except Exception as exc:  # noqa: BLE001 - Hub/network authentication boundary
        raise ArtifactValidationError(
            f"cannot resolve gated Hub artifact {repo_id}@{revision}: {exc}"
        ) from exc
    resolved = str(info.sha)
    if resolved != revision:
        raise ArtifactValidationError(
            f"Hub resolved {repo_id}@{revision} to unexpected commit {resolved}"
        )
    target.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "local_dir": target,
        "token": True,
        "local_files_only": False,
    }
    if allow_patterns is not None:
        kwargs["allow_patterns"] = allow_patterns
    try:
        snapshot_download(**kwargs)
    except Exception as exc:  # noqa: BLE001 - Hub/network download boundary
        raise ArtifactValidationError(
            f"cannot download gated Hub artifact {repo_id}@{revision}: {exc}"
        ) from exc
    validator(target)
    _write_snapshot_manifest(
        target,
        role=role,
        repo_id=repo_id,
        revision=revision,
        checkpoint_tag=checkpoint_tag,
    )
    validate_snapshot_manifest(
        target,
        role=role,
        repo_id=repo_id,
        revision=revision,
        checkpoint_tag=checkpoint_tag,
    )
    return ResolvedArtifact(target, repo_id, revision, revision, checkpoint_tag)


def _validate_groot_policy_config(config: dict[str, Any], train: dict[str, Any]) -> None:
    _reject_remote_code(config, "fine-tuned config")
    expected = {
        "type": POLICY_TYPE,
        "chunk_size": ACTION_CHUNK_SIZE,
        "n_action_steps": ACTION_CHUNK_SIZE,
        "max_state_dim": MODEL_MAX_STATE_ACTION_DIMENSION,
        "max_action_dim": MODEL_MAX_STATE_ACTION_DIMENSION,
        "embodiment_tag": EMBODIMENT_TAG,
        "use_relative_actions": False,
        "use_bf16": True,
        "model_params_fp32": False,
        "use_peft": False,
        "tune_llm": False,
        "tune_visual": False,
        "tune_projector": True,
        "tune_diffusion_model": True,
        "tune_vlln": True,
        "tune_top_llm_layers": 0,
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ArtifactValidationError(
                f"config.json {key}={config.get(key)!r}, expected {expected_value!r}"
            )
    if config.get("base_model_path") != BASE_MODEL_REPO_ID:
        raise ArtifactValidationError(
            "config.json base_model_path differs from the training base model"
        )
    if config.get("action_decode_transform") not in {None, "none", ""}:
        raise ArtifactValidationError("GR00T action decode transform must be disabled")
    if config.get("lora_rank", 0) != 0 or train.get("peft") not in (None, {}):
        raise ArtifactValidationError("GR00T artifact unexpectedly enables LoRA/PEFT")
    training_expected = {"seed": 1000, "batch_size": 8, "steps": 170_000}
    for key, expected_value in training_expected.items():
        if train.get(key) != expected_value:
            raise ArtifactValidationError(
                f"train_config.json {key}={train.get(key)!r}, expected {expected_value!r}"
            )
    dataset = train.get("dataset")
    if not isinstance(dataset, dict):
        raise ArtifactValidationError("train_config.json has no dataset contract")
    if dataset.get("repo_id") != DATASET_REPO_ID:
        raise ArtifactValidationError(
            "train_config.json dataset repo differs from the recorder dataset"
        )
    if dataset.get("revision") != DATASET_REVISION:
        raise ArtifactValidationError(
            "train_config.json dataset revision differs from the immutable training data"
        )
    train_policy = train.get("policy")
    if not isinstance(train_policy, dict):
        raise ArtifactValidationError("train_config.json has no saved policy config")
    for key in expected:
        if train_policy.get(key) != config.get(key):
            raise ArtifactValidationError(
                f"train_config.json policy.{key} differs from config.json"
            )

    inputs = config.get("input_features", {})
    state_shape = inputs.get("observation.state", {}).get("shape")
    if list(state_shape or []) != [STATE_DIMENSION]:
        raise ArtifactValidationError(
            f"GR00T physical state feature is {state_shape}, expected [{STATE_DIMENSION}]"
        )
    visual_keys = {
        key
        for key, value in inputs.items()
        if isinstance(value, dict) and str(value.get("type", "")).upper() == "VISUAL"
    }
    if visual_keys != {HEAD_CAMERA_FEATURE}:
        raise ArtifactValidationError(
            "GR00T visual feature set must contain only the recorder head camera; "
            f"found {sorted(visual_keys)}"
        )
    action_shape = config.get("output_features", {}).get("action", {}).get("shape")
    if list(action_shape or []) != [ACTION_DIMENSION]:
        raise ArtifactValidationError(
            f"GR00T physical action feature is {action_shape}, expected [{ACTION_DIMENSION}]"
        )


def _validate_groot_processor_configs(preprocessor: Any, postprocessor: Any) -> None:
    pre_text = json.dumps(preprocessor, sort_keys=True)
    post_text = json.dumps(postprocessor, sort_keys=True)
    if "groot_n1_7_action_decode_v1" in post_text:
        raise ArtifactValidationError(
            "saved GR00T postprocessor unexpectedly uses a raw-checkpoint decoder"
        )
    combined = pre_text + post_text
    if "relative_actions_processor" in combined or "absolute_actions_processor" in combined:
        raise ArtifactValidationError("saved GR00T processors unexpectedly use relative actions")
    if TASK in pre_text:
        raise ArtifactValidationError(
            "task text must remain a runtime input, not serialized processor state"
        )
    pre_steps = _processor_steps(preprocessor, "preprocessor")
    post_steps = _processor_steps(postprocessor, "postprocessor")
    pre_names = [step["registry_name"] for step in pre_steps]
    post_names = [step["registry_name"] for step in post_steps]
    if pre_names.count("groot_n1_7_pack_inputs_v1") != 1:
        raise ArtifactValidationError("saved GR00T preprocessor needs one pack step")
    if pre_names.count("groot_n1_7_vlm_encode_v1") != 1:
        raise ArtifactValidationError("saved GR00T preprocessor needs one VLM step")
    if pre_names.index("groot_n1_7_pack_inputs_v1") > pre_names.index(
        "groot_n1_7_vlm_encode_v1"
    ):
        raise ArtifactValidationError("saved GR00T processor step order is invalid")
    if post_names.count("groot_action_unpack_unnormalize_v2") != 1:
        raise ArtifactValidationError(
            "saved GR00T postprocessor lacks physical action unnormalization"
        )

    pack = pre_steps[pre_names.index("groot_n1_7_pack_inputs_v1")]
    pack_config = pack.get("config", {})
    pack_expected = {
        "action_horizon": ACTION_CHUNK_SIZE,
        "valid_action_horizon": ACTION_CHUNK_SIZE,
        "max_state_dim": MODEL_MAX_STATE_ACTION_DIMENSION,
        "max_action_dim": MODEL_MAX_STATE_ACTION_DIMENSION,
        "embodiment_tag": EMBODIMENT_TAG,
        "normalize_min_max": True,
    }
    for key, expected_value in pack_expected.items():
        if pack_config.get(key) != expected_value:
            raise ArtifactValidationError(
                f"saved GR00T pack step {key}={pack_config.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    vlm = pre_steps[pre_names.index("groot_n1_7_vlm_encode_v1")]
    if vlm.get("config", {}).get("model_name") != COSMOS_PROCESSOR_REPO_ID:
        raise ArtifactValidationError(
            "saved GR00T VLM processor differs from Cosmos-Reason2-2B"
        )
    unpack = post_steps[post_names.index("groot_action_unpack_unnormalize_v2")]
    unpack_config = unpack.get("config", {})
    unpack_expected = {
        "env_action_dim": ACTION_DIMENSION,
        "normalize_min_max": True,
        "libero_gripper_action": False,
    }
    for key, expected_value in unpack_expected.items():
        if unpack_config.get(key) != expected_value:
            raise ArtifactValidationError(
                f"saved GR00T unpack step {key}={unpack_config.get(key)!r}, "
                f"expected {expected_value!r}"
            )


def _validate_groot_statistics(root: Path) -> None:
    pre_config = _read_json(root / "policy_preprocessor.json")
    post_config = _read_json(root / "policy_postprocessor.json")
    pre_files = _processor_state_files(root, pre_config, "preprocessor")
    post_files = _processor_state_files(root, post_config, "postprocessor")
    if not pre_files or not post_files:
        raise ArtifactValidationError(
            "GR00T artifact is missing serialized preprocessor/postprocessor state"
        )
    pre_shapes = _merge_safetensors_shapes(pre_files)
    post_shapes = _merge_safetensors_shapes(post_files)
    expected_pre = {
        "observation.state.min": [STATE_DIMENSION],
        "observation.state.max": [STATE_DIMENSION],
        "action.min": [ACTION_DIMENSION],
        "action.max": [ACTION_DIMENSION],
    }
    for key, shape in expected_pre.items():
        if pre_shapes.get(key) != shape:
            raise ArtifactValidationError(
                f"saved GR00T preprocessor statistic {key} has shape "
                f"{pre_shapes.get(key)}, expected {shape}"
            )
    for key in ("action.min", "action.max"):
        if post_shapes.get(key) != [ACTION_DIMENSION]:
            raise ArtifactValidationError(
                f"saved GR00T postprocessor statistic {key} has shape "
                f"{post_shapes.get(key)}, expected [{ACTION_DIMENSION}]"
            )


def _merge_safetensors_shapes(paths: list[Path]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for path in paths:
        try:
            with path.open("rb") as stream:
                raw_length = stream.read(8)
                if len(raw_length) != 8:
                    raise ValueError("missing header length")
                header_length = struct.unpack("<Q", raw_length)[0]
                if header_length <= 0 or header_length > 64 * 1024 * 1024:
                    raise ValueError(f"invalid header length {header_length}")
                header = json.loads(stream.read(header_length).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ArtifactValidationError(
                f"cannot parse safetensors header {path}: {exc}"
            ) from exc
        for key, value in header.items():
            if key != "__metadata__" and isinstance(value, dict):
                result[key] = list(value.get("shape", []))
    return result


def _write_snapshot_manifest(
    root: Path,
    *,
    role: str,
    repo_id: str,
    revision: str,
    checkpoint_tag: str | None,
) -> None:
    manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_role": role,
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_commit": revision,
        "checkpoint_tag": checkpoint_tag,
        "files": _file_inventory(root),
    }
    temporary = root / f".{ARTIFACT_MANIFEST}.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(root / ARTIFACT_MANIFEST)


def _file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = item.relative_to(root)
        if relative in {
            Path(ARTIFACT_MANIFEST),
            Path(f".{ARTIFACT_MANIFEST}.tmp"),
        } or relative.parts[:2] == (
            ".cache",
            "huggingface",
        ):
            continue
        try:
            item.resolve().relative_to(root)
        except ValueError as exc:
            raise ArtifactValidationError(
                f"artifact file escapes its root: {relative}"
            ) from exc
        inventory[str(relative)] = {
            "size": item.stat().st_size,
            "sha256": _sha256(item),
        }
    return inventory


def _require_directory_and_files(
    root: Path, required: tuple[str, ...], label: str
) -> None:
    if not root.is_dir():
        raise ArtifactValidationError(f"{label} directory does not exist: {root}")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ArtifactValidationError(
            f"{label} is incomplete; missing: " + ", ".join(missing)
        )
    empty = [name for name in required if (root / name).stat().st_size <= 0]
    if empty:
        raise ArtifactValidationError(f"{label} contains empty files: " + ", ".join(empty))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"cannot parse {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{path.name} must contain a JSON object")
    return value


def _processor_steps(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
        raise ArtifactValidationError(f"saved GR00T {label} has no steps")
    steps = value["steps"]
    if not all(
        isinstance(step, dict) and isinstance(step.get("registry_name"), str)
        for step in steps
    ):
        raise ArtifactValidationError(f"saved GR00T {label} step is invalid")
    return steps


def _processor_state_files(
    root: Path, config: dict[str, Any], label: str
) -> list[Path]:
    files: list[Path] = []
    for step in _processor_steps(config, label):
        state_file = step.get("state_file")
        if state_file is None:
            continue
        if not isinstance(state_file, str) or Path(state_file).name != state_file:
            raise ArtifactValidationError(
                f"saved GR00T {label} has an invalid state file path"
            )
        path = root / state_file
        if path.suffix != ".safetensors" or not path.is_file():
            raise ArtifactValidationError(
                f"saved GR00T {label} state is missing: {state_file}"
            )
        files.append(path)
    return files


def _reject_remote_code(config: dict[str, Any], label: str) -> None:
    if "auto_map" in config:
        raise ArtifactValidationError(
            f"{label} requests dynamic remote code, which this runtime forbids"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
