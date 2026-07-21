"""Pinned Hugging Face acquisition and local PI0.5 artifact validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    ACTION_CHUNK_SIZE,
    ACTION_DIMENSION,
    HEAD_CAMERA_FEATURE,
    MODEL_HEAD_CAMERA_FEATURE,
    MODEL_MAX_STATE_ACTION_DIMENSION,
    STATE_DIMENSION,
    TASK,
    TOKENIZER_REPO_ID,
)


EXPECTED_MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
TOKENIZER_DOWNLOAD_PATTERNS = (
    "added_tokens.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
ARTIFACT_MANIFEST = "dexmate_artifact_manifest.json"
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ArtifactValidationError(RuntimeError):
    """Raised when a model/tokenizer source is incomplete or mutable."""


@dataclass(frozen=True)
class ResolvedArtifact:
    local_path: Path
    repo_id: str | None
    requested_revision: str | None
    resolved_commit: str | None
    checkpoint_tag: str | None


def configure_project_caches(project_root: str | Path) -> dict[str, Path]:
    """Route model/runtime caches inside the workspace."""
    root = Path(project_root).expanduser().resolve()
    cache_root = root / ".cache"
    paths = {
        "XDG_CACHE_HOME": cache_root,
        "HF_HOME": cache_root / "huggingface",
        "HF_HUB_CACHE": cache_root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "TRITON_CACHE_DIR": cache_root / "triton",
    }
    for variable, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    return paths


def require_project_local(path: str | Path, project_root: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactValidationError(
            f"path must remain inside project root {root}: {resolved}"
        ) from exc
    return resolved


def resolve_model_artifact(
    *,
    project_root: str | Path,
    local_path: str | Path | None,
    repo_id: str | None,
    revision: str | None,
    download_directory: str | Path | None,
    checkpoint_tag: str | None,
    allow_download: bool,
    allow_non_commit_revision: bool = False,
    local_files_only: bool = False,
) -> ResolvedArtifact:
    """Resolve either a complete local artifact or a private pinned Hub snapshot."""
    configure_project_caches(project_root)
    if local_path:
        path = require_project_local(local_path, project_root)
        manifest = _load_optional_manifest(path)
        resolved_commit = manifest.get("model_resolved_commit")
        requested_revision = manifest.get("model_requested_revision")
        recorded_tag = manifest.get("checkpoint_tag")
        if (
            not recorded_tag
            and requested_revision
            and not COMMIT_PATTERN.fullmatch(str(requested_revision))
        ):
            # Older download manifests recorded the human-readable checkpoint
            # tag as the requested Hub revision rather than checkpoint_tag.
            recorded_tag = requested_revision
        if revision:
            if not allow_non_commit_revision and not COMMIT_PATTERN.fullmatch(revision):
                raise ArtifactValidationError(
                    "configured local model revision must be a full commit SHA"
                )
            if revision not in {resolved_commit, requested_revision}:
                raise ArtifactValidationError(
                    "configured model revision differs from the local artifact manifest"
                )
        if checkpoint_tag and recorded_tag and checkpoint_tag != recorded_tag:
            raise ArtifactValidationError(
                "configured checkpoint tag differs from the local artifact manifest"
            )
        artifact = ResolvedArtifact(
            local_path=path,
            repo_id=manifest.get("model_repo_id"),
            requested_revision=requested_revision,
            resolved_commit=resolved_commit,
            checkpoint_tag=recorded_tag or checkpoint_tag,
        )
        validate_model_artifact(path)
        return artifact

    if not repo_id:
        raise ArtifactValidationError("set either model_local_path or model_repo_id")
    if not revision:
        raise ArtifactValidationError("Hub model source requires an explicit revision")
    if not allow_non_commit_revision and not COMMIT_PATTERN.fullmatch(revision):
        raise ArtifactValidationError(
            "production Hub revision must be a full 40-character commit SHA; "
            "tags require allow_non_commit_revision=true and are resolved before download"
        )
    if not allow_download and not local_files_only:
        raise ArtifactValidationError("Hub download is disabled; use a local artifact path")
    if not download_directory:
        raise ArtifactValidationError("Hub source requires a project-local download directory")
    target = require_project_local(download_directory, project_root)

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise ArtifactValidationError(
            "huggingface_hub is required to resolve a Hub source"
        ) from exc

    token: bool = True
    resolved_commit = revision
    if not local_files_only:
        info = HfApi().model_info(repo_id, revision=revision, token=token)
        resolved_commit = str(info.sha)
        if not COMMIT_PATTERN.fullmatch(resolved_commit):
            raise ArtifactValidationError(
                f"Hub returned a non-immutable model revision: {resolved_commit}"
            )
    snapshot_download(
        repo_id=repo_id,
        revision=resolved_commit,
        local_dir=target,
        token=token,
        local_files_only=local_files_only,
    )
    validate_model_artifact(target)
    _write_manifest(
        target,
        {
            "model_repo_id": repo_id,
            "model_requested_revision": revision,
            "model_resolved_commit": resolved_commit,
            "checkpoint_tag": checkpoint_tag,
        },
    )
    return ResolvedArtifact(target, repo_id, revision, resolved_commit, checkpoint_tag)


def resolve_tokenizer(
    *,
    project_root: str | Path,
    local_path: str | Path | None,
    revision: str | None,
    download_directory: str | Path | None,
    allow_download: bool,
    allow_non_commit_revision: bool = False,
    local_files_only: bool = False,
) -> ResolvedArtifact:
    """Resolve and pin PaliGemma tokenizer files independently of policy weights."""
    configure_project_caches(project_root)
    if local_path:
        path = require_project_local(local_path, project_root)
        validate_tokenizer_artifact(path)
        manifest = _load_optional_manifest(path)
        resolved_commit = manifest.get("tokenizer_resolved_commit")
        requested_revision = manifest.get("tokenizer_requested_revision")
        if revision:
            if not allow_non_commit_revision and not COMMIT_PATTERN.fullmatch(revision):
                raise ArtifactValidationError(
                    "configured local tokenizer revision must be a full commit SHA"
                )
            if revision not in {resolved_commit, requested_revision}:
                raise ArtifactValidationError(
                    "configured tokenizer revision differs from the local artifact manifest"
                )
        return ResolvedArtifact(
            path,
            TOKENIZER_REPO_ID,
            requested_revision,
            resolved_commit,
            None,
        )
    if not revision:
        raise ArtifactValidationError(
            "tokenizer_revision is required because the tokenizer is not embedded in the policy"
        )
    if not allow_non_commit_revision and not COMMIT_PATTERN.fullmatch(revision):
        raise ArtifactValidationError(
            "production tokenizer revision must be a full 40-character commit SHA"
        )
    if not allow_download and not local_files_only:
        raise ArtifactValidationError("tokenizer download is disabled; use tokenizer_local_path")
    if not download_directory:
        raise ArtifactValidationError("tokenizer source requires a project-local directory")
    target = require_project_local(download_directory, project_root)
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise ArtifactValidationError(
            "huggingface_hub is required to resolve the tokenizer"
        ) from exc
    resolved_commit = revision
    if not local_files_only:
        info = HfApi().model_info(TOKENIZER_REPO_ID, revision=revision, token=True)
        resolved_commit = str(info.sha)
        if not COMMIT_PATTERN.fullmatch(resolved_commit):
            raise ArtifactValidationError("Hub returned a non-immutable tokenizer revision")
    snapshot_download(
        repo_id=TOKENIZER_REPO_ID,
        revision=resolved_commit,
        local_dir=target,
        token=True,
        local_files_only=local_files_only,
        allow_patterns=TOKENIZER_DOWNLOAD_PATTERNS,
    )
    validate_tokenizer_artifact(target)
    _write_manifest(
        target,
        {
            "tokenizer_repo_id": TOKENIZER_REPO_ID,
            "tokenizer_requested_revision": revision,
            "tokenizer_resolved_commit": resolved_commit,
        },
    )
    return ResolvedArtifact(target, TOKENIZER_REPO_ID, revision, resolved_commit, None)


def validate_model_artifact(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactValidationError(f"model artifact directory does not exist: {root}")
    missing = [name for name in EXPECTED_MODEL_FILES if not (root / name).is_file()]
    if missing:
        raise ArtifactValidationError(
            "model artifact is incomplete; missing: " + ", ".join(missing)
        )
    if (root / "model.safetensors").stat().st_size <= 0:
        raise ArtifactValidationError("model.safetensors is empty")
    parsed: dict[str, Any] = {}
    for name in (
        "config.json",
        "train_config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        try:
            parsed[name] = json.loads((root / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError(f"cannot parse {name}: {exc}") from exc
    _validate_policy_config(parsed["config.json"])
    _validate_processor_configs(
        parsed["policy_preprocessor.json"], parsed["policy_postprocessor.json"]
    )
    _validate_statistics_shapes(root)
    return parsed


def validate_tokenizer_artifact(path: str | Path) -> None:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactValidationError(f"tokenizer directory does not exist: {root}")
    if not (root / "tokenizer_config.json").is_file():
        raise ArtifactValidationError(f"tokenizer_config.json is missing from {root}")
    vocabulary_files = (
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
    )
    if not any((root / name).is_file() for name in vocabulary_files):
        raise ArtifactValidationError(f"no tokenizer vocabulary file found in {root}")


def file_inventory(path: str | Path, *, hash_files: bool = False) -> dict[str, dict[str, Any]]:
    root = Path(path).expanduser().resolve()
    inventory: dict[str, dict[str, Any]] = {}
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = str(item.relative_to(root))
        details: dict[str, Any] = {"size": item.stat().st_size}
        if hash_files:
            details["sha256"] = _sha256(item)
        inventory[relative] = details
    return inventory


def _validate_policy_config(config: dict[str, Any]) -> None:
    expected = {
        "type": "pi05",
        "chunk_size": ACTION_CHUNK_SIZE,
        "n_action_steps": ACTION_CHUNK_SIZE,
        "max_state_dim": MODEL_MAX_STATE_ACTION_DIMENSION,
        "max_action_dim": MODEL_MAX_STATE_ACTION_DIMENSION,
        "use_relative_actions": False,
    }
    for key, value in expected.items():
        if key in config and config[key] != value:
            raise ArtifactValidationError(
                f"config.json {key}={config[key]!r}, expected {value!r}"
            )
    input_features = config.get("input_features", {})
    state_feature = input_features.get("observation.state", {})
    state_shape = state_feature.get("shape")
    if state_shape is not None and list(state_shape) != [MODEL_MAX_STATE_ACTION_DIMENSION]:
        raise ArtifactValidationError(
            "config model-facing state shape is "
            f"{state_shape}, expected [{MODEL_MAX_STATE_ACTION_DIMENSION}]"
        )
    output_features = config.get("output_features", {})
    action_shape = output_features.get("action", {}).get("shape")
    if action_shape is not None and list(action_shape) != [ACTION_DIMENSION]:
        raise ArtifactValidationError(
            f"config action shape is {action_shape}, expected [{ACTION_DIMENSION}]"
        )


def _validate_processor_configs(preprocessor: Any, postprocessor: Any) -> None:
    pre_text = json.dumps(preprocessor, sort_keys=True)
    post_text = json.dumps(postprocessor, sort_keys=True)
    required_preprocessor_markers = (
        HEAD_CAMERA_FEATURE,
        MODEL_HEAD_CAMERA_FEATURE,
        TOKENIZER_REPO_ID,
        "normalizer",
        "device",
    )
    missing = [marker for marker in required_preprocessor_markers if marker not in pre_text]
    if missing:
        raise ArtifactValidationError(
            "saved preprocessor contract is incomplete; missing markers: " + ", ".join(missing)
        )
    if "unnormalizer" not in post_text or "device" not in post_text:
        raise ArtifactValidationError("saved postprocessor lacks unnormalizer or CPU device step")
    if TASK in pre_text:
        raise ArtifactValidationError(
            "task text should be a runtime input, not serialized into the preprocessor"
        )


def _validate_statistics_shapes(root: Path) -> None:
    pre_shapes = _safetensors_shapes(
        root / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    )
    post_shapes = _safetensors_shapes(
        root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    expected = {
        "observation.state.q01": [STATE_DIMENSION],
        "observation.state.q99": [STATE_DIMENSION],
        "action.q01": [ACTION_DIMENSION],
        "action.q99": [ACTION_DIMENSION],
    }
    for key, shape in expected.items():
        actual = pre_shapes.get(key)
        if actual != shape:
            raise ArtifactValidationError(
                f"saved preprocessor statistic {key} has shape {actual}, expected {shape}"
            )
    for key in ("action.q01", "action.q99"):
        actual = post_shapes.get(key)
        if actual != expected[key]:
            raise ArtifactValidationError(
                f"saved postprocessor statistic {key} has shape {actual}, "
                f"expected {expected[key]}"
            )


def _safetensors_shapes(path: Path) -> dict[str, list[int]]:
    try:
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise ValueError("missing header length")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > 16 * 1024 * 1024:
                raise ValueError(f"invalid header length {header_length}")
            header = json.loads(stream.read(header_length).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactValidationError(f"cannot parse safetensors header {path}: {exc}") from exc
    return {
        key: list(value.get("shape", []))
        for key, value in header.items()
        if key != "__metadata__" and isinstance(value, dict)
    }


def _write_manifest(root: Path, values: dict[str, Any]) -> None:
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **values,
        "files": file_inventory(root, hash_files=False),
    }
    (root / ARTIFACT_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_optional_manifest(root: Path) -> dict[str, Any]:
    path = root / ARTIFACT_MANIFEST
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid artifact manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactValidationError(f"artifact manifest must contain an object: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
