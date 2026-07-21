import json
import struct

import pytest

from dex_vega_lerobot_inference.artifact import (
    EXPECTED_MODEL_FILES,
    TOKENIZER_DOWNLOAD_PATTERNS,
    ArtifactValidationError,
    require_project_local,
    resolve_model_artifact,
    validate_model_artifact,
)


def _write_valid_fixture(root):
    root.mkdir()
    config = {
        "type": "pi05",
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "use_relative_actions": False,
        "input_features": {"observation.state": {"shape": [32]}},
        "output_features": {"action": {"shape": [27]}},
    }
    preprocessor = {
        "steps": [
            {
                "rename": {
                    "observation.images.head": "observation.images.base_0_rgb"
                }
            },
            {"normalizer": "quantiles"},
            {"tokenizer": "google/paligemma-3b-pt-224"},
            {"device": "cuda"},
        ]
    }
    postprocessor = {"steps": [{"unnormalizer": "quantiles"}, {"device": "cpu"}]}
    values = {
        "config.json": config,
        "train_config.json": {},
        "policy_preprocessor.json": preprocessor,
        "policy_postprocessor.json": postprocessor,
    }
    for name, value in values.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    for name in EXPECTED_MODEL_FILES:
        path = root / name
        if not path.exists():
            if name.endswith("processor.safetensors"):
                shapes = {
                    "action.q01": [27],
                    "action.q99": [27],
                }
                if "preprocessor" in name:
                    shapes.update(
                        {
                            "observation.state.q01": [27],
                            "observation.state.q99": [27],
                        }
                    )
                header = {
                    key: {"dtype": "F32", "shape": shape, "data_offsets": [0, 0]}
                    for key, shape in shapes.items()
                }
                encoded = json.dumps(header).encode("utf-8")
                padding = b" " * ((8 - len(encoded) % 8) % 8)
                encoded += padding
                path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)
            else:
                path.write_bytes(b"fixture")


def test_complete_artifact_contract_is_accepted(tmp_path):
    root = tmp_path / "model"
    _write_valid_fixture(root)
    parsed = validate_model_artifact(root)
    assert parsed["config.json"]["type"] == "pi05"


def test_missing_processor_state_is_rejected(tmp_path):
    root = tmp_path / "model"
    _write_valid_fixture(root)
    (root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").unlink()
    with pytest.raises(ArtifactValidationError, match="incomplete"):
        validate_model_artifact(root)


def test_wrong_physical_action_shape_is_rejected(tmp_path):
    root = tmp_path / "model"
    _write_valid_fixture(root)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["output_features"]["action"]["shape"] = [32]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="action shape"):
        validate_model_artifact(root)


def test_paths_cannot_escape_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    assert require_project_local(project / "models", project) == project / "models"
    with pytest.raises(ArtifactValidationError, match="inside project"):
        require_project_local(tmp_path / "outside", project)


def test_tokenizer_download_does_not_include_paligemma_model_weights():
    assert "tokenizer.json" in TOKENIZER_DOWNLOAD_PATTERNS
    assert "tokenizer.model" in TOKENIZER_DOWNLOAD_PATTERNS
    assert not any("safetensors" in pattern for pattern in TOKENIZER_DOWNLOAD_PATTERNS)


def test_local_model_revision_and_tag_must_match_manifest(tmp_path):
    project = tmp_path / "project"
    root = project / "model"
    project.mkdir()
    _write_valid_fixture(root)
    commit = "a" * 40
    (root / "dexmate_artifact_manifest.json").write_text(
        json.dumps(
            {
                "model_repo_id": "Kasra99/pi05-dexmate-blue-bird",
                "model_requested_revision": commit,
                "model_resolved_commit": commit,
                "checkpoint_tag": "step-005000",
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_model_artifact(
        project_root=project,
        local_path=root,
        repo_id=None,
        revision=commit,
        download_directory=None,
        checkpoint_tag="step-005000",
        allow_download=False,
        local_files_only=True,
    )
    assert resolved.resolved_commit == commit

    with pytest.raises(ArtifactValidationError, match="revision differs"):
        resolve_model_artifact(
            project_root=project,
            local_path=root,
            repo_id=None,
            revision="b" * 40,
            download_directory=None,
            checkpoint_tag="step-005000",
            allow_download=False,
            local_files_only=True,
        )
    with pytest.raises(ArtifactValidationError, match="tag differs"):
        resolve_model_artifact(
            project_root=project,
            local_path=root,
            repo_id=None,
            revision=commit,
            download_directory=None,
            checkpoint_tag="step-015000",
            allow_download=False,
            local_files_only=True,
        )


def test_checkpoint_tag_is_inferred_from_older_requested_revision_manifest(tmp_path):
    project = tmp_path / "project"
    root = project / "model"
    project.mkdir()
    _write_valid_fixture(root)
    commit = "c" * 40
    (root / "dexmate_artifact_manifest.json").write_text(
        json.dumps(
            {
                "model_repo_id": "Kasra99/pi05-dexmate-blue-bird",
                "model_requested_revision": "step-030000",
                "model_resolved_commit": commit,
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_model_artifact(
        project_root=project,
        local_path=root,
        repo_id=None,
        revision=commit,
        download_directory=None,
        checkpoint_tag="step-030000",
        allow_download=False,
        local_files_only=True,
    )
    assert resolved.checkpoint_tag == "step-030000"

    with pytest.raises(ArtifactValidationError, match="tag differs"):
        resolve_model_artifact(
            project_root=project,
            local_path=root,
            repo_id=None,
            revision=commit,
            download_directory=None,
            checkpoint_tag="step-015000",
            allow_download=False,
            local_files_only=True,
        )
