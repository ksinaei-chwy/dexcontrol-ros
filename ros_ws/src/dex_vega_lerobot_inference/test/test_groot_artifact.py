import hashlib
import json
import os
import struct

import pytest

from dex_vega_lerobot_inference.artifact import ArtifactValidationError
from dex_vega_lerobot_inference.groot_artifact import (
    COSMOS_PROCESSOR_REQUIRED_FILES,
    _write_snapshot_manifest,
    resolve_groot_artifacts,
    validate_cosmos_processor_artifact,
    validate_groot_base_model_artifact,
    validate_groot_model_artifact,
    validate_snapshot_manifest,
)
from dex_vega_lerobot_inference.groot_contracts import (
    BASE_MODEL_REPO_ID,
    BASE_MODEL_REVISION,
    CHECKPOINT_TAG,
    COSMOS_PROCESSOR_REPO_ID,
    COSMOS_PROCESSOR_REVISION,
    MODEL_REPO_ID,
    MODEL_REVISION,
)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_safetensors_header(path, shapes):
    header = {
        key: {"dtype": "F32", "shape": shape, "data_offsets": [0, 0]}
        for key, shape in shapes.items()
    }
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def _policy_config():
    return {
        "type": "groot",
        "chunk_size": 40,
        "n_action_steps": 40,
        "max_state_dim": 132,
        "max_action_dim": 132,
        "embodiment_tag": "new_embodiment",
        "base_model_path": BASE_MODEL_REPO_ID,
        "action_decode_transform": None,
        "use_relative_actions": False,
        "use_bf16": True,
        "model_params_fp32": False,
        "use_peft": False,
        "lora_rank": 0,
        "tune_llm": False,
        "tune_visual": False,
        "tune_projector": True,
        "tune_diffusion_model": True,
        "tune_vlln": True,
        "tune_top_llm_layers": 0,
        "input_features": {
            "observation.images.head": {
                "type": "VISUAL",
                "shape": [3, 480, 640],
            },
            "observation.state": {"type": "STATE", "shape": [27]},
        },
        "output_features": {
            "action": {"type": "ACTION", "shape": [27]},
        },
    }


def _write_model_fixture(root):
    root.mkdir()
    config = _policy_config()
    train = {
        "seed": 1000,
        "batch_size": 8,
        "steps": 170_000,
        "peft": None,
        "dataset": {
            "repo_id": "Kasra99/dexmate_blue_bird",
            "revision": "72a97b1a916699c17177e311463729d757f3119c",
        },
        "policy": dict(config),
    }
    pre_state = "policy_preprocessor_step_2_groot_n1_7_pack_inputs_v1.safetensors"
    post_state = (
        "policy_postprocessor_step_0_"
        "groot_action_unpack_unnormalize_v2.safetensors"
    )
    pre = {
        "name": "policy_preprocessor",
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {}},
            },
            {"registry_name": "to_batch_processor", "config": {}},
            {
                "registry_name": "groot_n1_7_pack_inputs_v1",
                "config": {
                    "action_horizon": 40,
                    "valid_action_horizon": 40,
                    "max_state_dim": 132,
                    "max_action_dim": 132,
                    "embodiment_tag": "new_embodiment",
                    "normalize_min_max": True,
                    "video_modality_keys": ["observation.images.head"],
                },
                "state_file": pre_state,
            },
            {
                "registry_name": "groot_n1_7_vlm_encode_v1",
                "config": {"model_name": COSMOS_PROCESSOR_REPO_ID},
            },
            {"registry_name": "device_processor", "config": {"device": "cuda"}},
        ],
    }
    post = {
        "name": "policy_postprocessor",
        "steps": [
            {
                "registry_name": "groot_action_unpack_unnormalize_v2",
                "config": {
                    "env_action_dim": 27,
                    "normalize_min_max": True,
                    "clip_normalized_action": True,
                    "libero_gripper_action": False,
                },
                "state_file": post_state,
            },
            {"registry_name": "device_processor", "config": {"device": "cpu"}},
        ],
    }
    model_bytes = b"small-model-fixture"
    (root / "model.safetensors").write_bytes(model_bytes)
    _write_json(root / "config.json", config)
    _write_json(root / "train_config.json", train)
    _write_json(root / "policy_preprocessor.json", pre)
    _write_json(root / "policy_postprocessor.json", post)
    _write_safetensors_header(
        root / pre_state,
        {
            "observation.state.min": [27],
            "observation.state.max": [27],
            "action.min": [27],
            "action.max": [27],
        },
    )
    _write_safetensors_header(
        root / post_state,
        {"action.min": [27], "action.max": [27]},
    )
    return model_bytes


def _write_base_fixture(root):
    root.mkdir()
    _write_json(
        root / "config.json",
        {
            "model_type": "Gr00tN1d7",
            "architectures": ["Gr00tN1d7"],
            "model_name": COSMOS_PROCESSOR_REPO_ID,
        },
    )
    _write_json(root / "embodiment_id.json", {"new_embodiment": 31})
    shard_names = {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    for name in shard_names:
        (root / name).write_bytes(b"base-shard")
    _write_json(
        root / "model.safetensors.index.json",
        {"weight_map": {"a": sorted(shard_names)[0], "b": sorted(shard_names)[1]}},
    )


def _write_cosmos_fixture(root):
    root.mkdir()
    for name in COSMOS_PROCESSOR_REQUIRED_FILES:
        value = {"fixture": True}
        if name == "config.json":
            value = {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
            }
        if name.endswith(".json"):
            _write_json(root / name, value)
        else:
            (root / name).write_text("fixture\n", encoding="utf-8")


def test_complete_saved_groot_bundle_contract_is_accepted(tmp_path):
    root = tmp_path / "policy"
    model_bytes = _write_model_fixture(root)
    parsed = validate_groot_model_artifact(
        root,
        expected_weight_size=len(model_bytes),
        expected_weight_sha256=hashlib.sha256(model_bytes).hexdigest(),
    )
    assert parsed["config.json"]["type"] == "groot"
    assert parsed["model_weight_size"] == len(model_bytes)


def test_model_contract_rejects_changed_training_or_processor_state(tmp_path):
    root = tmp_path / "policy"
    model_bytes = _write_model_fixture(root)
    train_path = root / "train_config.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    train["dataset"]["revision"] = "b" * 40
    _write_json(train_path, train)
    with pytest.raises(ArtifactValidationError, match="dataset revision"):
        validate_groot_model_artifact(
            root,
            expected_weight_size=len(model_bytes),
            expected_weight_sha256=hashlib.sha256(model_bytes).hexdigest(),
        )


def test_model_contract_rejects_missing_serialized_unnormalization(tmp_path):
    root = tmp_path / "policy"
    model_bytes = _write_model_fixture(root)
    post_path = root / "policy_postprocessor.json"
    post = json.loads(post_path.read_text(encoding="utf-8"))
    post["steps"][0]["registry_name"] = "wrong_unpack_step"
    _write_json(post_path, post)
    with pytest.raises(ArtifactValidationError, match="unnormalization"):
        validate_groot_model_artifact(
            root,
            expected_weight_size=len(model_bytes),
            expected_weight_sha256=hashlib.sha256(model_bytes).hexdigest(),
        )


def test_base_and_processor_only_cosmos_contracts(tmp_path):
    base = tmp_path / "base"
    cosmos = tmp_path / "cosmos"
    _write_base_fixture(base)
    _write_cosmos_fixture(cosmos)
    assert len(validate_groot_base_model_artifact(base)["weight_files"]) == 2
    assert validate_cosmos_processor_artifact(cosmos)["config.json"]["model_type"] == (
        "qwen3_vl"
    )
    (cosmos / "model.safetensors").write_bytes(b"must-not-be-downloaded")
    with pytest.raises(ArtifactValidationError, match="only pinned processor"):
        validate_cosmos_processor_artifact(cosmos)


def test_base_rejects_orphaned_shard_and_cosmos_rejects_any_weight_shard(tmp_path):
    base = tmp_path / "orphaned-base"
    cosmos = tmp_path / "cosmos"
    base.mkdir()
    _write_json(
        base / "config.json",
        {
            "model_type": "Gr00tN1d7",
            "architectures": ["Gr00tN1d7"],
            "model_name": COSMOS_PROCESSOR_REPO_ID,
        },
    )
    _write_json(base / "embodiment_id.json", {"new_embodiment": 31})
    (base / "model-00001-of-00002.safetensors").write_bytes(b"orphan")
    with pytest.raises(ArtifactValidationError, match="orphaned weight shard"):
        validate_groot_base_model_artifact(base)

    _write_cosmos_fixture(cosmos)
    (cosmos / "model-00001-of-00002.safetensors").write_bytes(b"must-not-exist")
    with pytest.raises(ArtifactValidationError, match="only pinned processor"):
        validate_cosmos_processor_artifact(cosmos)


def test_manifest_detects_added_or_modified_local_files(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    _write_snapshot_manifest(
        root,
        role="fixture",
        repo_id="owner/repo",
        revision="a" * 40,
        checkpoint_tag=None,
    )
    validate_snapshot_manifest(
        root,
        role="fixture",
        repo_id="owner/repo",
        revision="a" * 40,
    )
    (root / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="inventory changed"):
        validate_snapshot_manifest(
            root,
            role="fixture",
            repo_id="owner/repo",
            revision="a" * 40,
        )


def test_resolver_rejects_any_revision_other_than_exact_deployment_pins(tmp_path):
    with pytest.raises(ArtifactValidationError, match="deployment pin"):
        resolve_groot_artifacts(
            project_root=tmp_path,
            model_local_path=tmp_path / "policy",
            base_model_local_path=tmp_path / "base",
            cosmos_processor_local_path=tmp_path / "cosmos",
            model_revision="f" * 40,
            base_model_revision=BASE_MODEL_REVISION,
            cosmos_processor_revision=COSMOS_PROCESSOR_REVISION,
            checkpoint_tag=CHECKPOINT_TAG,
        )
    with pytest.raises(ArtifactValidationError, match="checkpoint tag"):
        resolve_groot_artifacts(
            project_root=tmp_path,
            model_local_path=tmp_path / "policy",
            base_model_local_path=tmp_path / "base",
            cosmos_processor_local_path=tmp_path / "cosmos",
            model_revision=MODEL_REVISION,
            base_model_revision=BASE_MODEL_REVISION,
            cosmos_processor_revision=COSMOS_PROCESSOR_REVISION,
            checkpoint_tag="main",
        )


def test_download_requires_environment_token_before_creating_directories(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    model = tmp_path / "policy"
    with pytest.raises(ArtifactValidationError, match="HF_TOKEN is not set"):
        resolve_groot_artifacts(
            project_root=tmp_path,
            model_local_path=model,
            base_model_local_path=tmp_path / "base",
            cosmos_processor_local_path=tmp_path / "cosmos",
            allow_download=True,
            local_files_only=False,
        )
    assert not model.exists()
    assert "HF_TOKEN" not in os.environ


def test_contract_constants_retain_exact_repository_identity():
    assert MODEL_REPO_ID == "Kasra99/groot-n17-dexmate-blue-bird"
    assert len(MODEL_REVISION) == 40
    assert BASE_MODEL_REPO_ID == "nvidia/GR00T-N1.7-3B"
    assert COSMOS_PROCESSOR_REPO_ID == "nvidia/Cosmos-Reason2-2B"
