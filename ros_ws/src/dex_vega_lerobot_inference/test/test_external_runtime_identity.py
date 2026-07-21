from pathlib import Path
from types import SimpleNamespace

import pytest

from dex_vega_lerobot_inference import inference_node
from dex_vega_lerobot_inference.artifact import ResolvedArtifact
from dex_vega_lerobot_inference.contracts import MODEL_REPO_ID
from dex_vega_lerobot_inference.inference_node import (
    InferenceNode,
    project_local_path_from_server,
    project_relative_paths_match,
    resolve_external_pi05_artifacts,
)


def test_external_paths_match_across_workspace_bind(tmp_path: Path):
    local_root = tmp_path / "ros_ws"
    local_model = local_root / "data" / "models" / "policy" / "step-005000"
    local_model.mkdir(parents=True)

    assert project_relative_paths_match(
        "/workspace/data/models/policy/step-005000",
        "/workspace",
        local_model,
        local_root,
    )


def test_external_paths_reject_different_or_unscoped_artifacts(tmp_path: Path):
    local_root = tmp_path / "ros_ws"
    local_model = local_root / "data" / "models" / "policy" / "step-005000"
    local_model.mkdir(parents=True)

    assert not project_relative_paths_match(
        "/workspace/data/models/policy/step-015000",
        "/workspace",
        local_model,
        local_root,
    )
    assert not project_relative_paths_match(
        "/outside/data/models/policy/step-005000",
        "/workspace",
        local_model,
        local_root,
    )
    assert not project_relative_paths_match(
        "data/models/policy/step-005000",
        "/workspace",
        local_model,
        local_root,
    )


def test_server_path_maps_to_the_same_project_relative_host_path(tmp_path: Path):
    local_root = tmp_path / "ros_ws"
    expected = local_root / "data" / "models" / "policy" / "step-030000"
    expected.mkdir(parents=True)

    assert project_local_path_from_server(
        "/workspace/data/models/policy/step-030000",
        "/workspace",
        local_root,
    ) == expected

    with pytest.raises(ValueError, match="outside"):
        project_local_path_from_server(
            "/other/data/models/policy/step-030000",
            "/workspace",
            local_root,
        )


def test_pi05_artifacts_are_discovered_from_server_identity(monkeypatch, tmp_path):
    local_root = tmp_path / "ros_ws"
    model_path = local_root / "data" / "models" / "policy" / "step-030000"
    tokenizer_path = local_root / "data" / "models" / "tokenizer"
    model_path.mkdir(parents=True)
    tokenizer_path.mkdir(parents=True)
    model_commit = "a" * 40
    tokenizer_commit = "b" * 40
    calls = {}

    def fake_model_resolver(**kwargs):
        calls["model"] = kwargs
        return ResolvedArtifact(
            model_path,
            MODEL_REPO_ID,
            "step-030000",
            model_commit,
            "step-030000",
        )

    def fake_tokenizer_resolver(**kwargs):
        calls["tokenizer"] = kwargs
        return ResolvedArtifact(
            tokenizer_path,
            "google/paligemma-3b-pt-224",
            tokenizer_commit,
            tokenizer_commit,
            None,
        )

    monkeypatch.setattr(
        inference_node,
        "resolve_model_artifact",
        fake_model_resolver,
    )
    monkeypatch.setattr(
        inference_node,
        "resolve_tokenizer",
        fake_tokenizer_resolver,
    )
    info = SimpleNamespace(
        policy_type="pi05",
        model_path="/workspace/data/models/policy/step-030000",
        model_commit=model_commit,
        checkpoint_tag="step-030000",
        tokenizer_path="/workspace/data/models/tokenizer",
        tokenizer_commit=tokenizer_commit,
    )

    model, tokenizer = resolve_external_pi05_artifacts(
        info,
        local_project_root=local_root,
        server_project_root="/workspace",
    )

    assert model.local_path == model_path
    assert tokenizer.local_path == tokenizer_path
    assert calls["model"]["local_path"] == model_path
    assert calls["model"]["revision"] == model_commit
    assert calls["model"]["checkpoint_tag"] == "step-030000"
    assert calls["tokenizer"]["local_path"] == tokenizer_path
    assert calls["tokenizer"]["revision"] == tokenizer_commit


def test_external_identity_rejects_checkpoint_tag_disagreement(tmp_path: Path):
    local_root = tmp_path / "ros_ws"
    model_path = local_root / "model"
    tokenizer_path = local_root / "tokenizer"
    model_path.mkdir(parents=True)
    tokenizer_path.mkdir(parents=True)
    model_commit = "a" * 40
    tokenizer_commit = "b" * 40
    info = SimpleNamespace(
        policy_type="pi05",
        action_chunk_size=50,
        action_dimension=27,
        model_path="/workspace/model",
        model_commit=model_commit,
        checkpoint_tag="step-030000",
        tokenizer_path="/workspace/tokenizer",
        tokenizer_commit=tokenizer_commit,
    )
    runtime = SimpleNamespace(info=info)
    model = ResolvedArtifact(
        model_path,
        MODEL_REPO_ID,
        "step-015000",
        model_commit,
        "step-015000",
    )
    tokenizer = ResolvedArtifact(
        tokenizer_path,
        "google/paligemma-3b-pt-224",
        tokenizer_commit,
        tokenizer_commit,
        None,
    )

    with pytest.raises(RuntimeError, match="checkpoint tag"):
        InferenceNode._validate_external_runtime_identity(
            runtime,
            model,
            tokenizer,
            base_model=None,
            processor=None,
            expected_policy_type="pi05",
            local_project_root=local_root,
            server_project_root=Path("/workspace"),
        )
