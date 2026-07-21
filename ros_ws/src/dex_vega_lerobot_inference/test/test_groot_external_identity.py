from types import SimpleNamespace

import pytest

from dex_vega_lerobot_inference import inference_node
from dex_vega_lerobot_inference.artifact import ResolvedArtifact
from dex_vega_lerobot_inference.groot_artifact import GrootArtifactBundle
from dex_vega_lerobot_inference.groot_contracts import (
    BASE_MODEL_REPO_ID,
    BASE_MODEL_REVISION,
    CHECKPOINT_TAG,
    COSMOS_PROCESSOR_REPO_ID,
    COSMOS_PROCESSOR_REVISION,
    MODEL_REPO_ID,
    MODEL_REVISION,
)
from dex_vega_lerobot_inference.inference_node import (
    resolve_external_groot_artifacts,
)


def _runtime_info():
    return SimpleNamespace(
        policy_type="groot",
        model_path="/workspace/data/models/policy",
        model_commit=MODEL_REVISION,
        checkpoint_tag=CHECKPOINT_TAG,
        tokenizer_path="",
        tokenizer_commit=None,
        base_model_path="/workspace/data/models/base",
        base_model_commit=BASE_MODEL_REVISION,
        processor_path="/workspace/data/models/cosmos",
        processor_commit=COSMOS_PROCESSOR_REVISION,
    )


def test_groot_bundle_is_discovered_from_server_identity(monkeypatch, tmp_path):
    for name in ("policy", "base", "cosmos"):
        (tmp_path / "data" / "models" / name).mkdir(parents=True)
    calls = {}

    def fake_resolver(**kwargs):
        calls.update(kwargs)
        return GrootArtifactBundle(
            model=ResolvedArtifact(
                kwargs["model_local_path"],
                MODEL_REPO_ID,
                MODEL_REVISION,
                MODEL_REVISION,
                CHECKPOINT_TAG,
            ),
            base_model=ResolvedArtifact(
                kwargs["base_model_local_path"],
                BASE_MODEL_REPO_ID,
                BASE_MODEL_REVISION,
                BASE_MODEL_REVISION,
                None,
            ),
            cosmos_processor=ResolvedArtifact(
                kwargs["cosmos_processor_local_path"],
                COSMOS_PROCESSOR_REPO_ID,
                COSMOS_PROCESSOR_REVISION,
                COSMOS_PROCESSOR_REVISION,
                None,
            ),
        )

    monkeypatch.setattr(inference_node, "resolve_groot_artifacts", fake_resolver)
    bundle = resolve_external_groot_artifacts(
        _runtime_info(),
        local_project_root=tmp_path,
        server_project_root="/workspace",
    )
    assert bundle.model.local_path == tmp_path / "data" / "models" / "policy"
    assert calls["base_model_local_path"] == tmp_path / "data" / "models" / "base"
    assert calls["cosmos_processor_local_path"] == (
        tmp_path / "data" / "models" / "cosmos"
    )
    assert calls["allow_download"] is False
    assert calls["local_files_only"] is True


def test_groot_server_identity_must_match_every_exact_pin(tmp_path):
    info = _runtime_info()
    info.processor_commit = "f" * 40
    with pytest.raises(RuntimeError, match="exact GR00T deployment pin"):
        resolve_external_groot_artifacts(
            info,
            local_project_root=tmp_path,
            server_project_root="/workspace",
        )
