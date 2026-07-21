from types import SimpleNamespace

import pytest

from dex_vega_lerobot_inference.groot_policy_runtime import GrootPolicyRuntime
from dex_vega_lerobot_inference.policy_runtime import PolicyRuntimeError


def _runtime_config():
    return SimpleNamespace(
        type="groot",
        chunk_size=40,
        n_action_steps=40,
        max_state_dim=132,
        max_action_dim=132,
        n_obs_steps=1,
        embodiment_tag="new_embodiment",
        use_relative_actions=False,
        use_bf16=True,
        model_params_fp32=False,
        action_decode_transform=None,
        use_peft=False,
        base_model_path="nvidia/GR00T-N1.7-3B",
        output_features={"action": SimpleNamespace(shape=(27,))},
        input_features={"observation.state": SimpleNamespace(shape=(27,))},
        image_features={
            "observation.images.head": SimpleNamespace(shape=(3, 480, 640))
        },
    )


def test_runtime_accepts_exact_groot_n17_contract():
    GrootPolicyRuntime._validate_config(_runtime_config())


def test_runtime_rejects_relative_actions_or_wrong_camera():
    config = _runtime_config()
    config.use_relative_actions = True
    config.image_features = {
        "observation.images.left_wrist": SimpleNamespace(shape=(3, 480, 640))
    }
    with pytest.raises(PolicyRuntimeError, match="contract mismatch"):
        GrootPolicyRuntime._validate_config(config)
