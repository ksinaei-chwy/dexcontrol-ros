from types import SimpleNamespace

import pytest

from dex_vega_lerobot_inference.policy_runtime import (
    Pi05PolicyRuntime,
    PolicyRuntimeError,
)


def _config(**overrides):
    values = {
        "type": "pi05",
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "n_obs_steps": 1,
        "num_inference_steps": 10,
        "image_resolution": (224, 224),
        "dtype": "bfloat16",
        "use_relative_actions": False,
        "output_features": {"action": SimpleNamespace(shape=(27,))},
        "input_features": {"observation.state": SimpleNamespace(shape=(32,))},
        "image_features": {
            "observation.images.base_0_rgb": SimpleNamespace(shape=(3, 224, 224))
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_saved_policy_contract_accepts_model_facing_32():
    Pi05PolicyRuntime._validate_config(_config())


@pytest.mark.parametrize("key", ["max_state_dim", "max_action_dim"])
def test_saved_policy_contract_rejects_manual_27_model_padding(key):
    with pytest.raises(PolicyRuntimeError, match=key):
        Pi05PolicyRuntime._validate_config(_config(**{key: 27}))
