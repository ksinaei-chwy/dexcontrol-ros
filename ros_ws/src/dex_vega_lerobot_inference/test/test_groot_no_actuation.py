from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_groot_default_config_is_observe_only_and_unacknowledged():
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "groot_n17_blue_bird.yaml").read_text(
            encoding="utf-8"
        )
    )["dex_vega_lerobot_inference"]["ros__parameters"]
    assert config["policy_type"] == "groot"
    assert config["mode"] == "observe_only"
    assert config["allow_command_publication"] is False
    assert config["execution_readiness_acknowledged"] is False
    assert config["allow_model_download"] is False
    assert config["local_files_only"] is True
    assert config["inference_frequency_hz"] == 3.0
    assert config["execution_horizon"] == 21
    for server_owned_parameter in (
        "model_local_path",
        "model_revision",
        "checkpoint_tag",
        "base_model_local_path",
        "base_model_revision",
        "cosmos_processor_local_path",
        "cosmos_processor_revision",
    ):
        assert server_owned_parameter not in config


def test_groot_launches_keep_safe_defaults_explicit():
    observe = (PACKAGE_ROOT / "launch" / "groot_observe_only.launch.py").read_text(
        encoding="utf-8"
    )
    guarded = (
        PACKAGE_ROOT / "launch" / "groot_guarded_execution.launch.py"
    ).read_text(encoding="utf-8")
    assert '"mode": "observe_only"' in observe
    assert '"allow_command_publication": False' in observe
    assert '"execution_readiness_acknowledged": False' in observe
    assert '"allow_command_publication",\n                default_value="false"' in guarded
    assert '"execution_readiness_acknowledged",\n                default_value="false"' in guarded


def test_groot_replay_cannot_construct_command_publishers():
    replay = (PACKAGE_ROOT / "launch" / "groot_replay.launch.py").read_text(
        encoding="utf-8"
    )
    assert '"mode": "replay"' in replay
    assert '"camera_source": "ros_image"' in replay
    assert '"allow_command_publication": False' in replay
