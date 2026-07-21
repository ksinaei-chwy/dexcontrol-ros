from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PI05_LAUNCHES = (
    "dry_run.launch.py",
    "guarded_execution.launch.py",
    "observe_only.launch.py",
    "replay.launch.py",
)
CHECKPOINT_ARGUMENTS = (
    '"model_local_path"',
    '"model_revision"',
    '"checkpoint_tag"',
    '"tokenizer_local_path"',
)


@pytest.mark.parametrize("launch_name", PI05_LAUNCHES)
def test_pi05_launch_does_not_select_policy_server_artifacts(launch_name):
    text = (PACKAGE_ROOT / "launch" / launch_name).read_text(encoding="utf-8")
    for argument in CHECKPOINT_ARGUMENTS:
        assert argument not in text


def test_default_external_config_does_not_pin_a_checkpoint():
    text = (PACKAGE_ROOT / "config" / "pi05_blue_bird.yaml").read_text(
        encoding="utf-8"
    )
    for parameter in (
        "model_local_path:",
        "model_revision:",
        "checkpoint_tag:",
        "tokenizer_local_path:",
        "tokenizer_revision:",
    ):
        assert parameter not in text


def test_pi05_live_config_uses_measured_synchronization_skew_limit():
    text = (PACKAGE_ROOT / "config" / "pi05_blue_bird.yaml").read_text(
        encoding="utf-8"
    )
    assert "maximum_synchronization_skew_seconds: 0.20" in text


def test_policy_server_is_not_a_ros_launch():
    policy_server_launches = tuple(
        (PACKAGE_ROOT / "launch").glob("*policy_server*.launch.py")
    )
    assert policy_server_launches == ()
