import pytest

pytest.importorskip("rclpy")

from dex_vega_lerobot_inference.inference_node import (  # noqa: E402
    command_publication_capable,
    execution_duration_gate_failure,
    unexpected_command_publisher_counts,
)


@pytest.mark.parametrize("mode", ["observe_only", "dry_run", "replay"])
@pytest.mark.parametrize("allow", [False, True])
def test_non_execution_modes_cannot_construct_command_publishers(mode, allow):
    assert not command_publication_capable(mode, allow)


def test_guarded_mode_still_requires_explicit_publication_flag():
    assert not command_publication_capable("armed", False)
    assert command_publication_capable("armed", True)


def test_command_topics_must_have_only_the_inference_publisher():
    assert unexpected_command_publisher_counts({"/cmd_vel": 1}) == {}
    assert unexpected_command_publisher_counts(
        {"/cmd_vel": 2, "/left_arm/joint_commands": 0}
    ) == {"/cmd_vel": 2, "/left_arm/joint_commands": 0}


def test_execution_duration_limit_fails_closed():
    assert execution_duration_gate_failure(0, 1, 5.0) == ""
    assert execution_duration_gate_failure(1_000_000_000, 6_000_000_000, 5.0) == ""
    assert "exceeded" in execution_duration_gate_failure(
        1_000_000_000, 6_000_000_001, 5.0
    )
    assert "finite and positive" in execution_duration_gate_failure(0, 1, 0.0)
