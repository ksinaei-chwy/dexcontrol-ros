import pytest

from dex_vega_lerobot_inference.state_machine import (
    RuntimeState,
    SafetyStateMachine,
    StateTransitionError,
)


def test_observe_only_can_never_publish():
    state = SafetyStateMachine(execution_capable=False)
    state.begin_model_load()
    state.model_ready()
    assert state.state is RuntimeState.OBSERVE_ONLY
    assert not state.may_publish
    with pytest.raises(StateTransitionError):
        state.arm(True)


def test_execution_requires_ready_explicit_arm_and_rearm_after_fault():
    state = SafetyStateMachine(execution_capable=True)
    state.begin_model_load()
    state.model_ready()
    with pytest.raises(StateTransitionError, match="fresh data"):
        state.arm(False, "fresh data missing")
    state.arm(True)
    assert state.state is RuntimeState.ARMED and state.may_publish
    state.executing()
    state.fault("test fault")
    assert not state.may_publish
    state.recover(True)
    assert state.state is RuntimeState.READY and not state.may_publish


def test_estop_cannot_be_cleared_by_disarm():
    state = SafetyStateMachine(execution_capable=True)
    state.estop()
    state.disarm("ordinary stop")
    assert state.state is RuntimeState.ESTOP
