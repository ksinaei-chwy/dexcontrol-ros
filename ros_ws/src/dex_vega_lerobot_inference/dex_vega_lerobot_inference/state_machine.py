"""Small fail-closed inference lifecycle state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RuntimeState(str, Enum):
    UNCONFIGURED = "UNCONFIGURED"
    MODEL_LOADING = "MODEL_LOADING"
    OBSERVE_ONLY = "OBSERVE_ONLY"
    READY = "READY"
    ARMED = "ARMED"
    EXECUTING = "EXECUTING"
    FAULT = "FAULT"
    ESTOP = "ESTOP"


class StateTransitionError(RuntimeError):
    """Raised when a lifecycle transition would weaken a safety gate."""


@dataclass
class SafetyStateMachine:
    execution_capable: bool
    state: RuntimeState = RuntimeState.UNCONFIGURED
    reason: str = "not configured"

    def begin_model_load(self) -> None:
        self._require({RuntimeState.UNCONFIGURED, RuntimeState.FAULT})
        self.state = RuntimeState.MODEL_LOADING
        self.reason = "model loading"

    def model_ready(self) -> None:
        self._require({RuntimeState.MODEL_LOADING})
        self.state = RuntimeState.READY if self.execution_capable else RuntimeState.OBSERVE_ONLY
        self.reason = "model ready"

    def arm(self, gates_ok: bool, reason: str = "") -> None:
        self._require({RuntimeState.READY})
        if not self.execution_capable:
            raise StateTransitionError("this process was not launched execution-capable")
        if not gates_ok:
            raise StateTransitionError(reason or "arming gates are not satisfied")
        self.state = RuntimeState.ARMED
        self.reason = "explicitly armed"

    def executing(self) -> None:
        self._require({RuntimeState.ARMED, RuntimeState.EXECUTING})
        self.state = RuntimeState.EXECUTING
        self.reason = "publishing guarded commands"

    def disarm(self, reason: str) -> None:
        if self.state in {RuntimeState.ESTOP, RuntimeState.FAULT}:
            return
        self.state = RuntimeState.READY if self.execution_capable else RuntimeState.OBSERVE_ONLY
        self.reason = reason

    def fault(self, reason: str) -> None:
        self.state = RuntimeState.FAULT
        self.reason = reason

    def estop(self, reason: str = "bridge e-stop active") -> None:
        self.state = RuntimeState.ESTOP
        self.reason = reason

    def recover(self, gates_ok: bool, reason: str = "") -> None:
        self._require({RuntimeState.FAULT, RuntimeState.ESTOP})
        if not gates_ok:
            raise StateTransitionError(reason or "recovery gates are not satisfied")
        self.state = RuntimeState.READY if self.execution_capable else RuntimeState.OBSERVE_ONLY
        self.reason = "recovered; explicit re-arm required"

    @property
    def may_publish(self) -> bool:
        return self.execution_capable and self.state in {
            RuntimeState.ARMED,
            RuntimeState.EXECUTING,
        }

    def _require(self, allowed: set[RuntimeState]) -> None:
        if self.state not in allowed:
            expected = ", ".join(sorted(item.value for item in allowed))
            raise StateTransitionError(
                f"cannot transition from {self.state.value}; expected one of {expected}"
            )
