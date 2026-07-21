import threading
from types import MethodType, SimpleNamespace

import numpy as np
import pytest


rclpy = pytest.importorskip("rclpy")
pytest.importorskip("dexcontrol")

from geometry_msgs.msg import Twist  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from dex_vega_lerobot_inference.action_adapter import (  # noqa: E402
    ActionAdapter,
    load_joint_limits_from_urdf,
)
from dexcontrol_ros.dexcontrol_bridge import DexcontrolBridge  # noqa: E402


class _Clock:
    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds

    def now(self):
        return SimpleNamespace(nanoseconds=self.nanoseconds)


class _BridgeFixture:
    """Minimum state required to exercise bridge methods without Node/Robot init."""

    def __init__(self):
        self._lock = threading.Lock()
        self._warnings = []
        self._clock = _Clock(1_000_000_000)
        self._cmd_vel = np.zeros(3, dtype=np.float64)
        self._last_cmd_vel_time = None
        self._timeout = 0.5
        self._chassis = SimpleNamespace(max_lin_vel=0.4, max_ang_vel=0.8)
        self._joint_to_component = {}
        self._joint_targets = {}
        self._joint_limits = {}
        self._update_joint_target = MethodType(
            DexcontrolBridge._update_joint_target,
            self,
        )

    def get_clock(self):
        return self._clock

    def get_parameter(self, name):
        assert name == "cmd_vel_timeout_s"
        return SimpleNamespace(value=self._timeout)

    def _get_robot_component(self, name):
        assert name == "chassis"
        return self._chassis

    def _warn_throttled(self, key, message):
        self._warnings.append((key, message))


def test_adapter_joint_messages_enter_real_bridge_mapping(
    recorder_config,
    vega_urdf_path,
):
    limits = load_joint_limits_from_urdf(vega_urdf_path)
    adapter = ActionAdapter(recorder_config, limits)
    state = np.zeros(27, dtype=np.float64)
    state[20:24] = 0.5
    adapted = adapter.adapt(state.copy(), state, cycle_seconds=1.0 / 30.0)

    bridge = _BridgeFixture()
    for component, (names, positions) in adapted.component_positions.items():
        bridge._joint_targets[component] = np.zeros(len(names), dtype=np.float64)
        bridge._joint_limits[component] = np.asarray(
            [(limits[str(name)].lower, limits[str(name)].upper) for name in names]
        )
        for index, name in enumerate(names):
            bridge._joint_to_component[str(name)] = (component, index)

        message = JointState()
        message.name = [str(name) for name in names]
        message.position = [float(value) for value in positions]
        DexcontrolBridge._apply_named_joint_command(
            bridge,
            message,
            expected_component=component,
        )
        np.testing.assert_allclose(bridge._joint_targets[component], positions)


def test_real_bridge_clips_joint_and_base_commands_and_watchdog_zeros():
    bridge = _BridgeFixture()
    bridge._joint_targets["head"] = np.zeros(2, dtype=np.float64)
    bridge._joint_limits["head"] = np.asarray([[-0.5, 0.5], [-0.25, 0.25]])
    DexcontrolBridge._update_joint_target(
        bridge,
        "head",
        [(0, 1.0), (1, float("nan"))],
    )
    np.testing.assert_allclose(bridge._joint_targets["head"], [0.5, 0.0])

    command = Twist()
    command.linear.x = 1.0
    command.linear.y = -1.0
    command.angular.z = 2.0
    DexcontrolBridge._on_cmd_vel(bridge, command)
    np.testing.assert_allclose(bridge._cmd_vel, [0.4, -0.4, 0.8])

    bridge._clock.nanoseconds += 600_000_000
    current = DexcontrolBridge._current_cmd_vel_or_zero(bridge)
    np.testing.assert_allclose(current, np.zeros(3))
