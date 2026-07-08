#!/usr/bin/env python3
"""ROS 2 node for Pico XR teleoperation of Dexmate Vega."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from ament_index_python.packages import get_package_share_directory
import rclpy
from geometry_msgs.msg import Twist
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from dex_pico_teleop.calibration import CalibrationState, height_signal
from dex_pico_teleop.kinematics import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    ROBOT_ARM_REACH_M,
    ROBOT_SHOULDER_LATERAL_OFFSET_M,
    TORSO_JOINTS,
    VegaKinematics,
)
from dex_pico_teleop.log_frame import make_log_frame_payload
from dex_pico_teleop.network_receiver import NetworkReceiver
from dex_pico_teleop.pink_backend import PinkVegaKinematics
from dex_pico_teleop.retargeting import (
    fixed_head_joint_positions,
    normalized_reach_target,
    operator_arm_length_for_side,
    operator_shoulder_position,
    robot_shoulder_position,
)
from dex_pico_teleop.safety import VectorRateLimiter, base_twist_from_joysticks
from dex_pico_teleop.teleop_state import joint_values
from dex_pico_teleop.xr_packet import PicoPacket


PICO_BUTTON_ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("right", "a", "calibrate"),
    ("right", "b", "calibrate_reach"),
    ("left", "y", "enable"),
    ("left", "x", "disable"),
)


class PicoTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("pico_teleop_node")
        self._declare_parameters()

        qos = int(self.get_parameter("qos_depth").value)
        self._kin = self._make_kinematics()
        self._calibration = CalibrationState()
        self._enabled = False
        self._hold = False
        self._latest_packet: PicoPacket | None = None
        self._latest_packet_rx_ns = 0
        self._joint_positions: dict[str, float] = {}
        self._command_positions: dict[str, float] = {}
        self._status: dict[str, object] = {}
        self._retarget_debug: dict[str, object] = {}
        self._button_states: dict[tuple[str, str], bool] = {}
        self._button_states_initialized = False

        self._limiters = {
            "torso": VectorRateLimiter(
                float(self.get_parameter("max_torso_joint_delta_per_tick").value)
            ),
            "head": VectorRateLimiter(
                float(self.get_parameter("max_head_joint_delta_per_tick").value)
            ),
            "left_arm": VectorRateLimiter(
                float(self.get_parameter("max_arm_joint_delta_per_tick").value)
            ),
            "right_arm": VectorRateLimiter(
                float(self.get_parameter("max_arm_joint_delta_per_tick").value)
            ),
            "base": VectorRateLimiter(float(self.get_parameter("max_base_delta_per_tick").value)),
        }

        self._joint_pubs = {
            "torso": self.create_publisher(JointState, "/torso/joint_commands", qos),
            "head": self.create_publisher(JointState, "/head/joint_commands", qos),
            "left_arm": self.create_publisher(JointState, "/left_arm/joint_commands", qos),
            "right_arm": self.create_publisher(JointState, "/right_arm/joint_commands", qos),
            "left_hand": self.create_publisher(JointState, "/left_hand/joint_commands", qos),
            "right_hand": self.create_publisher(JointState, "/right_hand/joint_commands", qos),
        }
        self._cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", qos)
        self._status_pub = self.create_publisher(String, "/dex_pico_teleop/status", qos)
        self._log_frame_pub = self.create_publisher(String, "/dex_pico_teleop/log_frame", qos)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, qos)

        self.create_service(Trigger, "/dex_pico_teleop/calibrate", self._on_calibrate)
        self.create_service(
            Trigger,
            "/dex_pico_teleop/calibrate_reach",
            self._on_calibrate_reach,
        )
        self.create_service(SetBool, "/dex_pico_teleop/enabled", self._on_enabled)
        self.create_service(SetBool, "/dex_pico_teleop/hold", self._on_hold)
        self.create_service(Trigger, "/dex_pico_teleop/zero_base", self._on_zero_base)

        self._receiver: NetworkReceiver | None = None
        if bool(self.get_parameter("network_enabled").value):
            self._receiver = NetworkReceiver(
                transport=str(self.get_parameter("network_transport").value),
                host=str(self.get_parameter("network_host").value),
                port=int(self.get_parameter("network_port").value),
                log_info=self.get_logger().info,
                log_warn=self.get_logger().warn,
            )
            self._receiver.start()

        rate_hz = float(self.get_parameter("control_rate_hz").value)
        self._timer = self.create_timer(1.0 / rate_hz, self._on_timer)
        self.get_logger().info(
            "dex_pico_teleop ready; call /dex_pico_teleop/calibrate then "
            "/dex_pico_teleop/enabled true"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("qos_depth", 10)
        self.declare_parameter("publish_commands", True)
        self.declare_parameter("network_enabled", True)
        self.declare_parameter("network_transport", "tcp")
        self.declare_parameter("network_host", "0.0.0.0")
        self.declare_parameter("network_port", 63901)
        self.declare_parameter("kinematics_backend", "pink")
        self.declare_parameter("robot_urdf_path", "")
        self.declare_parameter("pink_qp_solver", "quadprog")
        self.declare_parameter("pink_self_collision_enabled", True)
        self.declare_parameter("pink_self_collision_components", ["left_arm", "right_arm"])
        self.declare_parameter("pink_self_collision_srdf_path", "")
        self.declare_parameter("pink_self_collision_urdf_path", "")
        self.declare_parameter("pink_self_collision_max_pairs", 24)
        self.declare_parameter("pink_self_collision_min_distance", 0.04)
        self.declare_parameter("pink_self_collision_gain", 1.0)
        self.declare_parameter("pink_self_collision_safe_displacement_gain", 0.0)
        self.declare_parameter("pink_velocity_limit_enabled", False)
        self.declare_parameter("pink_task_gain", 1.0)
        self.declare_parameter("pink_lm_damping", 1.0e-6)
        self.declare_parameter("pink_solve_damping", 1.0e-8)
        self.declare_parameter("pink_torso_max_iterations", 25)
        self.declare_parameter("pink_head_max_iterations", 8)
        self.declare_parameter("pink_arm_max_iterations", 20)
        self.declare_parameter("pink_arm_position_cost", 1.0)
        self.declare_parameter("pink_arm_orientation_cost", 0.1)
        self.declare_parameter("input_timeout_s", 0.35)
        self.declare_parameter("height_gain", 1.0)
        self.declare_parameter("torso_min_z", 0.72)
        self.declare_parameter("torso_max_z", 1.48)
        self.declare_parameter("head_tracking_enabled", False)
        self.declare_parameter("head_disabled_pitch_deg", 20.0)
        self.declare_parameter("operator_shoulder_width_m", 0.42)
        self.declare_parameter("operator_head_to_shoulder_z_m", 0.22)
        self.declare_parameter("operator_shoulder_x_m", 0.0)
        self.declare_parameter("operator_arm_length_ratio", 0.44)
        self.declare_parameter("operator_arm_length_min_m", 0.45)
        self.declare_parameter("operator_arm_length_max_m", 0.85)
        self.declare_parameter("robot_shoulder_lateral_offset_m", ROBOT_SHOULDER_LATERAL_OFFSET_M)
        self.declare_parameter("robot_arm_reach_m", ROBOT_ARM_REACH_M)
        self.declare_parameter("joystick_deadzone", 0.12)
        self.declare_parameter("base_vx_scale", 0.35)
        self.declare_parameter("base_vy_scale", 0.25)
        self.declare_parameter("base_wz_scale", 0.55)
        self.declare_parameter("max_joint_delta_per_tick", 0.014)
        self.declare_parameter("max_torso_joint_delta_per_tick", 0.028)
        self.declare_parameter("max_head_joint_delta_per_tick", 0.02)
        self.declare_parameter("max_arm_joint_delta_per_tick", 0.028)
        self.declare_parameter("max_base_delta_per_tick", 0.032)
        self.declare_parameter("left_hand_joint_names", [])
        self.declare_parameter("right_hand_joint_names", [])
        self.declare_parameter("left_hand_close_offsets", [])
        self.declare_parameter("right_hand_close_offsets", [])

    def _make_kinematics(self):
        backend = str(self.get_parameter("kinematics_backend").value).lower()
        if backend not in {"auto", "numeric", "pink"}:
            raise ValueError("kinematics_backend must be 'auto', 'numeric', or 'pink'")
        if backend == "numeric":
            self.get_logger().info("Using numeric IK backend")
            return VegaKinematics()

        try:
            urdf_path = self._robot_urdf_path()
            solver = str(self.get_parameter("pink_qp_solver").value)
            dt = 1.0 / float(self.get_parameter("control_rate_hz").value)
            self_collision_components = self._pink_self_collision_components()
            kin = PinkVegaKinematics(
                urdf_path,
                solver=solver,
                dt=dt,
                self_collision_components=self_collision_components,
                self_collision_srdf_path=(
                    self._self_collision_srdf_path() if self_collision_components else None
                ),
                self_collision_urdf_path=(
                    self._self_collision_urdf_path() if self_collision_components else None
                ),
                collision_package_dirs=(
                    self._collision_package_dirs() if self_collision_components else ()
                ),
                self_collision_n_pairs=int(
                    self.get_parameter("pink_self_collision_max_pairs").value
                ),
                self_collision_gain=float(
                    self.get_parameter("pink_self_collision_gain").value
                ),
                self_collision_safe_displacement_gain=float(
                    self.get_parameter("pink_self_collision_safe_displacement_gain").value
                ),
                self_collision_d_min=float(
                    self.get_parameter("pink_self_collision_min_distance").value
                ),
                velocity_limit_enabled=bool(
                    self.get_parameter("pink_velocity_limit_enabled").value
                ),
                task_gain=float(self.get_parameter("pink_task_gain").value),
                lm_damping=float(self.get_parameter("pink_lm_damping").value),
                solve_damping=float(self.get_parameter("pink_solve_damping").value),
                torso_max_iterations=int(
                    self.get_parameter("pink_torso_max_iterations").value
                ),
                head_max_iterations=int(self.get_parameter("pink_head_max_iterations").value),
                arm_max_iterations=int(self.get_parameter("pink_arm_max_iterations").value),
                arm_position_cost=float(self.get_parameter("pink_arm_position_cost").value),
                arm_orientation_cost=float(
                    self.get_parameter("pink_arm_orientation_cost").value
                ),
            )
            self.get_logger().info(
                f"Using Pinocchio/Pink IK backend with solver '{solver}' and URDF {urdf_path}"
            )
            if self_collision_components:
                collision_summary = ", ".join(
                    f"{component}="
                    f"{getattr(kin, component).collision_pair_count}/"
                    f"{getattr(kin, component).barrier_pair_count}"
                    for component in self_collision_components
                )
                self.get_logger().info(
                    "Pink self-collision barrier enabled for "
                    f"{', '.join(self_collision_components)} using SRDF "
                    f"{self._self_collision_srdf_path()} and collision URDF "
                    f"{self._self_collision_urdf_path()}"
                )
                self.get_logger().info(
                    "Pink self-collision reduced/barrier pair counts: "
                    f"{collision_summary}"
                )
            return kin
        except Exception as exc:  # noqa: BLE001 - optional backend fallback
            if backend == "pink":
                raise
            self.get_logger().warn(f"Pink IK backend unavailable, using numeric fallback: {exc}")
            return VegaKinematics()

    def _robot_urdf_path(self) -> Path:
        configured = str(self.get_parameter("robot_urdf_path").value)
        if configured:
            return Path(configured)
        description_share = Path(get_package_share_directory("dexmate_vega_description"))
        return description_share / "urdf" / "vega_1p_f5d6.package.urdf"

    def _self_collision_srdf_path(self) -> Path:
        configured = str(self.get_parameter("pink_self_collision_srdf_path").value)
        if configured:
            return Path(configured)
        moveit_share = Path(get_package_share_directory("dexmate_vega_moveit_config"))
        return moveit_share / "config" / "vega_1p_f5d6.srdf"

    def _self_collision_urdf_path(self) -> Path:
        configured = str(self.get_parameter("pink_self_collision_urdf_path").value)
        if configured:
            return Path(configured)
        description_share = Path(get_package_share_directory("dexmate_vega_description"))
        return (
            description_share
            / "robots"
            / "humanoid"
            / "vega_1p"
            / "vega_1p_f5d6_collision_spheres.collision.urdf"
        )

    def _collision_package_dirs(self) -> tuple[Path, ...]:
        description_share = Path(get_package_share_directory("dexmate_vega_description"))
        return (description_share.parent,)

    def _pink_self_collision_components(self) -> tuple[str, ...]:
        if not bool(self.get_parameter("pink_self_collision_enabled").value):
            return ()
        components = []
        for component in self._string_list_parameter("pink_self_collision_components"):
            normalized = component.lower()
            if normalized in {"left_arm", "right_arm"}:
                components.append(normalized)
            else:
                self.get_logger().warn(
                    "Ignoring Pink self-collision component "
                    f"'{component}'; torso/head are solved without Pink"
                )
        return tuple(components)

    def _on_joint_state(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            if math.isfinite(position):
                self._joint_positions[name] = float(position)

    def _on_enabled(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        response.success, response.message = self._set_enabled(bool(request.data))
        return response

    def _set_enabled(self, enabled: bool) -> tuple[bool, str]:
        self._enabled = enabled
        if not self._enabled:
            self._publish_zero_base()
        return True, f"teleop enabled={self._enabled}"

    def _on_hold(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self._hold = bool(request.data)
        if self._hold:
            self._publish_zero_base()
        response.success = True
        response.message = f"teleop hold={self._hold}"
        return response

    def _on_zero_base(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self._publish_zero_base()
        response.success = True
        response.message = "published zero cmd_vel"
        return response

    def _on_calibrate(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success, response.message = self._calibrate_from_latest_packet()
        return response

    def _on_calibrate_reach(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        response.success, response.message = self._calibrate_reach_from_latest_packet()
        return response

    def _calibrate_from_latest_packet(self) -> tuple[bool, str]:
        packet = self._latest_packet
        if packet is None:
            return False, "no Pico packet received yet"

        torso_q = self._current_q(TORSO_JOINTS, prefer_command=False)
        head_q = self._current_q(HEAD_JOINTS, prefer_command=False)
        left_q = self._current_q(LEFT_ARM_JOINTS, prefer_command=False)
        right_q = self._current_q(RIGHT_ARM_JOINTS, prefer_command=False)
        arm_center_pos, arm_center_rot = self._kin.arm_center_pose(torso_q)
        _head_pos, head_rot = self._kin.head.forward(head_q)
        arm_end_effector_rotations = {
            "left": self._kin.left_arm.forward(left_q)[1],
            "right": self._kin.right_arm.forward(right_q)[1],
        }
        hand_positions = {
            side: self._current_q(tuple(names), prefer_command=False)
            for side, names in self._hand_joint_names().items()
            if names
        }
        self._calibration.calibrate(
            packet,
            arm_center_pos,
            arm_center_rot,
            head_rot,
            hand_positions,
            arm_end_effector_rotations,
        )
        self._status.pop("operator_arm_lengths", None)
        self._status.pop("operator_arm_length_raw", None)
        for limiter in self._limiters.values():
            limiter.reset()
        return (
            True,
            "calibrated: neutral_height_signal="
            f"{self._calibration.neutral_height_signal:.3f} m",
        )

    def _calibrate_reach_from_latest_packet(self) -> tuple[bool, str]:
        packet = self._latest_packet
        if packet is None:
            return False, "no Pico packet received yet"
        if not self._calibration.calibrated:
            return False, "neutral calibration must be completed first"

        minimum = float(self.get_parameter("operator_arm_length_min_m").value)
        maximum = float(self.get_parameter("operator_arm_length_max_m").value)
        head_pose = self._calibration.to_operator_pose(packet.head)
        lengths: dict[str, float] = {}
        raw_lengths: dict[str, float] = {}
        for side in ("left", "right"):
            controller_pose = self._calibration.to_operator_pose(packet.controllers[side].pose)
            shoulder = operator_shoulder_position(
                side,
                head_pose.position,
                float(self.get_parameter("operator_shoulder_width_m").value),
                float(self.get_parameter("operator_head_to_shoulder_z_m").value),
                float(self.get_parameter("operator_shoulder_x_m").value),
            )
            raw = float(np.linalg.norm(controller_pose.position - shoulder))
            raw_lengths[side] = raw
            lengths[side] = float(np.clip(raw, minimum, maximum))

        self._calibration.set_operator_arm_lengths(lengths)
        self._status["operator_arm_lengths"] = lengths.copy()
        self._status["operator_arm_length_raw"] = raw_lengths.copy()
        return (
            True,
            "reach calibrated: "
            f"left={lengths['left']:.3f} m, right={lengths['right']:.3f} m",
        )

    def _on_timer(self) -> None:
        loop_start = time.perf_counter()
        packet = self._drain_packet()
        now_ns = self.get_clock().now().nanoseconds
        stale = self._packet_is_stale(now_ns)

        if packet is not None and not stale:
            self._handle_button_clicks(packet)

        if stale or not self._enabled or self._hold or not self._calibration.calibrated:
            self._publish_zero_base()
            self._publish_status(stale=stale)
            return

        assert packet is not None
        timings_ms: dict[str, float] = {}
        stage_start = time.perf_counter()
        torso_q = self._update_torso(packet)
        timings_ms["torso"] = _elapsed_ms(stage_start)
        stage_start = time.perf_counter()
        head_q = self._update_head(packet)
        timings_ms["head"] = _elapsed_ms(stage_start)
        stage_start = time.perf_counter()
        left_q = self._update_arm("left", packet)
        timings_ms["left_arm"] = _elapsed_ms(stage_start)
        stage_start = time.perf_counter()
        right_q = self._update_arm("right", packet)
        timings_ms["right_arm"] = _elapsed_ms(stage_start)
        stage_start = time.perf_counter()
        self._update_hands(packet)
        timings_ms["hands"] = _elapsed_ms(stage_start)
        stage_start = time.perf_counter()
        self._update_base(packet)
        timings_ms["base"] = _elapsed_ms(stage_start)
        timings_ms["loop"] = _elapsed_ms(loop_start)
        self._status["loop_ms"] = timings_ms["loop"]
        self._publish_log_frame(packet, torso_q, head_q, left_q, right_q, timings_ms)
        self._publish_status(stale=False)

    def _drain_packet(self) -> PicoPacket | None:
        if self._receiver is not None:
            packet = self._receiver.get_latest()
            if packet is not None:
                self._latest_packet = packet
                self._latest_packet_rx_ns = self.get_clock().now().nanoseconds
        return self._latest_packet

    def _packet_is_stale(self, now_ns: int) -> bool:
        if self._latest_packet is None or self._latest_packet_rx_ns == 0:
            return True
        timeout_s = float(self.get_parameter("input_timeout_s").value)
        age_s = (now_ns - self._latest_packet_rx_ns) / 1.0e9
        return age_s > timeout_s

    def _handle_button_clicks(self, packet: PicoPacket) -> None:
        current = self._button_states_from_packet(packet)
        if not self._button_states_initialized:
            self._button_states = current
            self._button_states_initialized = True
            return

        for side, button, action in PICO_BUTTON_ACTIONS:
            key = (side, button)
            if current.get(key, False) and not self._button_states.get(key, False):
                self._handle_button_action(side, button, action)
        self._button_states = current

    def _button_states_from_packet(self, packet: PicoPacket) -> dict[tuple[str, str], bool]:
        return {
            (side, button): packet.controllers[side].button(button)
            for side, button, _action in PICO_BUTTON_ACTIONS
        }

    def _handle_button_action(self, side: str, button: str, action: str) -> None:
        if action == "calibrate":
            success, message = self._calibrate_from_latest_packet()
        elif action == "calibrate_reach":
            success, message = self._calibrate_reach_from_latest_packet()
        elif action == "enable":
            success, message = self._set_enabled(True)
        elif action == "disable":
            success, message = self._set_enabled(False)
        else:
            return

        self._status["last_button_action"] = f"{side}_{button}_{action}"
        self._status["last_button_action_success"] = success
        self._status["last_button_action_message"] = message
        log = self.get_logger().info if success else self.get_logger().warn
        log(f"Pico {side} {button.upper()} click {action}: {message}")

    def _update_torso(self, packet: PicoPacket) -> np.ndarray:
        current = self._current_q(TORSO_JOINTS)
        delta_height = height_signal(packet) - self._calibration.neutral_height_signal
        target_z = self._calibration.neutral_arm_center_z + float(
            self.get_parameter("height_gain").value
        ) * delta_height
        target_z = float(
            np.clip(
                target_z,
                float(self.get_parameter("torso_min_z").value),
                float(self.get_parameter("torso_max_z").value),
            )
        )
        solution = self._kin.solve_torso_height(
            current,
            target_z,
            self._calibration.neutral_arm_center_pitch,
            self._calibration.neutral_arm_center_x,
        )
        limited = self._limiters["torso"].limit(solution.q)
        self._publish_joint_command("torso", TORSO_JOINTS, limited)
        self._status["torso_error"] = solution.error_norm
        self._status["torso_iterations"] = solution.iterations
        return limited

    def _update_head(self, packet: PicoPacket) -> np.ndarray:
        current = self._current_q(HEAD_JOINTS)
        if not bool(self.get_parameter("head_tracking_enabled").value):
            self._status["head_tracking_enabled"] = False
            target_q = fixed_head_joint_positions(
                float(self.get_parameter("head_disabled_pitch_deg").value)
            )
            limited = self._limiters["head"].limit(target_q)
            self._publish_joint_command("head", HEAD_JOINTS, limited)
            self._status["head_error"] = 0.0
            self._status["head_iterations"] = 0
            return limited
        else:
            self._status["head_tracking_enabled"] = True
            head_pose = self._calibration.to_operator_pose(packet.head)
            target_rotation = self._calibration.head_target_rotation(head_pose)
        solution = self._kin.solve_head_orientation(current, target_rotation)
        limited = self._limiters["head"].limit(solution.q)
        self._publish_joint_command("head", HEAD_JOINTS, limited)
        self._status["head_error"] = solution.error_norm
        self._status["head_iterations"] = solution.iterations
        return limited

    def _update_arm(self, side: str, packet: PicoPacket) -> np.ndarray:
        joint_names = LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS
        current = self._current_q(joint_names)
        controller = packet.controllers[side]

        controller_pose = self._calibration.to_operator_pose(controller.pose)
        head_pose = self._calibration.to_operator_pose(packet.head)
        operator_shoulder = operator_shoulder_position(
            side,
            head_pose.position,
            float(self.get_parameter("operator_shoulder_width_m").value),
            float(self.get_parameter("operator_head_to_shoulder_z_m").value),
            float(self.get_parameter("operator_shoulder_x_m").value),
        )
        operator_vector = controller_pose.position - operator_shoulder
        operator_arm_length = operator_arm_length_for_side(
            side,
            self._calibration.operator_arm_lengths,
            self._calibration.neutral_height_signal,
            float(self.get_parameter("operator_arm_length_ratio").value),
            float(self.get_parameter("operator_arm_length_min_m").value),
            float(self.get_parameter("operator_arm_length_max_m").value),
        )
        robot_shoulder = robot_shoulder_position(
            side,
            float(self.get_parameter("robot_shoulder_lateral_offset_m").value),
        )
        reach = normalized_reach_target(
            controller_pose.position,
            operator_shoulder,
            operator_arm_length.value,
            robot_shoulder,
            float(self.get_parameter("robot_arm_reach_m").value),
        )
        target_pos = reach.position
        target_rot = self._calibration.arm_target_rotation(side, controller_pose)
        solution = self._kin.solve_arm_pose(side, current, target_pos, target_rot)
        limited = self._limiters[f"{side}_arm"].limit(solution.q)
        self._publish_joint_command(f"{side}_arm", joint_names, limited)
        self._status[f"{side}_arm_error"] = solution.error_norm
        self._status[f"{side}_arm_iterations"] = solution.iterations
        self._status[f"{side}_arm_reach_fraction"] = reach.fraction
        self._status[f"{side}_arm_length_source"] = operator_arm_length.source
        self._retarget_debug[side] = {
            "operator_shoulder": operator_shoulder.tolist(),
            "controller_position": controller_pose.position.tolist(),
            "controller_rotation": controller_pose.rotation.tolist(),
            "operator_vector": operator_vector.tolist(),
            "operator_arm_length": operator_arm_length.value,
            "operator_arm_length_source": operator_arm_length.source,
            "reach_fraction": reach.fraction,
            "robot_shoulder": robot_shoulder.tolist(),
            "robot_target": target_pos.tolist(),
            "robot_target_rotation": target_rot.tolist(),
            "ik_error": solution.error_norm,
            "ik_iterations": solution.iterations,
        }
        return limited

    def _update_hands(self, packet: PicoPacket) -> None:
        names_by_side = self._hand_joint_names()
        offsets_by_side = {
            "left": self._float_list_parameter("left_hand_close_offsets"),
            "right": self._float_list_parameter("right_hand_close_offsets"),
        }
        for side in ("left", "right"):
            names = names_by_side[side]
            offsets = offsets_by_side[side]
            if not names or len(offsets) != len(names):
                continue
            open_pos = self._calibration.hand_open.get(side)
            if open_pos is None or open_pos.size != len(names):
                open_pos = self._current_q(tuple(names))
            close_pos = open_pos + np.asarray(offsets, dtype=np.float64)
            trigger = packet.controllers[side].trigger
            target = open_pos + trigger * (close_pos - open_pos)
            self._publish_joint_command(f"{side}_hand", tuple(names), target)

    def _update_base(self, packet: PicoPacket) -> None:
        twist = base_twist_from_joysticks(
            packet.controllers["left"].joystick,
            packet.controllers["right"].joystick,
            float(self.get_parameter("joystick_deadzone").value),
            float(self.get_parameter("base_vx_scale").value),
            float(self.get_parameter("base_vy_scale").value),
            float(self.get_parameter("base_wz_scale").value),
        )
        limited = self._limiters["base"].limit(twist)
        self._publish_twist(limited)

    def _publish_joint_command(
        self,
        component: str,
        names: Iterable[str],
        positions: np.ndarray,
    ) -> None:
        names_list = list(names)
        values = [
            float(value)
            for value in np.asarray(positions, dtype=np.float64).reshape(-1)
        ]
        for name, value in zip(names_list, values):
            self._command_positions[name] = value
        if not bool(self.get_parameter("publish_commands").value):
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names_list
        msg.position = values
        self._joint_pubs[component].publish(msg)

    def _publish_twist(self, values: np.ndarray) -> None:
        if not bool(self.get_parameter("publish_commands").value):
            return
        msg = Twist()
        msg.linear.x = float(values[0])
        msg.linear.y = float(values[1])
        msg.angular.z = float(values[2])
        self._cmd_vel_pub.publish(msg)

    def _publish_zero_base(self) -> None:
        self._limiters["base"].reset(np.zeros(3, dtype=np.float64))
        self._publish_twist(np.zeros(3, dtype=np.float64))

    def _publish_status(self, stale: bool) -> None:
        status = {
            "enabled": self._enabled,
            "hold": self._hold,
            "calibrated": self._calibration.calibrated,
            "stale_input": stale,
            "has_joint_state": bool(self._joint_positions),
            **self._status,
        }
        msg = String()
        msg.data = json.dumps(status, sort_keys=True)
        self._status_pub.publish(msg)

    def _publish_log_frame(
        self,
        packet: PicoPacket,
        torso_q: np.ndarray,
        head_q: np.ndarray,
        left_q: np.ndarray,
        right_q: np.ndarray,
        timings_ms: dict[str, float],
    ) -> None:
        msg = String()
        msg.data = json.dumps(
            make_log_frame_payload(
                packet.timestamp_ns,
                packet.sequence,
                torso_q,
                head_q,
                left_q,
                right_q,
                debug={
                    "retarget": self._retarget_debug,
                    "timing_ms": timings_ms,
                },
            ),
            sort_keys=True,
        )
        self._log_frame_pub.publish(msg)

    def _current_q(self, names: tuple[str, ...], prefer_command: bool = True) -> np.ndarray:
        return joint_values(
            names,
            self._joint_positions,
            self._command_positions,
            prefer_command=prefer_command,
        )

    def _hand_joint_names(self) -> dict[str, list[str]]:
        configured = {
            "left": self._string_list_parameter("left_hand_joint_names"),
            "right": self._string_list_parameter("right_hand_joint_names"),
        }
        if configured["left"] and configured["right"]:
            return configured
        all_names = list(self._joint_positions)
        if not configured["left"]:
            configured["left"] = [
                name
                for name in all_names
                if name.startswith(("L_th_", "L_ff_", "L_mf_", "L_rf_", "L_lf_"))
            ]
        if not configured["right"]:
            configured["right"] = [
                name
                for name in all_names
                if name.startswith(("R_th_", "R_ff_", "R_mf_", "R_rf_", "R_lf_"))
            ]
        return configured

    def _string_list_parameter(self, name: str) -> list[str]:
        return [str(value) for value in self._optional_list_parameter(name)]

    def _float_list_parameter(self, name: str) -> list[float]:
        return [float(value) for value in self._optional_list_parameter(name)]

    def _optional_list_parameter(self, name: str) -> list[object]:
        try:
            values = self.get_parameter(name).value
        except ParameterUninitializedException:
            return []
        if values is None:
            return []
        return list(values)

    def destroy_node(self) -> None:
        if self._receiver is not None:
            self._receiver.stop()
        super().destroy_node()


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: PicoTeleopNode | None = None
    try:
        node = PicoTeleopNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
