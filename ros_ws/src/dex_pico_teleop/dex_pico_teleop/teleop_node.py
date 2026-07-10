#!/usr/bin/env python3
"""ROS 2 node for Pico XR teleoperation of Dexmate Vega."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
from ament_index_python.packages import get_package_share_directory
import rclpy
from geometry_msgs.msg import Twist
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from dex_pico_teleop.calibration import (
    CalibrationState,
    average_calibration_packets,
    height_signal,
)
from dex_pico_teleop.hand_retargeting import (
    F5D6HandConfig,
    retarget_f5d6_hand,
)
from dex_pico_teleop.kinematics import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    ROBOT_ARM_REACH_M,
    ROBOT_SHOULDER_LATERAL_OFFSET_M,
    TORSO_JOINTS,
    VegaKinematics,
    clip_arm_j4_upper_limit,
)
from dex_pico_teleop.log_frame import make_log_frame_payload
from dex_pico_teleop.network_receiver import NetworkReceiver
from dex_pico_teleop.pink_backend import PinkVegaKinematics
from dex_pico_teleop.retargeting import (
    controller_hand_point,
    fit_two_pose_shoulder_reach,
    fixed_head_joint_positions,
    normalized_reach_target_from_vector,
    operator_arm_length_for_side,
    operator_shoulder_from_offset,
    operator_shoulder_position,
    robot_shoulder_position,
    vector_in_arm_center,
)
from dex_pico_teleop.safety import VectorRateLimiter, base_twist_from_joysticks
from dex_pico_teleop.teleop_state import (
    PositionTargetPlant,
    assess_arm_ik_solution,
    joint_values,
)
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
        self._controller_to_hand_offsets = self._load_controller_to_hand_offsets()

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
        self._calibration_samples: deque[tuple[int, PicoPacket]] = deque()
        self._last_calibration_packet_timestamp_ns: int | None = None
        self._arm_held_counts = {"left": 0, "right": 0}
        self._hand_configs = self._load_hand_configs()
        self._last_control_time = time.perf_counter()
        self._loop_samples_ms: deque[float] = deque(maxlen=250)
        self._dry_run_plant = (
            PositionTargetPlant(
                float(self.get_parameter("dry_run_max_joint_velocity_rad_s").value),
                float(self.get_parameter("dry_run_max_joint_acceleration_rad_s2").value),
            )
            if bool(self.get_parameter("dry_run_simulated_feedback_enabled").value)
            else None
        )

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
        self.declare_parameter("pink_qp_solver", "proxqp")
        self.declare_parameter("pink_self_collision_enabled", True)
        self.declare_parameter("pink_self_collision_components", ["left_arm", "right_arm"])
        self.declare_parameter("pink_self_collision_srdf_path", "")
        self.declare_parameter("pink_self_collision_urdf_path", "")
        self.declare_parameter("pink_self_collision_max_pairs", 24)
        self.declare_parameter("pink_self_collision_min_distance", 0.04)
        self.declare_parameter("pink_self_collision_gain", 6.0)
        self.declare_parameter("pink_self_collision_safe_displacement_gain", 0.0)
        self.declare_parameter("pink_collision_pipeline", "reduced_all_pairs")
        self.declare_parameter("pink_collision_sphere_count", 18)
        self.declare_parameter("pink_collision_sphere_inflation", 1.0)
        self.declare_parameter("pink_velocity_limit_enabled", True)
        self.declare_parameter("pink_task_gain", 1.0)
        self.declare_parameter("pink_lm_damping", 1.0e-6)
        self.declare_parameter("pink_solve_damping", 1.0e-8)
        self.declare_parameter("pink_torso_max_iterations", 25)
        self.declare_parameter("pink_head_max_iterations", 8)
        self.declare_parameter("pink_arm_max_iterations", 20)
        self.declare_parameter("pink_self_collision_arm_max_iterations", 2)
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
        self.declare_parameter("left_controller_to_hand_point_xyz_m", [0.0, 0.0, 0.0])
        self.declare_parameter("right_controller_to_hand_point_xyz_m", [0.0, 0.0, 0.0])
        self.declare_parameter("calibration_sample_window_s", 0.4)
        self.declare_parameter("calibration_min_samples", 10)
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
        self.declare_parameter("dry_run_simulated_feedback_enabled", False)
        self.declare_parameter("dry_run_max_joint_velocity_rad_s", 1.5)
        self.declare_parameter("dry_run_max_joint_acceleration_rad_s2", 6.0)
        self.declare_parameter("left_hand_joint_names", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("right_hand_joint_names", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("left_hand_open_positions", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("left_hand_closed_positions", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("right_hand_open_positions", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("right_hand_closed_positions", Parameter.Type.DOUBLE_ARRAY)

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
                collision_arm_max_iterations=int(
                    self.get_parameter("pink_self_collision_arm_max_iterations").value
                ),
                arm_position_cost=float(self.get_parameter("pink_arm_position_cost").value),
                arm_orientation_cost=float(
                    self.get_parameter("pink_arm_orientation_cost").value
                ),
                collision_pipeline=str(
                    self.get_parameter("pink_collision_pipeline").value
                ),
                collision_sphere_count=int(
                    self.get_parameter("pink_collision_sphere_count").value
                ),
                collision_sphere_inflation=float(
                    self.get_parameter("pink_collision_sphere_inflation").value
                ),
            )
            self.get_logger().info(
                f"Using Pinocchio/Pink IK backend with solver '{solver}' and URDF {urdf_path}"
            )
            if self_collision_components:
                self.get_logger().info(
                    "Pink bimanual self-collision barrier enabled using SRDF "
                    f"{self._self_collision_srdf_path()} and collision URDF "
                    f"{self._self_collision_urdf_path()}"
                )
                self.get_logger().info(
                    "Pink collision pipeline="
                    f"{kin.arms.collision_pipeline}, geometries="
                    f"{kin.arms.collision_geometry_count}, pairs="
                    f"{kin.arms.collision_pair_count}, barrier rows="
                    f"{kin.arms.barrier_pair_count}"
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
        try:
            packet, sample_stats = self._averaged_calibration_packet()
        except ValueError as exc:
            return False, str(exc)

        torso_q = self._current_q(TORSO_JOINTS, prefer_command=False)
        head_q = self._current_q(HEAD_JOINTS, prefer_command=False)
        left_q = clip_arm_j4_upper_limit(
            self._current_q(LEFT_ARM_JOINTS, prefer_command=False)
        )
        right_q = clip_arm_j4_upper_limit(
            self._current_q(RIGHT_ARM_JOINTS, prefer_command=False)
        )
        arm_center_pos, arm_center_rot = self._kin.arm_center_pose(torso_q)
        _head_pos, head_rot = self._kin.head.forward(head_q)
        arm_end_effector_rotations = {
            "left": self._kin.left_arm.forward(left_q)[1],
            "right": self._kin.right_arm.forward(right_q)[1],
        }
        self._calibration.calibrate(
            packet,
            arm_center_pos,
            arm_center_rot,
            head_rot,
            arm_end_effector_rotations,
            controller_to_hand_offsets=self._controller_to_hand_offsets,
        )
        self._seed_command_state("torso", TORSO_JOINTS, torso_q)
        self._seed_command_state("head", HEAD_JOINTS, head_q)
        self._seed_command_state("left_arm", LEFT_ARM_JOINTS, left_q)
        self._seed_command_state("right_arm", RIGHT_ARM_JOINTS, right_q)
        self._limiters["base"].reset(np.zeros(3, dtype=np.float64))
        self._status["calibration_sample_count"] = sample_stats.sample_count
        self._status["calibration_controller_dispersion_m"] = (
            sample_stats.controller_position_dispersion_m.copy()
        )
        return (
            True,
            "neutral calibrated from "
            f"{sample_stats.sample_count} samples: neutral_height_signal="
            f"{self._calibration.neutral_height_signal:.3f} m; "
            "keep elbows straight and extend both arms forward before B",
        )

    def _calibrate_reach_from_latest_packet(self) -> tuple[bool, str]:
        if not self._calibration.calibrated:
            return False, "neutral calibration must be completed first"
        try:
            packet, sample_stats = self._averaged_calibration_packet()
        except ValueError as exc:
            return False, str(exc)

        minimum = float(self.get_parameter("operator_arm_length_min_m").value)
        maximum = float(self.get_parameter("operator_arm_length_max_m").value)
        head_pose = self._calibration.to_operator_pose(packet.head)
        lengths: dict[str, float] = {}
        shoulder_offsets: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            controller_pose = self._calibration.to_operator_pose(packet.controllers[side].pose)
            hand_point = controller_hand_point(
                controller_pose.position,
                controller_pose.rotation,
                self._controller_to_hand_offsets[side],
            )
            neutral_hand = self._calibration.neutral_hand_relative_to_head.get(side)
            if neutral_hand is None:
                return False, f"neutral calibration is missing the {side} hand reference"
            try:
                fitted = fit_two_pose_shoulder_reach(
                    side,
                    neutral_hand,
                    hand_point - head_pose.position,
                    float(self.get_parameter("operator_shoulder_width_m").value),
                    float(self.get_parameter("operator_head_to_shoulder_z_m").value),
                    minimum,
                    maximum,
                )
            except ValueError as exc:
                return False, str(exc)
            shoulder_offsets[side] = fitted.shoulder_offset
            lengths[side] = fitted.arm_length

        if abs(lengths["left"] - lengths["right"]) > 0.12:
            return (
                False,
                "left/right fitted arm lengths differ by more than 0.12 m: "
                f"left={lengths['left']:.3f}, right={lengths['right']:.3f}",
            )

        self._calibration.set_operator_reach_calibration(shoulder_offsets, lengths)
        self._status["operator_arm_lengths"] = lengths.copy()
        self._status["operator_shoulder_offsets"] = {
            side: value.tolist() for side, value in shoulder_offsets.items()
        }
        self._status["calibration_sample_count"] = sample_stats.sample_count
        self._status["calibration_controller_dispersion_m"] = (
            sample_stats.controller_position_dispersion_m.copy()
        )
        return (
            True,
            "reach calibrated from "
            f"{sample_stats.sample_count} samples: "
            f"left={lengths['left']:.3f} m, right={lengths['right']:.3f} m",
        )

    def _averaged_calibration_packet(self):
        if self._latest_packet is None:
            raise ValueError("no Pico packet received yet")
        now_ns = self.get_clock().now().nanoseconds
        window_ns = int(
            max(0.0, float(self.get_parameter("calibration_sample_window_s").value))
            * 1.0e9
        )
        packets = [
            packet
            for received_ns, packet in self._calibration_samples
            if now_ns - received_ns <= window_ns
        ]
        minimum_samples = int(self.get_parameter("calibration_min_samples").value)
        if len(packets) < minimum_samples:
            raise ValueError(
                "not enough fresh calibration samples: "
                f"need {minimum_samples}, have {len(packets)}"
            )
        averaged, stats = average_calibration_packets(packets)
        moving = {
            side: value
            for side, value in stats.controller_position_dispersion_m.items()
            if value > 0.02
        }
        if moving:
            details = ", ".join(f"{side}={value:.3f} m" for side, value in moving.items())
            raise ValueError(f"hold controllers still during calibration ({details})")
        return averaged, stats

    def _seed_command_state(
        self,
        component: str,
        names: tuple[str, ...],
        positions: np.ndarray,
    ) -> None:
        values = np.asarray(positions, dtype=np.float64).reshape(len(names))
        if component in {"left_arm", "right_arm"}:
            values = clip_arm_j4_upper_limit(values)
        for name, value in zip(names, values):
            self._command_positions[name] = float(value)
        if self._dry_run_plant is not None:
            self._dry_run_plant.seed(names, values)
        self._limiters[component].reset(values)

    def _on_timer(self) -> None:
        loop_start = time.perf_counter()
        control_dt = loop_start - self._last_control_time
        self._last_control_time = loop_start
        nominal_dt = 1.0 / float(self.get_parameter("control_rate_hz").value)
        if not np.isfinite(control_dt) or control_dt <= 0.0:
            control_dt = nominal_dt
        if self._dry_run_plant is not None:
            self._dry_run_plant.advance(control_dt)
        self._status["control_dt_ms"] = control_dt * 1000.0
        packet = self._drain_packet()
        now_ns = self.get_clock().now().nanoseconds
        stale = self._packet_is_stale(now_ns)

        if packet is not None and not stale:
            self._update_hand_input_status(packet)
            self._handle_button_clicks(packet)

        if stale or not self._enabled or self._hold or not self._calibration.calibrated:
            self._publish_zero_base()
            self._publish_status(stale=stale)
            return

        assert packet is not None
        timings_ms: dict[str, float] = {}
        torso_feedback_q = self._current_q(TORSO_JOINTS)
        stage_start = time.perf_counter()
        torso_q = self._update_torso(packet)
        timings_ms["torso"] = _elapsed_ms(stage_start)
        _arm_center_position, arm_center_rotation = self._kin.arm_center_pose(torso_feedback_q)
        stage_start = time.perf_counter()
        head_q = self._update_head(packet)
        timings_ms["head"] = _elapsed_ms(stage_start)
        stage_start = time.perf_counter()
        left_q, right_q = self._update_arms(
            packet,
            torso_feedback_q,
            arm_center_rotation,
            control_dt,
        )
        timings_ms["arms"] = _elapsed_ms(stage_start)
        stage_start = time.perf_counter()
        hand_targets = self._update_hands(packet)
        timings_ms["hands"] = _elapsed_ms(stage_start)
        stage_start = time.perf_counter()
        self._update_base(packet)
        timings_ms["base"] = _elapsed_ms(stage_start)
        timings_ms["loop"] = _elapsed_ms(loop_start)
        self._loop_samples_ms.append(timings_ms["loop"])
        self._status["loop_ms"] = timings_ms["loop"]
        self._status["loop_p50_ms"] = _percentile(self._loop_samples_ms, 50.0)
        self._status["loop_p95_ms"] = _percentile(self._loop_samples_ms, 95.0)
        self._status["loop_p99_ms"] = _percentile(self._loop_samples_ms, 99.0)
        if hasattr(self._kin, "collision_diagnostics"):
            self._status.update(self._kin.collision_diagnostics())
        self._publish_log_frame(
            packet,
            torso_q,
            head_q,
            left_q,
            right_q,
            hand_targets,
            timings_ms,
        )
        self._publish_status(stale=False)

    def _drain_packet(self) -> PicoPacket | None:
        if self._receiver is not None:
            packets = self._receiver.get_available()
            if packets:
                received_ns = self.get_clock().now().nanoseconds
                self._latest_packet = packets[-1]
                self._latest_packet_rx_ns = received_ns
                for packet in packets:
                    self._record_calibration_packet(packet, received_ns)
        return self._latest_packet

    def _record_calibration_packet(self, packet: PicoPacket, received_ns: int) -> None:
        if packet.timestamp_ns == self._last_calibration_packet_timestamp_ns:
            return
        self._last_calibration_packet_timestamp_ns = packet.timestamp_ns
        self._calibration_samples.append((int(received_ns), packet))
        retention_ns = int(
            max(1.0, float(self.get_parameter("calibration_sample_window_s").value) * 2.0)
            * 1.0e9
        )
        cutoff = int(received_ns) - retention_ns
        while self._calibration_samples and self._calibration_samples[0][0] < cutoff:
            self._calibration_samples.popleft()

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
        log_message = f"Pico {side} {button.upper()} click {action}: {message}"
        if success:
            self.get_logger().info(log_message)
        else:
            self.get_logger().warn(log_message)

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

    def _build_arm_target(
        self,
        side: str,
        packet: PicoPacket,
        arm_center_rotation: np.ndarray,
    ) -> dict[str, object]:
        joint_names = LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS
        current = clip_arm_j4_upper_limit(self._current_q(joint_names))
        controller = packet.controllers[side]

        controller_pose = self._calibration.to_operator_pose(controller.pose)
        head_pose = self._calibration.to_operator_pose(packet.head)
        hand_point = controller_hand_point(
            controller_pose.position,
            controller_pose.rotation,
            self._controller_to_hand_offsets[side],
        )
        shoulder_offset = self._calibration.operator_shoulder_offsets.get(side)
        if shoulder_offset is None:
            shoulder_offset = operator_shoulder_position(
                side,
                np.zeros(3, dtype=np.float64),
                float(self.get_parameter("operator_shoulder_width_m").value),
                float(self.get_parameter("operator_head_to_shoulder_z_m").value),
                float(self.get_parameter("operator_shoulder_x_m").value),
            )
        operator_shoulder = operator_shoulder_from_offset(
            head_pose.position,
            shoulder_offset,
        )
        operator_vector = hand_point - operator_shoulder
        arm_center_vector = vector_in_arm_center(operator_vector, arm_center_rotation)
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
        reach = normalized_reach_target_from_vector(
            arm_center_vector,
            operator_arm_length.value,
            robot_shoulder,
            float(self.get_parameter("robot_arm_reach_m").value),
        )
        target_pos = reach.position
        target_rot = self._calibration.arm_target_rotation(
            side,
            controller_pose,
            arm_center_rotation,
        )
        return {
            "joint_names": joint_names,
            "current": current,
            "target_position": target_pos,
            "target_rotation": target_rot,
            "reach_fraction": reach.fraction,
            "operator_arm_length": operator_arm_length.value,
            "operator_arm_length_source": operator_arm_length.source,
            "operator_shoulder": operator_shoulder,
            "controller_position": controller_pose.position,
            "controller_rotation": controller_pose.rotation,
            "hand_point_position": hand_point,
            "operator_vector": operator_vector,
            "arm_center_vector": arm_center_vector,
            "arm_center_rotation": arm_center_rotation,
            "operator_shoulder_offset": shoulder_offset,
            "robot_shoulder": robot_shoulder,
        }

    def _update_arms(
        self,
        packet: PicoPacket,
        torso_feedback_q: np.ndarray,
        arm_center_rotation: np.ndarray,
        control_dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = {
            side: self._build_arm_target(side, packet, arm_center_rotation)
            for side in ("left", "right")
        }
        bimanual_solver = getattr(self._kin, "solve_bimanual_arm_poses", None)
        if bimanual_solver is None:
            solutions = {
                side: self._kin.solve_arm_pose(
                    side,
                    targets[side]["current"],
                    targets[side]["target_position"],
                    targets[side]["target_rotation"],
                )
                for side in ("left", "right")
            }
            self._status["arm_ik_mode"] = "independent_numeric"
        else:
            solutions = bimanual_solver(
                torso_feedback_q,
                targets["left"]["current"],
                targets["right"]["current"],
                targets["left"]["target_position"],
                targets["left"]["target_rotation"],
                targets["right"]["target_position"],
                targets["right"]["target_rotation"],
                dt=control_dt,
            )
            self._status["arm_ik_mode"] = "bimanual_one_step"
        left = self._publish_arm_solution("left", targets["left"], solutions["left"])
        right = self._publish_arm_solution("right", targets["right"], solutions["right"])
        return left, right

    def _update_arm(
        self,
        side: str,
        packet: PicoPacket,
        arm_center_rotation: np.ndarray,
    ) -> np.ndarray:
        """Compatibility helper for callers that still request one arm."""
        target = self._build_arm_target(side, packet, arm_center_rotation)
        solution = self._kin.solve_arm_pose(
            side,
            target["current"],
            target["target_position"],
            target["target_rotation"],
        )
        return self._publish_arm_solution(side, target, solution)

    def _publish_arm_solution(
        self,
        side: str,
        target_info: dict[str, object],
        solution,
    ) -> np.ndarray:
        joint_names = target_info["joint_names"]
        current = np.asarray(target_info["current"], dtype=np.float64)
        acceptance = assess_arm_ik_solution(solution)
        if acceptance.accepted:
            target = clip_arm_j4_upper_limit(solution.q)
        else:
            target = current.copy()
            self._arm_held_counts[side] += 1
        rate_limited = self._limiters[f"{side}_arm"].limit(target)
        limited = clip_arm_j4_upper_limit(rate_limited)
        if not np.array_equal(limited, rate_limited):
            self._limiters[f"{side}_arm"].reset(limited)
        self._publish_joint_command(f"{side}_arm", joint_names, limited)
        self._status[f"{side}_arm_error"] = solution.error_norm
        self._status[f"{side}_arm_iterations"] = solution.iterations
        self._status[f"{side}_arm_reach_fraction"] = target_info["reach_fraction"]
        self._status[f"{side}_arm_length_source"] = target_info["operator_arm_length_source"]
        self._status[f"{side}_arm_ik_termination"] = solution.termination
        self._status[f"{side}_arm_ik_acceptance"] = acceptance.mode
        self._status[f"{side}_arm_ik_initial_error"] = solution.initial_error_norm
        self._status[f"{side}_arm_ik_position_error"] = solution.position_error_norm
        self._status[f"{side}_arm_ik_orientation_error"] = (
            solution.orientation_error_norm
        )
        self._status[f"{side}_arm_held_command_count"] = self._arm_held_counts[side]
        self._retarget_debug[side] = {
            "operator_shoulder": np.asarray(target_info["operator_shoulder"]).tolist(),
            "controller_position": np.asarray(target_info["controller_position"]).tolist(),
            "controller_rotation": np.asarray(target_info["controller_rotation"]).tolist(),
            "controller_to_hand_point_offset": self._controller_to_hand_offsets[
                side
            ].tolist(),
            "hand_point_position": np.asarray(target_info["hand_point_position"]).tolist(),
            "operator_vector": np.asarray(target_info["operator_vector"]).tolist(),
            "arm_center_vector": np.asarray(target_info["arm_center_vector"]).tolist(),
            "arm_center_from_operator_rotation": np.asarray(
                target_info["arm_center_rotation"],
                dtype=np.float64,
            ).reshape(3, 3).T.tolist(),
            "operator_shoulder_offset": np.asarray(
                target_info["operator_shoulder_offset"],
                dtype=np.float64,
            ).reshape(3).tolist(),
            "operator_arm_length": target_info["operator_arm_length"],
            "operator_arm_length_source": target_info["operator_arm_length_source"],
            "reach_fraction": target_info["reach_fraction"],
            "robot_shoulder": np.asarray(target_info["robot_shoulder"]).tolist(),
            "robot_target": np.asarray(target_info["target_position"]).tolist(),
            "robot_target_rotation": np.asarray(target_info["target_rotation"]).tolist(),
            "ik_error": solution.error_norm,
            "ik_iterations": solution.iterations,
            "ik_termination": solution.termination,
            "ik_acceptance": acceptance.mode,
            "ik_initial_error": solution.initial_error_norm,
            "ik_initial_position_error": solution.initial_position_error_norm,
            "ik_initial_orientation_error": solution.initial_orientation_error_norm,
            "ik_position_error": solution.position_error_norm,
            "ik_orientation_error": solution.orientation_error_norm,
            "held_command_count": self._arm_held_counts[side],
            "ik_integrated_target": target.tolist(),
            "published_target": limited.tolist(),
            "feedback_posture": self._current_q(joint_names).tolist(),
        }
        return limited

    def _update_hands(self, packet: PicoPacket) -> dict[str, np.ndarray]:
        targets: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            config = self._hand_configs[side]
            if config is None:
                continue
            controller = packet.controllers[side]
            target = retarget_f5d6_hand(config, controller.trigger, controller.grip)
            self._publish_joint_command(f"{side}_hand", config.joint_names, target)
            targets[side] = target
        return targets

    def _update_hand_input_status(self, packet: PicoPacket) -> None:
        for side in ("left", "right"):
            controller = packet.controllers[side]
            self._status[f"{side}_hand_trigger"] = controller.trigger
            self._status[f"{side}_hand_grip"] = controller.grip

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
        command_values = np.asarray(positions, dtype=np.float64).reshape(-1)
        if component in {"left_arm", "right_arm"}:
            command_values = clip_arm_j4_upper_limit(command_values)
        values = [
            float(value)
            for value in command_values
        ]
        for name, value in zip(names_list, values):
            self._command_positions[name] = value
        if self._dry_run_plant is not None:
            self._dry_run_plant.set_target(tuple(names_list), command_values)
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
        hand_targets: dict[str, np.ndarray],
        timings_ms: dict[str, float],
    ) -> None:
        displayed_torso = self._current_q(TORSO_JOINTS)
        displayed_head = self._current_q(HEAD_JOINTS)
        displayed_left = clip_arm_j4_upper_limit(self._current_q(LEFT_ARM_JOINTS))
        displayed_right = clip_arm_j4_upper_limit(self._current_q(RIGHT_ARM_JOINTS))
        msg = String()
        msg.data = json.dumps(
            make_log_frame_payload(
                packet.timestamp_ns,
                packet.sequence,
                displayed_torso,
                displayed_head,
                displayed_left,
                displayed_right,
                left_hand_q=hand_targets.get("left"),
                right_hand_q=hand_targets.get("right"),
                debug={
                    "retarget": self._retarget_debug,
                    "timing_ms": timings_ms,
                    "commanded": {
                        "torso": np.asarray(torso_q).tolist(),
                        "head": np.asarray(head_q).tolist(),
                        "left_arm": np.asarray(left_q).tolist(),
                        "right_arm": np.asarray(right_q).tolist(),
                    },
                    "displayed_feedback": {
                        "torso": displayed_torso.tolist(),
                        "head": displayed_head.tolist(),
                        "left_arm": displayed_left.tolist(),
                        "right_arm": displayed_right.tolist(),
                    },
                },
            ),
            sort_keys=True,
        )
        self._log_frame_pub.publish(msg)

    def _current_q(self, names: tuple[str, ...], prefer_command: bool = False) -> np.ndarray:
        feedback_or_command = joint_values(
            names,
            self._joint_positions,
            self._command_positions,
            prefer_command=prefer_command,
        )
        if self._dry_run_plant is not None:
            return self._dry_run_plant.values(names, feedback_or_command)
        return feedback_or_command

    def _load_controller_to_hand_offsets(self) -> dict[str, np.ndarray]:
        offsets: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            parameter_name = f"{side}_controller_to_hand_point_xyz_m"
            values = np.asarray(
                self._float_list_parameter(parameter_name),
                dtype=np.float64,
            ).reshape(-1)
            if values.size != 3:
                raise ValueError(f"{parameter_name} must contain exactly 3 values")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{parameter_name} must contain finite values")
            offsets[side] = values.copy()

        sample_window = float(self.get_parameter("calibration_sample_window_s").value)
        minimum_samples = int(self.get_parameter("calibration_min_samples").value)
        if sample_window <= 0.0:
            raise ValueError("calibration_sample_window_s must be positive")
        if minimum_samples <= 0:
            raise ValueError("calibration_min_samples must be positive")
        return offsets

    def _load_hand_configs(self) -> dict[str, F5D6HandConfig | None]:
        configs: dict[str, F5D6HandConfig | None] = {}
        for side in ("left", "right"):
            self._status[f"{side}_hand_trigger"] = 0.0
            self._status[f"{side}_hand_grip"] = 0.0
            try:
                config = F5D6HandConfig.from_values(
                    side,
                    self._string_list_parameter(f"{side}_hand_joint_names"),
                    self._float_list_parameter(f"{side}_hand_open_positions"),
                    self._float_list_parameter(f"{side}_hand_closed_positions"),
                )
            except (TypeError, ValueError) as exc:
                configs[side] = None
                self._status[f"{side}_hand_config_valid"] = False
                self._status[f"{side}_hand_config_error"] = str(exc)
                self.get_logger().warn(
                    f"{side.capitalize()} F5D6 hand commands disabled: {exc}"
                )
                continue
            configs[side] = config
            self._status[f"{side}_hand_config_valid"] = True
            self._status.pop(f"{side}_hand_config_error", None)
        return configs

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


def _percentile(values: deque[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


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
