#!/usr/bin/env python3
"""ROS 2 bridge for Dexmate robots through the dexcontrol Python API."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Final

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Quaternion, TransformStamped, Twist, WrenchStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2, PointField
from std_srvs.srv import SetBool
from tf2_ros import TransformBroadcaster

from dexcontrol import Robot
from dexcontrol.core.config import get_robot_config


STATE_COMPONENTS: Final[tuple[str, ...]] = (
    "left_arm",
    "right_arm",
    "torso",
    "head",
    "left_hand",
    "right_hand",
)
COMMAND_COMPONENTS: Final[tuple[str, ...]] = (
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "torso",
    "head",
)
CHASSIS_COMPONENT: Final[str] = "chassis"


class DexcontrolBridge(Node):
    """Expose dexcontrol robot state, commands, sensors, odometry, and e-stop to ROS."""

    def __init__(self) -> None:
        super().__init__("dexcontrol_bridge")
        self._declare_parameters()

        self._lock = threading.Lock()
        self._last_warn_s: dict[str, float] = {}
        self._last_run_ns: dict[str, int] = {}
        self._lidar_status: dict[str, dict[str, Any]] = {}
        self._last_cmd_vel_time: Time | None = None
        self._cmd_vel = np.zeros(3, dtype=np.float64)
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0
        self._last_odom_time: Time | None = None
        self._estop_active = False

        configs = get_robot_config()
        self._enable_requested_sensors(configs)
        self.robot = Robot(configs=configs, auto_shutdown=False)

        self._state_components: dict[str, Any] = {}
        self._command_components: dict[str, Any] = {}
        self._component_joint_names: dict[str, list[str]] = {}
        self._joint_to_component: dict[str, tuple[str, int]] = {}
        self._joint_targets: dict[str, np.ndarray] = {}
        self._joint_limits: dict[str, np.ndarray | None] = {}

        self._discover_components()
        self._initialize_command_targets()
        self._discover_sensors()

        qos_depth = int(self.get_parameter("qos_depth").value)
        self.joint_state_pub = self.create_publisher(JointState, "joint_states", qos_depth)
        self.joint_feedback_pub = self.create_publisher(
            DiagnosticArray, "dexcontrol/joint_feedback", qos_depth
        )
        self.lidar_feedback_pub = self.create_publisher(
            DiagnosticArray, "dexcontrol/lidar_feedback", qos_depth
        )
        self.odom_pub = self.create_publisher(Odometry, "odom", qos_depth)
        self.pointcloud_pubs: dict[str, Any] = {}
        for sensor_name in self._lidar_3d_sensors:
            self.pointcloud_pubs[sensor_name] = self.create_publisher(
                PointCloud2, f"{sensor_name}/points", qos_depth
            )

        self.wrench_pubs = {
            "left": self.create_publisher(
                WrenchStamped, "left_arm/ft_sensor/wrench", qos_depth
            ),
            "right": self.create_publisher(
                WrenchStamped, "right_arm/ft_sensor/wrench", qos_depth
            ),
        }

        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(JointState, "joint_commands", self._on_joint_command, qos_depth)
        for component_name in self._command_components:
            self.create_subscription(
                JointState,
                f"{component_name}/joint_commands",
                lambda msg, name=component_name: self._on_component_joint_command(
                    name, msg
                ),
                qos_depth,
            )
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, qos_depth)
        self.create_service(SetBool, "soft_estop", self._on_soft_estop)

        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        if control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be positive")
        self._control_timer = self.create_timer(1.0 / control_rate_hz, self._on_timer)

        self.get_logger().info(
            "Dexcontrol ROS bridge started at "
            f"{control_rate_hz:.1f} Hz with command components: "
            f"{', '.join(self._command_components) or 'none'}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("control_rate_hz", 250.0)
        self.declare_parameter("state_publish_rate_hz", 100.0)
        self.declare_parameter("diagnostics_publish_rate_hz", 20.0)
        self.declare_parameter("wrench_publish_rate_hz", 100.0)
        self.declare_parameter("pointcloud_publish_rate_hz", 10.0)
        self.declare_parameter("pointcloud_debug_log_rate_hz", 0.0)
        self.declare_parameter("odom_publish_rate_hz", 50.0)
        self.declare_parameter("tf_publish_rate_hz", 50.0)
        self.declare_parameter("estop_poll_rate_hz", 10.0)
        self.declare_parameter("qos_depth", 10)
        self.declare_parameter("enable_joint_commands", True)
        self.declare_parameter("enable_cmd_vel", True)
        self.declare_parameter("cmd_vel_timeout_s", 0.5)
        self.declare_parameter("include_chassis_joint_states", True)
        self.declare_parameter("use_robot_timestamps", True)
        self.declare_parameter("use_measured_chassis_for_odom", True)
        self.declare_parameter("lidar_3d_sensors", ["lidar_3d_front", "lidar_3d_back"])
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("lidar_3d_front_frame", "front_lidar")
        self.declare_parameter("lidar_3d_back_frame", "back_lidar")
        self.declare_parameter("left_ft_frame", "L_ee")
        self.declare_parameter("right_ft_frame", "R_ee")

    def _enable_requested_sensors(self, configs: Any) -> None:
        lidar_3d_sensors = list(self.get_parameter("lidar_3d_sensors").value)
        for sensor_name in lidar_3d_sensors:
            self._enable_sensor_if_available(configs, str(sensor_name))

    def _enable_sensor_if_available(self, configs: Any, sensor_name: str) -> None:
        try:
            if configs.has_sensor(sensor_name):
                configs.enable_sensor(sensor_name)
            else:
                self.get_logger().debug(
                    f"Sensor '{sensor_name}' is not present in this robot config"
                )
        except Exception as exc:
            self._warn_throttled(
                f"enable_sensor_{sensor_name}",
                f"Could not enable sensor '{sensor_name}': {exc}",
            )

    def _discover_components(self) -> None:
        state_names = list(STATE_COMPONENTS)
        if bool(self.get_parameter("include_chassis_joint_states").value):
            state_names.append(CHASSIS_COMPONENT)

        for component_name in state_names:
            component = self._get_robot_component(component_name)
            if component is None:
                continue
            self._state_components[component_name] = component
            if hasattr(component, "joint_name"):
                self._component_joint_names[component_name] = list(component.joint_name)

        for component_name in COMMAND_COMPONENTS:
            component = self._get_robot_component(component_name)
            if component is None or not hasattr(component, "joint_name"):
                continue
            joint_names = list(component.joint_name)
            self._command_components[component_name] = component
            self._component_joint_names[component_name] = joint_names
            for joint_index, joint_name in enumerate(joint_names):
                self._joint_to_component[joint_name] = (component_name, joint_index)

    def _get_robot_component(self, component_name: str) -> Any | None:
        try:
            if component_name in {"left_hand", "right_hand"}:
                side = component_name.split("_", maxsplit=1)[0]
                if not self.robot.have_hand(side):  # type: ignore[arg-type]
                    return None
            elif not self.robot.has_component(component_name):
                return None
            return getattr(self.robot, component_name)
        except Exception as exc:
            self._warn_throttled(
                f"component_{component_name}",
                f"Component '{component_name}' is unavailable: {exc}",
            )
            return None

    def _initialize_command_targets(self) -> None:
        for component_name, component in self._command_components.items():
            try:
                target = np.asarray(component.get_joint_pos(), dtype=np.float64)
            except Exception as exc:
                self._warn_throttled(
                    f"target_init_{component_name}",
                    f"Could not initialize target for '{component_name}': {exc}",
                )
                continue
            self._joint_targets[component_name] = target.copy()
            self._joint_limits[component_name] = self._safe_joint_limits(component)

    def _safe_joint_limits(self, component: Any) -> np.ndarray | None:
        try:
            limits = component.joint_pos_limit
        except Exception:
            return None
        if limits is None:
            return None
        limits_array = np.asarray(limits, dtype=np.float64)
        if limits_array.ndim != 2 or limits_array.shape[1] != 2:
            return None
        return limits_array

    def _discover_sensors(self) -> None:
        self._lidar_3d_sensors: dict[str, Any] = {}
        for sensor_name in list(self.get_parameter("lidar_3d_sensors").value):
            sensor = self._get_sensor(str(sensor_name))
            if sensor is not None:
                self._lidar_3d_sensors[str(sensor_name)] = sensor
                self._lidar_status[str(sensor_name)] = self._initial_lidar_status()

    def _initial_lidar_status(self) -> dict[str, Any]:
        return {
            "status": "discovered",
            "attempt_count": 0,
            "publish_count": 0,
            "same_sequence_count": 0,
            "point_count": 0,
            "sequence": None,
            "timestamp_ns": None,
            "read_duration_ms": 0.0,
            "last_attempt_ns": None,
            "last_publish_ns": None,
            "last_debug_ns": None,
            "message": "",
        }

    def _get_sensor(self, sensor_name: str) -> Any | None:
        try:
            if self.robot.has_sensor(sensor_name):
                return getattr(self.robot.sensors, sensor_name)
        except Exception as exc:
            self._warn_throttled(
                f"sensor_{sensor_name}",
                f"Sensor '{sensor_name}' is unavailable: {exc}",
            )
        return None

    def _on_joint_command(self, msg: JointState) -> None:
        if msg.name:
            self._apply_named_joint_command(msg)
            return

        command_order = [
            joint
            for component_name in self._command_components
            for joint in self._component_joint_names.get(component_name, [])
        ]
        if len(msg.position) != len(command_order):
            self._warn_throttled(
                "joint_command_size",
                "Unnamed joint_commands must contain exactly "
                f"{len(command_order)} positions in bridge command order",
            )
            return
        named_msg = JointState()
        named_msg.name = command_order
        named_msg.position = msg.position
        self._apply_named_joint_command(named_msg)

    def _on_component_joint_command(self, component_name: str, msg: JointState) -> None:
        joint_names = self._component_joint_names.get(component_name, [])
        if not joint_names:
            return

        if msg.name:
            renamed = JointState()
            renamed.name = list(msg.name)
            renamed.position = msg.position
            self._apply_named_joint_command(renamed, expected_component=component_name)
            return

        if len(msg.position) != len(joint_names):
            self._warn_throttled(
                f"{component_name}_joint_command_size",
                f"{component_name}/joint_commands expected {len(joint_names)} positions",
            )
            return

        renamed = JointState()
        renamed.name = joint_names
        renamed.position = msg.position
        self._apply_named_joint_command(renamed, expected_component=component_name)

    def _apply_named_joint_command(
        self, msg: JointState, expected_component: str | None = None
    ) -> None:
        if not msg.position:
            self._warn_throttled("empty_joint_command", "Ignoring joint command with no positions")
            return

        updates: dict[str, list[tuple[int, float]]] = {}
        for index, joint_name in enumerate(msg.name):
            if index >= len(msg.position):
                break
            mapping = self._joint_to_component.get(joint_name)
            if mapping is None:
                self._warn_throttled(
                    f"unknown_joint_{joint_name}",
                    f"Ignoring unknown commanded joint '{joint_name}'",
                )
                continue
            component_name, joint_index = mapping
            if expected_component is not None and component_name != expected_component:
                self._warn_throttled(
                    f"wrong_component_{joint_name}",
                    f"Ignoring joint '{joint_name}' on {expected_component} topic",
                )
                continue
            updates.setdefault(component_name, []).append(
                (joint_index, float(msg.position[index]))
            )

        with self._lock:
            for component_name, component_updates in updates.items():
                self._update_joint_target(component_name, component_updates)

    def _update_joint_target(
        self, component_name: str, component_updates: list[tuple[int, float]]
    ) -> None:
        target = self._joint_targets.get(component_name)
        if target is None:
            return

        limits = self._joint_limits.get(component_name)
        for joint_index, value in component_updates:
            if not math.isfinite(value):
                self._warn_throttled(
                    f"nonfinite_{component_name}_{joint_index}",
                    f"Ignoring non-finite command for {component_name}[{joint_index}]",
                )
                continue

            command_value = value
            if limits is not None and joint_index < len(limits):
                lower, upper = limits[joint_index]
                if command_value < lower or command_value > upper:
                    self._warn_throttled(
                        f"joint_limit_{component_name}_{joint_index}",
                        f"Command for {component_name}[{joint_index}]={command_value:.4f} "
                        f"is outside [{lower:.4f}, {upper:.4f}]; limiting command",
                    )
                    command_value = float(np.clip(command_value, lower, upper))
            target[joint_index] = command_value

    def _on_cmd_vel(self, msg: Twist) -> None:
        vx = float(msg.linear.x)
        vy = float(msg.linear.y)
        wz = float(msg.angular.z)
        if not all(math.isfinite(value) for value in (vx, vy, wz)):
            self._warn_throttled("cmd_vel_nonfinite", "Ignoring non-finite cmd_vel")
            return

        chassis = self._get_robot_component(CHASSIS_COMPONENT)
        if chassis is not None:
            max_lin = float(getattr(chassis, "max_lin_vel", float("inf")))
            max_ang = float(getattr(chassis, "max_ang_vel", float("inf")))
            clipped_vx = float(np.clip(vx, -max_lin, max_lin))
            clipped_vy = float(np.clip(vy, -max_lin, max_lin))
            clipped_wz = float(np.clip(wz, -max_ang, max_ang))
            if (clipped_vx, clipped_vy, clipped_wz) != (vx, vy, wz):
                self._warn_throttled(
                    "cmd_vel_limit",
                    "cmd_vel exceeds chassis velocity limits; limiting command",
                )
            vx, vy, wz = clipped_vx, clipped_vy, clipped_wz

        with self._lock:
            self._cmd_vel[:] = (vx, vy, wz)
            self._last_cmd_vel_time = self.get_clock().now()

    def _on_soft_estop(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        estop = getattr(self.robot, "estop", None)
        if estop is None:
            response.success = False
            response.message = "Robot does not expose an estop component"
            return response

        try:
            if request.data:
                estop.activate()
            else:
                estop.deactivate()
            self._estop_active = bool(request.data)
            response.success = True
            response.message = (
                "Software e-stop activated"
                if request.data
                else "Software e-stop released"
            )
        except Exception as exc:
            response.success = False
            response.message = f"Failed to set software e-stop: {exc}"
        return response

    def _on_timer(self) -> None:
        self._refresh_estop_state()
        if not self._estop_active:
            self._dispatch_joint_targets()
            self._dispatch_cmd_vel()
        else:
            self._dispatch_zero_base()

        if self._should_run("joint_state", "state_publish_rate_hz"):
            self._publish_joint_state()
        if self._should_run("joint_feedback", "diagnostics_publish_rate_hz"):
            self._publish_joint_feedback()
            self._publish_lidar_feedback()
        if self._should_run("wrench", "wrench_publish_rate_hz"):
            self._publish_wrenches()
        if self._should_run("pointcloud", "pointcloud_publish_rate_hz"):
            self._publish_pointclouds()
        if self._should_run("odom", "odom_publish_rate_hz"):
            self._publish_odom()
        if self._should_run("tf", "tf_publish_rate_hz"):
            self._publish_dynamic_tf()

    def _refresh_estop_state(self) -> None:
        if not self._should_run("estop_poll", "estop_poll_rate_hz"):
            return
        estop = getattr(self.robot, "estop", None)
        if estop is None:
            self._estop_active = False
            return
        try:
            status = estop.get_status()
            self._estop_active = bool(
                status.get("button_pressed", False)
                or status.get("software_estop_enabled", False)
            )
        except Exception as exc:
            self._warn_throttled("estop_status", f"Could not read e-stop status: {exc}")

    def _dispatch_joint_targets(self) -> None:
        if not bool(self.get_parameter("enable_joint_commands").value):
            return
        with self._lock:
            targets = {
                name: target.copy() for name, target in self._joint_targets.items()
            }
        for component_name, target in targets.items():
            component = self._command_components.get(component_name)
            if component is None:
                continue
            try:
                component.set_joint_pos(target, wait_time=0.0)
            except Exception as exc:
                self._warn_throttled(
                    f"dispatch_{component_name}",
                    f"Failed to command {component_name}: {exc}",
                )

    def _dispatch_cmd_vel(self) -> None:
        if not bool(self.get_parameter("enable_cmd_vel").value):
            return
        chassis = self._get_robot_component(CHASSIS_COMPONENT)
        if chassis is None:
            return
        twist = self._current_cmd_vel_or_zero()
        try:
            chassis.set_velocity(
                vx=float(twist[0]),
                vy=float(twist[1]),
                wz=float(twist[2]),
                wait_time=0.0,
                sequential_steering=False,
            )
        except Exception as exc:
            self._warn_throttled("dispatch_cmd_vel", f"Failed to command chassis: {exc}")

    def _dispatch_zero_base(self) -> None:
        chassis = self._get_robot_component(CHASSIS_COMPONENT)
        if chassis is None:
            return
        try:
            chassis.set_velocity(0.0, 0.0, 0.0, wait_time=0.0, sequential_steering=False)
        except Exception as exc:
            self._warn_throttled("zero_base", f"Failed to send zero base command: {exc}")

    def _current_cmd_vel_or_zero(self) -> np.ndarray:
        now = self.get_clock().now()
        timeout_s = float(self.get_parameter("cmd_vel_timeout_s").value)
        with self._lock:
            if self._last_cmd_vel_time is None:
                return np.zeros(3, dtype=np.float64)
            age_s = (now.nanoseconds - self._last_cmd_vel_time.nanoseconds) / 1e9
            if timeout_s >= 0.0 and age_s > timeout_s:
                self._cmd_vel[:] = 0.0
            return self._cmd_vel.copy()

    def _publish_joint_state(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        efforts: list[float] = []
        for component_name, component in self._state_components.items():
            joint_names = self._component_joint_names.get(component_name, [])
            if not joint_names:
                continue

            pos = self._safe_component_array(component, "get_joint_pos", len(joint_names))
            vel = self._safe_component_array(component, "get_joint_vel", len(joint_names))
            effort = self._safe_effort_array(component, len(joint_names))
            if pos is None:
                continue

            msg.name.extend(joint_names)
            msg.position.extend(pos.tolist())
            msg.velocity.extend((vel if vel is not None else np.zeros_like(pos)).tolist())
            efforts.extend(effort.tolist())

        if msg.name:
            msg.effort = efforts
            self.joint_state_pub.publish(msg)

    def _publish_joint_feedback(self) -> None:
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [
            self._component_diagnostic(component_name, component)
            for component_name, component in self._state_components.items()
        ]
        self.joint_feedback_pub.publish(msg)

    def _component_diagnostic(self, component_name: str, component: Any) -> DiagnosticStatus:
        status = DiagnosticStatus()
        status.name = f"dexcontrol/{component_name}/joint_feedback"
        status.hardware_id = getattr(self.robot, "robot_name", "dexmate")
        status.level = DiagnosticStatus.OK
        status.message = "active"
        values: list[KeyValue] = []

        joint_names = ",".join(self._component_joint_names.get(component_name, []))
        values.append(self._kv("joint_names", joint_names))
        self._append_feedback(values, component, "position", "get_joint_pos")
        self._append_feedback(values, component, "velocity", "get_joint_vel")
        self._append_feedback(values, component, "current_A", "get_joint_current")
        self._append_feedback(values, component, "torque_Nm", "get_joint_torque")
        self._append_feedback(values, component, "error_code", "get_joint_err")

        try:
            values.append(self._kv("timestamp_ns", str(component.get_timestamp_ns())))
        except Exception:
            pass

        if component_name in {"left_hand", "right_hand"} and hasattr(
            component, "get_finger_tip_force"
        ):
            try:
                force = component.get_finger_tip_force()
                if force is not None:
                    values.append(self._kv("fingertip_force", self._array_to_csv(force)))
            except Exception:
                pass

        status.values = values
        return status

    def _append_feedback(
        self, values: list[KeyValue], component: Any, key: str, method_name: str
    ) -> None:
        try:
            method = getattr(component, method_name)
            values.append(self._kv(key, self._array_to_csv(method())))
        except Exception:
            return

    def _publish_wrenches(self) -> None:
        for side in ("left", "right"):
            arm = self._get_robot_component(f"{side}_arm")
            if arm is None or getattr(arm, "wrench_sensor", None) is None:
                continue
            sensor = arm.wrench_sensor
            try:
                wrench = np.asarray(sensor.get_wrench_state(), dtype=np.float64)
            except Exception as exc:
                self._warn_throttled(
                    f"wrench_{side}", f"Could not read {side} F/T sensor: {exc}"
                )
                continue
            if wrench.size < 6:
                continue

            msg = WrenchStamped()
            msg.header.stamp = self._stamp_from_timestamp_ns(
                self._safe_timestamp_ns(sensor)
            )
            msg.header.frame_id = str(self.get_parameter(f"{side}_ft_frame").value)
            msg.wrench.force.x = float(wrench[0])
            msg.wrench.force.y = float(wrench[1])
            msg.wrench.force.z = float(wrench[2])
            msg.wrench.torque.x = float(wrench[3])
            msg.wrench.torque.y = float(wrench[4])
            msg.wrench.torque.z = float(wrench[5])
            self.wrench_pubs[side].publish(msg)

    def _publish_pointclouds(self) -> None:
        for sensor_name, sensor in self._lidar_3d_sensors.items():
            read_start = time.monotonic()
            self._record_lidar_attempt(sensor_name)
            try:
                data = sensor.get_obs()
            except Exception as exc:
                self._record_lidar_status(
                    sensor_name,
                    "error",
                    read_start,
                    message=str(exc),
                )
                self._warn_throttled(
                    f"pointcloud_{sensor_name}", f"Could not read {sensor_name}: {exc}"
                )
                continue
            if not data:
                self._record_lidar_status(
                    sensor_name,
                    "no_data",
                    read_start,
                    message="get_obs returned no data",
                )
                continue
            msg = self._pointcloud2_from_lidar(sensor_name, data)
            if msg is not None:
                self.pointcloud_pubs[sensor_name].publish(msg)
                self._record_lidar_status(
                    sensor_name,
                    "published",
                    read_start,
                    data=data,
                    point_count=msg.width * msg.height,
                    published=True,
                )
            else:
                self._record_lidar_status(
                    sensor_name,
                    "invalid_cloud",
                    read_start,
                    data=data,
                    message="cloud missing non-empty x/y/z arrays",
                )

    def _record_lidar_attempt(self, sensor_name: str) -> None:
        status = self._lidar_status.setdefault(
            sensor_name, self._initial_lidar_status()
        )
        status["attempt_count"] = int(status.get("attempt_count", 0)) + 1
        status["last_attempt_ns"] = self.get_clock().now().nanoseconds

    def _record_lidar_status(
        self,
        sensor_name: str,
        state: str,
        read_start: float,
        data: dict[str, Any] | None = None,
        point_count: int = 0,
        published: bool = False,
        message: str = "",
    ) -> None:
        status = self._lidar_status.setdefault(
            sensor_name, self._initial_lidar_status()
        )
        previous_sequence = status.get("sequence")
        sequence = data.get("sequence") if data else None
        timestamp_ns = data.get("timestamp_ns") if data else None

        if sequence is not None and sequence == previous_sequence:
            count = int(status.get("same_sequence_count", 0))
            status["same_sequence_count"] = count + 1
        elif sequence is not None:
            status["same_sequence_count"] = 0

        status["status"] = state
        status["point_count"] = int(
            data.get("point_count", point_count) if data else point_count
        )
        status["sequence"] = sequence
        status["timestamp_ns"] = timestamp_ns
        status["read_duration_ms"] = (time.monotonic() - read_start) * 1000.0
        status["message"] = message

        if published:
            status["publish_count"] = int(status.get("publish_count", 0)) + 1
            status["last_publish_ns"] = self.get_clock().now().nanoseconds

        self._maybe_log_lidar_debug(sensor_name, status)

    def _maybe_log_lidar_debug(self, sensor_name: str, status: dict[str, Any]) -> None:
        rate_hz = float(self.get_parameter("pointcloud_debug_log_rate_hz").value)
        if rate_hz <= 0.0:
            return

        now_ns = self.get_clock().now().nanoseconds
        last_debug_ns = status.get("last_debug_ns")
        period_ns = int(1e9 / rate_hz)
        if last_debug_ns is not None and now_ns - int(last_debug_ns) < period_ns:
            return

        status["last_debug_ns"] = now_ns
        last_publish_ns = status.get("last_publish_ns")
        age_s = (
            (now_ns - int(last_publish_ns)) / 1e9
            if last_publish_ns is not None
            else float("inf")
        )
        self.get_logger().info(
            f"lidar {sensor_name}: status={status.get('status')} "
            f"points={status.get('point_count')} seq={status.get('sequence')} "
            f"same_seq={status.get('same_sequence_count')} "
            f"read_ms={float(status.get('read_duration_ms', 0.0)):.1f} "
            f"last_publish_age_s={age_s:.2f} msg={status.get('message', '')}"
        )

    def _publish_lidar_feedback(self) -> None:
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [
            self._lidar_diagnostic(sensor_name, status)
            for sensor_name, status in self._lidar_status.items()
        ]
        self.lidar_feedback_pub.publish(msg)

    def _lidar_diagnostic(
        self, sensor_name: str, status_data: dict[str, Any]
    ) -> DiagnosticStatus:
        status = DiagnosticStatus()
        status.name = f"dexcontrol/{sensor_name}/pointcloud"
        status.hardware_id = getattr(self.robot, "robot_name", "dexmate")
        status.message = str(status_data.get("status", "unknown"))

        now_ns = self.get_clock().now().nanoseconds
        last_publish_ns = status_data.get("last_publish_ns")
        publish_age_s = (
            (now_ns - int(last_publish_ns)) / 1e9
            if last_publish_ns is not None
            else float("inf")
        )
        if status_data.get("status") == "error":
            status.level = DiagnosticStatus.ERROR
        elif publish_age_s > 2.0:
            status.level = DiagnosticStatus.WARN
            status.message = f"stale pointcloud ({publish_age_s:.2f}s since publish)"
        else:
            status.level = DiagnosticStatus.OK

        status.values = [
            self._kv("status", str(status_data.get("status", "unknown"))),
            self._kv("point_count", str(status_data.get("point_count", 0))),
            self._kv("sequence", str(status_data.get("sequence"))),
            self._kv("timestamp_ns", str(status_data.get("timestamp_ns"))),
            self._kv("attempt_count", str(status_data.get("attempt_count", 0))),
            self._kv("publish_count", str(status_data.get("publish_count", 0))),
            self._kv(
                "same_sequence_count",
                str(status_data.get("same_sequence_count", 0)),
            ),
            self._kv(
                "read_duration_ms",
                f"{float(status_data.get('read_duration_ms', 0.0)):.3f}",
            ),
            self._kv("last_publish_age_s", f"{publish_age_s:.3f}"),
            self._kv("message", str(status_data.get("message", ""))),
        ]
        return status

    def _pointcloud2_from_lidar(self, sensor_name: str, data: dict[str, Any]) -> PointCloud2 | None:
        x = np.asarray(data.get("x", []), dtype=np.float32).reshape(-1)
        y = np.asarray(data.get("y", []), dtype=np.float32).reshape(-1)
        z = np.asarray(data.get("z", []), dtype=np.float32).reshape(-1)
        if x.size == 0 or y.size == 0 or z.size == 0:
            return None
        count = min(x.size, y.size, z.size)
        intensity = np.asarray(
            data.get("intensity", np.zeros(count, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if intensity.size < count:
            intensity = np.pad(intensity, (0, count - intensity.size))

        packed = np.empty(
            count,
            dtype=np.dtype(
                [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")]
            ),
        )
        packed["x"] = x[:count]
        packed["y"] = y[:count]
        packed["z"] = z[:count]
        packed["intensity"] = intensity[:count]

        height = int(data.get("height", 1) or 1)
        width = int(data.get("width", count) or count)
        if height * width != count:
            height = 1
            width = count

        msg = PointCloud2()
        msg.header.stamp = self._stamp_from_timestamp_ns(data.get("timestamp_ns"))
        msg.header.frame_id = self._frame_for_lidar_3d(sensor_name)
        msg.height = height
        msg.width = width
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="intensity", offset=12, datatype=PointField.FLOAT32, count=1
            ),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = bool(data.get("is_dense", False))
        msg.data = packed.tobytes()
        return msg

    def _frame_for_lidar_3d(self, sensor_name: str) -> str:
        parameter_name = f"{sensor_name}_frame"
        if self.has_parameter(parameter_name):
            return str(self.get_parameter(parameter_name).value)
        if sensor_name.endswith("front"):
            return str(self.get_parameter("lidar_3d_front_frame").value)
        if sensor_name.endswith("back"):
            return str(self.get_parameter("lidar_3d_back_frame").value)
        return f"{sensor_name}_link"

    def _publish_odom(self) -> None:
        now = self.get_clock().now()
        if self._last_odom_time is None:
            self._last_odom_time = now
            return
        dt = (now.nanoseconds - self._last_odom_time.nanoseconds) / 1e9
        self._last_odom_time = now
        if dt <= 0.0:
            return

        vx_body, vy_body, wz = self._estimate_base_twist()
        yaw_cos = math.cos(self._odom_yaw)
        yaw_sin = math.sin(self._odom_yaw)
        self._odom_x += (vx_body * yaw_cos - vy_body * yaw_sin) * dt
        self._odom_y += (vx_body * yaw_sin + vy_body * yaw_cos) * dt
        self._odom_yaw = self._normalize_angle(self._odom_yaw + wz * dt)

        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = str(self.get_parameter("odom_frame").value)
        msg.child_frame_id = str(self.get_parameter("base_frame").value)
        msg.pose.pose.position.x = self._odom_x
        msg.pose.pose.position.y = self._odom_y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = self._quaternion_from_yaw(self._odom_yaw)
        msg.twist.twist.linear.x = vx_body
        msg.twist.twist.linear.y = vy_body
        msg.twist.twist.angular.z = wz
        msg.pose.covariance = [
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1e6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1e6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1e6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
        ]
        msg.twist.covariance = msg.pose.covariance
        self.odom_pub.publish(msg)

    def _estimate_base_twist(self) -> tuple[float, float, float]:
        if bool(self.get_parameter("use_measured_chassis_for_odom").value):
            chassis = self._get_robot_component(CHASSIS_COMPONENT)
            if chassis is not None:
                try:
                    steering = np.asarray(chassis.steering_angle, dtype=np.float64)
                    wheel_velocity = np.asarray(chassis.wheel_velocity, dtype=np.float64)
                    return self._twist_from_chassis_state(chassis, steering, wheel_velocity)
                except Exception as exc:
                    self._warn_throttled(
                        "measured_odom",
                        f"Measured chassis odometry unavailable, using cmd_vel: {exc}",
                    )
        twist = self._current_cmd_vel_or_zero()
        return float(twist[0]), float(twist[1]), float(twist[2])

    def _twist_from_chassis_state(
        self, chassis: Any, steering: np.ndarray, wheel_velocity: np.ndarray
    ) -> tuple[float, float, float]:
        if steering.size < 2 or wheel_velocity.size < 2:
            raise ValueError("chassis state must contain left and right wheel values")
        half_wheels_dist = float(getattr(chassis, "_half_wheels_dist"))
        center_to_axis = float(getattr(chassis, "_center_to_wheel_axis_dist"))
        if abs(half_wheels_dist) < 1e-9:
            raise ValueError("invalid chassis wheel distance")

        left_vec = wheel_velocity[0] * np.array(
            [math.cos(-steering[0]), math.sin(-steering[0])]
        )
        right_vec = wheel_velocity[1] * np.array(
            [math.cos(-steering[1]), math.sin(-steering[1])]
        )
        wz = (right_vec[0] - left_vec[0]) / (2.0 * half_wheels_dist)
        vx = (left_vec[0] + right_vec[0]) / 2.0
        vy = (left_vec[1] + right_vec[1]) / 2.0 - wz * center_to_axis
        return float(vx), float(vy), float(wz)

    def _publish_dynamic_tf(self) -> None:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.get_parameter("odom_frame").value)
        msg.child_frame_id = str(self.get_parameter("base_frame").value)
        msg.transform.translation.x = self._odom_x
        msg.transform.translation.y = self._odom_y
        msg.transform.translation.z = 0.0
        msg.transform.rotation = self._quaternion_from_yaw(self._odom_yaw)
        self.tf_broadcaster.sendTransform(msg)

    def _should_run(self, key: str, rate_parameter: str) -> bool:
        rate_hz = float(self.get_parameter(rate_parameter).value)
        if rate_hz <= 0.0:
            return False
        now_ns = self.get_clock().now().nanoseconds
        period_ns = int(1e9 / rate_hz)
        last_ns = self._last_run_ns.get(key)
        if last_ns is None or now_ns - last_ns >= period_ns:
            self._last_run_ns[key] = now_ns
            return True
        return False

    def _safe_component_array(
        self, component: Any, method_name: str, expected_size: int
    ) -> np.ndarray | None:
        try:
            data = np.asarray(getattr(component, method_name)(), dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if data.size != expected_size:
            return None
        return data

    def _safe_effort_array(self, component: Any, expected_size: int) -> np.ndarray:
        for method_name in ("get_joint_torque", "get_joint_current"):
            data = self._safe_component_array(component, method_name, expected_size)
            if data is not None:
                return data
        return np.zeros(expected_size, dtype=np.float64)

    def _safe_timestamp_ns(self, component: Any) -> int | None:
        try:
            return int(component.get_timestamp_ns())
        except Exception:
            return None

    def _stamp_from_timestamp_ns(self, timestamp_ns: Any | None) -> Any:
        if bool(self.get_parameter("use_robot_timestamps").value) and timestamp_ns:
            try:
                ns = int(timestamp_ns)
                return Time(nanoseconds=ns).to_msg()
            except Exception:
                pass
        return self.get_clock().now().to_msg()

    def _warn_throttled(self, key: str, message: str, interval_s: float = 2.0) -> None:
        now = time.monotonic()
        last = self._last_warn_s.get(key, 0.0)
        if now - last >= interval_s:
            self._last_warn_s[key] = now
            self.get_logger().warn(message)

    @staticmethod
    def _kv(key: str, value: str) -> KeyValue:
        item = KeyValue()
        item.key = key
        item.value = value
        return item

    @staticmethod
    def _array_to_csv(data: Any) -> str:
        array = np.asarray(data).reshape(-1)
        return ",".join(str(float(value)) for value in array)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _quaternion_from_yaw(yaw: float) -> Quaternion:
        quat = Quaternion()
        quat.x = 0.0
        quat.y = 0.0
        quat.z = math.sin(yaw / 2.0)
        quat.w = math.cos(yaw / 2.0)
        return quat

    def destroy_node(self) -> None:
        try:
            self.robot.shutdown()
        except Exception as exc:
            self.get_logger().warn(f"Robot shutdown reported an error: {exc}")
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: DexcontrolBridge | None = None
    try:
        node = DexcontrolBridge()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
