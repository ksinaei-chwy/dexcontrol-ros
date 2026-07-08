#!/usr/bin/env python3
"""MeshCat visualizer for dry-run Pico teleoperation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from dex_pico_teleop.kinematics import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    TORSO_JOINTS,
)


ACTION_JOINTS = {
    "torso": TORSO_JOINTS,
    "head": HEAD_JOINTS,
    "left_arm": LEFT_ARM_JOINTS,
    "right_arm": RIGHT_ARM_JOINTS,
}


class PicoMeshcatVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("pico_meshcat_visualizer")
        self.declare_parameter("robot_urdf_path", "")
        self.declare_parameter("root_node_name", "vega_pico_teleop")
        self.declare_parameter("open_browser", False)
        self.declare_parameter("max_update_rate_hz", 30.0)
        self.declare_parameter("topic", "/dex_pico_teleop/log_frame")
        self.declare_parameter("meshcat_use_joint_state_initial_pose", True)
        self.declare_parameter("meshcat_show_visuals", True)
        self.declare_parameter("meshcat_show_collisions", False)

        self._pin = None
        self._model = None
        self._data = None
        self._viz = None
        self._q = None
        self._joint_indices: dict[str, int] = {}
        self._arm_center_frame_id = None
        self._ee_frame_ids: dict[str, int] = {}
        self._last_display_ns = 0
        self._has_log_frame = False
        self._meshcat_geometry = None
        self._warned_debug_overlay = False

        self._init_meshcat()
        if bool(self.get_parameter("meshcat_use_joint_state_initial_pose").value):
            self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        topic = str(self.get_parameter("topic").value)
        self.create_subscription(String, topic, self._on_log_frame, 10)
        self.get_logger().info(f"listening for Pico teleop action frames on {topic}")

    def _init_meshcat(self) -> None:
        try:
            import pinocchio as pin
            from pinocchio.visualize import MeshcatVisualizer
        except Exception as exc:  # noqa: BLE001 - optional visualization dependency
            raise RuntimeError(
                "Pinocchio with MeshCat visualization support is required"
            ) from exc

        self._pin = pin
        urdf_path = self._robot_urdf_path()
        package_dirs = [str(Path(get_package_share_directory("dexmate_vega_description")).parent)]
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            str(urdf_path),
            package_dirs,
        )
        self._model = model
        self._data = model.createData()
        self._q = pin.neutral(model)
        self._joint_indices = {
            name: int(model.joints[model.getJointId(name)].idx_q)
            for names in ACTION_JOINTS.values()
            for name in names
            if model.existJointName(name)
        }
        if model.existFrame("arm_center"):
            self._arm_center_frame_id = model.getFrameId("arm_center")
        self._ee_frame_ids = {
            "left": model.getFrameId("L_ee") if model.existFrame("L_ee") else None,
            "right": model.getFrameId("R_ee") if model.existFrame("R_ee") else None,
        }

        self._viz = MeshcatVisualizer(model, collision_model, visual_model)
        self._viz.initViewer(open=bool(self.get_parameter("open_browser").value))
        self._viz.loadViewerModel(rootNodeName=str(self.get_parameter("root_node_name").value))
        self._viz.displayVisuals(bool(self.get_parameter("meshcat_show_visuals").value))
        self._viz.displayCollisions(bool(self.get_parameter("meshcat_show_collisions").value))
        self._viz.display(self._q)

        url = self._viz.viewer.url()
        self.get_logger().info(f"MeshCat Vega visualizer ready: {url}")

    def _robot_urdf_path(self) -> Path:
        configured = str(self.get_parameter("robot_urdf_path").value)
        if configured:
            return Path(configured)
        description_share = Path(get_package_share_directory("dexmate_vega_description"))
        return description_share / "urdf" / "vega_1p_f5d6.package.urdf"

    def _on_log_frame(self, msg: String) -> None:
        if self._viz is None or self._q is None:
            return

        now_ns = self.get_clock().now().nanoseconds
        min_period_ns = int(1.0e9 / float(self.get_parameter("max_update_rate_hz").value))
        if now_ns - self._last_display_ns < min_period_ns:
            return

        try:
            payload = json.loads(msg.data)
            action = payload.get("action", {})
            debug = payload.get("debug", {})
            self._apply_action(action)
        except Exception as exc:  # noqa: BLE001 - ROS boundary
            self.get_logger().warn(f"ignoring malformed teleop log frame: {exc}")
            return

        self._viz.display(self._q)
        self._apply_debug(debug)
        self._last_display_ns = now_ns
        self._has_log_frame = True

    def _on_joint_state(self, msg: JointState) -> None:
        if self._viz is None or self._q is None or self._has_log_frame:
            return
        changed = False
        for name, position in zip(msg.name, msg.position):
            index = self._joint_indices.get(name)
            if index is not None and np.isfinite(position):
                self._q[index] = float(position)
                changed = True
        if changed:
            self._viz.display(self._q)

    def _apply_action(self, action: dict[str, object]) -> None:
        for component, joint_names in ACTION_JOINTS.items():
            if component not in action:
                continue
            values = np.asarray(action[component], dtype=np.float64).reshape(-1)
            if values.size != len(joint_names):
                raise ValueError(
                    f"{component} expected {len(joint_names)} values, got {values.size}"
                )
            for name, value in zip(joint_names, values):
                index = self._joint_indices.get(name)
                if index is not None:
                    self._q[index] = float(value)

    def _apply_debug(self, debug: object) -> None:
        if self._viz is None or self._q is None or not isinstance(debug, dict):
            return
        retarget = debug.get("retarget", {})
        if not isinstance(retarget, dict):
            return

        colors = {
            "left": 0x2CA7FF,
            "right": 0xFF6B35,
        }
        for side in ("left", "right"):
            data = retarget.get(side, {})
            if not isinstance(data, dict):
                continue
            color = colors[side]
            shoulder = _array3(data.get("operator_shoulder"))
            controller = _array3(data.get("controller_position"))
            controller_rotation = _matrix3(data.get("controller_rotation"))
            robot_shoulder = _array3(data.get("robot_shoulder"))
            robot_target = _array3(data.get("robot_target"))
            robot_target_rotation = _matrix3(data.get("robot_target_rotation"))

            if shoulder is not None:
                self._set_marker(f"debug/operator/{side}_shoulder", shoulder, color, 0.035)
            if controller is not None:
                self._set_marker(f"debug/operator/{side}_controller", controller, color, 0.03)
            if controller is not None and controller_rotation is not None:
                self._set_frame(
                    f"debug/operator/{side}_controller_frame",
                    controller,
                    controller_rotation,
                    0.10,
                )
            if shoulder is not None and controller is not None:
                self._set_line(f"debug/operator/{side}_reach", shoulder, controller, color)

            if robot_shoulder is not None:
                world_shoulder = self._arm_center_point_to_world(robot_shoulder)
                self._set_marker(f"debug/robot/{side}_shoulder", world_shoulder, color, 0.025)
            if robot_target is not None:
                world_target = self._arm_center_point_to_world(robot_target)
                self._set_marker(f"debug/robot/{side}_target", world_target, color, 0.035)
                if robot_target_rotation is not None:
                    world_target_pos, world_target_rot = self._arm_center_pose_to_world(
                        robot_target,
                        robot_target_rotation,
                    )
                    self._set_frame(
                        f"debug/robot/{side}_target_frame",
                        world_target_pos,
                        world_target_rot,
                        0.10,
                    )
            if robot_shoulder is not None and robot_target is not None:
                self._set_line(
                    f"debug/robot/{side}_reach",
                    self._arm_center_point_to_world(robot_shoulder),
                    self._arm_center_point_to_world(robot_target),
                    color,
                )
            ee_pose = self._ee_pose_to_world(side)
            if ee_pose is not None:
                ee_position, ee_rotation = ee_pose
                self._set_frame(f"debug/robot/{side}_ee_frame", ee_position, ee_rotation, 0.08)

    def _set_marker(self, path: str, position: np.ndarray, color: int, radius: float) -> None:
        geometry = self._geometry_module()
        if geometry is None or self._viz is None:
            return
        self._viz.viewer[path].set_object(
            geometry.Sphere(float(radius)),
            geometry.MeshLambertMaterial(color=color),
        )
        self._viz.viewer[path].set_transform(_translation_matrix(position))

    def _set_line(self, path: str, start: np.ndarray, end: np.ndarray, color: int) -> None:
        geometry = self._geometry_module()
        if geometry is None or self._viz is None:
            return
        try:
            self._viz.viewer[path].set_object(
                geometry.Line(
                    geometry.PointsGeometry(np.column_stack((start, end))),
                    geometry.LineBasicMaterial(color=color, linewidth=4),
                )
            )
        except Exception as exc:  # noqa: BLE001 - optional overlay boundary
            if not self._warned_debug_overlay:
                self.get_logger().warn(f"MeshCat debug overlay line unavailable: {exc}")
                self._warned_debug_overlay = True

    def _set_frame(
        self,
        path: str,
        position: np.ndarray,
        rotation: np.ndarray,
        length: float,
    ) -> None:
        origin = np.asarray(position, dtype=np.float64).reshape(3)
        rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        axes = (
            ("x", rot[:, 0], 0xFF3333),
            ("y", rot[:, 1], 0x33CC33),
            ("z", rot[:, 2], 0x3388FF),
        )
        for axis_name, direction, color in axes:
            self._set_line(
                f"{path}/{axis_name}",
                origin,
                origin + direction * float(length),
                color,
            )

    def _geometry_module(self):
        if self._meshcat_geometry is not None:
            return self._meshcat_geometry
        try:
            import meshcat.geometry as geometry
        except Exception as exc:  # noqa: BLE001 - optional visualization dependency
            if not self._warned_debug_overlay:
                self.get_logger().warn(f"MeshCat debug overlay unavailable: {exc}")
                self._warned_debug_overlay = True
            return None
        self._meshcat_geometry = geometry
        return geometry

    def _arm_center_point_to_world(self, point: np.ndarray) -> np.ndarray:
        if (
            self._pin is None
            or self._model is None
            or self._data is None
            or self._q is None
            or self._arm_center_frame_id is None
        ):
            return point
        self._pin.forwardKinematics(self._model, self._data, self._q)
        self._pin.updateFramePlacements(self._model, self._data)
        transform = self._data.oMf[self._arm_center_frame_id]
        return np.asarray(transform.rotation @ point + transform.translation, dtype=np.float64)

    def _arm_center_pose_to_world(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            self._pin is None
            or self._model is None
            or self._data is None
            or self._q is None
            or self._arm_center_frame_id is None
        ):
            return (
                np.asarray(position, dtype=np.float64).reshape(3),
                np.asarray(rotation, dtype=np.float64).reshape(3, 3),
            )
        self._pin.forwardKinematics(self._model, self._data, self._q)
        self._pin.updateFramePlacements(self._model, self._data)
        transform = self._data.oMf[self._arm_center_frame_id]
        world_position = transform.rotation @ np.asarray(position, dtype=np.float64).reshape(3)
        world_position = world_position + transform.translation
        world_rotation = transform.rotation @ np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        return (
            np.asarray(world_position, dtype=np.float64),
            np.asarray(world_rotation, dtype=np.float64),
        )

    def _ee_pose_to_world(self, side: str) -> tuple[np.ndarray, np.ndarray] | None:
        if (
            self._pin is None
            or self._model is None
            or self._data is None
            or self._q is None
        ):
            return None
        frame_id = self._ee_frame_ids.get(side)
        if frame_id is None:
            return None
        self._pin.forwardKinematics(self._model, self._data, self._q)
        self._pin.updateFramePlacements(self._model, self._data)
        transform = self._data.oMf[frame_id]
        return (
            np.asarray(transform.translation, dtype=np.float64).copy(),
            np.asarray(transform.rotation, dtype=np.float64).copy(),
        )


def _array3(value: object) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def _matrix3(value: object) -> np.ndarray | None:
    try:
        matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _translation_matrix(position: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: PicoMeshcatVisualizer | None = None
    try:
        node = PicoMeshcatVisualizer()
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
