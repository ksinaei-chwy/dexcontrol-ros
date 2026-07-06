#!/usr/bin/env python3
"""MeshCat visualizer for dry-run Pico teleoperation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
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

        self._pin = None
        self._viz = None
        self._q = None
        self._joint_indices: dict[str, int] = {}
        self._last_display_ns = 0

        self._init_meshcat()
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
        self._q = pin.neutral(model)
        self._joint_indices = {
            name: int(model.joints[model.getJointId(name)].idx_q)
            for names in ACTION_JOINTS.values()
            for name in names
            if model.existJointName(name)
        }

        self._viz = MeshcatVisualizer(model, collision_model, visual_model)
        self._viz.initViewer(open=bool(self.get_parameter("open_browser").value))
        self._viz.loadViewerModel(rootNodeName=str(self.get_parameter("root_node_name").value))
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
            action = json.loads(msg.data).get("action", {})
            self._apply_action(action)
        except Exception as exc:  # noqa: BLE001 - ROS boundary
            self.get_logger().warn(f"ignoring malformed teleop log frame: {exc}")
            return

        self._viz.display(self._q)
        self._last_display_ns = now_ns

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
