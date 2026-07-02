#!/usr/bin/env python3
"""Mask robot self-returns from a LaserScan in the robot base frame."""

from __future__ import annotations

import math
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


@dataclass(frozen=True)
class ExclusionBox:
    """Axis-aligned x/y exclusion box in the scan frame, using SI units."""

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


class ScanSelfFilter(Node):
    """Remove LaserScan returns that fall inside configured robot self boxes."""

    def __init__(self) -> None:
        super().__init__("front_scan_self_filter")
        self.declare_parameter("input_scan_topic", "/lidar_3d_front/scan_raw")
        self.declare_parameter("output_scan_topic", "/lidar_3d_front/scan")
        self.declare_parameter("target_frame", "base")
        self.declare_parameter("use_inf", True)
        self.declare_parameter(
            "exclusion_boxes",
            ["front_body_and_arms:-0.20,1.00,-0.55,0.55"],
        )
        self.declare_parameter("log_period_s", 2.0)

        input_topic = str(self.get_parameter("input_scan_topic").value)
        output_topic = str(self.get_parameter("output_scan_topic").value)
        self._target_frame = str(self.get_parameter("target_frame").value)
        self._use_inf = bool(self.get_parameter("use_inf").value)
        self._log_period_s = float(self.get_parameter("log_period_s").value)
        self._boxes = self._parse_boxes(self.get_parameter("exclusion_boxes").value)
        self._last_log_ns = 0

        self._publisher = self.create_publisher(LaserScan, output_topic, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan,
            input_topic,
            self._on_scan,
            qos_profile_sensor_data,
        )
        box_text = ", ".join(
            f"{box.name}[x={box.x_min:.2f}..{box.x_max:.2f}, "
            f"y={box.y_min:.2f}..{box.y_max:.2f}]"
            for box in self._boxes
        )
        self.get_logger().info(
            f"Masking {input_topic} -> {output_topic} in frame "
            f"'{self._target_frame}' using boxes: {box_text}"
        )

    def _parse_boxes(self, raw_value: object) -> list[ExclusionBox]:
        if not isinstance(raw_value, (list, tuple)):
            raise ValueError("exclusion_boxes must be a list of strings")
        boxes: list[ExclusionBox] = []
        for index, item in enumerate(raw_value):
            text = str(item).strip()
            if not text:
                continue
            if ":" in text:
                name, values_text = text.split(":", 1)
                name = name.strip() or f"box_{index}"
            else:
                name = f"box_{index}"
                values_text = text
            values = [part.strip() for part in values_text.split(",")]
            if len(values) != 4:
                raise ValueError(
                    "Each exclusion box must be 'name:x_min,x_max,y_min,y_max'"
                )
            x_min, x_max, y_min, y_max = (float(value) for value in values)
            if x_max <= x_min or y_max <= y_min:
                raise ValueError(f"Invalid exclusion box bounds for {name}")
            boxes.append(ExclusionBox(name, x_min, x_max, y_min, y_max))
        if not boxes:
            raise ValueError("At least one exclusion box is required")
        return boxes

    def _on_scan(self, msg: LaserScan) -> None:
        if msg.header.frame_id and msg.header.frame_id != self._target_frame:
            self.get_logger().warn(
                f"Passing scan through: frame '{msg.header.frame_id}' does not match "
                f"target_frame '{self._target_frame}'"
            )
            self._publisher.publish(msg)
            return

        filtered = LaserScan()
        filtered.header = msg.header
        filtered.angle_min = msg.angle_min
        filtered.angle_max = msg.angle_max
        filtered.angle_increment = msg.angle_increment
        filtered.time_increment = msg.time_increment
        filtered.scan_time = msg.scan_time
        filtered.range_min = msg.range_min
        filtered.range_max = msg.range_max
        filtered.ranges = list(msg.ranges)
        filtered.intensities = list(msg.intensities)

        replacement = math.inf if self._use_inf else msg.range_max + 1.0
        masked_count = 0
        valid_count = 0
        for index, range_value in enumerate(filtered.ranges):
            value = float(range_value)
            if not math.isfinite(value) or value < msg.range_min or value > msg.range_max:
                continue
            valid_count += 1
            angle = msg.angle_min + index * msg.angle_increment
            x = value * math.cos(angle)
            y = value * math.sin(angle)
            if any(box.contains(x, y) for box in self._boxes):
                filtered.ranges[index] = replacement
                if index < len(filtered.intensities):
                    filtered.intensities[index] = 0.0
                masked_count += 1

        self._publisher.publish(filtered)
        self._log_stats(masked_count, valid_count)

    def _log_stats(self, masked_count: int, valid_count: int) -> None:
        if self._log_period_s <= 0.0:
            return
        now_ns = self.get_clock().now().nanoseconds
        period_ns = int(self._log_period_s * 1e9)
        if now_ns - self._last_log_ns < period_ns:
            return
        self._last_log_ns = now_ns
        self.get_logger().info(
            f"Masked {masked_count}/{valid_count} finite front scan returns"
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: ScanSelfFilter | None = None
    try:
        node = ScanSelfFilter()
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
