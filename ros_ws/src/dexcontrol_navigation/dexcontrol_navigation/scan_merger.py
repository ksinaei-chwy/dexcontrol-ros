#!/usr/bin/env python3
"""Merge projected front/back lidar scans into one planar scan for SLAM."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


@dataclass(frozen=True)
class ScanGeometry:
    """Output scan geometry in radians and meters."""

    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float

    @property
    def bin_count(self) -> int:
        span = self.angle_max - self.angle_min
        return int(math.floor(span / self.angle_increment)) + 1


class LaserScanMerger(Node):
    """Combine multiple same-frame LaserScan inputs by taking nearest valid ranges."""

    DEFAULT_INPUTS: Final[list[str]] = [
        "/lidar_3d_front/scan",
        "/lidar_3d_back/scan",
    ]

    def __init__(self) -> None:
        super().__init__("dual_lidar_scan_merger")
        self._declare_parameters()

        input_topics = self._string_list_parameter("input_scan_topics", self.DEFAULT_INPUTS)
        output_topic = str(self.get_parameter("output_scan_topic").value)
        self._target_frame = str(self.get_parameter("target_frame").value)
        self._scan_time = float(self.get_parameter("scan_time").value)
        self._stale_timeout_s = float(self.get_parameter("stale_timeout_s").value)
        self._use_inf = bool(self.get_parameter("use_inf").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self._geometry = ScanGeometry(
            angle_min=float(self.get_parameter("angle_min").value),
            angle_max=float(self.get_parameter("angle_max").value),
            angle_increment=float(self.get_parameter("angle_increment").value),
            range_min=float(self.get_parameter("range_min").value),
            range_max=float(self.get_parameter("range_max").value),
        )
        self._validate_parameters(input_topics, publish_rate_hz)

        self._latest_scans: dict[str, LaserScan] = {}
        self._last_frame_warn_ns: dict[str, int] = {}
        self._publish = self.create_publisher(
            LaserScan,
            output_topic,
            qos_profile_sensor_data,
        )
        for topic in input_topics:
            self.create_subscription(
                LaserScan,
                topic,
                lambda msg, topic_name=topic: self._on_scan(topic_name, msg),
                qos_profile_sensor_data,
            )

        self.create_timer(1.0 / publish_rate_hz, self._on_timer)
        self.get_logger().info(
            f"Merging {len(input_topics)} projected scans into {output_topic} "
            f"with frame '{self._target_frame}'"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("input_scan_topics", self.DEFAULT_INPUTS)
        self.declare_parameter("output_scan_topic", "/scan")
        self.declare_parameter("target_frame", "base")
        self.declare_parameter("angle_min", -math.pi)
        self.declare_parameter("angle_max", math.pi)
        self.declare_parameter("angle_increment", math.radians(0.5))
        self.declare_parameter("scan_time", 0.1)
        self.declare_parameter("range_min", 0.15)
        self.declare_parameter("range_max", 20.0)
        self.declare_parameter("stale_timeout_s", 0.5)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("use_inf", True)

    def _string_list_parameter(self, name: str, default: list[str]) -> list[str]:
        value = self.get_parameter(name).value
        if isinstance(value, (list, tuple)):
            topics = [str(item) for item in value if str(item)]
            return topics or default
        return default

    def _validate_parameters(self, input_topics: list[str], publish_rate_hz: float) -> None:
        if not input_topics:
            raise ValueError("input_scan_topics must contain at least one topic")
        if publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        if self._geometry.angle_increment <= 0.0:
            raise ValueError("angle_increment must be positive")
        if self._geometry.angle_max <= self._geometry.angle_min:
            raise ValueError("angle_max must be greater than angle_min")
        if self._geometry.range_max <= self._geometry.range_min:
            raise ValueError("range_max must be greater than range_min")
        if self._geometry.bin_count <= 1:
            raise ValueError("output scan geometry must contain more than one bin")

    def _on_scan(self, topic: str, msg: LaserScan) -> None:
        if msg.header.frame_id and msg.header.frame_id != self._target_frame:
            self._warn_frame_mismatch(topic, msg.header.frame_id)
            return
        self._latest_scans[topic] = msg

    def _warn_frame_mismatch(self, topic: str, frame_id: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        last_ns = self._last_frame_warn_ns.get(topic)
        if last_ns is not None and now_ns - last_ns < 2_000_000_000:
            return
        self._last_frame_warn_ns[topic] = now_ns
        self.get_logger().warn(
            f"Ignoring {topic}: frame '{frame_id}' does not match "
            f"target_frame '{self._target_frame}'. Set pointcloud_to_laserscan "
            "target_frame to the merger target frame."
        )

    def _on_timer(self) -> None:
        fresh_scans = self._fresh_scans()
        if not fresh_scans:
            return

        ranges = self._empty_ranges()
        intensities = [0.0] * self._geometry.bin_count
        for scan in fresh_scans:
            self._merge_scan(scan, ranges, intensities)

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._target_frame
        msg.angle_min = self._geometry.angle_min
        msg.angle_max = self._geometry.angle_max
        msg.angle_increment = self._geometry.angle_increment
        msg.time_increment = 0.0
        msg.scan_time = self._scan_time
        msg.range_min = self._geometry.range_min
        msg.range_max = self._geometry.range_max
        msg.ranges = ranges
        msg.intensities = intensities
        self._publish.publish(msg)

    def _fresh_scans(self) -> list[LaserScan]:
        now_ns = self.get_clock().now().nanoseconds
        fresh: list[LaserScan] = []
        for scan in self._latest_scans.values():
            stamp_ns = int(
                scan.header.stamp.sec * 1_000_000_000 + scan.header.stamp.nanosec
            )
            age_s = (now_ns - stamp_ns) / 1e9
            if stamp_ns <= 0 or age_s <= self._stale_timeout_s:
                fresh.append(scan)
        return fresh

    def _empty_ranges(self) -> list[float]:
        empty = math.inf if self._use_inf else self._geometry.range_max + 1.0
        return [empty] * self._geometry.bin_count

    def _merge_scan(
        self,
        scan: LaserScan,
        output_ranges: list[float],
        output_intensities: list[float],
    ) -> None:
        for source_index, source_range in enumerate(scan.ranges):
            if not self._is_valid_range(float(source_range), scan.range_min, scan.range_max):
                continue
            angle = scan.angle_min + source_index * scan.angle_increment
            target_index = int(
                round(
                    (angle - self._geometry.angle_min)
                    / self._geometry.angle_increment
                )
            )
            if target_index < 0 or target_index >= len(output_ranges):
                continue
            if source_range < output_ranges[target_index]:
                output_ranges[target_index] = float(source_range)
                if source_index < len(scan.intensities):
                    output_intensities[target_index] = float(scan.intensities[source_index])

    def _is_valid_range(
        self,
        value: float,
        source_range_min: float,
        source_range_max: float,
    ) -> bool:
        lower = max(self._geometry.range_min, float(source_range_min))
        upper = min(self._geometry.range_max, float(source_range_max))
        return math.isfinite(value) and lower <= value <= upper


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: LaserScanMerger | None = None
    try:
        node = LaserScanMerger()
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
