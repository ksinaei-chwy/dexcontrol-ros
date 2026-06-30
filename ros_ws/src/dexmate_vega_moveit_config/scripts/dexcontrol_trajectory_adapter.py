#!/usr/bin/env python3
"""Expose MoveIt FollowJointTrajectory actions for the dexcontrol ROS bridge.

This node is intentionally part of the MoveIt config package rather than the
low-level dexcontrol bridge. It adapts MoveIt's standard trajectory execution
interface to the existing dexcontrol_ros JointState command topics:

- /left_arm_controller/follow_joint_trajectory -> /left_arm/joint_commands
- /right_arm_controller/follow_joint_trajectory -> /right_arm/joint_commands
- /torso_controller/follow_joint_trajectory -> /torso/joint_commands
- /head_cam_controller/follow_joint_trajectory -> /head/joint_commands

The adapter commands position targets only. The dexcontrol bridge owns the final
hardware dispatch loop and continuously sends the latest target to the robot.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


SUCCESSFUL = 0
INVALID_GOAL = -1
INVALID_JOINTS = -2
PATH_TOLERANCE_VIOLATED = -4


@dataclass(frozen=True)
class ControllerSpec:
    """MoveIt controller action mapped to one dexcontrol JointState command topic."""

    name: str
    action_name: str
    command_topic: str
    joints: tuple[str, ...]


class DexcontrolTrajectoryAdapter(Node):
    """Bridge MoveIt trajectory actions to dexcontrol component command topics."""

    def __init__(self) -> None:
        super().__init__("dexcontrol_trajectory_adapter")

        self.declare_parameter("command_publish_rate_hz", 100.0)
        self.declare_parameter("allowed_start_tolerance_rad", 0.35)
        self.declare_parameter("state_timeout_sec", 1.0)
        self.declare_parameter("require_current_state", True)
        self.declare_parameter("enforce_goal_tolerance", False)
        self.declare_parameter("goal_tolerance_rad", 0.08)
        self.declare_parameter("goal_settle_time_sec", 1.0)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("left_arm_command_topic", "/left_arm/joint_commands")
        self.declare_parameter("right_arm_command_topic", "/right_arm/joint_commands")
        self.declare_parameter("torso_command_topic", "/torso/joint_commands")
        self.declare_parameter("head_command_topic", "/head/joint_commands")

        self._state_lock = threading.Lock()
        self._latest_positions: dict[str, float] = {}
        self._latest_state_time = 0.0
        self._active_lock = threading.Lock()
        self._active_controllers: set[str] = set()

        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

        self._controllers = self._default_controllers()
        self._command_publishers = {
            spec.name: self.create_publisher(JointState, spec.command_topic, 10)
            for spec in self._controllers
        }
        self._servers = [
            ActionServer(
                self,
                FollowJointTrajectory,
                spec.action_name,
                execute_callback=lambda goal, spec=spec: self._execute(goal, spec),
                goal_callback=lambda goal, spec=spec: self._on_goal(goal, spec),
                cancel_callback=self._on_cancel,
            )
            for spec in self._controllers
        ]

        for spec in self._controllers:
            self.get_logger().info(
                f"Serving {spec.action_name} -> {spec.command_topic} "
                f"for joints: {', '.join(spec.joints)}"
            )

    def _default_controllers(self) -> tuple[ControllerSpec, ...]:
        left_topic = str(self.get_parameter("left_arm_command_topic").value)
        right_topic = str(self.get_parameter("right_arm_command_topic").value)
        torso_topic = str(self.get_parameter("torso_command_topic").value)
        head_topic = str(self.get_parameter("head_command_topic").value)
        return (
            ControllerSpec(
                name="left_arm_controller",
                action_name="/left_arm_controller/follow_joint_trajectory",
                command_topic=left_topic,
                joints=tuple(f"L_arm_j{i}" for i in range(1, 8)),
            ),
            ControllerSpec(
                name="right_arm_controller",
                action_name="/right_arm_controller/follow_joint_trajectory",
                command_topic=right_topic,
                joints=tuple(f"R_arm_j{i}" for i in range(1, 8)),
            ),
            ControllerSpec(
                name="torso_controller",
                action_name="/torso_controller/follow_joint_trajectory",
                command_topic=torso_topic,
                joints=tuple(f"torso_j{i}" for i in range(1, 4)),
            ),
            ControllerSpec(
                name="head_cam_controller",
                action_name="/head_cam_controller/follow_joint_trajectory",
                command_topic=head_topic,
                joints=tuple(f"head_j{i}" for i in range(1, 4)),
            ),
        )

    def _on_joint_state(self, msg: JointState) -> None:
        now = time.monotonic()
        with self._state_lock:
            for name, position in zip(msg.name, msg.position):
                if math.isfinite(position):
                    self._latest_positions[name] = float(position)
            self._latest_state_time = now

    def _on_goal(
        self, goal_request: FollowJointTrajectory.Goal, spec: ControllerSpec
    ) -> GoalResponse:
        trajectory = goal_request.trajectory
        error = self._validate_trajectory(trajectory.joint_names, trajectory.points, spec)
        if error:
            self.get_logger().error(f"Rejecting {spec.name} goal: {error}")
            return GoalResponse.REJECT

        with self._active_lock:
            if spec.name in self._active_controllers:
                self.get_logger().error(f"Rejecting {spec.name} goal: controller is busy")
                return GoalResponse.REJECT
            self._active_controllers.add(spec.name)
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle, spec: ControllerSpec) -> FollowJointTrajectory.Result:
        try:
            result = self._execute_locked(goal_handle, spec)
        finally:
            with self._active_lock:
                self._active_controllers.discard(spec.name)
        return result

    def _execute_locked(self, goal_handle, spec: ControllerSpec) -> FollowJointTrajectory.Result:
        goal = goal_handle.request
        trajectory = goal.trajectory
        points = list(trajectory.points)
        joint_order = list(trajectory.joint_names)
        result = FollowJointTrajectory.Result()

        start_error = self._check_current_state_start(joint_order, points[0], spec)
        if start_error:
            self.get_logger().error(f"Aborting {spec.name}: {start_error}")
            goal_handle.abort()
            result.error_code = INVALID_GOAL
            result.error_string = start_error
            return result

        dry_run = bool(self.get_parameter("dry_run").value)
        if dry_run:
            self.get_logger().warn(f"dry_run=true: accepting {spec.name} without publishing commands")

        start_time = time.monotonic()
        publish_period = 1.0 / float(self.get_parameter("command_publish_rate_hz").value)
        if publish_period <= 0.0 or not math.isfinite(publish_period):
            publish_period = 0.01

        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = list(spec.joints)

        previous = self._point_positions(points[0], joint_order, spec)
        self._publish_command(spec, previous, dry_run=dry_run)
        self._publish_feedback(goal_handle, spec, previous, feedback)

        for index in range(1, len(points)):
            if goal_handle.is_cancel_requested:
                self._hold_current_or_last(spec, previous, dry_run=dry_run)
                goal_handle.canceled()
                result.error_code = SUCCESSFUL
                result.error_string = "Trajectory canceled; holding current target"
                return result

            current = self._point_positions(points[index], joint_order, spec)
            segment_start = self._seconds_from_start(points[index - 1])
            segment_end = self._seconds_from_start(points[index])
            duration = max(0.0, segment_end - segment_start)
            wall_segment_start = start_time + segment_start
            wall_segment_end = start_time + segment_end

            while time.monotonic() < wall_segment_end:
                if goal_handle.is_cancel_requested:
                    self._hold_current_or_last(spec, previous, dry_run=dry_run)
                    goal_handle.canceled()
                    result.error_code = SUCCESSFUL
                    result.error_string = "Trajectory canceled; holding current target"
                    return result

                now = time.monotonic()
                if duration <= 1e-6:
                    command = current
                else:
                    ratio = min(max((now - wall_segment_start) / duration, 0.0), 1.0)
                    command = [
                        start + ratio * (end - start)
                        for start, end in zip(previous, current)
                    ]
                self._publish_command(spec, command, dry_run=dry_run)
                self._publish_feedback(goal_handle, spec, command, feedback)
                time.sleep(publish_period)

            self._publish_command(spec, current, dry_run=dry_run)
            self._publish_feedback(goal_handle, spec, current, feedback)
            previous = current

        tolerance_error = self._wait_for_goal_tolerance(spec, previous)
        if tolerance_error:
            self.get_logger().error(f"Aborting {spec.name}: {tolerance_error}")
            goal_handle.abort()
            result.error_code = PATH_TOLERANCE_VIOLATED
            result.error_string = tolerance_error
            return result

        goal_handle.succeed()
        result.error_code = SUCCESSFUL
        result.error_string = "Trajectory command completed"
        return result

    def _validate_trajectory(
        self,
        joint_names: Iterable[str],
        points: Iterable[JointTrajectoryPoint],
        spec: ControllerSpec,
    ) -> str | None:
        joint_names = list(joint_names)
        points = list(points)
        if not points:
            return "trajectory has no points"
        missing = [joint for joint in spec.joints if joint not in joint_names]
        extra = [joint for joint in joint_names if joint not in spec.joints]
        if missing or extra:
            return f"joint mismatch; missing={missing}, extra={extra}"

        previous_time = -1e-9
        for point_index, point in enumerate(points):
            if len(point.positions) != len(joint_names):
                return (
                    f"point {point_index} has {len(point.positions)} positions for "
                    f"{len(joint_names)} joints"
                )
            if any(not math.isfinite(value) for value in point.positions):
                return f"point {point_index} contains non-finite positions"
            current_time = self._seconds_from_start(point)
            if current_time + 1e-9 < previous_time:
                return "trajectory point times are not monotonic"
            previous_time = current_time
        return None

    def _check_current_state_start(
        self,
        joint_order: list[str],
        first_point: JointTrajectoryPoint,
        spec: ControllerSpec,
    ) -> str | None:
        require_state = bool(self.get_parameter("require_current_state").value)
        state_timeout = float(self.get_parameter("state_timeout_sec").value)
        allowed = float(self.get_parameter("allowed_start_tolerance_rad").value)

        with self._state_lock:
            age = time.monotonic() - self._latest_state_time if self._latest_state_time else math.inf
            current = {joint: self._latest_positions.get(joint) for joint in spec.joints}

        missing = [joint for joint, position in current.items() if position is None]
        if require_state and (missing or age > state_timeout):
            return f"missing/fresh joint state for {missing}; latest state age={age:.3f}s"

        if missing:
            return None

        target = self._point_positions(first_point, joint_order, spec)
        deltas = {
            joint: abs(float(current[joint]) - target[index])
            for index, joint in enumerate(spec.joints)
        }
        worst_joint, worst_delta = max(deltas.items(), key=lambda item: item[1])
        if worst_delta > allowed:
            return (
                f"trajectory start differs from current state by {worst_delta:.3f} rad "
                f"at {worst_joint}; allowed={allowed:.3f} rad"
            )
        return None

    def _wait_for_goal_tolerance(self, spec: ControllerSpec, target: list[float]) -> str | None:
        enforce = bool(self.get_parameter("enforce_goal_tolerance").value)
        if not enforce:
            return None

        tolerance = float(self.get_parameter("goal_tolerance_rad").value)
        timeout = float(self.get_parameter("goal_settle_time_sec").value)
        deadline = time.monotonic() + max(timeout, 0.0)
        while time.monotonic() <= deadline:
            with self._state_lock:
                current = [self._latest_positions.get(joint) for joint in spec.joints]
            if all(value is not None for value in current):
                errors = [abs(float(value) - desired) for value, desired in zip(current, target)]
                if max(errors, default=0.0) <= tolerance:
                    return None
            time.sleep(0.02)
        return f"final joint state did not settle within {tolerance:.3f} rad"

    def _point_positions(
        self, point: JointTrajectoryPoint, joint_order: list[str], spec: ControllerSpec
    ) -> list[float]:
        by_name = {joint: float(point.positions[index]) for index, joint in enumerate(joint_order)}
        return [by_name[joint] for joint in spec.joints]

    def _publish_command(self, spec: ControllerSpec, positions: list[float], *, dry_run: bool) -> None:
        if dry_run:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(spec.joints)
        msg.position = [float(value) for value in positions]
        self._command_publishers[spec.name].publish(msg)

    def _publish_feedback(
        self,
        goal_handle,
        spec: ControllerSpec,
        desired_positions: list[float],
        feedback: FollowJointTrajectory.Feedback,
    ) -> None:
        now_msg = self.get_clock().now().to_msg()
        feedback.header.stamp = now_msg
        feedback.desired = JointTrajectoryPoint()
        feedback.desired.positions = [float(value) for value in desired_positions]

        with self._state_lock:
            actual_positions = [self._latest_positions.get(joint) for joint in spec.joints]
        if all(position is not None for position in actual_positions):
            feedback.actual = JointTrajectoryPoint()
            feedback.actual.positions = [float(value) for value in actual_positions]
            feedback.error = JointTrajectoryPoint()
            feedback.error.positions = [
                float(actual - desired)
                for actual, desired in zip(actual_positions, desired_positions)
            ]
        goal_handle.publish_feedback(feedback)

    def _hold_current_or_last(self, spec: ControllerSpec, last: list[float], *, dry_run: bool) -> None:
        with self._state_lock:
            current = [self._latest_positions.get(joint) for joint in spec.joints]
        if all(position is not None for position in current):
            self._publish_command(spec, [float(value) for value in current], dry_run=dry_run)
        else:
            self._publish_command(spec, last, dry_run=dry_run)

    @staticmethod
    def _seconds_from_start(point: JointTrajectoryPoint) -> float:
        return float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9


def main() -> None:
    rclpy.init()
    node = DexcontrolTrajectoryAdapter()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
