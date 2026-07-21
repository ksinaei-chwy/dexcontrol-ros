#!/usr/bin/env python3
"""ROS 2 node that snapshots teleop demonstrations into LeRobotDataset v3."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from .camera_sources import (
    CameraValidationError,
    DirectRgbCameraSource,
    PlaceholderCameraSource,
)
from .configuration import CAMERA_FEATURES, CameraConfig, RecorderConfig, load_config
from .dataset_writer import LeRobotDatasetWriter
from .episode_controller import (
    EpisodeController,
    EpisodeState,
    InvalidTransition,
)
from .input_controller import KeyboardInputController
from .robot_feature_adapter import RobotFeatureAdapter, SnapshotValidationError


class _FrameWorker:
    """Bound disk/writer work away from ROS subscription and timer callbacks."""

    def __init__(
        self,
        controller: EpisodeController,
        *,
        capacity: int,
        on_error: Callable[[Exception], None],
    ) -> None:
        self.controller = controller
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(capacity)
        self._on_error = on_error
        self._thread = threading.Thread(
            target=self._run, name="lerobot-frame-writer", daemon=True
        )
        self._thread.start()

    def enqueue(self, frame: dict[str, Any]) -> bool:
        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            return False

    def flush(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        self._queue.put(None)
        self._queue.join()
        self._thread.join(timeout=5.0)

    @property
    def queued_frames(self) -> int:
        return self._queue.qsize()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self.controller.add_frame(item)
            except Exception as exc:  # noqa: BLE001 - worker boundary
                self._on_error(exc)
            finally:
                self._queue.task_done()


class VegaLeRobotRecorderNode(Node):
    """Observe applied commands, measured state, and RGB images at fixed FPS."""

    def __init__(
        self,
        *,
        cli_config_file: str | None = None,
        cli_no_hf_upload: bool = False,
        cli_overwrite: bool = False,
        cli_resume: bool = False,
        cli_input_backend: str | None = None,
        cli_input_device: str | None = None,
        writer_factory: Callable[..., LeRobotDatasetWriter] = LeRobotDatasetWriter,
        camera_source_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__("dex_vega_lerobot_recorder")
        if cli_config_file:
            default_config = cli_config_file
        else:
            try:
                share = Path(
                    get_package_share_directory("dex_vega_lerobot_recorder")
                )
            except Exception:
                share = Path(__file__).resolve().parents[1]
            default_config = str(share / "config" / "vega_lerobot_recording.yaml")
        self.declare_parameter("config_file", cli_config_file or default_config)
        self.declare_parameter("no_hf_upload", False)
        self.declare_parameter("overwrite", False)
        self.declare_parameter("resume", False)
        self.declare_parameter("input_backend", "")
        self.declare_parameter("input_device", "")

        config_path = cli_config_file or str(self.get_parameter("config_file").value)
        no_upload = cli_no_hf_upload or bool(self.get_parameter("no_hf_upload").value)
        overwrite = cli_overwrite or bool(self.get_parameter("overwrite").value)
        resume = cli_resume or bool(self.get_parameter("resume").value)
        input_backend = cli_input_backend or str(
            self.get_parameter("input_backend").value
        )
        input_device = cli_input_device or str(self.get_parameter("input_device").value)
        self.config: RecorderConfig = load_config(config_path).with_overrides(
            no_hf_upload=no_upload,
            input_backend=input_backend,
            input_device=input_device,
        )
        self._shutdown_complete = False
        self._lifecycle_lock = threading.RLock()
        self._last_warning: dict[str, float] = {}

        self._adapter = RobotFeatureAdapter(
            self.config.robot_features.joint_names,
            include_joint_velocities=(
                self.config.robot_features.include_joint_velocities
            ),
            hand_synergies=self.config.robot_features.hand_synergies,
        )
        self._camera_source_factory = camera_source_factory
        self._head_source = self._make_direct_camera(self.config.head_camera)
        self._left_source = self._make_optional_camera(
            self.config.left_wrist_camera
        )
        self._right_source = self._make_optional_camera(
            self.config.right_wrist_camera
        )

        writer = writer_factory(
            self.config,
            overwrite=overwrite,
            resume=resume,
            log_info=self.get_logger().info,
            log_warn=self.get_logger().warn,
        )
        self._controller = EpisodeController(
            writer,
            minimum_frames=self.config.episode_control.minimum_frames,
            minimum_duration_seconds=(
                self.config.episode_control.minimum_duration_seconds
            ),
        )
        self._frame_worker = _FrameWorker(
            self._controller,
            capacity=self.config.writer.frame_queue_size,
            on_error=self._on_worker_error,
        )

        self._create_subscriptions()
        self._create_services()
        self._record_timer = self.create_timer(
            1.0 / self.config.dataset.recording_fps,
            self._on_recording_tick,
        )
        self._status_timer = self.create_timer(1.0, self._report_status)

        controls = self.config.episode_control
        self._input = KeyboardInputController(
            backend=controls.input_backend,
            device_path=controls.input_device,
            debounce_seconds=controls.debounce_seconds,
            on_key=self._on_key,
            log_info=self.get_logger().info,
            log_warn=self.get_logger().warn,
        )
        self._input.start()

        for name, camera in (
            ("left_wrist", self.config.left_wrist_camera),
            ("right_wrist", self.config.right_wrist_camera),
        ):
            if camera.enabled and camera.placeholder_enabled:
                self.get_logger().warn(
                    f"{name} placeholder enabled: configured future resolution "
                    f"{camera.resolution.width}x{camera.resolution.height}; effective "
                    "black-frame resolution is matched to processed head camera "
                    f"{self.config.head_camera.resolution.width}x"
                    f"{self.config.head_camera.resolution.height}"
                )
        if any(
            camera.enabled and camera.placeholder_enabled
            for camera in (
                self.config.left_wrist_camera,
                self.config.right_wrist_camera,
            )
        ):
            self.get_logger().warn(
                "black wrist placeholders are real unmasked RGB pixels; do not mix "
                "placeholder episodes into production training without an explicit decision"
            )
        self.get_logger().info(
            f"recorder ready at {self.config.dataset.recording_fps} FPS; "
            f"dataset={self.config.local_dataset_path}; repo_id={self.config.repo_id}"
        )
        self.get_logger().info(
            "head camera uses direct DexComm latest-frame transport; "
            "no ROS image subscription is created"
        )

    def _make_direct_camera(self, config: CameraConfig) -> DirectRgbCameraSource:
        kwargs: dict[str, Any] = {
            "width": config.resolution.width,
            "height": config.resolution.height,
            "stream_name": config.stream_name,
            "topic": config.topic,
            "transport": config.transport,
            "rtc_channel": config.rtc_channel,
            "codec": config.codec,
        }
        if self._camera_source_factory is not None:
            kwargs["source_factory"] = self._camera_source_factory
        return DirectRgbCameraSource(**kwargs)

    def _make_optional_camera(
        self, config: CameraConfig
    ) -> PlaceholderCameraSource | DirectRgbCameraSource | None:
        if not config.enabled:
            return None
        if config.placeholder_enabled:
            return PlaceholderCameraSource()
        return self._make_direct_camera(config)

    def _create_subscriptions(self) -> None:
        topics = self.config.topics
        self.create_subscription(JointState, topics.joint_states, self._on_joint_state, 20)
        self.create_subscription(
            JointState,
            topics.applied_joint_commands,
            self._on_applied_joint_command,
            20,
        )
        self.create_subscription(
            TwistStamped,
            topics.measured_base_twist,
            self._on_measured_base_twist,
            20,
        )
        self.create_subscription(
            TwistStamped,
            topics.applied_base_twist,
            self._on_applied_base_twist,
            20,
        )

    def _create_services(self) -> None:
        prefix = "/dex_vega_lerobot_recorder"
        self.create_service(Trigger, f"{prefix}/start", self._service_start)
        self.create_service(Trigger, f"{prefix}/stop", self._service_stop)
        self.create_service(Trigger, f"{prefix}/save", self._service_save)
        self.create_service(Trigger, f"{prefix}/discard", self._service_discard)

    def _on_joint_state(self, message: JointState) -> None:
        try:
            self._adapter.update_measured_joints(
                message.name,
                message.position,
                message.velocity,
                self._stamp_or_now(message),
            )
        except ValueError as exc:
            self._warn_throttled("joint_state", f"invalid measured joint state: {exc}")

    def _on_applied_joint_command(self, message: JointState) -> None:
        try:
            self._adapter.update_applied_joints(
                message.name,
                message.position,
                self._stamp_or_now(message),
            )
        except ValueError as exc:
            self._warn_throttled("joint_action", f"invalid applied joint action: {exc}")

    def _on_measured_base_twist(self, message: TwistStamped) -> None:
        try:
            self._adapter.update_measured_base(
                _twist_values(message), self._stamp_or_now(message)
            )
        except ValueError as exc:
            self._warn_throttled("base_state", f"invalid measured base state: {exc}")

    def _on_applied_base_twist(self, message: TwistStamped) -> None:
        try:
            self._adapter.update_applied_base(
                _twist_values(message), self._stamp_or_now(message)
            )
        except ValueError as exc:
            self._warn_throttled("base_action", f"invalid applied base action: {exc}")

    def _on_recording_tick(self) -> None:
        with self._lifecycle_lock:
            if self._controller.state is not EpisodeState.RECORDING:
                return
            now_ns = self.get_clock().now().nanoseconds
            try:
                head = self._head_source.snapshot(
                    now_ns,
                    maximum_receive_age_seconds=(
                        self.config.validation.maximum_receive_age_seconds
                    ),
                    maximum_capture_age_seconds=(
                        self.config.validation.maximum_capture_age_seconds
                    ),
                    maximum_transport_delay_seconds=(
                        self.config.validation.maximum_transport_delay_seconds
                    ),
                )
                features = self._adapter.snapshot(
                    now_ns,
                    maximum_state_age_seconds=(
                        self.config.validation.maximum_state_age_seconds
                    ),
                    maximum_action_age_seconds=(
                        self.config.validation.maximum_action_age_seconds
                    ),
                )
                frame = {
                    CAMERA_FEATURES[0]: head.rgb,
                    "observation.state": features.state,
                    "action": features.action,
                    "task": self.config.dataset.task_instruction,
                }
                for key, source in (
                    (CAMERA_FEATURES[1], self._left_source),
                    (CAMERA_FEATURES[2], self._right_source),
                ):
                    if source is not None:
                        frame[key] = self._camera_rgb(source, head.rgb, now_ns)
            except (CameraValidationError, SnapshotValidationError) as exc:
                stale = (
                    exc.stale
                    if isinstance(exc, SnapshotValidationError)
                    else "stale" in str(exc) or "missing" in str(exc)
                )
                self._controller.note_drop(stale=stale)
                self._warn_throttled("sample", f"dropping recording sample: {exc}")
                return
            if not self._frame_worker.enqueue(frame):
                self._controller.note_drop(stale=False)
                self._warn_throttled(
                    "queue", "dropping recording sample: writer queue is full"
                )

    def _camera_rgb(
        self,
        source: PlaceholderCameraSource | DirectRgbCameraSource,
        head_rgb: Any,
        now_ns: int,
    ) -> Any:
        if isinstance(source, PlaceholderCameraSource):
            return source.frame_for(head_rgb)
        return source.snapshot(
            now_ns,
            maximum_receive_age_seconds=(
                self.config.validation.maximum_receive_age_seconds
            ),
            maximum_capture_age_seconds=(
                self.config.validation.maximum_capture_age_seconds
            ),
            maximum_transport_delay_seconds=(
                self.config.validation.maximum_transport_delay_seconds
            ),
        ).rgb

    def start_episode(self) -> str:
        with self._lifecycle_lock:
            summary = self._controller.start_episode()
            message = (
                f"candidate recording started; requested FPS="
                f"{self.config.dataset.recording_fps}"
            )
            self.get_logger().info(message)
            return f"{message}; state={summary.state.value}"

    def stop_episode(self) -> str:
        with self._lifecycle_lock:
            self._frame_worker.flush()
            summary = self._controller.stop_episode()
            message = self._format_candidate("candidate stopped", summary)
            self.get_logger().info(message)
            return message

    def save_episode(self) -> str:
        with self._lifecycle_lock:
            result = self._controller.save_episode()
            message = (
                f"committed episode {result.episode_index} at {result.local_path}"
            )
            self.get_logger().info(message)
            return message

    def discard_episode(self) -> str:
        with self._lifecycle_lock:
            if self._controller.state is EpisodeState.RECORDING:
                self._frame_worker.flush()
            self._controller.discard_episode()
            message = "discarded candidate episode; committed episode count unchanged"
            self.get_logger().warn(message)
            return message

    def _service_start(self, _request: Trigger.Request, response: Trigger.Response):
        return self._service_call(response, self.start_episode)

    def _service_stop(self, _request: Trigger.Request, response: Trigger.Response):
        return self._service_call(response, self.stop_episode)

    def _service_save(self, _request: Trigger.Request, response: Trigger.Response):
        return self._service_call(response, self.save_episode)

    def _service_discard(self, _request: Trigger.Request, response: Trigger.Response):
        return self._service_call(response, self.discard_episode)

    @staticmethod
    def _service_call(response: Trigger.Response, operation: Callable[[], str]):
        try:
            response.message = operation()
            response.success = True
        except Exception as exc:  # noqa: BLE001 - service boundary
            response.success = False
            response.message = str(exc)
        return response

    def _on_key(self, key: str) -> None:
        controls = self.config.episode_control
        operation = {
            controls.start_key: self.start_episode,
            controls.stop_key: self.stop_episode,
            controls.save_key: self.save_episode,
            controls.discard_key: self.discard_episode,
        }.get(key)
        if operation is None:
            return
        try:
            operation()
        except InvalidTransition as exc:
            self.get_logger().warn(str(exc))
        except Exception as exc:  # noqa: BLE001 - physical input boundary
            self.get_logger().error(f"episode control failed: {exc}")

    def _report_status(self) -> None:
        summary = self._controller.summary()
        camera = self._head_source.stats()
        achieved = (
            summary.frames / summary.duration_seconds
            if summary.duration_seconds > 0.0
            else 0.0
        )
        self.get_logger().info(
            f"state={summary.state.value} requested_fps="
            f"{self.config.dataset.recording_fps} achieved_fps={achieved:.2f} "
            f"pending_frames={summary.frames} duration={summary.duration_seconds:.2f}s "
            f"dropped={summary.dropped_samples} stale={summary.stale_samples} "
            f"writer_queue={self._frame_worker.queued_frames} "
            f"camera_fps={camera.source_fps:.2f} "
            f"camera_invalid={camera.invalid_frames}"
        )

    def _on_worker_error(self, error: Exception) -> None:
        self._controller.set_error(error)
        self.get_logger().error(f"LeRobot frame writer failed: {error}")

    def _stamp_or_now(self, message: Any) -> int:
        stamp = message.header.stamp
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        return value if value > 0 else self.get_clock().now().nanoseconds

    def _warn_throttled(self, key: str, message: str, period: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._last_warning.get(key, float("-inf")) >= period:
            self._last_warning[key] = now
            self.get_logger().warn(message)

    @staticmethod
    def _format_candidate(prefix: str, summary: Any) -> str:
        return (
            f"{prefix}: frames={summary.frames}, duration="
            f"{summary.duration_seconds:.3f}s, dropped={summary.dropped_samples}, "
            f"stale={summary.stale_samples}, validation="
            f"{summary.validation_message}"
        )

    def destroy_node(self) -> bool:
        if not self._shutdown_complete:
            self._shutdown_complete = True
            self._record_timer.cancel()
            self._status_timer.cancel()
            self._input.stop()
            for source in (
                self._head_source,
                self._left_source,
                self._right_source,
            ):
                if isinstance(source, DirectRgbCameraSource):
                    source.shutdown()
            with self._lifecycle_lock:
                state = self._controller.state
                if state in {EpisodeState.RECORDING, EpisodeState.REVIEW_PENDING}:
                    self.get_logger().warn(
                        f"shutdown with unsaved episode in {state.value}; "
                        f"autosave={self.config.episode_control.autosave_on_shutdown}"
                    )
                self._frame_worker.flush()
                self._frame_worker.stop()
                try:
                    resolution = self._controller.shutdown(
                        autosave=(
                            self.config.episode_control.autosave_on_shutdown
                        )
                    )
                    self.get_logger().info(
                        f"recorder shutdown: {resolution}; finalized committed data at "
                        f"{self.config.local_dataset_path}"
                    )
                except Exception as exc:  # noqa: BLE001 - shutdown boundary
                    self.get_logger().error(f"dataset finalization failed: {exc}")
        return super().destroy_node()


def _twist_values(message: TwistStamped) -> tuple[float, float, float]:
    return (
        float(message.twist.linear.x),
        float(message.twist.linear.y),
        float(message.twist.angular.z),
    )


def _parse_arguments(args: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file")
    parser.add_argument("--no-hf-upload", action="store_true")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--overwrite", action="store_true")
    policy.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--input-backend", choices=("disabled", "terminal", "linux_input_event")
    )
    parser.add_argument("--input-device")
    return parser.parse_known_args(args)


def main(args: list[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if args is None else args)
    options, ros_args = _parse_arguments(raw_args)
    rclpy.init(args=ros_args)
    node: VegaLeRobotRecorderNode | None = None
    try:
        node = VegaLeRobotRecorderNode(
            cli_config_file=options.config_file,
            cli_no_hf_upload=options.no_hf_upload,
            cli_overwrite=options.overwrite,
            cli_resume=options.resume,
            cli_input_backend=options.input_backend,
            cli_input_device=options.input_device,
        )
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
