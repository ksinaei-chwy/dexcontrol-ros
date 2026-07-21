#!/usr/bin/env python3
"""Direct DexTop head-camera bridge for XRoboToolkit video outputs."""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
import traceback
from fractions import Fraction
from typing import Any

import numpy as np
import rclpy
from dex_camera_transport import DexCommCameraSource, StreamKind
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool

from .output_worker import LatestFrameOutputWorker


RGB_STREAM_NAMES = {"left_rgb", "right_rgb", "rgb"}
CAMERA_TRANSPORTS = {"rtc", "zenoh"}
OUTPUT_CODECS = {"auto", "h264", "vp8"}
SOURCE_CODECS = {"auto", "h264", "vp8"}
XR_TCP_PACKET_HEADER = struct.Struct(">I")


def normalize_rgb_frame(frame: Any) -> np.ndarray:
    """Return a contiguous uint8 RGB frame from decoded camera data."""
    if hasattr(frame, "data"):
        frame = frame.data
    if isinstance(frame, dict) and "data" in frame:
        frame = frame["data"]
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected RGB image with shape HxWx3, got {image.shape}")
    if image.shape[2] > 3:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def resize_rgb_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize RGB using OpenCV, with a dependency-free nearest fallback."""
    frame = normalize_rgb_frame(frame)
    if width <= 0 or height <= 0:
        return frame
    src_height, src_width = frame.shape[:2]
    if src_width == width and src_height == height:
        return frame
    try:
        import cv2

        return np.ascontiguousarray(
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        )
    except ImportError:
        y_index = np.linspace(0, src_height - 1, height).astype(np.intp)
        x_index = np.linspace(0, src_width - 1, width).astype(np.intp)
        return np.ascontiguousarray(frame[y_index][:, x_index])


def make_side_by_side_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """Return a stereo side-by-side frame by duplicating one RGB image."""
    return np.ascontiguousarray(np.concatenate((frame, frame), axis=1))


def xrobotoolkit_packet_header(payload_size: int) -> bytes:
    """Build the 4-byte big-endian packet header used by ZED Mini mode."""
    if payload_size < 0 or payload_size > 0xFFFFFFFF:
        raise ValueError("payload_size must fit in an unsigned 32-bit integer")
    return XR_TCP_PACKET_HEADER.pack(payload_size)


class XRobotoolkitTcpH264Publisher:
    """Send H.264 packets to XRoboToolkit's optional ZED Mini TCP listener."""

    def __init__(
        self,
        host: str,
        port: int,
        width: int,
        height: int,
        fps: int,
        bitrate: int,
        side_by_side: bool = True,
        connect_timeout_s: float = 0.25,
        write_timeout_s: float = 2.0,
        reconnect_interval_s: float = 1.0,
    ) -> None:
        if not host:
            raise ValueError("xrtcp_host is required when xrtcp_enabled is true")
        if port <= 0 or port > 65535:
            raise ValueError("xrtcp_port must be between 1 and 65535")
        if width <= 0 or height <= 0:
            raise ValueError("XRoboToolkit TCP output needs fixed dimensions")
        self.host = host
        self.port = int(port)
        self.source_width = int(width)
        self.source_height = int(height)
        self.side_by_side = bool(side_by_side)
        self.width = self.source_width * 2 if self.side_by_side else self.source_width
        self.height = self.source_height
        self.fps = max(int(fps), 1)
        self.bitrate = int(bitrate)
        self.connect_timeout_s = max(float(connect_timeout_s), 0.05)
        self.write_timeout_s = max(float(write_timeout_s), 0.1)
        self.reconnect_interval_s = max(float(reconnect_interval_s), 0.1)
        self._socket: socket.socket | None = None
        self._next_connect_time_s = 0.0
        self._codec: Any | None = None
        self._frame_index = 0
        self._connected = False
        self.output_frames = 0
        self.failures = 0
        self.last_error = ""
        self.last_publish_time_ns = 0
        self._reset_encoder()

    def publish(self, frame: np.ndarray) -> bool:
        """Encode and send one frame; called only by the TCP output worker."""
        sock = self._ensure_socket()
        if sock is None:
            return False
        try:
            output = (
                make_side_by_side_rgb_frame(frame)
                if self.side_by_side
                else frame
            )
            video_frame = self._video_frame(output)
            sent_packet = False
            for packet in self._codec.encode(video_frame):
                payload = bytes(packet)
                if not payload:
                    continue
                sock.sendall(xrobotoolkit_packet_header(len(payload)))
                sock.sendall(payload)
                sent_packet = True
            if sent_packet:
                self.output_frames += 1
                self.last_publish_time_ns = time.time_ns()
                self.last_error = ""
            return sent_packet
        except Exception as exc:  # noqa: BLE001 - network/encoder boundary
            self.failures += 1
            self.last_error = str(exc)
            self._close_socket()
            self._reset_encoder()
            return False

    def is_connected(self) -> bool:
        return self._connected and self._socket is not None

    def shutdown(self) -> None:
        self._close_socket()
        self._codec = None

    def _video_frame(self, frame: np.ndarray) -> Any:
        import av

        video_frame = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(frame), format="rgb24"
        )
        video_frame.pts = self._frame_index
        video_frame.time_base = Fraction(1, self.fps)
        self._frame_index += 1
        return video_frame

    def _ensure_socket(self) -> socket.socket | None:
        if self._socket is not None:
            return self._socket
        now_s = time.monotonic()
        if now_s < self._next_connect_time_s:
            return None
        self._next_connect_time_s = now_s + self.reconnect_interval_s
        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout_s
            )
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            sock.settimeout(self.write_timeout_s)
            self._socket = sock
            self._connected = True
            self.last_error = ""
            self._reset_encoder()
            return sock
        except OSError as exc:
            self.failures += 1
            self.last_error = f"connect: {exc}"
            self._connected = False
            return None

    def _close_socket(self) -> None:
        sock, self._socket = self._socket, None
        self._connected = False
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _reset_encoder(self) -> None:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "PyAV is required for XRoboToolkit ZED Mini TCP output. "
                "Install it with: python3 -m pip install av==17.1.0"
            ) from exc
        codec = av.CodecContext.create("libx264", "w")
        codec.width = self.width
        codec.height = self.height
        codec.time_base = Fraction(1, self.fps)
        codec.framerate = Fraction(self.fps, 1)
        codec.pix_fmt = "yuv420p"
        codec.bit_rate = self.bitrate
        codec.gop_size = self.fps
        codec.max_b_frames = 0
        codec.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "annexb": "1",
            "sc_threshold": "0",
            "x264-params": f"repeat-headers=1:keyint={self.fps}:scenecut=0",
        }
        codec.open()
        self._codec = codec
        self._frame_index = 0


class DexmateHeadCameraVisionNode(Node):
    """Read DexTop directly and fan frames into isolated headset workers."""

    def __init__(self) -> None:
        super().__init__("dexmate_head_camera_vision")
        self._declare_parameters()
        self._status_pub = self.create_publisher(
            String,
            "/dex_pico_teleop/head_camera_vision/status",
            int(self.get_parameter("qos_depth").value),
        )
        self.create_service(
            SetBool,
            "/dex_pico_teleop/head_camera_vision/enabled",
            self._on_enabled,
        )
        self._state_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._status = "stopped"
        self._last_error = ""
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._rgb_source: DexCommCameraSource | None = None
        self._depth_source: DexCommCameraSource | None = None
        self._rtc_publisher: Any | None = None
        self._rtc_codec = "disabled"
        self._rtc_worker: LatestFrameOutputWorker | None = None
        self._xrtcp_publisher: XRobotoolkitTcpH264Publisher | None = None
        self._xrtcp_worker: LatestFrameOutputWorker | None = None
        self._output_width = 0
        self._output_height = 0
        self._last_submitted_sequence = 0
        status_rate_hz = float(self.get_parameter("status_rate_hz").value)
        self._status_timer = self.create_timer(
            1.0 / max(status_rate_hz, 0.1), self._publish_status
        )
        if bool(self.get_parameter("enabled").value):
            self._start_stream()
        else:
            self._set_status("disabled")

    def _declare_parameters(self) -> None:
        self.declare_parameter("enabled", True)
        self.declare_parameter("qos_depth", 10)
        self.declare_parameter("sensor_name", "head_camera")
        self.declare_parameter("stream_name", "left_rgb")
        self.declare_parameter("camera_transport", "zenoh")
        self.declare_parameter("camera_topic", "sensors/head_camera/left_rgb")
        self.declare_parameter(
            "source_rtc_channel", "sensors/head_camera/left_rgb_rtc"
        )
        self.declare_parameter("source_codec", "auto")
        self.declare_parameter("depth_enabled", True)
        self.declare_parameter("depth_topic", "sensors/head_camera/depth")
        self.declare_parameter("rtc_enabled", True)
        self.declare_parameter(
            "rtc_channel",
            "xrobotoolkit/remote_vision/head_camera/left_rgb_rtc",
        )
        self.declare_parameter("rtc_profile", "local")
        self.declare_parameter("codec", "auto")
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("bitrate", 1500000)
        self.declare_parameter("xrtcp_enabled", False)
        self.declare_parameter("xrtcp_host", "")
        self.declare_parameter("xrtcp_port", 12345)
        self.declare_parameter("xrtcp_side_by_side", True)
        self.declare_parameter("xrtcp_bitrate", 4000000)
        self.declare_parameter("xrtcp_connect_timeout_s", 0.25)
        self.declare_parameter("xrtcp_write_timeout_s", 2.0)
        self.declare_parameter("xrtcp_reconnect_interval_s", 1.0)
        self.declare_parameter("first_frame_timeout_s", 5.0)
        self.declare_parameter("status_rate_hz", 1.0)

    def _on_enabled(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        if request.data:
            started = self._start_stream()
            response.success = started
            response.message = (
                "head camera vision enabled" if started else "stream already running"
            )
        else:
            self._stop_stream()
            self._set_status("disabled")
            response.success = True
            response.message = "head camera vision disabled"
        return response

    def _start_stream(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_stream,
            name="dexmate_head_camera_vision",
            daemon=True,
        )
        self._thread.start()
        return True

    def _stop_stream(self) -> None:
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            try:
                thread.join(timeout=3.0)
            except KeyboardInterrupt:
                self.get_logger().warn("interrupted while stopping vision worker")
        self._shutdown_io()

    def _run_stream(self) -> None:
        try:
            self._set_status("starting")
            self._rgb_source = self._make_rgb_source()
            if bool(self.get_parameter("depth_enabled").value):
                self._depth_source = DexCommCameraSource(
                    stream_name="depth",
                    stream_kind=StreamKind.DEPTH,
                    topic=str(self.get_parameter("depth_topic").value),
                    transport="zenoh",
                )
            timeout = float(self.get_parameter("first_frame_timeout_s").value)
            first = self._rgb_source.wait_for_frame(timeout)
            if first is None:
                raise RuntimeError(
                    f"no direct RGB frame received within {timeout:.1f}s"
                )
            self._output_width, self._output_height = self._output_dimensions(
                first.data
            )
            self._prepare_outputs()
            if self._rtc_worker is None and self._xrtcp_worker is None:
                raise RuntimeError("no headset video outputs are enabled")
            self._set_status("streaming")
            while not self._stop_event.is_set():
                frame = self._rgb_source.latest()
                if frame is not None and frame.sequence != self._last_submitted_sequence:
                    self._last_submitted_sequence = frame.sequence
                    if self._rtc_worker is not None:
                        self._rtc_worker.submit(frame.data, frame.sequence)
                    if self._xrtcp_worker is not None:
                        self._xrtcp_worker.submit(frame.data, frame.sequence)
                self._stop_event.wait(0.001)
        except Exception as exc:  # noqa: BLE001 - runtime hardware boundary
            self.get_logger().error(f"head camera vision failed: {exc}")
            self.get_logger().debug(traceback.format_exc())
            with self._state_lock:
                self._status = "error"
                self._last_error = str(exc)
        finally:
            self._shutdown_io()
            if self._stop_event.is_set():
                self._set_status("stopped")

    def _make_rgb_source(self) -> DexCommCameraSource:
        stream_name = str(self.get_parameter("stream_name").value).lower()
        transport = str(self.get_parameter("camera_transport").value).lower()
        source_codec = str(self.get_parameter("source_codec").value).lower()
        if stream_name not in RGB_STREAM_NAMES:
            raise ValueError(f"stream_name must be one of {sorted(RGB_STREAM_NAMES)}")
        if transport not in CAMERA_TRANSPORTS:
            raise ValueError(
                f"camera_transport must be one of {sorted(CAMERA_TRANSPORTS)}"
            )
        if source_codec not in SOURCE_CODECS:
            raise ValueError(f"source_codec must be one of {sorted(SOURCE_CODECS)}")
        topic = str(self.get_parameter("camera_topic").value)
        self.get_logger().info(
            f"subscribing directly to DexTop RGB '{topic}' via {transport}; "
            "no ROS image mirror is created"
        )
        return DexCommCameraSource(
            stream_name=stream_name,
            stream_kind=StreamKind.RGB,
            topic=topic,
            transport=transport,
            rtc_channel=str(self.get_parameter("source_rtc_channel").value),
            codec=source_codec,
        )

    def _output_dimensions(self, first_frame: np.ndarray) -> tuple[int, int]:
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        if width > 0 and height > 0:
            return width, height
        src_height, src_width = first_frame.shape[:2]
        return src_width, src_height

    def _prepare_outputs(self) -> None:
        def transform(frame: np.ndarray) -> np.ndarray:
            return resize_rgb_frame(
                frame, self._output_width, self._output_height
            )
        rtc_enabled = bool(self.get_parameter("rtc_enabled").value)
        xrtcp_enabled = bool(self.get_parameter("xrtcp_enabled").value)
        if rtc_enabled:
            try:
                self._rtc_publisher, self._rtc_codec = self._make_rtc_publisher()
                self._rtc_worker = LatestFrameOutputWorker(
                    name="rtc",
                    publish=self._publish_rtc,
                    transform=transform,
                )
            except Exception:
                if not xrtcp_enabled:
                    raise
                self.get_logger().warn(
                    "RTC output failed to start; continuing with optional ZED TCP"
                )
                self.get_logger().debug(traceback.format_exc())
        if xrtcp_enabled:
            self._xrtcp_publisher = self._make_xrtcp_publisher()
            self._xrtcp_worker = LatestFrameOutputWorker(
                name="xrtcp",
                publish=self._xrtcp_publisher.publish,
                transform=transform,
            )

    def _make_rtc_publisher(self) -> tuple[Any, str]:
        from dexcomm.rtc import VideoPublisher

        channel = str(self.get_parameter("rtc_channel").value)
        fps = int(round(float(self.get_parameter("fps").value)))
        bitrate = int(self.get_parameter("bitrate").value)
        errors: list[str] = []
        for codec_name, codec in self._video_codecs():
            try:
                publisher = VideoPublisher(
                    channel,
                    codec,
                    self._output_width,
                    self._output_height,
                    fps,
                    bitrate,
                    self._rtc_config(),
                )
                self.get_logger().info(
                    "publishing XRoboToolkit Remote Vision RTC channel "
                    f"'{channel}' {self._output_width}x{self._output_height}@{fps} "
                    f"{codec_name.upper()} {bitrate}bps on an isolated worker"
                )
                return publisher, codec_name
            except Exception as exc:  # noqa: BLE001 - codec boundary
                errors.append(f"{codec_name}: {exc}")
                self.get_logger().warn(
                    f"failed to create {codec_name.upper()} RTC publisher: {exc}"
                )
        raise RuntimeError(
            "failed to create RTC video publisher " f"({'; '.join(errors)})"
        )

    def _publish_rtc(self, frame: np.ndarray) -> bool:
        assert self._rtc_publisher is not None
        self._rtc_publisher.publish(frame, bgr=False)
        return True

    def _make_xrtcp_publisher(self) -> XRobotoolkitTcpH264Publisher:
        publisher = XRobotoolkitTcpH264Publisher(
            host=str(self.get_parameter("xrtcp_host").value).strip(),
            port=int(self.get_parameter("xrtcp_port").value),
            width=self._output_width,
            height=self._output_height,
            fps=int(round(float(self.get_parameter("fps").value))),
            bitrate=int(self.get_parameter("xrtcp_bitrate").value),
            side_by_side=bool(self.get_parameter("xrtcp_side_by_side").value),
            connect_timeout_s=float(
                self.get_parameter("xrtcp_connect_timeout_s").value
            ),
            write_timeout_s=float(self.get_parameter("xrtcp_write_timeout_s").value),
            reconnect_interval_s=float(
                self.get_parameter("xrtcp_reconnect_interval_s").value
            ),
        )
        self.get_logger().warn(
            "ZED Mini TCP compatibility output is enabled on an isolated worker; "
            "RTC Remote Vision is the production path"
        )
        return publisher

    def _video_codecs(self) -> list[tuple[str, Any]]:
        from dexcomm.rtc import VideoCodec

        codec_name = str(self.get_parameter("codec").value).lower()
        if codec_name not in OUTPUT_CODECS:
            raise ValueError(f"codec must be one of {sorted(OUTPUT_CODECS)}")
        if codec_name in {"auto", "h264"}:
            return [("h264", VideoCodec.H264), ("vp8", VideoCodec.VP8)]
        return [("vp8", VideoCodec.VP8)]

    def _rtc_config(self) -> Any:
        from dexcomm.rtc import RtcConfig

        profile = str(self.get_parameter("rtc_profile").value).lower()
        if profile == "local":
            return RtcConfig.local()
        if profile == "internet":
            return RtcConfig.internet()
        raise ValueError("rtc_profile must be 'local' or 'internet'")

    def _set_status(self, status: str) -> None:
        with self._state_lock:
            self._status = status
            if status not in {"error"}:
                self._last_error = ""

    def _publish_status(self) -> None:
        now_ns = time.time_ns()
        with self._state_lock:
            payload: dict[str, Any] = {
                "status": self._status,
                "last_error": self._last_error,
                "transport": "direct_dexcomm",
                "raw_ros_image_transport": "removed",
                "output_width": self._output_width,
                "output_height": self._output_height,
                "rtc_codec": self._rtc_codec,
            }
        if self._rgb_source is not None:
            stats = self._rgb_source.stats()
            payload["rgb"] = self._source_status(stats, now_ns)
        if self._depth_source is not None:
            stats = self._depth_source.stats()
            payload["depth"] = self._source_status(stats, now_ns)
        if self._rtc_worker is not None:
            payload["rtc"] = self._worker_status(self._rtc_worker)
            if self._rtc_publisher is not None:
                try:
                    payload["rtc"]["connected"] = bool(
                        self._rtc_publisher.is_connected()
                    )
                    payload["rtc"]["subscriber_count"] = int(
                        self._rtc_publisher.subscriber_count()
                    )
                except Exception:
                    pass
        if self._xrtcp_worker is not None and self._xrtcp_publisher is not None:
            payload["xrtcp"] = self._worker_status(self._xrtcp_worker)
            payload["xrtcp"].update(
                {
                    "connected": self._xrtcp_publisher.is_connected(),
                    "encoder_failures": self._xrtcp_publisher.failures,
                    "encoder_last_error": self._xrtcp_publisher.last_error,
                }
            )
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self._status_pub.publish(msg)

    @staticmethod
    def _source_status(stats: Any, now_ns: int) -> dict[str, Any]:
        capture_age = (
            (now_ns - stats.last_source_stamp_ns) / 1.0e9
            if stats.last_source_stamp_ns
            else None
        )
        receive_age = (
            (now_ns - stats.last_receive_stamp_ns) / 1.0e9
            if stats.last_receive_stamp_ns
            else None
        )
        transport_delay = (
            (stats.last_receive_stamp_ns - stats.last_source_stamp_ns) / 1.0e9
            if stats.last_source_stamp_ns and stats.last_receive_stamp_ns
            else None
        )
        return {
            "unique_frames": stats.unique_frames,
            "invalid_frames": stats.invalid_frames,
            "source_fps": stats.source_fps,
            "last_sequence": stats.last_sequence,
            "capture_age_seconds": capture_age,
            "receive_age_seconds": receive_age,
            "transport_delay_seconds": transport_delay,
            "shape": list(stats.shape) if stats.shape else None,
            "dtype": stats.dtype,
            "last_error": stats.last_error,
        }

    @staticmethod
    def _worker_status(worker: LatestFrameOutputWorker) -> dict[str, Any]:
        stats = worker.stats()
        return {
            "enqueued_frames": stats.enqueued_frames,
            "published_frames": stats.published_frames,
            "replaced_frames": stats.replaced_frames,
            "failures": stats.failures,
            "last_sequence": stats.last_sequence,
            "queue_age_seconds": stats.last_queue_age_seconds,
            "processing_seconds": stats.last_processing_seconds,
            "last_error": stats.last_error,
        }

    def _shutdown_io(self) -> None:
        with self._io_lock:
            rtc_worker, self._rtc_worker = self._rtc_worker, None
            xrtcp_worker, self._xrtcp_worker = self._xrtcp_worker, None
            rtc_publisher, self._rtc_publisher = self._rtc_publisher, None
            xrtcp_publisher, self._xrtcp_publisher = self._xrtcp_publisher, None
            rgb_source, self._rgb_source = self._rgb_source, None
            depth_source, self._depth_source = self._depth_source, None
        for worker in (rtc_worker, xrtcp_worker):
            if worker is not None:
                worker.shutdown()
        for name, publisher in (
            ("RTC", rtc_publisher),
            ("XRoboToolkit TCP", xrtcp_publisher),
        ):
            if publisher is not None:
                try:
                    publisher.shutdown()
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"error shutting down {name}: {exc}")
        for source in (rgb_source, depth_source):
            if source is not None:
                source.shutdown()

    def destroy_node(self) -> None:
        try:
            self._stop_stream()
        finally:
            super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: DexmateHeadCameraVisionNode | None = None
    try:
        node = DexmateHeadCameraVisionNode()
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
