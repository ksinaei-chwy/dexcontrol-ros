#!/usr/bin/env python3
"""Dexmate head-camera bridge for XRoboToolkit video outputs."""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
import traceback
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool


RGB_STREAM_NAMES = {"left_rgb", "right_rgb", "rgb"}
CAMERA_TRANSPORTS = {"rtc", "zenoh"}
OUTPUT_CODECS = {"auto", "h264", "vp8"}
SOURCE_CODECS = {"auto", "h264", "vp8"}
XR_TCP_PACKET_HEADER = struct.Struct(">I")


def normalize_rgb_frame(frame: Any) -> np.ndarray:
    """Return a contiguous uint8 RGB frame from Dexmate camera data."""
    if hasattr(frame, "data"):
        frame = frame.data
    if isinstance(frame, dict) and "data" in frame:
        frame = frame["data"]

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(
            f"expected RGB image with shape HxWx3, got {image.shape}"
        )
    if image.shape[2] > 3:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def resize_rgb_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an RGB frame with a dependency-free nearest-neighbor fallback."""
    if width <= 0 or height <= 0:
        return np.ascontiguousarray(frame)

    src_height, src_width = frame.shape[:2]
    if src_width == width and src_height == height:
        return np.ascontiguousarray(frame)

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


def configure_head_camera_stream(
    configs: Any,
    sensor_name: str,
    stream_name: str,
    transport: str,
) -> Any:
    """Enable the head camera config and return one RGB stream config."""
    stream_name = stream_name.lower()
    transport = transport.lower()
    if stream_name not in RGB_STREAM_NAMES:
        raise ValueError(
            f"stream_name must be one of {sorted(RGB_STREAM_NAMES)}"
        )
    if transport not in CAMERA_TRANSPORTS:
        raise ValueError(
            f"camera_transport must be one of {sorted(CAMERA_TRANSPORTS)}"
        )
    if not configs.has_sensor(sensor_name):
        raise ValueError(
            f"Dexmate config does not define sensor '{sensor_name}'"
        )

    configs.enable_sensor(sensor_name)
    sensor_config = configs.sensors[sensor_name]
    if not hasattr(sensor_config, stream_name):
        raise ValueError(
            f"sensor '{sensor_name}' has no stream '{stream_name}'"
        )

    sensor_config.enabled = True
    if hasattr(sensor_config, "transport"):
        sensor_config.transport = transport
    if hasattr(sensor_config, "enable_rgb"):
        sensor_config.enable_rgb = True
    if hasattr(sensor_config, "enable_depth"):
        sensor_config.enable_depth = False

    stream_config = getattr(sensor_config, stream_name)
    stream_config.transport = transport
    return stream_config


class DexmateCameraStreamReader:
    """Read one Dexmate camera stream without creating robot components."""

    def __init__(
        self,
        sensor_name: str,
        stream_name: str,
        transport: str,
        codec: str = "h264",
    ) -> None:
        from dexcomm import Node as DexCommNode
        if transport == "rtc":
            from dexcomm.rtc import VideoCodec

            if not hasattr(VideoCodec, "H265") and hasattr(VideoCodec, "H264"):
                # dexcontrol versions that know about h265 build their codec
                # map eagerly. Older dexcomm builds expose only H264 and VP8.
                VideoCodec.H265 = VideoCodec.H264
        from dexcontrol.core.config import get_robot_config
        from dexcontrol.sensors.camera.base_camera import (
            StreamSubscriber,
            StreamType,
            TransportType,
        )

        configs = get_robot_config()
        stream_config = configure_head_camera_stream(
            configs,
            sensor_name=sensor_name,
            stream_name=stream_name,
            transport=transport,
        )

        self.stream_name = stream_name
        self._node = None
        if transport == "zenoh":
            self._node = DexCommNode(
                name=f"{sensor_name}_{stream_name}_vision_node"
            )

        self._stream = StreamSubscriber(
            stream_name=stream_name,
            transport=TransportType(transport),
            stream_type=StreamType.RGB,
            node=self._node,
            topic=getattr(stream_config, "topic", None),
            rtc_channel=getattr(stream_config, "rtc_channel", None),
            codec=codec,
            buffer_size=1,
        )

    def wait_for_active(
        self,
        timeout: float = 5.0,
        require_all: bool = False,
    ) -> bool:
        del require_all
        return self._stream.wait_for_message(timeout=timeout) is not None

    def get_obs(self, obs_keys: list[str] | None = None) -> dict[str, Any]:
        del obs_keys
        return {self.stream_name: self._stream.get_latest()}

    def shutdown(self) -> None:
        self._stream.shutdown()
        if self._node is not None:
            self._node.shutdown()


class XRobotoolkitTcpH264Publisher:
    """Send H.264 frames to XRoboToolkit's ZED Mini TCP listener."""

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
        self.port = port
        self.source_width = width
        self.source_height = height
        self.side_by_side = side_by_side
        self.width = width * 2 if side_by_side else width
        self.height = height
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
        sock = self._ensure_socket()
        if sock is None:
            return False

        try:
            output = make_side_by_side_rgb_frame(frame) if self.side_by_side else frame
            video_frame = self._video_frame(output)
            packets = self._codec.encode(video_frame)
            sent_packet = False
            for packet in packets:
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
            np.ascontiguousarray(frame),
            format="rgb24",
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
                (self.host, self.port),
                timeout=self.connect_timeout_s,
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


@dataclass
class VisionStats:
    status: str = "stopped"
    input_frames: int = 0
    output_frames: int = 0
    rtc_output_frames: int = 0
    read_failures: int = 0
    publish_failures: int = 0
    last_error: str = ""
    source_shape: tuple[int, ...] | None = None
    output_width: int = 0
    output_height: int = 0
    output_codec: str = ""
    last_input_time_ns: int = 0
    last_publish_time_ns: int = 0
    connected: bool = False
    subscriber_count: int = 0
    rtc_enabled: bool = True
    xrtcp_enabled: bool = False
    xrtcp_host: str = ""
    xrtcp_port: int = 12345
    xrtcp_connected: bool = False
    xrtcp_output_frames: int = 0
    xrtcp_failures: int = 0
    xrtcp_last_error: str = ""
    xrtcp_last_publish_time_ns: int = 0
    xrtcp_output_width: int = 0
    xrtcp_output_height: int = 0
    xrtcp_side_by_side: bool = True

    def to_dict(self) -> dict[str, object]:
        now_ns = time.time_ns()
        input_age_s = (
            (now_ns - self.last_input_time_ns) / 1.0e9
            if self.last_input_time_ns
            else None
        )
        publish_age_s = (
            (now_ns - self.last_publish_time_ns) / 1.0e9
            if self.last_publish_time_ns
            else None
        )
        xrtcp_publish_age_s = (
            (now_ns - self.xrtcp_last_publish_time_ns) / 1.0e9
            if self.xrtcp_last_publish_time_ns
            else None
        )
        return {
            "status": self.status,
            "input_frames": self.input_frames,
            "output_frames": self.output_frames,
            "rtc_output_frames": self.rtc_output_frames,
            "read_failures": self.read_failures,
            "publish_failures": self.publish_failures,
            "last_error": self.last_error,
            "source_shape": (
                list(self.source_shape) if self.source_shape else None
            ),
            "output_width": self.output_width,
            "output_height": self.output_height,
            "output_codec": self.output_codec,
            "last_input_age_s": input_age_s,
            "last_publish_age_s": publish_age_s,
            "connected": self.connected,
            "subscriber_count": self.subscriber_count,
            "rtc_enabled": self.rtc_enabled,
            "xrtcp_enabled": self.xrtcp_enabled,
            "xrtcp_host": self.xrtcp_host,
            "xrtcp_port": self.xrtcp_port,
            "xrtcp_connected": self.xrtcp_connected,
            "xrtcp_output_frames": self.xrtcp_output_frames,
            "xrtcp_failures": self.xrtcp_failures,
            "xrtcp_last_error": self.xrtcp_last_error,
            "xrtcp_last_publish_age_s": xrtcp_publish_age_s,
            "xrtcp_output_width": self.xrtcp_output_width,
            "xrtcp_output_height": self.xrtcp_output_height,
            "xrtcp_side_by_side": self.xrtcp_side_by_side,
        }


class DexmateHeadCameraVisionNode(Node):
    """Read Dexmate head-camera RGB frames and publish headset video outputs."""

    def __init__(self) -> None:
        super().__init__("dexmate_head_camera_vision")
        self._declare_parameters()

        qos_depth = int(self.get_parameter("qos_depth").value)
        self._status_pub = self.create_publisher(
            String,
            "/dex_pico_teleop/head_camera_vision/status",
            qos_depth,
        )
        self.create_service(
            SetBool,
            "/dex_pico_teleop/head_camera_vision/enabled",
            self._on_enabled,
        )

        self._stats = VisionStats()
        self._stats_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._camera: Any | None = None
        self._publisher: Any | None = None
        self._xrtcp_publisher: XRobotoolkitTcpH264Publisher | None = None

        status_rate_hz = float(self.get_parameter("status_rate_hz").value)
        self._status_timer = self.create_timer(
            1.0 / max(status_rate_hz, 0.1),
            self._publish_status,
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
        self.declare_parameter("rtc_enabled", True)
        self.declare_parameter(
            "rtc_channel",
            "xrobotoolkit/remote_vision/head_camera/left_rgb_rtc",
        )
        self.declare_parameter("rtc_profile", "local")
        self.declare_parameter("source_codec", "h264")
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
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        if request.data:
            started = self._start_stream()
            response.success = started
            response.message = (
                "head camera vision enabled"
                if started
                else "stream already running"
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
        if self._thread is not None:
            try:
                self._thread.join(timeout=2.0)
            except KeyboardInterrupt:
                self.get_logger().warn(
                    "interrupted while waiting for vision worker shutdown"
                )
            self._thread = None
        self._shutdown_io()

    def _run_stream(self) -> None:
        try:
            self._set_status("starting")
            self._camera = self._make_camera()
            self._wait_for_camera()

            first_frame = self._wait_for_first_frame()
            output_width, output_height = self._output_dimensions(first_frame)
            self._prepare_outputs(output_width, output_height)
            if self._publisher is None and self._xrtcp_publisher is None:
                raise RuntimeError("no video outputs are enabled")
            with self._stats_lock:
                self._stats.output_width = output_width
                self._stats.output_height = output_height
            self._set_status("streaming")

            if first_frame is not None:
                self._publish_frame(first_frame)

            period_s = 1.0 / max(float(self.get_parameter("fps").value), 1.0)
            next_publish_s = time.monotonic()
            while not self._stop_event.is_set():
                frame = self._read_frame()
                if frame is None:
                    with self._stats_lock:
                        self._stats.read_failures += 1
                else:
                    self._publish_frame(frame)

                next_publish_s += period_s
                sleep_s = next_publish_s - time.monotonic()
                if sleep_s > 0.0:
                    self._stop_event.wait(sleep_s)
                else:
                    next_publish_s = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - runtime hardware boundary
            self.get_logger().error(f"head camera vision failed: {exc}")
            self.get_logger().debug(traceback.format_exc())
            with self._stats_lock:
                self._stats.status = "error"
                self._stats.last_error = str(exc)
        finally:
            self._shutdown_io()
            if self._stop_event.is_set():
                self._set_status("stopped")

    def _prepare_outputs(self, output_width: int, output_height: int) -> None:
        rtc_enabled = bool(self.get_parameter("rtc_enabled").value)
        xrtcp_enabled = bool(self.get_parameter("xrtcp_enabled").value)
        with self._stats_lock:
            self._stats.rtc_enabled = rtc_enabled
            self._stats.xrtcp_enabled = xrtcp_enabled
            if not rtc_enabled:
                self._stats.output_codec = "disabled"

        if rtc_enabled:
            try:
                publisher, output_codec = self._make_publisher(
                    output_width,
                    output_height,
                )
                self._publisher = publisher
                with self._stats_lock:
                    self._stats.output_codec = output_codec
            except Exception:
                if not xrtcp_enabled:
                    raise
                self.get_logger().warn(
                    "RTC publisher could not be started; continuing with "
                    "XRoboToolkit ZED Mini TCP output"
                )
                self.get_logger().debug(traceback.format_exc())

        if xrtcp_enabled:
            self._xrtcp_publisher = self._make_xrtcp_publisher(
                output_width,
                output_height,
            )

    def _make_camera(self) -> Any:
        sensor_name = str(self.get_parameter("sensor_name").value)
        stream_name = str(self.get_parameter("stream_name").value)
        transport = str(self.get_parameter("camera_transport").value)
        source_codec = str(self.get_parameter("source_codec").value).lower()
        if source_codec not in SOURCE_CODECS:
            raise ValueError(
                f"source_codec must be one of {sorted(SOURCE_CODECS)}"
            )

        self.get_logger().info(
            "starting Dexmate camera "
            f"'{sensor_name}.{stream_name}' via {transport}"
        )
        return DexmateCameraStreamReader(
            sensor_name=sensor_name,
            stream_name=stream_name,
            transport=transport,
            codec=source_codec,
        )

    def _wait_for_camera(self) -> None:
        timeout_s = float(self.get_parameter("first_frame_timeout_s").value)
        if hasattr(self._camera, "wait_for_active"):
            active = bool(
                self._camera.wait_for_active(
                    timeout=timeout_s,
                    require_all=False,
                )
            )
            if not active:
                self.get_logger().warn(
                    "camera stream did not become active within "
                    f"{timeout_s:.1f}s"
                )

    def _wait_for_first_frame(self) -> np.ndarray | None:
        timeout_s = float(self.get_parameter("first_frame_timeout_s").value)
        deadline_s = time.monotonic() + max(timeout_s, 0.0)
        while not self._stop_event.is_set() and time.monotonic() <= deadline_s:
            frame = self._read_frame()
            if frame is not None:
                return frame
            time.sleep(0.02)

        if (
            int(self.get_parameter("width").value) <= 0
            or int(self.get_parameter("height").value) <= 0
        ):
            raise RuntimeError(
                "no camera frame available to auto-detect output dimensions"
            )
        self.get_logger().warn(
            "no first camera frame yet; starting RTC publisher with "
            "configured dimensions"
        )
        return None

    def _read_frame(self) -> np.ndarray | None:
        if self._camera is None:
            return None
        stream_name = str(self.get_parameter("stream_name").value)
        try:
            obs = self._camera.get_obs(obs_keys=[stream_name])
            raw_frame = obs.get(stream_name) if isinstance(obs, dict) else obs
            if raw_frame is None:
                return None
            frame = normalize_rgb_frame(raw_frame)
            with self._stats_lock:
                self._stats.input_frames += 1
                self._stats.source_shape = tuple(
                    int(value) for value in frame.shape
                )
                self._stats.last_input_time_ns = time.time_ns()
                self._stats.last_error = ""
            return frame
        except Exception as exc:  # noqa: BLE001 - runtime hardware boundary
            with self._stats_lock:
                self._stats.read_failures += 1
                self._stats.last_error = f"read: {exc}"
            return None

    def _output_dimensions(
        self,
        first_frame: np.ndarray | None,
    ) -> tuple[int, int]:
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        if width > 0 and height > 0:
            return width, height
        assert first_frame is not None
        src_height, src_width = first_frame.shape[:2]
        return src_width, src_height

    def _make_publisher(self, width: int, height: int) -> tuple[Any, str]:
        from dexcomm.rtc import VideoPublisher

        channel = str(self.get_parameter("rtc_channel").value)
        fps = int(round(float(self.get_parameter("fps").value)))
        bitrate = int(self.get_parameter("bitrate").value)
        config = self._rtc_config()
        errors: list[str] = []
        for codec_name, codec in self._video_codecs():
            try:
                publisher = VideoPublisher(
                    channel,
                    codec,
                    width,
                    height,
                    fps,
                    bitrate,
                    config,
                )
                self.get_logger().info(
                    "publishing XRoboToolkit Remote Vision RTC channel "
                    f"'{channel}' {width}x{height}@{fps} "
                    f"{codec_name.upper()} {bitrate}bps"
                )
                return publisher, codec_name
            except Exception as exc:  # noqa: BLE001 - codec boundary
                errors.append(f"{codec_name}: {exc}")
                self.get_logger().warn(
                    f"failed to create {codec_name.upper()} RTC publisher: {exc}"
                )

        raise RuntimeError(
            "failed to create an RTC video publisher with available codecs "
            f"({'; '.join(errors)}). Install the encoder-enabled dexcomm-video "
            "package or choose a codec supported by this runtime."
        )

    def _make_xrtcp_publisher(
        self,
        width: int,
        height: int,
    ) -> XRobotoolkitTcpH264Publisher:
        host = str(self.get_parameter("xrtcp_host").value).strip()
        port = int(self.get_parameter("xrtcp_port").value)
        fps = int(round(float(self.get_parameter("fps").value)))
        bitrate = int(self.get_parameter("xrtcp_bitrate").value)
        side_by_side = bool(self.get_parameter("xrtcp_side_by_side").value)
        connect_timeout_s = float(
            self.get_parameter("xrtcp_connect_timeout_s").value
        )
        write_timeout_s = float(
            self.get_parameter("xrtcp_write_timeout_s").value
        )
        reconnect_interval_s = float(
            self.get_parameter("xrtcp_reconnect_interval_s").value
        )

        publisher = XRobotoolkitTcpH264Publisher(
            host=host,
            port=port,
            width=width,
            height=height,
            fps=fps,
            bitrate=bitrate,
            side_by_side=side_by_side,
            connect_timeout_s=connect_timeout_s,
            write_timeout_s=write_timeout_s,
            reconnect_interval_s=reconnect_interval_s,
        )
        with self._stats_lock:
            self._stats.xrtcp_host = host
            self._stats.xrtcp_port = port
            self._stats.xrtcp_side_by_side = side_by_side
            self._stats.xrtcp_output_width = publisher.width
            self._stats.xrtcp_output_height = publisher.height

        self.get_logger().info(
            "publishing XRoboToolkit ZED Mini TCP stream to "
            f"{host}:{port} {publisher.width}x{publisher.height}@{fps} "
            f"H264 {bitrate}bps"
        )
        return publisher

    def _video_codecs(self) -> list[tuple[str, Any]]:
        from dexcomm.rtc import VideoCodec

        codec_name = str(self.get_parameter("codec").value).lower()
        if codec_name not in OUTPUT_CODECS:
            raise ValueError(f"codec must be one of {sorted(OUTPUT_CODECS)}")
        if codec_name == "auto":
            return [("h264", VideoCodec.H264), ("vp8", VideoCodec.VP8)]
        if codec_name == "h264":
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

    def _publish_frame(self, frame: np.ndarray) -> None:
        if self._publisher is None and self._xrtcp_publisher is None:
            return
        width = int(self._stats.output_width)
        height = int(self._stats.output_height)
        try:
            output = resize_rgb_frame(frame, width, height)
        except Exception as exc:  # noqa: BLE001 - runtime hardware boundary
            with self._stats_lock:
                self._stats.publish_failures += 1
                self._stats.last_error = f"resize: {exc}"
            return

        frame_published = False
        if self._publisher is not None:
            try:
                self._publisher.publish(output, bgr=False)
                connected = bool(self._publisher.is_connected())
                subscriber_count = int(self._publisher.subscriber_count())
                with self._stats_lock:
                    self._stats.rtc_output_frames += 1
                    self._stats.connected = connected
                    self._stats.subscriber_count = subscriber_count
                    self._stats.last_error = ""
                frame_published = True
            except Exception as exc:  # noqa: BLE001 - runtime hardware boundary
                with self._stats_lock:
                    self._stats.publish_failures += 1
                    self._stats.last_error = f"rtc publish: {exc}"

        if self._xrtcp_publisher is not None:
            sent = self._xrtcp_publisher.publish(output)
            with self._stats_lock:
                self._stats.xrtcp_connected = (
                    self._xrtcp_publisher.is_connected()
                )
                self._stats.xrtcp_output_frames = (
                    self._xrtcp_publisher.output_frames
                )
                self._stats.xrtcp_failures = self._xrtcp_publisher.failures
                self._stats.xrtcp_last_error = (
                    self._xrtcp_publisher.last_error
                )
                self._stats.xrtcp_last_publish_time_ns = (
                    self._xrtcp_publisher.last_publish_time_ns
                )
            frame_published = frame_published or sent

        if frame_published:
            with self._stats_lock:
                self._stats.output_frames += 1
                self._stats.last_publish_time_ns = time.time_ns()
        elif self._xrtcp_publisher is None:
            with self._stats_lock:
                self._stats.publish_failures += 1
                self._stats.last_error = "no output accepted frame"

    def _refresh_output_status(self) -> None:
        if self._publisher is not None:
            try:
                with self._stats_lock:
                    self._stats.connected = bool(
                        self._publisher.is_connected()
                    )
                    self._stats.subscriber_count = int(
                        self._publisher.subscriber_count()
                    )
            except Exception:
                pass

        if self._xrtcp_publisher is not None:
            with self._stats_lock:
                self._stats.xrtcp_connected = (
                    self._xrtcp_publisher.is_connected()
                )
                self._stats.xrtcp_output_frames = (
                    self._xrtcp_publisher.output_frames
                )
                self._stats.xrtcp_failures = self._xrtcp_publisher.failures
                self._stats.xrtcp_last_error = (
                    self._xrtcp_publisher.last_error
                )
                self._stats.xrtcp_last_publish_time_ns = (
                    self._xrtcp_publisher.last_publish_time_ns
                )

    def _set_status(self, status: str) -> None:
        with self._stats_lock:
            self._stats.status = status

    def _publish_status(self) -> None:
        self._refresh_output_status()
        msg = String()
        with self._stats_lock:
            msg.data = json.dumps(self._stats.to_dict(), sort_keys=True)
        self._status_pub.publish(msg)

    def _shutdown_io(self) -> None:
        publisher, self._publisher = self._publisher, None
        xrtcp_publisher, self._xrtcp_publisher = self._xrtcp_publisher, None
        camera, self._camera = self._camera, None
        if publisher is not None:
            try:
                publisher.shutdown()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"error shutting down RTC publisher: {exc}"
                )
        if xrtcp_publisher is not None:
            try:
                xrtcp_publisher.shutdown()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"error shutting down XRoboToolkit TCP publisher: {exc}"
                )
        if camera is not None:
            try:
                camera.shutdown()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"error shutting down camera sensor: {exc}"
                )

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
