"""Direct, latest-frame camera sources backed by DexComm.

This module deliberately has no ROS dependency.  DexTop timestamps are kept
alongside the decoded array so consumers can distinguish capture latency from
time spent waiting locally.
"""

from __future__ import annotations

import threading
import time
import os
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np


class CameraSourceError(RuntimeError):
    """Raised when a camera sample is missing, stale, or malformed."""


class StreamKind(str, Enum):
    """Payload kinds supported by the direct camera transport."""

    RGB = "rgb"
    DEPTH = "depth"


@dataclass(frozen=True)
class CameraFrame:
    """One decoded camera frame and its transport timing metadata."""

    data: np.ndarray
    source_stamp_ns: int
    receive_stamp_ns: int
    sequence: int

    def capture_age_seconds(self, now_ns: int) -> float:
        return (int(now_ns) - self.source_stamp_ns) / 1.0e9

    def receive_age_seconds(self, now_ns: int) -> float:
        return (int(now_ns) - self.receive_stamp_ns) / 1.0e9

    @property
    def transport_delay_seconds(self) -> float:
        return (self.receive_stamp_ns - self.source_stamp_ns) / 1.0e9


@dataclass(frozen=True)
class CameraSourceStats:
    """Thread-safe snapshot of source health and throughput."""

    unique_frames: int
    invalid_frames: int
    source_fps: float
    last_sequence: int
    last_source_stamp_ns: int
    last_receive_stamp_ns: int
    last_error: str
    shape: tuple[int, ...] | None
    dtype: str


class Subscriber(Protocol):
    """Subset of StreamSubscriber used by the source pump."""

    def get_latest(self) -> Any:
        """Return the latest decoded transport message."""

    def wait_for_message(self, timeout: float) -> Any:
        """Wait for the first decoded transport message."""

    def shutdown(self) -> None:
        """Release subscriber resources."""


SubscriberFactory = Callable[[], tuple[Subscriber, Callable[[], None]]]


class DexCommCameraSource:
    """Continuously capture the newest RGB or depth frame from DexComm.

    The internal pump never queues historical frames.  A consumer that is
    slower than the sensor always observes the newest decoded frame.
    """

    def __init__(
        self,
        *,
        stream_name: str,
        stream_kind: StreamKind | str,
        topic: str,
        transport: str = "zenoh",
        rtc_channel: str = "",
        codec: str = "auto",
        poll_interval_seconds: float = 0.001,
        subscriber_factory: SubscriberFactory | None = None,
        start: bool = True,
    ) -> None:
        self.stream_name = str(stream_name).strip()
        self.stream_kind = StreamKind(stream_kind)
        self.topic = str(topic).strip()
        self.transport = str(transport).strip().lower()
        self.rtc_channel = str(rtc_channel).strip()
        self.codec = str(codec).strip().lower()
        self.poll_interval_seconds = float(poll_interval_seconds)
        if not self.stream_name:
            raise ValueError("stream_name must not be empty")
        if self.transport not in {"zenoh", "rtc"}:
            raise ValueError("transport must be 'zenoh' or 'rtc'")
        if self.stream_kind is StreamKind.DEPTH and self.transport != "zenoh":
            raise ValueError("depth streams only support Zenoh transport")
        if self.transport == "zenoh" and not self.topic:
            raise ValueError("topic is required for Zenoh camera transport")
        if self.transport == "rtc" and not self.rtc_channel:
            raise ValueError("rtc_channel is required for RTC camera transport")
        if self.poll_interval_seconds <= 0.0:
            raise ValueError("poll_interval_seconds must be positive")

        self._subscriber_factory = subscriber_factory
        self._subscriber: Subscriber | None = None
        self._subscriber_shutdown: Callable[[], None] = lambda: None
        self._lock = threading.Lock()
        self._frame: CameraFrame | None = None
        self._last_transport_key: tuple[int, int] | None = None
        self._last_seen_key: tuple[int, int] | None = None
        self._unique_frames = 0
        self._invalid_frames = 0
        self._last_error = ""
        self._arrival_times: deque[float] = deque()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        if start:
            self.start()

    def start(self) -> None:
        """Create the transport subscriber and start the latest-frame pump."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self._subscriber is None:
            factory = self._subscriber_factory or self._make_subscriber
            self._subscriber, self._subscriber_shutdown = factory()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._pump,
            name=f"dex_camera_{self.stream_name}",
            daemon=True,
        )
        self._thread.start()

    def wait_for_frame(self, timeout_seconds: float = 5.0) -> CameraFrame | None:
        """Wait for and return the first valid frame."""
        deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
        while not self._stop_event.is_set() and time.monotonic() <= deadline:
            with self._lock:
                frame = self._frame
            if frame is not None:
                return frame
            time.sleep(min(self.poll_interval_seconds, 0.01))
        return None

    def latest(self) -> CameraFrame | None:
        """Return the current immutable frame envelope without copying pixels."""
        with self._lock:
            return self._frame

    def snapshot(
        self,
        *,
        now_ns: int,
        maximum_receive_age_seconds: float,
        maximum_capture_age_seconds: float,
        maximum_transport_delay_seconds: float,
    ) -> CameraFrame:
        """Return the newest frame after validating all freshness dimensions."""
        frame = self.latest()
        if frame is None:
            raise CameraSourceError(f"missing {self.stream_name} frame")

        capture_age = frame.capture_age_seconds(now_ns)
        receive_age = frame.receive_age_seconds(now_ns)
        transport_delay = frame.transport_delay_seconds
        for label, value in (
            ("capture age", capture_age),
            ("receive age", receive_age),
            ("transport delay", transport_delay),
        ):
            if value < 0.0:
                raise CameraSourceError(
                    f"{self.stream_name} {label} is negative ({value:.3f}s)"
                )
        if receive_age > maximum_receive_age_seconds:
            raise CameraSourceError(
                f"stale {self.stream_name} receive age "
                f"({receive_age:.3f}s > {maximum_receive_age_seconds:.3f}s)"
            )
        if capture_age > maximum_capture_age_seconds:
            raise CameraSourceError(
                f"stale {self.stream_name} capture age "
                f"({capture_age:.3f}s > {maximum_capture_age_seconds:.3f}s)"
            )
        if transport_delay > maximum_transport_delay_seconds:
            raise CameraSourceError(
                f"stale {self.stream_name} transport delay "
                f"({transport_delay:.3f}s > "
                f"{maximum_transport_delay_seconds:.3f}s)"
            )
        return frame

    def stats(self) -> CameraSourceStats:
        """Return current source statistics."""
        with self._lock:
            frame = self._frame
            return CameraSourceStats(
                unique_frames=self._unique_frames,
                invalid_frames=self._invalid_frames,
                source_fps=self._source_fps_locked(time.monotonic()),
                last_sequence=frame.sequence if frame is not None else 0,
                last_source_stamp_ns=(
                    frame.source_stamp_ns if frame is not None else 0
                ),
                last_receive_stamp_ns=(
                    frame.receive_stamp_ns if frame is not None else 0
                ),
                last_error=self._last_error,
                shape=(
                    tuple(int(value) for value in frame.data.shape)
                    if frame is not None
                    else None
                ),
                dtype=str(frame.data.dtype) if frame is not None else "",
            )

    def shutdown(self) -> None:
        """Stop capture and release DexComm resources."""
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        try:
            self._subscriber_shutdown()
        finally:
            self._subscriber = None

    def _pump(self) -> None:
        assert self._subscriber is not None
        while not self._stop_event.is_set():
            raw = self._subscriber.get_latest()
            if raw is not None:
                try:
                    self._accept(raw)
                except CameraSourceError as exc:
                    with self._lock:
                        self._invalid_frames += 1
                        self._last_error = str(exc)
            self._stop_event.wait(self.poll_interval_seconds)

    def _accept(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            raise CameraSourceError(
                f"{self.stream_name} transport omitted timestamp metadata"
            )
        if "data" not in raw:
            raise CameraSourceError(f"{self.stream_name} payload has no data")
        source_stamp_ns = int(raw.get("timestamp_ns", 0))
        receive_stamp_ns = int(raw.get("receive_time_ns", 0))
        if source_stamp_ns <= 0 or receive_stamp_ns <= 0:
            raise CameraSourceError(
                f"{self.stream_name} source/receive timestamps are unavailable"
            )
        transport_key = (source_stamp_ns, receive_stamp_ns)
        with self._lock:
            if transport_key == self._last_seen_key:
                return
            self._last_seen_key = transport_key

        data = self._validate_data(raw["data"])
        arrival = time.monotonic()
        with self._lock:
            if transport_key == self._last_transport_key:
                return
            self._unique_frames += 1
            self._last_transport_key = transport_key
            self._frame = CameraFrame(
                data=data,
                source_stamp_ns=source_stamp_ns,
                receive_stamp_ns=receive_stamp_ns,
                sequence=self._unique_frames,
            )
            self._arrival_times.append(arrival)
            self._trim_arrivals_locked(arrival)
            self._last_error = ""

    def _validate_data(self, data: Any) -> np.ndarray:
        array = np.asarray(data)
        if self.stream_kind is StreamKind.RGB:
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise CameraSourceError(
                    f"expected uint8 RGB HxWx3 for {self.stream_name}, "
                    f"got dtype={array.dtype}, shape={array.shape}"
                )
        else:
            if array.dtype != np.float32 or array.ndim != 2:
                raise CameraSourceError(
                    f"expected float32 depth HxW for {self.stream_name}, "
                    f"got dtype={array.dtype}, shape={array.shape}"
                )
        return np.ascontiguousarray(array)

    def _source_fps_locked(self, now: float) -> float:
        self._trim_arrivals_locked(now)
        if len(self._arrival_times) < 2:
            return 0.0
        elapsed = self._arrival_times[-1] - self._arrival_times[0]
        return (len(self._arrival_times) - 1) / elapsed if elapsed > 0.0 else 0.0

    def _trim_arrivals_locked(self, now: float) -> None:
        cutoff = now - 2.0
        while self._arrival_times and self._arrival_times[0] < cutoff:
            self._arrival_times.popleft()

    def _make_subscriber(self) -> tuple[Subscriber, Callable[[], None]]:
        from dexcomm import Node as DexCommNode
        from dexcontrol.sensors.camera.base_camera import (
            StreamSubscriber,
            StreamType,
            TransportType,
        )

        node = None
        if self.transport == "zenoh":
            node = DexCommNode(
                name=(
                    f"{self.stream_name}_direct_camera_source_"
                    f"{os.getpid()}_{id(self):x}"
                )
            )
        subscriber = StreamSubscriber(
            stream_name=self.stream_name,
            transport=TransportType(self.transport),
            stream_type=(
                StreamType.RGB
                if self.stream_kind is StreamKind.RGB
                else StreamType.DEPTH
            ),
            node=node,
            topic=self.topic or None,
            rtc_channel=self.rtc_channel or None,
            codec=self.codec,
            buffer_size=1,
        )

        def close() -> None:
            subscriber.shutdown()
            if node is not None:
                node.shutdown()

        return subscriber, close
