"""Direct DexComm RGB camera sources and cached wrist placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from dex_camera_transport import CameraSourceError, DexCommCameraSource, StreamKind


class CameraValidationError(RuntimeError):
    """Raised when a camera frame is missing, stale, or malformed."""


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    source_stamp_ns: int
    receive_stamp_ns: int
    capture_age_seconds: float
    receive_age_seconds: float
    transport_delay_seconds: float


def resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize RGB with OpenCV when available and deterministic nearest fallback."""
    _validate_rgb(image)
    if width <= 0 or height <= 0:
        raise CameraValidationError("target image resolution must be positive")
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image)
    try:
        import cv2

        return np.ascontiguousarray(
            cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        )
    except ImportError:
        y_index = np.linspace(0, image.shape[0] - 1, height).astype(np.intp)
        x_index = np.linspace(0, image.shape[1] - 1, width).astype(np.intp)
        return np.ascontiguousarray(image[y_index][:, x_index])


class DirectRgbCameraSource:
    """Processed RGB view over a direct latest-frame DexComm source."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        stream_name: str,
        topic: str,
        transport: str = "zenoh",
        rtc_channel: str = "",
        codec: str = "auto",
        source_factory: Callable[..., Any] = DexCommCameraSource,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self._source = source_factory(
            stream_name=stream_name,
            stream_kind=StreamKind.RGB,
            topic=topic,
            transport=transport,
            rtc_channel=rtc_channel,
            codec=codec,
        )

    def snapshot(
        self,
        now_ns: int,
        *,
        maximum_receive_age_seconds: float,
        maximum_capture_age_seconds: float,
        maximum_transport_delay_seconds: float,
    ) -> CameraFrame:
        try:
            frame = self._source.snapshot(
                now_ns=int(now_ns),
                maximum_receive_age_seconds=maximum_receive_age_seconds,
                maximum_capture_age_seconds=maximum_capture_age_seconds,
                maximum_transport_delay_seconds=maximum_transport_delay_seconds,
            )
            rgb = resize_rgb(frame.data, self.width, self.height)
        except CameraSourceError as exc:
            raise CameraValidationError(str(exc)) from exc
        except CameraValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport/resize boundary
            raise CameraValidationError(f"camera conversion failed: {exc}") from exc
        return CameraFrame(
            rgb=rgb,
            source_stamp_ns=frame.source_stamp_ns,
            receive_stamp_ns=frame.receive_stamp_ns,
            capture_age_seconds=frame.capture_age_seconds(now_ns),
            receive_age_seconds=frame.receive_age_seconds(now_ns),
            transport_delay_seconds=frame.transport_delay_seconds,
        )

    def stats(self) -> Any:
        return self._source.stats()

    def shutdown(self) -> None:
        self._source.shutdown()


class PlaceholderCameraSource:
    """Cached black RGB frame whose effective size follows the head frame."""

    def __init__(self) -> None:
        self._frame: np.ndarray | None = None
        self._shape: tuple[int, int, int] | None = None

    def frame_for(self, reference_rgb: np.ndarray) -> np.ndarray:
        _validate_rgb(reference_rgb)
        shape = tuple(int(value) for value in reference_rgb.shape)
        if self._frame is None or self._shape != shape:
            self._frame = np.zeros(shape, dtype=np.uint8)
            self._shape = shape
        return self._frame


def _validate_rgb(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise CameraValidationError("camera frame is not a numpy array")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise CameraValidationError(
            f"expected uint8 RGB HxWx3, got dtype={image.dtype}, shape={image.shape}"
        )
