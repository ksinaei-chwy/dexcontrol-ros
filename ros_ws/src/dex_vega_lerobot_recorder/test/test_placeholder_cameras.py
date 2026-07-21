from types import SimpleNamespace

import numpy as np
import pytest
from dex_camera_transport import CameraFrame, CameraSourceError

from dex_vega_lerobot_recorder.camera_sources import (
    CameraValidationError,
    DirectRgbCameraSource,
    PlaceholderCameraSource,
)


class FakeTransport:
    def __init__(self, frame=None, error=None, **_kwargs):
        self.frame = frame
        self.error = error
        self.shutdown_called = False

    def snapshot(self, **_kwargs):
        if self.error is not None:
            raise self.error
        return self.frame

    def stats(self):
        return SimpleNamespace(source_fps=30.0, invalid_frames=0)

    def shutdown(self):
        self.shutdown_called = True


def test_placeholder_is_cached_black_uint8_rgb_matching_head():
    source = PlaceholderCameraSource()
    head = np.full((12, 16, 3), 127, dtype=np.uint8)
    first = source.frame_for(head)
    second = source.frame_for(head)
    assert first is second
    assert first.shape == head.shape
    assert first.dtype == np.uint8
    assert np.count_nonzero(first) == 0
    resized = source.frame_for(np.zeros((8, 9, 3), dtype=np.uint8))
    assert resized.shape == (8, 9, 3)
    assert resized is not first


def test_direct_source_resizes_rgb_and_preserves_transport_timing():
    backend = FakeTransport(
        frame=CameraFrame(
            data=np.full((4, 6, 3), 12, dtype=np.uint8),
            source_stamp_ns=700_000_000,
            receive_stamp_ns=850_000_000,
            sequence=3,
        )
    )
    source = DirectRgbCameraSource(
        width=3,
        height=2,
        stream_name="left_rgb",
        topic="sensors/head_camera/left_rgb",
        source_factory=lambda **_kwargs: backend,
    )
    frame = source.snapshot(
        900_000_000,
        maximum_receive_age_seconds=0.1,
        maximum_capture_age_seconds=0.3,
        maximum_transport_delay_seconds=0.2,
    )
    assert frame.rgb.shape == (2, 3, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.capture_age_seconds == pytest.approx(0.2)
    assert frame.receive_age_seconds == pytest.approx(0.05)
    assert frame.transport_delay_seconds == pytest.approx(0.15)
    source.shutdown()
    assert backend.shutdown_called


def test_direct_source_maps_transport_validation_errors():
    backend = FakeTransport(error=CameraSourceError("stale left_rgb receive age"))
    source = DirectRgbCameraSource(
        width=2,
        height=2,
        stream_name="left_rgb",
        topic="sensors/head_camera/left_rgb",
        source_factory=lambda **_kwargs: backend,
    )
    with pytest.raises(CameraValidationError, match="stale"):
        source.snapshot(
            1,
            maximum_receive_age_seconds=0.1,
            maximum_capture_age_seconds=0.3,
            maximum_transport_delay_seconds=0.2,
        )
