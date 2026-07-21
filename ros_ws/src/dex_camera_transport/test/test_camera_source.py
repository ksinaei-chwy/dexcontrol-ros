from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from dex_camera_transport import CameraSourceError, DexCommCameraSource, StreamKind


class FakeSubscriber:
    def __init__(self) -> None:
        self.latest = None
        self.shutdown_called = False
        self.lock = threading.Lock()

    def get_latest(self):
        with self.lock:
            return self.latest

    def wait_for_message(self, timeout: float):
        del timeout
        return self.get_latest()

    def shutdown(self) -> None:
        self.shutdown_called = True

    def publish(self, data, stamp: int, receive: int) -> None:
        with self.lock:
            self.latest = {
                "data": data,
                "timestamp_ns": stamp,
                "receive_time_ns": receive,
            }


def make_source(kind=StreamKind.RGB):
    fake = FakeSubscriber()
    source = DexCommCameraSource(
        stream_name="test",
        stream_kind=kind,
        topic="sensors/test",
        poll_interval_seconds=0.0005,
        subscriber_factory=lambda: (fake, fake.shutdown),
    )
    return source, fake


def test_latest_frame_preserves_timestamps_and_replaces_old_frame():
    source, fake = make_source()
    try:
        fake.publish(np.zeros((2, 3, 3), dtype=np.uint8), 1_000, 1_100)
        first = source.wait_for_frame(0.2)
        assert first is not None
        fake.publish(np.ones((2, 3, 3), dtype=np.uint8), 2_000, 2_100)
        deadline = time.monotonic() + 0.2
        while source.latest().sequence < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        latest = source.latest()
        assert latest.sequence == 2
        assert latest.source_stamp_ns == 2_000
        assert latest.receive_stamp_ns == 2_100
        assert np.all(latest.data == 1)
    finally:
        source.shutdown()
    assert fake.shutdown_called


def test_rgb_and_depth_payload_validation():
    rgb_source, rgb = make_source()
    depth_source, depth = make_source(StreamKind.DEPTH)
    try:
        rgb.publish(np.zeros((2, 2), dtype=np.uint8), 1, 2)
        depth.publish(np.zeros((2, 2), dtype=np.uint16), 1, 2)
        time.sleep(0.02)
        assert rgb_source.latest() is None
        assert depth_source.latest() is None
        assert rgb_source.stats().invalid_frames > 0
        assert depth_source.stats().invalid_frames > 0
    finally:
        rgb_source.shutdown()
        depth_source.shutdown()


def test_snapshot_validates_capture_receive_and_transport_age():
    source, fake = make_source()
    try:
        fake.publish(np.zeros((2, 2, 3), dtype=np.uint8), 700_000_000, 850_000_000)
        assert source.wait_for_frame(0.2) is not None
        frame = source.snapshot(
            now_ns=900_000_000,
            maximum_receive_age_seconds=0.1,
            maximum_capture_age_seconds=0.3,
            maximum_transport_delay_seconds=0.2,
        )
        assert frame.transport_delay_seconds == pytest.approx(0.15)
        with pytest.raises(CameraSourceError, match="receive age"):
            source.snapshot(
                now_ns=1_000_000_000,
                maximum_receive_age_seconds=0.1,
                maximum_capture_age_seconds=0.4,
                maximum_transport_delay_seconds=0.2,
            )
        with pytest.raises(CameraSourceError, match="transport delay"):
            source.snapshot(
                now_ns=900_000_000,
                maximum_receive_age_seconds=0.1,
                maximum_capture_age_seconds=0.3,
                maximum_transport_delay_seconds=0.1,
            )
    finally:
        source.shutdown()


def test_depth_requires_zenoh():
    with pytest.raises(ValueError, match="depth streams only support Zenoh"):
        DexCommCameraSource(
            stream_name="depth",
            stream_kind=StreamKind.DEPTH,
            topic="",
            transport="rtc",
            rtc_channel="depth_rtc",
            start=False,
        )
