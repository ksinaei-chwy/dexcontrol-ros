import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dex_pico_teleop.head_camera_vision_node import (  # noqa: E402
    configure_head_camera_stream,
    make_side_by_side_rgb_frame,
    normalize_rgb_frame,
    resize_rgb_frame,
    xrobotoolkit_packet_header,
)


class TestHeadCameraVision(unittest.TestCase):
    def test_configure_head_camera_stream_enables_one_rgb_stream(self):
        from dexcontrol.core.config import get_robot_config

        configs = get_robot_config()
        stream_config = configure_head_camera_stream(
            configs,
            sensor_name="head_camera",
            stream_name="left_rgb",
            transport="rtc",
        )
        sensor_config = configs.sensors["head_camera"]

        self.assertTrue(sensor_config.enabled)
        self.assertEqual(stream_config.name, "left_rgb")
        self.assertTrue(stream_config.enabled)
        self.assertEqual(stream_config.transport, "rtc")
        self.assertEqual(
            stream_config.rtc_channel,
            "sensors/head_camera/left_rgb_rtc",
        )
        self.assertFalse(sensor_config.enable_depth)

    def test_normalize_rgb_frame_extracts_data_and_clips_dtype(self):
        raw = np.array([[[0.0, 127.2, 999.0, 42.0]]], dtype=np.float32)
        frame = normalize_rgb_frame({"data": raw})

        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(frame.shape, (1, 1, 3))
        np.testing.assert_array_equal(frame[0, 0], [0, 127, 255])
        self.assertTrue(frame.flags["C_CONTIGUOUS"])

    def test_normalize_rgb_frame_extracts_dexcomm_message_data(self):
        class Message:
            def __init__(self, data):
                self.data = data

        raw = np.array([[[1, 2, 3]]], dtype=np.uint8)
        frame = normalize_rgb_frame(Message({"data": raw}))

        np.testing.assert_array_equal(frame, raw)
        self.assertTrue(frame.flags["C_CONTIGUOUS"])

    def test_normalize_rgb_frame_rejects_non_rgb_shape(self):
        with self.assertRaises(ValueError):
            normalize_rgb_frame(np.zeros((2, 2), dtype=np.uint8))

    def test_resize_rgb_frame_changes_dimensions(self):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(6, dtype=np.uint8)

        resized = resize_rgb_frame(frame, width=3, height=2)

        self.assertEqual(resized.shape, (2, 3, 3))
        self.assertEqual(resized.dtype, np.uint8)
        self.assertTrue(resized.flags["C_CONTIGUOUS"])

    def test_make_side_by_side_rgb_frame_duplicates_mono_view(self):
        frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((2, 3, 3))

        stereo = make_side_by_side_rgb_frame(frame)

        self.assertEqual(stereo.shape, (2, 6, 3))
        np.testing.assert_array_equal(stereo[:, :3], frame)
        np.testing.assert_array_equal(stereo[:, 3:], frame)
        self.assertTrue(stereo.flags["C_CONTIGUOUS"])

    def test_xrobotoolkit_packet_header_is_big_endian_payload_size(self):
        self.assertEqual(
            xrobotoolkit_packet_header(0x01020304),
            b"\x01\x02\x03\x04",
        )


if __name__ == "__main__":
    unittest.main()
