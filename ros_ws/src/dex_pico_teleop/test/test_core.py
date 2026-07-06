import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dex_pico_teleop.kinematics import VegaKinematics
from dex_pico_teleop.pink_backend import PinkUnavailableError, PinkVegaKinematics
from dex_pico_teleop.safety import joystick_with_deadzone
from dex_pico_teleop.transforms import OPENXR_TO_ROBOT, Pose, pose_openxr_to_robot, rot_z
from dex_pico_teleop.xr_packet import PicoPacket


class TestCore(unittest.TestCase):
    def test_openxr_position_conversion_to_robot_frame(self):
        pose = Pose.from_list([1.0, 2.0, -3.0, 0.0, 0.0, 0.0, 1.0])
        converted = pose_openxr_to_robot(pose)
        np.testing.assert_allclose(converted.position, OPENXR_TO_ROBOT @ pose.position)

    def test_packet_parses_minimal_schema_and_buttons(self):
        packet = PicoPacket.from_dict(
            {
                "timestamp_ns": 10,
                "frame": "robot_z_up",
                "head": {"pose": [0, 0, 1.6, 0, 0, 0, 1]},
                "controllers": {
                    "left": {
                        "pose": [0.1, 0.2, 1.1, 0, 0, 0, 1],
                        "trigger": 0.5,
                        "grip": 0.9,
                        "joystick": [0.2, -0.3],
                        "buttons": {"stick": True},
                    },
                    "right": {"pose": [0, 0, 1.1, 0, 0, 0, 1]},
                },
                "trackers": {
                    "left_ankle": {"pose": [0, 0.1, 0.2, 0, 0, 0, 1]},
                    "right_ankle": {"pose": [0, -0.1, 0.2, 0, 0, 0, 1]},
                },
            }
        )
        self.assertEqual(packet.timestamp_ns, 10)
        self.assertTrue(packet.controllers["left"].button("stick"))
        self.assertTrue(math.isclose(packet.controllers["left"].trigger, 0.5))
        np.testing.assert_allclose(packet.controllers["left"].joystick, [0.2, -0.3])

    def test_packet_parses_xrobotoolkit_tracking_payload(self):
        packet = PicoPacket.from_xrobotoolkit_tracking(
            {
                "timeStampNs": 1234,
                "Head": {"pose": "0,1.6,0,0,0,0,1"},
                "Controller": {
                    "left": {
                        "pose": "-0.2,1.2,-0.3,0,0,0,1",
                        "axisX": 0.1,
                        "axisY": -0.2,
                        "axisClick": True,
                        "grip": 0.8,
                        "trigger": 0.4,
                        "primaryButton": True,
                    },
                    "right": {"pose": "0.2,1.2,-0.3,0,0,0,1"},
                },
                "Motion": {
                    "joints": [
                        {"p": "-0.1,0.1,0,0,0,0,1"},
                        {"p": "0.1,0.1,0,0,0,0,1"},
                    ]
                },
            }
        )
        self.assertEqual(packet.timestamp_ns, 1234)
        self.assertEqual(packet.frame, "robot_z_up")
        self.assertTrue(packet.controllers["left"].button("stick"))
        self.assertTrue(packet.controllers["left"].button("x"))
        self.assertTrue(math.isclose(packet.controllers["left"].trigger, 0.4))
        np.testing.assert_allclose(packet.controllers["left"].joystick, [0.1, -0.2])
        self.assertIn("left_ankle", packet.trackers)
        self.assertIn("right_ankle", packet.trackers)

    def test_joystick_deadzone_rescales_after_threshold(self):
        result = joystick_with_deadzone(np.array([0.12, -0.56]), 0.2)
        self.assertEqual(result[0], 0.0)
        self.assertLess(result[1], -0.4)

    def test_torso_solver_tracks_height_with_fixed_pitch(self):
        kin = VegaKinematics()
        seed = np.array([0.0, 0.0, 0.0])
        solution = kin.solve_torso_height(seed, 1.05, target_pitch=0.0)
        pos, rot = kin.arm_center_pose(solution.q)
        pitch = math.atan2(rot[0, 2], rot[0, 0])
        self.assertTrue(solution.success)
        self.assertLess(abs(pos[2] - 1.05), 0.02)
        self.assertLess(abs(pitch), 0.03)

    def test_head_solver_tracks_simple_yaw(self):
        kin = VegaKinematics()
        seed = np.zeros(3)
        solution = kin.solve_head_orientation(seed, rot_z(0.25))
        _pos, rot = kin.head.forward(solution.q)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        self.assertTrue(solution.success)
        self.assertLess(abs(yaw - 0.25), 0.04)

    def test_arm_solver_can_follow_small_relative_motion(self):
        kin = VegaKinematics()
        seed = np.zeros(7)
        start_pos, start_rot = kin.left_arm.forward(seed)
        target_pos = start_pos + np.array([0.02, 0.0, 0.01])
        solution = kin.solve_arm_pose("left", seed, target_pos, start_rot)
        end_pos, _end_rot = kin.left_arm.forward(solution.q)
        self.assertLess(solution.error_norm, 0.05)
        self.assertLess(np.linalg.norm(end_pos - target_pos), 0.04)

    def test_pink_backend_tracks_torso_height_when_available(self):
        repo_root = Path(__file__).resolve().parents[4]
        urdf_path = repo_root / "ros_ws/src/dexmate_vega_description/urdf/vega_1p_f5d6.package.urdf"
        try:
            kin = PinkVegaKinematics(urdf_path)
        except (PinkUnavailableError, ImportError, RuntimeError) as exc:
            self.skipTest(f"Pink backend unavailable: {exc}")
        solution = kin.solve_torso_height(np.zeros(3), 1.05)
        pos, _rot = kin.arm_center_pose(solution.q)
        self.assertTrue(solution.success)
        self.assertLess(abs(pos[2] - 1.05), 0.02)
