import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dex_pico_teleop.calibration import CalibrationState
from dex_pico_teleop.kinematics import VegaKinematics
from dex_pico_teleop.log_frame import make_log_frame_payload
from dex_pico_teleop.pink_backend import PinkUnavailableError, PinkVegaKinematics
from dex_pico_teleop.retargeting import (
    fixed_head_joint_positions,
    fixed_head_rotation,
    normalized_reach_target,
    operator_arm_length_for_side,
    operator_shoulder_position,
    robot_shoulder_position,
)
from dex_pico_teleop.safety import base_twist_from_joysticks, joystick_with_deadzone
from dex_pico_teleop.teleop_state import joint_values
from dex_pico_teleop.transforms import OPENXR_TO_ROBOT, Pose, pose_openxr_to_robot, rot_y, rot_z
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
                        "secondaryButton": True,
                    },
                    "right": {
                        "pose": "0.2,1.2,-0.3,0,0,0,1",
                        "primaryButton": True,
                        "secondaryButton": True,
                    },
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
        self.assertTrue(packet.controllers["left"].button("y"))
        self.assertTrue(packet.controllers["right"].button("a"))
        self.assertTrue(packet.controllers["right"].button("b"))
        self.assertTrue(math.isclose(packet.controllers["left"].trigger, 0.4))
        np.testing.assert_allclose(packet.controllers["left"].joystick, [0.1, -0.2])
        self.assertIn("left_ankle", packet.trackers)
        self.assertIn("right_ankle", packet.trackers)

    def test_joystick_deadzone_rescales_after_threshold(self):
        result = joystick_with_deadzone(np.array([0.12, -0.56]), 0.2)
        self.assertEqual(result[0], 0.0)
        self.assertLess(result[1], -0.4)

    def test_base_twist_uses_left_and_right_controller_joysticks(self):
        twist = base_twist_from_joysticks(
            left_joystick=np.array([0.5, -1.0]),
            right_joystick=np.array([0.25, 0.75]),
            deadzone=0.0,
            vx_scale=0.35,
            vy_scale=0.25,
            wz_scale=0.55,
        )
        np.testing.assert_allclose(twist, [-0.35, -0.125, -0.1375])

    def test_base_twist_applies_deadzone_before_scaling(self):
        twist = base_twist_from_joysticks(
            left_joystick=np.array([0.1, -0.1]),
            right_joystick=np.array([0.1, 1.0]),
            deadzone=0.2,
            vx_scale=0.35,
            vy_scale=0.25,
            wz_scale=0.55,
        )
        np.testing.assert_allclose(twist, [0.0, 0.0, 0.0])

    def test_normalized_reach_maps_operator_sizes_to_same_robot_extent(self):
        robot_shoulder = robot_shoulder_position("right", 0.16946)
        short_shoulder = np.array([0.0, -0.2, 1.2])
        tall_shoulder = np.array([0.0, -0.25, 1.45])
        direction = np.array([1.0, 0.0, 0.0])

        short_target = normalized_reach_target(
            short_shoulder + direction * 0.50,
            short_shoulder,
            0.50,
            robot_shoulder,
            0.80,
        )
        tall_target = normalized_reach_target(
            tall_shoulder + direction * 0.75,
            tall_shoulder,
            0.75,
            robot_shoulder,
            0.80,
        )

        self.assertAlmostEqual(short_target.fraction, 1.0)
        self.assertAlmostEqual(tall_target.fraction, 1.0)
        np.testing.assert_allclose(short_target.position, tall_target.position)
        np.testing.assert_allclose(short_target.position, robot_shoulder + [0.80, 0.0, 0.0])

    def test_operator_shoulder_position_tracks_side_and_head_height(self):
        head = np.array([0.1, -0.05, 1.65])
        left = operator_shoulder_position("left", head, 0.42, 0.22, 0.03)
        right = operator_shoulder_position("right", head, 0.42, 0.22, 0.03)

        np.testing.assert_allclose(left, [0.13, 0.16, 1.43])
        np.testing.assert_allclose(right, [0.13, -0.26, 1.43])

    def test_operator_arm_length_uses_calibrated_value_and_clips(self):
        left = operator_arm_length_for_side(
            "left",
            {"left": 1.2},
            operator_height=1.7,
            ratio=0.44,
            minimum=0.45,
            maximum=0.85,
        )
        right = operator_arm_length_for_side(
            "right",
            {"left": 1.2},
            operator_height=1.7,
            ratio=0.44,
            minimum=0.45,
            maximum=0.85,
        )

        self.assertEqual(left.source, "calibrated")
        self.assertAlmostEqual(left.value, 0.85)
        self.assertEqual(right.source, "height_estimate")
        self.assertAlmostEqual(right.value, 1.7 * 0.44)

    def test_joint_values_prefer_command_warm_start_with_feedback_fallback(self):
        names = ("joint_1", "joint_2", "joint_3")
        feedback = {"joint_1": 0.1, "joint_2": 0.2}
        commands = {"joint_1": 1.1, "joint_3": 1.3}

        np.testing.assert_allclose(
            joint_values(names, feedback, commands, prefer_command=True),
            [1.1, 0.2, 1.3],
        )
        np.testing.assert_allclose(
            joint_values(names, feedback, commands, prefer_command=False),
            [0.1, 0.2, 1.3],
        )

    def test_log_frame_payload_keeps_action_and_adds_optional_debug(self):
        payload = make_log_frame_payload(
            timestamp_ns=123,
            sequence=7,
            torso_q=np.array([0.1, 0.2, 0.3]),
            head_q=np.array([0.4, 0.5, 0.6]),
            left_q=np.zeros(7),
            right_q=np.ones(7),
            debug={"retarget": {}, "timing_ms": {"loop": 1.5}},
        )

        self.assertEqual(payload["timestamp_ns"], 123)
        self.assertEqual(payload["sequence"], 7)
        self.assertEqual(set(payload["action"]), {"torso", "head", "left_arm", "right_arm"})
        self.assertIn("debug", payload)
        self.assertIn("timing_ms", payload["debug"])

    def test_calibration_orientation_offsets_make_current_pose_neutral_target(self):
        packet = PicoPacket.from_dict(
            {
                "timestamp_ns": 10,
                "frame": "robot_z_up",
                "head": {"pose": [0.0, 0.0, 1.6, 0.0, 0.0, 0.0, 1.0]},
                "controllers": {
                    "left": {"pose": [0.25, 0.25, 1.25, 0.0, 0.0, 0.0, 1.0]},
                    "right": {"pose": [0.25, -0.25, 1.25, 0.0, 0.0, 0.0, 1.0]},
                },
            }
        )
        calibration = CalibrationState()
        arm_rotations = {"left": rot_z(0.2), "right": rot_y(0.3)}
        head_rotation = rot_y(0.1)

        calibration.calibrate(
            packet,
            np.array([0.0, 0.0, 0.9]),
            np.eye(3),
            head_rotation,
            {},
            arm_rotations,
        )

        left_controller = calibration.to_operator_pose(packet.controllers["left"].pose)
        right_controller = calibration.to_operator_pose(packet.controllers["right"].pose)
        head_pose = calibration.to_operator_pose(packet.head)
        np.testing.assert_allclose(
            calibration.arm_target_rotation("left", left_controller),
            arm_rotations["left"],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            calibration.arm_target_rotation("right", right_controller),
            arm_rotations["right"],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            calibration.head_target_rotation(head_pose),
            head_rotation,
            atol=1.0e-12,
        )

    def test_fixed_head_rotation_looks_forward_and_down(self):
        rotation = fixed_head_rotation(20.0)
        forward = rotation @ np.array([1.0, 0.0, 0.0])
        lateral = rotation @ np.array([0.0, 1.0, 0.0])

        self.assertLess(forward[2], 0.0)
        self.assertAlmostEqual(math.atan2(forward[1], forward[0]), 0.0)
        np.testing.assert_allclose(lateral, [0.0, 1.0, 0.0], atol=1.0e-12)

    def test_fixed_head_joint_positions_pitch_head_down(self):
        kin = VegaKinematics()
        q = fixed_head_joint_positions(20.0)
        _pos, rotation = kin.head.forward(q)
        forward = rotation @ np.array([1.0, 0.0, 0.0])

        self.assertLess(forward[2], 0.0)
        self.assertAlmostEqual(q[0], math.radians(20.0))
        np.testing.assert_allclose(q[1:], [0.0, 0.0], atol=1.0e-12)

    def test_torso_solver_tracks_height_with_fixed_pitch(self):
        kin = VegaKinematics()
        seed = np.array([0.0, 0.0, 0.0])
        solution = kin.solve_torso_height(seed, 1.05, target_pitch=0.0)
        pos, rot = kin.arm_center_pose(solution.q)
        pitch = math.atan2(rot[0, 2], rot[0, 0])
        self.assertTrue(solution.success)
        self.assertLess(abs(pos[2] - 1.05), 0.02)
        self.assertLess(abs(pos[0] - kin.arm_center_pose(seed)[0][0]), 0.002)
        self.assertLess(abs(pitch), 0.03)
        self.assertEqual(solution.iterations, 0)

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

    def test_pink_backend_builds_arm_self_collision_barriers_when_available(self):
        repo_root = Path(__file__).resolve().parents[4]
        urdf_path = repo_root / "ros_ws/src/dexmate_vega_description/urdf/vega_1p_f5d6.package.urdf"
        srdf_path = repo_root / "ros_ws/src/dexmate_vega_moveit_config/config/vega_1p_f5d6.srdf"
        collision_urdf_path = (
            repo_root
            / "ros_ws/src/dexmate_vega_description/robots/humanoid/vega_1p"
            / "vega_1p_f5d6_collision_spheres.collision.urdf"
        )
        try:
            kin = PinkVegaKinematics(
                urdf_path,
                self_collision_components=("left_arm", "right_arm"),
                self_collision_srdf_path=srdf_path,
                self_collision_urdf_path=collision_urdf_path,
                collision_package_dirs=(repo_root / "ros_ws/src",),
                self_collision_n_pairs=8,
            )
        except (PinkUnavailableError, ImportError, RuntimeError) as exc:
            self.skipTest(f"Pink backend unavailable: {exc}")

        self.assertGreater(kin.left_arm.collision_pair_count, 1000)
        self.assertGreater(kin.right_arm.collision_pair_count, 1000)
        self.assertEqual(kin.left_arm.barrier_pair_count, 8)
        self.assertEqual(kin.right_arm.barrier_pair_count, 8)
        self.assertFalse(hasattr(kin.torso, "barrier_pair_count"))

        seed = np.zeros(7)
        start_pos, start_rot = kin.left_arm.forward(seed)
        solution = kin.solve_arm_pose(
            "left",
            seed,
            start_pos + np.array([0.01, 0.0, -0.01]),
            start_rot,
        )
        self.assertTrue(solution.success)
        self.assertLess(solution.error_norm, 0.05)

    def test_pink_backend_clips_slightly_out_of_range_arm_seed_when_available(self):
        repo_root = Path(__file__).resolve().parents[4]
        urdf_path = repo_root / "ros_ws/src/dexmate_vega_description/urdf/vega_1p_f5d6.package.urdf"
        try:
            kin = PinkVegaKinematics(urdf_path)
        except (PinkUnavailableError, ImportError, RuntimeError) as exc:
            self.skipTest(f"Pink backend unavailable: {exc}")

        seed = np.zeros(7)
        seed[3] = 0.24423788487911224
        target_pos, target_rot = kin.right_arm.forward(seed)
        solution = kin.solve_arm_pose("right", seed, target_pos, target_rot)

        self.assertTrue(solution.success)
        self.assertLessEqual(solution.q[3], 0.244)
