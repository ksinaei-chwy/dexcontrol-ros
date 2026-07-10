import math
import sys
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dex_pico_teleop.calibration import CalibrationState, average_calibration_packets
from dex_pico_teleop.collision_profiles import collision_sphere_names
from dex_pico_teleop.kinematics import (
    ARM_J4_UPPER_LIMIT_RAD,
    IKSolution,
    VegaKinematics,
    clip_arm_j4_upper_limit,
)
from dex_pico_teleop.log_frame import make_log_frame_payload
from dex_pico_teleop.network_receiver import NetworkReceiver
from dex_pico_teleop.pink_backend import PinkUnavailableError, PinkVegaKinematics
from dex_pico_teleop.retargeting import (
    controller_hand_point,
    fit_two_pose_shoulder_reach,
    fixed_head_joint_positions,
    fixed_head_rotation,
    normalized_reach_target,
    operator_arm_length_for_side,
    operator_shoulder_position,
    robot_shoulder_position,
    vector_in_arm_center,
)
from dex_pico_teleop.safety import base_twist_from_joysticks, joystick_with_deadzone
from dex_pico_teleop.teleop_state import (
    PositionTargetPlant,
    assess_arm_ik_solution,
    joint_values,
)
from dex_pico_teleop.transforms import (
    OPENXR_TO_ROBOT,
    Pose,
    matrix_to_quat,
    pose_openxr_to_robot,
    rot_y,
    rot_z,
)
from dex_pico_teleop.xr_packet import PicoPacket


class TestCore(unittest.TestCase):
    def test_compact_collision_profile_prioritizes_elbows_palms_and_torso(self):
        profile = collision_sphere_names(18)
        self.assertEqual(len(profile), 18)
        self.assertTrue(
            {
                "L_arm_l3_0",
                "L_arm_l4_0",
                "R_arm_l3_0",
                "R_arm_l4_0",
                "L_hand_base_0",
                "L_hand_base_1",
                "R_hand_base_0",
                "R_hand_base_1",
                "torso_l3_0",
                "torso_l3_1",
                "torso_l3_2",
                "torso_l3_4",
                "torso_l3_6",
                "torso_l1_1",
            }.issubset(profile)
        )
        self.assertFalse(any(name.startswith(("head_", "base_")) for name in profile))

    def _minimal_packet(self, timestamp_ns: int) -> PicoPacket:
        return PicoPacket.from_dict(
            {
                "timestamp_ns": timestamp_ns,
                "frame": "robot_z_up",
                "head": {"pose": [0, 0, 1.6, 0, 0, 0, 1]},
                "controllers": {
                    "left": {"pose": [0.1, 0.2, 1.1, 0, 0, 0, 1]},
                    "right": {"pose": [0.1, -0.2, 1.1, 0, 0, 0, 1]},
                },
            }
        )

    def test_openxr_position_conversion_to_robot_frame(self):
        pose = Pose.from_list([1.0, 2.0, -3.0, 0.0, 0.0, 0.0, 1.0])
        converted = pose_openxr_to_robot(pose)
        np.testing.assert_allclose(converted.position, OPENXR_TO_ROBOT @ pose.position)

    def test_network_receiver_drains_all_available_packets(self):
        receiver = NetworkReceiver(max_queue=4)
        for timestamp_ns in (1, 2, 3):
            receiver._push_packet(self._minimal_packet(timestamp_ns))

        packets = receiver.get_available()
        self.assertEqual([packet.timestamp_ns for packet in packets], [1, 2, 3])
        self.assertEqual(receiver.get_available(), [])

    def test_network_receiver_get_latest_still_drains_queue(self):
        receiver = NetworkReceiver(max_queue=2)
        for timestamp_ns in (1, 2, 3):
            receiver._push_packet(self._minimal_packet(timestamp_ns))

        packet = receiver.get_latest()
        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.timestamp_ns, 3)
        self.assertEqual(receiver.get_available(), [])

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

    def test_controller_hand_point_uses_controller_local_offset(self):
        offset = np.array([0.12, -0.03, 0.02])
        identity_point = controller_hand_point(
            np.array([1.0, 2.0, 3.0]),
            np.eye(3),
            offset,
        )
        rotated_point = controller_hand_point(
            np.array([1.0, 2.0, 3.0]),
            rot_z(math.pi / 2.0),
            offset,
        )
        np.testing.assert_allclose(identity_point, [1.12, 1.97, 3.02])
        np.testing.assert_allclose(rotated_point, [1.03, 2.12, 3.02], atol=1.0e-12)
        with self.assertRaises(ValueError):
            controller_hand_point(np.zeros(3), np.eye(3), [0.0, math.nan, 0.0])
        with self.assertRaises(ValueError):
            controller_hand_point(np.zeros(3), np.eye(3), [0.0, 0.0])

    def test_operator_vector_is_rotated_into_arm_center_frame(self):
        arm_center_rotation = rot_y(math.radians(30.0))
        operator_vector = np.array([0.6, -0.1, 0.2])
        mapped = vector_in_arm_center(operator_vector, arm_center_rotation)
        np.testing.assert_allclose(
            arm_center_rotation @ mapped,
            operator_vector,
            atol=1.0e-12,
        )

    def test_two_pose_calibration_recovers_shoulder_and_arm_length(self):
        for side, shoulder_y in (("left", 0.21), ("right", -0.21)):
            shoulder = np.array([-0.10, shoulder_y, -0.22])
            arm_length = 0.65
            neutral = shoulder + np.array([0.0, 0.0, -arm_length])
            reach = shoulder + np.array([arm_length, 0.0, 0.0])
            fitted = fit_two_pose_shoulder_reach(
                side,
                neutral,
                reach,
                shoulder_width=0.42,
                head_to_shoulder_z=0.22,
                arm_length_min=0.45,
                arm_length_max=0.85,
            )
            np.testing.assert_allclose(fitted.shoulder_offset, shoulder, atol=1.0e-12)
            self.assertAlmostEqual(fitted.arm_length, arm_length)

        with self.assertRaisesRegex(ValueError, "move at least"):
            fit_two_pose_shoulder_reach(
                "left",
                np.array([-0.1, 0.21, -0.8]),
                np.array([-0.05, 0.21, -0.3]),
                0.42,
                0.22,
                0.45,
                0.85,
            )

    def test_two_pose_calibration_handles_head_translation_and_small_noise(self):
        rng = np.random.default_rng(7)
        head_a = np.array([0.4, -0.2, 1.65])
        head_b = np.array([0.48, -0.16, 1.67])
        for side, shoulder_y in (("left", 0.21), ("right", -0.21)):
            shoulder = np.array([-0.11, shoulder_y, -0.22])
            arm_length = 0.64
            hand_a_world = head_a + shoulder + [0.0, 0.0, -arm_length]
            hand_b_world = head_b + shoulder + [arm_length, 0.0, 0.0]
            neutral_samples = [
                hand_a_world + rng.normal(0.0, 0.002, 3)
                - (head_a + rng.normal(0.0, 0.001, 3))
                for _sample in range(21)
            ]
            reach_samples = [
                hand_b_world + rng.normal(0.0, 0.002, 3)
                - (head_b + rng.normal(0.0, 0.001, 3))
                for _sample in range(21)
            ]
            fitted = fit_two_pose_shoulder_reach(
                side,
                np.median(neutral_samples, axis=0),
                np.median(reach_samples, axis=0),
                shoulder_width=0.42,
                head_to_shoulder_z=0.22,
                arm_length_min=0.45,
                arm_length_max=0.85,
            )
            np.testing.assert_allclose(fitted.shoulder_offset, shoulder, atol=0.01)
            self.assertAlmostEqual(fitted.arm_length, arm_length, delta=0.01)

    def test_invalid_reach_calibration_does_not_replace_previous_values(self):
        calibration = CalibrationState()
        original_offsets = {
            "left": np.array([-0.1, 0.21, -0.22]),
            "right": np.array([-0.1, -0.21, -0.22]),
        }
        original_lengths = {"left": 0.64, "right": 0.65}
        calibration.set_operator_reach_calibration(
            original_offsets,
            original_lengths,
        )

        with self.assertRaises(ValueError):
            calibration.set_operator_reach_calibration(
                {"left": np.zeros(3)},
                {"left": 0.6},
            )

        for side in ("left", "right"):
            np.testing.assert_allclose(
                calibration.operator_shoulder_offsets[side],
                original_offsets[side],
            )
            self.assertEqual(
                calibration.operator_arm_lengths[side],
                original_lengths[side],
            )

    def test_calibration_packet_average_reports_motion_dispersion(self):
        packets = []
        for index in range(10):
            x = 0.001 * index
            packets.append(
                PicoPacket.from_dict(
                    {
                        "timestamp_ns": index + 1,
                        "frame": "robot_z_up",
                        "head": {"pose": [0.0, 0.0, 1.6, 0.0, 0.0, 0.0, 1.0]},
                        "controllers": {
                            "left": {"pose": [x, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0]},
                            "right": {"pose": [x, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0]},
                        },
                    }
                )
            )
        averaged, stats = average_calibration_packets(packets)
        self.assertEqual(stats.sample_count, 10)
        self.assertLess(stats.controller_position_dispersion_m["left"], 0.01)
        self.assertAlmostEqual(averaged.controllers["left"].pose.position[0], 0.0045)

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

    def test_calibration_maps_controller_rotation_through_arm_center_frame(self):
        controller_rotation = rot_z(0.3)
        arm_center_rotation = rot_y(math.radians(28.0))
        ee_rotation = rot_y(-0.4) @ rot_z(0.2)
        packet = PicoPacket.from_dict(
            {
                "timestamp_ns": 10,
                "frame": "robot_z_up",
                "head": {"pose": [0.0, 0.0, 1.6, 0.0, 0.0, 0.0, 1.0]},
                "controllers": {
                    "left": {
                        "pose": [
                            0.2,
                            0.2,
                            1.1,
                            *matrix_to_quat(controller_rotation).tolist(),
                        ]
                    },
                    "right": {"pose": [0.2, -0.2, 1.1, 0.0, 0.0, 0.0, 1.0]},
                },
            }
        )
        calibration = CalibrationState()
        calibration.calibrate(
            packet,
            np.array([0.0, 0.0, 1.0]),
            arm_center_rotation,
            np.eye(3),
            {"left": ee_rotation, "right": np.eye(3)},
        )
        neutral_controller = calibration.to_operator_pose(packet.controllers["left"].pose)
        np.testing.assert_allclose(
            calibration.arm_target_rotation(
                "left",
                neutral_controller,
                arm_center_rotation,
            ),
            ee_rotation,
            atol=1.0e-12,
        )

        delta = rot_z(0.15)
        moved_controller = Pose(
            neutral_controller.position,
            matrix_to_quat(delta @ neutral_controller.rotation),
        )
        target_arm_center = calibration.arm_target_rotation(
            "left",
            moved_controller,
            arm_center_rotation,
        )
        np.testing.assert_allclose(
            arm_center_rotation @ target_arm_center,
            delta @ arm_center_rotation @ ee_rotation,
            atol=1.0e-12,
        )

    def test_arm_ik_acceptance_sends_finite_integrated_steps_and_holds_failures(self):
        converged = IKSolution(
            np.zeros(7),
            True,
            0.001,
            2,
            termination="converged",
            initial_error_norm=0.1,
            initial_position_error_norm=0.08,
            initial_orientation_error_norm=0.2,
            position_error_norm=0.001,
            orientation_error_norm=0.005,
        )
        self.assertEqual(assess_arm_ik_solution(converged).mode, "converged")

        progress = IKSolution(
            np.ones(7) * 0.01,
            False,
            0.07,
            6,
            termination="max_iterations",
            initial_error_norm=0.1,
            initial_position_error_norm=0.08,
            initial_orientation_error_norm=0.2,
            position_error_norm=0.06,
            orientation_error_norm=0.205,
        )
        self.assertEqual(
            assess_arm_ik_solution(progress).mode,
            "integrated_step",
        )

        temporarily_worse = IKSolution(
            progress.q,
            False,
            0.15,
            6,
            termination="max_iterations",
            initial_error_norm=0.1,
            initial_position_error_norm=0.08,
            initial_orientation_error_norm=0.2,
            position_error_norm=0.09,
            orientation_error_norm=0.25,
        )
        self.assertEqual(
            assess_arm_ik_solution(temporarily_worse).mode,
            "integrated_step",
        )

        no_solution = IKSolution(
            progress.q,
            False,
            0.05,
            1,
            termination="no_solution",
            initial_error_norm=0.1,
            initial_position_error_norm=0.08,
            initial_orientation_error_norm=0.2,
            position_error_norm=0.04,
            orientation_error_norm=0.19,
        )
        self.assertEqual(
            assess_arm_ik_solution(no_solution).mode,
            "held_no_solution",
        )

        nonfinite = IKSolution(
            np.full(7, np.nan),
            False,
            float("inf"),
            1,
            termination="max_iterations",
            initial_error_norm=0.1,
            initial_position_error_norm=0.08,
            initial_orientation_error_norm=0.2,
            position_error_norm=0.04,
            orientation_error_norm=0.19,
        )
        self.assertEqual(
            assess_arm_ik_solution(nonfinite).mode,
            "held_nonfinite",
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

    def test_pink_backend_builds_bimanual_fixed_pair_barrier_when_available(self):
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

        self.assertEqual(kin.arms.collision_geometry_count, 18)
        self.assertEqual(kin.arms.collision_pair_count, 107)
        self.assertEqual(kin.arms.barrier_pair_count, kin.arms.collision_pair_count)
        self.assertAlmostEqual(float(kin.arms.barriers[0].gain[0]), 6.0)
        self.assertEqual(kin.left_arm.barrier_pair_count, kin.arms.barrier_pair_count)
        self.assertEqual(kin.right_arm.barrier_pair_count, kin.arms.barrier_pair_count)
        self.assertFalse(hasattr(kin.torso, "barrier_pair_count"))

        torso_seed = np.array([0.02, 0.0, 0.0])
        left_seed = np.array([0.45, 0.25, 0.0, -1.0, 0.0, 0.25, 0.0])
        right_seed = np.array([-0.45, -0.25, 0.0, -1.0, 0.0, -0.25, 0.0])
        left_position, left_rotation = kin.arms.forward_arm(
            "left", torso_seed, left_seed, right_seed
        )
        right_position, right_rotation = kin.arms.forward_arm(
            "right", torso_seed, left_seed, right_seed
        )
        solutions = kin.solve_bimanual_arm_poses(
            torso_seed,
            left_seed,
            right_seed,
            left_position + np.array([0.01, 0.0, -0.01]),
            left_rotation,
            right_position + np.array([0.01, 0.0, 0.005]),
            right_rotation,
            dt=0.02,
        )
        for solution in solutions.values():
            self.assertTrue(np.all(np.isfinite(solution.q)))
            self.assertEqual(solution.iterations, 1)
            self.assertTrue(assess_arm_ik_solution(solution).accepted)
            self.assertLess(
                solution.position_error_norm,
                solution.initial_position_error_norm,
            )
        for name, expected in zip(("torso_j1", "torso_j2", "torso_j3"), torso_seed):
            self.assertAlmostEqual(
                kin.arms.configuration.q[kin.arms._indices[name]], expected
            )

        elapsed_samples_ms = []
        for _sample in range(5):
            start = time.perf_counter()
            kin.solve_bimanual_arm_poses(
                torso_seed,
                left_seed,
                right_seed,
                left_position + [0.01, 0.0, 0.005],
                left_rotation,
                right_position + [0.01, 0.0, 0.005],
                right_rotation,
                dt=0.02,
            )
            elapsed_samples_ms.append((time.perf_counter() - start) * 1000.0)
        self.assertLess(float(np.mean(elapsed_samples_ms)), 20.0)

    def test_dry_run_position_plant_returns_delayed_feedback(self):
        plant = PositionTargetPlant(max_velocity=1.0, max_acceleration=4.0)
        plant.seed(("joint",), np.array([0.0]))
        plant.set_target(("joint",), np.array([1.0]))
        plant.advance(0.02)
        self.assertGreater(plant.values(("joint",))[0], 0.0)
        self.assertLess(plant.values(("joint",))[0], 1.0)
        for _ in range(100):
            plant.advance(0.02)
        self.assertAlmostEqual(plant.values(("joint",))[0], 1.0)

    def test_pink_closest_pair_pipeline_remains_available_when_possible(self):
        repo_root = Path(__file__).resolve().parents[4]
        try:
            kin = PinkVegaKinematics(
                repo_root
                / "ros_ws/src/dexmate_vega_description/urdf/vega_1p_f5d6.package.urdf",
                self_collision_components=("left_arm", "right_arm"),
                self_collision_srdf_path=(
                    repo_root / "ros_ws/src/dexmate_vega_moveit_config/config/vega_1p_f5d6.srdf"
                ),
                self_collision_urdf_path=(
                    repo_root
                    / "ros_ws/src/dexmate_vega_description/robots/humanoid/vega_1p"
                    / "vega_1p_f5d6_collision_spheres.collision.urdf"
                ),
                collision_package_dirs=(repo_root / "ros_ws/src",),
                self_collision_n_pairs=8,
                collision_pipeline="closest_pairs",
            )
        except (PinkUnavailableError, ImportError, RuntimeError) as exc:
            self.skipTest(f"Pink backend unavailable: {exc}")

        self.assertEqual(kin.arms.collision_pipeline, "closest_pairs")
        self.assertEqual(kin.arms.collision_geometry_count, 182)
        self.assertGreater(kin.arms.collision_pair_count, 1000)
        self.assertEqual(kin.arms.barrier_pair_count, 8)

    def test_arm_j4_robot_limits_are_zero_for_both_ik_backends(self):
        numeric_kinematics = VegaKinematics()
        for chain in (numeric_kinematics.left_arm, numeric_kinematics.right_arm):
            seed = np.zeros(7)
            seed[3] = 0.244
            self.assertEqual(chain.clamp(seed)[3], ARM_J4_UPPER_LIMIT_RAD)
            self.assertEqual(clip_arm_j4_upper_limit(seed)[3], ARM_J4_UPPER_LIMIT_RAD)

        repo_root = Path(__file__).resolve().parents[4]
        for relative_path in (
            "ros_ws/src/dexmate_vega_description/urdf/vega_1p_f5d6.package.urdf",
            "ros_ws/src/dexmate_vega_description/robots/humanoid/vega_1p/vega_1p_f5d6.urdf",
            "ros_ws/src/dexmate_vega_description/robots/humanoid/vega_1p/"
            "vega_1p_f5d6_collision_spheres.collision.urdf",
        ):
            root = ET.parse(repo_root / relative_path).getroot()
            limits = {
                joint.attrib["name"]: float(joint.find("limit").attrib["upper"])
                for joint in root.findall("joint")
                if joint.attrib["name"] in {"L_arm_j4", "R_arm_j4"}
            }
            self.assertEqual(
                limits,
                {
                    "L_arm_j4": ARM_J4_UPPER_LIMIT_RAD,
                    "R_arm_j4": ARM_J4_UPPER_LIMIT_RAD,
                },
            )

    def test_pink_backend_reads_zero_arm_j4_limit_when_available(self):
        repo_root = Path(__file__).resolve().parents[4]
        urdf_path = repo_root / "ros_ws/src/dexmate_vega_description/urdf/vega_1p_f5d6.package.urdf"
        try:
            kin = PinkVegaKinematics(urdf_path)
        except (PinkUnavailableError, ImportError, RuntimeError) as exc:
            self.skipTest(f"Pink backend unavailable: {exc}")

        left_index = kin.left_arm._indices["L_arm_j4"]
        right_index = kin.right_arm._indices["R_arm_j4"]
        self.assertAlmostEqual(
            kin.left_arm._upper_position_limits[left_index],
            ARM_J4_UPPER_LIMIT_RAD,
        )
        self.assertAlmostEqual(
            kin.right_arm._upper_position_limits[right_index],
            ARM_J4_UPPER_LIMIT_RAD,
        )

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
        self.assertLessEqual(solution.q[3], ARM_J4_UPPER_LIMIT_RAD + 1.0e-12)
