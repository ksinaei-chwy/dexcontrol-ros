import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dex_pico_teleop.hand_retargeting import (  # noqa: E402
    F5D6_CLOSED_POSITIONS,
    F5D6_OPEN_POSITIONS,
    F5D6HandConfig,
    f5d6_joint_names,
    f5d6_visual_joint_positions,
    retarget_f5d6_hand,
)
from dex_pico_teleop.log_frame import make_log_frame_payload  # noqa: E402
from dex_pico_teleop.teleop_node import PicoTeleopNode  # noqa: E402
from dex_pico_teleop.xr_packet import PicoPacket  # noqa: E402


def _config(side: str) -> F5D6HandConfig:
    return F5D6HandConfig.from_values(
        side,
        f5d6_joint_names(side),
        F5D6_OPEN_POSITIONS,
        F5D6_CLOSED_POSITIONS,
    )


class TestHandRetargeting(unittest.TestCase):
    def test_fully_open_and_fully_closed_targets(self):
        config = _config("left")

        np.testing.assert_allclose(
            retarget_f5d6_hand(config, trigger=0.0, grip=0.0),
            F5D6_OPEN_POSITIONS,
        )
        np.testing.assert_allclose(
            retarget_f5d6_hand(config, trigger=1.0, grip=1.0),
            F5D6_CLOSED_POSITIONS,
        )

    def test_trigger_flexes_five_joints_and_grip_only_opposes_thumb(self):
        config = _config("right")
        open_positions = np.asarray(F5D6_OPEN_POSITIONS)
        closed_positions = np.asarray(F5D6_CLOSED_POSITIONS)

        trigger_only = retarget_f5d6_hand(config, trigger=0.25, grip=0.0)
        expected_trigger = open_positions.copy()
        expected_trigger[:5] += 0.25 * (closed_positions[:5] - open_positions[:5])
        np.testing.assert_allclose(trigger_only, expected_trigger)
        self.assertEqual(trigger_only[5], open_positions[5])

        grip_only = retarget_f5d6_hand(config, trigger=0.0, grip=0.75)
        expected_grip = open_positions.copy()
        expected_grip[5] += 0.75 * (closed_positions[5] - open_positions[5])
        np.testing.assert_allclose(grip_only, expected_grip)
        np.testing.assert_allclose(grip_only[:5], open_positions[:5])

    def test_left_and_right_controller_values_are_independent(self):
        left = retarget_f5d6_hand(_config("left"), trigger=0.2, grip=0.8)
        right = retarget_f5d6_hand(_config("right"), trigger=0.9, grip=0.1)

        self.assertFalse(np.allclose(left[:5], right[:5]))
        self.assertNotEqual(left[5], right[5])
        np.testing.assert_allclose(
            left[:5],
            np.asarray(F5D6_OPEN_POSITIONS)[:5]
            + 0.2
            * (
                np.asarray(F5D6_CLOSED_POSITIONS)[:5]
                - np.asarray(F5D6_OPEN_POSITIONS)[:5]
            ),
        )

    def test_invalid_joint_names_disable_configuration(self):
        invalid_names = list(f5d6_joint_names("left"))
        invalid_names[-1] = "L_th_j2"

        with self.assertRaisesRegex(ValueError, "joint names must be"):
            F5D6HandConfig.from_values(
                "left",
                invalid_names,
                F5D6_OPEN_POSITIONS,
                F5D6_CLOSED_POSITIONS,
            )

    def test_mismatched_or_non_finite_endpoints_disable_configuration(self):
        with self.assertRaisesRegex(ValueError, "must contain 6 values"):
            F5D6HandConfig.from_values(
                "right",
                f5d6_joint_names("right"),
                F5D6_OPEN_POSITIONS[:-1],
                F5D6_CLOSED_POSITIONS,
            )

        non_finite = list(F5D6_CLOSED_POSITIONS)
        non_finite[2] = math.nan
        with self.assertRaisesRegex(ValueError, "only finite values"):
            F5D6HandConfig.from_values(
                "right",
                f5d6_joint_names("right"),
                F5D6_OPEN_POSITIONS,
                non_finite,
            )

    def test_disabled_hand_configuration_prevents_publication(self):
        packet = PicoPacket.from_dict(
            {
                "frame": "robot_z_up",
                "controllers": {
                    "left": {"trigger": 1.0, "grip": 1.0},
                    "right": {"trigger": 0.5, "grip": 0.25},
                },
            }
        )
        published: list[tuple[str, tuple[str, ...], np.ndarray]] = []

        class FakeNode:
            _hand_configs = {"left": None, "right": _config("right")}

            @staticmethod
            def _publish_joint_command(component, names, positions):
                published.append((component, names, positions))

        targets = PicoTeleopNode._update_hands(FakeNode(), packet)

        self.assertNotIn("left", targets)
        self.assertIn("right", targets)
        self.assertEqual([item[0] for item in published], ["right_hand"])

    def test_packet_clamps_trigger_and_grip_to_unit_interval(self):
        packet = PicoPacket.from_dict(
            {
                "frame": "robot_z_up",
                "controllers": {
                    "left": {"trigger": -0.4, "grip": 1.7},
                    "right": {"trigger": 4.2, "grip": -3.0},
                },
            }
        )

        self.assertEqual(packet.controllers["left"].trigger, 0.0)
        self.assertEqual(packet.controllers["left"].grip, 1.0)
        self.assertEqual(packet.controllers["right"].trigger, 1.0)
        self.assertEqual(packet.controllers["right"].grip, 0.0)

    def test_log_frame_contains_optional_hand_targets(self):
        left = retarget_f5d6_hand(_config("left"), trigger=0.3, grip=0.4)
        right = retarget_f5d6_hand(_config("right"), trigger=0.7, grip=0.6)
        payload = make_log_frame_payload(
            123,
            9,
            np.zeros(3),
            np.zeros(3),
            np.zeros(7),
            np.zeros(7),
            left_hand_q=left,
            right_hand_q=right,
        )

        np.testing.assert_allclose(payload["action"]["left_hand"], left)
        np.testing.assert_allclose(payload["action"]["right_hand"], right)

    def test_meshcat_positions_include_drivers_and_correct_mimics(self):
        commands = np.array([0.1, -0.2, -0.3, -0.4, -0.5, 0.6])
        positions = f5d6_visual_joint_positions("left", commands)

        self.assertEqual(positions["L_th_j1"], commands[0])
        self.assertEqual(positions["L_th_j0"], commands[5])
        self.assertAlmostEqual(positions["L_th_j2"], commands[0] * 1.35316 + 0.00765)
        self.assertAlmostEqual(positions["L_ff_j2"], commands[1] * 1.13028 - 0.00053)
        self.assertAlmostEqual(positions["L_mf_j2"], commands[2] * 1.13311 - 0.00079)
        self.assertAlmostEqual(positions["L_rf_j2"], commands[3] * 1.12935 + 0.00065)
        self.assertAlmostEqual(positions["L_lf_j2"], commands[4] * 1.15037 + 0.00186)


if __name__ == "__main__":
    unittest.main()
