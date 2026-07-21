"""Immutable policy and bridge contracts for the blue-bird task."""

from __future__ import annotations

from typing import Final


TASK: Final[str] = "put the blue bird on the meeting desk"
ROBOT_TYPE: Final[str] = "dexmate_vega_1_pro"
DATASET_REPO_ID: Final[str] = "Kasra99/dexmate_blue_bird"
DATASET_REVISION: Final[str] = "72a97b1a916699c17177e311463729d757f3119c"
MODEL_REPO_ID: Final[str] = "Kasra99/pi05-dexmate-blue-bird"
BASE_MODEL_REPO_ID: Final[str] = "lerobot/pi05_base"
BASE_MODEL_REVISION: Final[str] = "7de663972b7817d2c4cf2d84c821153dfea772e9"
TOKENIZER_REPO_ID: Final[str] = "google/paligemma-3b-pt-224"

BODY_JOINT_NAMES: Final[tuple[str, ...]] = (
    "torso_j1",
    "torso_j2",
    "torso_j3",
    "head_j1",
    "head_j2",
    "head_j3",
    "L_arm_j1",
    "L_arm_j2",
    "L_arm_j3",
    "L_arm_j4",
    "L_arm_j5",
    "L_arm_j6",
    "L_arm_j7",
    "R_arm_j1",
    "R_arm_j2",
    "R_arm_j3",
    "R_arm_j4",
    "R_arm_j5",
    "R_arm_j6",
    "R_arm_j7",
)

HAND_RATIO_NAMES: Final[tuple[str, ...]] = (
    "left_hand.open_close_ratio",
    "left_hand.thumb_opposition_ratio",
    "right_hand.open_close_ratio",
    "right_hand.thumb_opposition_ratio",
)
BASE_NAMES: Final[tuple[str, ...]] = ("base_vx", "base_vy", "base_wz")
ACTION_NAMES: Final[tuple[str, ...]] = BODY_JOINT_NAMES + HAND_RATIO_NAMES + BASE_NAMES
STATE_NAMES: Final[tuple[str, ...]] = (
    tuple(f"{name}.position" for name in BODY_JOINT_NAMES)
    + HAND_RATIO_NAMES
    + BASE_NAMES
)

STATE_DIMENSION: Final[int] = 27
ACTION_DIMENSION: Final[int] = 27
ACTION_CHUNK_SIZE: Final[int] = 50
MODEL_MAX_STATE_ACTION_DIMENSION: Final[int] = 32
HEAD_CAMERA_FEATURE: Final[str] = "observation.images.head"
MODEL_HEAD_CAMERA_FEATURE: Final[str] = "observation.images.base_0_rgb"

COMPONENT_JOINT_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "torso": BODY_JOINT_NAMES[0:3],
    "head": BODY_JOINT_NAMES[3:6],
    "left_arm": BODY_JOINT_NAMES[6:13],
    "right_arm": BODY_JOINT_NAMES[13:20],
}

COMMAND_TOPICS: Final[dict[str, str]] = {
    "torso": "/torso/joint_commands",
    "head": "/head/joint_commands",
    "left_arm": "/left_arm/joint_commands",
    "right_arm": "/right_arm/joint_commands",
    "left_hand": "/left_hand/joint_commands",
    "right_hand": "/right_hand/joint_commands",
}
BASE_COMMAND_TOPIC: Final[str] = "/cmd_vel"


def validate_contract_lengths() -> None:
    """Fail immediately if an edited contract no longer has the trained shape."""
    if len(STATE_NAMES) != STATE_DIMENSION:
        raise RuntimeError(f"state contract has {len(STATE_NAMES)} values, expected 27")
    if len(ACTION_NAMES) != ACTION_DIMENSION:
        raise RuntimeError(f"action contract has {len(ACTION_NAMES)} values, expected 27")
    if len(BODY_JOINT_NAMES) != 20:
        raise RuntimeError("body joint contract must have 20 values")


validate_contract_lengths()
