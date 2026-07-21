"""Immutable GR00T N1.7 training-to-deployment contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .contracts import ACTION_DIMENSION, HEAD_CAMERA_FEATURE, STATE_DIMENSION


POLICY_TYPE: Final[str] = "groot"
MODEL_REPO_ID: Final[str] = "Kasra99/groot-n17-dexmate-blue-bird"
MODEL_REVISION: Final[str] = "7f0f318540355031f189693e5623c1c5e8a17e93"
CHECKPOINT_TAG: Final[str] = "step-034000"
MODEL_WEIGHT_SIZE: Final[int] = 9_335_183_176
MODEL_WEIGHT_SHA256: Final[str] = (
    "549616cb8e8aebab8d3fe35207f8389b18275f5e9a770fada51a9e62faeeca94"
)

BASE_MODEL_REPO_ID: Final[str] = "nvidia/GR00T-N1.7-3B"
BASE_MODEL_REVISION: Final[str] = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
COSMOS_PROCESSOR_REPO_ID: Final[str] = "nvidia/Cosmos-Reason2-2B"
COSMOS_PROCESSOR_REVISION: Final[str] = (
    "9ce19a195e423419c349abfc86fd07178b230561"
)

ACTION_CHUNK_SIZE: Final[int] = 40
MODEL_MAX_STATE_ACTION_DIMENSION: Final[int] = 132
EMBODIMENT_TAG: Final[str] = "new_embodiment"


@dataclass(frozen=True)
class GrootDeploymentContract:
    """Small value object used by validators and status reporting."""

    policy_type: str = POLICY_TYPE
    state_dimension: int = STATE_DIMENSION
    action_dimension: int = ACTION_DIMENSION
    action_chunk_size: int = ACTION_CHUNK_SIZE
    max_state_dimension: int = MODEL_MAX_STATE_ACTION_DIMENSION
    max_action_dimension: int = MODEL_MAX_STATE_ACTION_DIMENSION
    camera_feature: str = HEAD_CAMERA_FEATURE
    embodiment_tag: str = EMBODIMENT_TAG


CONTRACT: Final[GrootDeploymentContract] = GrootDeploymentContract()
