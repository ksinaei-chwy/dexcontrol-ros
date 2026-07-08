"""Helpers for Pico teleop log-frame payloads."""

from __future__ import annotations

from typing import Any

import numpy as np


def make_log_frame_payload(
    timestamp_ns: int,
    sequence: int | None,
    torso_q: np.ndarray,
    head_q: np.ndarray,
    left_q: np.ndarray,
    right_q: np.ndarray,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp_ns": int(timestamp_ns),
        "sequence": sequence,
        "action": {
            "torso": np.asarray(torso_q, dtype=np.float64).reshape(-1).tolist(),
            "head": np.asarray(head_q, dtype=np.float64).reshape(-1).tolist(),
            "left_arm": np.asarray(left_q, dtype=np.float64).reshape(-1).tolist(),
            "right_arm": np.asarray(right_q, dtype=np.float64).reshape(-1).tolist(),
        },
    }
    if debug is not None:
        payload["debug"] = debug
    return payload
