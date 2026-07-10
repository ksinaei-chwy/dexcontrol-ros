#!/usr/bin/env python3
"""Compare reduced Pink collision profiles with the full sphere reference.

This is an engineering coverage check, not a collision-safety certification.
It reports how often a reduced profile raises a d_min event that is also
raised by the full, SRDF-filtered collision-sphere model over a reproducible
set of broad upper-body samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dex_pico_teleop.pink_backend import PinkVegaKinematics


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--inflation", type=float, default=1.0)
    parser.add_argument("--d-min", type=float, default=0.04)
    parser.add_argument("--profiles", type=int, nargs="+", default=[18, 30, 40, 50])
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--srdf", type=Path, required=True)
    parser.add_argument("--collision-urdf", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, action="append", default=[])
    return parser.parse_args()


def _kinematics(
    args: argparse.Namespace,
    pipeline: str,
    sphere_count: int = 30,
) -> PinkVegaKinematics:
    return PinkVegaKinematics(
        args.urdf,
        self_collision_components=("left_arm", "right_arm"),
        self_collision_srdf_path=args.srdf,
        self_collision_urdf_path=args.collision_urdf,
        collision_package_dirs=tuple(args.package_dir),
        self_collision_n_pairs=24,
        self_collision_d_min=args.d_min,
        collision_pipeline=pipeline,
        collision_sphere_count=sphere_count,
        collision_sphere_inflation=args.inflation,
    )


def _samples(count: int, seed: int):
    rng = np.random.default_rng(seed)
    torso_center = np.array([0.02, 0.0, 0.0])
    left_center = np.array([0.45, 0.25, 0.0, -1.0, 0.0, 0.25, 0.0])
    right_center = np.array([-0.45, -0.25, 0.0, -1.0, 0.0, -0.25, 0.0])
    torso_spread = np.array([0.04, 0.05, 0.03])
    arm_spread = np.array([0.45, 0.45, 0.50, 0.50, 0.65, 0.70, 0.70])
    for _ in range(count):
        yield (
            torso_center + rng.normal(0.0, torso_spread),
            left_center + rng.normal(0.0, arm_spread),
            right_center + rng.normal(0.0, arm_spread),
        )


def _event(kinematics: PinkVegaKinematics, q_values) -> bool:
    torso_q, left_q, right_q = q_values
    q = kinematics.arms.q_from_values(torso_q, left_q, right_q)
    kinematics.arms.configuration.update(q)
    kinematics.arms._update_collision_diagnostics()
    return kinematics.arms.last_collision_distance < float(kinematics.arms.barriers[0].d_min)


def main() -> None:
    args = _arguments()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if not args.package_dir:
        args.package_dir = [args.urdf.parent.parent]
    samples = list(_samples(args.samples, args.seed))
    reference = _kinematics(args, "closest_pairs")
    reference_events = [_event(reference, values) for values in samples]
    event_count = sum(reference_events)
    results: dict[str, object] = {
        "samples": args.samples,
        "seed": args.seed,
        "d_min": args.d_min,
        "reference_events": event_count,
        "profiles": {},
    }
    for profile in args.profiles:
        reduced = _kinematics(args, "reduced_all_pairs", profile)
        events = [_event(reduced, values) for values in samples]
        recalled = sum(reference_event and event for reference_event, event in zip(reference_events, events))
        false_positive = sum(
            event and not reference_event for reference_event, event in zip(reference_events, events)
        )
        results["profiles"][str(profile)] = {
            "geometries": reduced.arms.collision_geometry_count,
            "pairs": reduced.arms.collision_pair_count,
            "recalled_events": recalled,
            "recall": None if event_count == 0 else recalled / event_count,
            "false_positive_events": false_positive,
        }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
