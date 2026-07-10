"""Reduced collision-sphere profiles for bimanual Vega IK.

The source collision URDF remains the reference model. These profiles retain
only geometry relevant to the teleoperation collision cases we actively want
to prevent. The compact profile protects elbows/forearms/palms against the
torso and opposite arm; larger profiles trade additional coverage for QP cost.
"""

from __future__ import annotations

from collections.abc import Iterable


_ELBOW_PALM_18 = (
    # Elbow, forearm, wrist, and palm geometry: the high-frequency contact
    # cases for bimanual Pico teleoperation.
    "L_arm_l3_0",
    "L_arm_l4_0",
    "L_arm_l6_1",
    "L_arm_l7_1",
    "L_hand_base_0",
    "L_hand_base_1",
    "R_arm_l3_0",
    "R_arm_l4_0",
    "R_arm_l6_1",
    "R_arm_l7_1",
    "R_hand_base_0",
    "R_hand_base_1",
    # Torso regions approached by both elbows and palms. Head/base are
    # deliberately excluded from this real-time profile.
    "torso_l3_0",
    "torso_l3_1",
    "torso_l3_2",
    "torso_l3_4",
    "torso_l3_6",
    "torso_l1_1",
)


_CORE_30 = (
    # Left arm: representative endpoints along every moving arm section.
    "L_arm_l1_1",
    "L_arm_l2_0",
    "L_arm_l2_2",
    "L_arm_l3_0",
    "L_arm_l3_1",
    "L_arm_l4_0",
    "L_arm_l4_1",
    "L_arm_l5_1",
    "L_arm_l6_1",
    "L_arm_l7_1",
    # Right arm, mirrored.
    "R_arm_l1_1",
    "R_arm_l2_0",
    "R_arm_l2_2",
    "R_arm_l3_0",
    "R_arm_l3_1",
    "R_arm_l4_0",
    "R_arm_l4_1",
    "R_arm_l5_1",
    "R_arm_l6_1",
    "R_arm_l7_1",
    # Palms and the upper-body obstacles most often approached by the arms.
    "L_hand_base_0",
    "R_hand_base_0",
    "torso_l1_1",
    "torso_l1_5",
    "torso_l2_1",
    "torso_l3_1",
    "torso_l3_4",
    "torso_l3_6",
    "head_l3_0",
    "base_4",
)

_DETAIL_40 = (
    "L_arm_l5_0",
    "R_arm_l5_0",
    "L_arm_l6_3",
    "R_arm_l6_3",
    "L_arm_l8_0",
    "R_arm_l8_0",
    "L_hand_base_1",
    "R_hand_base_1",
    "torso_l3_0",
    "base_0",
)

_DETAIL_50 = (
    "L_arm_l1_0",
    "R_arm_l1_0",
    "L_arm_l3_2",
    "R_arm_l3_2",
    "L_arm_l4_2",
    "R_arm_l4_2",
    "L_arm_l5_2",
    "R_arm_l5_2",
    "torso_l1_0",
    "head_l3_1",
)

ORDERED_COLLISION_SPHERES = _CORE_30 + _DETAIL_40 + _DETAIL_50
SUPPORTED_PROFILE_SIZES = (18, 30, 40, 50)


def collision_sphere_names(size: int) -> tuple[str, ...]:
    """Return the ordered geometry names for a supported profile size."""
    requested = int(size)
    if requested not in SUPPORTED_PROFILE_SIZES:
        raise ValueError(
            f"collision sphere profile must be one of {SUPPORTED_PROFILE_SIZES}, "
            f"got {requested}"
        )
    if requested == 18:
        return _ELBOW_PALM_18
    return ORDERED_COLLISION_SPHERES[:requested]


def filter_geometry_model(
    geometry_model,
    size: int,
    radius_inflation: float = 1.0,
) -> tuple[str, ...]:
    """Keep one reduced profile in a Pinocchio geometry model in place."""
    selected = collision_sphere_names(size)
    selected_set = set(selected)
    available = {geometry.name for geometry in geometry_model.geometryObjects}
    missing = selected_set - available
    if missing:
        raise ValueError(
            "collision URDF is missing reduced-profile geometries: "
            + ", ".join(sorted(missing))
        )
    for name in tuple(available - selected_set):
        geometry_model.removeGeometryObject(name)

    inflation = float(radius_inflation)
    if inflation < 1.0:
        raise ValueError("collision sphere inflation must be at least 1.0")
    for geometry in geometry_model.geometryObjects:
        if hasattr(geometry.geometry, "radius"):
            geometry.geometry.radius = float(geometry.geometry.radius) * inflation
    return selected


def profile_link_names(geometry_names: Iterable[str]) -> tuple[str, ...]:
    """Return the unique link names represented by a sphere-name sequence."""
    return tuple(dict.fromkeys(name.rsplit("_", maxsplit=1)[0] for name in geometry_names))
