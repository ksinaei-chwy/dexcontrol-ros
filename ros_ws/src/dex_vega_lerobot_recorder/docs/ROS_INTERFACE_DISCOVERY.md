# Vega 1 Pro teleoperation recording interface discovery

This document records the repository inspection performed before the dataset
recorder was implemented.  It is the source of truth for the recorder's ROS
interfaces and vector layout.

## Existing packages

- Hardware wrapper: `dexcontrol_ros`, implemented by
  `dexcontrol_ros/dexcontrol_bridge.py`.
- Pico teleoperation: `dex_pico_teleop`, implemented by
  `dex_pico_teleop/teleop_node.py`.
- Head-camera transport: `dex_pico_teleop/head_camera_vision_node.py`.

The recorder is an observer.  It does not publish a robot command or alter the
teleoperation control loop.

## Command path found in the repository

Pico teleoperation publishes absolute joint-position targets as
`sensor_msgs/msg/JointState` on:

- `/torso/joint_commands`
- `/head/joint_commands`
- `/left_arm/joint_commands`
- `/right_arm/joint_commands`
- `/left_hand/joint_commands`
- `/right_hand/joint_commands`

It publishes a base velocity as `geometry_msgs/msg/Twist` on `/cmd_vel`.
Joint targets have already passed Pico retargeting, IK, collision constraints,
the teleop rate limiters, and the explicit arm-joint-4 cap.  The bridge then
performs another finite-value check and clips targets to the limits reported by
the Dexcontrol component.  It clips base linear and angular velocities to the
Dexcontrol chassis limits and replaces a stale base command with zero after
`cmd_vel_timeout_s` (0.5 s by default).  The resulting values are passed to
`component.set_joint_pos(..., wait_time=0.0)` and
`chassis.set_velocity(..., wait_time=0.0, sequential_steering=False)`.

Consequently, the teleop topics are intermediate commands, not an exact audit
of what is handed to the hardware API.  The implementation adds two read-only
audit topics to `dexcontrol_ros`:

- `/dexcontrol/applied_joint_commands` (`sensor_msgs/msg/JointState`)
- `/dexcontrol/applied_base_twist` (`geometry_msgs/msg/TwistStamped`)

These audit publishers do not feed back into control. A joint message normally
contains every successfully applied component; after a partial vendor failure
it contains only the successful subset, causing the strict recorder adapter to
invalidate that tick. A failed chassis call publishes non-finite invalidation
values. Those values are never recorded: they clear the cached base action and
the whole tick is dropped. The recorder uses valid, complete audit messages for
`action`.

## Measured feedback found in the repository

- `/joint_states` (`sensor_msgs/msg/JointState`): measured position, velocity,
  and effort from the Dexcontrol components at 100 Hz by default. Live F5D6
  validation showed that both hands provide position but no joint velocity.
  The compatibility topic substitutes zeros for those unavailable velocities.
- `/odom` (`nav_msgs/msg/Odometry`): integrated odometry at 50 Hz.  The twist
  is computed from measured steering angle and wheel velocity when available,
  but the existing bridge can fall back to the cached command.

The implementation also adds `/dexcontrol/measured_joint_states`
(`sensor_msgs/msg/JointState`), which publishes only when every real joint
position and velocity read succeeds. It is empty on the live F5D6 because hand
velocities are unavailable. The recorder therefore reads measured positions
from `/joint_states`, ignores its entire velocity array, and stores no joint
velocity. This avoids labeling compatibility zeros as measured values. The
implementation also adds
`/dexcontrol/measured_base_twist` (`geometry_msgs/msg/TwistStamped`).  It is
published only when the chassis steering and wheel feedback calculation
succeeds.  The recorder uses this topic, not `/odom`, in `observation.state`.

The base convention is the ROS `base` child frame: +x forward, +y left, +z up;
linear x/y are m/s and angular z is rad/s.

## Camera interface found in the repository

Live measurement on 2026-07-16 confirmed direct DexComm Zenoh inputs:

- `sensors/head_camera/left_rgb`: 30.0 Hz, 960 x 600 x 3, `uint8 RGB`;
  source-to-local-receive delay 156-177 ms.
- `sensors/head_camera/depth`: 29.7 Hz, 960 x 600, `float32` metres;
  source-to-local-receive delay 160-188 ms.

The former `/head_camera/image_rgb` raw DDS mirror was removed. It copied about
2.76 MB per 1280 x 720 frame and coupled recorder throughput to synchronous
headset encoding/network writes. The recorder now subscribes directly to RGB,
while `dexmate_head_camera_vision` uses a separate direct subscription and
capacity-one RTC output worker. Source, receive, and transport ages remain
distinct metadata. Depth is monitored but is not part of the current LeRobot
feature schema.

The vendor configuration already names future streams
`sensors/left_wrist_camera/rgb` and `sensors/right_wrist_camera/rgb`, but there
is no installed wrist-camera hardware to validate them. They are therefore not
claimed as current direct camera interfaces.

## Dataset action vector

`action` has 27 `float32` values. Torso, head, and arm entries are absolute
position targets in radians after bridge validation/clipping. Each six-driver
hand target is reconstructed from the post-bridge audit as the same two logical
controls used by Pico teleoperation. Base entries are the velocity most recently
handed to the chassis API.

| Indices | Values | Units | Frame | Semantics |
| --- | --- | --- | --- | --- |
| 0:3 | `torso_j1..torso_j3` | rad | joint local | absolute, commanded |
| 3:6 | `head_j1..head_j3` | rad | joint local | absolute, commanded |
| 6:13 | `L_arm_j1..L_arm_j7` | rad | joint local | absolute, commanded |
| 13:20 | `R_arm_j1..R_arm_j7` | rad | joint local | absolute, commanded |
| 20:22 | left `open_close_ratio,thumb_opposition_ratio` | ratio `[0,1]` | left-hand synergy | reconstructed from applied targets |
| 22:24 | right `open_close_ratio,thumb_opposition_ratio` | ratio `[0,1]` | right-hand synergy | reconstructed from applied targets |
| 24:26 | `base_vx,base_vy` | m/s | `base` | commanded velocity |
| 26:27 | `base_wz` | rad/s | `base` | commanded velocity |

Pico `trigger` is `open_close_ratio`: `0` is the configured open endpoint and
`1` is the configured closed endpoint for thumb/finger flexion. Pico `grip` is
`thumb_opposition_ratio`: `0` is unopposed and `1` is opposed. The five applied
flexion drivers must agree on one ratio within the configured tolerance. An
off-synergy applied hand target invalidates the sample rather than being
compressed lossily.

## Dataset observation-state vector

`observation.state` has 27 `float32` values. It contains measured torso, head,
and arm positions; two measured-derived synergy coordinates per hand; and
measured chassis velocity. Joint velocity is intentionally omitted because it
is not available for both hands. The measured hand coordinates are projected
from the six driver positions and clipped to their physical ratio range.

| Indices | Values | Units | Frame | Semantics |
| --- | --- | --- | --- | --- |
| 0:3 | `torso_j1..torso_j3` | rad | joint local | absolute, measured |
| 3:6 | `head_j1..head_j3` | rad | joint local | absolute, measured |
| 6:13 | `L_arm_j1..L_arm_j7` | rad | joint local | absolute, measured |
| 13:20 | `R_arm_j1..R_arm_j7` | rad | joint local | absolute, measured |
| 20:22 | left `open_close_ratio,thumb_opposition_ratio` | ratio `[0,1]` | left-hand synergy | derived from measured positions |
| 22:24 | right `open_close_ratio,thumb_opposition_ratio` | ratio `[0,1]` | right-hand synergy | derived from measured positions |
| 24:26 | `base_vx,base_vy` | m/s | `base` | measured velocity |
| 26:27 | `base_wz` | rad/s | `base` | measured velocity |

The action and state adapters look up all source joints by name before
reordering and reducing the hands; they never trust incoming `JointState` array
order. Missing, duplicate, non-finite, wrong-size, off-synergy applied, or stale
inputs cause a sample drop rather than a partially populated frame.
