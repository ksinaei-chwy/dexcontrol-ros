from pathlib import Path
import os

import numpy as np
import pytest
import yaml
from dex_camera_transport import CameraFrame, CameraSourceStats

rclpy = pytest.importorskip("rclpy")
from geometry_msgs.msg import TwistStamped  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from dex_vega_lerobot_recorder.hand_synergy import expand_hand_synergy  # noqa: E402
from dex_vega_lerobot_recorder.recorder_node import (  # noqa: E402
    VegaLeRobotRecorderNode,
)

from helpers import CONFIG_PATH, FakeEpisodeWriter  # noqa: E402


def write_test_config(tmp_path: Path) -> Path:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data["dataset"]["name"] = "mocked_ros"
    data["dataset"]["local_save_directory"] = str(tmp_path)
    data["dataset"]["recording_fps"] = 10
    data["episode_control"]["minimum_frames"] = 1
    data["episode_control"]["minimum_duration_seconds"] = 0.0
    data["episode_control"]["input_backend"] = "disabled"
    for camera in data["cameras"].values():
        camera["resolution"] = {"width": 4, "height": 3}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_synthetic_ros_messages_record_one_committed_episode(tmp_path):
    os.environ["ROS_LOG_DIR"] = str(tmp_path / "ros_logs")
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    if not rclpy.ok():
        rclpy.init()
    created = {}

    def writer_factory(*_args, **_kwargs):
        writer = FakeEpisodeWriter()
        created["writer"] = writer
        return writer

    class FakeCameraSource:
        def __init__(self, **_kwargs):
            self.image = np.zeros((3, 4, 3), dtype=np.uint8)
            self.image[:, :, 0] = 255

        def snapshot(self, *, now_ns, **_kwargs):
            return CameraFrame(
                data=self.image,
                source_stamp_ns=now_ns - 10_000_000,
                receive_stamp_ns=now_ns - 1_000_000,
                sequence=1,
            )

        def stats(self):
            return CameraSourceStats(
                unique_frames=1,
                invalid_frames=0,
                source_fps=30.0,
                last_sequence=1,
                last_source_stamp_ns=1,
                last_receive_stamp_ns=1,
                last_error="",
                shape=self.image.shape,
                dtype="uint8",
            )

        def shutdown(self):
            pass

    recorder = VegaLeRobotRecorderNode(
        cli_config_file=str(write_test_config(tmp_path)),
        cli_no_hf_upload=True,
        writer_factory=writer_factory,
        camera_source_factory=FakeCameraSource,
    )
    publisher = Node("synthetic_recorder_inputs")
    topics = recorder.config.topics
    state_pub = publisher.create_publisher(JointState, topics.joint_states, 20)
    action_pub = publisher.create_publisher(
        JointState, topics.applied_joint_commands, 20
    )
    measured_base_pub = publisher.create_publisher(
        TwistStamped, topics.measured_base_twist, 20
    )
    applied_base_pub = publisher.create_publisher(
        TwistStamped, topics.applied_base_twist, 20
    )
    executor = SingleThreadedExecutor()
    executor.add_node(recorder)
    executor.add_node(publisher)

    names = list(recorder.config.robot_features.joint_names)
    measured_positions = {name: float(index) for index, name in enumerate(names)}
    applied_positions = {
        name: float(index + 100) for index, name in enumerate(names)
    }
    measured_ratios = ((0.25, 0.40), (0.35, 0.50))
    applied_ratios = ((0.60, 0.70), (0.80, 0.90))
    for synergy, measured_ratio, applied_ratio in zip(
        recorder.config.robot_features.hand_synergies,
        measured_ratios,
        applied_ratios,
    ):
        measured_positions.update(
            zip(synergy.joint_names, expand_hand_synergy(synergy, *measured_ratio))
        )
        applied_positions.update(
            zip(synergy.joint_names, expand_hand_synergy(synergy, *applied_ratio))
        )
    for _ in range(8):
        stamp = publisher.get_clock().now().to_msg()
        state = JointState()
        state.header.stamp = stamp
        state.name = list(reversed(names))
        state.position = [measured_positions[name] for name in reversed(names)]
        state_pub.publish(state)

        action = JointState()
        action.header.stamp = stamp
        action.name = list(reversed(names))
        action.position = [applied_positions[name] for name in reversed(names)]
        action_pub.publish(action)

        measured = TwistStamped()
        measured.header.stamp = stamp
        measured.twist.linear.x = 0.1
        measured.twist.linear.y = 0.2
        measured.twist.angular.z = 0.3
        measured_base_pub.publish(measured)
        applied = TwistStamped()
        applied.header.stamp = stamp
        applied.twist.linear.x = 0.4
        applied.twist.linear.y = 0.5
        applied.twist.angular.z = 0.6
        applied_base_pub.publish(applied)

        executor.spin_once(timeout_sec=0.05)

    recorder.start_episode()
    recorder._on_recording_tick()
    recorder._frame_worker.flush()
    recorder.stop_episode()
    recorder.save_episode()

    writer = created["writer"]
    assert writer.committed_episodes == 1
    assert len(writer.committed[0]) == 1
    frame = writer.committed[0][0]
    assert frame["observation.state"].shape == (27,)
    assert frame["action"].shape == (27,)
    np.testing.assert_allclose(frame["observation.state"][20:24], [0.25, 0.4, 0.35, 0.5])
    np.testing.assert_allclose(frame["action"][20:24], [0.6, 0.7, 0.8, 0.9])
    assert frame["observation.images.head"].shape == (3, 4, 3)
    assert frame["observation.images.head"][0, 0].tolist() == [255, 0, 0]
    assert np.count_nonzero(frame["observation.images.left_wrist"]) == 0
    assert np.count_nonzero(frame["observation.images.right_wrist"]) == 0
    assert frame["task"] == recorder.config.dataset.task_instruction

    executor.remove_node(recorder)
    executor.remove_node(publisher)
    recorder.destroy_node()
    publisher.destroy_node()
    executor.shutdown()
    if rclpy.ok():
        rclpy.shutdown()
