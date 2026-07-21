from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("dex_vega_lerobot_recorder"))
    default_config = package_share / "config" / "vega_lerobot_recording.yaml"
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=str(default_config),
                description="Recorder application YAML file.",
            ),
            DeclareLaunchArgument(
                "no_hf_upload",
                default_value="false",
                description="Force local-only mode even if YAML enables upload.",
            ),
            DeclareLaunchArgument(
                "overwrite",
                default_value="false",
                description="Explicitly replace an existing local dataset.",
            ),
            DeclareLaunchArgument(
                "resume",
                default_value="false",
                description="Safely append to a finalized local LeRobotDataset.",
            ),
            DeclareLaunchArgument(
                "input_backend",
                default_value="",
                description="Override YAML: terminal, linux_input_event, or disabled.",
            ),
            DeclareLaunchArgument(
                "input_device",
                default_value="",
                description="Override Linux input-event device path.",
            ),
            DeclareLaunchArgument(
                "shutdown_upload_timeout_s",
                default_value="1800",
                description=(
                    "Grace period after Ctrl+C before ROS launch sends SIGTERM. "
                    "Allows a configured on_session_end Hub upload to complete."
                ),
            ),
            Node(
                package="dex_vega_lerobot_recorder",
                executable="record_teleop_dataset",
                name="dex_vega_lerobot_recorder",
                output="screen",
                emulate_tty=True,
                # LeRobot's Hub push is intentionally synchronous so a clean
                # process exit means the configured session-end upload finished.
                # The ROS launch defaults (5 s SIGINT, 10 s SIGTERM) are much
                # shorter than a multi-hundred-MB video upload.
                sigterm_timeout=LaunchConfiguration("shutdown_upload_timeout_s"),
                sigkill_timeout="10",
                parameters=[
                    {
                        "config_file": LaunchConfiguration("config_file"),
                        "no_hf_upload": ParameterValue(
                            LaunchConfiguration("no_hf_upload"), value_type=bool
                        ),
                        "overwrite": ParameterValue(
                            LaunchConfiguration("overwrite"), value_type=bool
                        ),
                        "resume": ParameterValue(
                            LaunchConfiguration("resume"), value_type=bool
                        ),
                        "input_backend": LaunchConfiguration("input_backend"),
                        "input_device": LaunchConfiguration("input_device"),
                    }
                ],
            ),
        ]
    )
