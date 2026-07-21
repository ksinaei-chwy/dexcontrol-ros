from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("dex_vega_lerobot_inference"))
    default_config = str(share / "config" / "pi05_blue_bird.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument(
                "allow_command_publication",
                default_value="false",
                description=(
                    "Must be explicitly true to create bridge command publishers. "
                    "The arm service is still required afterward."
                ),
            ),
            DeclareLaunchArgument(
                "require_teleop_disabled",
                default_value="true",
                description=(
                    "Set false only when Pico teleop is confirmed absent. "
                    "Exclusive command-publisher checks remain enabled."
                ),
            ),
            DeclareLaunchArgument(
                "maximum_execution_duration_seconds",
                default_value="5.0",
                description=(
                    "Maximum armed interval before an automatic FAULT. "
                    "Increase only in staged, operator-supervised trials."
                ),
            ),
            Node(
                package="dex_vega_lerobot_inference",
                executable="inference_node",
                name="dex_vega_lerobot_inference",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "mode": "armed",
                        "allow_command_publication": ParameterValue(
                            LaunchConfiguration("allow_command_publication"),
                            value_type=bool,
                        ),
                        "require_teleop_disabled": ParameterValue(
                            LaunchConfiguration("require_teleop_disabled"),
                            value_type=bool,
                        ),
                        "maximum_execution_duration_seconds": ParameterValue(
                            LaunchConfiguration(
                                "maximum_execution_duration_seconds"
                            ),
                            value_type=float,
                        ),
                    },
                ],
            ),
        ]
    )
