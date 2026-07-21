from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("dex_vega_lerobot_inference"))
    default_config = str(share / "config" / "groot_n17_blue_bird.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument(
                "allow_command_publication",
                default_value="false",
                description="Explicitly permit construction of command publishers.",
            ),
            DeclareLaunchArgument(
                "execution_readiness_acknowledged",
                default_value="false",
                description=(
                    "Set true only after the pinned GR00T candidate's offline and "
                    "live observe-only results, limits, and rollout plan are reviewed."
                ),
            ),
            DeclareLaunchArgument(
                "require_teleop_disabled",
                default_value="true",
                description="Require a fresh disabled Pico status before arming.",
            ),
            DeclareLaunchArgument(
                "maximum_execution_duration_seconds",
                default_value="5.0",
                description="Finite first-stage armed interval before automatic FAULT.",
            ),
            Node(
                package="dex_vega_lerobot_inference",
                executable="inference_node",
                name="dex_vega_lerobot_inference",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "policy_type": "groot",
                        "mode": "armed",
                        "allow_command_publication": ParameterValue(
                            LaunchConfiguration("allow_command_publication"),
                            value_type=bool,
                        ),
                        "execution_readiness_acknowledged": ParameterValue(
                            LaunchConfiguration("execution_readiness_acknowledged"),
                            value_type=bool,
                        ),
                        "require_teleop_disabled": ParameterValue(
                            LaunchConfiguration("require_teleop_disabled"),
                            value_type=bool,
                        ),
                        "maximum_execution_duration_seconds": ParameterValue(
                            LaunchConfiguration("maximum_execution_duration_seconds"),
                            value_type=float,
                        ),
                    },
                ],
            ),
        ]
    )
