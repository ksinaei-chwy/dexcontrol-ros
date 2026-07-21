from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("dex_vega_lerobot_inference"))
    default_config = str(share / "config" / "groot_n17_blue_bird.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            Node(
                package="dex_vega_lerobot_inference",
                executable="inference_node",
                name="dex_vega_lerobot_inference",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "policy_type": "groot",
                        "mode": "observe_only",
                        "allow_command_publication": False,
                        "execution_readiness_acknowledged": False,
                    },
                ],
            ),
        ]
    )
