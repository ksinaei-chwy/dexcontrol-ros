from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("dex_pico_teleop"))
    default_config = package_share / "config" / "vega_pico_teleop.yaml"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=str(default_config),
                description="Path to the Pico teleop ROS parameter file.",
            ),
            DeclareLaunchArgument(
                "publish_commands",
                default_value="true",
                description="Set false for dry-run packet/IK/status testing.",
            ),
            DeclareLaunchArgument(
                "network_transport",
                default_value="tcp",
                description="Pico packet transport: udp or tcp.",
            ),
            DeclareLaunchArgument(
                "network_host",
                default_value="0.0.0.0",
                description="Local interface address to bind the Pico receiver.",
            ),
            DeclareLaunchArgument(
                "network_port",
                default_value="63901",
                description="Local TCP/UDP port for Pico packets.",
            ),
            Node(
                package="dex_pico_teleop",
                executable="pico_teleop_node",
                name="pico_teleop_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("config_file"),
                    {
                        "publish_commands": ParameterValue(
                            LaunchConfiguration("publish_commands"),
                            value_type=bool,
                        ),
                        "network_transport": LaunchConfiguration("network_transport"),
                        "network_host": LaunchConfiguration("network_host"),
                        "network_port": ParameterValue(
                            LaunchConfiguration("network_port"),
                            value_type=int,
                        ),
                    },
                ],
            ),
        ]
    )
