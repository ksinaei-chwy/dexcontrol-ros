from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from pathlib import Path


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("dexcontrol_ros"))
    description_share = Path(get_package_share_directory("dexmate_vega_description"))
    config_file = package_share / "config" / "vega_bridge.yaml"
    robot_description_file = description_share / "urdf" / "vega_1p_f5d6.package.urdf"
    robot_description = robot_description_file.read_text(encoding="utf-8")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_name",
                default_value=EnvironmentVariable(
                    "ROBOT_NAME", default_value="dm/vg150fef71c9-1p"
                ),
                description="Dexmate robot namespace, for example dm/vg150fef71c9-1p.",
            ),
            DeclareLaunchArgument(
                "zenoh_config",
                default_value=EnvironmentVariable(
                    "ZENOH_CONFIG",
                    default_value=[
                        EnvironmentVariable("HOME"),
                        "/.dexmate/comm/zenoh/chewy/zenoh_peer_config.json5",
                    ],
                ),
                description="Path to the DexComm/Zenoh config file.",
            ),
            DeclareLaunchArgument(
                "robot_ip",
                default_value=EnvironmentVariable("ROBOT_IP", default_value=""),
                description=(
                    "Optional direct robot endpoint as <ip>:<port>, for example "
                    "192.168.1.42:7447. Leave empty to use mDNS/multicast discovery."
                ),
            ),
            DeclareLaunchArgument(
                "publish_robot_description",
                default_value="true",
                description=(
                    "Start robot_state_publisher with the Vega 1p F5D6 URDF so "
                    "static and joint transforms come from dexmate_vega_description."
                ),
            ),
            SetEnvironmentVariable("ROBOT_NAME", LaunchConfiguration("robot_name")),
            SetEnvironmentVariable("DM_COMM_CONFIG", LaunchConfiguration("zenoh_config")),
            SetEnvironmentVariable("ZENOH_CONFIG", LaunchConfiguration("zenoh_config")),
            SetEnvironmentVariable(
                "ROBOT_IP",
                LaunchConfiguration("robot_ip"),
                condition=IfCondition(
                    PythonExpression(["'", LaunchConfiguration("robot_ip"), "' != ''"])
                ),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
                condition=IfCondition(
                    LaunchConfiguration("publish_robot_description")
                ),
            ),
            Node(
                package="dexcontrol_ros",
                executable="dexcontrol_bridge",
                name="dexcontrol_bridge",
                output="screen",
                parameters=[str(config_file)],
            )
        ]
    )
