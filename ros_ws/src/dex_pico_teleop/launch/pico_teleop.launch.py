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
                "control_rate_hz",
                default_value="50.0",
                description="Pico retargeting and joint-position-target rate in Hz.",
            ),
            DeclareLaunchArgument(
                "network_enabled",
                default_value="true",
                description="Enable the Pico packet network receiver.",
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
            DeclareLaunchArgument(
                "pink_self_collision_enabled",
                default_value="true",
                description="Enable the unified Pink bimanual self-collision barrier.",
            ),
            DeclareLaunchArgument(
                "pink_self_collision_arm_max_iterations",
                default_value="1",
                description="Deprecated compatibility parameter; bimanual Pink always integrates once.",
            ),
            DeclareLaunchArgument(
                "pink_collision_pipeline",
                default_value="reduced_all_pairs",
                description="Collision pipeline: reduced_all_pairs or closest_pairs.",
            ),
            DeclareLaunchArgument(
                "pink_collision_sphere_count",
                default_value="18",
                description="Reduced collision profile size: 18, 30, 40, or 50 spheres.",
            ),
            DeclareLaunchArgument(
                "pink_collision_sphere_inflation",
                default_value="1.0",
                description="Radius multiplier for reduced collision spheres.",
            ),
            DeclareLaunchArgument(
                "head_tracking_enabled",
                default_value="false",
                description="Enable Pico headset orientation tracking for Vega head joints.",
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
                        "control_rate_hz": ParameterValue(
                            LaunchConfiguration("control_rate_hz"),
                            value_type=float,
                        ),
                        "network_enabled": ParameterValue(
                            LaunchConfiguration("network_enabled"),
                            value_type=bool,
                        ),
                        "network_transport": LaunchConfiguration("network_transport"),
                        "network_host": LaunchConfiguration("network_host"),
                        "network_port": ParameterValue(
                            LaunchConfiguration("network_port"),
                            value_type=int,
                        ),
                        "pink_self_collision_enabled": ParameterValue(
                            LaunchConfiguration("pink_self_collision_enabled"),
                            value_type=bool,
                        ),
                        "pink_self_collision_arm_max_iterations": ParameterValue(
                            LaunchConfiguration("pink_self_collision_arm_max_iterations"),
                            value_type=int,
                        ),
                        "pink_collision_pipeline": LaunchConfiguration(
                            "pink_collision_pipeline"
                        ),
                        "pink_collision_sphere_count": ParameterValue(
                            LaunchConfiguration("pink_collision_sphere_count"),
                            value_type=int,
                        ),
                        "pink_collision_sphere_inflation": ParameterValue(
                            LaunchConfiguration("pink_collision_sphere_inflation"),
                            value_type=float,
                        ),
                        "head_tracking_enabled": ParameterValue(
                            LaunchConfiguration("head_tracking_enabled"),
                            value_type=bool,
                        ),
                    },
                ],
            ),
        ]
    )
