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
    description_share = Path(get_package_share_directory("dexmate_vega_description"))
    default_urdf = description_share / "urdf" / "vega_1p_f5d6.package.urdf"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=str(default_config),
                description="Path to the Pico teleop ROS parameter file.",
            ),
            DeclareLaunchArgument(
                "control_rate_hz",
                default_value="50.0",
                description="Pico retargeting and joint-position-target rate in Hz.",
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
                "robot_urdf_path",
                default_value=str(default_urdf),
                description="Path to the Vega URDF for IK and MeshCat visuals.",
            ),
            DeclareLaunchArgument(
                "pink_self_collision_enabled",
                default_value="true",
                description="Enable Pink self-collision barriers for diagnostics.",
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
                "open_browser",
                default_value="false",
                description="Ask MeshCat to open a browser window.",
            ),
            DeclareLaunchArgument(
                "max_update_rate_hz",
                default_value="30.0",
                description="Maximum MeshCat display update rate.",
            ),
            Node(
                package="dex_pico_teleop",
                executable="pico_teleop_node",
                name="pico_teleop_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("config_file"),
                    {
                        "publish_commands": False,
                        "dry_run_simulated_feedback_enabled": True,
                        "kinematics_backend": "pink",
                        "control_rate_hz": ParameterValue(
                            LaunchConfiguration("control_rate_hz"),
                            value_type=float,
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
                        "network_transport": LaunchConfiguration("network_transport"),
                        "network_host": LaunchConfiguration("network_host"),
                        "network_port": ParameterValue(
                            LaunchConfiguration("network_port"),
                            value_type=int,
                        ),
                        "robot_urdf_path": LaunchConfiguration("robot_urdf_path"),
                    },
                ],
            ),
            Node(
                package="dex_pico_teleop",
                executable="pico_meshcat_visualizer",
                name="pico_meshcat_visualizer",
                output="screen",
                parameters=[
                    {
                        "robot_urdf_path": LaunchConfiguration("robot_urdf_path"),
                        "open_browser": ParameterValue(
                            LaunchConfiguration("open_browser"),
                            value_type=bool,
                        ),
                        "max_update_rate_hz": ParameterValue(
                            LaunchConfiguration("max_update_rate_hz"),
                            value_type=float,
                        ),
                        "topic": "/dex_pico_teleop/log_frame",
                        "meshcat_use_joint_state_initial_pose": True,
                        "meshcat_show_visuals": True,
                        "meshcat_show_collisions": True,
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
                    }
                ],
            ),
        ]
    )
