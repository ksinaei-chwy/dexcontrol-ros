from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    description_share = Path(get_package_share_directory("dexmate_vega_description"))
    default_urdf = description_share / "urdf" / "vega_1p_f5d6.package.urdf"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_urdf_path",
                default_value=str(default_urdf),
                description="Path to the Vega URDF to display in MeshCat.",
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
            DeclareLaunchArgument(
                "topic",
                default_value="/dex_pico_teleop/log_frame",
                description="Teleop log frame topic to visualize.",
            ),
            DeclareLaunchArgument(
                "meshcat_use_joint_state_initial_pose",
                default_value="true",
                description="Use /joint_states only before the first teleop log frame.",
            ),
            DeclareLaunchArgument(
                "meshcat_show_visuals",
                default_value="true",
                description="Display real Vega visual meshes in MeshCat.",
            ),
            DeclareLaunchArgument(
                "meshcat_show_collisions",
                default_value="false",
                description="Display collision geometry in MeshCat.",
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
                        "topic": LaunchConfiguration("topic"),
                        "meshcat_use_joint_state_initial_pose": ParameterValue(
                            LaunchConfiguration("meshcat_use_joint_state_initial_pose"),
                            value_type=bool,
                        ),
                        "meshcat_show_visuals": ParameterValue(
                            LaunchConfiguration("meshcat_show_visuals"),
                            value_type=bool,
                        ),
                        "meshcat_show_collisions": ParameterValue(
                            LaunchConfiguration("meshcat_show_collisions"),
                            value_type=bool,
                        ),
                    }
                ],
            ),
        ]
    )
