from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _pkg_launch(package_name: str, launch_file: str) -> PythonLaunchDescriptionSource:
    return PythonLaunchDescriptionSource(
        str(Path(get_package_share_directory(package_name)) / "launch" / launch_file)
    )


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "dry_run",
                default_value="false",
                description="If true, adapter accepts trajectories without publishing commands.",
            ),
            IncludeLaunchDescription(
                _pkg_launch("dexmate_vega_moveit_config", "static_virtual_joint_tfs.launch.py")
            ),
            IncludeLaunchDescription(
                _pkg_launch("dexmate_vega_moveit_config", "rsp.launch.py")
            ),
            IncludeLaunchDescription(
                _pkg_launch("dexmate_vega_moveit_config", "move_group.launch.py")
            ),
            IncludeLaunchDescription(
                _pkg_launch("dexmate_vega_moveit_config", "moveit_rviz.launch.py")
            ),
            IncludeLaunchDescription(
                _pkg_launch("dexmate_vega_moveit_config", "dexcontrol_trajectory_adapter.launch.py"),
                launch_arguments={"dry_run": LaunchConfiguration("dry_run")}.items(),
            ),
        ]
    )
