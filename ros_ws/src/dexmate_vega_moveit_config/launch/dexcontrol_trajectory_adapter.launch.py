from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "dry_run",
                default_value="false",
                description="If true, accept trajectories without publishing hardware commands.",
            ),
            DeclareLaunchArgument(
                "require_current_state",
                default_value="true",
                description="Require fresh /joint_states before accepting a trajectory.",
            ),
            DeclareLaunchArgument(
                "command_publish_rate_hz",
                default_value="100.0",
                description="Interpolation and JointState command publish rate.",
            ),
            Node(
                package="dexmate_vega_moveit_config",
                executable="dexcontrol_trajectory_adapter.py",
                name="dexcontrol_trajectory_adapter",
                output="screen",
                parameters=[
                    {
                        "dry_run": ParameterValue(
                            LaunchConfiguration("dry_run"), value_type=bool
                        ),
                        "require_current_state": ParameterValue(
                            LaunchConfiguration("require_current_state"), value_type=bool
                        ),
                        "command_publish_rate_hz": ParameterValue(
                            LaunchConfiguration("command_publish_rate_hz"), value_type=float
                        ),
                    }
                ],
            ),
        ]
    )
