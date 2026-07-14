from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nav_share = Path(get_package_share_directory("dexcontrol_navigation"))
    bridge_share = Path(get_package_share_directory("dexcontrol_ros"))
    nav2_share = Path(get_package_share_directory("nav2_bringup"))

    bridge_launch = bridge_share / "launch" / "dexcontrol_bridge.launch.py"
    bridge_config = nav_share / "config" / "vega_mapping_bridge.yaml"
    cloud_to_scan_config = nav_share / "config" / "cloud_to_scan.yaml"
    scan_merger_config = nav_share / "config" / "scan_merger.yaml"
    scan_self_filter_config = nav_share / "config" / "scan_self_filter.yaml"
    nav2_params = nav_share / "config" / "nav2_params.yaml"
    default_map = nav_share / "maps" / "lab_test.yaml"
    bringup_launch = nav2_share / "launch" / "bringup_launch.py"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_bridge",
                default_value="true",
                description="Launch dexcontrol_bridge before Nav2.",
            ),
            DeclareLaunchArgument(
                "robot_name",
                default_value=EnvironmentVariable(
                    "ROBOT_NAME", default_value="dm/vg150fef71c9-1p"
                ),
                description="Dexmate robot namespace.",
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
                description="Optional direct robot endpoint as <ip>:<port>.",
            ),
            DeclareLaunchArgument(
                "publish_robot_description",
                default_value="true",
                description="Let dexcontrol_bridge launch robot_state_publisher.",
            ),
            DeclareLaunchArgument(
                "map",
                default_value=str(default_map),
                description="Path to the saved occupancy-grid YAML map.",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=str(nav2_params),
                description="Path to Nav2 parameters YAML.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="false",
                description="Launch RViz with the Nav2 config.",
            ),
            DeclareLaunchArgument(
                "front_points_topic",
                default_value="/lidar_3d_front/points",
                description="Front 3D lidar PointCloud2 topic.",
            ),
            DeclareLaunchArgument(
                "back_points_topic",
                default_value="/lidar_3d_back/points",
                description="Back 3D lidar PointCloud2 topic.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(bridge_launch)),
                launch_arguments={
                    "robot_name": LaunchConfiguration("robot_name"),
                    "zenoh_config": LaunchConfiguration("zenoh_config"),
                    "robot_ip": LaunchConfiguration("robot_ip"),
                    "publish_robot_description": LaunchConfiguration("publish_robot_description"),
                    "config_file": str(bridge_config),
                }.items(),
                condition=IfCondition(LaunchConfiguration("use_bridge")),
            ),
            Node(
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                name="front_cloud_to_scan",
                output="screen",
                parameters=[str(cloud_to_scan_config)],
                remappings=[
                    ("cloud_in", LaunchConfiguration("front_points_topic")),
                    ("scan", "/lidar_3d_front/scan_raw"),
                ],
            ),
            Node(
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                name="back_cloud_to_scan",
                output="screen",
                parameters=[str(cloud_to_scan_config)],
                remappings=[
                    ("cloud_in", LaunchConfiguration("back_points_topic")),
                    ("scan", "/lidar_3d_back/scan"),
                ],
            ),
            Node(
                package="dexcontrol_navigation",
                executable="scan_self_filter",
                name="front_scan_self_filter",
                output="screen",
                parameters=[str(scan_self_filter_config)],
            ),
            Node(
                package="dexcontrol_navigation",
                executable="scan_merger",
                name="dual_lidar_scan_merger",
                output="screen",
                parameters=[str(scan_merger_config)],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(bringup_launch)),
                launch_arguments={
                    "map": LaunchConfiguration("map"),
                    "params_file": LaunchConfiguration("params_file"),
                    "use_sim_time": "false",
                    "autostart": "true",
                    "slam": "False",
                    "use_composition": "False",
                    "use_respawn": "False",
                }.items(),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", str(nav_share / "config" / "nav2.rviz")],
                condition=IfCondition(LaunchConfiguration("use_rviz")),
            ),
        ]
    )
