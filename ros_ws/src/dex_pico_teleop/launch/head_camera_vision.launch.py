from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("dex_pico_teleop"))
    default_config = package_share / "config" / "head_camera_vision.yaml"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=str(default_config),
                description=(
                    "Path to the head camera Remote Vision parameter file."
                ),
            ),
            DeclareLaunchArgument(
                "enabled",
                default_value="true",
                description="Start the RTC stream immediately.",
            ),
            DeclareLaunchArgument(
                "stream_name",
                default_value="left_rgb",
                description=(
                    "Dexmate head camera RGB stream: left_rgb or right_rgb."
                ),
            ),
            DeclareLaunchArgument(
                "camera_transport",
                default_value="zenoh",
                description="Dexmate camera source transport: rtc or zenoh.",
            ),
            DeclareLaunchArgument(
                "camera_topic",
                default_value="sensors/head_camera/left_rgb",
                description="Direct DexTop/DexComm RGB Zenoh topic.",
            ),
            DeclareLaunchArgument(
                "source_rtc_channel",
                default_value="sensors/head_camera/left_rgb_rtc",
                description="Direct DexTop RGB RTC channel when RTC input is used.",
            ),
            DeclareLaunchArgument(
                "depth_enabled",
                default_value="true",
                description="Monitor the direct DexTop depth stream.",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="sensors/head_camera/depth",
                description="Direct DexTop/DexComm float32 depth Zenoh topic.",
            ),
            DeclareLaunchArgument(
                "rtc_enabled",
                default_value="true",
                description="Enable DexComm RTC Remote Vision output.",
            ),
            DeclareLaunchArgument(
                "rtc_channel",
                default_value=(
                    "xrobotoolkit/remote_vision/head_camera/left_rgb_rtc"
                ),
                description=(
                    "RTC output channel for XRoboToolkit Remote Vision."
                ),
            ),
            DeclareLaunchArgument(
                "source_codec",
                default_value="h264",
                description="Dexmate camera source codec: h264, vp8, or auto.",
            ),
            DeclareLaunchArgument(
                "codec",
                default_value="auto",
                description="Headset output codec: auto, h264, or vp8.",
            ),
            DeclareLaunchArgument(
                "width",
                default_value="1280",
                description="Output video width. Set <=0 to use source width.",
            ),
            DeclareLaunchArgument(
                "height",
                default_value="720",
                description=(
                    "Output video height. Set <=0 to use source height."
                ),
            ),
            DeclareLaunchArgument(
                "fps",
                default_value="30.0",
                description="Output video frame rate.",
            ),
            DeclareLaunchArgument(
                "bitrate",
                default_value="1500000",
                description="Output video bitrate in bits per second.",
            ),
            DeclareLaunchArgument(
                "xrtcp_enabled",
                default_value="false",
                description="Enable XRoboToolkit ZED Mini TCP H.264 output.",
            ),
            DeclareLaunchArgument(
                "xrtcp_host",
                default_value="",
                description="Pico headset IP for ZED Mini TCP output.",
            ),
            DeclareLaunchArgument(
                "xrtcp_port",
                default_value="12345",
                description="Pico ZED Mini TCP streaming port.",
            ),
            DeclareLaunchArgument(
                "xrtcp_side_by_side",
                default_value="true",
                description="Duplicate mono RGB into stereo side-by-side.",
            ),
            DeclareLaunchArgument(
                "xrtcp_bitrate",
                default_value="4000000",
                description="ZED Mini TCP H.264 bitrate in bits per second.",
            ),
            DeclareLaunchArgument(
                "xrtcp_write_timeout_s",
                default_value="2.0",
                description="Socket write timeout for ZED Mini TCP streaming.",
            ),
            Node(
                package="dex_pico_teleop",
                executable="dexmate_head_camera_vision",
                name="dexmate_head_camera_vision",
                output="screen",
                parameters=[
                    LaunchConfiguration("config_file"),
                    {
                        "enabled": ParameterValue(
                            LaunchConfiguration("enabled"),
                            value_type=bool,
                        ),
                        "stream_name": LaunchConfiguration("stream_name"),
                        "camera_transport": LaunchConfiguration(
                            "camera_transport"
                        ),
                        "camera_topic": LaunchConfiguration("camera_topic"),
                        "source_rtc_channel": LaunchConfiguration(
                            "source_rtc_channel"
                        ),
                        "depth_enabled": ParameterValue(
                            LaunchConfiguration("depth_enabled"),
                            value_type=bool,
                        ),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        "rtc_enabled": ParameterValue(
                            LaunchConfiguration("rtc_enabled"),
                            value_type=bool,
                        ),
                        "rtc_channel": LaunchConfiguration("rtc_channel"),
                        "source_codec": LaunchConfiguration("source_codec"),
                        "codec": LaunchConfiguration("codec"),
                        "width": ParameterValue(
                            LaunchConfiguration("width"),
                            value_type=int,
                        ),
                        "height": ParameterValue(
                            LaunchConfiguration("height"),
                            value_type=int,
                        ),
                        "fps": ParameterValue(
                            LaunchConfiguration("fps"),
                            value_type=float,
                        ),
                        "bitrate": ParameterValue(
                            LaunchConfiguration("bitrate"),
                            value_type=int,
                        ),
                        "xrtcp_enabled": ParameterValue(
                            LaunchConfiguration("xrtcp_enabled"),
                            value_type=bool,
                        ),
                        "xrtcp_host": LaunchConfiguration("xrtcp_host"),
                        "xrtcp_port": ParameterValue(
                            LaunchConfiguration("xrtcp_port"),
                            value_type=int,
                        ),
                        "xrtcp_side_by_side": ParameterValue(
                            LaunchConfiguration("xrtcp_side_by_side"),
                            value_type=bool,
                        ),
                        "xrtcp_bitrate": ParameterValue(
                            LaunchConfiguration("xrtcp_bitrate"),
                            value_type=int,
                        ),
                        "xrtcp_write_timeout_s": ParameterValue(
                            LaunchConfiguration("xrtcp_write_timeout_s"),
                            value_type=float,
                        ),
                    },
                ],
            ),
        ]
    )
