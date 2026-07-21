from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "dex_pico_teleop"


def package_files(pattern: str) -> list[str]:
    return [str(path) for path in glob(pattern, recursive=True) if Path(path).is_file()]


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/config", package_files("config/*.yaml")),
        (f"share/{package_name}/launch", package_files("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Dexmate ROS User",
    maintainer_email="contact@dexmate.ai",
    description="Pico XR teleoperation retargeting for Dexmate Vega.",
    license="AGPL-3.0-or-later",
    entry_points={
        "console_scripts": [
            "pico_teleop_node = dex_pico_teleop.teleop_node:main",
            "pico_meshcat_visualizer = dex_pico_teleop.meshcat_visualizer_node:main",
            "dexmate_head_camera_vision = dex_pico_teleop.head_camera_vision_node:main",
        ],
    },
)
