from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "dex_vega_lerobot_recorder"


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
        (f"share/{package_name}/docs", package_files("docs/*")),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="Dexmate ROS User",
    maintainer_email="contact@dexmate.ai",
    description="LeRobot v3 episodic teleoperation recorder for Dexmate Vega 1 Pro.",
    license="AGPL-3.0-or-later",
    entry_points={
        "console_scripts": [
            "record_teleop_dataset = dex_vega_lerobot_recorder.recorder_node:main",
            "upload_dataset = dex_vega_lerobot_recorder.upload_dataset:main",
        ],
    },
)
