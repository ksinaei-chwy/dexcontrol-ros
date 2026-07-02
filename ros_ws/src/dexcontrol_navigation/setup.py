from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "dexcontrol_navigation"


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
        (f"share/{package_name}/config", package_files("config/*.rviz")),
        (f"share/{package_name}/launch", package_files("launch/*.launch.py")),
        (f"share/{package_name}/maps", package_files("maps/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Dexmate ROS User",
    maintainer_email="contact@dexmate.ai",
    description="SLAM and navigation bringup for Dexmate Vega robots.",
    license="AGPL-3.0-or-later",
    entry_points={
        "console_scripts": [
            "scan_merger = dexcontrol_navigation.scan_merger:main",
            "scan_self_filter = dexcontrol_navigation.scan_self_filter:main",
        ],
    },
)
