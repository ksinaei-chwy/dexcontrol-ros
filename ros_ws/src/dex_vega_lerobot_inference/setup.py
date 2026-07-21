from glob import glob
from pathlib import Path

from setuptools import find_packages, setup


package_name = "dex_vega_lerobot_inference"


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
        (f"share/{package_name}/requirements", package_files("requirements/*")),
        (f"share/{package_name}/scripts", package_files("scripts/*")),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="Dexmate ROS User",
    maintainer_email="contact@dexmate.ai",
    description="Guarded LeRobot PI0.5 and GR00T inference for Dexmate Vega 1 Pro.",
    license="AGPL-3.0-or-later",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "inference_node = dex_vega_lerobot_inference.inference_node:main",
            "download_model = dex_vega_lerobot_inference.download_model:main",
            "download_groot_model = "
            "dex_vega_lerobot_inference.download_groot_model:main",
            "benchmark_groot = dex_vega_lerobot_inference.benchmark_groot:main",
            "policy_server = dex_vega_lerobot_inference.policy_server:main",
            "validate_runtime = dex_vega_lerobot_inference.validate_runtime:main",
        ],
    },
)
