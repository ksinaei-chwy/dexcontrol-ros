from setuptools import find_packages, setup


package_name = "dex_camera_transport"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Dexmate ROS User",
    maintainer_email="contact@dexmate.ai",
    description="ROS-independent latest-frame DexComm camera sources.",
    license="AGPL-3.0-or-later",
)
