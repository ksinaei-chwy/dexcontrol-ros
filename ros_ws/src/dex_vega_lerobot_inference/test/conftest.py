from pathlib import Path

import pytest

from dex_vega_lerobot_recorder.configuration import load_config


SRC_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def recorder_config():
    return load_config(
        SRC_ROOT
        / "dex_vega_lerobot_recorder"
        / "config"
        / "dexmate_blue_bird.yaml"
    )


@pytest.fixture()
def vega_urdf_path() -> Path:
    return (
        SRC_ROOT
        / "dexmate_vega_description"
        / "urdf"
        / "vega_1p_f5d6.package.urdf"
    )
