#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <ros_ws_project_root> [python3.12]" >&2
  exit 2
fi

project_root="$(realpath "$1")"
python_bin="${2:-python3.12}"
venv_path="${project_root}/.venvs/lerobot"
package_root="${project_root}/src/dex_vega_lerobot_inference"

if [[ ! -f "${package_root}/package.xml" ]]; then
  echo "project root does not contain src/dex_vega_lerobot_inference: ${project_root}" >&2
  exit 2
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "${python_bin} is unavailable; LeRobot 0.6.0 requires Python >=3.12" >&2
  exit 2
fi

"${python_bin}" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit("LeRobot 0.6.0 requires Python >=3.12")
PY

"${python_bin}" -m venv --system-site-packages "${venv_path}"

# Fail before installing anything if the inherited framework is a CPU/generic
# build. On JetPack 7, use an NVIDIA iGPU PyTorch container/build as documented.
"${venv_path}/bin/python" - <<'PY'
import torch
import torchvision
if torch.version.cuda is None or not torch.cuda.is_available():
    raise SystemExit(
        "CUDA-enabled NVIDIA PyTorch must already be visible through "
        "--system-site-packages; refusing to install a generic torch wheel"
    )
if not torch.cuda.is_bf16_supported():
    raise SystemExit("the CUDA device does not report bfloat16 support")
print(
    f"validated NVIDIA torch={torch.__version__}, "
    f"torchvision={torchvision.__version__}, CUDA={torch.version.cuda}"
)
PY

"${venv_path}/bin/python" -m pip install \
  --constraint "${package_root}/requirements/lerobot-0.6-pi-no-torch.constraints.txt" \
  -r "${package_root}/requirements/lerobot-0.6-pi-no-torch.txt"
"${venv_path}/bin/python" -m pip install --no-deps "lerobot==0.6.0"
"${venv_path}/bin/python" -m pip install --no-deps \
  "${project_root}/src/dex_vega_lerobot_recorder"
"${venv_path}/bin/python" -m pip install --no-deps "${package_root}"

"${venv_path}/bin/python" - <<'PY'
import diffusers
import peft
import timm
import tree
print(
    "validated GR00T dependencies "
    f"diffusers={diffusers.__version__}, peft={peft.__version__}, "
    f"timm={timm.__version__}, dm-tree={tree.__file__}"
)
PY

"${venv_path}/bin/validate_runtime" --require-groot
