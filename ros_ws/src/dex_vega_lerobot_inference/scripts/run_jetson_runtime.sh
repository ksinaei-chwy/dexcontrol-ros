#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <ros_ws_project_root> <guest-command> [args ...]" >&2
  exit 2
fi

project_root="$(realpath "$1")"
shift
runtime_root="${project_root}/.runtime/rootfs/nvidia-pytorch-25.11-arm64"
driver_root="/usr/lib/aarch64-linux-gnu/nvidia"

if [[ ! -f "${project_root}/src/dex_vega_lerobot_inference/package.xml" ]]; then
  echo "not a Dexmate ROS workspace: ${project_root}" >&2
  exit 2
fi
if [[ ! -x "${runtime_root}/usr/bin/python3.12" ]]; then
  echo "missing project-local NVIDIA runtime: ${runtime_root}" >&2
  echo "run bootstrap_jetson_runtime.sh first" >&2
  exit 2
fi
if [[ ! -s "${driver_root}/libcuda.so.1" ]]; then
  echo "Jetson driver mount is unavailable: ${driver_root}/libcuda.so.1" >&2
  exit 2
fi

mkdir -p \
  "${project_root}/.cache/huggingface" \
  "${project_root}/.cache/torch" \
  "${project_root}/.cache/triton" \
  "${project_root}/.runtime/home" \
  "${project_root}/.runtime/ros-log" \
  "${runtime_root}/usr/lib/aarch64-linux-gnu/nvidia"

# NVIDIA's Jetson container integration injects driver libraries as child
# mounts. The recursive bind is intentional: a plain bind exposes zero-byte
# overlay placeholders instead of the real libcuda files. All mounts live in a
# private namespace and disappear when the guest command exits.
exec unshare --mount --propagation private /bin/bash -c '
  set -euo pipefail
  guest_root="$1"
  workspace_root="$2"
  shift 2

  mount --rbind /dev "${guest_root}/dev"
  mount --make-rslave "${guest_root}/dev"
  mount -t proc proc "${guest_root}/proc"
  mount --rbind /sys "${guest_root}/sys"
  mount --make-rslave "${guest_root}/sys"
  mount --bind "${workspace_root}" "${guest_root}/workspace"
  mount --rbind /usr/lib/aarch64-linux-gnu/nvidia \
    "${guest_root}/usr/lib/aarch64-linux-gnu/nvidia"
  mount --make-rslave "${guest_root}/usr/lib/aarch64-linux-gnu/nvidia"

  if [[ -e /etc/resolv.conf && -e "${guest_root}/etc/resolv.conf" ]]; then
    mount --bind /etc/resolv.conf "${guest_root}/etc/resolv.conf"
  fi

  exec chroot "${guest_root}" /usr/bin/env \
    HOME=/workspace/.runtime/home \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    XDG_CACHE_HOME=/workspace/.cache \
    TORCH_HOME=/workspace/.cache/torch \
    TRITON_CACHE_DIR=/workspace/.cache/triton \
    ROS_LOG_DIR=/workspace/.runtime/ros-log \
    NVIDIA_IMEX_CHANNELS=0 \
    LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64 \
    PATH=/workspace/.venvs/lerobot/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    "$@"
' jetson-runtime "${runtime_root}" "${project_root}" "$@"
