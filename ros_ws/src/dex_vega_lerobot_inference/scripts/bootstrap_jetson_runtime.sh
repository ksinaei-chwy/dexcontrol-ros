#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <ros_ws_project_root>" >&2
  exit 2
fi

project_root="$(realpath "$1")"
runtime_dir="${project_root}/.runtime"
tool_dir="${runtime_dir}/tools/crane-v0.21.7"
image_dir="${runtime_dir}/images"
rootfs_dir="${runtime_dir}/rootfs/nvidia-pytorch-25.11-arm64"
archive="${image_dir}/nvidia-pytorch-25.11-arm64-rootfs.tar"

crane_version="0.21.7"
crane_archive="go-containerregistry_Linux_arm64.tar.gz"
crane_sha256="b6ee979d9411dfb05ce35ab9e156fe5de7def11a230764a7856ffa2eb971fa88"
crane_url="https://github.com/google/go-containerregistry/releases/download/v${crane_version}/${crane_archive}"
nvidia_image="nvcr.io/nvidia/pytorch"
nvidia_tag="25.11-py3"
nvidia_arm64_digest="sha256:4a85d8cf6fb3a943280960b8948cf4e9b6eca77b4414c68c9b2c7bb863f79b70"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "this runtime is pinned for Jetson aarch64" >&2
  exit 2
fi
if [[ ! -f "${project_root}/src/dex_vega_lerobot_inference/package.xml" ]]; then
  echo "not a Dexmate ROS workspace: ${project_root}" >&2
  exit 2
fi

mkdir -p "${tool_dir}" "${image_dir}" "${runtime_dir}/downloads"
if [[ ! -x "${tool_dir}/crane" ]]; then
  download_path="${runtime_dir}/downloads/${crane_archive}"
  curl --fail --location --retry 3 --output "${download_path}" "${crane_url}"
  echo "${crane_sha256}  ${download_path}" | sha256sum --check --strict
  tar --no-same-owner -xzf "${download_path}" -C "${tool_dir}" crane
  chmod 0755 "${tool_dir}/crane"
fi

resolved_digest="$(${tool_dir}/crane digest --platform linux/arm64 "${nvidia_image}:${nvidia_tag}")"
if [[ "${resolved_digest}" != "${nvidia_arm64_digest}" ]]; then
  echo "NVIDIA tag digest changed: expected ${nvidia_arm64_digest}, got ${resolved_digest}" >&2
  exit 2
fi

if [[ ! -f "${archive}" ]]; then
  partial_archive="${archive}.partial"
  if [[ -e "${partial_archive}" ]]; then
    echo "refusing to overwrite partial archive: ${partial_archive}" >&2
    exit 2
  fi
  "${tool_dir}/crane" export --platform linux/arm64 \
    "${nvidia_image}@${nvidia_arm64_digest}" "${partial_archive}"
  mv "${partial_archive}" "${archive}"
fi

if [[ ! -x "${rootfs_dir}/usr/bin/python3.12" ]]; then
  if [[ -e "${rootfs_dir}" ]]; then
    echo "refusing to overwrite incomplete rootfs: ${rootfs_dir}" >&2
    exit 2
  fi
  partial_rootfs="${rootfs_dir}.partial"
  mkdir -p "${partial_rootfs}"
  tar --no-same-owner -xf "${archive}" -C "${partial_rootfs}"
  mv "${partial_rootfs}" "${rootfs_dir}"
fi

printf '%s\n' \
  "runtime ready" \
  "  image: ${nvidia_image}:${nvidia_tag}" \
  "  arm64 digest: ${nvidia_arm64_digest}" \
  "  rootfs: ${rootfs_dir}"
