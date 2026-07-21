"""Non-actuating platform and Python runtime validation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_runtime_report(
    run_bfloat16_probe: bool = True,
    run_groot_import_probe: bool = False,
) -> dict[str, object]:
    report: dict[str, object] = {
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "lerobot": _version("lerobot"),
        "torch": _version("torch"),
        "torchvision": _version("torchvision"),
        "transformers": _version("transformers"),
        "huggingface_hub": _version("huggingface_hub"),
        "accelerate": _version("accelerate"),
        "diffusers": _version("diffusers"),
        "dm_tree": _version("dm-tree"),
        "peft": _version("peft"),
        "timm": _version("timm"),
    }
    try:
        import torch

        report.update(
            {
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
            }
        )
        if torch.cuda.is_available():
            report["cuda_device_name"] = torch.cuda.get_device_name(0)
            report["bfloat16_reported_supported"] = torch.cuda.is_bf16_supported()
            if run_bfloat16_probe:
                left = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
                right = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
                value = torch.matmul(left, right)
                torch.cuda.synchronize()
                report["bfloat16_matmul_ok"] = bool(
                    value.dtype == torch.bfloat16 and value.isfinite().all().item()
                )
                del left, right, value
        else:
            report["bfloat16_matmul_ok"] = False
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        report["torch_probe_error"] = repr(exc)
    report["ready"] = bool(
        report.get("architecture") == "aarch64"
        and report.get("lerobot") == "0.6.0"
        and report.get("cuda_available") is True
        and report.get("torch_cuda_version")
        and report.get("bfloat16_matmul_ok") is True
    )
    report["groot_dependencies_ready"] = all(
        report.get(name)
        for name in ("accelerate", "diffusers", "dm_tree", "peft", "timm")
    )
    if run_groot_import_probe and report["groot_dependencies_ready"]:
        try:
            from lerobot.policies.groot.configuration_groot import GrootConfig
            from lerobot.policies.groot.modeling_groot import GrootPolicy
            from lerobot.policies.groot.processor_groot import GrootN17VLMEncodeStep

            report["groot_import_probe"] = [
                GrootConfig.__name__,
                GrootPolicy.__name__,
                GrootN17VLMEncodeStep.__name__,
            ]
            report["groot_import_ready"] = True
        except Exception as exc:  # noqa: BLE001 - diagnostic import boundary
            report["groot_import_ready"] = False
            report["groot_import_error"] = repr(exc)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-groot",
        action="store_true",
        help="Also require every LeRobot GR00T N1.7 inference dependency.",
    )
    args = parser.parse_args(argv)
    report = collect_runtime_report(run_groot_import_probe=args.require_groot)
    print(json.dumps(report, indent=2, sort_keys=True))
    ready = bool(report["ready"])
    if args.require_groot:
        ready = (
            ready
            and bool(report["groot_dependencies_ready"])
            and bool(report.get("groot_import_ready"))
        )
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
