"""Download the exact GR00T policy and gated offline dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifact import ArtifactValidationError
from .groot_artifact import resolve_groot_artifacts
from .groot_contracts import (
    BASE_MODEL_REVISION,
    CHECKPOINT_TAG,
    COSMOS_PROCESSOR_REVISION,
    MODEL_REVISION,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify the immutable Dexmate GR00T N1.7 bundle. "
            "HF_TOKEN is read only from the environment."
        )
    )
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument(
        "--model-dir",
        default=f"data/models/groot-n17-dexmate-blue-bird/{MODEL_REVISION}",
    )
    parser.add_argument(
        "--base-model-dir",
        default=f"data/models/groot-n1.7-3b/{BASE_MODEL_REVISION}",
    )
    parser.add_argument(
        "--cosmos-processor-dir",
        default=f"data/models/cosmos-reason2-2b/{COSMOS_PROCESSOR_REVISION}",
    )
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--base-model-revision", default=BASE_MODEL_REVISION)
    parser.add_argument(
        "--cosmos-processor-revision", default=COSMOS_PROCESSOR_REVISION
    )
    parser.add_argument("--checkpoint-tag", default=CHECKPOINT_TAG)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    try:
        bundle = resolve_groot_artifacts(
            project_root=root,
            model_local_path=root / args.model_dir,
            base_model_local_path=root / args.base_model_dir,
            cosmos_processor_local_path=root / args.cosmos_processor_dir,
            model_revision=args.model_revision,
            base_model_revision=args.base_model_revision,
            cosmos_processor_revision=args.cosmos_processor_revision,
            checkpoint_tag=args.checkpoint_tag,
            allow_download=True,
            local_files_only=False,
        )
    except ArtifactValidationError as exc:
        print(f"GR00T artifact validation failed: {exc}", file=sys.stderr)
        return 2
    summary = {
        "policy": {
            "path": str(bundle.model.local_path),
            "commit": bundle.model.resolved_commit,
            "checkpoint_tag": bundle.model.checkpoint_tag,
        },
        "base_model": {
            "path": str(bundle.base_model.local_path),
            "commit": bundle.base_model.resolved_commit,
        },
        "cosmos_processor": {
            "path": str(bundle.cosmos_processor.local_path),
            "commit": bundle.cosmos_processor.resolved_commit,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
