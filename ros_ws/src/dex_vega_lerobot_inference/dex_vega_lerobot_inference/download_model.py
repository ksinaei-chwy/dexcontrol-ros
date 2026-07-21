"""Download immutable private policy/tokenizer snapshots into the repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact import (
    file_inventory,
    resolve_model_artifact,
    resolve_tokenizer,
)
from .contracts import MODEL_REPO_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve a private PI0.5 revision and PaliGemma tokenizer revision to "
            "project-local directories. Authentication is read from HF_TOKEN/HF_HOME."
        )
    )
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--model-repo-id", default=MODEL_REPO_ID)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--checkpoint-tag", default="")
    parser.add_argument("--model-dir", default="data/models/pi05-dexmate-blue-bird")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--tokenizer-dir", default="data/models/paligemma-3b-pt-224")
    parser.add_argument(
        "--allow-tag",
        action="store_true",
        help="Allow a tag as input; the downloader records and uses its resolved commit SHA.",
    )
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help="Compute SHA-256 for every downloaded file (the 9.35 GB model may take time).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    model_dir = root / args.model_dir
    tokenizer_dir = root / args.tokenizer_dir
    model = resolve_model_artifact(
        project_root=root,
        local_path=None,
        repo_id=args.model_repo_id,
        revision=args.model_revision,
        download_directory=model_dir,
        checkpoint_tag=args.checkpoint_tag or None,
        allow_download=True,
        allow_non_commit_revision=args.allow_tag,
    )
    tokenizer = resolve_tokenizer(
        project_root=root,
        local_path=None,
        revision=args.tokenizer_revision,
        download_directory=tokenizer_dir,
        allow_download=True,
        allow_non_commit_revision=args.allow_tag,
    )
    summary = {
        "model_path": str(model.local_path),
        "model_repo_id": model.repo_id,
        "model_requested_revision": model.requested_revision,
        "model_resolved_commit": model.resolved_commit,
        "checkpoint_tag": model.checkpoint_tag,
        "tokenizer_path": str(tokenizer.local_path),
        "tokenizer_resolved_commit": tokenizer.resolved_commit,
        "model_files": file_inventory(model.local_path, hash_files=args.hash_files),
        "tokenizer_files": file_inventory(tokenizer.local_path, hash_files=args.hash_files),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
