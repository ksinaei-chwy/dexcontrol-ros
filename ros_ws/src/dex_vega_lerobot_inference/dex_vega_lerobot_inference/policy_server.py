"""Python 3.12 LeRobot/CUDA server isolated from the ROS Humble interpreter."""

from __future__ import annotations

import argparse
import math
import os
import signal
import socket
import sys
from pathlib import Path

from .artifact import (
    resolve_model_artifact,
    resolve_tokenizer,
    require_project_local,
)
from .contracts import MODEL_REPO_ID
from .groot_artifact import resolve_groot_artifacts
from .groot_contracts import (
    BASE_MODEL_REVISION,
    CHECKPOINT_TAG,
    COSMOS_PROCESSOR_REVISION,
    MODEL_REVISION,
)
from .groot_policy_runtime import GrootPolicyRuntime
from .policy_rpc import PolicyRpcError, _receive_message, _send_message, serve_request
from .policy_runtime import Pi05PolicyRuntime


def build_parser() -> argparse.ArgumentParser:
    """Build the local-only policy server command-line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-type", choices=("pi05", "groot"), default="pi05")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer-dir", default="")
    parser.add_argument("--base-model-dir", default="")
    parser.add_argument("--cosmos-processor-dir", default="")
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--model-commit", default="")
    parser.add_argument("--tokenizer-commit", default="")
    parser.add_argument("--base-model-commit", default="")
    parser.add_argument("--cosmos-processor-commit", default="")
    parser.add_argument("--checkpoint-tag", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--request-io-timeout-seconds", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load local artifacts, serve inference, and never connect to ROS or hardware."""
    args = build_parser().parse_args(argv)
    if (
        not math.isfinite(args.request_io_timeout_seconds)
        or args.request_io_timeout_seconds <= 0.0
    ):
        raise ValueError("request I/O timeout must be finite and positive")
    root = Path(args.project_root).expanduser().resolve()
    model_path = require_project_local(args.model_dir, root)
    socket_path = require_project_local(args.socket_path, root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.policy_type == "groot":
        if not args.base_model_dir or not args.cosmos_processor_dir:
            raise ValueError(
                "GR00T requires --base-model-dir and --cosmos-processor-dir"
            )
        artifacts = resolve_groot_artifacts(
            project_root=root,
            model_local_path=model_path,
            base_model_local_path=args.base_model_dir,
            cosmos_processor_local_path=args.cosmos_processor_dir,
            model_revision=args.model_commit or MODEL_REVISION,
            base_model_revision=args.base_model_commit or BASE_MODEL_REVISION,
            cosmos_processor_revision=(
                args.cosmos_processor_commit or COSMOS_PROCESSOR_REVISION
            ),
            checkpoint_tag=args.checkpoint_tag or CHECKPOINT_TAG,
            allow_download=False,
            local_files_only=True,
        )
        runtime = GrootPolicyRuntime(
            project_root=root,
            artifacts=artifacts,
            device=args.device,
            require_cuda=True,
            require_bfloat16=True,
        )
    else:
        if not args.tokenizer_dir:
            raise ValueError("PI0.5 requires --tokenizer-dir")
        tokenizer_path = require_project_local(args.tokenizer_dir, root)
        model = resolve_model_artifact(
            project_root=root,
            local_path=model_path,
            repo_id=None,
            revision=args.model_commit or None,
            download_directory=None,
            checkpoint_tag=args.checkpoint_tag or None,
            allow_download=False,
            local_files_only=True,
        )
        if model.repo_id != MODEL_REPO_ID:
            raise ValueError(
                "PI0.5 model manifest does not identify the expected private Hub repo"
            )
        tokenizer = resolve_tokenizer(
            project_root=root,
            local_path=tokenizer_path,
            revision=args.tokenizer_commit or None,
            download_directory=None,
            allow_download=False,
            local_files_only=True,
        )
        runtime = Pi05PolicyRuntime(
            project_root=root,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            require_cuda=True,
            require_bfloat16=True,
        )
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_socket_path(socket_path)

    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    bound_socket_inode: int | None = None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            bound_socket_inode = socket_path.stat().st_ino
            socket_path.chmod(0o600)
            server.listen(1)
            server.settimeout(0.5)
            print(
                f"{runtime.info.policy_type} policy server ready at {socket_path}",
                flush=True,
            )
            while not stopping:
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(args.request_io_timeout_seconds)
                    try:
                        request, payload = _receive_message(connection)
                        response, response_payload = serve_request(runtime, request, payload)
                        response["payload_bytes"] = len(response_payload)
                    except Exception as exc:  # noqa: BLE001 - RPC/model boundary
                        response = {"status": "error", "error": str(exc), "payload_bytes": 0}
                        response_payload = b""
                    try:
                        _send_message(connection, response, response_payload)
                    except (OSError, PolicyRpcError) as exc:
                        print(
                            f"policy RPC response could not be delivered: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
    finally:
        try:
            if (
                bound_socket_inode is not None
                and socket_path.is_socket()
                and socket_path.stat().st_ino == bound_socket_inode
            ):
                socket_path.unlink()
        except FileNotFoundError:
            pass
    return 0


def _prepare_socket_path(socket_path: Path) -> None:
    """Fail closed rather than unlinking an existing or ambiguous endpoint."""
    if not socket_path.exists():
        return
    if not socket_path.is_socket():
        raise RuntimeError(f"refusing to replace non-socket path: {socket_path}")
    raise RuntimeError(
        f"policy socket already exists at {socket_path}; refuse to unlink it because "
        "another server may own it"
    )


if __name__ == "__main__":
    raise SystemExit(main())
