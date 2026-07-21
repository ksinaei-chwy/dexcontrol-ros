#!/usr/bin/env python3
"""Upload an already-finalized local LeRobot dataset."""

from __future__ import annotations

import argparse

from .dataset_writer import upload_existing_dataset


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-directory", required=True)
    parser.add_argument("--repo-id", required=True)
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", action="store_true", default=True)
    visibility.add_argument("--public", action="store_false", dest="private")
    options = parser.parse_args(args)
    upload_existing_dataset(
        options.local_directory,
        options.repo_id,
        private=options.private,
    )
    print(f"uploaded {options.local_directory} to {options.repo_id}")


if __name__ == "__main__":
    main()
