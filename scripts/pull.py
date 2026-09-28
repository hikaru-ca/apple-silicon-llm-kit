"""Download a model into the local Hugging Face cache."""

from __future__ import annotations

import sys

from huggingface_hub import snapshot_download
from resolve_model import load


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: pull.py <model-key|hf-id>", file=sys.stderr)
        return 1

    key = sys.argv[1]
    models = load()
    repo_id = models[key]["id"] if key in models else key

    print(f"Downloading {repo_id} ...")
    path = snapshot_download(repo_id=repo_id)
    print(f"Cached at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
