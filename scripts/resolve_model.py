"""Resolve a short model key from configs/models.toml into a Hugging Face id."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "models.toml"


def load() -> dict[str, dict]:
    with CONFIG.open("rb") as fh:
        return tomllib.load(fh)["models"]


def main() -> int:
    models = load()
    args = sys.argv[1:]

    if not args or args[0] in {"--list", "-l"}:
        width = max(len(k) for k in models)
        for key, spec in models.items():
            print(
                f"    {key:<{width}}  {spec['mem_gb']:>5.1f} GB  ~{spec['tok_s']:<8} {spec['id']}"
            )
        return 0

    key = args[0]
    if key in models:
        print(models[key]["id"])
        return 0

    # Allow passing a full HF id or a local path straight through.
    if "/" in key or Path(key).exists():
        print(key)
        return 0

    print(f"error: unknown model key {key!r}. Known keys: {', '.join(models)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
