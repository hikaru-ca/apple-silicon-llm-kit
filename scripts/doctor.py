"""Report whether this machine can run the models in configs/models.toml."""

from __future__ import annotations

import platform
import subprocess
import sys
import tomllib
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "models.toml"


def sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value or None


def gpu_cores() -> str:
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    for line in out.stdout.splitlines():
        if "Total Number of Cores" in line:
            return line.split(":")[-1].strip()
    return "unknown"


def main() -> int:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print(f"FAIL  MLX requires Apple silicon. Found {platform.system()}/{platform.machine()}.")
        return 1

    chip = sysctl("machdep.cpu.brand_string") or "unknown"
    mem_raw = sysctl("hw.memsize")
    mem_gb = int(mem_raw) / 1024**3 if mem_raw and mem_raw.isdigit() else 0.0

    print("Machine")
    print(f"  chip        {chip}")
    print(f"  memory      {mem_gb:.0f} GB unified")
    print(f"  gpu cores   {gpu_cores()}")
    print(f"  python      {platform.python_version()}")

    # Default Metal wired limit is roughly 75% of physical memory.
    limit_raw = sysctl("iogpu.wired_limit_mb")
    if limit_raw and limit_raw.isdigit() and int(limit_raw) > 0:
        budget = int(limit_raw) / 1024
        print(f"  gpu budget  {budget:.0f} GB (explicit iogpu.wired_limit_mb)")
    else:
        budget = mem_gb * 0.75
        print(f"  gpu budget  ~{budget:.0f} GB (default, ~75% of physical)")

    try:
        import mlx.core as mx
    except ImportError:
        print("\nFAIL  mlx is not installed. Run `make setup`.")
        return 1

    # Touch the GPU to confirm Metal actually works, not just that the import resolved.
    probe = (mx.ones((256, 256)) @ mx.ones((256, 256))).sum()
    mx.eval(probe)
    print(f"  mlx         {mx.__version__} (Metal probe OK, result={probe.item():.0f})")

    print("\nModel fit  (weights only; KV cache grows on top with context length)")
    with CONFIG.open("rb") as fh:
        models = tomllib.load(fh)["models"]

    width = max(len(k) for k in models)
    for key, spec in models.items():
        need = float(spec["mem_gb"])
        # LoRA needs the base weights plus optimizer state and activations.
        train_need = need * 1.8 + 6
        infer = "ok  " if need < budget * 0.9 else "tight"
        train = "ok  " if train_need < budget * 0.9 else "tight"
        print(
            f"  {key:<{width}}  weights {need:>5.1f} GB  infer {infer}  "
            f"lora ~{train_need:>5.1f} GB {train}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
