"""Enforce the dependency policy: every dependency is pinned to an exact version.

Ranges (^ ~ >= < *), bare names and `latest` are rejected. This is a supply
chain control -- an unpinned dependency lets a compromised or yanked release
enter the build without any change to this repository.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
LOCKFILE = ROOT / "uv.lock"
WORKFLOWS = ROOT / ".github" / "workflows"

# name==1.2.3, optionally with an environment marker.
EXACT = re.compile(r"^[A-Za-z0-9._-]+==[A-Za-z0-9._+!-]+(\s*;.*)?$")

# `uses: owner/repo@<40 hex> # vX.Y.Z` -- SHA pinned, with the tag in a comment
# so Renovate can still propose updates.
USES = re.compile(r"^\s*-?\s*uses:\s*(?P<ref>\S+)(?P<rest>.*)$")
SHA_PINNED = re.compile(r"^[^@]+@[0-9a-f]{40}$")
HAS_VERSION_COMMENT = re.compile(r"#\s*v?\d+(\.\d+)*")


def check_python_deps(errors: list[str]) -> None:
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)

    project = data.get("project", {})
    groups = {"dependencies": project.get("dependencies", [])}
    for name, deps in project.get("optional-dependencies", {}).items():
        groups[f"optional-dependencies.{name}"] = deps

    for group, deps in groups.items():
        for dep in deps:
            if not EXACT.match(dep.strip()):
                errors.append(f"{PYPROJECT.name} [{group}] not pinned to an exact version: {dep!r}")


def check_lockfile(errors: list[str]) -> None:
    if not LOCKFILE.exists():
        errors.append("uv.lock is missing. Run `uv sync` and commit the lockfile.")


def check_actions(errors: list[str]) -> None:
    if not WORKFLOWS.is_dir():
        return
    for wf in sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml")):
        for lineno, line in enumerate(wf.read_text(encoding="utf-8").splitlines(), 1):
            m = USES.match(line)
            if not m:
                continue
            ref = m.group("ref")
            if ref.startswith("./") or ref.startswith("docker://"):
                continue
            where = f"{wf.relative_to(ROOT)}:{lineno}"
            if not SHA_PINNED.match(ref):
                errors.append(f"{where} action is not pinned to a 40-char SHA: {ref}")
            elif not HAS_VERSION_COMMENT.search(m.group("rest")):
                errors.append(f"{where} SHA pin is missing its `# vX.Y.Z` comment: {ref}")


def main() -> int:
    errors: list[str] = []
    check_python_deps(errors)
    check_lockfile(errors)
    check_actions(errors)

    if errors:
        print("DEPENDENCY PIN CHECK FAILED\n")
        for err in errors:
            print(f"  {err}")
        return 1

    print("Dependency pin check passed (exact Python pins, SHA-pinned actions, lockfile present).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
