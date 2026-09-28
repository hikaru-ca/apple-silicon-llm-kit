"""Scan git-tracked files for information that should not be in a public repo.

Two layers, deliberately separated:

  1. Generic patterns, defined below and committed. They describe *shapes* of
     private data (credentials, addresses, internal hostnames) and name no
     specific person, employer or system.

  2. An optional private denylist at `.leakcheck-local.txt`, which is
     gitignored. Concrete internal names belong there and only there -- a
     committed denylist of secret words is itself a disclosure, so this file
     must never be tracked.

Layer 1 runs in CI. Layer 2 runs only on the machine that has the file.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_DENYLIST = ROOT / ".leakcheck-local.txt"

# Files whose content is allowed to mention these shapes.
SKIP_FILES = {"scripts/check_leak.py", "uv.lock", ".gitignore"}
SKIP_SUFFIXES = {".jsonl", ".safetensors", ".npz", ".bin", ".gguf", ".png", ".jpg", ".pdf"}

# Matches that are fine in a public repo.
ALLOWED = re.compile(
    r"""
      users\.noreply\.github\.com
    | \bexample\.(com|org|net)\b
    | \bnoreply@
    | \blocalhost\b
    | 127\.0\.0\.1
    | \b0\.0\.0\.0\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

GENERIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("corporate domain", re.compile(r"\b[\w-]+\.(co\.jp|ne\.jp|or\.jp)\b", re.IGNORECASE)),
    ("internal hostname", re.compile(r"\b[\w-]+\.(internal|corp|intra|lan)\b", re.IGNORECASE)),
    ("private IPv4", re.compile(r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d+\.\d+\b")),
    ("absolute home path", re.compile(r"/Users/[A-Za-z0-9._-]+/")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GCP service account", re.compile(r'"type"\s*:\s*"service_account"')),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer token", re.compile(r"\b(?:ghp|gho|github_pat|sk-|xoxb-|xoxp-)[A-Za-z0-9_-]{16,}")),
    ("slack/meet link", re.compile(r"https://[\w-]+\.slack\.com/\S+", re.IGNORECASE)),
]


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    paths = [ROOT / p for p in out.stdout.split("\0") if p]
    return [
        p
        for p in paths
        if p.is_file()
        and p.suffix not in SKIP_SUFFIXES
        and str(p.relative_to(ROOT)) not in SKIP_FILES
    ]


def load_local_patterns() -> list[tuple[str, re.Pattern[str]]]:
    if not LOCAL_DENYLIST.exists():
        return []
    patterns = []
    for raw in LOCAL_DENYLIST.read_text(encoding="utf-8").splitlines():
        term = raw.strip()
        if not term or term.startswith("#"):
            continue
        patterns.append(("private denylist", re.compile(re.escape(term), re.IGNORECASE)))
    return patterns


def redact(text: str) -> str:
    """Show enough to locate the hit without reprinting the secret in full."""
    if len(text) <= 6:
        return text[0] + "*" * (len(text) - 1)
    return f"{text[:3]}{'*' * (len(text) - 6)}{text[-3:]}"


def main() -> int:
    local = load_local_patterns()
    patterns = GENERIC_PATTERNS + local
    findings: list[str] = []

    for path in tracked_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(ROOT)
        for lineno, line in enumerate(lines, 1):
            for label, pattern in patterns:
                for match in pattern.finditer(line):
                    hit = match.group(0)
                    if ALLOWED.search(hit):
                        continue
                    findings.append(f"  {rel}:{lineno}  [{label}] {redact(hit)}")

    scope = f"{len(GENERIC_PATTERNS)} generic"
    scope += f" + {len(local)} private" if local else " (no .leakcheck-local.txt on this machine)"

    if findings:
        print(f"LEAK CHECK FAILED  ({scope} patterns)\n")
        print("\n".join(findings))
        print(f"\n{len(findings)} finding(s). Remove them before pushing.")
        return 1

    print(f"Leak check passed. {scope} patterns, {len(tracked_files())} tracked files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
