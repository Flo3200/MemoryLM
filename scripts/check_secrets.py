"""Block commits that contain access tokens or passwords.

The repository is public, so a leaked Hugging Face or GitHub token could be
abused within minutes. This script is run by the git pre-commit hook
(.githooks/pre-commit) on the staged changes and by GitHub Actions on every
push. It exits with code 1 if it finds anything that looks like a secret.

Usage:
  python scripts/check_secrets.py --staged      # what is about to be committed
  python scripts/check_secrets.py --all         # every tracked file
  python scripts/check_secrets.py FILE [FILE...]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Each pattern matches one well-known token format. They are written so that
# they do not match themselves (the character classes need real token text).
PATTERNS = {
    "Hugging Face token": r"hf_[A-Za-z0-9]{30,}",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "GitHub fine-grained token": r"github_pat_[A-Za-z0-9_]{40,}",
    "OpenAI/Anthropic-style key": r"sk-(?:ant-)?[A-Za-z0-9_\-]{30,}",
    "Google API key": r"AIza[0-9A-Za-z_\-]{35}",
    "AWS access key": r"AKIA[0-9A-Z]{16}",
    "Private key block": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "Hard-coded password": r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"\s]{6,}['\"]",
    "Hard-coded token assignment": r"(?i)(?:hf_token|api_key|access_token|secret)\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,}['\"]",
}
COMPILED = {name: re.compile(p) for name, p in PATTERNS.items()}

# Files that may legitimately be large/binary or contain example patterns.
SKIP_SUFFIXES = {".bin", ".pt", ".safetensors", ".png", ".jpg", ".npy", ".gz", ".zip"}


def scan_text(text: str) -> list[tuple[int, str]]:
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for name, rx in COMPILED.items():
            if rx.search(line):
                hits.append((lineno, name))
    return hits


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def staged_files() -> list[tuple[str, str]]:
    names = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR").split()
    out = []
    for n in names:
        if Path(n).suffix in SKIP_SUFFIXES:
            continue
        # Read the STAGED version, not the working copy: that is what gets committed.
        out.append((n, _git("show", f":{n}")))
    return out


def tracked_files() -> list[tuple[str, str]]:
    out = []
    for n in _git("ls-files").split():
        p = Path(n)
        if p.suffix in SKIP_SUFFIXES or not p.is_file():
            continue
        out.append((n, p.read_text(errors="ignore")))
    return out


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "--staged":
        files = staged_files()
    elif argv[0] == "--all":
        files = tracked_files()
    else:
        files = [(a, Path(a).read_text(errors="ignore")) for a in argv]
    found = 0
    for name, text in files:
        for lineno, kind in scan_text(text):
            print(f"BLOCKED: {name}:{lineno}: looks like a {kind}")
            found += 1
    if found:
        print("\nCommit blocked. Remove the secret (use Colab's secrets manager or an env variable).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
