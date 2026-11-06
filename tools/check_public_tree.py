"""Fail when a release tree contains common private or generated artifacts."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip"}
BANNED_NAMES = {".env", "api.txt", "credentials", "secrets", "id_rsa"}
BANNED_PARTS = {"__pycache__", ".pytest_cache", ".venv", "venv", "milestone"}
SKIP_DIRS = {".git"}
TEXT_PATTERNS = {
    "Windows absolute path": re.compile(r"[A-Za-z]:\\"),
    "Unix home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "OpenAI-style secret": re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    "Bearer credential": re.compile(r"Bearer\s+[A-Za-z0-9._-]{16,}", re.I),
}


def main() -> int:
    findings = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file()):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        lowered_parts = {part.lower() for part in rel.parts}
        if path.name.lower() in BANNED_NAMES:
            findings.append(f"banned file: {rel}")
        if any(any(token in part for token in BANNED_PARTS) for part in lowered_parts):
            findings.append(f"generated/private path: {rel}")
        if path.stat().st_size > MAX_FILE_BYTES:
            findings.append(f"large file: {rel} ({path.stat().st_size} bytes)")
        if path.suffix.lower() in SKIP_SUFFIXES or path == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"unreviewed binary file: {rel}")
            continue
        for label, pattern in TEXT_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {rel}")
    if findings:
        print("Public-tree check failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("Public-tree check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
