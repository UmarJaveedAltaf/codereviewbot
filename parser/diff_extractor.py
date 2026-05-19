from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class HunkRange:
    start: int
    length: int


@dataclass
class ChangedFile:
    filename: str
    language: str                      # "python" | "javascript" | "unknown"
    patch: str                         # raw unified diff text
    added_lines: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)
    hunks: list[HunkRange] = field(default_factory=list)


_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "javascript",
    ".jsx": "javascript",
    ".tsx": "javascript",
}

_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def extract_changed_files(files: list[dict]) -> list[ChangedFile]:
    """Convert PyGithub file objects (dicts) into ChangedFile dataclasses.

    Args:
        files: List of file dicts from PyGithub PullRequest.get_files().
                Expected keys: filename, patch (may be absent for binary files).

    Returns:
        List of ChangedFile with parsed hunks and line buckets.
    """
    results: list[ChangedFile] = []
    for f in files:
        patch = f.get("patch", "") or ""
        if not patch:
            continue

        ext = "." + f["filename"].rsplit(".", 1)[-1] if "." in f["filename"] else ""
        lang = _LANG_MAP.get(ext, "unknown")

        added: list[str] = []
        removed: list[str] = []
        hunks: list[HunkRange] = []

        for line in patch.splitlines():
            m = _HUNK_HEADER.match(line)
            if m:
                start = int(m.group(1))
                length = int(m.group(2)) if m.group(2) is not None else 1
                hunks.append(HunkRange(start=start, length=length))
            elif line.startswith("+") and not line.startswith("+++"):
                content = line[1:]
                if content:  # skip blank added lines — no code content for analysis
                    added.append(content)
            elif line.startswith("-") and not line.startswith("---"):
                content = line[1:]
                if content:  # skip blank removed lines for the same reason
                    removed.append(content)

        results.append(
            ChangedFile(
                filename=f["filename"],
                language=lang,
                patch=patch,
                added_lines=added,
                removed_lines=removed,
                hunks=hunks,
            )
        )
    return results
