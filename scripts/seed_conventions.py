#!/usr/bin/env python3
"""Seed team conventions from conventions.yaml into the ChromaDB vector store.

Usage
-----
Seed all conventions (global scope):
    python scripts/seed_conventions.py

Scope to a single repo (overrides the per-rule repo field):
    python scripts/seed_conventions.py --repo myorg/myrepo

Preview without writing anything:
    python scripts/seed_conventions.py --dry-run

Clear the collection and re-seed from scratch:
    python scripts/seed_conventions.py --reset

The script is idempotent when run without --reset: conventions are upserted by
content hash so re-running will not create duplicates.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import textwrap
from pathlib import Path

# ── Make project root importable when run directly ────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is not installed.  Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from memory.vector_store import ConventionMetadata, add_convention

CONVENTIONS_FILE = _ROOT / "conventions.yaml"

# ── Helpers ───────────────────────────────────────────────────────────────────


def _convention_id(rule_text: str, language: str, category: str) -> str:
    """Deterministic UUID-shaped ID derived from the convention's content.

    Using a content hash means re-running the seeder upserts rather than
    duplicating conventions that haven't changed.
    """
    raw = f"{category}:{language}:{rule_text.strip()}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    # Format as UUID4-like string for readability
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _truncate(text: str, width: int = 72) -> str:
    return textwrap.shorten(text.strip(), width=width, placeholder="...")


# ── Core logic ─────────────────────────────────────────────────────────────────


def load_conventions(path: Path = CONVENTIONS_FILE) -> list[dict]:
    """Parse and return the list of conventions from the YAML file."""
    if not path.exists():
        print(f"ERROR: conventions file not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    conventions = data.get("conventions", [])
    if not conventions:
        print("WARNING: conventions.yaml contains no conventions.", file=sys.stderr)
    return conventions


def seed(
    conventions: list[dict],
    repo_override: str = "",
    dry_run: bool = False,
) -> int:
    """Insert conventions into the vector store.

    Args:
        conventions:   Parsed list of convention dicts from the YAML file.
        repo_override: If non-empty, all conventions are scoped to this repo.
        dry_run:       When True, print what would be seeded but write nothing.

    Returns:
        Number of conventions seeded (0 for dry-run).
    """
    seeded = 0

    for item in conventions:
        rule_text: str = item.get("rule", "").strip()
        if not rule_text:
            print("  SKIP: convention has no 'rule' field")
            continue

        language: str = item.get("language", "")
        category: str = item.get("category", "general")
        added_by: str = item.get("added_by", "system")
        repo: str = repo_override or item.get("repo", "")

        meta = ConventionMetadata(
            repo=repo,
            language=language,
            category=category,
            added_by=added_by,
        )

        lang_label = language if language else "all languages"
        repo_label = repo if repo else "global"

        if dry_run:
            print(
                f"  [DRY RUN] [{category}] ({lang_label}) ({repo_label})\n"
                f"            {_truncate(rule_text)}"
            )
            continue

        convention_id = add_convention(rule_text, meta)
        seeded += 1
        print(
            f"  + {convention_id[:8]}...  [{category}] ({lang_label})\n"
            f"    {_truncate(rule_text)}"
        )

    return seeded


def reset_conventions() -> None:
    """Delete and re-create the team_conventions ChromaDB collection."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from config import settings as app_settings
    from memory import vector_store as vs

    client = chromadb.PersistentClient(
        path=app_settings.CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection("team_conventions")
        print("  Deleted existing team_conventions collection.")
    except Exception:
        pass

    # Reset cached collection reference so it is re-created on next access.
    vs._conventions_col = None
    print("  Collection reset complete.")


# ── CLI entry point ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed team coding conventions into CodeReviewBot's vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo",
        default="",
        metavar="OWNER/NAME",
        help="Scope all conventions to this repo (overrides per-rule repo field).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be seeded without writing to the store.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing team_conventions collection before seeding.",
    )
    parser.add_argument(
        "--file",
        default=str(CONVENTIONS_FILE),
        metavar="PATH",
        help=f"Path to conventions YAML file (default: {CONVENTIONS_FILE}).",
    )
    args = parser.parse_args()

    print(f"CodeReviewBot — convention seeder")
    print(f"  Source : {args.file}")
    print(f"  Repo   : {args.repo or '(global)'}")
    print(f"  Mode   : {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    conventions = load_conventions(Path(args.file))
    print(f"Loaded {len(conventions)} convention(s) from YAML.\n")

    if args.reset and not args.dry_run:
        print("Resetting collection…")
        reset_conventions()
        print()

    print("Seeding…")
    count = seed(conventions, repo_override=args.repo, dry_run=args.dry_run)

    print()
    if args.dry_run:
        print(f"Dry run complete — {len(conventions)} convention(s) would be seeded.")
    else:
        print(f"Done. {count} convention(s) seeded into team_conventions.")


if __name__ == "__main__":
    main()
