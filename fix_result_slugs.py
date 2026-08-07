"""One-time script to fix model_slug='unknown' in existing result files."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results/ directory found.")
        return

    fixed = 0
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("model_slug") == "unknown":
            data["model_slug"] = "sonnet-4-20250514.parity"
            p.write_text(json.dumps(data, indent=2))
            print(f"Fixed: {p.name}")
            fixed += 1

    print(f"\nDone. Fixed {fixed} file(s).")


if __name__ == "__main__":
    main()
