"""Check if local main branch is behind origin/main."""

import subprocess
import sys


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def main() -> int:
    run(["git", "fetch", "origin", "main", "--quiet"])

    result = run(
        ["git", "rev-list", "--count", "main..origin/main"],
        check=False,
    )
    if result.returncode != 0:
        print(f"ERROR: could not compare branches: {result.stderr.strip()}")
        return 1

    behind = int(result.stdout.strip())

    if behind == 0:
        print("OK: local main is up to date with origin/main")
        return 0

    print(f"WARNING: local main is {behind} commit(s) behind origin/main")
    print()

    log_result = run(
        [
            "git", "log", "--oneline", "--merges",
            "main..origin/main",
            "--format=%h %s",
        ],
        check=False,
    )
    if log_result.returncode == 0 and log_result.stdout.strip():
        print("Missing PR merges:")
        for line in log_result.stdout.strip().splitlines():
            print(f"  {line}")
    else:
        log_all = run(
            ["git", "log", "--oneline", "main..origin/main"],
            check=False,
        )
        if log_all.returncode == 0 and log_all.stdout.strip():
            print("Missing commits:")
            for line in log_all.stdout.strip().splitlines():
                print(f"  {line}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
