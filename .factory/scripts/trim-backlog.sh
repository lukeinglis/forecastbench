#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 '<backlog item text>'"
    echo "Trims backlog to a single item (saves backup first)."
    exit 1
fi

BACKLOG=".factory/strategy/backlog.md"
BACKUP=".factory/strategy/backlog-full.md"

if [ ! -f "$BACKLOG" ]; then
    echo "ERROR: $BACKLOG not found"
    exit 1
fi

cp "$BACKLOG" "$BACKUP"
echo "Saved backup to $BACKUP"

cat > "$BACKLOG" <<EOF
## Backlog

- $1
EOF

echo "Trimmed backlog to single item: $1"
echo ""
echo "To restore the full backlog:"
echo "  cp $BACKUP $BACKLOG"
