#!/usr/bin/env sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
cd "$SCRIPT_DIR" || exit 1

if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "Python 3 was not found. Install Python or create the .venv environment first." >&2
    exit 1
fi

"$PYTHON" -m rbr2beamng.gui
status=$?
if [ "$status" -ne 0 ]; then
    echo >&2
    echo "RBR2BeamNG failed to start. Review the error above." >&2
    echo "For missing-package errors, install uv and run: uv sync --locked" >&2
fi
exit "$status"
