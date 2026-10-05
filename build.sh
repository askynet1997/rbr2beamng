#!/bin/sh
root=$(dirname "$0")
exec uv run --project "$root" --locked --extra dev "$root/packaging/build_windows.py"
