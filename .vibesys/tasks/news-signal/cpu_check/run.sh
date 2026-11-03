#!/bin/sh
# Fast CPU check: replay the synthetic fixture through the candidate with a tiny random model.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../../.." && pwd)
export UV_PROJECT_ENVIRONMENT="${XDG_CACHE_HOME:-$HOME/.cache}/news-signal/venv"
exec uv run --frozen --project "$root" python "$here/check.py" "$@"
