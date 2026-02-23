#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x ".venv/bin/streamlit" ]]; then
  STREAMLIT_CMD=".venv/bin/streamlit"
elif command -v streamlit >/dev/null 2>&1; then
  STREAMLIT_CMD="streamlit"
elif [[ -x ".venv/bin/python" ]]; then
  exec ".venv/bin/python" -m streamlit run app.py "$@"
else
  echo "Error: streamlit not found. Activate/install dependencies first (see README.md)." >&2
  exit 1
fi

exec "$STREAMLIT_CMD" run app.py "$@"
