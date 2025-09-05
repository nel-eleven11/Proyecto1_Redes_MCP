#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../.venv/bin/activate"
export PYTHONPATH="$(dirname "$0")/../src"
python -m futboliq_mcp.server
