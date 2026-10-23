#!/bin/sh
# One-time local setup: Python environment + the commit hook that blocks secrets.
set -e
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
if [ ! -d .venv ]; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python "torch==2.2.2" "numpy<2" -r requirements.txt
fi
echo "Setup done. Run tests with: .venv/bin/python -m pytest -q"
