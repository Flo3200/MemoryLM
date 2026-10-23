#!/bin/bash
# MemoryLM app for Mac: double-click this file. The first start installs
# everything (a few minutes); later starts take seconds. Then open
# http://localhost:7860 (the browser opens automatically).
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First start: installing (a few minutes)..."
  if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi
  uv venv --python 3.12 .venv
  if [ "$(uname -m)" = "x86_64" ]; then TORCH="torch==2.2.2"; else TORCH="torch"; fi
  uv pip install --python .venv/bin/python "$TORCH" "numpy<2" -r requirements.txt -r app/requirements.txt
fi
.venv/bin/python -c "import gradio" 2>/dev/null || uv pip install --python .venv/bin/python -r app/requirements.txt
.venv/bin/python app/app.py
