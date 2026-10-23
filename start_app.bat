@echo off
REM MemoryLM app for Windows: double-click this file. The first start installs
REM everything (a few minutes). Then open http://localhost:7860
cd /d "%~dp0"
if not exist .venv (
  echo First start: installing...
  py -3.12 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv\Scripts\python -m pip install -r requirements.txt -r app\requirements.txt
)
.venv\Scripts\python app\app.py
pause
