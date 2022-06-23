$ErrorActionPreference = "Stop"

$env:PYTHONUNBUFFERED = 1

. venv\Scripts\activate.ps1
python -c "import taichi"
ti diagnose
