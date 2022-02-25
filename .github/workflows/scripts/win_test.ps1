$ErrorActionPreference = "Stop"

. venv\Scripts\activate.ps1
python -c "import taichi"
ti diagnose
ti changelog
echo wanted arch: $env:TI_WANTED_ARCHS
pip install -r requirements_test.txt
# TODO relax this when torch supports 3.10
if ("$env:TI_WANTED_ARCHS".Contains("cuda")) {
    pip install "torch==1.10.1+cu113; python_version < '3.10'" -f https://download.pytorch.org/whl/cu113/torch_stable.html
} else {
    pip install torch
}
}
if ("$env:TI_WANTED_ARCHS".Contains("cuda")) {
  python tests/run_tests.py -vr2 -t4 -k "not torch" -a cuda
  if (-not $?) { exit 1 }
}
if ("$env:TI_WANTED_ARCHS".Contains("cpu")) {
  python tests/run_tests.py -vr2 -t6 -k "not torch" -a cpu
  if (-not $?) { exit 1 }
}
if ("$env:TI_WANTED_ARCHS".Contains("opengl")) {
  python tests/run_tests.py -vr2 -t4 -k "not torch" -a opengl
  if (-not $?) { exit 1 }
}
python tests/run_tests.py -vr2 -t2 -k "torch" -a "$env:TI_WANTED_ARCHS"
if (-not $?) { exit 1 }
