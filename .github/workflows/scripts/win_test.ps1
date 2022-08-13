$ErrorActionPreference = "Stop"

$env:PYTHONUNBUFFERED = 1
$env:TI_CI = 1
$env:TI_OFFLINE_CACHE_FILE_PATH = Join-Path -Path $pwd -ChildPath ".cache\taichi"

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
    pip install "torch; python_version < '3.10'"
    pip install "paddlepaddle==2.3.0; python_version < '3.10'"
}


if ("$env:TI_RUN_RELEASE_TESTS" -eq "1" -and -not "$env:TI_LITE_TEST") {
    echo "Running release tests"
    # release tests
    pip install PyYAML
    git clone https://github.com/taichi-dev/taichi-release-tests
    mkdir -p repos/taichi/python/taichi
    $EXAMPLES = & python -c 'import taichi.examples as e; print(e.__path__._path[0])' | Select-Object -Last 1
    New-Item -Target $EXAMPLES -Path repos/taichi/python/taichi/examples -ItemType Junction
    New-Item -Target taichi-release-tests/truths -Path truths -ItemType Junction
    python taichi-release-tests/run.py --log=DEBUG --runners 1 taichi-release-tests/timelines
    if (-not $?) { exit 1 }
}

# Run C++ tests
python tests/run_tests.py --cpp
if (-not $?) { exit 1 }

# Fail fast, give priority to the error-prone tests
python tests/run_tests.py -vr2 -t1 -k "paddle" -a cpu
if (-not $?) { exit 1 }

# Disable paddle for the remaining test
$env:TI_ENABLE_PADDLE = "0"

if ("$env:TI_WANTED_ARCHS".Contains("cuda")) {
  python tests/run_tests.py -vr2 -t4 -k "not torch and not paddle" -a cuda
  if (-not $?) { exit 1 }
}
if ("$env:TI_WANTED_ARCHS".Contains("cpu")) {
  python tests/run_tests.py -vr2 -t6 -k "not torch and not paddle" -a cpu
  if (-not $?) { exit 1 }
}
if ("$env:TI_WANTED_ARCHS".Contains("opengl")) {
  python tests/run_tests.py -vr2 -t4 -k "not torch and not paddle" -a opengl
  if (-not $?) { exit 1 }
}
python tests/run_tests.py -vr2 -t1 -k "torch" -a "$env:TI_WANTED_ARCHS"
if (-not $?) { exit 1 }
