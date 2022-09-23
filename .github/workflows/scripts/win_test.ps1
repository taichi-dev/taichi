param (
    [string]$libsDir = "."
)

$ErrorActionPreference = "Stop"

. $PSScriptRoot\common-utils.ps1

Setup-VS

$env:PYTHONUNBUFFERED = 1
$env:TI_CI = 1
$env:TI_OFFLINE_CACHE_FILE_PATH = Join-Path -Path $pwd -ChildPath ".cache\taichi"

Setup-Python $libsDir $env:PY

$os = Get-CimInstance -Class Win32_OperatingSystem
Info "Total system memory: $($os.TotalVisibleMemorySize / 1024 / 1024) GB"

$whl = & Get-ChildItem -Filter '*.whl' -Path dist | Select-Object -First 1
echo $whl
Invoke python -m pip install $whl.FullName
Invoke python -c "import taichi"
Invoke ti diagnose
# Invoke ti changelog
echo wanted arch: $env:TI_WANTED_ARCHS
Invoke pip install -r requirements_test.txt
Invoke pip install "paddlepaddle==2.3.0; python_version < '3.10'"

# Run C++ tests
Invoke python tests/run_tests.py --cpp

# Fail fast, give priority to the error-prone tests
Invoke python tests/run_tests.py -vr2 -t1 -k "paddle" -a cpu

# Disable paddle for the remaining test
$env:TI_ENABLE_PADDLE = "0"

if ("$env:TI_WANTED_ARCHS".Contains("cpu")) {
  # NOTE: Always test CPU with non-CUDA version of PyTorch,
  #       since CUDA version of PyTorch will load a lot of CUDA libraries,
  #       which inflates required commited memory usage (not physical memory, but still relevant)
  #       to 5GiB per test process (compared to 1.4GiB for non-CUDA version).
  #       This greatly improves test paralllism.
  #       This is a non-issue on Linux, since Linux overcommits.
  # TODO relax this when torch supports 3.10
  Invoke pip install "torch==1.12.1; python_version < '3.10'"
  Invoke python tests/run_tests.py -vr2 "-t$(EstimateNumProcs)" -k "not torch and not paddle" -a cpu
}
if ("$env:TI_WANTED_ARCHS".Contains("cuda")) {
  # TODO relax this when torch supports 3.10
  Invoke pip install "torch==1.10.1+cu113; python_version < '3.10'" -f https://download.pytorch.org/whl/cu113/torch_stable.html
  Invoke python tests/run_tests.py -vr2 -t4 -k "not torch and not paddle" -a cuda
}
if ("$env:TI_WANTED_ARCHS".Contains("opengl")) {
  Invoke python tests/run_tests.py -vr2 -t4 -k "not torch and not paddle" -a opengl
}
if ("$env:TI_WANTED_ARCHS".Contains("vulkan")) {
  Invoke python tests/run_tests.py -vr2 -t4 -k "not torch and not paddle" -a vulkan
}
Invoke python tests/run_tests.py -vr2 -t1 -k "torch" -a "$env:TI_WANTED_ARCHS"

if ("$env:TI_RUN_RELEASE_TESTS" -eq "1" -and -not "$env:TI_LITE_TEST") {
    echo "Running release tests"
    # release tests
    Invoke pip install PyYAML
    Invoke git clone https://github.com/taichi-dev/taichi-release-tests
    mkdir -p repos/taichi/python/taichi
    $EXAMPLES = & python -c 'import taichi.examples as e; print(e.__path__._path[0])' | Select-Object -Last 1
    New-Item -Target $EXAMPLES -Path repos/taichi/python/taichi/examples -ItemType Junction
    New-Item -Target taichi-release-tests/truths -Path truths -ItemType Junction
    Invoke python taichi-release-tests/run.py --log=DEBUG --runners 1 taichi-release-tests/timelines
}
