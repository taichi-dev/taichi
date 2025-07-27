param (
    [string]$libsDir = "."
)

$ErrorActionPreference = "Stop"

. $PSScriptRoot\common-utils.ps1

Setup-VS

$env:PYTHONUNBUFFERED = 1
$env:TI_CI = 1
$env:TI_OFFLINE_CACHE_FILE_PATH = Join-Path -Path $pwd -ChildPath ".cache\taichi"

Invoke python .github/workflows/scripts/build.py --permissive --write-env=ti-env.ps1
. .\ti-env.ps1

Invoke python -m pip install -U pip wheel
Invoke python -m pip uninstall taichi taichi-nightly -y
# These have to be re-installed to avoid strange certificate issue
# on CPU docker environment
Invoke python -m pip install --upgrade --force-reinstall numpy cmake wheel

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
# Invoke pip install "paddlepaddle==2.3.0; python_version < '3.10'"

if ($env:EXTRA_TEST_MARKERS) {
    $EXTRA_TEST_MARKERS_SOLO = @("-m", $env:EXTRA_TEST_MARKERS)
    $EXTRA_TEST_MARKERS_AND = "and $env:EXTRA_TEST_MARKERS"
} else {
    $EXTRA_TEST_MARKERS_SOLO = @()
    $EXTRA_TEST_MARKERS_AND = ""
}

# Run C++ tests
#
Invoke python tests/run_tests.py --cpp -vr2 -t4 @EXTRA_TEST_MARKERS_SOLO

# Fail fast, give priority to the error-prone tests
# FIXME(proton): Disable paddle tests, out of sync for too long.
# Invoke python tests/run_tests.py -vr2 -t1 -k "paddle" -a cpu @EXTRA_TEST_MARKERS_SOLO

# Disable paddle for the remaining test
$env:TI_ENABLE_PADDLE = "0"

function RunIt($arch, $parallelism) {
    if ("$env:TI_WANTED_ARCHS".Contains("cuda")) {
        Invoke python tests/run_tests.py -vr2 -t"$parallelism" -k "not torch and not paddle" -m "not run_in_serial $EXTRA_TEST_MARKERS_AND" -a $arch
        Invoke python tests/run_tests.py -vr2 -t1 -k "not torch and not paddle" -m "run_in_serial $EXTRA_TEST_MARKERS_AND" -a $arch
    }
}

if ("$env:TI_WANTED_ARCHS".Contains("cpu")) {
  # NOTE: Always test CPU with non-CUDA version of PyTorch,
  #       since CUDA version of PyTorch will load a lot of CUDA libraries,
  #       which inflates required commited memory usage (not physical memory, but still relevant)
  #       to 5GiB per test process (compared to 1.4GiB for non-CUDA version).
  #       This greatly improves test paralllism.
  #       This is a non-issue on Linux, since Linux overcommits.
  Invoke pip install "torch==2.6.0" "torchvision==0.20.1" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cpu
  RunIt cpu (EstimateNumProcs)
}

if ("$env:TI_WANTED_ARCHS".Contains("cuda")) {
  Invoke pip install --force-reinstall "torch==2.6.0" "torchvision==0.20.1" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu121
  RunIt cuda 8
}

RunIt vulkan 8

Invoke python tests/run_tests.py -vr2 -t1 -k "torch" -a "$env:TI_WANTED_ARCHS" @EXTRA_TEST_MARKERS_SOLO

if ("$env:TI_RUN_RELEASE_TESTS" -eq "1") {
    Info "Running release tests"
    # release tests
    Invoke pip install PyYAML
    Invoke git clone https://github.com/taichi-dev/taichi-release-tests
    Push-Location taichi-release-tests
    Invoke git checkout 20230619
    mkdir -p repos/taichi/python/taichi
    $EXAMPLES = & python -c 'import taichi.examples as e; print(e.__path__._path[0])' | Select-Object -Last 1
    Push-Location repos
    Invoke git clone --depth=1 https://github.com/taichi-dev/quantaichi
    Invoke git clone --depth=1 https://github.com/taichi-dev/difftaichi
    Invoke git clone --depth=1 https://github.com/taichi-dev/games201
    Invoke git clone --depth=1 https://github.com/taichiCourse01/--Galaxy
    Invoke git clone --depth=1 https://github.com/taichiCourse01/--Shadertoys
    Invoke git clone --depth=1 https://github.com/taichiCourse01/taichi_ray_tracing
    Pop-Location
    Push-Location repos/difftaichi
    Invoke pip install -r requirements.txt
    Pop-Location
    Invoke python run.py --log=DEBUG --runners 1 timelines
    Pop-Location
}
