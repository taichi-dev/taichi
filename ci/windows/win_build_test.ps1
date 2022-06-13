# Build script for windows

param (
    [switch]$clone = $true,
    [switch]$installVulkan = $false,
    [switch]$develop = $false,
    [switch]$install = $false,
    [string]$libsDir = "."
)

$ErrorActionPreference = "Stop"

$RepoURL = 'https://github.com/taichi-dev/taichi'

function WriteInfo($text) {
    Write-Host -ForegroundColor Green "[BUILD] $text"
}

$libsDir = (Resolve-Path $libsDir).Path
if (-not (Test-Path $libsDir)) {
    New-Item -ItemType Directory -Path $libsDir
}
Push-Location $libsDir

if (-not (Test-Path "taichi_llvm")) {
    WriteInfo("Download and extract LLVM")
    curl.exe --retry 10 --retry-delay 5 https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/taichi-llvm-10.0.0-msvc2019.zip -LO
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE; }
    7z x taichi-llvm-10.0.0-msvc2019.zip -otaichi_llvm
}
if (-not (Test-Path "taichi_clang")) {
    WriteInfo("Download and extract Clang")
    curl.exe --retry 10 --retry-delay 5 https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/clang-10.0.0-win.zip -LO
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE; }
    7z x clang-10.0.0-win.zip -otaichi_clang
}

$env:LLVM_DIR = "C://taichi_llvm"
$env:TAICHI_CMAKE_ARGS =' -DCMAKE_CXX_COMPILER=C:/Program\ Files\ (x86)/Microsoft\ Visual\ Studio/2019/BuildTools/vc/Tools/Llvm/x64/bin/clang++.exe -DCMAKE_C_COMPILER=C:/Program\ Files\ (x86)/Microsoft\ Visual\ Studio/2019/BuildTools/vc/Tools/Llvm/x64/bin/clang.exe'
$env:TAICHI_CMAKE_ARGS += " -DCLANG_EXECUTABLE=C:\\taichi_clang\\bin\\clang++.exe"
$env:TAICHI_CMAKE_ARGS += " -DLLVM_AS_EXECUTABLE=C:\\taichi_llvm\\bin\\llvm-as.exe -DTI_WITH_VULKAN:BOOL=OFF"
$env:TAICHI_CMAKE_ARGS += " -DTI_WITH_VULKAN:BOOL=OFF -DTI_WITH_CUDA:BOOL=OFF -DTI_WITH_OPENGL:BOOL=OFF"

Pop-Location
clang --version

WriteInfo("Clone the repository")
git clone --recurse-submodules $RepoURL
Set-Location .\taichi

WriteInfo("Setting up Python environment")
conda activate py37
python -m pip install numpy
python -m pip install wheel
python -m pip install -r requirements_dev.txt
python -m pip install -r requirements_test.txt
if (-not $?) { exit 1 }

WriteInfo("Building Taichi")
python setup.py develop
WriteInfo("Build finished")

WriteInfo("Testing Taichi")
python tests/run_tests.py --cpp
python tests/run_tests.py -vr2 -t2 -k "not torch and not paddle" -a cpu
WriteInfo("Test finished")
