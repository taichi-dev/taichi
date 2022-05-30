# Build script for windows

param (
    [switch]$clone = $false,
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

# Get sccache
$env:CCACHE_DIR="${pwd}/ccache_cache"
$env:CCACHE_MAXSIZE="128M"
$env:CCACHE_LOGFILE="${pwd}/ccache_error.log"
WriteInfo("ccache dir: $Env:CCACHE_DIR")
md "$Env:CCACHE_DIR" -ea 0
if (-not (Test-Path "ccache-4.5.1-windows-64")) {
    curl.exe --retry 10 --retry-delay 5 https://github.com/ccache/ccache/releases/download/v4.5.1/ccache-4.5.1-windows-64.zip -LO
    7z x ccache-4.5.1-windows-64.zip
    $env:PATH += ";${pwd}/ccache-4.5.1-windows-64"
}
ccache -v -s

# WriteInfo("Install 7Zip")
# Install-Module 7Zip4PowerShell -Force -Verbose -Scope CurrentUser

if ($clone) {
    WriteInfo("Clone the repository")
    git clone --recurse-submodules $RepoURL
    Set-Location .\taichi
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
$env:LLVM_DIR = "$libsDir\taichi_llvm"
$env:TAICHI_CMAKE_ARGS += " -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang"
if ($installVulkan) {
    WriteInfo("Download and install Vulkan")
    if (-not (Test-Path "VulkanSDK")) {
        curl.exe --retry 10 --retry-delay 5 https://sdk.lunarg.com/sdk/download/1.2.189.0/windows/VulkanSDK-1.2.189.0-Installer.exe -Lo VulkanSDK.exe
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE; }
        $installer = Start-Process -FilePath VulkanSDK.exe -Wait -PassThru -ArgumentList @("/S");
        $installer.WaitForExit();
    }
    $env:VULKAN_SDK = "$libsDir\VulkanSDK\1.2.189.0"
    $env:PATH += ";$env:VULKAN_SDK\Bin"
    $env:TAICHI_CMAKE_ARGS += " -DTI_WITH_VULKAN:BOOL=ON"
}

Pop-Location
clang --version

WriteInfo("Setting up Python environment")
python -m venv venv
. venv\Scripts\activate.ps1
python -m pip install wheel
python -m pip install -r requirements_dev.txt
if (-not $?) { exit 1 }
WriteInfo("Building Taichi")
$env:TAICHI_CMAKE_ARGS += " -DCLANG_EXECUTABLE=$libsDir\\taichi_clang\\bin\\clang++.exe"
$env:TAICHI_CMAKE_ARGS += " -DLLVM_AS_EXECUTABLE=$libsDir\\taichi_llvm\\bin\\llvm-as.exe"
if ($install) {
    if ($develop) {
        python setup.py develop
    } else {
        python setup.py install
    }
    if (-not $?) { exit 1 }
    WriteInfo("Build and install finished")
} else {
    if ($env:PROJECT_NAME -eq "taichi-nightly") {
        python setup.py egg_info --tag-date bdist_wheel
    } else {
        python setup.py bdist_wheel
    }
    if (-not $?) { exit 1 }
    WriteInfo("Build finished")
}
ccache -s -v
