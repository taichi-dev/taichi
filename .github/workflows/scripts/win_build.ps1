# Build script for windows

param (
    [switch]$clone = $false,
    [switch]$vulkan = $false,
    [switch]$develop = $false,
    [string]$build = "_build"
)

$RepoURL = 'https://github.com/taichi-dev/taichi'

function WriteInfo($text) {
    Write-Host -ForegroundColor Green "[BUILD] $text"
}

WriteInfo("Install 7Zip")
Install-Module 7Zip4PowerShell -Force -Verbose -Scope CurrentUser

if ($clone) {
    WriteInfo("Clone the repository")
    git clone --recurse-submodules $RepoURL
    Set-Location .\taichi
}
if (-not (Test-Path $build)) {
    New-Item -ItemType Directory -Path $build
}
Push-Location $build
WriteInfo("Download and extract LLVM")
if (-not (Test-Path "taichi_llvm")) {
    curl.exe --retry 10 --retry-delay 5 https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/taichi-llvm-10.0.0-msvc2019.zip -LO
    7z x taichi-llvm-10.0.0-msvc2019.zip -otaichi_llvm
}
WriteInfo("Download and extract Clang")
if (-not (Test-Path "taichi_clang")) {
    curl.exe --retry 10 --retry-delay 5 https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/clang-10.0.0-win.zip -LO
    7z x clang-10.0.0-win.zip -otaichi_clang
}
$env:PATH = "$build\taichi_llvm\bin;$build\taichi_clang\bin;$env:PATH"
$env:TAICHI_CMAKE_ARGS = "-G 'Visual Studio 16 2019' -A x64 -DLLVM_DIR=$build\taichi_llvm\lib\cmake\llvm"
if ($vulkan) {
    WriteInfo("Download and install Vulkan")
    if (-not (Test-Path "VulkanSDK.exe")) {
        curl.exe --retry 10 --retry-delay 5 https://sdk.lunarg.com/sdk/download/1.2.189.0/windows/VulkanSDK-1.2.189.0-Installer.exe -Lo VulkanSDK.exe
    }
    $installer = Start-Process -FilePath VulkanSDK.exe -Wait -PassThru -ArgumentList @("/S");
    $installer.WaitForExit();
    $env:VULKAN_SDK = "$build\VulkanSDK\1.2.189.0"
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
python -m pip install -r requirements_test.txt
WriteInfo("Building Taichi")
$env:CXX = "$build\taichi_clang\bin\clang++.exe"
if ($develop) {
    python -m pip install -e .
}
else {
    python -m pip install .
}
WriteInfo("Build finished")
