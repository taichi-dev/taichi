# Build script for windows

param (
    [switch]$clone = $false,
    [switch]$installVulkan = $false,
    [switch]$develop = $false,
    [switch]$install = $false,
    [string]$libsDir = ".",
    [string]$llvmVer = "10"
)

$ErrorActionPreference = "Stop"

. $PSScriptRoot\common-utils.ps1

$libsDir = Resolve-Path-String-Force $libsDir
if (-not (Test-Path $libsDir)) {
    New-Item -ItemType Directory -Path $libsDir
}

$RepoURL = 'https://github.com/taichi-dev/taichi'

SetupCCacheLocal "$libsDir/ccache"

if ($clone) {
    Info("Clone the repository")
    Invoke git clone --recurse-submodules $RepoURL
    Set-Location .\taichi
}

Setup-VS

Push-Location $libsDir

function DownloadDep {
    param (
        [string]$name,
        [string]$outfile,
        [string]$dir,
        [string]$url
    )
    if (-not (Test-Path $dir)) {
        Info("Download and extract $name")
        Invoke-WebRequest `
            -Uri $url `
            -MaximumRetryCount 10 -RetryIntervalSec 5 `
            -OutFile $outfile
        Expand-Archive -Force $outfile $dir
    }
}

if ($llvmVer -eq "10") {
    DownloadDep LLVM llvm.zip taichi_llvm `
        https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/taichi-llvm-10.0.0-msvc2019.zip
    DownloadDep Clang clang.zip taichi_clang `
        https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/clang-10.0.0-win.zip
    $env:LLVM_DIR = "$libsDir\taichi_llvm"
	$env:TAICHI_CMAKE_ARGS += " -DCLANG_EXECUTABLE=$($libsDir -replace "\\", "\\")\\taichi_clang\\bin\\clang++.exe"
	$env:TAICHI_CMAKE_ARGS += " -DLLVM_AS_EXECUTABLE=$($libsDir -replace "\\", "\\")\\taichi_llvm\\bin\\llvm-as.exe"
} elseif ($llvmVer -eq "15") {
    DownloadDep LLVM-15 llvm-15.zip taichi_llvm_15 `
        https://github.com/python3kgae/taichi_assets/releases/download/llvm15_vs2019_clang/taichi-llvm-15.0.0-msvc2019.zip
    DownloadDep Clang-15 clang-15.zip taichi_clang_15 `
		https://github.com/python3kgae/taichi_assets/releases/download/llvm15_vs2022_clang/clang-15.0.0-win.zip
    $env:LLVM_DIR = "$libsDir\taichi_llvm_15"
	$env:TAICHI_CMAKE_ARGS += " -DCLANG_EXECUTABLE=$($libsDir -replace "\\", "\\")\\taichi_clang_15\\bin\\clang++.exe"
	$env:TAICHI_CMAKE_ARGS += " -DLLVM_AS_EXECUTABLE=$($libsDir -replace "\\", "\\")\\taichi_llvm_15\\bin\\llvm-as.exe"
} else {
    throw "Unsupported LLVM version"
}

$env:TAICHI_CMAKE_ARGS += " -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang"

if ($installVulkan) {
    $env:VULKAN_SDK = "C:\VulkanSDK\1.2.189.0"
    if (-not (Test-Path $env:VULKAN_SDK)) {
        Info("Download and install Vulkan")
        Invoke-WebRequest `
            -Uri 'https://sdk.lunarg.com/sdk/download/1.2.189.0/windows/VulkanSDK-1.2.189.0-Installer.exe' `
            -MaximumRetryCount 10 -RetryIntervalSec 5 `
            -OutFile VulkanSDK.exe
        $installer = Start-Process -FilePath VulkanSDK.exe -Wait -PassThru -ArgumentList @("/S")
        $installer.WaitForExit();
    }
    $env:PATH += ";$env:VULKAN_SDK\Bin"
    $env:TAICHI_CMAKE_ARGS += " -DTI_WITH_VULKAN:BOOL=ON"
}

Pop-Location
Invoke clang --version

Setup-Python $libsDir $env:PY

Invoke python -m pip install -r requirements_dev.txt

Info("Building Taichi")

if ($install) {
    if ($develop) {
        Invoke python setup.py develop
    } else {
        Invoke python setup.py install
    }
    Info("Build and install finished")
} else {
    if ($env:PROJECT_NAME -eq "taichi-nightly") {
        Invoke python setup.py egg_info --tag-date bdist_wheel
    } else {
        Invoke python setup.py bdist_wheel
    }
    Info("Build finished")
}

ccache -s -v
