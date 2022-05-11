param(
    # Debug, Release, RelWithDebInfo, MinSizeRel
    [string] $BuildType = "Release",
    [string] $LlvmDir = "",
    [string] $ClangDir = "",
    # Install python package in user-space.
    [switch] $UserSpace = $false,
    # Clean up compilation intermediates instead of building Taichi. Note that
    # downloaded artifacts (like LLVM and Clang) will not be removed.
    [switch] $Clean = $false
)

$ErrorActionPreference = "Stop"

if ($Clean) {
    & python setup.py clean
    exit
}

$TempDir = "${pwd}/tmp"
$DownloadDir = "${TempDir}/download"

function EnsureDir($Dir) {
    if (-not (Test-Path $Dir)) {
        New-Item $Dir -ItemType Directory
    }
}
function DownloadFile($Uri, $DstFileName) {
    EnsureDir $TempDir
    EnsureDir $DownloadDir
    # Download only if the file is in absence.
    $DstPath = "$DownloadDir/$DstFileName"
    if (-not (Test-Path $DstPath)) {
        Invoke-WebRequest -MaximumRetryCount 10 -RetryIntervalSec 5 $Uri -OutFile $DstPath
    }
}
function DownloadArchiveAndExpand($Uri, $ArchiveName) {
    DownloadFile $Uri "$ArchiveName.zip";
    # Expand archive only if we haven't done it before.
    $ExpandDir = "$TempDir/$ArchiveName";
    if (-not (Test-Path $ExpandDir)) {
        Expand-Archive "$DownloadDir/$ArchiveName.zip" -DestinationPath $ExpandDir
    }
}



# Select build type, by default it's `Release`.
switch ($BuildType) {
    "Debug" { $env:DEBUG = 1; }
    "Release" {}
    "RelWithDebInfo" { $env:RELWITHDEBINFO = 1; }
    "MinSizeRel" { $env:MINSIZEREL = 1; }
    Default {
        Write-Error "Unknown build type '$BuildType'"
    }
}

# Prepare LLVM.
if ($env:LLVM_DIR) {
    # Compatible with previous building process, where `LLVM_DIR` and
    # `LLVM_AS_EXECUTABLE` are set externally.
    $LlvmDir = $env:LLVM_DIR;
}
if (-not $LlvmDir) {
    DownloadArchiveAndExpand -Uri "https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/taichi-llvm-10.0.0-msvc2019.zip" -ArchiveName "taichi-llvm"
    $LlvmDir = "$TempDir/taichi-llvm"
}
if (-not $LlvmDir -or -not (Test-Path $LlvmDir)) {
    throw "LLVM cannot be found in local environment and the script failed to download a prebuilt archive. " +
        "Please follow the instructions at 'https://docs.taichi-lang.org/lang/articles/dev_install' to manually configure LLVM for Taichi."
} else {
    $LlvmDir = (Resolve-Path $LlvmDir).Path;
    $env:LLVM_DIR = $LlvmDir
    Write-Host "Using LLVM at '$LlvmDir'."
}

#Prepare Clang.
if (-not $ClangDir) {
    DownloadArchiveAndExpand -Uri "https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/clang-10.0.0-win.zip" -ArchiveName "taichi-clang"
    $ClangDir = "$TempDir/taichi-clang"
}
if (-not $ClangDir -or -not (Test-Path $ClangDir)) {
    throw "Clang cannot be found in local environment and the script failed to download a prebuilt archive. " +
        "Please follow the instructions at 'https://docs.taichi-lang.org/lang/articles/dev_install' to manually configure Clang for Taichi."
} else {
    $ClangDir = (Resolve-Path $ClangDir).Path;
    Write-Host "Using Clang at '$ClangDir'."
}

$CMakeArgs = @{
    "CLANG_EXECUTABLE" = "$ClangDir/bin/clang++.exe";
    "LLVM_AS_EXECUTABLE" = "$LlvmDir/bin/llvm-as.exe";
}

# Build Vulkan backend if Vulkan SDK is installed.
if ($env:VK_SDK_PATH) {
    Write-Host "Found existing Vulkan SDK isntalltion at '$env:VK_SDK_PATH', Vulkan backend will be built."
    $env:VULKAN_SDK = $env:VK_SDK_PATH;
    $CMakeArgs["TI_WITH_VULKAN:BOOL"] = "ON";
}

# Chain up the cmake arguments.
Write-Host "Will build Taichi ($BuildType) with the following CMake args:"
$env:TAICHI_CMAKE_ARGS = ""
foreach ($Pair in $CMakeArgs.GetEnumerator()) {
    $Key = $Pair | Select-Object -ExpandProperty Key
    $Value = ($Pair | Select-Object -ExpandProperty Value) -replace "\\", "/"
    Write-Host "  $Key = $Value"
    $env:TAICHI_CMAKE_ARGS += " -D$Key=`"$Value`""
}

# Install in userspace?
$BuildExpr = "python setup.py develop";
if ($UserSpace) {
    Write-Host "Taichi Python package will be installed in user-space."
    $BuildExpr += " --user"
}

Write-Host

# Do the job.
$stopwatch = [system.diagnostics.stopwatch]::startNew()
Write-Host $BuildExpr
Invoke-Expression $BuildExpr
$stopwatch.Elapsed
