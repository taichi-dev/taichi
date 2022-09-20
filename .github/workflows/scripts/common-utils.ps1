function Info($text, $prefix="BUILD") {
    Write-Host -ForegroundColor Green "[$prefix] $text"
}

function ProbeVirtualMemory {
    $MethodDefinition = @"
using System;
using System.Runtime.InteropServices;

public class VirtualProbe {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll", SetLastError=true, ExactSpelling=true)]
    public static extern IntPtr VirtualAllocEx(IntPtr hProcess, IntPtr lpAddress, IntPtr dwSize, uint flAllocationType, uint flProtect);
    [DllImport("kernel32.dll", SetLastError=true, ExactSpelling=true)]
    public static extern bool VirtualFreeEx(IntPtr hProcess, IntPtr lpAddress, IntPtr dwSize, uint dwFreeType);
}
"@
    if(-not ('VirtualProbe' -as [type])) {
        Add-Type -TypeDefinition $MethodDefinition -Language CSharp
    }
    $totalSz = 0
    $attempt = [int64](512 * 1024 * 1024 * 1024)
    $allocated = @()
    $me = [VirtualProbe]::GetCurrentProcess()
    while($attempt -gt 256 * 1024 * 1024) {
        $ptr = [VirtualProbe]::VirtualAllocEx($me, 0, $attempt, 0x3000, 0x01)  # MEM_COMMIT | MEM_RESERVE, PAGE_NOACCESS
        if($ptr -eq 0) {
            $attempt = [int64]($attempt / 2)
        } else {
            $allocated += $ptr
            $totalSz += $attempt
        }
    }

    foreach($ptr in $allocated) {
        $_ = [VirtualProbe]::VirtualFreeEx($me, $ptr, 0, 0x8000)  # MEM_RELEASE
    }

    return $totalSz
}

function EstimateNumProcs($expectedPhysicalMemoryUsage = 512, $expectedVirtualMemoryUsage = 2048) {
    # Things on Windows is complicated.
    # Windows never overcommits, so we can't just use physical memory to estimate
    # parallelism, pagefile (aka. swap on Linux but with subtle difference) must
    # also be considered. What's more irratating is that pagefile is not
    # a fixed size, it can grow on demand, Win32_OperatingSystem.FreeVirtualMemory
    # is therefore almost always smaller than what you can really use.
    # Aaaand, there are containers, where you can't even get the disk size or
    # pagefile status.
    # So we just probe the available virtual memory size.
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $phys = $os.TotalVisibleMemorySize / 1024
    $maxVirt = [int]($(ProbeVirtualMemory) / 1024 / 1024)
    $procsByVirtMem = [int]($maxVirt / $expectedVirtualMemoryUsage)
    $procsByPhysMem = [int](($phys - 1024) / $expectedPhysicalMemoryUsage)
    $procsByProcessors = [int]($env:NUMBER_OF_PROCESSORS)
    Info "Physical memory: $phys MB" "ESTIMATE"
    Info "Allocatable (virtual) memory: $maxVirt MB" "ESTIMATE"
    Info "Processors: $procsByProcessors" "ESTIMATE"
    Info "Parallelism by physical memory: $procsByPhysMem" "ESTIMATE"
    Info "Parallelism by allocatable memory: $procsByVirtMem" "ESTIMATE"
    $procs = [int]([Math]::Min($procsByPhysMem, [Math]::Min($procsByVirtMem, $procsByProcessors)))
    Info "Parallelism: $procs" "ESTIMATE"
    # TODO: GPU memory
    return $procs
}

function Setup-VS {
    Info "Setting up Visual Studio"
    foreach($progRoot in $env:ProgramFiles, ${env:ProgramFiles(x86)}) {
        $vsBase = Join-Path $progRoot 'Microsoft Visual Studio'
        foreach($ver in '2022','2019') {
            foreach($edition in 'Enterprise','Professional','Community','BuildTools') {
                $vsPath = Join-Path $vsBase $ver $edition
                $clangPath = Join-Path $vsPath "VC\Tools\Llvm\x64\bin\clang.exe"
                if (Test-Path $clangPath) {
                    $devShellModule = Get-Item $(Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
                    Import-Module $devShellModule
                    Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64"
                    return
                }
            }
        }
    }

    throw "Could not find Visual Studio with Clang"
}

function Setup-Python($libsDir, $version = "3.7") {
    if ([string]::IsNullOrEmpty($version)) {
        throw "Should specify Python version"
    }

    Info "Setting up Python environment $version"

    function PipOps {
        Invoke python -m pip install -U pip wheel
        Invoke python -m pip uninstall taichi taichi-nightly -y
        # These have to be re-installed to avoid strange certificate issue
        # on CPU docker environment
        Invoke python -m pip install --upgrade --force-reinstall numpy cmake wheel
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        $ver = & python --version
        Info "Found $ver" "Python"
        if ($ver.Startswith("Python ${version}.")) {
            Info "Using $ver" "Python"
            $venv = "$libsDir/taichi-venv-$version"
            if(-not (Test-Path $venv)) {
                Invoke python -m venv $venv
            }
            . "$libsDir/taichi-venv-$version/Scripts/activate.ps1"
            PipOps
            return
        }
    }

    if (Get-Command conda -ErrorAction SilentlyContinue) {
        Info "Using conda environment" "Python"
        # <Workaround> bad conda in container
        Invoke conda shell.powershell hook | Out-String | Invoke-Expression
        # </Workaround>
        $condaEnv = "$libsDir/taichi-conda-$version"
        if (-not (Test-Path $condaEnv)) {
            conda create -y -q --prefix=$condaEnv python=$version
        }
        conda activate $condaEnv
        PipOps
        return
    }

    throw "Could not setup Python"
}

function Resolve-Path-String-Force {
    <#
    .SYNOPSIS
        Calls Resolve-Path but works for files that don't exist.
    .REMARKS
        From http://devhawk.net/blog/2010/1/22/fixing-powershells-busted-resolve-path-cmdlet
    #>
    param (
        [string] $FileName
    )

    $FileName = Resolve-Path $FileName -ErrorAction SilentlyContinue `
                                       -ErrorVariable _frperror
    if (-not($FileName)) {
        $FileName = $_frperror[0].TargetObject
    }

    return $FileName
}

function Invoke() {
    # https://stackoverflow.com/questions/47032005/why-does-a-powershell-script-not-end-when-there-is-a-non-zero-exit-code-using-th
    # A handy way to run a command, and automatically throw an error if the
    # exit code is non-zero.
    #
    #
    if ($args.Count -eq 0) {
        throw "Must supply some arguments."
    }

    Write-Host -ForegroundColor Blue "[CMD] $args"

    $command = $args[0]
    $commandArgs = @()
    if ($args.Count -gt 1) {
        $commandArgs = $args[1..($args.Count - 1)]
    }

    & $command $commandArgs
    $ok = $?
    $result = $LASTEXITCODE

    if (-not $ok) {
        throw "$command $commandArgs failed."
    }

    if ($result -ne 0) {
        throw "$command $commandArgs exited with code $result."
    }
}


# SCCache not working on Windows (reports UnknownFlag with -Xclang)
# Using CCache here
function SetupCCacheLocal($root) {
    $root = Resolve-Path-String-Force $root

    New-Item -ItemType Directory -Path "$root\cache" -ErrorAction SilentlyContinue

    $env:CCACHE_DIR = "$root/cache"
    $env:CCACHE_TEMPDIR = "$env:TEMP/ccache-temp"
    $env:CCACHE_MAXSIZE= "10G"
    $env:CCACHE_LOG_FILE = "$root/ccache_error.log"

    if (-not (Test-Path -Path "$root/bin/ccache.exe")) {
        Info "Downloading sccache"
        Push-Location "$root"

        Invoke-WebRequest `
            -Uri "https://github.com/ccache/ccache/releases/download/v4.5.1/ccache-4.5.1-windows-64.zip" `
            -MaximumRetryCount 10 -RetryIntervalSec 300 `
            -OutFile ccache.zip
        Expand-Archive -Force ccache.zip .
        Rename-Item -Force -Path "ccache-4.5.1-windows-64" -NewName "bin"
        Pop-Location
    }

    $env:PATH += ";$root/bin"
    $env:TAICHI_CMAKE_ARGS += " -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"

    ccache -s -v
}

function ClearTaichiOfflineCache {
    # Remove-Item -Force -Path "$env:LocalAppData/build-cache/dot-cache/" -Recurse -ErrorAction SilentlyContinue
}

function PrepareBuildCache {
    "git-cache","pip-cache" | % {
        New-Item -ItemType Directory -Path "$env:LocalAppData/build-cache/$_" -ErrorAction SilentlyContinue
    }
    Push-Location $env:LocalAppData/build-cache/git-cache
    if (Test-Path -Path objects) {
        Invoke git init --bare
    }
    Pop-Location
}

function CIDockerRun {
    $containerName = $null
    $shouldRm = $true

    for($i = 0; $i -lt $args.Count; $i++) {
        $v = $args[$i]
        if($v -eq "-n" -or $v -eq "--name") {
            $containerName = $args[$i+1]
            $shouldRm = $false
            $i++
        }
    }

    if($containerName) {
        Invoke docker rm -f $containerName
    }

    if($shouldRm -and -not $args.Contains("--rm")) {
        $args = ,("--rm") + $args
    }

    $TiEnvs = @()
    Get-ChildItem "env:*" | % {
        if($_.Key.Startswith("TI_")) {
            $TiEnvs += "-e", $_.Key
        }
    }

    $extraArgs = ($env:CI_DOCKER_RUN_EXTRA_ARGS ?? "").Trim().Split()

    Invoke docker run `
        -i `
        -e PY `
        -e PROJECT_NAME `
        -e TAICHI_CMAKE_ARGS `
        -e IN_DOCKER=true `
        -e PIP_CACHE_DIR=X:/pip-cache `
        -e GIT_ALTERNATE_OBJECT_DIRECTORIES=X:/git-cache/objects `
        -e TI_CI=1 `
        @TiEnvs `
        -v (($env:LocalAppData -replace "\\", "/") + "/build-cache:X:") `
        @extraArgs `
        @args
}
