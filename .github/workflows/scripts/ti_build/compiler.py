# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
import os
import json
import platform
import shutil
import tempfile
import sys

# -- third party --
# -- own --
from .cmake import cmake_args
from .dep import download_dep
from .misc import banner, error, get_cache_home, warn
from .tinysh import powershell


# -- code --
@banner("Setup Clang")
def setup_clang(as_compiler=True) -> None:
    """
    Setup Clang.
    """
    u = platform.uname()
    if u.system in ("Linux", "Darwin"):
        for v in ("", "-14", "-13", "-12", "-11", "-10"):
            clang = shutil.which(f"clang{v}")
            if clang is not None:
                clangpp = shutil.which(f"clang++{v}")
                assert clangpp
                break
        else:
            error("Could not find clang of any version")
            return

    elif (u.system, u.machine) == ("Windows", "AMD64"):
        out = get_cache_home() / "clang-14-v2"
        url = "https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/clang-14.0.6-win-complete.zip"
        download_dep(url, out, force=True)
        clang = str(out / "bin" / "clang++.exe").replace("\\", "\\\\")
        clangpp = clang
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    cmake_args["CLANG_EXECUTABLE"] = clang

    if as_compiler:
        cc = os.environ.get("CC")
        cxx = os.environ.get("CXX")
        if cc:
            warn(f"Explicitly specified compiler via environment variable CC={cc}, not configuring clang.")
        else:
            cmake_args["CMAKE_C_COMPILER"] = clang

        if cxx:
            warn(f"Explicitly specified compiler via environment variable CXX={cxx}, not configuring clang++.")
        else:
            cmake_args["CMAKE_CXX_COMPILER"] = clangpp


ENV_EXTRACT_SCRIPT = """
param ([string]$DevShell, [string]$VsPath, [string]$OutFile)
$WarningPreference = 'SilentlyContinue'
Import-Module $DevShell
Enter-VsDevShell -VsInstallPath $VsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64"
Get-ChildItem env:* | ConvertTo-Json -Depth 1 | Out-File $OutFile
"""


def _vs_devshell(vs):
    dll = vs / "Common7" / "Tools" / "Microsoft.VisualStudio.DevShell.dll"

    if not dll.exists():
        error("Could not find Visual Studio DevShell")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        script = tmp / "extract.ps1"
        with script.open("w") as f:
            f.write(ENV_EXTRACT_SCRIPT)
        outfile = tmp / "env.json"
        powershell(
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-DevShell",
            str(dll),
            "-VsPath",
            str(vs),
            "-OutFile",
            str(outfile),
        )
        with outfile.open(encoding="utf-16") as f:
            envs = json.load(f)

    for v in envs:
        os.environ[v["Key"]] = v["Value"]


@banner("Setup MSVC")
def setup_msvc() -> None:
    assert platform.system() == "Windows"

    base = Path("C:\\Program Files (x86)\\Microsoft Visual Studio")
    for ver in ("2022",):
        for edition in ("Enterprise", "Professional", "Community", "BuildTools"):
            vs = base / ver / edition
            if not vs.exists():
                continue

            if os.environ.get("TI_CI") and not os.environ.get("TAICHI_USE_MSBUILD"):
                # Use Ninja + MSVC in CI, for better caching
                _vs_devshell(vs)
                cmake_args["CMAKE_C_COMPILER"] = "cl.exe"
                cmake_args["CMAKE_CXX_COMPILER"] = "cl.exe"
            else:
                os.environ["TAICHI_USE_MSBUILD"] = "1"

            return
    else:
        url = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
        out = base / "2022" / "BuildTools"
        download_dep(
            url,
            out,
            elevate=True,
            args=[
                "--passive",
                "--wait",
                "--norestart",
                "--includeRecommended",
                "--add",
                "Microsoft.VisualStudio.Workload.VCTools",
                # NOTE: We are using the custom built Clang++,
                #       so components below are not necessary anymore.
                # '--add',
                # 'Microsoft.VisualStudio.Component.VC.Llvm.Clang',
                # '--add',
                # 'Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Llvm.Clang',
                # '--add',
                # 'Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset',
            ],
        )
        warn("Please restart build.py after Visual Studio Build Tools is installed.")
        sys.exit(1)
