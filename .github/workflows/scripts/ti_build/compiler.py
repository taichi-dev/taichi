# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
import os
import platform
import shutil

# -- third party --
# -- own --
from .cmake import cmake_args
from .dep import download_dep
from .misc import banner, get_cache_home, warn


# -- code --
@banner("Setup Clang")
def setup_clang(as_compiler=True) -> None:
    """
    Setup Clang.
    """
    u = platform.uname()
    if u.system in ("Linux", "Darwin"):
        for v in ("", "-15", "-14", "-13", "-12", "-11", "-10"):
            clang = shutil.which(f"clang{v}")
            if clang is not None:
                clangpp = shutil.which(f"clang++{v}")
                assert clangpp
                break
        else:
            warn("Cannot find clang, compiling with system default compiler (or $CC/$CXX if set).")
            return

    elif (u.system, u.machine) == ("Windows", "AMD64"):
        out = get_cache_home() / "clang-15-v2"
        url = "https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/clang-15.0.0-win-complete.zip"
        download_dep(url, out, force=True)
        clang = str(out / "bin" / "clang++.exe").replace("\\", "\\\\")
        clangpp = clang
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    cmake_args["CLANG_EXECUTABLE"] = clang

    if as_compiler:
        if os.environ.get("CC"):
            warn(
                f"Explicitly specified compiler via environment variable CC={os.environ['CC']}, not configuring clang."
            )
        else:
            cmake_args["CMAKE_C_COMPILER"] = clang

        if os.environ.get("CXX"):
            warn(
                f"Explicitly specified compiler via environment variable CXX={os.environ['CXX']}, not configuring clang++."
            )
        else:
            cmake_args["CMAKE_CXX_COMPILER"] = clangpp


@banner("Setup MSVC")
def setup_msvc() -> None:
    assert platform.system() == "Windows"
    os.environ["TAICHI_USE_MSBUILD"] = "1"

    base = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools")
    for edition in ("Enterprise", "Professional", "Community", "BuildTools"):
        if (base / edition).exists():
            return

    url = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
    out = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools")
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
