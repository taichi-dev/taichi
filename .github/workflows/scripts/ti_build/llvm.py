# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform

# -- third party --
# -- own --
from .bootstrap import get_cache_home
from .cmake import cmake_args
from .dep import download_dep
from .misc import banner, get_cache_home, is_manylinux2014


# -- code --
@banner("Setup LLVM")
def setup_llvm() -> None:
    """
    Download and install LLVM.
    """
    u = platform.uname()
    if u.system == "Linux":
        if cmake_args.get_effective("TI_WITH_AMDGPU"):
            out = get_cache_home() / "llvm15-amdgpu-005"
            url = "https://github.com/GaleSeLee/assets/releases/download/v0.0.5/taichi-llvm-15.0.0-linux.zip"
        elif is_manylinux2014():
            # FIXME: prebuilt llvm15 on ubuntu didn't work on manylinux2014 image of centos. Once that's fixed, remove this hack.
            out = get_cache_home() / "llvm15-manylinux2014"
            url = "https://github.com/ailzhang/torchhub_example/releases/download/0.3/taichi-llvm-15-linux.zip"
        else:
            out = get_cache_home() / "llvm15"
            url = "https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-linux.zip"
        download_dep(url, out, strip=1)
    elif (u.system, u.machine) == ("Darwin", "arm64"):
        out = get_cache_home() / "llvm15-m1-nozstd"
        url = "https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-m1-nozstd.zip"
        download_dep(url, out, strip=1)
    elif (u.system, u.machine) == ("Darwin", "x86_64"):
        out = get_cache_home() / "llvm15-mac"
        url = "https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/llvm-15-mac10.15.zip"
        download_dep(url, out, strip=1)
    elif (u.system, u.machine) == ("Windows", "AMD64"):
        out = get_cache_home() / "llvm15"
        url = "https://github.com/python3kgae/taichi_assets/releases/download/llvm15_vs2019_clang/taichi-llvm-15.0.0-msvc2019.zip"
        download_dep(url, out, strip=0)
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    # We should use LLVM toolchains shipped with OS.
    # path_prepend('PATH', out / 'bin')
    os.environ["LLVM_DIR"] = str(out)
