# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform
from pathlib import Path

# -- third party --
# -- own --
from .cmake import cmake_args
from .misc import banner, path_prepend
from .python import path_prepend
from .tinysh import Command, sh


# -- code --
@banner("Setup Android NDK")
def setup_android_ndk() -> None:
    # TODO: Auto install
    s = platform.system()
    if s != "Linux":
        raise RuntimeError(f"Android NDK is only supported on Linux, but the current platform is {s}.")

    ndkroot = Path(os.environ.get("ANDROID_NDK_ROOT", "/android-sdk/ndk-bundle"))
    toolchain = ndkroot / "build/cmake/android.toolchain.cmake"
    if not toolchain.exists():
        raise RuntimeError(f"ANDROID_NDK_ROOT is set to {ndkroot}, but the path does not exist.")

    p = ndkroot.resolve()
    os.environ["ANDROID_NDK_ROOT"] = str(p)
    cmake_args["CMAKE_TOOLCHAIN_FILE"] = str(toolchain)
    cmake_args["ANDROID_NATIVE_API_LEVEL"] = "29"
    cmake_args["ANDROID_ABI"] = "arm64-v8a"
    path_prepend("PATH", p / "toolchains/llvm/prebuilt/linux-x86_64/bin")


@banner("Build Taichi Android C-API Shared Library")
def build_android(python: Command, pip: Command) -> None:
    """
    Build the Taichi Android C-API shared library
    """
    cmake_args["TI_WITH_BACKTRACE"] = False
    cmake_args["TI_WITH_LLVM"] = False
    cmake_args["TI_WITH_C_API"] = True
    cmake_args["TI_BUILD_TESTS"] = False
    cmake_args.writeback()
    os.environ["TAICHI_FORCE_PLAT_NAME"] = "android-arm64"  # affects setup.py
    pip.install("-r", "requirements_dev.txt")
    python("setup.py", "clean")
    python("setup.py", "build_ext")
    for p in Path(os.getcwd()).glob("**/libtaichi_c_api.so"):
        sh("aarch64-linux-android-strip", p)
