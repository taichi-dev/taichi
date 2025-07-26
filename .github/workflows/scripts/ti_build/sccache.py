# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform

from .cmake import cmake_args

# -- third party --
# -- own --
from .dep import download_dep
from .misc import banner, get_cache_home
from .tinysh import Command, sh


# -- code --
@banner("Setup sccache")
def setup_sccache() -> Command:
    """
    Download and install sccache, setup compiler wrappers, and return the `sccache` command.
    """
    root = get_cache_home() / "sccache-v0-10-0"
    bin = root / "bin"

    u = platform.uname()
    if u.system in ("Linux", "Darwin"):
        exe = bin / "sccache"
    elif u.system == "Windows":
        exe = bin / "sccache.exe"
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    if not exe.exists():
        if (u.system, u.machine) == ("Linux", "x86_64"):
            url = "https://github.com/mozilla/sccache/releases/download/v0.10.0/sccache-v0.10.0-x86_64-unknown-linux-musl.tar.gz"
        elif (u.system, u.machine) in (("Linux", "arm64"), ("Linux", "aarch64")):
            url = "https://github.com/mozilla/sccache/releases/download/v0.10.0/sccache-v0.10.0-aarch64-unknown-linux-musl.tar.gz"
        elif (u.system, u.machine) == ("Darwin", "arm64"):
            url = "https://github.com/mozilla/sccache/releases/download/v0.10.0/sccache-v0.10.0-aarch64-apple-darwin.tar.gz"
        elif (u.system, u.machine) == ("Darwin", "x86_64"):
            url = "https://github.com/mozilla/sccache/releases/download/v0.10.0/sccache-v0.10.0-x86_64-apple-darwin.tar.gz"
        elif u.system == "Windows":
            url = "https://github.com/mozilla/sccache/releases/download/v0.10.0/sccache-v0.10.0-x86_64-pc-windows-msvc.tar.gz"
        else:
            raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

        download_dep(url, bin, strip=1)
        exe.chmod(0o755)

    os.environ["SCCACHE_LOG"] = "error"
    exepath = str(exe).replace("\\", "\\\\")
    cmake_args["CMAKE_C_COMPILER_LAUNCHER"] = exepath
    cmake_args["CMAKE_CXX_COMPILER_LAUNCHER"] = exepath

    # <LocalCache>
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["SCCACHE_DIR"] = str(cache)
    os.environ["SCCACHE_CACHE_SIZE"] = "40G"
    # </LocalCache>

    return sh.bake(str(exe))
