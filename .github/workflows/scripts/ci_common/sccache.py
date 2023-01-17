# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform

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
    root = get_cache_home() / "sccache"
    bin = root / "bin"

    u = platform.uname()
    if u.system in ("Linux", "Darwin"):
        exe = bin / "sccache"
    elif u.system == "Windows":
        exe = bin / "sccache.exe"
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    if not exe.exists():
        if u.system == "Linux":
            download_dep(
                "https://github.com/mozilla/sccache/releases/download/v0.3.1/sccache-v0.3.1-x86_64-unknown-linux-musl.tar.gz",
                bin,
                strip=1,
            )
        elif (u.system, u.machine) == ("Darwin", "arm64"):
            download_dep(
                "https://github.com/mozilla/sccache/releases/download/v0.3.1/sccache-v0.3.1-aarch64-apple-darwin.tar.gz",
                bin,
                strip=1,
            )
        elif (u.system, u.machine) == ("Darwin", "x86_64"):
            download_dep(
                "https://github.com/mozilla/sccache/releases/download/v0.3.1/sccache-v0.3.1-x86_64-apple-darwin.tar.gz",
                bin,
                strip=1,
            )
        elif u.system == "Windows":
            download_dep(
                "https://github.com/mozilla/sccache/releases/download/v0.3.1/sccache-v0.3.1-x86_64-pc-windows-msvc.tar.gz",
                bin,
                strip=1,
            )
        else:
            raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

        exe.chmod(0o755)

    os.environ["SCCACHE_LOG"] = "error"
    os.environ["TAICHI_CMAKE_ARGS"] += (
        f" -DCMAKE_C_COMPILER_LAUNCHER={exe}"
        f" -DCMAKE_CXX_COMPILER_LAUNCHER={exe}")

    # <LocalCache>
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["SCCACHE_DIR"] = str(cache)
    os.environ["SCCACHE_CACHE_SIZE"] = "40G"
    # </LocalCache>

    return sh.bake(str(exe))
