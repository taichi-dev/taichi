# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
import os
import platform
import re
import sys

# -- third party --
# -- own --
from .misc import banner, error, info
from .tinysh import apt


# -- code --
UBUNTU_PACKAGES = {
    "git",
    "freeglut3-dev",
    "libglfw3-dev",
    "libglm-dev",
    "libglu1-mesa-dev",
    "libjpeg-dev",
    "liblz4-dev",
    "libpng-dev",
    "libssl-dev",
    "libtinfo-dev",
    "libwayland-dev",
    "libx11-xcb-dev",
    "libxcb-dri3-dev",
    "libxcb-ewmh-dev",
    "libxcb-keysyms1-dev",
    "libxcb-randr0-dev",
    "libxcursor-dev",
    "libxi-dev",
    "libxinerama-dev",
    "libxrandr-dev",
    "libzstd-dev",
}


RE_DPKG_STATUS = re.compile(r"Package: (.+?)\nStatus: install ok installed\n")
DISTRIB_ID = re.compile(r"DISTRIB_ID=(.+?)\n")


def get_installed_apt_pkgs() -> set:
    with open("/var/lib/dpkg/status") as f:
        return set(RE_DPKG_STATUS.findall(f.read()))


def install_ubuntu_pkgs():
    installed = get_installed_apt_pkgs()
    to_install = UBUNTU_PACKAGES - installed
    if not to_install:
        return

    if os.isatty(sys.stdin.fileno()):
        apt.install(*sorted(to_install))
    else:
        info("The necessary packages for the build process are missing and need to be installed:")
        p = lambda s: print(s, file=sys.stderr, flush=True)
        p("")
        for v in sorted(to_install):
            p(f"    {v}")
        p("")
        error("Please install them manually.")


@banner("Install Required OS Packages")
def setup_os_pkgs() -> None:
    u = platform.uname()
    if u.system == "Linux":
        lsb = Path("/etc/lsb-release")
        if lsb.exists():
            distro = DISTRIB_ID.findall(lsb.read_text())[-1]
            if distro == "Ubuntu":
                install_ubuntu_pkgs()
