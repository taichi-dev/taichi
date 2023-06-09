# -*- coding: utf-8 -*-

# -- stdlib --
import argparse
import datetime
import os
import platform
import subprocess
import sys

# -- third party --
# -- own --
from . import misc
from .alter import handle_alternate_actions
from .android import build_android, setup_android_ndk
from .cmake import cmake_args
from .compiler import setup_clang, setup_msvc
from .ios import build_ios, setup_ios
from .llvm import setup_llvm
from .misc import banner, is_manylinux2014
from .ospkg import setup_os_pkgs
from .python import get_desired_python_version, setup_python
from .sccache import setup_sccache
from .tinysh import Command, CommandFailed, git, nice
from .vulkan import setup_vulkan


# -- code --
@banner("Build Taichi Wheel")
def build_wheel(python: Command, pip: Command) -> None:
    """
    Build the Taichi wheel
    """

    git.fetch("origin", "master", "--tags", "--force")
    proj_tags = []
    extra = []

    cmake_args.writeback()
    if misc.options.tag_local:
        wheel_tag = f"+{misc.options.tag_local}"
    elif misc.options.tag_config:
        wheel_tag = f"+{cmake_args.render_wheel_tag()}"
    else:
        wheel_tag = ""

    if misc.options.nightly:
        os.environ["PROJECT_NAME"] = "taichi-nightly"
        now = datetime.datetime.now().strftime("%Y%m%d")
        proj_tags.extend(["egg_info", f"--tag-build=.post{now}{wheel_tag}"])
    elif wheel_tag:
        proj_tags.extend(["egg_info", f"--tag-build={wheel_tag}"])

    if platform.system() == "Linux":
        if is_manylinux2014():
            extra.extend(["-p", "manylinux2014_x86_64"])
        else:
            extra.extend(["-p", "manylinux_2_27_x86_64"])

    python("setup.py", "clean")
    python("misc/make_changelog.py", "--ver", "origin/master", "--repo_dir", "./", "--save")

    with nice():
        python("setup.py", *proj_tags, "bdist_wheel", *extra)


@banner("Install Build Wheel Dependencies")
def install_build_wheel_deps(python: Command, pip: Command) -> None:
    pip.install("-U", "pip")
    pip.install("-r", "requirements_dev.txt")


def setup_basic_build_env():
    u = platform.uname()
    if (u.system, u.machine) == ("Windows", "AMD64"):
        # Use MSVC on Windows
        setup_clang(as_compiler=False)
        setup_msvc()
    else:
        # Use Clang on all other platforms
        setup_clang()

    setup_llvm()
    setup_vulkan()

    sccache = setup_sccache()

    # NOTE: We use conda/venv to build wheels, which may not be the same python
    #       running this script.
    python, pip = setup_python(get_desired_python_version())

    return sccache, python, pip


def action_wheel():
    setup_os_pkgs()
    sccache, python, pip = setup_basic_build_env()
    install_build_wheel_deps(python, pip)
    handle_alternate_actions()
    build_wheel(python, pip)
    try:
        sccache("-s")
    except CommandFailed:
        pass


def action_android():
    sccache, python, pip = setup_basic_build_env()
    setup_android_ndk()
    handle_alternate_actions()
    build_android(python, pip)
    try:
        sccache("-s")
    except CommandFailed:
        pass


def action_ios():
    sccache, python, pip = setup_basic_build_env()
    setup_ios(python, pip)
    handle_alternate_actions()
    build_ios()


def action_open_cache_dir():
    d = misc.get_cache_home()
    misc.info(f"Opening cache directory: {d}")

    if sys.platform == "win32":
        os.startfile(d)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", d])
    else:
        subprocess.Popen(["xdg-open", d])


def parse_args():
    parser = argparse.ArgumentParser()

    # Possible actions:
    #   wheel: build the wheel
    #   android: build the Android C-API shared library
    #   ios: build the iOS C-API shared library
    #   cache: open the cache directory
    help = 'Action, may be build target "wheel" / "android" / "ios", or "cache" for opening the cache directory.'
    parser.add_argument("action", type=str, nargs="?", default="wheel", help=help)

    help = "Do not build, write environment variables to file instead"
    parser.add_argument("-w", "--write-env", type=str, default=None, help=help)

    help = "Do not build, start a shell with environment variables set instead"
    parser.add_argument("-s", "--shell", action="store_true", help=help)

    help = (
        "Python version to use, e.g. '3.7', '3.11', or 'native' to not use an isolated python environment. "
        "Defaults to the same version of the current python interpreter."
    )
    parser.add_argument("--python", default=None, help=help)

    help = "Continue when encounters error."
    parser.add_argument("--permissive", action="store_true", default=False, help=help)

    help = "Tag built wheel with TI_WITH_xxx config."
    parser.add_argument("--tag-config", action="store_true", default=False, help=help)

    help = "Set a local version. Overrides --tag-config."
    parser.add_argument("--tag-local", type=str, default=None, help=help)

    help = "Build nightly wheel."
    parser.add_argument("--nightly", action="store_true", default=False, help=help)

    options = parser.parse_args()
    return options


def main() -> int:
    options = parse_args()
    misc.options = options

    def action_notimpl():
        raise RuntimeError(f"Unknown action: {options.action}")

    dispatch = {
        "wheel": action_wheel,
        "android": action_android,
        "ios": action_ios,
        "cache": action_open_cache_dir,
    }

    dispatch.get(options.action, action_notimpl)()

    return 0
