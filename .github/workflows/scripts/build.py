#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -- prioritized --
import ci_common  # isort: skip, early initialization happens here

# -- stdlib --
import argparse
import glob
import os
import platform

# -- third party --
# -- own --
from ci_common import misc
from ci_common.alter import handle_alternate_actions
from ci_common.android import build_android, setup_android_ndk
from ci_common.cmake import cmake_args
from ci_common.compiler import setup_clang, setup_msvc
from ci_common.ios import build_ios, setup_ios
from ci_common.llvm import setup_llvm
from ci_common.misc import banner, is_manylinux2014
from ci_common.python import setup_python
from ci_common.sccache import setup_sccache
from ci_common.tinysh import Command, git
from ci_common.vulkan import setup_vulkan


# -- code --
@banner('Build Taichi Wheel')
def build_wheel(python: Command, pip: Command) -> None:
    '''
    Build the Taichi wheel
    '''
    pip.install('-U', 'pip')
    pip.uninstall('-y', 'taichi', 'taichi-nightly')
    pip.install('-r', 'requirements_dev.txt')
    git.fetch('origin', 'master', '--tags')
    proj = os.environ.get('PROJECT_NAME', 'taichi')
    proj_tags = []
    extra = []

    if proj == 'taichi-nightly':
        proj_tags.extend(['egg_info', '--tag-date', '--tag-build=.post'])
        # Include C-API in nightly builds
        cmake_args['TI_WITH_C_API'] = True

    if platform.system() == 'Linux':
        if is_manylinux2014():
            extra.extend(['-p', 'manylinux2014_x86_64'])
        else:
            extra.extend(['-p', 'manylinux_2_27_x86_64'])

    cmake_args.writeback()
    python('setup.py', 'clean')
    python('misc/make_changelog.py', '--ver', 'origin/master', '--repo_dir',
           './', '--save')

    python('setup.py', *proj_tags, 'bdist_wheel', *extra)


def setup_basic_build_env(force_vulkan=False):
    u = platform.uname()
    if (u.system, u.machine) == ('Windows', 'AMD64'):
        # Use MSVC on Windows
        setup_clang(as_compiler=False)
        setup_msvc()
    else:
        # Use Clang on all other platforms
        setup_clang()

    setup_llvm()
    if force_vulkan or cmake_args.get_effective('TI_WITH_VULKAN'):
        setup_vulkan()

    sccache = setup_sccache()

    # NOTE: We use conda/venv to build wheels, which may not be the same python
    #       running this script.
    python, pip = setup_python(os.environ['PY'])

    return sccache, python, pip


def action_wheel():
    sccache, python, pip = setup_basic_build_env()
    handle_alternate_actions()
    build_wheel(python, pip)
    sccache('-s')

    distfiles = glob.glob('dist/*.whl')
    if len(distfiles) != 1:
        raise RuntimeError(
            f'Failed to produce exactly one wheel file: {distfiles}')


def action_android():
    sccache, python, pip = setup_basic_build_env(force_vulkan=True)
    setup_android_ndk()
    handle_alternate_actions()
    build_android(python, pip)
    sccache('-s')


def action_ios():
    sccache, python, pip = setup_basic_build_env()
    setup_ios(python, pip)
    handle_alternate_actions()
    build_ios()


def parse_args():
    parser = argparse.ArgumentParser()
    # Possible actions:
    #   wheel: build the wheel
    #   android: build the Android C-API shared library
    #   ios: build the iOS C-API shared library
    parser.add_argument(
        'action',
        type=str,
        nargs='?',
        default='wheel',
        help='Build target, may be "wheel" / "android" / "ios"')
    parser.add_argument(
        '-w',
        '--write-env',
        type=str,
        default=None,
        help='Do not build, write environment variables to file instead')
    parser.add_argument(
        '-s',
        '--shell',
        action='store_true',
        help=
        'Do not build, start a shell with environment variables set instead')
    options = parser.parse_args()
    return options


def main() -> None:
    options = parse_args()
    misc.options = options

    def action_notimpl():
        raise RuntimeError(f'Unknown action: {options.action}')

    dispatch = {
        'wheel': action_wheel,
        'android': action_android,
        'ios': action_ios,
    }

    dispatch.get(options.action, action_notimpl)()


if __name__ == '__main__':
    main()
