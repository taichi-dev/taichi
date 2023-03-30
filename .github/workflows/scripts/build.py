#!/usr/bin/env python3

import ci_common  # isort: skip, early initialization happens here

import argparse
import glob
import os
import platform
import sys
from pathlib import Path

from ci_common import misc
from ci_common.dep import download_dep
from ci_common.misc import (banner, get_cache_home, is_manylinux2014,
                            path_prepend)
from ci_common.python import path_prepend, setup_python
from ci_common.sccache import setup_sccache
from ci_common.tinysh import Command, git, sh


# -- code --
@banner('Setup Clang')
def setup_clang(as_compiler=True) -> None:
    '''
    Setup Clang.
    '''
    u = platform.uname()
    if u.system == 'Linux':
        pass
    elif (u.system, u.machine) == ('Windows', 'AMD64'):
        out = get_cache_home() / 'clang-15-v2'
        url = 'https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/clang-15.0.0-win-complete.zip'
        download_dep(url, out, force=True)
        clang = str(out / 'bin' / 'clang++.exe').replace('\\', '\\\\')
        os.environ['TAICHI_CMAKE_ARGS'] += f' -DCLANG_EXECUTABLE={clang}'

        if as_compiler:
            os.environ['TAICHI_CMAKE_ARGS'] += (
                f' -DCMAKE_CXX_COMPILER={clang}'
                f' -DCMAKE_C_COMPILER={clang}')
    else:
        # TODO: unify all
        pass


@banner('Setup MSVC')
def setup_msvc() -> None:
    assert platform.system() == 'Windows'
    os.environ['TAICHI_USE_MSBUILD'] = '1'

    url = 'https://aka.ms/vs/17/release/vs_BuildTools.exe'
    out = Path(
        r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools')
    download_dep(
        url,
        out,
        args=[
            '--passive',
            '--wait',
            '--norestart',
            '--includeRecommended',
            '--add',
            'Microsoft.VisualStudio.Workload.VCTools',
            # NOTE: We are using the custom built Clang++,
            #       so components below are not necessary anymore.
            # '--add',
            # 'Microsoft.VisualStudio.Component.VC.Llvm.Clang',
            # '--add',
            # 'Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Llvm.Clang',
            # '--add',
            # 'Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset',
        ])


@banner('Setup LLVM')
def setup_llvm() -> None:
    '''
    Download and install LLVM.
    '''
    u = platform.uname()
    if u.system == 'Linux':
        if 'AMDGPU_TEST' in os.environ:
            out = get_cache_home() / 'llvm15-amdgpu'
            url = 'https://github.com/GaleSeLee/assets/releases/download/v0.0.4/taichi-llvm-15.0.0-linux.zip'
        elif is_manylinux2014():
            # FIXME: prebuilt llvm15 on ubuntu didn't work on manylinux2014 image of centos. Once that's fixed, remove this hack.
            out = get_cache_home() / 'llvm15-manylinux2014'
            url = 'https://github.com/ailzhang/torchhub_example/releases/download/0.3/taichi-llvm-15-linux.zip'
        else:
            out = get_cache_home() / 'llvm15'
            url = 'https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-linux.zip'
        download_dep(url, out, strip=1)
    elif (u.system, u.machine) == ('Darwin', 'arm64'):
        out = get_cache_home() / 'llvm15-m1'
        url = 'https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-m1.zip'
        download_dep(url, out, strip=1)
    elif (u.system, u.machine) == ('Darwin', 'x86_64'):
        out = get_cache_home() / 'llvm15-mac'
        url = 'https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/llvm-15-mac10.15.zip'
        download_dep(url, out, strip=1)
    elif (u.system, u.machine) == ('Windows', 'AMD64'):
        out = get_cache_home() / 'llvm15'
        url = 'https://github.com/python3kgae/taichi_assets/releases/download/llvm15_vs2019_clang/taichi-llvm-15.0.0-msvc2019.zip'
        download_dep(url, out, strip=0)
    else:
        raise RuntimeError(f'Unsupported platform: {u.system} {u.machine}')

    # We should use LLVM toolchains shipped with OS.
    # path_prepend('PATH', out / 'bin')
    os.environ['LLVM_DIR'] = str(out)


@banner('Setup Android NDK')
def setup_android_ndk() -> None:
    # TODO: Auto install
    s = platform.system()
    if s != 'Linux':
        raise RuntimeError(
            f'Android NDK is only supported on Linux, but the current platform is {s}.'
        )

    ndkroot = Path(
        os.environ.get('ANDROID_NDK_ROOT', '/android-sdk/ndk-bundle'))
    toolchain = ndkroot / 'build/cmake/android.toolchain.cmake'
    if not toolchain.exists():
        raise RuntimeError(
            f'ANDROID_NDK_ROOT is set to {ndkroot}, but the path does not exist.'
        )

    p = ndkroot.resolve()
    os.environ['ANDROID_NDK_ROOT'] = str(p)
    cmake_args = (f' -DCMAKE_TOOLCHAIN_FILE={toolchain}'
                  ' -DANDROID_NATIVE_API_LEVEL=29'
                  ' -DANDROID_ABI=arm64-v8a')
    os.environ['ANDROID_CMAKE_ARGS'] = cmake_args.strip()
    os.environ['TAICHI_CMAKE_ARGS'] += cmake_args
    path_prepend('PATH', p / 'toolchains/llvm/prebuilt/linux-x86_64/bin')


@banner('Setup Vulkan 1.3.236.0')
def setup_vulkan():
    u = platform.uname()
    if u.system == "Linux":
        url = 'https://sdk.lunarg.com/sdk/download/1.3.236.0/linux/vulkansdk-linux-x86_64-1.3.236.0.tar.gz'
        prefix = get_cache_home() / 'vulkan-1.3.236.0'
        download_dep(url, prefix, strip=1)
        sdk = prefix / 'x86_64'
        os.environ['VULKAN_SDK'] = str(sdk)
        path_prepend('PATH', sdk / "bin")
        path_prepend('LD_LIBRARY_PATH', sdk / 'lib')
        os.environ['VK_LAYER_PATH'] = str(sdk / 'etc' / 'vulkan' /
                                          'explicit_layer.d')
    # elif (u.system, u.machine) == ("Darwin", "arm64"):
    # elif (u.system, u.machine) == ("Darwin", "x86_64"):
    elif (u.system, u.machine) == ('Windows', 'AMD64'):
        url = 'https://sdk.lunarg.com/sdk/download/1.3.236.0/windows/VulkanSDK-1.3.236.0-Installer.exe'
        prefix = get_cache_home() / 'vulkan-1.3.236.0'
        download_dep(
            url,
            prefix,
            args=[
                '--accept-licenses',
                '--default-answer',
                '--confirm-command',
                '--root',
                prefix,
                'install',
                'com.lunarg.vulkan.sdl2',
                'com.lunarg.vulkan.glm',
                'com.lunarg.vulkan.volk',
                'com.lunarg.vulkan.vma',
                # 'com.lunarg.vulkan.debug',
            ])
        os.environ['VULKAN_SDK'] = str(prefix)
        os.environ['VK_SDK_PATH'] = str(prefix)
        path_prepend('PATH', prefix / "Bin")
    else:
        return


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
        os.environ['TAICHI_CMAKE_ARGS'] += ' -DTI_WITH_C_API=ON'

    if platform.system() == 'Linux':
        if is_manylinux2014():
            extra.extend(['-p', 'manylinux2014_x86_64'])
        else:
            extra.extend(['-p', 'manylinux_2_27_x86_64'])

    python('setup.py', 'clean')
    python('misc/make_changelog.py', '--ver', 'origin/master', '--repo_dir',
           './', '--save')

    python('setup.py', *proj_tags, 'bdist_wheel', *extra)


def setup_basic_build_env():
    u = platform.uname()
    if (u.system, u.machine) == ('Windows', 'AMD64'):
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


@banner('Build Taichi Android C-API Shared Library')
def build_android(python: Command, pip: Command) -> None:
    '''
    Build the Taichi Android C-API shared library
    '''
    pip.install('-r', 'requirements_dev.txt')
    python('setup.py', 'clean')
    python('setup.py', 'build_ext')
    sh('aarch64-linux-android-strip', 'build/libtaichi_c_api.so')


def action_android():
    sccache, python, pip = setup_basic_build_env()
    setup_android_ndk()
    os.environ['TAICHI_CMAKE_ARGS'] += (' -DTI_WITH_BACKTRACE:BOOL=OFF'
                                        ' -DTI_WITH_LLVM:BOOL=OFF'
                                        ' -DTI_WITH_C_API:BOOL=ON'
                                        ' -DTI_BUILD_TESTS:BOOL=OFF')
    handle_alternate_actions()
    build_android(python, pip)
    sccache('-s')


@banner('Add AOT Demo Related Environment Variables')
def add_aot_env():
    os.environ['TAICHI_REPO_DIR'] = os.getcwd()
    for p in Path(os.getcwd()).glob('**/cmake-install/c_api'):
        if p.is_dir() and p.parent.parent.name.endswith(os.environ['PY']):
            os.environ['TAICHI_C_API_INSTALL_DIR'] = str(p)
            break
    else:
        misc.warn(
            "Failed to find TAICHI_C_API_INSTALL_DIR (did't build C-API?), skipping"
        )


def enter_shell():
    misc.info('Entering shell...')
    import pwd
    shell = pwd.getpwuid(os.getuid()).pw_shell
    os.execl(shell, shell)


def write_env(path):
    envs = os.environ.get_changed_envs()
    envstr = ''
    if path.endswith('.ps1'):
        envstr = '\n'.join([f'$env:{k}="{v}"' for k, v in envs.items()])
    elif path.endswith('.sh'):
        envstr = '\n'.join([f'export {k}="{v}"' for k, v in envs.items()])
    elif path.endswith('.json'):
        import json
        envstr = json.dumps(envs, indent=2)
    else:
        raise RuntimeError(f'Unsupported format')

    with open(path, 'w') as f:
        f.write(envstr)

    misc.info(f'Environment variables written to {path}')


def handle_alternate_actions():
    if misc.options.write_env:
        add_aot_env()
        write_env(misc.options.write_env)
    elif misc.options.shell:
        add_aot_env()
        enter_shell()
    else:
        return

    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser()
    # Possible actions:
    #   wheel: build the wheel
    #   android: build the Android C-API shared library
    parser.add_argument('action',
                        type=str,
                        nargs='?',
                        default='wheel',
                        help='Build target, may be "wheel" / "android"')
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
    }

    dispatch.get(options.action, action_notimpl)()


if __name__ == '__main__':
    main()
