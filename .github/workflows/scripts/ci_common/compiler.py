# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform
from pathlib import Path

# -- third party --
# -- own --
from .cmake import cmake_args
from .dep import download_dep
from .misc import banner, get_cache_home


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
        cmake_args['CLANG_EXECUTABLE'] = clang

        if as_compiler:
            cmake_args['CMAKE_CXX_COMPILER'] = clang
            cmake_args['CMAKE_C_COMPILER'] = clang
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
