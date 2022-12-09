#!/usr/bin/python3 -u

# -- prioritized --
import sys
import os.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import common

# -- stdlib --
import glob
import os
import platform

# -- third party --
# -- own --
from common.dep import download_dep
from common.misc import get_cache_home, banner
from common.python import setup_python
from common.sccache import setup_sccache
from common.tinysh import git, sccache, sh, sudo


# -- code --
env = os.environ


@banner('Setup LLVM')
def setup_llvm():
    u = platform.uname()
    if u.system == 'Linux':
        if 'AMDGPU_TEST' in env:
            # FIXME: AMDGPU bots are currently maintained separately,
            #        we should unify them with the rest of the bots.
            lnsf = sudo(sh.ln.bake('-sf'))
            lnsf('/usr/bin/clang++-10', '/usr/bin/clang++')
            lnsf('/usr/bin/clang-10',   '/usr/bin/clang')
            lnsf('/usr/bin/ld.lld-10',  '/usr/bin/ld.lld')
            env['LLVM_DIR'] = '/taichi-llvm-15'
            return
        elif common.is_manylinux2014():
            # FIXME: prebuilt llvm15 on ubuntu didn't work on manylinux2014 image of centos. Once that's fixed, remove this hack.
            out = get_cache_home() / 'llvm15-manylinux2014'
            url = 'https://github.com/ailzhang/torchhub_example/releases/download/0.3/taichi-llvm-15-linux.zip'
        else:
            out = get_cache_home() / 'llvm15'
            url = 'https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-linux.zip'
    elif (u.system, u.machine) == ('Darwin', 'arm64'):
        out = get_cache_home() / 'llvm15-m1'
        url = 'https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-m1.zip'
    elif (u.system, u.machine) == ('Darwin', 'x86_64'):
        out = get_cache_home() / 'llvm15-mac'
        url = 'https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/llvm-15-mac10.15.zip'
    else:
        raise RuntimeError(f'Unsupported platform: {u.system} {u.machine}')

    download_dep(url, out, strip=1)
    env['LLVM_DIR'] = str(out)


@banner('Build Taichi Wheel')
def build_wheel(python, pip):
    pip.install('-r', 'requirements_dev.txt')
    git.fetch('origin', 'master', '--tags')
    proj = env.get('PROJECT_NAME', 'taichi')
    proj_tags = []
    extra = []

    if proj == 'taichi-nightly':
        proj_tags.extend(['egg_info', '--tag-date'])
        # Include C-API in nightly builds
        env['TAICHI_CMAKE_ARGS'] += ' -DTI_WITH_C_API=ON'

    if platform.system() == 'Linux':
        if common.is_manylinux2014():
            extra.extend(['-p', 'manylinux2014_x86_64'])
        else:
            extra.extend(['-p', 'manylinux_2_27_x86_64'])

    python('misc/make_changelog.py', '--ver', 'origin/master', '--repo_dir', './', '--save')
    python('setup.py', *proj_tags, 'bdist_wheel', *extra)


setup_llvm()
sccache = setup_sccache()
python, pip = setup_python(env['PY'])
build_wheel(python, pip)
sccache('-s')

distfiles = glob.glob('dist/*.whl')
if len(distfiles) != 1:
    raise RuntimeError(f'Failed to produce exactly one wheel file: {distfiles}')
