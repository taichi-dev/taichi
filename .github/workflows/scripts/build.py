#!/usr/bin/python3

# -- prioritized --
import sys
import os.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import common
common.init()

# -- stdlib --
from pathlib import Path
import os
import platform

# -- third party --
from sh import git, python, sccache

# -- own --

# -- code --
env = os.environ
pip = python.bake('-m', 'pip')


def download_dep(url, outdir, *, strip_leading=False):
    pass


def setup_llvm():
    u = platform.uname()
    if u.system == 'Linux':
        if Path('/etc/centos-release').exists():
            # FIXME: prebuilt llvm15 on ubuntu didn't work on manylinux2014 image of centos. Once that's fixed, remove this hack.
            url = 'https://github.com/ailzhang/torchhub_example/releases/download/0.3/taichi-llvm-15-linux.zip'
        else:
            url = 'https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-linux.zip'
    elif (u.system, u.machine) == ('Darwin', 'arm64'):
        url = 'https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-m1.zip'
    elif (u.system, u.machine) == ('Darwin', 'x86_64'):
        url = 'https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/llvm-15-mac10.15.zip'
    else:
        raise RuntimeError(f'Unsupported platform: {u.system} {u.machine}')

    download_dep(url, 'taichi-llvm-15')
    env['LLVM_DIR'] = str(Path.home() / "taichi-llvm-15")


def build_wheel():
    pip.install('-r', 'requirements_dev.txt')
    git.fetch('origin', 'master', '--tags')
    proj = env['PROJECT_NAME']
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

    # python3 misc/make_changelog.py --ver origin/master --repo_dir ./ --save
    python('setup.py', *proj_tags, 'bdist_wheel', *extra)
    sccache('-s')



setup_llvm()


# setup sccache
# setup_python

build_wheel()

# NUM_WHL=$(ls dist/*.whl | wc -l)
# if [ $NUM_WHL -ne 1 ]; then echo "ERROR: created more than 1 whl." && exit 1; fi

# chmod -R 777 "$SCCACHE_DIR"
# rm -f python/CHANGELOG.md
