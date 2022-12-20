#!/usr/bin/python3 -u

# -- prioritized --
import ci_common  # isort: skip, early initialization happens here

# -- stdlib --
import glob
import os
import platform

# -- third party --
# -- own --
from ci_common.dep import download_dep
from ci_common.misc import banner, get_cache_home, is_manylinux2014
from ci_common.python import setup_python
from ci_common.sccache import setup_sccache
from ci_common.tinysh import Command, environ, git, sh


# -- code --
@banner('Setup LLVM')
def setup_llvm(env_out: dict) -> None:
    '''
    Download and install LLVM.
    '''
    u = platform.uname()
    if u.system == 'Linux':
        if 'AMDGPU_TEST' in os.environ:
            # FIXME: AMDGPU bots are currently maintained separately,
            #        we should unify them with the rest of the bots.
            lnsf = sh.sudo.ln.bake('-sf')
            lnsf('/usr/bin/clang++-10', '/usr/bin/clang++')
            lnsf('/usr/bin/clang-10', '/usr/bin/clang')
            lnsf('/usr/bin/ld.lld-10', '/usr/bin/ld.lld')
            env_out['LLVM_DIR'] = '/taichi-llvm-15'
            return
        elif is_manylinux2014():
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
    env_out['LLVM_DIR'] = str(out)


@banner('Build Taichi Wheel')
def build_wheel(python: Command, pip: Command, env: dict) -> None:
    '''
    Build the Taichi wheel
    '''
    pip.install('-r', 'requirements_dev.txt')
    git.fetch('origin', 'master', '--tags')
    proj = env['PROJECT_NAME']
    proj_tags = []
    extra = []

    if proj == 'taichi-nightly':
        proj_tags.extend(['egg_info', '--tag-date'])
        # Include C-API in nightly builds
        os.environ['TAICHI_CMAKE_ARGS'] += ' -DTI_WITH_C_API=ON'

    if platform.system() == 'Linux':
        if is_manylinux2014():
            extra.extend(['-p', 'manylinux2014_x86_64'])
        else:
            extra.extend(['-p', 'manylinux_2_27_x86_64'])

    python('misc/make_changelog.py', '--ver', 'origin/master', '--repo_dir',
           './', '--save')

    with environ(env):
        python('setup.py', *proj_tags, 'bdist_wheel', *extra)


def main() -> None:
    env = {
        'TAICHI_CMAKE_ARGS': os.environ.get('TAICHI_CMAKE_ARGS', ''),
        'PROJECT_NAME': os.environ.get('PROJECT_NAME', 'taichi'),
    }
    setup_llvm(env)
    sccache = setup_sccache(env)

    # NOTE: We use conda/venv to build wheels, which may not be the same python
    #       running this script.
    python, pip = setup_python(os.environ['PY'])
    build_wheel(python, pip, env)

    sccache('-s')

    distfiles = glob.glob('dist/*.whl')
    if len(distfiles) != 1:
        raise RuntimeError(
            f'Failed to produce exactly one wheel file: {distfiles}')


if __name__ == '__main__':
    main()
