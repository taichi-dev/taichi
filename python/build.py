import argparse
import os
import platform
import re
import shutil
import sys


def get_os_name():
    name = platform.platform()
    # in python 3.8, platform.platform() uses mac_ver() on macOS
    # it will return 'macOS-XXXX' instead of 'Darwin-XXXX'
    if name.lower().startswith('darwin') or name.lower().startswith('macos'):
        return 'osx'
    elif name.lower().startswith('windows'):
        return 'win'
    elif name.lower().startswith('linux'):
        return 'linux'
    assert False, "Unknown platform name %s" % name


def get_python_executable():
    return '"' + sys.executable.replace('\\', '/') + '"'


def build(project_name):
    """Build and package the wheel file in root `dist` dir"""
    if platform.system() == 'Linux':
        if re.search("^clang\+\+-*\d*", str(os.environ.get('CXX'))) is None:
            raise RuntimeError(
                'Only the wheel with clang will be released to PyPI')

    print("Using python executable", get_python_executable())
    os.system(
        '{} -m pip install --user --upgrade twine setuptools wheel'.format(
            get_python_executable()))

    os.system(
        f'{get_python_executable()} ../misc/make_changelog.py origin/master ../ True'
    )

    # This env var is used in setup.py below.
    os.environ['PROJECT_NAME'] = project_name
    project_tag = ''
    if project_name == 'taichi-nightly':
        project_tag = 'egg_info --tag-date'
    if get_os_name() == 'linux':
        os.system(
            f'cd ..; {get_python_executable()} setup.py {project_tag} bdist_wheel -p manylinux1_x86_64'
        )
    else:
        os.system(
            f'cd .. && {get_python_executable()} setup.py {project_tag} bdist_wheel'
        )

    try:
        os.remove('taichi/CHANGELOG.md')
    except FileNotFoundError:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description=(
        'Build and uploads wheels to PyPI. Make sure to run this script '
        'inside `python/`'))
    parser.add_argument('mode',
                        type=str,
                        default='',
                        help=('Choose one of the modes: '
                              '[build, test, try_upload, upload]'))
    parser.add_argument('--skip_build',
                        action='store_true',
                        help=('Skip the build process if this is enabled'))
    parser.add_argument('--testpypi',
                        action='store_true',
                        help='Upload to test server if this is enabled')
    parser.add_argument('--project_name',
                        action='store',
                        dest='project_name',
                        default='taichi',
                        help='Set the project name')
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode
    pypi_user = '__token__'
    pypi_repo = ''
    project_name = args.project_name

    env_pypi_pwd = os.environ.get('PYPI_PWD', '')

    if not args.skip_build:
        shutil.rmtree('../dist', ignore_errors=True)

    if mode == 'try_upload':
        if env_pypi_pwd == '':
            print("Missing environment variable PYPI_PWD")
            print("Giving up and exiting 0 [try_upload mode]")
            exit(0)
        mode = 'upload'

    if mode == 'upload' and env_pypi_pwd == '':
        raise RuntimeError("Missing environment variable PYPI_PWD")

    os.environ['TWINE_PASSWORD'] = env_pypi_pwd

    if mode == 'upload' and args.testpypi:
        pypi_repo = '--repository testpypi'

    if not args.skip_build:
        build(project_name)

    if mode == 'build':
        return
    elif mode == 'upload':
        os.system('{} -m twine upload {} ../dist/* --verbose -u {}'.format(
            get_python_executable(), pypi_repo, pypi_user))
    elif mode == 'test':
        print('Uninstalling old taichi packages...')
        os.system(
            f'{get_python_executable()} -m pip uninstall -y taichi-nightly')
        os.system(f'{get_python_executable()} -m pip uninstall -y taichi')
        dists = os.listdir('../dist')
        assert len(dists) == 1
        dist = dists[0]
        print('Installing ', dist)
        os.environ['PYTHONPATH'] = ''
        os.makedirs('test_env', exist_ok=True)
        os.system(
            'cd test_env && {} -m pip install ../../dist/{} --user'.format(
                get_python_executable(), dist))
        print('Entering test environment...')
        if get_os_name() == 'win':
            os.system(
                'cmd /V /C "set PYTHONPATH=&& set TAICHI_REPO_DIR=&& cd test_env && cmd"'
            )
        else:
            os.system(
                'cd test_env && PYTHONPATH= TAICHI_REPO_DIR= bash --noprofile --norc '
            )
    else:
        raise ValueError("Unknown mode: %s" % mode)


if __name__ == '__main__':
    main()
