import argparse
import os
import platform
import shutil
import sys

import taichi as ti


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


def build():
    """Build and package the wheel file in `python/dist`"""
    if platform.system() == 'Linux':
        if os.environ.get(
                'CXX', 'clang++') not in ['clang++-8', 'clang++-7', 'clang++']:
            raise RuntimeError(
                'Only the wheel with clang will be released to PyPI')

    version = ti.core.get_version_string()
    with open('../setup.py') as fin:
        with open('setup.py', 'w') as fout:
            project_name = 'taichi'
            print("project_name = '{}'".format(project_name), file=fout)
            print("version = '{}'".format(version), file=fout)
            for l in fin:
                print(l, file=fout, end='')

    print("*** project_name = '{}'".format(project_name))

    try:
        os.remove('taichi/CHANGELOG.md')
    except FileNotFoundError:
        pass
    shutil.rmtree('taichi/lib', ignore_errors=True)
    shutil.rmtree('taichi/tests', ignore_errors=True)
    shutil.rmtree('taichi/examples', ignore_errors=True)
    shutil.rmtree('taichi/assets', ignore_errors=True)
    os.makedirs('taichi/lib', exist_ok=True)
    shutil.rmtree('build', ignore_errors=True)
    shutil.rmtree('dist', ignore_errors=True)
    shutil.rmtree('taichi/include', ignore_errors=True)
    # shutil.copytree('../include/', 'taichi/include')
    build_dir = '../build'

    if get_os_name() == 'linux':
        shutil.copy('../build/libtaichi_core.so', 'taichi/lib/taichi_core.so')
    elif get_os_name() == 'osx':
        shutil.copy('../build/libtaichi_core.dylib',
                    'taichi/lib/taichi_core.so')
    else:
        shutil.copy('../runtimes/RelWithDebInfo/taichi_core.dll',
                    'taichi/lib/taichi_core.pyd')

    os.system(f'cd .. && {get_python_executable()} -m taichi changelog --save')

    try:
        with open('../CHANGELOG.md') as f:
            print(f.read())
    except FileNotFoundError:
        print('CHANGELOG.md not found')
        pass

    try:
        shutil.copy('../CHANGELOG.md', './taichi/CHANGELOG.md')
    except FileNotFoundError:
        pass
    shutil.copytree('../tests/python', './taichi/tests')
    shutil.copytree('../examples', './taichi/examples')
    shutil.copytree('../external/assets', './taichi/assets')

    if get_os_name() != 'osx':
        libdevice_path = ti.core.libdevice_path()
        print("copying libdevice:", libdevice_path)
        assert os.path.exists(libdevice_path)
        shutil.copy(libdevice_path, 'taichi/lib/slim_libdevice.10.bc')

    ti.core.compile_runtimes()
    runtime_dir = ti.core.get_runtime_dir()
    for f in os.listdir(runtime_dir):
        if f.startswith('runtime_') and f.endswith('.bc'):
            print(f"Fetching runtime file {f}")
            shutil.copy(os.path.join(runtime_dir, f), 'taichi/lib')

    print("Using python executable", get_python_executable())
    os.system(
        '{} -m pip install --user --upgrade twine setuptools wheel'.format(
            get_python_executable()))

    if get_os_name() == 'linux':
        os.system('{} setup.py bdist_wheel -p manylinux1_x86_64'.format(
            get_python_executable()))
    else:
        os.system('{} setup.py bdist_wheel'.format(get_python_executable()))

    shutil.rmtree('taichi/lib')
    shutil.rmtree('taichi/tests')
    shutil.rmtree('taichi/examples')
    shutil.rmtree('taichi/assets')
    try:
        os.remove('taichi/CHANGELOG.md')
    except FileNotFoundError:
        pass
    shutil.rmtree('./build')


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
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    env_pypi_pwd = os.environ.get('PYPI_PWD', '')
    if mode == 'try_upload':
        if env_pypi_pwd == '':
            print("Missing environment variable PYPI_PWD")
            print("Giving up and exiting 0 [try_upload mode]")
            exit(0)
        mode = 'upload'

    if mode == 'upload' and env_pypi_pwd == '':
        raise RuntimeError("Missing environment variable PYPI_PWD")

    if not args.skip_build:
        build()

    if mode == 'build':
        return
    elif mode == 'upload':
        os.system(
            '{} -m twine upload dist/* --verbose -u yuanming-hu -p {}'.format(
                get_python_executable(),
                '%PYPI_PWD%' if get_os_name() == 'win' else '$PYPI_PWD'))
    elif mode == 'test':
        print('Uninstalling old taichi packages...')
        os.system(f'{get_python_executable()} -m pip uninstall taichi-nightly')
        os.system(f'{get_python_executable()} -m pip uninstall taichi')
        dists = os.listdir('dist')
        assert len(dists) == 1
        dist = dists[0]
        print('Installing ', dist)
        os.environ['PYTHONPATH'] = ''
        os.makedirs('test_env', exist_ok=True)
        os.system('cd test_env && {} -m pip install ../dist/{} --user'.format(
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
