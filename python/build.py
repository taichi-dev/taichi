import argparse
import os
import platform
import sys
import shutil
import taichi as ti

if len(sys.argv) != 2:
    print("Usage: python3 build.py [upload/test]")
    exit(-1)

version = ti.core.get_version_string()
gpu = ti.core.with_cuda()
mode = sys.argv[1]

env_pypi_pwd = os.environ.get('PYPI_PWD', '')

if mode == 'try_upload':
    if env_pypi_pwd == '':
        print("Missing environment variable PYPI_PWD")
        print("Giving up and exiting 0 [try_upload mode]")
        exit(0)
    mode = 'upload'

if env_pypi_pwd == '':
    assert False, "Missing environment variable PYPI_PWD"


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


if platform.system() == 'Linux':
    if os.environ['CXX'] not in ['clang++-8', 'clang++-7', 'clang++']:
        print('Only the wheel with clang will be released to PyPI.')
        sys.exit(-1)

    if not gpu:
        print('Linux release must ship with the CUDA backend.')
        sys.exit(-1)

with open('../setup.py') as fin:
    with open('setup.py', 'w') as fout:
        project_name = 'taichi'
        print("project_name = '{}'".format(project_name), file=fout)
        print("version = '{}'".format(version), file=fout)
        for l in fin:
            print(l, file=fout, end='')

print("*** project_name = '{}'".format(project_name))

shutil.rmtree('taichi/lib', ignore_errors=True)
shutil.rmtree('taichi/tests', ignore_errors=True)
os.makedirs('taichi/lib', exist_ok=True)
shutil.rmtree('build', ignore_errors=True)
shutil.rmtree('dist', ignore_errors=True)
shutil.rmtree('taichi/include', ignore_errors=True)
# shutil.copytree('../include/', 'taichi/include')
build_dir = '../build'

if get_os_name() == 'linux':
    shutil.copy('../build/libtaichi_core.so', 'taichi/lib/taichi_core.so')
elif get_os_name() == 'osx':
    shutil.copy('../build/libtaichi_core.dylib', 'taichi/lib/taichi_core.so')
else:
    shutil.copy('../runtimes/RelWithDebInfo/taichi_core.dll',
                'taichi/lib/taichi_core.pyd')

shutil.copytree('../tests/python', './taichi/tests')

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
os.system('{} -m pip install --user --upgrade twine setuptools wheel'.format(
    get_python_executable()))

if get_os_name() == 'linux':
    os.system('{} setup.py bdist_wheel -p manylinux1_x86_64'.format(
        get_python_executable()))
else:
    os.system('{} setup.py bdist_wheel'.format(get_python_executable()))

shutil.rmtree('taichi/lib')
shutil.rmtree('taichi/tests')
shutil.rmtree('./build')

if mode == 'upload':
    os.system(
        '{} -m twine upload dist/* --verbose -u yuanming-hu -p {}'.format(
            get_python_executable(),
            '%PYPI_PWD%' if get_os_name() == 'win' else '$PYPI_PWD'))
elif mode == 'test':
    print('Uninstalling old taichi packages...')
    os.system('{} -m pip uninstall taichi-nightly'.format(
        get_python_executable()))
    os.system('{} -m pip uninstall taichi'.format(get_python_executable()))
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
elif mode == '':
    pass
else:
    print("Unknown mode: ", mode)
    exit(-1)
