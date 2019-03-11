import os
import platform
import sys

def get_python_executable():
    return sys.executable.replace('\\','/')

if platform.system() == 'Linux':
    if os.environ['CXX'] != 'g++-6':
        print('Only the wheel with g++-6 will be released to PyPI.')
        sys.exit(0)

os.makedirs('taichi/lib', exist_ok=True)
os.system('cp ../build/libtaichi_core.so taichi/lib/taichi_core.so')
os.system('cp ../build/libtaichi_core.dylib taichi/lib/taichi_core.so')
os.system('cp ../external/lib/* taichi/lib')
os.system('rm taichi/lib/ffmpeg')
os.system('rm dist/*.whl')
os.system('{} -m pip install --user --upgrade twine setuptools wheel'.format(get_python_executable()))

if platform.system() == 'Linux':
    os.system('{} setup.py bdist_wheel -p manylinux1_x86_64'.format(get_python_executable()))
else:
    os.system('{} setup.py bdist_wheel'.format(get_python_executable()))
os.system('{} -m twine upload dist/* --verbose -u yuanming -p $PYPI_PWD'.format(get_python_executable()))
