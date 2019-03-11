import os
import platform
import sys

def get_python_executable():
    return sys.executable.replace('\\','/')

os.makedirs('python/lib', exist_ok=True)
os.system('cp ../build/libtaichi_core.so python/lib/taichi_core.so')
os.system('cp ../build/libtaichi_core.dylib python/lib/taichi_core.so')
os.system('cp ../external/lib/* python/lib')
os.system('rm python/lib/ffmpeg')
os.system('rm dist/*.whl')
os.system('{} -m pip install --user --upgrade twine setuptools wheel'.format(get_python_executable()))

if platform.system() == 'Linux':
    os.system('{} setup.py bdist_wheel -p manylinux1_x86_64'.format(get_python_executable()))
else:
    os.system('{} setup.py bdist_wheel'.format(get_python_executable()))
os.system('{} -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose -u yuanming -p $PYPI_PWD'.format(get_python_executable()))
