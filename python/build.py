import os
import platform
import sys

if os.environ.get('PYPI_PWD', '') is '':
  assert False, "Missing environment variable PYPI_PWD"


def get_python_executable():
  return sys.executable.replace('\\', '/')


if platform.system() == 'Linux':
  if os.environ['CXX'] != 'clang++-7':
    print('Only the wheel with clang-7 will be released to PyPI.')
    sys.exit(0)

os.makedirs('taichi/lib', exist_ok=True)
os.system('cp -r ../lang/include taichi/')
os.system('cp ../build/libtaichi_core.so taichi/lib/taichi_core.so')
os.system('cp ../build/libtaichi_core.dylib taichi/lib/taichi_core.so')
os.system('rm dist/*.whl')
os.system('{} -m pip install --user --upgrade twine setuptools wheel'.format(
  get_python_executable()))

if platform.system() == 'Linux':
  os.system('{} setup.py bdist_wheel -p manylinux1_x86_64'.format(
    get_python_executable()))
else:
  os.system('{} setup.py bdist_wheel'.format(get_python_executable()))


mode = ''
if len(sys.argv) > 1:
  mode = sys.argv[1]
  
if mode == 'upload':
  os.system('{} -m twine upload dist/* --verbose -u yuanming-hu -p $PYPI_PWD'.format(get_python_executable()))
elif mode == 'test':
  print('Uninstalling old taichi package...')
  os.system('pip3 uninstall taichi-nightly')
  dists = os.listdir('dist')
  assert len(dists) == 1
  dist = dists[0]
  print('Installing ', dist)
  os.system('pip3 install dist/{} --user'.format(dist))
  print('Entering test environment...')
  os.system('PYTHONPATH= TAICHI_REPO_DIR= bash')
elif mode == '':
  pass
else:
  print("Unknown mode: ", mode)
  exit(-1)