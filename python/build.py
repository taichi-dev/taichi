import argparse
import os
import platform
import sys
import shutil

if len(sys.argv) != 4:
  print("Usage: python3 build.py [version: 0.0.50] [gpu=0] [mode=upload/test]")
  exit(-1)
  
version = sys.argv[1]
gpu = int(sys.argv[2])
assert gpu in [0, 1]
mode = sys.argv[3]

def get_os_name():
  name = platform.platform()
  if name.lower().startswith('darwin'):
    return 'osx'
  elif name.lower().startswith('windows'):
    return 'win'
  elif name.lower().startswith('linux'):
    return 'linux'
  assert False, "Unknown platform name %s" % name

if os.environ.get('PYPI_PWD', '') is '':
  assert False, "Missing environment variable PYPI_PWD"

def get_python_executable():
  return sys.executable.replace('\\', '/')


if platform.system() == 'Linux':
  if os.environ['CXX'] not in ['clang++-7', 'clang++']:
    print('Only the wheel with clang-7 will be released to PyPI.')
    sys.exit(0)

with open('setup.temp.py') as fin:
  with open('setup.py', 'w') as fout:
    if gpu:
      project_name = 'taichi-gpu-nightly'
    else:
      project_name = 'taichi-nightly'
    print("project_name = '{}'".format(project_name), file=fout)
    print("version = '{}'".format(version), file=fout)
    for l in fin:
      print(l, file=fout, end='')

os.makedirs('taichi/lib', exist_ok=True)
shutil.rmtree('build')
shutil.rmtree('dist')
os.system('cp -r ../lang/include taichi/')
build_dir = '../build'

if get_os_name() == 'linux':
  shutil.copy('../build/libtaichi_core.so', 'taichi/lib/taichi_core.so')
elif get_os_name() == 'osx':
  shutil.copy('../build/libtaichi_core.dylib', 'taichi/lib/taichi_core.so')
else:
  print('not implemented')
  exit(-1)

os.system('{} -m pip install --user --upgrade twine setuptools wheel'.format(
  get_python_executable()))

if get_os_name() == 'linux':
  os.system('{} setup.py bdist_wheel -p manylinux1_x86_64'.format(
    get_python_executable()))
else:
  os.system('{} setup.py bdist_wheel'.format(get_python_executable()))

if mode == 'upload':
  os.system('{} -m twine upload dist/* --verbose -u yuanming-hu -p $PYPI_PWD'.format(get_python_executable()))
elif mode == 'test':
  print('Uninstalling old taichi package...')
  os.system('pip3 uninstall taichi-nightly taichi-gpu-nightly')
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
