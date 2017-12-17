import atexit
import os
import shutil
import sys
import ctypes
import subprocess

try:
  import pip
except Exception as e:
  print(e)
  print('  Please install pip3.')
  print('    [Ubuntu] sudo apt-get install python-pip3')
  print('    [Arch Linux] sudo pacman -S python-pip')
  exit(-1)

required_packages = [
    'numpy', ('Pillow', 'PIL'), 'scipy', 'pybind11', 'flask', 'flask_cors',
    ('GitPython', 'git'), 'yapf'
]


def format():
  import os
  import sys
  import taichi as tc
  from git import Repo
  from yapf.yapflib.yapf_api import FormatFile

  repo = Repo(tc.get_repo_directory())

  print('* Formatting code', end='')
  for item in repo.index.diff('HEAD'):
    fn = os.path.join(tc.get_repo_directory(), item.a_path)
    print(end='.')
    if fn.endswith('.py'):
      FormatFile(
          fn,
          in_place=True,
          style_config=os.path.join(tc.get_repo_directory(), '.style.yapf'))
    if fn.endswith('.cpp'):
      os.system('clang-format -i -style=file {}'.format(fn))

  print('* Done!')


def install_package(pkg):
  pip.main(['install', '--user', pkg])


def check_for_packages():
  for pkg in required_packages:
    if isinstance(pkg, tuple):
      import_name = pkg[1]
      pkg = pkg[0]
    else:
      import_name = pkg

    try:
      exec('import {}'.format(import_name))
    except ImportError as e:
      print("Installing package: ", pkg)
      install_package(pkg)


check_for_packages()

from taichi.misc.settings import get_output_directory, get_bin_directory, get_root_directory
from taichi.misc.util import get_os_name, get_unique_task_id

CREATE_SAND_BOX_ON_WINDOWS = True


def build():
  tmp_cwd = os.getcwd()
  bin_dir = get_bin_directory()

  try:
    os.mkdir(bin_dir)
  except:
    pass
  os.chdir(bin_dir)

  if os.environ.get('TC_CI', '') == '1':
    print('  Note: building for CI. SIMD disabled.')
    cmake_ret = os.system('cmake .. -DTC_DISABLE_SIMD:BOOL=1')
  else:
    cmake_ret = os.system('cmake ..')
  if cmake_ret != 0:
    print('  Error: CMake failed.')
    exit(-1)

  make_ret = os.system('make -j8')
  if make_ret != 0:
    print('  Error: Build failed.')
    exit(-1)

  os.chdir(tmp_cwd)


if get_os_name() == 'osx':
  bin_dir = get_bin_directory()
  if not os.path.exists(os.path.join(bin_dir, 'libtaichi_core.dylib')):
    build()
  tmp_cwd = os.getcwd()
  os.chdir(bin_dir)
  shutil.copy('libtaichi_core.dylib', 'taichi_core.so')
  sys.path.append(bin_dir)
  import taichi_core as tc_core
  os.chdir(tmp_cwd)
elif get_os_name() == 'linux':
  bin_dir = get_bin_directory()
  os.environ['LD_LIBRARY_PATH'] = '/usr/lib64/'
  if not os.path.exists(os.path.join(bin_dir, 'libtaichi_core.so')):
    build()
  tmp_cwd = os.getcwd()
  os.chdir(bin_dir)
  sys.path.append(bin_dir)
  # https://stackoverflow.com/questions/3855004/overwriting-library-file-causes-segmentation-fault
  if os.path.exists('taichi_core.so'):
    os.unlink('taichi_core.so')
  shutil.copy('libtaichi_core.so', 'taichi_core.so')
  try:
    import taichi_core as tc_core
  except Exception as e:
    print()
    print("\033[91m*Please make sure you are using python3 "
          "instead of python2.\033[0m")
    print()
    print(e)

  os.chdir(tmp_cwd)
elif get_os_name() == 'win':
  bin_dir = get_bin_directory()
  dll_path = os.path.join(bin_dir, 'Release', 'taichi_core.dll')
  if not os.path.exists(dll_path):
    dll_path = os.path.join(bin_dir, 'taichi_core.dll')
    print(dll_path)
    if not os.path.exists(dll_path):
      dll_path = os.path.join(bin_dir, 'libtaichi_core.dll')
      if not os.path.exists(dll_path):
        assert False, "Library taichi_core doesn't exist."

  # The problem here is, on windows, when an dll/pyd is loaded, we can not write to it any more...

  # Ridiculous...
  old_wd = os.getcwd()
  os.chdir(bin_dir)

  if CREATE_SAND_BOX_ON_WINDOWS:
    # So let's just create a sandbox for separated core lib development and loading
    dir = os.path.join(get_output_directory(), 'tmp', get_unique_task_id())
    os.makedirs(dir)
    '''
        for fn in os.listdir(bin_dir):
            if fn.endswith('.dll') and fn != 'taichi_core.dll':
                print dir + fn, bin_dir + fn
                # Why can we create symbolic links....
                # if not ctypes.windll.kernel32.CreateSymbolicLinkW(bin_dir + fn, dir + fn, 0):
                #    raise OSError
                shutil.copy(bin_dir + fn, dir + fn)
        '''
    shutil.copy(dll_path, os.path.join(dir, 'taichi_core.pyd'))
    sys.path.append(dir)
  else:
    shutil.copy(dll_path, os.path.join(bin_dir, 'taichi_core.pyd'))
    sys.path.append(bin_dir)
  import taichi_core as tc_core

  os.chdir(old_wd)


def get_dll_name(name):
  if get_os_name() == 'linux':
    return 'libtaichi_%s.so' % name
  else:
    assert False


def at_startup():
  assert os.path.exists(get_root_directory(
  )), 'Please make sure $TAICHI_ROOT_DIR [' + get_root_directory() + '] exists.'
  output_dir = get_output_directory()
  if not os.path.exists(output_dir):
    print('Making output directory')
    os.mkdir(output_dir)

  # Load modules
  f = open(os.path.join(get_root_directory(), 'taichi', 'modules.txt'), 'r')
  modules = f.readline().strip().split(';')
  for module in modules:
    if module != '':
      print('Loading module', module)
      try:
        ctypes.PyDLL(
            os.path.join(get_root_directory(), 'taichi', 'build',
                         get_dll_name(module)))
      except Exception as e:
        print(e)
        print("Warning: module", module, "loading failed!")

  tc_core.set_core_state_python_imported(True)
  f.close()


at_startup()


@atexit.register
def clean_libs():
  pass
