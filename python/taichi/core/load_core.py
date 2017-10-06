from __future__ import print_function
import atexit
import os
import shutil
import sys
import ctypes

from taichi.misc.settings import get_output_directory, get_bin_directory, get_root_directory
from taichi.misc.util import get_os_name, get_unique_task_id

CREATE_SAND_BOX_ON_WINDOWS = True

if get_os_name() == 'osx':
  bin_dir = get_bin_directory()
  if os.path.exists(os.path.join(bin_dir, 'libtaichi_core.dylib')):
    tmp_cwd = os.getcwd()
    os.chdir(bin_dir)
    shutil.copy('libtaichi_core.dylib', 'taichi_core.so')
    sys.path.append(bin_dir)
    import taichi_core as tc_core

    os.chdir(tmp_cwd)
  else:
    assert False, "Library taichi_core doesn't exist."
elif get_os_name() == 'linux':
  bin_dir = get_bin_directory()
  os.environ['LD_LIBRARY_PATH'] = '/usr/lib64/'
  if os.path.exists(os.path.join(bin_dir, 'libtaichi_core.so')):
    tmp_cwd = os.getcwd()
    os.chdir(bin_dir)
    sys.path.append(bin_dir)
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
  else:
    assert False, "Library taichi_core doesn't exist."
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
  modules = f.readline().split(';')
  for module in modules:
    print('Loading module', module)
    ctypes.PyDLL(
        os.path.join(get_root_directory(), 'taichi', 'build',
                     get_dll_name(module)))

  f.close()


at_startup()


@atexit.register
def clean_libs():
  pass
