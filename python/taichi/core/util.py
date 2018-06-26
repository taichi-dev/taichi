from __future__ import print_function
import atexit
import os
import shutil
import sys
import ctypes
import subprocess

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
  print("\nPlease restart with python3. \n(Taichi supports Python 3.5+)\n")
  print("Current version:", sys.version_info)
  exit(-1)

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
    ('GitPython', 'git'), 'yapf', 'colorama', 'pyglet', 'PyQt5', 'psutil', 'requests'
]


def is_ci():
  return os.environ.get('TC_CI', '') == '1'

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
      print("Installing package:", pkg)
      install_package(pkg)
      print('Checking installation of "{}"'.format(import_name))
      exec('import {}'.format(import_name))


check_for_packages()
from colorama import Fore, Back, Style


def get_repo():
  from git import Repo
  import taichi as tc
  repo = Repo(tc.get_repo_directory())
  return repo


def print_red_bold(*args, **kwargs):
  print(Fore.RED + Style.BRIGHT, end='')
  print(*args, **kwargs)
  print(Style.RESET_ALL, end='')

def get_projects(active=True):
  import taichi as tc
  ret = []
  for proj in os.listdir(tc.get_project_directory()):
    if proj not in ['examples', 'toys'] and os.path.isdir(tc.get_project_directory(proj)):
      activation = not proj.startswith('_')
      if not activation:
        proj = proj[1:]
      if activation == active:
        ret.append(proj)
  return ret

def update(include_projects=False):
  import git
  import taichi as tc

  g = git.cmd.Git(tc.get_repo_directory())
  print(Fore.GREEN + "Updating [taichi]..." + Style.RESET_ALL)
  try:
    g.pull('--rebase')
    print(Fore.GREEN + "   ...Done" + Style.RESET_ALL)
  except git.exc.GitCommandError as e:
    if 'You have unstaged changes' in e.stderr:
      print_red_bold("   You have unstaged changes in the Taichi main repo. Please commit your changes first.")
      exit(-1)

  for proj in os.listdir(tc.get_project_directory()):
    if proj in ['examples', 'toys'] or proj.startswith('_') or not os.path.isdir(
        tc.get_project_directory(proj)):
      continue
    print(
        Fore.GREEN + "Updating project [{}]...".format(proj) + Style.RESET_ALL)
    g = git.cmd.Git(os.path.join(tc.get_project_directory(proj)))
    try:
      g.pull('--rebase')
      print(Fore.GREEN + "   ...Done" + Style.RESET_ALL)
    except git.exc.GitCommandError as e:
      if 'You have unstaged changes' in e.stderr:
        print_red_bold("   You have committed changes in project [{}]. Please commit your changes first.".format(proj))
        exit(-1)


def format():
  import os
  import taichi as tc
  from yapf.yapflib.yapf_api import FormatFile
  repo = get_repo()

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
    repo.git.add(item.a_path)

  print('* Done!')


from taichi.misc.settings import get_output_directory, get_build_directory, get_bin_directory, get_repo_directory, get_runtime_directory
from taichi.misc.util import get_os_name, get_unique_task_id

CREATE_SAND_BOX_ON_WINDOWS = True


def build():
  tmp_cwd = os.getcwd()
  bin_dir = get_build_directory()

  try:
    os.mkdir(bin_dir)
  except:
    pass
  os.chdir(bin_dir)

  flags = ' -DPYTHON_EXECUTABLE:FILEPATH="{}"'.format(sys.executable)

  print('Running cmake...')
  if is_ci():
    print('  Note: building for CI. SIMD disabled.')
    flags += ' -DTC_DISABLE_SIMD:BOOL=1'
  if get_os_name() == 'win':
    flags += ' -G "Visual Studio 15 Win64"'
  cmake_ret = os.system('cmake .. ' + flags)
  if cmake_ret != 0:
    print('  Error: CMake failed.')
    exit(-1)

  import multiprocessing
  print('Building taichi...')
  num_make_threads = min(20, multiprocessing.cpu_count())
  if get_os_name() == 'win':
    make_ret = os.system("msbuild /p:Configuration=Release /p:Platform=x64 /m taichi.sln")
  else:
    make_ret = os.system('make -j {}'.format(num_make_threads))
  if make_ret != 0:
    print('  Error: Build failed.')
    exit(-1)

  os.chdir(tmp_cwd)


if get_os_name() == 'osx':
  bin_dir = get_bin_directory()
  os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = get_runtime_directory()
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
    from colorama import Fore, Back, Style
    print(e)
    print()
    print(Fore.RED + "Please make sure you are using python3 "
          "instead of python2." + Style.RESET_ALL)
    print()
    exit(-1)

  os.chdir(tmp_cwd)
elif get_os_name() == 'win':
  bin_dir = get_bin_directory()
  dll_path1 = os.path.join(bin_dir, 'Release', 'taichi_core.dll')
  dll_path2 = os.path.join(bin_dir, 'libtaichi_core.dll')
  if not os.path.exists(dll_path1) and not os.path.exists(dll_path2):
    build()

  # On windows when an dll/pyd is loaded, we can not write to it any more
  old_wd = os.getcwd()
  os.chdir(bin_dir)

  if CREATE_SAND_BOX_ON_WINDOWS:
    # Create a sandbox for separated core lib development and loading
    dir = os.path.join(get_output_directory(), 'tmp', get_unique_task_id())

    lib_dir = os.path.join(get_repo_directory(), 'external', 'lib')
    os.environ['PATH'] += ';' + lib_dir

    os.makedirs(dir)
    if os.path.exists(dll_path1):
      shutil.copy(dll_path1, os.path.join(dir, 'taichi_core.pyd'))
    else:
      shutil.copy(dll_path2, os.path.join(dir, 'taichi_core.pyd'))
    os.environ['PATH'] += ';' + dir
    sys.path.append(dir)
  else:
    shutil.copy(dll_path, os.path.join(bin_dir, 'taichi_core.pyd'))
    sys.path.append(bin_dir)
  try:
    import taichi_core as tc_core
  except Exception as e:
    print(e)
    print()
    print('Is taichi\external\lib correctly set to branch msvc or mingw?')
    print()
    raise e

  os.chdir(old_wd)


def get_dll_name(name):
  if get_os_name() == 'linux':
    return 'libtaichi_%s.so' % name
  else:
    assert False

def load_module(name, verbose=True):
  if verbose:
    print('Loading module', name)
  try:
    if '.so' in name:
      ctypes.PyDLL(name)
    else:
      ctypes.PyDLL(
        os.path.join(get_repo_directory(), 'build',
                     get_dll_name(name)))
  except Exception as e:
    print(Fore.YELLOW + "Warning: module [{}] loading failed: {}".format(
      name, e) + Style.RESET_ALL)

def at_startup():
  assert os.path.exists(get_repo_directory(
  )), 'Please make sure $TAICHI_REPO_DIR [' + get_repo_directory() + '] exists.'
  output_dir = get_output_directory()
  if not os.path.exists(output_dir):
    print('Making output directory')
    os.mkdir(output_dir)

  # Load modules
  f = open(os.path.join(get_repo_directory(), 'modules.txt'), 'r')
  modules = f.readline().strip().split(';')
  for module in modules:
    if module != '':
      load_module(module)

  tc_core.set_core_state_python_imported(True)
  f.close()


at_startup()


def start_memory_monitoring(output_fn, pid=-1, interval=1):
  import os, psutil, time
  if pid == -1:
    pid = os.getpid()
  import multiprocessing
  def task():
    with open(output_fn, 'w') as f:
      process = psutil.Process(pid)
      while True:
        try:
          mem = process.memory_info().rss
        except:
          mem = -1
        time.sleep(interval)
        print(time.time(), mem, file=f)
        f.flush()
  proc = multiprocessing.Process(target=task, daemon=True)
  proc.start()


@atexit.register
def clean_libs():
  pass
