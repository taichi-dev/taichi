import atexit
import os
import shutil
import sys
import ctypes

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
  print("\nPlease restart with python3. \n(Taichi supports Python 3.6+)\n")
  print("Current version:", sys.version_info)
  exit(-1)
  
tc_core = None

def in_docker():
  if os.environ.get("TI_IN_DOCKER", "") == "":
    return False
  else:
    return True

def import_tc_core():
  global tc_core
  old_flags = sys.getdlopenflags()
  sys.setdlopenflags(258) # 258 = RTLD_NOW | RTLD_GLOBAL
  # We
  import taichi_core as core
  tc_core = core
  sys.setdlopenflags(old_flags)
  core.set_lib_dir(os.path.join(package_root(), 'lib'))
  
def is_ci():
  return os.environ.get('TC_CI', '') == '1'

def package_root():
  return os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')

def is_release():
  return os.environ.get('TAICHI_REPO_DIR', '') == ''

from colorama import Fore, Back, Style

def get_core_shared_object():
  if is_release():
    directory = os.path.join(package_root(), 'lib')
  else:
    directory = get_bin_directory()
  return os.path.join(directory, 'libtaichi_core.so')

def get_repo():
  from git import Repo
  import taichi as tc
  repo = Repo(tc.get_repo_directory())
  return repo


def print_red_bold(*args, **kwargs):
  print(Fore.RED + Style.BRIGHT, end='')
  print(*args, **kwargs)
  print(Style.RESET_ALL, end='')


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

if is_release():
  print("[Release mode]")
  sys.path.append(os.path.join(package_root(), 'lib'))
  link_src = os.path.join(package_root(), 'lib', 'taichi_core.so')
  link_dst = os.path.join(package_root(), 'lib', 'libtaichi_core.so')
  if not os.path.exists(link_dst):
    os.symlink(link_src, link_dst)
  import_tc_core()
  
  # For llvm jit to find the runtime symbols
  dll = ctypes.CDLL(get_core_shared_object(), mode=ctypes.RTLD_GLOBAL)

  tc_core.set_python_package_dir(package_root())
  os.makedirs(tc_core.get_repo_dir(), exist_ok=True)
else:
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
    if 'LD_LIBRARY_PATH' in os.environ:
      os.environ['LD_LIBRARY_PATH'] += ':/usr/lib64/'
    else:
      os.environ['LD_LIBRARY_PATH'] = '/usr/lib64/'
    if not os.path.exists(os.path.join(bin_dir, 'libtaichi_core.so')):
      build()
    tmp_cwd = os.getcwd()
    os.chdir(bin_dir)
    sys.path.append(bin_dir)
    # https://stackoverflow.com/questions/3855004/overwriting-library-file-causes-segmentation-fault
    if os.path.exists('taichi_core.so'):
      try:
        os.unlink('taichi_core.so')
      except:
        print('Warning: taichi_core.so already removed. This may be caused by'
              'simultaneously starting two taichi instances.')
        pass
    shutil.copy('libtaichi_core.so', 'taichi_core.so')
    try:
      import_tc_core()
    except Exception as e:
      from colorama import Fore, Back, Style
      print_red_bold("Taichi core import failed: ", end='')
      print(e)
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
  elif get_os_name() == 'osx':
    return 'libtaichi_%s.dylib' % name
  elif get_os_name() == 'win':
    return 'taichi_%s.dll' % name
  else:
    assert False, "Unknown OS"

def load_module(name, verbose=True):
  if verbose:
    print('Loading module', name)
  try:
    if get_os_name() == 'osx':
      mode = ctypes.RTLD_LOCAL
    else:
      mode = ctypes.RTLD_GLOBAL
    if '.so' in name:
      ctypes.PyDLL(name, mode=mode)
    else:
      ctypes.PyDLL(
        os.path.join(get_repo_directory(), 'build',
                     get_dll_name(name)), mode=mode)
  except Exception as e:
    print(Fore.YELLOW + "Warning: module [{}] loading failed: {}".format(
      name, e) + Style.RESET_ALL)

def at_startup():
  if not is_release():
    output_dir = get_output_directory()
    if not os.path.exists(output_dir):
      print('Making output directory')
      os.mkdir(output_dir)

  # Load modules
  # load_module('lang_core')

  tc_core.set_core_state_python_imported(True)


def start_memory_monitoring(output_fn, pid=-1, interval=1):
  # removing dependency on psutil
  return
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

at_startup()

if not is_release():
  tc_core.set_core_trigger_gdb_when_crash(True)
