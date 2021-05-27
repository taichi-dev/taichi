import ctypes
import os
import re
import shutil
import sys
from pathlib import Path

from colorama import Back, Fore, Style

from .settings import *

if sys.version_info[0] < 3 or sys.version_info[1] <= 5:
    raise RuntimeError(
        "\nPlease restart with Python 3.6+\n" + "Current Python version:",
        sys.version_info)

ti_core = None


def in_docker():
    if os.environ.get("TI_IN_DOCKER", "") == "":
        return False
    else:
        return True


def import_ti_core(tmp_dir=None):
    global ti_core
    if get_os_name() != 'win':
        old_flags = sys.getdlopenflags()
        sys.setdlopenflags(2 | 8)  # RTLD_NOW | RTLD_DEEPBIND
    else:
        pyddir = os.path.join(package_root(), 'lib')
        os.environ['PATH'] += ';' + pyddir
    try:
        import taichi_core as core
    except Exception as e:
        if isinstance(e, ImportError):
            print(
                Fore.YELLOW + "Share object taichi_core import failed, "
                "check this page for possible solutions:\n"
                "https://taichi.readthedocs.io/en/stable/install.html#troubleshooting"
                + Fore.RESET)
            if get_os_name() == 'win':
                e.msg += '\nConsider installing Microsoft Visual C++ Redistributable: https://aka.ms/vs/16/release/vc_redist.x64.exe'
            elif get_os_name() == 'linux':
                e.msg += '\nConsider installing libtinfo5: sudo apt-get install libtinfo5'
        raise e from None
    ti_core = core
    if get_os_name() != 'win':
        sys.setdlopenflags(old_flags)
    lib_dir = os.path.join(package_root(), 'lib')
    core.set_lib_dir(locale_encode(lib_dir))
    if tmp_dir is not None:
        core.set_tmp_dir(locale_encode(tmp_dir))


def locale_encode(path):
    try:
        import locale
        return path.encode(locale.getdefaultlocale()[1])
    except:
        try:
            import sys
            return path.encode(sys.getfilesystemencoding())
        except:
            try:
                return path.encode()
            except:
                return path


def is_ci():
    return os.environ.get('TI_CI', '') == '1'


def package_root():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')


def is_release():
    return os.environ.get('TAICHI_REPO_DIR', '') == ''


def get_core_shared_object():
    if is_release():
        directory = os.path.join(package_root(), 'lib')
    else:
        directory = get_bin_directory()
    return os.path.join(directory, 'libtaichi_core.so')


def get_repo():
    from git import Repo
    repo = Repo(get_repo_directory())
    return repo


def print_red_bold(*args, **kwargs):
    print(Fore.RED + Style.BRIGHT, end='')
    print(*args, **kwargs)
    print(Style.RESET_ALL, end='')


create_sand_box_on_windows = True


def build():
    tmp_cwd = os.getcwd()
    bin_dir = get_build_directory()

    try:
        os.mkdir(bin_dir)
    except:
        pass
    os.chdir(bin_dir)

    import multiprocessing
    print('Building taichi...')
    num_make_threads = min(20, multiprocessing.cpu_count())
    if get_os_name() == 'win':
        make_ret = os.system(
            "msbuild /p:Configuration=Release /p:Platform=x64 /m taichi.sln")
    else:
        make_ret = os.system('make -j {}'.format(num_make_threads))
    if make_ret != 0:
        print('  Error: Build failed.')
        exit(-1)

    os.chdir(tmp_cwd)


def check_exists(src):
    if not os.path.exists(src):
        raise FileNotFoundError(
            f'File "{src}" not exist. Installation corrupted or build incomplete?'
        )


def prepare_sandbox():
    '''
    Returns a temporary directory, which will be automatically deleted on exit.
    It may contain the taichi_core shared object or some misc. files.
    '''
    import atexit
    import shutil
    from tempfile import mkdtemp
    tmp_dir = mkdtemp(prefix='taichi-')
    atexit.register(shutil.rmtree, tmp_dir)
    print(f'[Taichi] preparing sandbox at {tmp_dir}')
    os.mkdir(os.path.join(tmp_dir, 'runtime/'))
    return tmp_dir


def get_unique_task_id():
    import datetime
    import random
    return datetime.datetime.now().strftime('task-%Y-%m-%d-%H-%M-%S-r') + (
        '%05d' % random.randint(0, 10000))


if is_release():
    print("[Taichi] mode=release")
    sys.path.append(os.path.join(package_root(), 'lib'))
    if get_os_name() != 'win':
        link_src = os.path.join(package_root(), 'lib', 'taichi_core.so')
        link_dst = os.path.join(package_root(), 'lib', 'libtaichi_core.so')
        # For llvm jit to find the runtime symbols
        if not os.path.exists(link_dst):
            os.symlink(link_src, link_dst)
    import_ti_core()
    if get_os_name() != 'win':
        dll = ctypes.CDLL(get_core_shared_object(), mode=ctypes.RTLD_LOCAL)
        # The C backend needs a temporary directory for the generated .c and compiled .so files:
        ti_core.set_tmp_dir(locale_encode(prepare_sandbox(
        )))  # TODO: always allocate a tmp_dir for all situations

    ti_core.set_python_package_dir(package_root())
    os.makedirs(ti_core.get_repo_dir(), exist_ok=True)
else:
    print("[Taichi] mode=development")
    if get_os_name() == 'osx':
        bin_dir = get_bin_directory()
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = get_runtime_directory()
        lib_path = os.path.join(bin_dir, 'libtaichi_core.dylib')
        tmp_cwd = os.getcwd()
        tmp_dir = prepare_sandbox()
        check_exists(lib_path)
        shutil.copy(lib_path, os.path.join(tmp_dir, 'taichi_core.so'))
        os.chdir(tmp_dir)
        sys.path.append(tmp_dir)
        import taichi_core as ti_core
        os.chdir(tmp_cwd)

    # TODO: unify importing infrastructure:
    elif get_os_name() == 'linux':
        bin_dir = get_bin_directory()
        if 'LD_LIBRARY_PATH' in os.environ:
            os.environ['LD_LIBRARY_PATH'] += ':/usr/lib64/'
        else:
            os.environ['LD_LIBRARY_PATH'] = '/usr/lib64/'
        lib_path = os.path.join(bin_dir, 'libtaichi_core.so')
        check_exists(lib_path)
        tmp_cwd = os.getcwd()
        tmp_dir = prepare_sandbox()
        check_exists(lib_path)
        shutil.copy(lib_path, os.path.join(tmp_dir, 'taichi_core.so'))
        os.chdir(tmp_dir)
        sys.path.append(tmp_dir)
        try:
            import_ti_core(tmp_dir)
        except Exception as e:
            print_red_bold("Taichi core import failed: ", end='')
            print(e)
            print(
                Fore.YELLOW + "check this page for possible solutions:\n"
                "https://taichi.readthedocs.io/en/stable/install.html#troubleshooting"
                + Fore.RESET)
            raise e from None
        os.chdir(tmp_cwd)

    elif get_os_name() == 'win':
        bin_dir = get_bin_directory()
        dll_path_invalid = os.path.join(bin_dir, 'libtaichi_core.dll')
        assert not os.path.exists(dll_path_invalid)

        possible_folders = ['Debug', 'RelWithDebInfo', 'Release']
        detected_dlls = []
        for folder in possible_folders:
            dll_path = os.path.join(bin_dir, folder, 'taichi_core.dll')
            if os.path.exists(dll_path):
                detected_dlls.append(dll_path)

        if len(detected_dlls) == 0:
            raise FileNotFoundError(
                f'Cannot find Taichi core dll under {get_bin_directory()}/{possible_folders}'
            )
        elif len(detected_dlls) != 1:
            print('Warning: multiple Taichi core dlls found:')
            for dll in detected_dlls:
                print(' ', dll)
            print(f'Using {detected_dlls[0]}')

        dll_path = detected_dlls[0]

        # On windows when an dll/pyd is loaded, we cannot write to it any more
        old_wd = os.getcwd()
        os.chdir(bin_dir)

        if create_sand_box_on_windows:
            # Create a sandbox for separated core lib development and loading
            folder = os.path.join(get_output_directory(), 'tmp',
                                  get_unique_task_id())

            lib_dir = os.path.join(get_repo_directory(), 'external', 'lib')
            os.environ['PATH'] += ';' + lib_dir

            os.makedirs(folder)
            shutil.copy(dll_path, os.path.join(folder, 'taichi_core.pyd'))
            os.environ['PATH'] += ';' + folder
            sys.path.append(folder)
        else:
            shutil.copy(dll_path, os.path.join(bin_dir, 'taichi_core.pyd'))
            sys.path.append(bin_dir)
        try:
            import taichi_core as ti_core
        except Exception as e:
            print(e)
            print()
            print(
                'Hint: please make sure the major and minor versions of the Python executable is correct.'
            )
            print()
            raise e

        os.chdir(old_wd)

log_level = os.environ.get('TI_LOG_LEVEL', '')
if log_level:
    ti_core.set_logging_level(log_level)


def get_dll_name(name):
    if get_os_name() == 'linux':
        return 'libtaichi_%s.so' % name
    elif get_os_name() == 'osx':
        return 'libtaichi_%s.dylib' % name
    elif get_os_name() == 'win':
        return 'taichi_%s.dll' % name
    else:
        raise Exception(f"Unknown OS: {get_os_name()}")


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
            ctypes.PyDLL(os.path.join(get_repo_directory(), 'build',
                                      get_dll_name(name)),
                         mode=mode)
    except Exception as e:
        print(Fore.YELLOW +
              "Warning: module [{}] loading failed: {}".format(name, e) +
              Style.RESET_ALL)


def at_startup():
    if not is_release():
        output_dir = get_output_directory()
        if not os.path.exists(output_dir):
            print('Making output directory')
            os.mkdir(output_dir)

    ti_core.set_core_state_python_imported(True)


def start_memory_monitoring(output_fn, pid=-1, interval=1):
    # removing dependency on psutil
    return
    import os
    import time

    import psutil
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


def require_version(major, minor=None, patch=None):
    versions = [
        int(ti_core.get_version_major()),
        int(ti_core.get_version_minor()),
        int(ti_core.get_version_patch()),
    ]
    match = major == versions[0] and (
        minor < versions[1] or minor == versions[1] and patch <= versions[2])
    if match:
        return
    else:
        print("Taichi version mismatch. required >= {}.{}.{}".format(
            major, minor, patch))
        print("Installed =", ti_core.get_version_string())
        raise Exception("Taichi version mismatch")


at_startup()


def _print_taichi_header():
    dev_mode = not is_release()

    header = '[Taichi] '
    if dev_mode:
        header += '<dev mode>, '
    else:
        header += f'version {ti_core.get_version_string()}, '

    llvm_version = ti_core.get_llvm_version_string()
    header += f'llvm {llvm_version}, '

    commit_hash = ti_core.get_commit_hash()
    commit_hash = commit_hash[:8]
    header += f'commit {commit_hash}, '

    header += f'{get_os_name()}, '

    py_ver = '.'.join(str(x) for x in sys.version_info[:3])
    header += f'python {py_ver}'

    print(header)


_print_taichi_header()

__all__ = [
    'ti_core',
    'build',
    'load_module',
    'start_memory_monitoring',
    'is_release',
    'package_root',
    'require_version',
]
