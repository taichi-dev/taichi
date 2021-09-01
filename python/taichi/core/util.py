import ctypes
import os
import shutil
import sys

from colorama import Back, Fore, Style
from taichi.core import settings

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


def import_ti_core():
    global ti_core
    if settings.get_os_name() != 'win':
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
            if settings.get_os_name() == 'win':
                e.msg += '\nConsider installing Microsoft Visual C++ Redistributable: https://aka.ms/vs/16/release/vc_redist.x64.exe'
            elif settings.get_os_name() == 'linux':
                e.msg += '\nConsider installing libtinfo5: sudo apt-get install libtinfo5'
        raise e from None
    ti_core = core
    if settings.get_os_name() != 'win':
        sys.setdlopenflags(old_flags)
    lib_dir = os.path.join(package_root(), 'lib')
    core.set_lib_dir(locale_encode(lib_dir))


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


def get_core_shared_object():
    directory = os.path.join(package_root(), 'lib')
    return os.path.join(directory, 'libtaichi_core.so')


def print_red_bold(*args, **kwargs):
    print(Fore.RED + Style.BRIGHT, end='')
    print(*args, **kwargs)
    print(Style.RESET_ALL, end='')


def build():
    tmp_cwd = os.getcwd()
    bin_dir = settings.get_build_directory()

    try:
        os.mkdir(bin_dir)
    except:
        pass
    os.chdir(bin_dir)

    import multiprocessing
    print('Building taichi...')
    num_make_threads = min(20, multiprocessing.cpu_count())
    if settings.get_os_name() == 'win':
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


def get_unique_task_id():
    import datetime
    import random
    return datetime.datetime.now().strftime('task-%Y-%m-%d-%H-%M-%S-r') + (
        '%05d' % random.randint(0, 10000))


sys.path.append(os.path.join(package_root(), 'lib'))
if settings.get_os_name() != 'win':
    link_src = os.path.join(package_root(), 'lib', 'taichi_core.so')
    link_dst = os.path.join(package_root(), 'lib', 'libtaichi_core.so')
    # For llvm jit to find the runtime symbols
    if not os.path.exists(link_dst):
        os.symlink(link_src, link_dst)
import_ti_core()

ti_core.set_python_package_dir(package_root())
os.makedirs(ti_core.get_repo_dir(), exist_ok=True)

log_level = os.environ.get('TI_LOG_LEVEL', '')
if log_level:
    ti_core.set_logging_level(log_level)


def get_dll_name(name):
    if settings.get_os_name() == 'linux':
        return 'libtaichi_%s.so' % name
    elif settings.get_os_name() == 'osx':
        return 'libtaichi_%s.dylib' % name
    elif settings.get_os_name() == 'win':
        return 'taichi_%s.dll' % name
    else:
        raise Exception(f"Unknown OS: {settings.get_os_name()}")


def load_module(name, verbose=True):
    if verbose:
        print('Loading module', name)
    try:
        if settings.get_os_name() == 'osx':
            mode = ctypes.RTLD_LOCAL
        else:
            mode = ctypes.RTLD_GLOBAL
        if '.so' in name:
            ctypes.PyDLL(name, mode=mode)
        else:
            ctypes.PyDLL(os.path.join(settings.get_repo_directory(), 'build',
                                      get_dll_name(name)),
                         mode=mode)
    except Exception as e:
        print(Fore.YELLOW +
              "Warning: module [{}] loading failed: {}".format(name, e) +
              Style.RESET_ALL)


def at_startup():
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
    header = '[Taichi] '
    header += f'version {ti_core.get_version_string()}, '

    llvm_version = ti_core.get_llvm_version_string()
    header += f'llvm {llvm_version}, '

    commit_hash = ti_core.get_commit_hash()
    commit_hash = commit_hash[:8]
    header += f'commit {commit_hash}, '

    header += f'{settings.get_os_name()}, '

    py_ver = '.'.join(str(x) for x in sys.version_info[:3])
    header += f'python {py_ver}'

    print(header)


_print_taichi_header()

__all__ = [
    'ti_core',
    'build',
    'load_module',
    'start_memory_monitoring',
    'package_root',
    'require_version',
]
