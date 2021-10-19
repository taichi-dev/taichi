import datetime
import os
import platform
import random
import sys

from colorama import Fore, Style

if sys.version_info[0] < 3 or sys.version_info[1] <= 5:
    raise RuntimeError(
        "\nPlease restart with Python 3.6+\n" + "Current Python version:",
        sys.version_info)


def in_docker():
    if os.environ.get("TI_IN_DOCKER", "") == "":
        return False
    else:
        return True


def get_os_name():
    name = platform.platform()
    # in python 3.8, platform.platform() uses mac_ver() on macOS
    # it will return 'macOS-XXXX' instead of 'Darwin-XXXX'
    if name.lower().startswith('darwin') or name.lower().startswith('macos'):
        return 'osx'
    elif name.lower().startswith('windows'):
        return 'win'
    elif name.lower().startswith('linux'):
        return 'linux'
    assert False, "Unknown platform name %s" % name


def import_ti_core():
    if get_os_name() != 'win':
        old_flags = sys.getdlopenflags()
        sys.setdlopenflags(2 | 8)  # RTLD_NOW | RTLD_DEEPBIND
    else:
        pyddir = os.path.join(package_root(), 'lib')
        os.environ['PATH'] += os.pathsep + pyddir
    try:
        from taichi.lib import taichi_core as core  # pylint: disable=C0415
    except Exception as e:
        if isinstance(e, ImportError):
            print(Fore.YELLOW + "Share object taichi_core import failed, "
                  "check this page for possible solutions:\n"
                  "https://docs.taichi.graphics/lang/articles/misc/install" +
                  Fore.RESET)
            if get_os_name() == 'win':
                e.msg += '\nConsider installing Microsoft Visual C++ Redistributable: https://aka.ms/vs/16/release/vc_redist.x64.exe'
        raise e from None

    if get_os_name() != 'win':
        sys.setdlopenflags(old_flags)
    lib_dir = os.path.join(package_root(), 'lib')
    core.set_lib_dir(locale_encode(lib_dir))
    return core


def locale_encode(path):
    try:
        import locale  # pylint: disable=C0415
        return path.encode(locale.getdefaultlocale()[1])
    except (UnicodeEncodeError, TypeError):
        try:
            return path.encode(sys.getfilesystemencoding())
        except UnicodeEncodeError:
            try:
                return path.encode()
            except UnicodeEncodeError:
                return path


def is_ci():
    return os.environ.get('TI_CI', '') == '1'


def package_root():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def get_core_shared_object():
    directory = os.path.join(package_root(), 'lib')
    return os.path.join(directory, 'libtaichi_core.so')


def print_red_bold(*args, **kwargs):
    print(Fore.RED + Style.BRIGHT, end='')
    print(*args, **kwargs)
    print(Style.RESET_ALL, end='')


def check_exists(src):
    if not os.path.exists(src):
        raise FileNotFoundError(
            f'File "{src}" not exist. Installation corrupted or build incomplete?'
        )


def get_unique_task_id():
    return datetime.datetime.now().strftime('task-%Y-%m-%d-%H-%M-%S-r') + (
        '%05d' % random.randint(0, 10000))


ti_core = import_ti_core()

ti_core.set_python_package_dir(package_root())
os.makedirs(ti_core.get_repo_dir(), exist_ok=True)

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


def at_startup():
    ti_core.set_core_state_python_imported(True)


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

    header += f'{get_os_name()}, '

    py_ver = '.'.join(str(x) for x in sys.version_info[:3])
    header += f'python {py_ver}'

    print(header)


_print_taichi_header()

__all__ = [
    'ti_core',
    'get_os_name',
    'package_root',
    'require_version',
]
