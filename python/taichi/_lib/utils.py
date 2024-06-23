import os
import platform
import re
import sys
import warnings

from colorama import Fore, Style

if sys.version_info[0] < 3 or sys.version_info[1] <= 5:
    raise RuntimeError(
        "\nPlease restart with Python 3.6+\n" + "Current Python version:",
        sys.version_info,
    )


def in_docker():
    if os.environ.get("TI_IN_DOCKER", "") == "":
        return False
    return True


def get_os_name():
    name = platform.platform()
    # in python 3.8, platform.platform() uses mac_ver() on macOS
    # it will return 'macOS-XXXX' instead of 'Darwin-XXXX'
    if name.lower().startswith("darwin") or name.lower().startswith("macos"):
        return "osx"
    if name.lower().startswith("windows"):
        return "win"
    if name.lower().startswith("linux"):
        return "linux"
    if "bsd" in name.lower():
        return "unix"
    assert False, f"Unknown platform name {name}"


def import_ti_python_core():
    if get_os_name() != "win":
        # pylint: disable=E1101
        old_flags = sys.getdlopenflags()
        sys.setdlopenflags(2 | 8)  # RTLD_NOW | RTLD_DEEPBIND
    else:
        pyddir = os.path.dirname(os.path.realpath(__file__))
        os.environ["PATH"] += os.pathsep + pyddir
    try:
        from taichi._lib.core import taichi_python as core  # pylint: disable=C0415
    except Exception as e:
        if isinstance(e, ImportError):
            print(
                Fore.YELLOW + "Share object taichi_python import failed, "
                "check this page for possible solutions:\n"
                "https://docs.taichi-lang.org/docs/install" + Fore.RESET
            )
            if get_os_name() == "win":
                # pylint: disable=E1101
                e.msg += "\nConsider installing Microsoft Visual C++ Redistributable: https://aka.ms/vs/16/release/vc_redist.x64.exe"
        raise e from None

    if get_os_name() != "win":
        sys.setdlopenflags(old_flags)  # pylint: disable=E1101
    lib_dir = os.path.join(package_root, "_lib", "runtime")
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
    return os.environ.get("TI_CI", "") == "1"


package_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def get_core_shared_object():
    directory = os.path.join(package_root, "_lib")
    return os.path.join(directory, "libtaichi_python.so")


def print_red_bold(*args, **kwargs):
    print(Fore.RED + Style.BRIGHT, end="")
    print(*args, **kwargs)
    print(Style.RESET_ALL, end="")


def print_yellow_bold(*args, **kwargs):
    print(Fore.YELLOW + Style.BRIGHT, end="")
    print(*args, **kwargs)
    print(Style.RESET_ALL, end="")


def check_exists(src):
    if not os.path.exists(src):
        raise FileNotFoundError(f'File "{src}" not exist. Installation corrupted or build incomplete?')


ti_python_core = import_ti_python_core()

ti_python_core.set_python_package_dir(package_root)

log_level = os.environ.get("TI_LOG_LEVEL", "")
if log_level:
    ti_python_core.set_logging_level(log_level)


def get_dll_name(name):
    if get_os_name() == "linux":
        return f"libtaichi_{name}.so"
    if get_os_name() == "osx":
        return f"libtaichi_{name}.dylib"
    if get_os_name() == "win":
        return f"taichi_{name}.dll"
    raise Exception(f"Unknown OS: {get_os_name()}")


def at_startup():
    ti_python_core.set_core_state_python_imported(True)


at_startup()


def compare_version(latest, current):
    latest_num = map(int, latest.split("."))
    current_num = map(int, current.split("."))
    return tuple(latest_num) > tuple(current_num)


def _print_taichi_header():
    header = "[Taichi] "
    header += f"version {ti_python_core.get_version_string()}, "

    try:
        timestamp_path = os.path.join(ti_python_core.get_repo_dir(), "timestamp")
        if os.path.exists(timestamp_path):
            latest_version = ""
            with open(timestamp_path, "r") as f:
                latest_version = f.readlines()[1].rstrip()
            if compare_version(latest_version, ti_python_core.get_version_string()):
                header += f"latest version {latest_version}, "
    except:
        pass

    llvm_target_support = ti_python_core.get_llvm_target_support()
    header += f"llvm {llvm_target_support}, "

    commit_hash = ti_python_core.get_commit_hash()
    commit_hash = commit_hash[:8]
    header += f"commit {commit_hash}, "

    header += f"{get_os_name()}, "

    py_ver = ".".join(str(x) for x in sys.version_info[:3])
    header += f"python {py_ver}"

    print(header)


if os.getenv("ENABLE_TAICHI_HEADER_PRINT", "True").lower() not in ("false", "0", "f"):
    _print_taichi_header()


def try_get_wheel_tag(module):
    try:
        from email.parser import Parser  # pylint: disable=import-outside-toplevel

        wheel_path = f'{module.__path__[0]}-{".".join(map(str, module.__version__))}.dist-info/WHEEL'
        with open(wheel_path, "r") as f:
            meta = Parser().parse(f)
        return meta.get("Tag")
    except Exception:
        return None


def try_get_loaded_libc_version():
    assert platform.system() == "Linux"
    with open("/proc/self/maps") as f:
        content = f.read()

    try:
        libc_path = next(v for v in content.split() if "libc-" in v)
        ver = re.findall(r"\d+\.\d+", libc_path)
        if not ver:
            return None
        return tuple([int(v) for v in ver[0].split(".")])
    except StopIteration:
        return None


def try_get_pip_version():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pip  # pylint: disable=import-outside-toplevel
        return tuple([int(v) for v in pip.__version__.split(".")])
    except ImportError:
        return None


def warn_restricted_version():
    if os.environ.get("TI_MANYLINUX2014_OK", ""):
        return

    if get_os_name() == "linux":
        try:
            import taichi as ti  # pylint: disable=import-outside-toplevel

            wheel_tag = try_get_wheel_tag(ti)
            if wheel_tag and "manylinux2014" in wheel_tag:
                print_yellow_bold(
                    "You have installed a restricted version of taichi, certain features (e.g. Vulkan & GGUI) will not work."
                )
                libc_ver = try_get_loaded_libc_version()
                if libc_ver and libc_ver < (2, 27):
                    print_yellow_bold(
                        "!! Taichi requires glibc >= 2.27 to run, please try upgrading your OS to a recent one (e.g. Ubuntu 18.04 or later) if possible."
                    )

                pip_ver = try_get_pip_version()
                if pip_ver and pip_ver < (20, 3, 0):
                    print_yellow_bold(
                        f"!! Your pip (version {'.'.join(map(str, pip_ver))}) is outdated (20.3.0 or later required), "
                        "try upgrading pip and install taichi again."
                    )
                    print()
                    print_yellow_bold("    $ python3 -m pip install --upgrade pip")
                    print_yellow_bold("    $ python3 -m pip install --force-reinstall taichi")
                    print()

                print_yellow_bold(
                    "You can suppress this warning by setting the environment variable TI_MANYLINUX2014_OK=1."
                )
        except Exception:
            pass
