# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
from types import ModuleType
from typing import Optional
import importlib
import os
import platform
import re
import subprocess
import sys
import sysconfig

# -- third party --
# -- own --
from .escapes import escape_codes


# -- code --
def is_in_venv() -> bool:
    """
    Are we in a virtual environment?
    """
    return hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)


def get_cache_home() -> Path:
    """
    Get the cache home directory. All intermediate files should be stored here.
    """
    if platform.system() == "Windows":
        return Path(os.environ["LOCALAPPDATA"]) / "ti-build-cache"
    else:
        return Path.home() / ".cache" / "ti-build-cache"


def run(*args, env=None):
    args = list(map(str, args))
    if env is None:
        return subprocess.Popen(args).wait()
    else:
        e = os.environ.copy()
        e.update(env)
        return subprocess.Popen(args, env=e).wait()


def restart():
    """
    Restart the current process.
    """
    if platform.system() == "Windows":
        # GitHub Actions will treat the step as completed when doing os.execl in Windows,
        # since Windows does not have real execve, its behavior is emulated by spawning a new process and
        # terminating the current process. So we do not use os.execl in Windows.
        os._exit(run(sys.executable, "-S", *sys.argv))
    else:
        os.execl(sys.executable, sys.executable, "-S", *sys.argv)


def _try_import(name: str) -> Optional[ModuleType]:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


def ensure_dependencies(*deps: str):
    """
    Automatically install dependencies if they are not installed.
    """

    pip = _try_import("pip")
    ensurepip = _try_import("ensurepip")

    if not sys.flags.no_site:
        # First run, restart with no_site
        if not pip and not ensurepip:
            print(
                "!! pip or ensurepip not found, build.py needs at least a functional pip/ensurepip to work.", flush=True
            )
            sys.exit(1)

        restart()

    # Second run
    v = sys.version_info
    bootstrap_root = get_cache_home() / "bootstrap" / f"{v.major}.{v.minor}"
    bootstrap_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(bootstrap_root))

    try:
        for dep in deps:
            dep = re.split(r"[><=]=?", dep)[0]
            importlib.import_module(dep)
        return
    except ModuleNotFoundError:
        pass

    print("Installing dependencies...", flush=True)
    py = sys.executable
    pip_install = ["-m", "pip", "install", "--no-user", f"--target={bootstrap_root}", "-U"]

    if ensurepip:
        search_path = sysconfig.get_config_var("WHEEL_PKG_DIR")
        if search_path is None:
            search_path = ensurepip.__path__[0]
        wheels = Path(search_path).glob("**/*.whl")
        wheels = os.pathsep.join(map(str, wheels))
        if run(py, "-S", *pip_install, "pip", env={"PYTHONPATH": wheels}):
            raise Exception("Unable to install pip! (ensurepip method)")
    else:  # pip must exist
        if run(py, *pip_install, "pip"):
            raise Exception("Unable to install pip! (pip method)")

    if run(py, "-S", *pip_install, *deps, env={"PYTHONPATH": str(bootstrap_root)}):
        raise Exception("Unable to install dependencies!")

    restart()


def chdir_to_root():
    """
    Change working directory to the root of the repository
    """
    root = Path("/")
    p = Path(__file__).resolve()
    while p != root:
        if (p / "setup.py").exists():
            os.chdir(p)
            break
        p = p.parent


_Environ = os.environ.__class__

_CHANGED_ENV = {}


class _EnvironWrapper(_Environ):
    def __setitem__(self, name: str, value: str) -> None:
        orig = self.get(name, None)
        _Environ.__setitem__(self, name, value)
        new = self[name]
        self._print_diff(name, orig, new)

    def __delitem__(self, name: str) -> None:
        orig = self.get(name, None)
        _Environ.__delitem__(self, name)
        new = self.get(name, None)
        self._print_diff(name, orig, new)

    def pop(self, name: str, default: Optional[str] = None) -> Optional[str]:
        orig = self.get(name, None)
        value = _Environ.pop(self, name, default)
        new = self.get(name, None)
        self._print_diff(name, orig, new)
        return value

    def _print_diff(self, name, orig, new):
        G = escape_codes["bold_green"]
        R = escape_codes["bold_red"]
        N = escape_codes["reset"]

        if orig == new:
            return

        _CHANGED_ENV[name] = new

        p = lambda v: print(v, file=sys.stderr, flush=True)

        if orig == None:
            p(f"{G}:: ENV+ {name}={new}{N}")
        elif new == None:
            p(f"{R}:: ENV- {name}={orig}{N}")
        elif new.startswith(orig):
            l = len(orig)
            p(f"{G}:: ENV{N} {name}={new[:l]}{G}{new[l:]}{N}")
        elif new.endswith(orig):
            l = len(new) - len(orig)
            p(f"{G}:: ENV{N} {name}={G}{new[:l]}{N}{new[l:]}")
        else:
            p(f"{R}:: ENV- {name}={orig}{N}")
            p(f"{G}:: ENV+ {name}={new}{N}")

    def get_changed_envs(self):
        return dict(_CHANGED_ENV)


def monkey_patch_environ():
    """
    Monkey patch os.environ to print changes.
    """
    os.environ.__class__ = _EnvironWrapper


def detect_crippled_python():
    if platform.system() == "Windows" and "Microsoft\\WindowsApps" in sys.executable:
        print(
            ":: ERROR Using Python installed from Microsoft Store to run build.py is not supported. "
            "Please use Python from https://python.org/downloads/",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


def windows_enable_long_paths():
    import winreg

    key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
    try:
        enabled = winreg.QueryValueEx(key, "LongPathsEnabled") == (1, 4)
    except FileNotFoundError:
        enabled = False

    if not enabled:
        from .misc import info, warn
        from .tinysh import Command, sudo

        info("Enabling long paths on Windows")
        reg = Command("reg.exe")
        try:
            with sudo():
                reg.add(
                    r"HKLM\SYSTEM\CurrentControlSet\Control\FileSystem",
                    "/v",
                    "LongPathsEnabled",
                    "/t",
                    "REG_DWORD",
                    "/d",
                    "1",
                    "/f",
                )
        except OSError as e:
            if e.winerror == 1223:
                warn("Enabling long paths cancelled, you may encounter compile errors later")
            else:
                raise


def early_init():
    """
    Do early initialization.
    This must be called before any other non-stdlib imports.
    """
    detect_crippled_python()
    ensure_dependencies("tqdm", "requests", "mslex", "psutil>=5.9.5")
    chdir_to_root()
    monkey_patch_environ()

    if platform.system() == "Windows":
        windows_enable_long_paths()
