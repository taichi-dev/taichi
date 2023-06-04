# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Mapping, Sequence

# -- third party --
# -- own --
from .escapes import escape_codes

# -- code --
# A minimal and naive imitiation of the sh library, which can work on Windows.
# NOT written as a general purpose library, wild assumptions are made.

IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    import ctypes

    import mslex

    quote = mslex.quote

    SW_SHOWNORMAL = 1
    SEE_MASK_NOCLOSEPROCESS = 0x00000040
    INFINITE = -1

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_uint32),
            ("fMask", ctypes.c_ulong),
            ("hwnd", ctypes.c_void_p),
            ("lpVerb", ctypes.c_wchar_p),
            ("lpFile", ctypes.c_wchar_p),
            ("lpParameters", ctypes.c_wchar_p),
            ("lpDirectory", ctypes.c_wchar_p),
            ("nShow", ctypes.c_int),
            ("hInstApp", ctypes.c_void_p),
            ("lpIDList", ctypes.c_void_p),
            ("lpClass", ctypes.c_wchar_p),
            ("hkeyClass", ctypes.c_void_p),
            ("dwHotKey", ctypes.c_uint32),
            ("DUMMYUNIONNAME", ctypes.c_void_p),
            ("hProcess", ctypes.c_void_p),
        ]

    def win32_is_user_admin():
        return ctypes.windll.shell32.IsUserAnAdmin() != 0

    def win32_run_elevated(exe: str, params: str):
        kernel32 = ctypes.windll.kernel32
        shell32 = ctypes.windll.shell32

        sei = SHELLEXECUTEINFOW()
        sei.cbSize = ctypes.sizeof(SHELLEXECUTEINFOW)
        sei.lpVerb = "runas"
        sei.lpFile = exe
        sei.lpParameters = params
        sei.nShow = SW_SHOWNORMAL
        sei.fMask = SEE_MASK_NOCLOSEPROCESS

        if not shell32.ShellExecuteExW(ctypes.byref(sei)):
            raise ctypes.WinError()

        hProcess = sei.hProcess
        kernel32.WaitForSingleObject(hProcess, -1)
        rc = ctypes.c_uint(-1)
        kernel32.GetExitCodeProcess(hProcess, ctypes.byref(rc))
        kernel32.CloseHandle(hProcess)
        return rc.value

else:
    import shlex

    quote = shlex.quote


class CommandFailed(Exception):
    def __init__(self, cmd, code):
        self.cmd = cmd
        self.code = code

    def __str__(self):
        return f"Command {self.cmd} failed with code {self.code}"


ENVIRON_STACK = []
PREFIX_STACK = []
OPTIONS_STACK = []

P = escape_codes["bold_purple"]
N = escape_codes["reset"]

pr = lambda *args: print(*args, file=sys.stderr, flush=True)


class Command:
    def __init__(self, *args: str):
        self.args = list((map(str, args)))

    def __getattribute__(self, name: str) -> Any:
        if name in ("args", "bake") or name.startswith("__"):
            return object.__getattribute__(self, name)

        return self.bake(name)

    def bake(self, *moreargs: Sequence[str]) -> "Command":
        args = object.__getattribute__(self, "args")
        cls = object.__getattribute__(self, "__class__")
        return cls(*args, *moreargs)

    def __call__(self, *moreargs: Sequence[str]) -> None:
        args = object.__getattribute__(self, "args")
        args = args + list(map(str, moreargs))

        prefixes = []
        for v in PREFIX_STACK:
            prefixes.extend(v)

        overlay = {}
        for v in ENVIRON_STACK:
            overlay.update(v)

        args = prefixes + args
        cmd = " ".join([quote(v) for v in args])

        pr(f"{P}:: RUN {cmd}{N}")
        if overlay:
            pr(f"{P}>> WITH ADDITIONAL ENVS:{N}")
            for k, v in overlay.items():
                pr(f"{P}       {k}={v}{N}")

        env = os.environ.copy()
        env.update(overlay)

        options = {}
        for o in OPTIONS_STACK:
            options.update(o)

        exe = shutil.which(args[0])
        assert exe, f"Cannot find executable {args[0]}"

        runas = IS_WINDOWS and options.get("runas")
        assert not (runas and overlay), "Cannot run with both elevated privileges and additional envs"

        if runas and not win32_is_user_admin():
            pr(f"{P}>> !! WITH ELEVATED PRIVILEGES !!{N}")
            code = win32_run_elevated(exe, " ".join([quote(v) for v in args[1:]]))
            if code:
                raise CommandFailed(cmd, code)
        else:
            proc = subprocess.Popen(args, executable=exe, env=env)
            code = proc.wait()
            if code:
                raise CommandFailed(cmd, code)

    def __repr__(self) -> str:
        return f"<Command '{shlex.join(self.args)}'>"


@contextmanager
def environ(*envs: Mapping[str, str]):
    """
    Set command environment variables.
    """
    global ENVIRON_STACK

    this = {}
    for env in envs:
        this.update(env)

    try:
        ENVIRON_STACK.append(this)
        yield
    finally:
        assert ENVIRON_STACK[-1] is this
        ENVIRON_STACK.pop()


@contextmanager
def prefix(*args: str):
    """
    Set command prefixes.
    """
    global PREFIX_STACK

    l = list(map(str, args))

    try:
        PREFIX_STACK.insert(0, l)
        yield
    finally:
        assert PREFIX_STACK[0] is l
        PREFIX_STACK.pop(0)


@contextmanager
def with_options(options: Mapping[str, Any]):
    try:
        OPTIONS_STACK.append(options)
        yield
    finally:
        OPTIONS_STACK.pop()


def sudo():
    """
    Wrap a command with sudo.
    """
    if IS_WINDOWS:
        return with_options({"runas": True})
    elif os.geteuid() != 0:
        return prefix("sudo")
    else:
        return with_options({})


def nice():
    """
    Wrap a command with sudo.
    """
    if IS_WINDOWS:
        from .misc import warn

        warn("nice is not yet implemented on Windows")
        return with_options({})
    else:
        return prefix("nice")


sh = Command()
git = sh.git
# Use setup_python !
# python = sh.bake(sys.executable)
# pip = python.bake('-m', 'pip')
sccache = sh.sccache
tar = sh.tar
bash = sh.bash
start = sh.start.bake("/wait")
apt = sh.sudo.apt
powershell = Command("powershell.exe")
pwsh = Command("pwsh.exe")
