# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform
import sys
from pathlib import Path

# -- third party --
import psutil

# -- own --
from . import misc
from .cmake import cmake_args
from .compiler import get_cache_home
from .misc import banner
from .python import get_desired_python_version
from .tinysh import Command


# -- code --
@banner("Add AOT Related Environment Variables")
def add_aot_env():
    os.environ["TAICHI_REPO_DIR"] = os.getcwd()
    pyver = get_desired_python_version()
    for p in Path(os.getcwd()).glob("**/cmake-install/python/taichi/_lib/c_api"):
        if p.is_dir() and any(v.name.endswith(f"-{pyver}") for v in p.parents):
            os.environ["TAICHI_C_API_INSTALL_DIR"] = str(p)
            break
    else:
        misc.warn("Failed to find TAICHI_C_API_INSTALL_DIR (did't build C-API?), skipping")


def _write_ti_bashrc():
    path = get_cache_home() / "ti.bashrc"
    envs = get_cache_home() / "ti-env.sh"
    _write_env(envs)
    with open(path, "w") as f:
        f.write(
            "[ -f /etc/bashrc ] && source /etc/bashrc\n"
            "[ -f ~/.bashrc ] && source ~/.bashrc\n"
            r'export PS1="\[\e]0;[Taichi Build Environment]\a\]\[\033[01;31m\][Taichi Build] \[\033[00m\]$PS1"'
            "\n"
            f"source {envs}\n"
        )

    return path


def _write_ti_zshrc():
    dotdir = get_cache_home() / "zdotdir"
    dotdir.mkdir(parents=True, exist_ok=True)
    path = dotdir / ".zshrc"
    envs = get_cache_home() / "ti-env.sh"
    _write_env(envs)
    with open(path, "w") as f:
        f.write(
            "[ -f /etc/zsh/zshrc ] && source /etc/zsh/zshrc\n"
            "[ -f $HOME/.zshrc ] && source $HOME/.zshrc\n"
            r"export PROMPT='%{$fg_bold[red]%}[Taichi Build] %{$reset_color%}'$PROMPT"
            "\n"
            f"source {envs}\n"
        )
    return dotdir


def _write_ti_pwshrc():
    path = get_cache_home() / "ti.ps1"
    with open(path, "w") as f:
        f.write(
            "\n".join(
                [
                    r"function Prompt {",
                    r'    return "TiBuild $($executionContext.SessionState.Path.CurrentLocation)$(">" * ($nestedPromptLevel + 1)) "'
                    r"}",
                ]
            )
        )

    return path


KNOWN_SHELLS = (
    "bash",
    "zsh",
    "sh",
    "pwsh",
    "cmd.exe",
    "pwsh.exe",
    "powershell.exe",
)


class Shell:
    def __init__(self, name: str, exe: str):
        self.name = name
        self.exe = exe


def _find_shell():
    proc = psutil.Process()
    while proc:
        exe = proc.exe()
        name = exe.split(os.path.sep)[-1]
        if name in KNOWN_SHELLS:
            return Shell(name, exe)
        proc = proc.parent()

    return None


def enter_shell():
    cmake_args.writeback()
    misc.info("Entering shell...")
    if platform.system() == "Windows":
        shell = _find_shell() or Shell("cmd.exe", "cmd.exe")
        if shell.name in ("pwsh.exe", "powershell.exe"):
            pwsh = Command(shell.exe)
            path = _write_ti_pwshrc()
            pwsh("-ExecutionPolicy", "Bypass", "-NoExit", "-File", str(path))
        elif shell.name == "cmd.exe":
            cmd = Command(shell.exe)
            cmd("/k", "set", "PROMPT=TaichiBuild $P$G")
        else:
            # Unknown shell, not doing anything fancy
            os.execl(shell.exe, shell.exe)
    else:
        shell = _find_shell() or Shell("bash", "/bin/bash")
        if shell.name not in ("sh", "bash", "zsh"):
            import pwd

            path = pwd.getpwuid(os.getuid()).pw_shell.split("/")[-1]
            name = path.split("/")[-1]
            shell = Shell(name, path)

        if shell.name == "bash":
            path = _write_ti_bashrc()
            os.execl(shell.exe, shell.exe, "--rcfile", str(path))
        elif shell.name == "zsh":
            path = _write_ti_zshrc()
            env = os.environ.copy()
            env["ZDOTDIR"] = str(path)
            os.execle(shell.exe, shell.exe, env)
        else:
            # Unknown shell, not doing anything fancy
            os.execl(shell.exe, shell.exe)


def _write_env(path):
    envs = os.environ.get_changed_envs()
    envstr = ""

    if isinstance(path, Path):
        path = str(path)

    if path.endswith(".ps1"):
        envstr = "\n".join([f'$env:{k}="{v}"' for k, v in envs.items()])
    elif path.endswith(".sh"):
        envstr = "\n".join([f'export {k}="{v}"' for k, v in envs.items()])
    elif path.endswith(".json"):
        import json

        envstr = json.dumps(envs, indent=2)
    else:
        raise RuntimeError(f"Unsupported format")

    with open(path, "w") as f:
        f.write(envstr)


def write_env(path):
    cmake_args.writeback()
    _write_env(path)
    misc.info(f"Environment variables written to {path}")


def handle_alternate_actions():
    if misc.options.write_env:
        add_aot_env()
        write_env(misc.options.write_env)
    elif misc.options.shell:
        add_aot_env()
        enter_shell()
    else:
        return

    sys.exit(0)
