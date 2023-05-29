# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
from typing import Any, Callable
import inspect
import os
import platform
import sys

# -- third party --
# -- own --
from .bootstrap import get_cache_home  # noqa, this is a re-export
from .escapes import escape_codes


# -- code --
options = None


def is_manylinux2014() -> bool:
    """
    Are we in a manylinux2014 environment?
    This means a particular CentOS docker image.
    """
    return platform.system() == "Linux" and Path("/etc/centos-release").exists()


def info(msg: str) -> None:
    B = escape_codes["bold_blue"]
    N = escape_codes["reset"]
    print(f"{B}:: INFO {msg}{N}", file=sys.stderr, flush=True)


def warn(msg: str) -> None:
    Y = escape_codes["bold_yellow"]
    N = escape_codes["reset"]
    print(f"{Y}:: WARN {msg}{N}", file=sys.stderr, flush=True)


def error(msg: str) -> None:
    if options and options.permissive:
        warn(msg)
    else:
        R = escape_codes["bold_red"]
        N = escape_codes["reset"]
        print(f"{R}!! ERROR {msg}{N}", file=sys.stderr, flush=True)
        sys.exit(1)


def banner(msg: str) -> Callable:
    """
    Decorate a function to print a banner before and after it.
    """
    p = lambda s: print(s, file=sys.stderr, flush=True)

    def decorate(f: Callable) -> Callable:
        sig = inspect.signature(f)
        C = escape_codes["bold_cyan"]
        R = escape_codes["bold_red"]
        N = escape_codes["reset"]

        def wrapper(*args, **kwargs):
            _args = sig.bind(*args, **kwargs)
            p(f"{C}:: -----BEGIN {msg}-----{N}".format(**_args.arguments))
            try:
                ret = f(*args, **kwargs)
                p(f"{C}:: -----END {msg}-----{N}".format(**_args.arguments))
                return ret
            except BaseException as e:
                p(f"{R}!! -----EXCEPTION {msg}-----{N}".format(**_args.arguments))
                raise

        return wrapper

    return decorate


def path_prepend(var: str, *paths: Any) -> None:
    """
    Prepend paths to the environment variable.
    """
    value = os.pathsep.join(str(p) for p in paths if p)
    orig = os.environ.get(var, "")
    if orig:
        value += os.pathsep + orig
    os.environ[var] = value
