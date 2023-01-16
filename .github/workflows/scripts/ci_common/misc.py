# -*- coding: utf-8 -*-

# -- stdlib --
import inspect
import os
import platform
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from .bootstrap import get_cache_home  # noqa
# -- third party --
# -- own --
from .escapes import escape_codes


# -- code --
def is_manylinux2014() -> bool:
    '''
    Are we in a manylinux2014 environment?
    This means a particular CentOS docker image.
    '''
    return platform.system() == 'Linux' and Path(
        '/etc/centos-release').exists()


def banner(msg: str) -> Callable:
    '''
    Decorate a function to print a banner before and after it.
    '''
    def decorate(f: Callable) -> Callable:
        sig = inspect.signature(f)
        C = escape_codes['bold_cyan']
        R = escape_codes['bold_red']
        N = escape_codes['reset']

        def wrapper(*args, **kwargs):
            _args = sig.bind(*args, **kwargs)
            print(f'{C}:: -----BEGIN {msg}-----{N}'.format(**_args.arguments),
                  file=sys.stderr,
                  flush=True)
            try:
                ret = f(*args, **kwargs)
                print(
                    f'{C}:: -----END {msg}-----{N}'.format(**_args.arguments),
                    file=sys.stderr,
                    flush=True)
                return ret
            except BaseException as e:
                print(f'{R}!! -----EXCEPTION {msg}-----{N}'.format(
                    **_args.arguments),
                      file=sys.stderr,
                      flush=True)
                raise

        return wrapper

    return decorate


def path_prepend(var: str, *paths: Any) -> None:
    '''
    Prepend paths to the environment variable.
    '''
    value = os.pathsep.join(str(p) for p in paths if p)
    orig = os.environ.get(var, '')
    if orig:
        value += os.pathsep + orig
    os.environ[var] = value


def hook(module, name=None):
    '''
    Hook a function
    '''
    def inner(hooker):
        funcname = name or hooker.__name__
        hookee = getattr(module, funcname)

        @wraps(hookee)
        def real_hooker(*args, **kwargs):
            return hooker(hookee, *args, **kwargs)

        real_hooker.orig = hookee
        setattr(module, funcname, real_hooker)
        return real_hooker

    return inner
