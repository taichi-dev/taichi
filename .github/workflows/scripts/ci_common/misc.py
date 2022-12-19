# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
from typing import Callable
import inspect
import os
import platform
import sys

# -- third party --
# -- own --
from .escapes import escape_codes


# -- code --
def is_manylinux2014() -> bool:
    # manylinux2014 builds in a special CentOS docker image
    return platform.system() == 'Linux' and Path('/etc/centos-release').exists()


def get_cache_home() -> Path:
    if platform.system() == 'Windows':
        return Path(os.environ['LOCALAPPDATA']) / 'build-cache'
    else:
        return Path.home() / '.cache' / 'build-cache'


def banner(msg: str) -> Callable:
    def decorate(f: Callable) -> Callable:
        sig = inspect.signature(f)
        C = escape_codes['bold_cyan']
        R = escape_codes['bold_red']
        N = escape_codes['reset']
        def wrapper(*args, **kwargs):
            _args = sig.bind(*args, **kwargs)
            print(f'{C}:: -----BEGIN {msg}-----{N}'.format(**_args.arguments), file=sys.stderr)
            try:
                ret = f(*args, **kwargs)
                print(f'{C}:: -----END {msg}-----{N}'.format(**_args.arguments), file=sys.stderr)
                return ret
            except BaseException as e:
                print(f'{R}!! -----EXCEPTION {msg}-----{N}'.format(**_args.arguments), file=sys.stderr)
                raise

        return wrapper

    return decorate
