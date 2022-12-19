# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
from typing import Callable, Optional
import inspect
import os
import platform
import sys

# -- third party --
# -- own --
from .escapes import escape_codes


# -- code --
def is_manylinux2014() -> bool:
    '''
    Are we in a manylinux2014 environment?
    This means a particular CentOS docker image.
    '''
    return platform.system() == 'Linux' and Path('/etc/centos-release').exists()


def get_cache_home() -> Path:
    '''
    Get the cache home directory. All intermediate files should be stored here.
    '''
    if platform.system() == 'Windows':
        return Path(env['LOCALAPPDATA']) / 'build-cache'
    else:
        return Path.home() / '.cache' / 'build-cache'


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
            print(f'{C}:: -----BEGIN {msg}-----{N}'.format(**_args.arguments), file=sys.stderr, flush=True)
            try:
                ret = f(*args, **kwargs)
                print(f'{C}:: -----END {msg}-----{N}'.format(**_args.arguments), file=sys.stderr, flush=True)
                return ret
            except BaseException as e:
                print(f'{R}!! -----EXCEPTION {msg}-----{N}'.format(**_args.arguments), file=sys.stderr, flush=True)
                raise

        return wrapper

    return decorate


class _EnvWrapper:

    def __getitem__(self, name: str) -> str:
        return os.environ[name]

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return os.environ.get(name, default)

    def __setitem__(self, name: str, value: str) -> None:
        orig = os.environ.get(name)
        os.environ[name] = value
        new = os.environ[name]

        G = escape_codes['bold_green']
        R = escape_codes['bold_red']
        N = escape_codes['reset']
        print(f'{R}:: ENV -{name}={orig}{N}', file=sys.stderr, flush=True)
        print(f'{G}:: ENV +{name}={new}{N}', file=sys.stderr, flush=True)


env = _EnvWrapper()
