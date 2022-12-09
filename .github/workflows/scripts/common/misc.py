# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
import inspect
import os
import platform

# -- third party --
# -- own --

# -- code --
def is_manylinux2014():
    # manylinux2014 builds in a special CentOS docker image
    return platform.system() == 'Linux' and Path('/etc/centos-release').exists()


def get_cache_home():
    if platform.system() == 'Windows':
        return Path(os.environ['LOCALAPPDATA']) / 'build-cache'
    else:
        return Path.home() / '.cache' / 'build-cache'


def banner(msg):
    def decorate(f):
        sig = inspect.signature(f)
        def wrapper(*args, **kwargs):
            _args = sig.bind(*args, **kwargs)
            print(f':: -----BEGIN {msg}-----'.format(**_args.arguments))
            try:
                ret = f(*args, **kwargs)
                print(f':: -----END {msg}-----'.format(**_args.arguments))
                return ret
            except BaseException as e:
                print(f'!! -----EXCEPTION {msg}-----'.format(**_args.arguments))
                raise

        return wrapper

    return decorate
