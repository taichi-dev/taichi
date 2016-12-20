from taichi.util import get_os_name

import random
import shutil
import os
import sys

if get_os_name() == 'osx':
    if os.path.exists('libtaichi_core.dylib'):
        shutil.copy('libtaichi_core.dylib', 'taichi_core.so')
        sys.path.append(".")
    import taichi_core as tc_core
elif get_os_name() == 'linux':
    if os.path.exists('libtaichi_core.so'):
        shutil.copy('libtaichi_core.so', 'taichi_core.so')
        sys.path.append(".")
    import taichi_core as tc_core
elif get_os_name() == 'win':
    dll_path = 'Release/taichi_core.dll'
    d = 'tmp' + str(random.randint(0, 100000000)) + '/'
    try:
        os.mkdir(d)
    except:
        pass

    if os.path.exists(dll_path):
        shutil.copy(dll_path, d + 'taichi_core.pyd')
        sys.path.append(os.getcwd() + '/' + d)
        import taichi_core as tc_core
    else:
        assert False, "Library taichi_core doesn't exists."

__all__ = [
    'tc_core'
]
