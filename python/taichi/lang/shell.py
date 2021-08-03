import atexit
import functools
import os
import sys

from taichi.core.util import ti_core as _ti_core

import taichi as ti

try:
    import sourceinspect as oinspect
except ImportError:
    ti.warn('`sourceinspect` not installed!')
    ti.warn(
        'Without this package Taichi may not function well in Python IDLE interactive shell, '
        'Blender scripting module and Python native shell.')
    ti.warn('Please run `python3 -m pip install sourceinspect` to install.')
    import inspect as oinspect

pybuf_enabled = False
_env_enable_pybuf = os.environ.get('TI_ENABLE_PYBUF', '1')
if not _env_enable_pybuf or int(_env_enable_pybuf):
    # When using in Jupyter / IDLE, the sys.stdout will be their wrapped ones.
    # While sys.__stdout__ should always be the raw console stdout.
    pybuf_enabled = sys.stdout is not sys.__stdout__

_ti_core.toggle_python_print_buffer(pybuf_enabled)


def _shell_pop_print(old_call):
    if not pybuf_enabled:
        # zero-overhead!
        return old_call

    ti.info('Graphical python shell detected, using wrapped sys.stdout')

    @functools.wraps(old_call)
    def new_call(*args, **kwargs):
        _taichi_skip_traceback = 1
        ret = old_call(*args, **kwargs)
        # print's in kernel won't take effect until ti.sync(), discussion:
        # https://github.com/taichi-dev/taichi/pull/1303#discussion_r444897102
        print(_ti_core.pop_python_print_buffer(), end='')
        return ret

    return new_call
