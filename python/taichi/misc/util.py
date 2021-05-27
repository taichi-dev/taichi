import copy
import inspect
import sys
import traceback

from colorama import Fore, Style
from taichi.core import ti_core as _ti_core

import taichi


def config_from_dict(args):
    d = copy.copy(args)
    for k in d:
        if isinstance(d[k], _ti_core.Vector2f):
            d[k] = '({}, {})'.format(d[k].x, d[k].y)
        if isinstance(d[k], _ti_core.Vector3f):
            d[k] = '({}, {}, {})'.format(d[k].x, d[k].y, d[k].z)
        d[k] = str(d[k])
    return _ti_core.config_from_dict(d)


def core_veci(*args):
    if isinstance(args[0], _ti_core.Vector2i):
        return args[0]
    if isinstance(args[0], _ti_core.Vector3i):
        return args[0]
    if isinstance(args[0], tuple):
        args = tuple(*args)
    if len(args) == 2:
        return _ti_core.Vector2i(int(args[0]), int(args[1]))
    elif len(args) == 3:
        return _ti_core.Vector3i(int(args[0]), int(args[1]), int(args[2]))
    elif len(args) == 4:
        return _ti_core.Vector4i(int(args[0]), int(args[1]), int(args[2]),
                                 int(args[3]))
    else:
        assert False, type(args[0])


def core_vec(*args):
    if isinstance(args[0], _ti_core.Vector2f):
        return args[0]
    if isinstance(args[0], _ti_core.Vector3f):
        return args[0]
    if isinstance(args[0], _ti_core.Vector4f):
        return args[0]
    if isinstance(args[0], _ti_core.Vector2d):
        return args[0]
    if isinstance(args[0], _ti_core.Vector3d):
        return args[0]
    if isinstance(args[0], _ti_core.Vector4d):
        return args[0]
    if isinstance(args[0], tuple):
        args = tuple(*args)
    if _ti_core.get_default_float_size() == 4:
        if len(args) == 2:
            return _ti_core.Vector2f(float(args[0]), float(args[1]))
        elif len(args) == 3:
            return _ti_core.Vector3f(float(args[0]), float(args[1]),
                                     float(args[2]))
        elif len(args) == 4:
            return _ti_core.Vector4f(float(args[0]), float(args[1]),
                                     float(args[2]), float(args[3]))
        else:
            assert False, type(args[0])
    else:
        if len(args) == 2:
            return _ti_core.Vector2d(float(args[0]), float(args[1]))
        elif len(args) == 3:
            return _ti_core.Vector3d(float(args[0]), float(args[1]),
                                     float(args[2]))
        elif len(args) == 4:
            return _ti_core.Vector4d(float(args[0]), float(args[1]),
                                     float(args[2]), float(args[3]))
        else:
            assert False, type(args[0])


class Tee():
    def __init__(self, name):
        self.file = open(name, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()

    def write_to_file(self, data):
        self.file.write(data)


# The builtin `warnings` module is unreliable since it may be suppressed
# by other packages such as IPython.
def warning(msg, type=UserWarning, stacklevel=1):
    s = traceback.extract_stack()[:-stacklevel]
    raw = ''.join(traceback.format_list(s))
    print(Fore.YELLOW + Style.BRIGHT, end='')
    print(f'{type.__name__}: {msg}')
    print(f'\n{raw}')
    print(Style.RESET_ALL, end='')


def deprecated(old, new, warning_type=DeprecationWarning):
    """
    Mark an API as deprecated. Usage:

    @deprecated('ti.sqr(x)', 'x**2')
    def sqr(x):
        return x**2
    """
    import functools

    def decorator(foo):
        @functools.wraps(foo)
        def wrapped(*args, **kwargs):
            _taichi_skip_traceback = 1
            msg = f'{old} is deprecated. Please use {new} instead.'
            warning(msg, warning_type, stacklevel=2)
            return foo(*args, **kwargs)

        return wrapped

    return decorator


def obsolete(old, new):
    """
    Mark an API as obsolete. Usage:

    sqr = obsolete('ti.sqr(x)', 'x**2')
    """
    def wrapped(*args, **kwargs):
        _taichi_skip_traceback = 1
        msg = f'{old} is obsolete. Please use {new} instead.'
        raise SyntaxError(msg)

    return wrapped


def get_traceback(stacklevel=1):
    s = traceback.extract_stack()[:-1 - stacklevel]
    return ''.join(traceback.format_list(s))


def duplicate_stdout_to_file(fn):
    _ti_core.duplicate_stdout_to_file(fn)


def set_gdb_trigger(on=True):
    _ti_core.set_core_trigger_gdb_when_crash(on)


def print_profile_info():
    _ti_core.print_profile_info()


def clear_profile_info():
    _ti_core.clear_profile_info()


@deprecated('ti.vec(x, y)', 'ti.core_vec(x, y)')
def vec(*args, **kwargs):
    return core_vec(*args, **kwargs)


@deprecated('ti.veci(x, y)', 'ti.core_veci(x, y)')
def veci(*args, **kwargs):
    return core_veci(*args, **kwargs)


def dump_dot(filepath=None, rankdir=None, embed_states_threshold=0):
    d = _ti_core.dump_dot(rankdir, embed_states_threshold)
    if filepath is not None:
        with open(filepath, 'w') as fh:
            fh.write(d)
    return d


def dot_to_pdf(dot, filepath):
    assert filepath.endswith('.pdf')
    import subprocess
    p = subprocess.Popen(['dot', '-Tpdf'],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE)
    pdf_contents = p.communicate(input=dot.encode())[0]
    with open(filepath, 'wb') as fh:
        fh.write(pdf_contents)


def get_kernel_stats():
    return _ti_core.get_kernel_stats()


def print_async_stats(include_kernel_profiler=False):
    import taichi as ti
    if include_kernel_profiler:
        ti.kernel_profiler_print()
        print()
    stat = ti.get_kernel_stats()
    counters = stat.get_counters()
    print('=======================')
    print('Async benchmark metrics')
    print('-----------------------')
    print(f'Async mode:           {ti.current_cfg().async_mode}')
    print(f'Kernel time:          {ti.kernel_profiler_total_time():.3f} s')
    print(f'Tasks launched:       {int(counters["launched_tasks"])}')
    print(f'Instructions emitted: {int(counters["codegen_statements"])}')
    print(f'Tasks compiled:       {int(counters["codegen_offloaded_tasks"])}')
    NUM_FUSED_TASKS_KEY = 'num_fused_tasks'
    if NUM_FUSED_TASKS_KEY in counters:
        print(f'Tasks fused:          {int(counters["num_fused_tasks"])}')
    print('=======================')


__all__ = [
    'vec',
    'veci',
    'core_vec',
    'core_veci',
    'deprecated',
    'dump_dot',
    'dot_to_pdf',
    'obsolete',
    'get_kernel_stats',
    'get_traceback',
    'set_gdb_trigger',
    'print_profile_info',
    'clear_profile_info',
]
