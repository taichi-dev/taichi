import taichi
import sys
import copy


def config_from_dict(args):
    from taichi.core import ti_core
    d = copy.copy(args)
    for k in d:
        if isinstance(d[k], ti_core.Vector2f):
            d[k] = '({}, {})'.format(d[k].x, d[k].y)
        if isinstance(d[k], ti_core.Vector3f):
            d[k] = '({}, {}, {})'.format(d[k].x, d[k].y, d[k].z)
        d[k] = str(d[k])
    return ti_core.config_from_dict(d)


def veci(*args):
    from taichi.core import ti_core
    if isinstance(args[0], ti_core.Vector2i):
        return args[0]
    if isinstance(args[0], ti_core.Vector3i):
        return args[0]
    if isinstance(args[0], tuple):
        args = tuple(*args)
    if len(args) == 2:
        return ti_core.Vector2i(int(args[0]), int(args[1]))
    elif len(args) == 3:
        return ti_core.Vector3i(int(args[0]), int(args[1]), int(args[2]))
    elif len(args) == 4:
        return ti_core.Vector4i(int(args[0]), int(args[1]), int(args[2]),
                                int(args[3]))
    else:
        assert False, type(args[0])


def vec(*args):
    from taichi.core import ti_core
    if isinstance(args[0], ti_core.Vector2f):
        return args[0]
    if isinstance(args[0], ti_core.Vector3f):
        return args[0]
    if isinstance(args[0], ti_core.Vector4f):
        return args[0]
    if isinstance(args[0], ti_core.Vector2d):
        return args[0]
    if isinstance(args[0], ti_core.Vector3d):
        return args[0]
    if isinstance(args[0], ti_core.Vector4d):
        return args[0]
    if isinstance(args[0], tuple):
        args = tuple(*args)
    if ti_core.get_default_float_size() == 4:
        if len(args) == 2:
            return ti_core.Vector2f(float(args[0]), float(args[1]))
        elif len(args) == 3:
            return ti_core.Vector3f(float(args[0]), float(args[1]),
                                    float(args[2]))
        elif len(args) == 4:
            return ti_core.Vector4f(float(args[0]), float(args[1]),
                                    float(args[2]), float(args[3]))
        else:
            assert False, type(args[0])
    else:
        if len(args) == 2:
            return ti_core.Vector2d(float(args[0]), float(args[1]))
        elif len(args) == 3:
            return ti_core.Vector3d(float(args[0]), float(args[1]),
                                    float(args[2]))
        elif len(args) == 4:
            return ti_core.Vector4d(float(args[0]), float(args[1]),
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


import inspect


def get_file_name(asc=0):
    return inspect.stack()[1 + asc][1]


def get_function_name(asc=0):
    return inspect.stack()[1 + asc][3]


def get_line_number(asc=0):
    return inspect.stack()[1 + asc][2]


def warning(msg, type=UserWarning, stacklevel=1):
    import warnings
    import traceback
    import taichi as ti
    use_spdlog = False
    if use_spdlog:
        s = traceback.extract_stack()[:-stacklevel]
        raw = ''.join(traceback.format_list(s))
        ti.warn(f'{type.__name__}: {msg}')
        ti.warn(f'\n{raw}')
    else:
        warnings.warn(msg, type, stacklevel=stacklevel + 1)


def deprecated(old, new):
    import functools

    def decorator(foo):
        @functools.wraps(foo)
        def wrapped(*args, **kwargs):
            _taichi_skip_traceback = 1
            msg = f'{old} is deprecated, please use {new} instead'
            warning(msg, DeprecationWarning, stacklevel=2)
            return foo(*args, **kwargs)

        return wrapped

    return decorator


def get_logging(name):
    def logger(msg, *args, **kwargs):
        # Python inspection takes time (~0.1ms) so avoid it as much as possible
        if taichi.ti_core.logging_effective(name):
            import os
            msg_formatted = msg.format(*args, **kwargs)
            func = getattr(taichi.ti_core, name)
            frame = inspect.currentframe().f_back
            file_name, lineno, func_name, _, _ = inspect.getframeinfo(frame)
            file_name = os.path.basename(file_name)
            msg = f'[{file_name}:{func_name}@{lineno}] {msg_formatted}'
            func(msg)

    return logger


DEBUG = 'debug'
TRACE = 'trace'
INFO = 'info'
WARN = 'warn'
ERROR = 'error'
CRITICAL = 'critical'

debug = get_logging(DEBUG)
trace = get_logging(TRACE)
info = get_logging(INFO)
warn = get_logging(WARN)
error = get_logging(ERROR)
critical = get_logging(CRITICAL)


def redirect_print_to_log():
    class Logger:
        def write(self, msg):
            taichi.core.info('[{}:{}@{}] {}'.format(get_file_name(1),
                                                    get_function_name(1),
                                                    get_line_number(1), msg))

        def flush(self):
            taichi.core.flush_log()

    sys.stdout = Logger()


def duplicate_stdout_to_file(fn):
    taichi.ti_core.duplicate_stdout_to_file(fn)


def set_logging_level(level):
    taichi.ti_core.set_logging_level(level)


def set_gdb_trigger(on=True):
    taichi.ti_core.set_core_trigger_gdb_when_crash(on)


def print_profile_info():
    taichi.ti_core.print_profile_info()


__all__ = [
    'vec',
    'veci',
    'set_gdb_trigger',
    'print_profile_info',
    'set_logging_level',
    'info',
    'warn',
    'error',
    'debug',
    'trace',
    'INFO',
    'WARN',
    'ERROR',
    'DEBUG',
    'TRACE',
]
