import functools
import subprocess
import traceback

from colorama import Fore, Style
from taichi._lib import core as _ti_core


def core_veci(*args):
    if isinstance(args[0], _ti_core.Vector2i):
        return args[0]
    if isinstance(args[0], _ti_core.Vector3i):
        return args[0]
    if isinstance(args[0], tuple):
        args = tuple(*args)
    if len(args) == 2:
        return _ti_core.Vector2i(int(args[0]), int(args[1]))
    if len(args) == 3:
        return _ti_core.Vector3i(int(args[0]), int(args[1]), int(args[2]))
    if len(args) == 4:
        return _ti_core.Vector4i(int(args[0]), int(args[1]), int(args[2]),
                                 int(args[3]))
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
        if len(args) == 3:
            return _ti_core.Vector3f(float(args[0]), float(args[1]),
                                     float(args[2]))
        if len(args) == 4:
            return _ti_core.Vector4f(float(args[0]), float(args[1]),
                                     float(args[2]), float(args[3]))
        assert False, type(args[0])
    else:
        if len(args) == 2:
            return _ti_core.Vector2d(float(args[0]), float(args[1]))
        if len(args) == 3:
            return _ti_core.Vector3d(float(args[0]), float(args[1]),
                                     float(args[2]))
        if len(args) == 4:
            return _ti_core.Vector4d(float(args[0]), float(args[1]),
                                     float(args[2]), float(args[3]))
        assert False, type(args[0])


# The builtin `warnings` module is unreliable since it may be suppressed
# by other packages such as IPython.
def warning(msg, warning_type=UserWarning, stacklevel=1):
    """Print warning message

    Args:
        msg (str): massage to print.
        warning_type (builtin warning type):  type of warning.
        stacklevel (int): warning stack level from the caller.
    """
    s = traceback.extract_stack()[:-stacklevel]
    raw = ''.join(traceback.format_list(s))
    print(Fore.YELLOW + Style.BRIGHT, end='')
    print(f'{warning_type.__name__}: {msg}')
    print(f'\n{raw}')
    print(Style.RESET_ALL, end='')


def deprecated(old, new, warning_type=DeprecationWarning):
    """Mark an API as deprecated.

    Args:
        old (str): old method.
        new (str): new method.
        warning_type (builtin warning type): type of warning.

    Example::

        >>> @deprecated('ti.sqr(x)', 'x**2')
        >>> def sqr(x):
        >>>     return x**2

    Returns:
        Decorated fuction with warning message
    """
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


def set_gdb_trigger(on=True):
    _ti_core.set_core_trigger_gdb_when_crash(on)


def print_profile_info():
    """Print time elapsed on the host tasks in a hierarchical format.

    This profiler is automatically on.

    Call function imports from C++ : _ti_core.print_profile_info()

    Example::

            >>> import taichi as ti
            >>> ti.init(arch=ti.cpu)
            >>> var = ti.field(ti.f32, shape=1)
            >>> @ti.kernel
            >>> def compute():
            >>>     var[0] = 1.0
            >>>     print("Setting var[0] =", var[0])
            >>> compute()
            >>> ti.print_profile_info()
    """
    _ti_core.print_profile_info()


def clear_profile_info():
    """Clear profiler's records about time elapsed on the host tasks.

    Call function imports from C++ : _ti_core.clear_profile_info()
    """
    _ti_core.clear_profile_info()


def dump_dot(filepath=None, rankdir=None, embed_states_threshold=0):
    d = _ti_core.dump_dot(rankdir, embed_states_threshold)
    if filepath is not None:
        with open(filepath, 'w') as fh:
            fh.write(d)
    return d


def dot_to_pdf(dot, filepath):
    assert filepath.endswith('.pdf')
    with subprocess.Popen(['dot', '-Tpdf'],
                          stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE) as p:
        pdf_contents = p.communicate(input=dot.encode())[0]
        with open(filepath, 'wb') as fh:
            fh.write(pdf_contents)


def get_kernel_stats():
    return _ti_core.get_kernel_stats()
