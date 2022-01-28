import functools
import subprocess
import traceback

from colorama import Fore, Style
from taichi._lib import core as _ti_core


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
        msg = f'{old} is obsolete. Please use {new} instead.'
        raise SyntaxError(msg)

    return wrapped


def get_traceback(stacklevel=1):
    s = traceback.extract_stack()[:-1 - stacklevel]
    return ''.join(traceback.format_list(s))


def set_gdb_trigger(on=True):
    _ti_core.set_core_trigger_gdb_when_crash(on)


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
