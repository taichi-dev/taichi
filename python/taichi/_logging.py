import inspect
import os

from taichi._lib import core as ti_python_core


def _get_logging(name):
    """Generates a decorator to decorate a specific logger function.

    Args:
        name (str): The string represents logging level.
            Effective levels include: 'trace', 'debug', 'info', 'warn', 'error', 'critical'.

    Returns:
        Callabe: The decorated function.
    """

    def logger(msg, *args, **kwargs):
        # Python inspection takes time (~0.1ms) so avoid it as much as possible
        if ti_python_core.logging_effective(name):
            msg_formatted = msg.format(*args, **kwargs)
            func = getattr(ti_python_core, name)
            frame = inspect.currentframe().f_back
            file_name, lineno, func_name, _, _ = inspect.getframeinfo(frame)
            file_name = os.path.basename(file_name)
            msg = f"[{file_name}:{func_name}@{lineno}] {msg_formatted}"
            func(msg)

    return logger


def set_logging_level(level):
    """Setting the logging level to a specified value.
    Available levels are: 'trace', 'debug', 'info', 'warn', 'error', 'critical'.

    Note that after calling this function, logging levels below the specified one will
    also be effective. For example if `level` is set to 'warn', then the levels below
    it, which are 'error' and 'critical' in this case, will also be effective.

    See also https://docs.taichi-lang.org/docs/developer_utilities#logging.

    Args:
        level (str): Logging level.

    Example::

        >>> set_logging_level('debug')
    """
    ti_python_core.set_logging_level(level)


def is_logging_effective(level):
    """Check if the specified logging level is effective.
    All levels below current level will be effective.
    The default level is 'info'.

    See also https://docs.taichi-lang.org/docs/developer_utilities#logging.

    Args:
        level (str): The string represents logging level. \
            Effective levels include: 'trace', 'debug', 'info', 'warn', 'error', 'critical'.

    Returns:
        Bool: Indicate whether the logging level is effective.

    Example::

        >>> # assume current level is 'info'
        >>> print(ti.is_logging_effective("trace"))     # False
        >>> print(ti.is_logging_effective("debug"))     # False
        >>> print(ti.is_logging_effective("info"))      # True
        >>> print(ti.is_logging_effective("warn"))      # True
        >>> print(ti.is_logging_effective("error"))     # True
        >>> print(ti.is_logging_effective("critical"))  # True
    """
    return ti_python_core.logging_effective(level)


# ------------------------

DEBUG = "debug"
"""The `str` 'debug', used for the `debug` logging level.
"""
# ------------------------

TRACE = "trace"
"""The `str` 'trace', used for the `debug` logging level.
"""
# ------------------------

INFO = "info"
"""The `str` 'info', used for the `info` logging level.
"""
# ------------------------

WARN = "warn"
"""The `str` 'warn', used for the `warn` logging level.
"""
# ------------------------

ERROR = "error"
"""The `str` 'error', used for the `error` logging level.
"""
# ------------------------

CRITICAL = "critical"
"""The `str` 'critical', used for the `critical` logging level.
"""
# ------------------------

supported_log_levels = [DEBUG, TRACE, INFO, WARN, ERROR, CRITICAL]

debug = _get_logging(DEBUG)
trace = _get_logging(TRACE)
info = _get_logging(INFO)
warn = _get_logging(WARN)
error = _get_logging(ERROR)
critical = _get_logging(CRITICAL)

__all__ = [
    "DEBUG",
    "TRACE",
    "INFO",
    "WARN",
    "ERROR",
    "CRITICAL",
    "set_logging_level",
    "is_logging_effective",
]
