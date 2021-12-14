import inspect
import os

from taichi._lib import core as ti_core


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
        if ti_core.logging_effective(name):
            msg_formatted = msg.format(*args, **kwargs)
            func = getattr(ti_core, name)
            frame = inspect.currentframe().f_back
            file_name, lineno, func_name, _, _ = inspect.getframeinfo(frame)
            file_name = os.path.basename(file_name)
            msg = f'[{file_name}:{func_name}@{lineno}] {msg_formatted}'
            func(msg)

    return logger


def set_logging_level(level):
    """Controls the level of detail in logs.

    See also https://docs.taichi.graphics/lang/articles/contribution/utilities#logging.

    Args:
        level (str): The string represents logging level.
            Effective levels include: 'trace', 'debug', 'info', 'warn', 'error', 'critical'.

    Example:
            >>> set_logging_level('debug')

        If call this function, then everything below 'debug' will be effective,
        and 'trace' won't since it's above debug.
    """
    ti_core.set_logging_level(level)


def is_logging_effective(level):
    """Check if the level is effective. The level below current level will be effective.
        If not set by manual, the pre-set logging level is 'info'.

    See also https://docs.taichi.graphics/lang/articles/contribution/utilities#logging.

    Args:
        level (str): The string represents logging level.
            Effective levels include: 'trace', 'debug', 'info', 'warn', 'error', 'critical'.

    Returns:
        Bool: Indicate whether the logging level is supported.

    Example:
        If current level is 'info':

            >>> print(ti.is_logging_effective("trace"))     # False
            >>> print(ti.is_logging_effective("debug"))     # False
            >>> print(ti.is_logging_effective("info"))      # True
            >>> print(ti.is_logging_effective("warn"))      # True
            >>> print(ti.is_logging_effective("error"))     # True
            >>> print(ti.is_logging_effective("critical"))  # True
    """
    return ti_core.logging_effective(level)


DEBUG = 'debug'
TRACE = 'trace'
INFO = 'info'
WARN = 'warn'
ERROR = 'error'
CRITICAL = 'critical'

supported_log_levels = [DEBUG, TRACE, INFO, WARN, ERROR, CRITICAL]

debug = _get_logging(DEBUG)
trace = _get_logging(TRACE)
info = _get_logging(INFO)
warn = _get_logging(WARN)
error = _get_logging(ERROR)
critical = _get_logging(CRITICAL)

__all__ = [
    'DEBUG', 'TRACE', 'INFO', 'WARN', 'ERROR', 'CRITICAL', 'debug', 'trace',
    'info', 'warn', 'error', 'critical', 'supported_log_levels',
    'set_logging_level', 'is_logging_effective'
]
