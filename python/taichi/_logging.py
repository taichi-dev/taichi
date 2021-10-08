import inspect
import os

from taichi.core import ti_core


def _get_logging(name):
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
    ti_core.set_logging_level(level)


def is_logging_effective(level):
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
