import inspect
import os

from taichi.core import util


def get_logging(name):
    def logger(msg, *args, **kwargs):
        # Python inspection takes time (~0.1ms) so avoid it as much as possible
        if util.ti_core.logging_effective(name):
            msg_formatted = msg.format(*args, **kwargs)
            func = getattr(util.ti_core, name)
            frame = inspect.currentframe().f_back
            file_name, lineno, func_name, _, _ = inspect.getframeinfo(frame)
            file_name = os.path.basename(file_name)
            msg = f'[{file_name}:{func_name}@{lineno}] {msg_formatted}'
            func(msg)

    return logger


def set_logging_level(level):
    util.ti_core.set_logging_level(level)


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


def _get_file_name(asc=0):
    return inspect.stack()[1 + asc][1]


def _get_function_name(asc=0):
    return inspect.stack()[1 + asc][3]


def _get_line_number(asc=0):
    return inspect.stack()[1 + asc][2]
