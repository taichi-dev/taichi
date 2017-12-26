import taichi as tc

import inspect

def get_file_name(asc=0):
    return inspect.stack()[1 + asc][1]

def get_function_name(asc=0):
    return inspect.stack()[1 + asc][3]

def get_line_number(asc=0):
    return inspect.stack()[1 + asc][2]

def log_info(fmt, *args, **kwargs):
    tc.core.log_info(fmt.format(*args, **kwargs))

def get_logging(name):
    def logger(msg, *args, **kwargs):
        msg_formatted = msg.format(*args, **kwargs)
        func = getattr(tc.core, name)
        func('[{}:{}@{}] {}'.format(get_file_name(1), get_function_name(1), get_line_number(1), msg_formatted))

    return logger

log_info = get_logging('info')

def main():
    log_info('test_logging, a = {}, b = {b}', 10, b=123)

main()
