import functools
import sys
import traceback

from colorama import Fore, Style


def enable_excepthook():
    def excepthook(exctype, value, tb):
        skip = 0
        back = 4
        forward = 2
        bar = f'{Fore.LIGHTBLACK_EX}{"-"*44}{Fore.RESET}'
        print(
            f'{Fore.LIGHTBLACK_EX}========== Taichi Stack Traceback =========={Fore.RESET}'
        )
        for frame, lineno in traceback.walk_tb(tb):
            name = frame.f_code.co_name
            filename = frame.f_code.co_filename
            if '_taichi_skip_traceback' in frame.f_locals:
                skip = frame.f_locals['_taichi_skip_traceback']
            if skip > 0:
                skip -= 1
                continue
            print(
                f'In {Fore.LIGHTYELLOW_EX}{name}{Fore.RESET}() at {Fore.LIGHTMAGENTA_EX}{filename}{Fore.RESET}:{Fore.LIGHTCYAN_EX}{lineno}{Fore.RESET}:\n{bar}'
            )
            with open(filename) as f:
                lines = [''] + f.readlines()
                if lines[lineno][-1] == '\n':
                    lines[lineno] = lines[lineno][:-1]
                lines[lineno] = f'{Fore.LIGHTRED_EX}' + lines[
                    lineno] + f'  {Fore.LIGHTYELLOW_EX}<--{Fore.LIGHTBLACK_EX}\n'
                line = ''.join(lines[max(1, lineno -
                                         back):min(len(lines), lineno +
                                                   forward + 1)])
            if line[-1] != '\n':
                line += '\n'
            print(f'{Fore.LIGHTWHITE_EX}{line}{bar}')
        value = str(value)
        if len(value):
            value = f': {Fore.LIGHTRED_EX}{value}'
        print(
            f'{Fore.LIGHTGREEN_EX}{exctype.__name__}{Fore.RESET}{value}{Fore.RESET}'
        )

    if sys.excepthook is not excepthook:
        sys.excepthook = excepthook
