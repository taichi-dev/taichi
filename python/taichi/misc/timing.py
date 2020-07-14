import time
from colorama import Fore, Back, Style


def _m(sec):
    ms = sec * 1000
    if sec > 0.099:
        return f'{Fore.LIGHTRED_EX}{sec:6.3f}{Fore.RESET}s'
    else:
        return f'{Fore.LIGHTMAGENTA_EX}{ms:5.2f}{Fore.RESET}ms'


def _c(cnt):
    return f'{Fore.MAGENTA}{cnt:3d}{Fore.RESET}x'


def _n(name):
    return f'{Fore.MAGENTA}{name}{Fore.RESET}'


class PythonProfiler:
    def __init__(self, timer=time.time):
        self.timer = timer
        self.records = {}
        self.started = {}
        self.last_started = None

    def start(self, name):
        t = self.timer()
        self.started[name] = t
        self.last_started = name

    def stop(self, name=None):
        name = name or self.last_started
        t = self.timer()
        diff = t - self.started[name]
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(diff)

    def __call__(self, name=None):
        if self.last_started:
            self.stop()
        if name:
            self.start(name)

    def print(self, name=None):
        print('  min   |   avg   |   max   |  nr  |    name')
        for name in self.records.keys() if name is None else [name]:
            rec = self.records[name]
            maximum = max(rec)
            minimum = min(rec)
            avg = sum(rec) / len(rec)
            cnt = len(rec)
            print(f'{_m(minimum)} | {_m(avg)} | {_m(maximum)} | {_c(cnt)} | {_n(name)}')

    def timed(self, name):
        if callable(name):
            foo = name
            name = foo.__name__
            return decorator(name)

        def decorator(foo):
            def wrapped(*args, **kwargs):
                self.start(name)
                ret = foo(*args, **kwargs)
                self.stop(name)
                self.print(name)
                return ret

            return wrapped

        return decorator



profiler = PythonProfiler()
