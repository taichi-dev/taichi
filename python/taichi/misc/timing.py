import time, functools
from colorama import Fore, Back, Style


def _m(sec):
    if sec > 9.9:
        return f'{Fore.LIGHTRED_EX}{sec:6.1f}{Fore.RESET}s'
    elif sec > 0.099:
        return f'{Fore.LIGHTRED_EX}{sec:6.3f}{Fore.RESET}s'
    else:
        ms = sec * 1000
        if ms > 0.099:
            return f'{Fore.LIGHTMAGENTA_EX}{ms:5.2f}{Fore.RESET}ms'
        else:
            ns = ms * 1000
            return f'{Fore.BLUE}{ns:5.2f}{Fore.RESET}ns'


def _c(cnt):
    return f'{Fore.MAGENTA}{cnt:4d}{Fore.RESET}x'


def _n(name):
    return f'{Fore.MAGENTA}{name}{Fore.RESET}'


class RecordStatistics:
    def __init__(self, profiler, name):
        self.record = profiler.records[name]
        self.options = profiler.options[name]
        self.rec = self.record

        warmup = self.options.get('warmup', 0)
        self.rec = self.rec[warmup:]

    @property
    def min(self):
        return min(self.rec)

    @property
    def max(self):
        return max(self.rec)

    @property
    def total(self):
        return sum(self.rec)

    @property
    def avg(self):
        return sum(self.rec) / len(self.rec)

    @property
    def count(self):
        return len(self.rec)


class PythonProfiler:
    def __init__(self, timer=time.time):
        self.timer = timer
        self.records = {}
        self.started = {}
        self.options = {}
        self.last_started = None

    def start(self, name, **options):
        self.last_started = name
        self.options[name] = self.options.get(name, options)
        t = self.timer()
        self.started[name] = t

    def stop(self, name=None):
        t = self.timer()
        if not name and self.last_started:
            name = self.last_started
            self.last_started = None
        diff = t - self.started[name]
        del self.started[name]
        self.last_started = None
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(diff)

    def __call__(self, name=None, **options):
        if self.last_started:
            self.stop()
        if name is not None:
            self.start(name, **options)

    def print(self, name=None):
        print('  min   |   avg   |   max   |  num  |  total  |    name')
        if name is not None:
            names = [name]
        else:
            def keyfunc(name):
                rec = RecordStatistics(self, name)
                return -rec.total
            names = sorted(self.records.keys(), key=keyfunc)
        for name in names:
            rec = RecordStatistics(self, name)
            print(f'{_m(rec.min)} | {_m(rec.avg)} | {_m(rec.max)} | {_c(rec.count)} '
                  f'| {_m(rec.total)} | {_n(name)}')

    def timed(self, name=None, **options):
        if callable(name):
            foo = name
            name = None
        else:
            foo = None

        def decorator(foo):
            if decorator.name is None:
                name = foo.__name__
            else:
                name = decorator.name

            @functools.wraps(foo)
            def wrapped(*args, **kwargs):
                self.start(name, **decorator.options)
                ret = foo(*args, **kwargs)
                self.stop(name)
                return ret

            return wrapped

        decorator.name = name
        decorator.options = options

        return decorator(foo) if foo is not None else decorator



profiler = PythonProfiler()
