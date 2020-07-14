import time, functools
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


class RecordStatistics:
    def __init__(self, rec):
        self.rec = rec

    @property
    def min(self):
        return min(self.rec)

    @property
    def max(self):
        return max(self.rec)

    @property
    def total(self):
        return sum(rec)

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
        if name is not None:
            names = [name]
        else:
            def keyfunc(name):
                rec = RecordStatistics(self.records[name])
                return -rec.avg
            names = sorted(self.records.keys(), key=keyfunc)
        for name in names:
            rec = RecordStatistics(self.records[name])
            print(f'{_m(rec.min)} | {_m(rec.avg)} | {_m(rec.max)} | {_c(rec.count)} '
                  f'| {_n(name)}')

    def timed(self, name=None, warmup=0):
        if callable(name):
            foo = name
            name = None
        else:
            foo = None

        def deco(foo):
            if deco.name is None:
                name = foo.__name__
            else:
                name = deco.name

            @functools.wraps(foo)
            def wrapped(*args, **kwargs):
                if deco.warmup > 0:
                    deco.warmup -= 1
                    return foo(*args, **kwargs)

                self.start(name)
                ret = foo(*args, **kwargs)
                self.stop(name)
                return ret

            return wrapped

        deco.name = name
        deco.warmup = warmup

        return deco(foo) if foo is not None else deco



profiler = PythonProfiler()
