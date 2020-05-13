import os
import taichi as ti


class Case:
    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.records = {}

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name

    def pprint(self):
        print(f' * {self.name[10:]:33}', end='')
        for i, arch in enumerate(sorted(self.records.keys())):
            ms = self.records[arch] * 1000
            arch_name = str(arch)[5:]
            print(f' {arch_name:8} {ms:7.3f} ms', end='')
            if i < len(self.records) - 1:
                print('      ', end='')
        print()

    def run(self, arch):
        ti.init(arch=arch)
        t = self.func()
        self.records[arch] = t


class Suite:
    def __init__(self, filename):
        self.cases = []
        print(filename)
        self.name = filename[:-3]
        loc = {}
        exec(f'import {self.name} as suite', {}, loc)
        suite = loc['suite']
        case_keys = list(
            sorted(filter(lambda x: x.startswith('benchmark_'), dir(suite))))
        self.cases = [Case(k, getattr(suite, k)) for k in case_keys]

    def print(self):
        print(f'{self.name}:')
        for b in self.cases:
            b.pprint()

    def run(self, arch):
        print(f'{self.name}:')
        for case in sorted(self.cases):
            case.run(arch)


class TaichiBenchmark:
    def __init__(self):
        self.suites = []
        benchmark_dir = os.path.dirname(__file__)
        for f in map(os.path.basename, sorted(os.listdir(benchmark_dir))):
            if f != 'run.py' and f.endswith('.py') and f[0] != '_':
                self.suites.append(Suite(f))

    def pprint(self):
        for s in self.suites:
            s.print()

    def run(self, arch):
        print("Running...")
        for s in self.suites:
            s.run(arch)


b = TaichiBenchmark()
b.pprint()
b.run(ti.x64)
b.run(ti.cuda)
print()
b.pprint()
