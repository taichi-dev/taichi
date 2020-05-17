import os
import taichi as ti


def get_benchmark_dir():
    return os.path.join(ti.core.get_repo_dir(), 'benchmarks')


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

    def save_result(self):
        output_dir = os.environ.get('TI_BENCHMARK_OUTPUT_DIR', '.')
        for i, arch in enumerate(sorted(self.records.keys())):
            arch_name = str(arch)[5:]
            filename = f'{output_dir}/{self.name}__arch_{arch_name}.dat'
            with open(filename, 'w') as f:
                ms = self.records[arch] * 1000
                f.write(f'record_time: {ms:8.4f}\n')

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

    def save_result(self):
        for b in self.cases:
            b.save_result()

    def run(self, arch):
        print(f'{self.name}:')
        for case in sorted(self.cases):
            case.run(arch)


class TaichiBenchmark:
    def __init__(self):
        self.suites = []
        benchmark_dir = get_benchmark_dir()
        for f in map(os.path.basename, sorted(os.listdir(benchmark_dir))):
            if f != 'run.py' and f.endswith('.py') and f[0] != '_':
                self.suites.append(Suite(f))

    def pprint(self):
        for s in self.suites:
            s.print()

    def save_result(self):
        for s in self.suites:
            s.save_result()

    def run(self, arch):
        print("Running...")
        for s in self.suites:
            s.run(arch)


b = TaichiBenchmark()
b.pprint()
b.run(ti.x64)
#b.run(ti.cuda)
print()
b.pprint()
b.save_result()
