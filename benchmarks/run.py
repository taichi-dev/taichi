import os
import taichi as ti


def get_benchmark_dir():
    return os.path.join(ti.core.get_repo_dir(), 'benchmarks')


class Case:
    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.records = {}

    def stat_write(self, avg):
        arch_name = ti.core.arch_name(ti.cfg.arch)
        output_dir = os.environ.get('TI_BENCHMARK_OUTPUT_DIR', '.')
        filename = f'{output_dir}/{self.name}__arch_{arch_name}.dat'
        with open(filename, 'w') as f:
            f.write(f'time_avg: {avg:.4f}')

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name

    def run(self):
        avg = self.func()
        self.stat_write(avg)


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

    def run(self):
        print(f'{self.name}:')
        for case in sorted(self.cases):
            case.run()


class TaichiBenchmark:
    def __init__(self):
        self.suites = []
        benchmark_dir = get_benchmark_dir()
        for f in map(os.path.basename, sorted(os.listdir(benchmark_dir))):
            if f != 'run.py' and f.endswith('.py') and f[0] != '_':
                self.suites.append(Suite(f))

    def run(self):
        print("Running...")
        for s in self.suites:
            s.run()


b = TaichiBenchmark()
b.run()
