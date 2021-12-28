import os
import sys

from utils import get_benchmark_dir

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

    def run(self):
        print(f'==> {self.name}:')
        os.environ['TI_CURRENT_BENCHMARK'] = self.name
        self.func()


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
        output_dir = os.environ.get('TI_BENCHMARK_OUTPUT_DIR', '.')
        filename = f'{output_dir}/benchmark.yml'
        try:
            with open(filename, 'r+') as f:
                f.truncate()  # clear the previous result
        except FileNotFoundError:
            pass
        print("Running...")
        for s in self.suites:
            s.run()


b = TaichiBenchmark()
b.run()
