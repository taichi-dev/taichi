import os
import warnings
import argparse

from suite_microbenchmarks import MicroBenchmark
from taichi._lib import core as ti_python_core
from utils import datatime_with_format, dump2json

benchmark_suites = [MicroBenchmark]


class BenchmarkInfo:
    def __init__(self):
        """init with commit info"""
        self.commit_hash = ti_python_core.get_commit_hash()
        self.datetime = datatime_with_format()
        self.suites = {}
        print(f"commit_hash = {self.commit_hash}")


class BenchmarkSuites:
    def __init__(self):
        self._suites = []
        for suite in benchmark_suites:
            self._suites.append(suite())

    def run(self, arch, benchmark_plan):
        for suite in self._suites:
            suite.run(arch, benchmark_plan)

    def save(self, benchmark_dir="./"):
        for suite in self._suites:
            suite_dir = os.path.join(benchmark_dir, suite.suite_name)
            os.makedirs(suite_dir, exist_ok=True)
            suite.save_as_json(suite_dir)

    def get_suites_info(self):
        info_dict = {}
        for suite in self._suites:
            info_dict[suite.suite_name] = suite.get_benchmark_info()
        return info_dict

def parse_cmdln():
    parser = argparse.ArgumentParser(prog='run.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--arch",
            choices=['amdgpu', 'cuda', 'vulkan', 'opengl', 'metal', 'x64'],
            required=True, help="Architecture to benchmark")
    parser.add_argument("--benchmark_plan", 
            choices=['AtomicOpsPlan', 'FillPlan', 'MathOpsPlan', 
                'MatrixOpsPlan', 'MemcpyPlan', 'SaxpyPlan', 'Stencil2DPlan'],
            required=True, help="Benchmark plan to run")
    args = parser.parse_args()
    return args

def main():
    args = parse_cmdln()
    benchmark_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(benchmark_dir, exist_ok=True)

    # init & run
    info = BenchmarkInfo()
    suites = BenchmarkSuites()
    suites.run(args.arch, args.benchmark_plan)
    # save benchmark results & info
    suites.save(benchmark_dir)
    info.suites = suites.get_suites_info()
    info_path = os.path.join(benchmark_dir, "_info.json")
    info_str = dump2json(info)
    with open(info_path, "w") as f:
        print(info_str, file=f)


if __name__ == "__main__":
    main()
