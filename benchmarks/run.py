import os
import warnings

from suite_microbenchmarks import MicroBenchmark
from utils import datatime_with_format, dump2json, get_commit_hash

import taichi as ti

benchmark_suites = [MicroBenchmark]


class BenchmarkInfo:
    def __init__(self):
        """init with commit info"""
        self.commit_hash = get_commit_hash()  #[:8]
        self.datetime = [datatime_with_format()]  #init with start time
        self.suites = {}
        print(f'commit_hash = {self.commit_hash}')


class BenchmarkSuites:
    def __init__(self):
        self._suites = []
        for suite in benchmark_suites:
            self._suites.append(suite())

    def run(self):
        for suite in self._suites:
            suite.run()

    def save(self, benchmark_dir='./'):
        for suite in self._suites:
            suite_dir = os.path.join(benchmark_dir, suite.suite_name)
            os.makedirs(suite_dir, exist_ok=True)
            suite.save_as_json(suite_dir)

    def get_suites_info(self):
        info_dict = {}
        for suite in self._suites:
            info_dict[suite.suite_name] = suite.get_benchmark_info()
        return info_dict


def main():

    benchmark_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(benchmark_dir, exist_ok=True)

    #init & run
    info = BenchmarkInfo()
    suites = BenchmarkSuites()
    suites.run()
    #save result
    suites.save(benchmark_dir)
    #add benchmark info
    info.suites = suites.get_suites_info()
    info.datetime.append(datatime_with_format())  #end time

    #save benchmark info
    info_path = os.path.join(benchmark_dir, '_info.json')
    info_str = dump2json(info)
    with open(info_path, 'w') as f:
        print(info_str, file=f)


if __name__ == '__main__':
    main()
