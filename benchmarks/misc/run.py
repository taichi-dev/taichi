import os
import warnings

from membound import MemoryBound
from utils import arch_name, datatime_with_format, dump2json, get_commit_hash

import taichi as ti

benchmark_suites = [MemoryBound]
benchmark_archs = [ti.x64, ti.cuda]


class BenchmarksInfo:

    def __init__(self, pull_request_id: str, commit_hash: str):
        """init with commit info"""
        self.pull_request_id = pull_request_id
        self.commit_hash = commit_hash
        self.datetime = []  #[start, end]
        self.archs = {}
        # "archs": {
        #     "x64": ["memorybound"], #arch:[suites name]
        #     "cuda": ["memorybound"]
        # }
    def add_suites_info(self, arch, suites):
        self.archs[arch_name(arch)] = suites.get_suites_name()


class BenchmarkSuites:

    def __init__(self, arch):
        self._suites = []
        self._arch = arch
        for suite in benchmark_suites:
            if self._check_supported(arch, suite):
                self._suites.append(suite(arch))

    def run(self):
        print(f'Arch [{arch_name(self._arch)}] Running...')
        for suite in self._suites:
            suite.run()

    def save(self, benchmark_dir='./'):
        #folder of archs
        arch_dir = os.path.join(benchmark_dir, arch_name(self._arch))
        os.makedirs(arch_dir)
        for suite in self._suites:
            suite.save_as_json(arch_dir)
            suite.save_as_markdown(arch_dir)

    def get_suites_name(self):
        return [suite.suite_name for suite in self._suites]

    def _check_supported(self, arch, suite):
        if arch in suite.supported_archs:
            return True
        else:
            warnings.warn(
                f'Arch [{arch_name(arch)}] does not exist in {suite.__name__}.supported_archs.',
                UserWarning,
                stacklevel=2)
            return False


def main():

    benchmark_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(benchmark_dir)

    pull_request_id = os.environ.get('PULL_REQUEST_NUMBER')
    commit_hash = get_commit_hash()  #[:8]
    info = BenchmarksInfo(pull_request_id, commit_hash)
    info.datetime.append(datatime_with_format())  #start time

    print(f'pull_request_id = {pull_request_id}')
    print(f'commit_hash = {commit_hash}')

    for arch in benchmark_archs:
        #init & run
        suites = BenchmarkSuites(arch)
        suites.run()
        #save result
        suites.save(benchmark_dir)
        #add benchmark info
        info.add_suites_info(arch, suites)

    info.datetime.append(datatime_with_format())  #end time
    #save commit and benchmark info
    info_path = os.path.join(benchmark_dir, '_info.json')
    info_str = dump2json(info)
    with open(info_path, 'w') as f:
        print(info_str, file=f)


if __name__ == '__main__':
    main()
