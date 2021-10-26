import os

from membound import MemoryBound
from taichi.core import ti_core as _ti_core
from utils import arch_name, datatime_with_format

import taichi as ti

benchmark_suites = [MemoryBound]
benchmark_archs = [ti.cpu, ti.cuda]


class BenchmarkInfo:
    def __init__(self, pull_request_id, commit_hash):
        self.pull_request_id = pull_request_id  #int
        self.commit_hash = commit_hash  #str
        self.archs = []  #list ['x64','CUDA','Vulkan', ...]
        self.datetime = []  #list [begin, end]


class PerformanceMonitoring:
    def __init__(self, arch):
        self.suites = []
        self.arch = arch
        for suite in benchmark_suites:
            if self.check_supported(arch, suite):
                self.suites.append(suite(arch))

    def check_supported(self, arch, suite):
        if arch in suite.supported_archs:
            return True
        else:
            RuntimeWarning(
                'arch[' + arch_name(arch) +
                '] does not exist in SuiteInfo.supported_archs of class ' +
                suite.__name__)
            return False

    def run(self):
        print(f'Arch : {arch_name(self.arch)} Running...')
        for suite in self.suites:
            suite.run()

    def save_to_markdown(self, arch_dir='./'):
        current_time = datatime_with_format()
        commit_hash = _ti_core.get_commit_hash()  #[:8]
        for s in self.suites:
            file_name = f'{s.suite_name}.md'
            path = os.path.join(arch_dir, file_name)
            with open(path, 'w') as f:
                lines = [
                    f'commit_hash: {commit_hash}\n',
                    f'datatime: {current_time}\n'
                ] + s.get_markdown_str()
                for line in lines:
                    print(line, file=f)


def main():

    benchmark_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(benchmark_dir)

    for arch in benchmark_archs:
        #make dir
        arch_dir = os.path.join(benchmark_dir, arch_name(arch))
        os.makedirs(arch_dir)
        #init & run
        impl = PerformanceMonitoring(arch)
        impl.run()
        #save result
        impl.save_to_markdown(arch_dir)


if __name__ == '__main__':
    main()
