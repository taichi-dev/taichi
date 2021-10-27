import os
import warnings

from membound import MemoryBound
from taichi.core import ti_core as _ti_core
from utils import arch_name, datatime_with_format, dump2json

import taichi as ti

benchmark_suites = [MemoryBound]
benchmark_archs = [ti.cpu, ti.cuda]


class CommitInfo:
    def __init__(self, pull_request_id, commit_hash):
        self.pull_request_id = pull_request_id
        self.commit_hash = commit_hash  #str
        self.archs = []  #['x64','cuda','vulkan', ...]
        self.datetime = []  #[start, end]


class BenchmarkSuites:
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
            warnings.warn(
                f'Arch [{arch_name(arch)}] does not exist in {suite.__name__}.supported_archs.',
                UserWarning,
                stacklevel=2)
            return False

    def run(self):
        print(f'Arch [{arch_name(self.arch)}] Running...')
        for suite in self.suites:
            suite.run()

    def save_to_markdown(self, arch_dir='./'):
        current_time = datatime_with_format()
        commit_hash = _ti_core.get_commit_hash()  #[:8]
        for suite in self.suites:
            file_name = f'{suite.suite_name}.md'
            path = os.path.join(arch_dir, file_name)
            with open(path, 'w') as f:
                lines = [
                    f'commit_hash: {commit_hash}\n',
                    f'datatime: {current_time}\n'
                ]
                lines += suite.get_markdown_lines()
                for line in lines:
                    print(line, file=f)

    def save_to_json(self, file_dir='./'):
        #arch info
        arch_dict = {}
        arch_dict['arch_name'] = arch_name(self.arch)
        arch_dict['suites'] = [suite.suite_name for suite in self.suites]
        info_path = os.path.join(file_dir, '_info.json')
        info_str = dump2json(arch_dict)
        with open(info_path, 'w') as f:
            print(info_str, file=f)
        #suite info
        for suite in self.suites:
            #suite folder
            suite_path = os.path.join(file_dir, suite.suite_name)
            os.makedirs(suite_path)
            #suite info
            info_path = os.path.join(suite_path, '_info.json')
            info_str = suite.get_suite_info()
            with open(info_path, 'w') as f:
                print(info_str, file=f)
            #cases info
            suite.save_to_json(suite_path)


def main():

    benchmark_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(benchmark_dir)

    pull_request_id = os.environ.get('PULL_REQUEST_NUMBER')
    commit_hash = _ti_core.get_commit_hash()  #[:8]
    print(f'pull_request_id = {pull_request_id}')
    print(f'commit_hash = {commit_hash}')
    info = CommitInfo(pull_request_id, commit_hash)
    info.datetime.append(datatime_with_format())  #start time

    for arch in benchmark_archs:
        #init & run
        suites = BenchmarkSuites(arch)
        suites.run()
        #make dir
        arch_dir = os.path.join(benchmark_dir, arch_name(arch))
        os.makedirs(arch_dir)
        #save result
        suites.save_to_markdown(arch_dir)
        suites.save_to_json(arch_dir)
    info.datetime.append(datatime_with_format())  #end time
    #save commit and benchmark info
    info_path = os.path.join(benchmark_dir, '_info.json')
    info_str = dump2json(info)
    with open(info_path, 'w') as f:
        print(info_str, file=f)


if __name__ == '__main__':
    main()
