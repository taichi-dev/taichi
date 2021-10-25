import datetime
import os
import sys

from membound import Membound
from taichi.core import ti_core as _ti_core

import taichi as ti

test_suites = [Membound]
test_archs = [ti.cuda]


class PerformanceMonitoring:
    suites = []

    def __init__(self):
        for s in test_suites:
            self.suites.append(s())

    def run(self):
        print("Running...")
        for s in self.suites:
            s.run()

    def store_to_path(self, path_with_file_name='./performance_result.md'):
        with open(path_with_file_name, 'w') as f:
            for arch in test_archs:
                for s in self.suites:
                    lines = s.mdlines(arch)
                    for line in lines:
                        print(line, file=f)

    def store_with_date_and_commit_id(self, file_dir='./'):
        current_time = datetime.datetime.now().strftime("%Y%m%dd%Hh%Mm%Ss")
        commit_hash = _ti_core.get_commit_hash()[:8]
        file_name = f'perfresult_{current_time}_{commit_hash}.md'
        path = os.path.join(file_dir, file_name)
        print('Storing benchmark result to: ' + path)
        self.store_to_path(path)


def main():
    file_dir = sys.argv[1] if len(sys.argv) > 1 else './'
    p = PerformanceMonitoring()
    p.run()
    p.store_to_path()  # for /benchmark
    p.store_with_date_and_commit_id(file_dir)  #for postsubmit


if __name__ == '__main__':
    main()
