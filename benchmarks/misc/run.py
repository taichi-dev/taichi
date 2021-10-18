import datetime
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

    def write_to_path(self, path_with_file_name='./performance_result.md'):
        with open(path_with_file_name, 'w') as f:
            for arch in test_archs:
                for s in self.suites:
                    lines = s.mdlines(arch)
                    for line in lines:
                        print(line, file=f)

    def store_with_date_and_commit_hash(self, path='./'):
        current_time = datetime.datetime.now().strftime("%Y%m%dd%Hh%Mm%Ss")
        commit_hash = _ti_core.get_commit_hash()[:8]
        filename = f'perfresult_{current_time}_{commit_hash}.md'
        print('store to: ' + path + filename)
        self.write_to_path(path + filename)


path_to_store = sys.argv[1] if len(sys.argv) > 1 else './'
p = PerformanceMonitoring()
p.run()
p.write_to_path()  # for /benchamark
p.store_with_date_and_commit_hash(path_to_store)  #for post-submit
