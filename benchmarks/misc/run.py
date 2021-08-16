from membound import Membound

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

    def write_md(self):
        filename = f'performance_result.md'
        with open(filename, 'w') as f:
            for arch in test_archs:
                for s in self.suites:
                    lines = s.mdlines(arch)
                    for line in lines:
                        print(line, file=f)


p = PerformanceMonitoring()
p.run()
p.write_md()
