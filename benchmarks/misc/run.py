from membound import Membound

import taichi as ti

test_suites = [Membound]
test_archs = [ti.cuda]


class PerformanceMonitoring:
    impls = []
    filename = f'performance_result.md'

    def __init__(self):
        for s in test_suites:
            self.impls.append(s())

    def run(self):
        print("Running...")
        for s in self.impls:
            s.run()

    def write_md(self):
        try:
        with open(self.filename, 'w') as f:
            for arch in test_archs:
                for impl in self.impls:
                    lines = impl.mdlines(arch)
                    for line in lines:
                        print(line, file=f)


p = PerformanceMonitoring()
p.run()
p.write_md()
