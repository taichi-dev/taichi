import taichi as ti
from baseline import baseline

test_suites = [baseline]

class PerformanceMonitor:
    def run(self):
        for s in test_suites:
            s()

p = PerformanceMonitor()
p.run()