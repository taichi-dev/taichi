import os

from baseline import Baseline

test_suites = [Baseline]


class PerformanceMonitoring:
    def run(self):
        filename = f'performance_result.md'
        try:
            with open(filename, 'r+') as f:
                f.truncate()  # clear the previous result
        except FileNotFoundError:
            pass
        print("Running...")
        for s in test_suites:
            imp = s()
            imp.run()
            lines = imp.mdlines()
            f = open(filename, 'a')
            for line in lines:
                f.write(line + '\n')
            f.close()


p = PerformanceMonitoring()
p.run()
