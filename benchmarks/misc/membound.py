import time

from membound_cases import fill, reduction, saxpy
from utils import *

import taichi as ti

test_cases = [fill, saxpy, reduction]
test_archs = [ti.cuda]
test_dtype = [ti.i32, ti.i64, ti.f32, ti.f64]
test_dsize = [(4**i) * kibibyte for i in range(1, 11)]  #[4KB,16KB...1GB]
test_repeat = 10
results_evaluation = [geometric_mean]


class BenchmarkResult:
    def __init__(self, name, arch, dtype, dsize, results_evaluation):
        self.test_name = name
        self.test_arch = arch
        self.data_type = dtype
        self.data_size = dsize
        self.min_time_in_us = []
        self.results_evaluation = results_evaluation

    def time2mdtableline(self):
        string = '|' + self.test_name + '.' + dtype2str[self.data_type] + '|'
        string += ''.join(
            str(round(time, 4)) + '|' for time in self.min_time_in_us)
        string += ''.join(
            str(round(item(self.min_time_in_us), 4)) + '|'
            for item in self.results_evaluation)
        return string


class BenchmarkImpl:
    def __init__(self, func, archs, data_type, data_size):
        self.func = func
        self.name = func.__name__
        self.env = None
        self.device = None
        self.archs = archs
        self.data_type = data_type
        self.data_size = data_size
        self.benchmark_results = []

    def run(self):
        for arch in self.archs:
            for dtype in self.data_type:
                ti.init(kernel_profiler=True, arch=arch)
                print("TestCase[%s.%s.%s]" %
                      (self.func.__name__, ti.core.arch_name(arch),
                       dtype2str[dtype]))
                result = BenchmarkResult(self.name, arch, dtype,
                                         self.data_size, results_evaluation)
                for size in self.data_size:
                    print("data_size = %s" % (size2str(size)))
                    result.min_time_in_us.append(
                        self.func(arch, dtype, size, test_repeat))
                    time.sleep(0.2)
                self.benchmark_results.append(result)

    def print(self):
        i = 0
        for arch in self.archs:
            for dtype in self.data_type:
                for idx in range(len(self.data_size)):
                    print(
                        "    test_case:[%s] arch:[%s] dtype:[%s] dsize:[%7s] >>> time:[%4.4f]"
                        %
                        (self.name, ti.core.arch_name(arch), dtype2str[dtype],
                         size2str(self.benchmark_results[i].data_size[idx]),
                         self.benchmark_results[i].min_time_in_us[idx]))
                i = i + 1

    def save2markdown(self, arch):
        header = '|kernel elapsed time(ms)' + ''.join(
            '|' for i in range(len(self.data_size) + len(results_evaluation)))
        lines = [header]
        for result in self.benchmark_results:
            if (result.test_arch == arch):
                lines.append(result.time2mdtableline())
        return lines


class Membound:
    benchmark_imps = []

    def __init__(self):
        for case in test_cases:
            self.benchmark_imps.append(
                BenchmarkImpl(case, test_archs, test_dtype, test_dsize))

    def run(self):
        for case in self.benchmark_imps:
            case.run()

    def mdlines(self, arch):
        lines = []
        lines += md_table_header(self.__class__.__name__, arch, test_dsize,
                                 test_repeat, results_evaluation)
        for case in self.benchmark_imps:
            if arch in case.archs:
                lines += case.save2markdown(arch)
            else:
                continue
        lines.append('')
        return lines
