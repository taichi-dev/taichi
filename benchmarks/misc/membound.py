import time

from membound_cases import memory_bound_cases_list
from utils import (arch_name, dtype2str, geometric_mean, kibibyte,
                   md_table_header, size2str)

import taichi as ti


class MemoryBound:
    suite_name = 'memorybound'
    supported_archs = [ti.cpu, ti.cuda]
    test_cases = memory_bound_cases_list
    test_dtype_list = [ti.i32, ti.i64, ti.f32, ti.f64]
    test_dsize_list = [(4**i) * kibibyte
                       for i in range(1, 10)]  #[4KB,16KB...256MB]
    basic_repeat_times = 10
    evaluator = [geometric_mean]

    def __init__(self, arch):
        self.arch = arch
        self.cases_impl = []
        for case in self.test_cases:
            for dtype in self.test_dtype_list:
                impl = CaseImpl(case, arch, dtype, self.test_dsize_list,
                                self.evaluator)
                self.cases_impl.append(impl)

    def run(self):
        for case in self.cases_impl:
            case.run()

    def get_markdown_lines(self):
        lines = []
        lines += md_table_header(self.suite_name, self.arch,
                                 self.test_dsize_list, self.basic_repeat_times,
                                 self.evaluator)

        result_header = '|kernel elapsed time(ms)' + ''.join(
            '|' for i in range(
                len(self.test_dsize_list) + len(MemoryBound.evaluator)))
        lines += [result_header]
        for case in self.cases_impl:
            lines += case.get_markdown_lines()
        lines.append('')
        return lines


class CaseImpl:
    def __init__(self, func, arch, test_dtype, test_dsize_list, evaluator):
        self.func = func
        self.name = func.__name__
        self.arch = arch
        self.test_dtype = test_dtype
        self.test_dsize_list = test_dsize_list
        self.min_time_in_us = []  #test results
        self.evaluator = evaluator

    def run(self):
        ti.init(kernel_profiler=True, arch=self.arch)
        print("TestCase[%s.%s.%s]" % (self.func.__name__, arch_name(
            self.arch), dtype2str[self.test_dtype]))
        for test_dsize in self.test_dsize_list:
            print("test_dsize = %s" % (size2str(test_dsize)))
            self.min_time_in_us.append(
                self.func(self.arch, self.test_dtype, test_dsize,
                          MemoryBound.basic_repeat_times))
            time.sleep(0.2)
        ti.reset()

    def get_markdown_lines(self):
        string = '|' + self.name + '.' + dtype2str[self.test_dtype] + '|'
        string += ''.join(
            str(round(time, 4)) + '|' for time in self.min_time_in_us)
        string += ''.join(
            str(round(item(self.min_time_in_us), 4)) + '|'
            for item in self.evaluator)
        return [string]
