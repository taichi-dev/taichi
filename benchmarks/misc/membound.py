import time

from membound_cases import fill, reduction, saxpy
from utils import (arch_name, dtype2str, geometric_mean, kibibyte,
                   md_table_header, size2str)

import taichi as ti


class SuiteInfo:
    cases = [fill, saxpy, reduction]
    supported_archs = [ti.cpu, ti.cuda]
    dtype = [ti.i32, ti.i64, ti.f32, ti.f64]
    dsize = [(4**i) * kibibyte for i in range(1, 10)]  #[4KB,16KB...256MB]
    repeat = 10
    evaluator = [geometric_mean]


class CaseResult:
    def __init__(self, name, arch, dtype, dsize, evaluator):
        self.test_name = name
        self.test_arch = arch
        self.data_type = dtype
        self.data_size = dsize  #list
        self.min_time_in_us = []  #list
        self.evaluator = evaluator

    def result_to_markdown(self):
        string = '|' + self.test_name + '.' + dtype2str[self.data_type] + '|'
        string += ''.join(
            str(round(time, 4)) + '|' for time in self.min_time_in_us)
        string += ''.join(
            str(round(item(self.min_time_in_us), 4)) + '|'
            for item in self.evaluator)
        return string


class CaseImpl:
    def __init__(self, func, arch, data_type, data_size):
        self.func = func
        self.name = func.__name__
        self.env = None
        self.device = None
        self.arch = arch
        self.data_type = data_type
        self.data_size = data_size
        self.case_results = []

    def run(self):
        for dtype in self.data_type:
            ti.init(kernel_profiler=True, arch=self.arch)
            print("TestCase[%s.%s.%s]" %
                  (self.func.__name__, arch_name(self.arch), dtype2str[dtype]))
            result = CaseResult(self.name, self.arch, dtype, self.data_size,
                                SuiteInfo.evaluator)
            for size in self.data_size:
                print("data_size = %s" % (size2str(size)))
                result.min_time_in_us.append(
                    self.func(self.arch, dtype, size, SuiteInfo.repeat))
                time.sleep(0.2)
            self.case_results.append(result)

    def to_markdown(self):
        header = '|kernel elapsed time(ms)' + ''.join(
            '|' for i in range(len(self.data_size) + len(SuiteInfo.evaluator)))
        lines = [header]
        for result in self.case_results:
            lines.append(result.result_to_markdown())
        return lines


class MemoryBound:
    suite_name = 'memorybound'
    supported_archs = SuiteInfo.supported_archs

    def __init__(self, arch):
        self.arch = arch
        self.cases_impl = []
        for case in SuiteInfo.cases:
            self.cases_impl.append(
                CaseImpl(case, arch, SuiteInfo.dtype, SuiteInfo.dsize))

    def run(self):
        for case in self.cases_impl:
            case.run()

    def get_markdown_str(self):
        lines = []
        lines += md_table_header(self.suite_name, self.arch, SuiteInfo.dsize,
                                 SuiteInfo.repeat, SuiteInfo.evaluator)
        for case in self.cases_impl:
            lines += case.to_markdown()
        lines.append('')
        return lines
