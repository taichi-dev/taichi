import time

from baseline_fill import fill
from baseline_reduction import reduction
from utils import *

import taichi as ti

ti.init(kernel_profiler=True)

test_cases = [fill, reduction]
test_archs = [ti.cpu, ti.cuda]
test_dtype = [ti.i32, ti.i64, ti.f32, ti.f64]
test_dsize = [
    4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864,
    268435456, 1073741824
]
# size in byte [4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, 64MB, 256MB, 1024MB]
test_repeat = 10


class TestResult:
    def __init__(self, name, arch, dtype, dsize):
        self.test_name = name
        self.test_arch = arch
        self.data_type = dtype
        self.data_size = dsize
        self.min_time_in_us = []

    def time2mdtableline(self):
        string = '|' + self.test_name + '.' + dtype2str[self.data_type] + '|'
        string += ''.join(
            str(round(time, 4)) + '|' for time in self.min_time_in_us)
        return string


class CaseImpl:
    def __init__(self, func, archs, data_type, data_size):
        self.func = func
        self.name = func.__name__
        self.env = None
        self.device = None
        self.archs = archs
        self.data_type = data_type
        self.data_size = data_size
        self.test_result = []

    def run(self):
        for arch in self.archs:
            for dtype in self.data_type:
                ti.init(kernel_profiler=True, arch=arch)
                print("TestCase[%s.%s.%s]" %
                      (self.func.__name__, ti.core.arch_name(arch),
                       dtype2str[dtype]))
                result = TestResult(self.name, arch, dtype, self.data_size)
                for size in self.data_size:
                    print("data_size = %s" % (size2str(size)))
                    result.min_time_in_us.append(
                        self.func(arch, dtype, size, test_repeat))
                    time.sleep(0.2)
                self.test_result.append(result)

    def print(self):
        i = 0
        for arch in self.archs:
            for dtype in self.data_type:
                for idx in range(len(self.data_size)):
                    print(
                        "    test_case:[%s] arch:[%s] dtype:[%s] dsize:[%7s] >>> time:[%4.4f]"
                        %
                        (self.name, ti.core.arch_name(arch), dtype2str[dtype],
                         size2str(self.test_result[i].data_size[idx]),
                         self.test_result[i].min_time_in_us[idx]))
                i = i + 1

    def save2markdown(self, arch):
        header = '|kernel elapsed time(ms)' + ''.join(
            '|' for i in range(len(self.data_size)))
        lines = [header]
        for result in self.test_result:
            if (result.test_arch == arch):
                lines.append(result.time2mdtableline())
        return lines


class Baseline:
    case_impl = []

    def __init__(self):
        for case in test_cases:
            self.case_impl.append(
                CaseImpl(case, test_archs, test_dtype, test_dsize))

    def md_table_header(suite_name, arch, test_dsize):
        col = len(test_dsize) + 1
        header = '|' + suite_name + '.' + ti.core.arch_name(arch) + ''.join(
            '|' for i in range(col))
        layout = '|:--:|' + ''.join(':--:|' for size in test_dsize)
        size = '|**data_size**|' + ''.join(
            size2str(size) + '|' for size in test_dsize)
        repeat = '|**repeat**|' + ''.join(
            str(scale_repeat(arch, size, test_repeat)) + '|'
            for size in test_dsize)
        lines = [header, layout, size, repeat]
        return lines

    def run(self):
        for case in self.case_impl:
            case.run()

    def mdlines(self):
        lines = []
        for arch in test_archs:
            lines += self.__class__.md_table_header(self.__class__.__name__,
                                                    arch, test_dsize)
            for case in self.case_impl:
                lines += case.save2markdown(arch)
            lines.append('')
        return lines
