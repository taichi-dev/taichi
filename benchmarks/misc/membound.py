import os
import time

from membound_cases import memory_bound_cases_list
from utils import (arch_name, dtype2str, dump2json, geometric_mean, kibibyte,
                   md_table_header, scaled_repeat_times, size2str)

import taichi as ti


class MemoryBound:
    suite_name = 'memorybound'
    supported_archs = [ti.cpu]
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

    def get_suite_info(self):
        info_dict = {
            'cases': [func.__name__ for func in self.test_cases],
            'dtype': [dtype2str[type] for type in self.test_dtype_list],
            'dsize': [size for size in self.test_dsize_list],
            'repeat': [
                scaled_repeat_times(self.arch, size, self.basic_repeat_times)
                for size in self.test_dsize_list
            ],
            'evaluator': [func.__name__ for func in self.evaluator]
        }
        return dump2json(info_dict)

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

    def save_to_json(self, suite_path='./'):
        '''save suite benchmark result to case.json.'''
        for case in self.test_cases:  #for case [fill,saxpy,reduction]
            results_dict = {}
            case_name = case.__name__
            case_path = os.path.join(suite_path, (case_name + '.json'))
            for impl in self.cases_impl:  #find [ti.i32, ti.i64, ti.f32, ti.f64]
                if impl.name is not case_name:
                    continue
                type_str = dtype2str[impl.test_dtype]
                result_name = self.suite_name + '.' + impl.name + '.' + arch_name(
                    self.arch) + '.' + type_str
                results_dict[result_name] = impl.get_results_dict()
            with open(case_path, 'w') as f:
                case_str = dump2json(results_dict)
                print(case_str, file=f)


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

    def get_results_dict(self):
        results_dict = {}
        for i in range(len(self.test_dsize_list)):
            dsize = self.test_dsize_list[i]
            repeat = scaled_repeat_times(self.arch, dsize,
                                         MemoryBound.basic_repeat_times)
            elapsed_time = self.min_time_in_us[i]
            item_name = size2str(dsize).replace('.0', '')
            item_dict = {
                'dsize_byte': dsize,
                'repeat': repeat,
                'elapsed_time_ms': elapsed_time
            }
            results_dict[item_name] = item_dict
        return results_dict
