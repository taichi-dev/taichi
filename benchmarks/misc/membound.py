import os
import time

from membound_cases import memory_bound_cases_list
from utils import (arch_name, datatime_with_format, dtype2str, dump2json,
                   geometric_mean, get_commit_hash, md_table_header,
                   scaled_repeat_times, size2str)

import taichi as ti


class MemoryBound:
    suite_name = 'memorybound'
    supported_archs = [ti.x64, ti.cuda]
    test_cases = memory_bound_cases_list
    test_dtype_list = [ti.i32, ti.i64, ti.f32, ti.f64]
    test_dsize_list = [
        (4**i) * 1024  # kibibytes(KiB) = 1024
        for i in range(1, 10)  # [4KB,16KB...256MB]
    ]
    basic_repeat_times = 10
    evaluator = [geometric_mean]

    def __init__(self, arch):
        self._arch = arch
        self._cases_impl = []
        for case in self.test_cases:
            for dtype in self.test_dtype_list:
                impl = CaseImpl(case, arch, dtype, self.test_dsize_list,
                                self.evaluator)
                self._cases_impl.append(impl)

    def run(self):
        for case in self._cases_impl:
            case.run()

    def save_as_json(self, arch_dir='./'):
        #folder of suite
        suite_path = os.path.join(arch_dir, self.suite_name)
        os.makedirs(suite_path)
        #json files
        self._save_suite_info_as_json(suite_path)
        self._save_cases_info_as_json(suite_path)

    def save_as_markdown(self, arch_dir='./'):
        current_time = datatime_with_format()
        commit_hash = get_commit_hash()  #[:8]
        file_name = f'{self.suite_name}.md'
        file_path = os.path.join(arch_dir, file_name)
        with open(file_path, 'w') as f:
            lines = [
                f'commit_hash: {commit_hash}\n', f'datatime: {current_time}\n'
            ]
            lines += self._get_markdown_lines()
            for line in lines:
                print(line, file=f)

    def _save_suite_info_as_json(self, suite_path='./'):
        info_dict = {
            'cases': [func.__name__ for func in self.test_cases],
            'dtype': [dtype2str(dtype) for dtype in self.test_dtype_list],
            'dsize': [size for size in self.test_dsize_list],
            'repeat': [
                scaled_repeat_times(self._arch, size, self.basic_repeat_times)
                for size in self.test_dsize_list
            ],
            'evaluator': [func.__name__ for func in self.evaluator]
        }
        info_path = os.path.join(suite_path, '_info.json')
        with open(info_path, 'w') as f:
            print(dump2json(info_dict), file=f)

    def _save_cases_info_as_json(self, suite_path='./'):
        for case in self.test_cases:  #for case [fill,saxpy,reduction]
            results_dict = {}
            for impl in self._cases_impl:  #find [ti.i32, ti.i64, ti.f32, ti.f64]
                if impl._name != case.__name__:
                    continue
                result_name = dtype2str(impl._test_dtype)
                results_dict[result_name] = impl.get_results_dict()
            case_path = os.path.join(suite_path, (case.__name__ + '.json'))
            with open(case_path, 'w') as f:
                case_str = dump2json(results_dict)
                print(case_str, file=f)

    def _get_markdown_lines(self):
        lines = []
        lines += md_table_header(self.suite_name, self._arch,
                                 self.test_dsize_list, self.basic_repeat_times,
                                 self.evaluator)

        result_header = '|kernel elapsed time(ms)' + ''.join(
            '|' for i in range(
                len(self.test_dsize_list) + len(MemoryBound.evaluator)))
        lines += [result_header]
        for case in self._cases_impl:
            lines += case.get_markdown_lines()
        lines.append('')
        return lines


class CaseImpl:

    def __init__(self, func, arch, test_dtype, test_dsize_list, evaluator):
        self._func = func
        self._name = func.__name__
        self._arch = arch
        self._test_dtype = test_dtype
        self._test_dsize_list = test_dsize_list
        self._min_time_in_us = []  #test results
        self._evaluator = evaluator

    def run(self):
        ti.init(kernel_profiler=True, arch=self._arch)
        print("TestCase[%s.%s.%s]" % (self._func.__name__, arch_name(
            self._arch), dtype2str(self._test_dtype)))
        for test_dsize in self._test_dsize_list:
            print("test_dsize = %s" % (size2str(test_dsize)))
            self._min_time_in_us.append(
                self._func(self._arch, self._test_dtype, test_dsize,
                           MemoryBound.basic_repeat_times))
            time.sleep(0.2)
        ti.reset()

    def get_markdown_lines(self):
        string = '|' + self._name + '.' + dtype2str(self._test_dtype) + '|'
        string += ''.join(
            str(round(time, 4)) + '|' for time in self._min_time_in_us)
        string += ''.join(
            str(round(item(self._min_time_in_us), 4)) + '|'
            for item in self._evaluator)
        return [string]

    def get_results_dict(self):
        results_dict = {}
        for i in range(len(self._test_dsize_list)):
            dsize = self._test_dsize_list[i]
            repeat = scaled_repeat_times(self._arch, dsize,
                                         MemoryBound.basic_repeat_times)
            elapsed_time = self._min_time_in_us[i]
            item_name = size2str(dsize).replace('.0', '')
            item_dict = {
                'dsize_byte': dsize,
                'repeat': repeat,
                'elapsed_time_ms': elapsed_time
            }
            results_dict[item_name] = item_dict
        return results_dict
