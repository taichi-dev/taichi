from taichi.core import ti_core as _ti_core
from taichi.lang import impl

import taichi as ti


class StatisticalResult:
    def __init__(self, name):
        self.name = name
        self.counter = 0
        self.min_time = 0.0
        self.max_time = 0.0
        self.total_time = 0.0

    def __lt__(self, other):
        if (self.total_time < other.total_time):
            return True
        else:
            return False

    def insert_record(self, time):
        if self.counter == 0:
            self.min_time = time
            self.max_time = time
        self.counter += 1
        self.total_time += time
        self.min_time = min(self.min_time, time)
        self.max_time = max(self.max_time, time)


class Profiler:
    '''_profiling_mode now is boolean value
    `trace` and `count` are two modes for `print_info(mode)`
    '''

    trace = 'trace'
    count = 'count'

    def __init__(self):
        self._profiling_mode = False
        self._total_time_ms = 0.0
        self._traced_records = []
        self._statistical_results = {}

    def set_kernel_profiler_mode(self, kernel_profiler=False):
        if type(kernel_profiler) is bool:
            self._profiling_mode = kernel_profiler
        else:
            raise TypeError(f'Arg kernel_profiler must be of type boolean. '
                            f'Type {type(kernel_profiler)} '
                            f'is not supported')

    def get_kernel_profiler_mode(self):
        return self._profiling_mode

    def clear_fronted(self):
        self._total_time_ms = 0.0
        self._traced_records.clear()
        self._statistical_results.clear()

    def get_total_time(self):
        self.update_records()  # traced records
        self.count_results()  # _total_time_ms is counted here
        return self._total_time_ms

    def query_info(self, name):
        self.update_records()  # traced records
        self.count_results()  # statistical results
        # TODO : query self.StatisticalResult in python scope
        return impl.get_runtime().prog.query_kernel_profile_info(name)

    def clear_info(self):
        impl.get_runtime().sync()  #sync first
        #clear backend & frontend
        impl.get_runtime().prog.clear_kernel_profile_info()
        self.clear_fronted()

    def update_records(self):
        impl.get_runtime().sync()
        self.clear_fronted()
        self._traced_records = impl.get_runtime(
        ).prog.get_kernel_profiler_records()

    def count_results(self):
        """count statistical results
        kernels with the same name will be counted in a StatisticalResult
        """
        for record in self._traced_records:
            if self._statistical_results.get(record.name) == None:
                self._statistical_results[record.name] = StatisticalResult(
                    record.name)
            self._statistical_results[record.name].insert_record(
                record.kernel_time)
            self._total_time_ms += record.kernel_time
        self._statistical_results = {
            k: v
            for k, v in sorted(self._statistical_results.items(),
                               key=lambda item: item[1],
                               reverse=True)
        }

    def print_info(self, mode=count):
        '''`count` mode : print the statistical results (min,max,avg time) of Taichi kernels on devices.
        `trace` mode : print records of launched Taichi kernels
        Default print mode is `count` mode
        '''
        self.update_records()  # trace records
        self.count_results()  # statistical results
        #count mode (default) : print statistical results of all kernel
        if mode == self.count:
            print(f"====================================="
                  f"====================================")
            print(f"{_ti_core.arch_name(ti.cfg.arch).upper()} Profiler(count)")
            print(f"====================================="
                  f"====================================")
            print(f"[      %     total   count |"
                  f"      min       avg       max   ]"
                  f" Kernel name")
            for key in self._statistical_results:
                result = self._statistical_results[key]
                fraction = result.total_time / self._total_time_ms * 100.0
                print(f"["
                      f"{fraction:6.2f}% "
                      f"{result.total_time / 1000.0:7.3f} s "
                      f"{result.counter:6d}x |"
                      f"{result.min_time:9.3f} "
                      f"{result.total_time / result.counter:9.3f} "
                      f"{result.max_time:9.3f} ms] "
                      f"{result.name}")
            print(f"-------------------------------------"
                  f"------------------------------------")
            print(f"[100.00%] Total kernel execution time: "
                  f"{self._total_time_ms:7.3f} s   "
                  f"number of records:  "
                  f"{len(self._statistical_results)}")
            print(f"====================================="
                  f"====================================")

        #trace mode : print records of launched kernel
        if mode == self.trace:
            print(f"====================================="
                  f"====================================")
            print(f"{_ti_core.arch_name(ti.cfg.arch).upper()} Profiler(trace)")
            print(f"====================================="
                  f"====================================")
            print(f"[      % |     time    ] Kernel name")
            for record in self._traced_records:
                fraction = record.kernel_time / self._total_time_ms * 100.0
                print(f"["
                      f"{fraction:6.2f}% |"
                      f"{record.kernel_time:9.3f}  ms] "
                      f"{record.name}")
            print(f"-------------------------------------"
                  f"------------------------------------")
            print(f"[100.00%] Total kernel execution time: "
                  f"{self._total_time_ms:7.3f} s   "
                  f"number of records:  {len(self._traced_records)}")
            print(f"====================================="
                  f"====================================")


_ti_profiler = Profiler()


def get_default_profiler():
    return _ti_profiler
