from taichi.core import ti_core as _ti_core
from taichi.lang import impl

import taichi as ti


class StatisticalResult:
    """Statistical result of records.

    Profiling records with the same kernel name will be counted in a ``StatisticalResult`` instance via ``insert_record(time)``.
    Currently, only the kernel elapsed time is counted, other statistics related to the kernel will be added in the feature.
    """
    def __init__(self, name):
        self.name = name
        self.counter = 0
        self.min_time = 0.0
        self.max_time = 0.0
        self.total_time = 0.0

    def __lt__(self, other):
        """For sorted()"""
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


class KernelProfiler:
    """Kernel profiler of Taichi.

    Kernel profiler acquires kernel profiling records from backend, counts records in python scope,
    and outputs results by printing : `~taichi.profiler.kernelprofiler.KernelProfiler.print_info`.
    """
    def __init__(self):
        """Constructor of class KernelProfiler.

        ``_profiling_mode`` is a boolean value, turned ON/OFF by : `~taichi.profiler.kernelprofiler.KernelProfiler.set_kernel_profiler_mode`.
        ``_total_time_ms`` is a float value, get the value in seconds with : `~taichi.profiler.kernelprofiler.KernelProfiler.get_total_time`.
        ``_traced_records`` is a list of profiling records, acquired from backend by : `~taichi.profiler.kernelprofiler.KernelProfiler.update_records`.
        ``_statistical_results`` is a dict of statistical profiling results,  statistics via : `~taichi.profiler.kernelprofiler.KernelProfiler.count_results`.
        """
        self._profiling_mode = False
        self._total_time_ms = 0.0
        self._traced_records = []
        self._statistical_results = {}

    def set_kernel_profiler_mode(self, mode=False):
        if type(mode) is bool:
            self._profiling_mode = mode
        else:
            raise TypeError(f'Arg `mode` must be of type boolean. '
                            f'Type {type(mode)} '
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
        return self._total_time_ms / 1000  # ms to s

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
        """Acquires profiling records from a backend."""
        impl.get_runtime().sync()
        self.clear_fronted()
        self._traced_records = impl.get_runtime(
        ).prog.get_kernel_profiler_records()

    def count_results(self):
        """Counts the statistical results.

        Profiling records with the same kernel name will be counted in a instance of class:`~taichi.profiler.kernelprofiler.StatisticalResult`.
        Presenting kernel profiling results in a statistical perspactive.
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

    # print info mode
    COUNT = 'count'  # print the statistical results (min,max,avg time) of Taichi kernels.
    TRACE = 'trace'  # print the records of launched Taichi kernels with specific profiling metrics (time, memory load/store and core utilization etc.)

    def print_info(self, mode=COUNT):
        """Print the profiling results of Taichi kernels.

        To enable this profiler, set ``kernel_profiler=True`` in ``ti.init()``.
        The default print mode is ``'count'`` mode: print the statistical results (min,max,avg time) of Taichi kernels,
        another mode ``'trace'``: print the records of launched Taichi kernels with specific profiling metrics (time, memory load/store and core utilization etc.)

        Args:
            mode (str): the way to print profiling results
        """
        def patition_line(character, num):
            return character * num

        self.update_records()  # trace records
        self.count_results()  # statistical results

        #count mode (default) : print statistical results of all kernel
        if mode == self.COUNT:
            print(f"{patition_line('=',73)}"\
                  f"\n{_ti_core.arch_name(ti.cfg.arch).upper()} Profiler(count)\n"\
                  f"{patition_line('=',73)}")
            print(f"[      %     total   count |"\
                  f"      min       avg       max   ]"\
                  f" Kernel name\n")
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
            print(f"{patition_line('-',73)}")
            print(f"[100.00%] Total kernel execution time: "
                  f"{self._total_time_ms/1000:7.3f} s   "
                  f"number of records:  "
                  f"{len(self._statistical_results)}")
            print(f"{patition_line('=',73)}")

        #trace mode : print records of launched kernel
        if mode == self.TRACE:
            print(f"{patition_line('=',73)}\n"\
                  f"{_ti_core.arch_name(ti.cfg.arch).upper()} Profiler(trace)\n"\
                  f"{patition_line('=',73)}\n")
            print(f"[      % |     time    ] Kernel name")
            for record in self._traced_records:
                fraction = record.kernel_time / self._total_time_ms * 100.0
                print(f"["
                      f"{fraction:6.2f}% |"
                      f"{record.kernel_time:9.3f}  ms] "
                      f"{record.name}")
            print(f"{patition_line('-',73)}")
            print(f"[100.00%] Total kernel execution time: "
                  f"{self._total_time_ms/1000:7.3f} s   "
                  f"number of records:  {len(self._traced_records)}")
            print(f"{patition_line('=',73)}")


_ti_kernel_profiler = KernelProfiler()


def get_default_kernel_profiler():
    """We have only one ``KernelProfiler`` instance(i.e. ``_ti_kernel_profiler``) now.

    For ``KernelProfiler`` using ``CuptiToolkit``, GPU devices can only work in a certain configuration,
    profiling mode and metrics are configured by the host(CPU) via CUPTI APIs, and device(GPU) will use
    its counter registers to collect specific metrics.
    So if there are multiple instances of ``KernelProfiler``,the device will work in the last configuration mode,
    the configuration of other instance will also be changed as a result.
    For data retention purposes, support for multiple instances may be considered in the future.
    """
    return _ti_kernel_profiler
