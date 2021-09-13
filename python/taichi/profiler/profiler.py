from taichi.core import ti_core as _ti_core
from taichi.lang import impl
import taichi as ti

class StatisticalResult:
        def __init__(self, name):
            self.name     = name
            self.counter  = 0
            self.min_time = 0.0
            self.max_time = 0.0
            self.total_time = 0.0

        def __lt__(self, other):
            if(self.total_time<other.total_time):
                return True
            else:
                return False

        def insert_record(self,time):
            if self.counter == 0:
                self.min_time = time
                self.max_time = time
            self.counter += 1
            self.total_time += time
            self.min_time = min(self.min_time, time)
            self.max_time = max(self.max_time, time)

class Profiler:

    profiling_modes = [True, False]

    trace = 'trace'
    count = 'count'
    print_modes = [count, trace]

    profiling_mode_ = False
    total_time_ms_ = 0.0
    traced_records_ = []
    statistical_results_ = {}

    def __init__(self):
        _ti_core.trace(f'Profiler')

    def set_kernel_profiler_mode(self, kernel_profiler=None):
        if kernel_profiler is None:
            self.profiling_mode_ = False
        if kernel_profiler is False:
            self.profiling_mode_ = False
        elif kernel_profiler is True:
            self.profiling_mode_ = True
        else:
            _ti_core.warn(
                f'kernel_profiler mode error : {type(kernel_profiler)}')
    
    def get_kernel_profiler_mode(self):
        return self.profiling_mode_

    def get_total_time(self):
        self.update_records() # traced records
        self.count_results()  # statistical results : total_time_ms_ is counted here
        return self.total_time_ms_
    
    def query_info(self, name):
        self.update_records() # traced records
        self.count_results()  # statistical results
        # TODO : query self.StatisticalResult in python scope 
        return impl.get_runtime().prog.query_kernel_profile_info(name)

    def clear_info(self):
        impl.get_runtime().sync()#sync first
        impl.get_runtime().prog.clear_kernel_profile_info()#backend
        #frontend
        self.total_time_ms_ = 0.0
        self.traced_records_.clear()
        self.statistical_results_.clear()

    def update_records(self):
        #sync & clear
        impl.get_runtime().sync()
        self.total_time_ms_ = 0.0
        self.traced_records_.clear()
        self.statistical_results_.clear()
        #update
        records_size = impl.get_runtime().prog.kernel_profile_record_len()
        func_get_record = impl.get_runtime().prog.get_kernel_profile_record
        for i in range(records_size):
            self.traced_records_.append(func_get_record(i))
    
    def count_results(self):
        """count statistical results
        kernels with the same name will be counted in a StatisticalResult
        """
        for record in self.traced_records_:
            if self.statistical_results_.get(record.name) == None:
                self.statistical_results_[record.name] = StatisticalResult(record.name)
            self.statistical_results_[record.name].insert_record(record.kernel_time)
            self.total_time_ms_ += record.kernel_time
        self.statistical_results_ = {k: v 
            for k, v in sorted(
                self.statistical_results_.items(), 
                key=lambda item: item[1], 
                reverse = True)}
            
    def print_info(self,mode=count):
        self.update_records() # trace records
        self.count_results()  # statistical results 
        #count mode (default) : print statistical results of all kernel
        if mode == self.count:
            print("=========================================================================")
            print(_ti_core.arch_name(ti.cfg.arch).upper() + " Profiler(count)")
            print("=========================================================================")
            print("[      %     total   count |      min       avg       max   ] Kernel name")
            for key in self.statistical_results_:
                result = self.statistical_results_[key]
                fraction = result.total_time / self.total_time_ms_ * 100.0
                print("[{:6.2f}% {:7.3f} s {:6d}x |{:9.3f} {:9.3f} {:9.3f} ms] {}".format(
                    fraction,
                    result.total_time / 1000.0,
                    result.counter,
                    result.min_time,
                    result.total_time / result.counter, # avg_time
                    result.max_time,
                    result.name))
            print("------------------------------------------------------------------------")
            print("[100.00%] Total kernel execution time: {:7.3f} s   number of records:  {}".format(
                self.total_time_ms_, len(self.statistical_results_)))
            print("=========================================================================")

        #trace mode : print records of launched kernel 
        if mode == self.trace:
            print("====================================")
            print(_ti_core.arch_name(ti.cfg.arch).upper() + " Profiler(trace)")
            print("====================================")
            print("[      % |     time    ] Kernel name")
            for record in self.traced_records_:
                fraction = record.kernel_time / self.total_time_ms_ * 100.0
                print("[{:6.2f}% |{:9.3f}  ms] {}".format(
                  fraction,
                  record.kernel_time,
                  record.name))
            print("====================================")


_ti_profiler = Profiler()