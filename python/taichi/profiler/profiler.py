from taichi.core import ti_core as _ti_core
from taichi.lang import impl

class Profiler:

    profiling_modes = [
        True, False
    ]

    trace = 'trace'
    count = 'count'
    print_modes = [
        count, trace
    ]

    profiling_mode_ = False
    traced_records_ = []
    statistical_results_ = []

    def __init__(self):
        _ti_core.info(f'Profiler')

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

    def query_info(self, name):
        return impl.get_runtime().prog.query_kernel_profile_info(name)

    def print_info(self):
        impl.get_runtime().prog.print_kernel_profile_info()

    def clear_info(self):
        #sync first
        impl.get_runtime().sync()
        #backend
        impl.get_runtime().prog.clear_kernel_profile_info()
        #frontend
        self.traced_records_.clear()
        self.statistical_results_.clear()

    def record_len(self):
        return impl.get_runtime().prog.kernel_profile_record_len()

    def get_record(self,index):
        return impl.get_runtime().prog.get_kernel_profile_record(index)

    def update_records(self):
        #sync & clear
        impl.get_runtime().sync()
        self.traced_records_.clear()
        self.statistical_results_.clear()
        #update
        rlen = self.record_len()
        for i in range(rlen):
            self.traced_records_.append(self.get_record(i))
    
    def count_records(self):
        print('count')

    def print_records(self,mode=count):
        self.update_records()
        #defaut mode : count traced record >>> statistical results
        if mode == self.trace:
            for r in self.traced_records_:
                print(r.name + " : " + str(r.kernel_time) + " ms ")


_ti_profiler = Profiler()