from microbenchmarks._items import BenchmarkItem, DataType, MathOps
from microbenchmarks._metric import MetricType
from microbenchmarks._plan import BenchmarkPlan
from microbenchmarks._utils import dtype_size, scaled_repeat_times

import taichi as ti


def unary_ops_throughput_default(arch, repeat, math_op, dtype, element_num,
                                 thread_for_loop, get_metric):
    local_data_num = 16  #enough data to fill the instruction pipeline
    global_vector = ti.Vector.field(local_data_num, dtype, element_num)

    @ti.kernel
    def op_throughput():
        for e in global_vector:
            local_vector = global_vector[e]
            #loop
            for j in range(thread_for_loop):
                for k in ti.static(range(local_data_num)):
                    local_vector[k] = math_op(local_vector[k])
            #epilogue
            global_vector[e] = local_vector

    @ti.kernel
    def fill_random():
        for e in global_vector:
            for i in ti.static(range(local_data_num)):
                global_vector[e][i] = ti.random()

    fill_random()
    return get_metric(repeat, op_throughput)


class ElementNum(BenchmarkItem):
    name = 'element_num'

    def __init__(self):
        self._items = {
            'element16384': 16384,
            #enough threads for filling CUDA cores
        }


class ForLoopCycle(BenchmarkItem):
    name = 'thread_for_loop'

    def __init__(self):
        self._items = {}
        for i in range(1, 7):
            cycles = 4 * pow(2, i)  # [8 16 32 64 128 256]
            self._items['threadloop' + str(cycles)] = cycles


class MathOpsThroughputPlan(BenchmarkPlan):
    def __init__(self, arch: str):
        super().__init__('math_ops_throughput', arch, basic_repeat_times=10)
        math_dtype = DataType()
        math_dtype.remove_integer()
        self.create_plan(MathOps(), math_dtype, ElementNum(), ForLoopCycle(),
                         MetricType())
        self.add_func(['element16384'], unary_ops_throughput_default)
