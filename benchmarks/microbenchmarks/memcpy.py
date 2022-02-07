from microbenchmarks._items import BenchmarkItem, Container, DataSize, DataType
from microbenchmarks._metric import MetricType
from microbenchmarks._plan import BenchmarkPlan
from microbenchmarks._utils import dtype_size, scaled_repeat_times

import taichi as ti

def memcpy_default(arch, repeat, container, dtype, dsize, get_metric):
    @ti.kernel
    def memcpy_field(dst: ti.template(), src: ti.template()):
        for I in ti.grouped(dst):
            dst[I] = src[I]

    @ti.kernel
    def memcpy_array(dst: ti.any_arr(), src: ti.any_arr()):
        for I in ti.grouped(dst):
            dst[I] = src[I]

    repeat = scaled_repeat_times(arch, dsize, repeat)
    num_elements = dsize // dtype_size(dtype)

    x = container(dtype, num_elements)
    y = container(dtype, num_elements)

    func = memcpy_field if container == ti.field else memcpy_array

    return get_metric(repeat, func, y, x)


class MemcpyPlan(BenchmarkPlan):
    def __init__(self, arch: str):
        super().__init__('memcpy', arch, basic_repeat_times=10)
        self.create_plan(Container(), DataType(), DataSize(), MetricType())
        self.add_func(['field'], memcpy_default)
        self.add_func(['ndarray'], memcpy_default)
