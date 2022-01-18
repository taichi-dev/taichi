from microbenchmarks._items import DataSize, DataType
from microbenchmarks._plan import BenchmarkPlan
from microbenchmarks._result import ResultType
from microbenchmarks._utils import dtype_size, scaled_repeat_times

import taichi as ti


def fill_default(arch, repeat, dtype, dsize, get_result):
    @ti.kernel
    def fill_field(dst: ti.template()):
        for I in ti.grouped(dst):
            dst[I] = ti.cast(0.7, dtype)

    repeat = scaled_repeat_times(arch, dsize, repeat)
    num_elements = dsize // dtype_size(dtype)
    x = ti.field(dtype, num_elements)
    return get_result(repeat, fill_field, x)


class FillPlan(BenchmarkPlan):
    def __init__(self, arch: str):
        super().__init__('fill', arch, basic_repeat_times=10)
        self.create_plan(DataType(), DataSize(), ResultType())
        self.set_func(fill_default)
