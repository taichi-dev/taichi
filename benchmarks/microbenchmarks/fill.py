from microbenchmarks._items import Container, DataSize, DataType
from microbenchmarks._metric import MetricType
from microbenchmarks._plan import BenchmarkPlan
from microbenchmarks._utils import dtype_size, scaled_repeat_times

import taichi as ti


def fill_default(arch, repeat, container, dtype, dsize, get_metric):
    @ti.kernel
    def fill_field(dst: ti.template()):
        for I in ti.grouped(dst):
            dst[I] = ti.cast(0.7, dtype)

    @ti.kernel
    def fill_array(dst: ti.any_arr()):
        for i in dst:
            dst[i] = ti.cast(0.7, dtype)

    repeat = scaled_repeat_times(arch, dsize, repeat)
    num_elements = dsize // dtype_size(dtype)
    x = container(dtype, num_elements)
    func = fill_field if container == ti.field else fill_array
    return get_metric(repeat, func, x)


def fill_sparse(arch, repeat, container, dtype, dsize, get_metric):
    repeat = scaled_repeat_times(arch, dsize, repeat=1)
    # basic_repeat_time = 1: sparse-specific parameter
    num_elements = dsize // dtype_size(dtype) // 8

    block = ti.root.pointer(ti.i, num_elements)
    x = ti.field(dtype)
    block.dense(ti.i, 8).place(x)

    @ti.kernel
    def active_all():
        for i in ti.ndrange(num_elements):
            ti.activate(block, [i])

    active_all()

    @ti.kernel
    def fill_const(dst: ti.template()):
        for i in x:
            dst[i] = ti.cast(0.7, dtype)

    return get_metric(repeat, fill_const, x)


# use container_tag to get customized implementation
func_lut = {
    'field': fill_default,
    'ndarray': fill_default,
    'sparse': fill_sparse
}


class FillPlan(BenchmarkPlan):
    def __init__(self, arch: str):
        super().__init__('fill', arch, basic_repeat_times=10)
        fill_container = Container()
        # fill_container.update({'sparse': None})  # None: implement by feature
        self.create_plan(fill_container, DataType(), DataSize(), MetricType())
        self.set_func(func_lut)
