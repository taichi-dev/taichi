from microbenchmarks._items import AtomicOps, Container, DataSize, DataType
from microbenchmarks._metric import MetricType
from microbenchmarks._plan import BenchmarkPlan
from microbenchmarks._utils import dtype_size, fill_random, scaled_repeat_times

import taichi as ti


def reduction_default(arch, repeat, atomic_op, container, dtype, dsize,
                      get_metric):
    repeat = scaled_repeat_times(arch, dsize, repeat)
    num_elements = dsize // dtype_size(dtype)

    x = container(dtype, shape=num_elements)
    y = container(dtype, shape=())
    y[None] = 0

    @ti.kernel
    def reduction_field(y: ti.template(), x: ti.template()):
        for i in x:
            atomic_op(y[None], x[i])

    @ti.kernel
    def reduction_array(y: ti.any_arr(), x: ti.any_arr()):
        for i in x:
            atomic_op(y[None], x[i])

    fill_random(x, dtype, container)
    func = reduction_field if container == ti.field else reduction_array
    return get_metric(repeat, func, y, x)


class AtomicOpsPlan(BenchmarkPlan):
    def __init__(self, arch: str):
        super().__init__('atomic_ops', arch, basic_repeat_times=10)
        atomic_ops = AtomicOps()
        atomic_ops.remove(
            ['atomic_sub', 'atomic_and', 'atomic_xor', 'atomic_max'])
        self.create_plan(atomic_ops, Container(), DataType(), DataSize(),
                         MetricType())
        self.add_func(['field'], reduction_default)
        self.add_func(['ndarray'], reduction_default)
