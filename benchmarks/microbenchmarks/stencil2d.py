from microbenchmarks._items import BenchmarkItem, Container, DataType
from microbenchmarks._metric import MetricType
from microbenchmarks._plan import BenchmarkPlan
from microbenchmarks._utils import (dtype_size, fill_random,
                                    scaled_repeat_times, size2tag)

import taichi as ti

stencil_common = [(0, 0), (0, -1), (0, 1), (1, 0)]


def stencil_2d_default(arch, repeat, scatter, bls, container, dtype, dsize_2d,
                       get_metric):

    dsize = dsize_2d[0] * dsize_2d[1]
    repeat = scaled_repeat_times(arch, dsize, repeat)
    num_elements_2d = (dsize_2d[0] // dtype_size(dtype), dsize_2d[1] // 2)

    y = container(dtype, shape=num_elements_2d)
    x = container(dtype, shape=num_elements_2d)

    @ti.kernel
    def stencil_2d_field(y: ti.template(), x: ti.template()):
        for I in ti.grouped(x):
            if ti.static(scatter):
                for offset in ti.static(stencil_common):
                    y[I + ti.Vector(offset)] += x[I]
            else:  # gather
                s = ti.cast(0.0, dtype)
                for offset in ti.static(stencil_common):
                    s = s + x[I + ti.Vector(offset)]
                y[I] = s

    @ti.kernel
    def stencil_2d_array(y: ti.any_arr(), x: ti.any_arr()):
        for I in ti.grouped(x):
            if ti.static(scatter):
                for offset in ti.static(stencil_common):
                    y[I + ti.Vector(offset)] += x[I]
            else:  # gather
                s = ti.cast(0.0, dtype)
                for offset in ti.static(stencil_common):
                    s = s + x[I + ti.Vector(offset)]
                y[I] = s

    fill_random(x, dtype, container)
    func = stencil_2d_field if container == ti.field else stencil_2d_array
    return get_metric(repeat, func, y, x)


def stencil_2d_sparse_bls(arch, repeat, scatter, bls, container, dtype,
                          dsize_2d, get_metric):

    dsize = dsize_2d[0] * dsize_2d[1]
    if dsize <= 4096 or dsize > 67108864:  # 16KB <= dsize <= 64 MB: Sparse-specific parameters
        return None
    repeat = scaled_repeat_times(
        arch, dsize, 1)  # basic_repeat_time = 1: Sparse-specific parameters
    block_elements_2d = (dsize_2d[0] // dtype_size(dtype) // 8,
                         dsize_2d[1] // 2 // 8)

    block = ti.root.pointer(ti.ij, block_elements_2d)
    y = ti.field(dtype)
    x = ti.field(dtype)
    block.dense(ti.ij, 8).place(y)
    block.dense(ti.ij, 8).place(x)

    @ti.kernel
    def active_all():
        for i, j in ti.ndrange(block_elements_2d[0], block_elements_2d[0]):
            ti.activate(block, [i, j])

    active_all()

    @ti.kernel
    def stencil_2d(y: ti.template(), x: ti.template()):
        #reference: tests/python/bls_test_template.py
        if ti.static(bls and not scatter):
            ti.block_local(x)
        if ti.static(bls and scatter):
            ti.block_local(y)
        ti.block_dim(64)  # 8*8=64

        for I in ti.grouped(x):
            if ti.static(scatter):
                for offset in ti.static(stencil_common):
                    y[I + ti.Vector(offset)] += x[I]
            else:  # gather
                s = ti.cast(0.0, dtype)
                for offset in ti.static(stencil_common):
                    s = s + x[I + ti.Vector(offset)]
                y[I] = s

    fill_random(x, dtype, container)
    return get_metric(repeat, stencil_2d, y, x)


class Scatter(BenchmarkItem):
    name = 'scatter'

    def __init__(self):
        self._items = {'scatter': True, 'gether': False}


class BloclLocalStorage(BenchmarkItem):
    name = 'bls'

    def __init__(self):
        self._items = {'bls_on': True, 'bls_off': False}


class DataSize2D(BenchmarkItem):
    name = 'dsize_2d'

    def __init__(self):
        self._items = {}
        for i in range(2, 10, 2):  # [16KB,256KB,4MB,64MB]
            size_bytes_2d = 32 * (2**i), 32 * (2**i)
            size_bytes = size_bytes_2d[0] * size_bytes_2d[1]
            self._items[size2tag(size_bytes)] = size_bytes_2d


class Stencil2DPlan(BenchmarkPlan):
    def __init__(self, arch: str):
        super().__init__('stencil_2d', arch, basic_repeat_times=10)
        container = Container()
        container.update({'sparse': None})  # None: implement by feature
        self.create_plan(Scatter(), BloclLocalStorage(), container, DataType(),
                         DataSize2D(), MetricType())
        # no use for field & ndarray
        self.remove_cases_with_tags(['field', 'bls1'])
        self.remove_cases_with_tags(['ndarray', 'bls1'])
        self.add_func(['field'], stencil_2d_default)
        self.add_func(['ndarray'], stencil_2d_default)
        self.add_func(['sparse'], stencil_2d_sparse_bls)
