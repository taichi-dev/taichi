from time import perf_counter

from taichi._lib import core as ti_core

import taichi as ti


class End2EndTimer:
    def __init__(self):
        self._ts1 = 0
        self._ts2 = 0

    def tick(self):
        ti.sync()
        self._ts1 = perf_counter()
        return self._ts1

    def tock(self):
        ti.sync()
        self._ts2 = perf_counter()
        return self._ts2 - self._ts1


def size2tag(size_in_byte):
    size_subsection = [(0.0, 'B'), (1024.0, 'KB'), (1048576.0, 'MB'),
                       (1073741824.0, 'GB'),
                       (float('inf'), 'INF')]  #B KB MB GB
    for dsize, unit in reversed(size_subsection):
        if size_in_byte >= dsize:
            return str(int(size_in_byte / dsize)) + unit


def tags2name(tag_list):
    return '_'.join(tag_list)


def dtype_size(ti_dtype):
    dtype_size_dict = {ti.i32: 4, ti.i64: 8, ti.f32: 4, ti.f64: 8}
    if ti_dtype not in dtype_size_dict:
        raise RuntimeError('Unsupported ti.dtype: ' + str(type(ti_dtype)))
    else:
        return dtype_size_dict[ti_dtype]


def get_ti_arch(arch: str):
    arch_dict = {
        'cuda': ti.cuda,
        'vulkan': ti.vulkan,
        'opengl': ti.opengl,
        'metal': ti.metal,
        'x64': ti.x64,
        'cc': ti.cc
    }
    return arch_dict[arch]


def scaled_repeat_times(arch: str, datasize, repeat=1):
    if (arch == 'cuda') | (arch == 'vulkan') | (arch == 'opengl'):
        repeat *= 10
    if datasize <= 4 * 1024 * 1024:
        repeat *= 10
    return repeat


def fill_random(dst, dtype, container):
    @ti.kernel
    def fill_template(dst: ti.template()):
        for I in ti.grouped(dst):
            dst[I] = ti.random(dtype)

    @ti.kernel
    def fill_1d_array(dst: ti.types.ndarray()):
        for i in dst:
            dst[i] = ti.random(dtype)

    @ti.kernel
    def fill_2d_array(dst: ti.types.ndarray()):
        for i, j in dst:
            dst[i, j] = ti.random(dtype)

    if container == ti.ndarray:
        if len(dst.shape) == 1:
            fill_1d_array(dst)
        elif len(dst.shape) == 2:
            fill_2d_array(dst)
    else:
        fill_template(dst)
