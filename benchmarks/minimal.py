# Copyright (c) 2020 The Taichi Authors
# Use of this software is governed by the LICENSE file.

import taichi as ti


@ti.all_archs
def benchmark_fill_scalar():
    a = ti.var(dt=ti.f32, shape=())

    @ti.kernel
    def fill():
        a[None] = 1.0

    return ti.benchmark(fill, repeat=1000)
