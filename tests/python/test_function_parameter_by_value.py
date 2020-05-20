# Copyright (c) 2019 The Taichi Authors
# Use of this software is governed by the LICENSE file.

import taichi as ti


@ti.all_archs
def test_pass_by_value():
    @ti.func
    def set_val(x, i):
        x = i

    ret = ti.var(ti.i32, shape=())

    @ti.kernel
    def task():
        set_val(ret[None], 112)

    task()
    assert ret[None] == 0
