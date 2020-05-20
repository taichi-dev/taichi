# Copyright (c) 2019 The Taichi Authors
# Use of this software is governed by the LICENSE file.

import taichi as ti


@ti.must_throw(ti.TaichiSyntaxError)
def test_nested_kernel_error():
    @ti.kernel
    def B():
        pass

    @ti.kernel
    def A():
        B()

    A()
