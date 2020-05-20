# Copyright (c) 2019 The Taichi Authors
# Use of this software is governed by the LICENSE file.

import taichi as ti


@ti.host_arch_only
def test_while():
    assert ti.core.test_threading()
