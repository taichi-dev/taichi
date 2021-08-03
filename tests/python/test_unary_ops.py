import numpy as np

import taichi as ti


def _test_op(dt, taichi_op, np_op):
    print('arch={} default_fp={}'.format(ti.cfg.arch, ti.cfg.default_fp))
    n = 4
    val = ti.field(dt, shape=n)

    def f(i):
        return i * 0.1 + 0.4

    @ti.kernel
    def fill():
        for i in range(n):
            val[i] = taichi_op(f(ti.cast(i, dt)))

    fill()

    # check that it is double precision
    for i in range(n):
        if dt == ti.f64:
            assert abs(np_op(float(f(i))) - val[i]) < 1e-15
        else:
            assert abs(np_op(float(f(i))) -
                       val[i]) < 1e-6 if ti.cfg.arch != ti.opengl else 1e-5


def test_f64_trig():
    op_pairs = [
        (ti.sin, np.sin),
        (ti.cos, np.cos),
        (ti.asin, np.arcsin),
        (ti.acos, np.arccos),
        (ti.tan, np.tan),
        (ti.tanh, np.tanh),
        (ti.exp, np.exp),
        (ti.log, np.log),
    ]
    for dt in [ti.f32, ti.f64]:
        for taichi_op, np_op in op_pairs:

            @ti.all_archs_with(default_fp=dt)
            def wrapped():
                _test_op(dt, taichi_op, np_op)

            wrapped()
