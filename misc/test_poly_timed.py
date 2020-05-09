import taichi as ti
from taichi import approx

from autograd import grad


# Note: test happens at v = 0.2
def grad_test(tifunc, npfunc=None, default_fp=ti.f32):
    if npfunc is None:
        npfunc = tifunc

    @ti.all_archs_with(default_fp=default_fp)
    def impl():
        print(f'arch={ti.cfg.arch} default_fp={ti.cfg.default_fp}')
        x = ti.var(default_fp)
        y = ti.var(default_fp)

        @ti.layout
        def place():
            ti.root.dense(ti.i, 1).place(x, x.grad, y, y.grad)

        @ti.kernel
        def func():
            for i in x:
                y[i] = tifunc(x[i])

        v = 0.234

        y.grad[0] = 1
        x[0] = v
        func()
        func.grad()

        assert y[0] == approx(npfunc(v))
        assert x.grad[0] == approx(grad(npfunc)(v))

    impl()


def test_poly():
    import time
    t = time.time()
    grad_test(lambda x: x)
    grad_test(lambda x: -x)
    grad_test(lambda x: x * x)
    grad_test(lambda x: x**2)
    grad_test(lambda x: x * x * x)
    grad_test(lambda x: x * x * x * x)
    grad_test(lambda x: 0.4 * x * x - 3)
    grad_test(lambda x: (x - 3) * (x - 1))
    grad_test(lambda x: (x - 3) * (x - 1) + x * x)
    ti.core.print_profile_info()
    print('total_time', time.time() - t)


test_poly()
