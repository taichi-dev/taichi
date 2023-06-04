import taichi as ti

import pytest
from taichi.lang.exception import TaichiRuntimeError

from tests import test_utils
from taichi.lang.util import has_pytorch

if has_pytorch():
    import torch

archs_support_ndarray_ad = [ti.cpu, ti.cuda]


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64)
def test_simple_demo():
    @test_utils.torch_op(output_shapes=[(1,)])
    @ti.kernel
    def test(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in x:
            a = 2.0
            for j in range(1):
                a += x[i] / 3
            y[0] += a

    device = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"
    input = torch.rand(4, dtype=torch.double, device=device, requires_grad=True)
    torch.autograd.gradcheck(test, input)


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64)
def test_ad_reduce():
    @test_utils.torch_op(output_shapes=[(1,)])
    @ti.kernel
    def test(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in x:
            y[0] += x[i] ** 2

    device = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"
    input = torch.rand(4, dtype=torch.double, device=device, requires_grad=True)
    torch.autograd.gradcheck(test, input)


@pytest.mark.parametrize(
    "tifunc",
    [
        lambda x: x,
        lambda x: ti.abs(-x),
        lambda x: -x,
        lambda x: x * x,
        lambda x: x**2,
        lambda x: x * x * x,
        lambda x: x * x * x * x,
        lambda x: 0.4 * x * x - 3,
        lambda x: (x - 3) * (x - 1),
        lambda x: (x - 3) * (x - 1) + x * x,
        lambda x: ti.tanh(x),
        lambda x: ti.sin(x),
        lambda x: ti.cos(x),
        lambda x: ti.acos(x),
        lambda x: ti.asin(x),
        lambda x: 1 / x,
        lambda x: (x + 1) / (x - 1),
        lambda x: (x + 1) * (x + 2) / ((x - 1) * (x + 3)),
        lambda x: ti.sqrt(x),
        lambda x: ti.rsqrt(x),
        lambda x: ti.exp(x),
        lambda x: ti.log(x),
        lambda x: ti.min(x, 0),
        lambda x: ti.min(x, 1),
        lambda x: ti.min(0, x),
        lambda x: ti.min(1, x),
        lambda x: ti.max(x, 0),
        lambda x: ti.max(x, 1),
        lambda x: ti.max(0, x),
        lambda x: ti.max(1, x),
        lambda x: x % 3,
        lambda x: ti.atan2(0.4, x),
        lambda y: ti.atan2(y, 0.4),
        lambda x: 0.4**x,
        lambda y: y**0.4,
    ],
)
@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64)
def test_poly(tifunc):
    s = (4,)

    @test_utils.torch_op(output_shapes=[s])
    @ti.kernel
    def test(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in x:
            y[i] = tifunc(x[i])

    device = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"
    input = torch.rand(s, dtype=torch.double, device=device, requires_grad=True)
    torch.autograd.gradcheck(test, input)


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64)
def test_ad_select():
    s = (4,)

    @test_utils.torch_op(output_shapes=[s])
    @ti.kernel
    def test(x: ti.types.ndarray(), y: ti.types.ndarray(), z: ti.types.ndarray()):
        for i in x:
            z[i] = ti.select(i % 2, x[i], y[i])

    device = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"
    x = torch.rand(s, dtype=torch.double, device=device, requires_grad=True)
    y = torch.rand(s, dtype=torch.double, device=device, requires_grad=True)
    torch.autograd.gradcheck(test, [x, y])


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64)
def test_ad_sum():
    N = 10

    @ti.kernel
    def compute_sum(a: ti.types.ndarray(), b: ti.types.ndarray(), p: ti.types.ndarray()):
        for i in range(N):
            ret = 1.0
            for j in range(b[i]):
                ret = ret + a[i]
            p[i] = ret

    a = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    b = ti.ndarray(ti.i32, shape=N)
    p = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    for i in range(N):
        a[i] = 3
        b[i] = i

    compute_sum(a, b, p)

    for i in range(N):
        assert p[i] == a[i] * b[i] + 1
        p.grad[i] = 1

    compute_sum.grad(a, b, p)

    for i in range(N):
        assert a.grad[i] == b[i]


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64)
def test_ad_sum_local_atomic():
    N = 10
    a = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    b = ti.ndarray(ti.i32, shape=N)
    p = ti.ndarray(ti.f32, shape=N, needs_grad=True)

    @ti.kernel
    def compute_sum(a: ti.types.ndarray(), b: ti.types.ndarray(), p: ti.types.ndarray()):
        for i in range(N):
            ret = 1.0
            for j in range(b[i]):
                ret += a[i]
            p[i] = ret

    for i in range(N):
        a[i] = 3
        b[i] = i

    compute_sum(a, b, p)

    for i in range(N):
        assert p[i] == 3 * b[i] + 1
        p.grad[i] = 1

    compute_sum.grad(a, b, p)

    for i in range(N):
        assert a.grad[i] == b[i]


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64, require=ti.extension.adstack)
def test_ad_power():
    N = 10
    a = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    b = ti.ndarray(ti.i32, shape=N)
    p = ti.ndarray(ti.f32, shape=N, needs_grad=True)

    @ti.kernel
    def power(a: ti.types.ndarray(), b: ti.types.ndarray(), p: ti.types.ndarray()):
        for i in range(N):
            ret = 1.0
            for j in range(b[i]):
                ret = ret * a[i]
            p[i] = ret

    for i in range(N):
        a[i] = 3
        b[i] = i

    power(a, b, p)

    for i in range(N):
        assert p[i] == 3 ** b[i]
        p.grad[i] = 1

    power.grad(a, b, p)

    for i in range(N):
        assert a.grad[i] == b[i] * 3 ** (b[i] - 1)


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64, require=ti.extension.adstack)
def test_ad_fibonacci():
    N = 15
    a = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    b = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    c = ti.ndarray(ti.i32, shape=N)
    f = ti.ndarray(ti.f32, shape=N, needs_grad=True)

    @ti.kernel
    def fib(a: ti.types.ndarray(), b: ti.types.ndarray(), c: ti.types.ndarray(), f: ti.types.ndarray()):
        for i in range(N):
            p = a[i]
            q = b[i]
            for j in range(c[i]):
                p, q = q, p + q
            f[i] = q

    b.fill(1)

    for i in range(N):
        c[i] = i

    fib(a, b, c, f)

    for i in range(N):
        f.grad[i] = 1

    fib.grad(a, b, c, f)

    for i in range(N):
        print(a.grad[i], b.grad[i])
        if i == 0:
            assert a.grad[i] == 0
        else:
            assert a.grad[i] == f[i - 1]
        assert b.grad[i] == f[i]


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f32, require=ti.extension.adstack)
def test_ad_fibonacci_index():
    N = 5
    M = 10
    a = ti.ndarray(ti.f32, shape=M, needs_grad=True)
    b = ti.ndarray(ti.f32, shape=M, needs_grad=True)
    f = ti.ndarray(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def fib(a: ti.types.ndarray(), b: ti.types.ndarray(), f: ti.types.ndarray()):
        for i in range(N):
            p = 0
            q = 1
            for j in range(5):
                p, q = q, p + q
                b[q] += a[q]

        for i in range(M):
            f[None] += b[i]

    f.grad[None] = 1
    a.fill(1)

    fib(a, b, f)
    fib.grad(a, b, f)

    for i in range(M):
        is_fib = int(i in [1, 2, 3, 5, 8])
        assert a.grad[i] == is_fib * N
        assert b[i] == is_fib * N


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64, require=ti.extension.adstack)
def test_integer_stack():
    N = 5
    a = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    b = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    c = ti.ndarray(ti.i32, shape=N)
    f = ti.ndarray(ti.f32, shape=N, needs_grad=True)

    @ti.kernel
    def int_stack(a: ti.types.ndarray(), b: ti.types.ndarray(), c: ti.types.ndarray(), f: ti.types.ndarray()):
        for i in range(N):
            weight = 1
            s = 0.0
            for j in range(c[i]):
                s += weight * a[i] + b[i]
                weight *= 10
            f[i] = s

    a.fill(1)
    b.fill(1)

    for i in range(N):
        c[i] = i

    int_stack(a, b, c, f)

    for i in range(N):
        print(f[i])
        f.grad[i] = 1

    int_stack.grad(a, b, c, f)

    t = 0
    for i in range(N):
        assert a.grad[i] == t
        assert b.grad[i] == i
        t = t * 10 + 1


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64, require=ti.extension.adstack)
def test_double_for_loops():
    N = 5
    a = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    b = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    c = ti.ndarray(ti.i32, shape=N)
    f = ti.ndarray(ti.f32, shape=N, needs_grad=True)

    @ti.kernel
    def double_for(a: ti.types.ndarray(), b: ti.types.ndarray(), c: ti.types.ndarray(), f: ti.types.ndarray()):
        for i in range(N):
            weight = 1.0
            for j in range(c[i]):
                weight *= a[i]
            s = 0.0
            for j in range(c[i] * 2):
                s += weight + b[i]
            f[i] = s

    a.fill(2)
    b.fill(1)

    for i in range(N):
        c[i] = i

    double_for(a, b, c, f)

    for i in range(N):
        assert f[i] == 2 * i * (1 + 2**i)
        f.grad[i] = 1

    double_for.grad(a, b, c, f)

    for i in range(N):
        assert a.grad[i] == 2 * i * i * 2 ** (i - 1)
        assert b.grad[i] == 2 * i


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64, require=ti.extension.adstack)
def test_double_for_loops_more_nests():
    N = 6
    a = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    b = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    c = ti.ndarray(ti.i32, shape=(N, N // 2))
    f = ti.ndarray(ti.f32, shape=(N, N // 2), needs_grad=True)

    @ti.kernel
    def double_for(a: ti.types.ndarray(), b: ti.types.ndarray(), c: ti.types.ndarray(), f: ti.types.ndarray()):
        for i in range(N):
            for k in range(N // 2):
                weight = 1.0
                for j in range(c[i, k]):
                    weight *= a[i]
                s = 0.0
                for j in range(c[i, k] * 2):
                    s += weight + b[i]
                f[i, k] = s

    a.fill(2)
    b.fill(1)

    for i in range(N):
        for k in range(N // 2):
            c[i, k] = i + k

    double_for(a, b, c, f)

    for i in range(N):
        for k in range(N // 2):
            assert f[i, k] == 2 * (i + k) * (1 + 2 ** (i + k))
            f.grad[i, k] = 1

    double_for.grad(a, b, c, f)

    for i in range(N):
        total_grad_a = 0
        total_grad_b = 0
        for k in range(N // 2):
            total_grad_a += 2 * (i + k) ** 2 * 2 ** (i + k - 1)
            total_grad_b += 2 * (i + k)
        assert a.grad[i] == total_grad_a
        assert b.grad[i] == total_grad_b


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64, require=ti.extension.adstack)
def test_complex_body():
    N = 5
    a = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    c = ti.ndarray(ti.i32, shape=N)
    f = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    g = ti.ndarray(ti.f32, shape=N, needs_grad=True)

    @ti.kernel
    def complex(a: ti.types.ndarray(), c: ti.types.ndarray(), f: ti.types.ndarray(), g: ti.types.ndarray()):
        for i in range(N):
            weight = 2.0
            tot = 0.0
            tot_weight = 0.0
            for j in range(c[i]):
                tot_weight += weight + 1
                tot += (weight + 1) * a[i]
                weight = weight + 1
                weight = weight * 4
                weight = ti.cast(weight, ti.f64)
                weight = ti.cast(weight, ti.f32)

            g[i] = tot_weight
            f[i] = tot

    a.fill(2)

    for i in range(N):
        c[i] = i
        f.grad[i] = 1

    complex(a, c, f, g)
    complex.grad(a, c, f, g)

    for i in range(N):
        print(a.grad.to_numpy())
        # assert a.grad[i] == g[i]


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64, require=ti.extension.adstack)
def test_mixed_inner_loops():
    x = ti.ndarray(dtype=ti.f32, shape=(1,), needs_grad=True)
    arr = ti.ndarray(dtype=ti.f32, shape=(5))
    loss = ti.ndarray(dtype=ti.f32, shape=(1,), needs_grad=True)

    @ti.kernel
    def mixed_inner_loops(x: ti.types.ndarray(), arr: ti.types.ndarray(), loss: ti.types.ndarray()):
        for i in arr:
            loss[0] += ti.sin(x[0])
            for j in range(2):
                loss[0] += ti.sin(x[0]) + 1.0

    loss.grad[0] = 1.0
    x[0] = 0.0
    mixed_inner_loops(x, arr, loss)
    mixed_inner_loops.grad(x, arr, loss)

    assert loss[0] == 10.0
    assert x.grad[0] == 15.0


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=ti.f64, require=ti.extension.adstack)
def test_mixed_inner_loops_tape():
    x = ti.ndarray(dtype=ti.f32, shape=(1,), needs_grad=True)
    arr = ti.ndarray(dtype=ti.f32, shape=(5))
    loss = ti.ndarray(dtype=ti.f32, shape=(1,), needs_grad=True)

    @ti.kernel
    def mixed_inner_loops_tape(x: ti.types.ndarray(), arr: ti.types.ndarray(), loss: ti.types.ndarray()):
        for i in arr:
            loss[0] += ti.sin(x[0])
            for j in range(2):
                loss[0] += ti.sin(x[0]) + 1.0

    x[0] = 0.0
    with ti.ad.Tape(loss=loss):
        mixed_inner_loops_tape(x, arr, loss)
    assert loss[0] == 10.0
    assert x.grad[0] == 15.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack, ad_stack_size=32)
def test_inner_loops_local_variable_fixed_stack_size_kernel_grad():
    x = ti.ndarray(dtype=float, shape=(1), needs_grad=True)
    arr = ti.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = ti.ndarray(dtype=float, shape=(1), needs_grad=True)

    @ti.kernel
    def test_inner_loops_local_variable(x: ti.types.ndarray(), arr: ti.types.ndarray(), loss: ti.types.ndarray()):
        for i in arr:
            for j in range(3):
                s = 0.0
                t = 0.0
                for k in range(3):
                    s += ti.sin(x[0]) + 1.0
                    t += ti.sin(x[0])
                loss[0] += s + t

    loss.grad[0] = 1.0
    x[0] = 0.0
    test_inner_loops_local_variable(x, arr, loss)
    test_inner_loops_local_variable.grad(x, arr, loss)

    assert loss[0] == 18.0
    assert x.grad[0] == 36.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack, ad_stack_size=0)
def test_inner_loops_local_variable_adaptive_stack_size_tape():
    x = ti.ndarray(dtype=float, shape=(1), needs_grad=True)
    arr = ti.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = ti.ndarray(dtype=float, shape=(1), needs_grad=True)

    @ti.kernel
    def test_inner_loops_local_variable(x: ti.types.ndarray(), arr: ti.types.ndarray(), loss: ti.types.ndarray()):
        for i in arr:
            for j in range(3):
                s = 0.0
                t = 0.0
                for k in range(3):
                    s += ti.sin(x[0]) + 1.0
                    t += ti.sin(x[0])
                loss[0] += s + t

    x[0] = 0.0
    with ti.ad.Tape(loss=loss):
        test_inner_loops_local_variable(x, arr, loss)

    assert loss[0] == 18.0
    assert x.grad[0] == 36.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack, ad_stack_size=0)
def test_more_inner_loops_local_variable_adaptive_stack_size_tape():
    x = ti.ndarray(dtype=float, shape=(1), needs_grad=True)
    arr = ti.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = ti.ndarray(dtype=float, shape=(1), needs_grad=True)

    @ti.kernel
    def test_more_inner_loops_local_variable(x: ti.types.ndarray(), arr: ti.types.ndarray(), loss: ti.types.ndarray()):
        for i in arr:
            for j in range(2):
                s = 0.0
                for k in range(3):
                    u = 0.0
                    s += ti.sin(x[0]) + 1.0
                    for l in range(2):
                        u += ti.sin(x[0])
                    loss[0] += u
                loss[0] += s

    x[0] = 0.0
    with ti.ad.Tape(loss=loss):
        test_more_inner_loops_local_variable(x, arr, loss)

    assert loss[0] == 12.0
    assert x.grad[0] == 36.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack, ad_stack_size=32)
def test_more_inner_loops_local_variable_fixed_stack_size_tape():
    x = ti.ndarray(dtype=float, shape=(1), needs_grad=True)
    arr = ti.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = ti.ndarray(dtype=float, shape=(1), needs_grad=True)

    @ti.kernel
    def test_more_inner_loops_local_variable(x: ti.types.ndarray(), arr: ti.types.ndarray(), loss: ti.types.ndarray()):
        for i in arr:
            for j in range(2):
                s = 0.0
                for k in range(3):
                    u = 0.0
                    s += ti.sin(x[0]) + 1.0
                    for l in range(2):
                        u += ti.sin(x[0])
                    loss[0] += u
                loss[0] += s

    x[0] = 0.0
    with ti.ad.Tape(loss=loss):
        test_more_inner_loops_local_variable(x, arr, loss)

    assert loss[0] == 12.0
    assert x.grad[0] == 36.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack, ad_stack_size=32)
def test_stacked_inner_loops_local_variable_fixed_stack_size_kernel_grad():
    x = ti.ndarray(dtype=float, shape=(), needs_grad=True)
    arr = ti.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = ti.ndarray(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_stacked_inner_loops_local_variable(
        x: ti.types.ndarray(), arr: ti.types.ndarray(), loss: ti.types.ndarray()
    ):
        for i in arr:
            loss[None] += ti.sin(x[None])
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                loss[None] += s
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                loss[None] += s

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_stacked_inner_loops_local_variable(x, arr, loss)
    test_stacked_inner_loops_local_variable.grad(x, arr, loss)

    assert loss[None] == 36.0
    assert x.grad[None] == 38.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack, ad_stack_size=32)
def test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable_fixed_stack_size_kernel_grad():
    x = ti.ndarray(dtype=float, shape=(), needs_grad=True)
    arr = ti.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = ti.ndarray(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable(
        x: ti.types.ndarray(), arr: ti.types.ndarray(), loss: ti.types.ndarray()
    ):
        for i in arr:
            loss[None] += ti.sin(x[None])
            for j in range(3):
                for k in range(3):
                    loss[None] += ti.sin(x[None]) + 1.0
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                loss[None] += s
            for j in range(3):
                for k in range(3):
                    loss[None] += ti.sin(x[None]) + 1.0

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable(x, arr, loss)
    test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable.grad(x, arr, loss)

    assert loss[None] == 54.0
    assert x.grad[None] == 56.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack, ad_stack_size=0)
def test_stacked_inner_loops_local_variable_adaptive_stack_size_kernel_grad():
    x = ti.ndarray(dtype=float, shape=(), needs_grad=True)
    arr = ti.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = ti.ndarray(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_stacked_inner_loops_local_variable(
        x: ti.types.ndarray(), arr: ti.types.ndarray(), loss: ti.types.ndarray()
    ):
        for i in arr:
            loss[None] += ti.sin(x[None])
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                loss[None] += s
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                loss[None] += s

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_stacked_inner_loops_local_variable(x, arr, loss)
    test_stacked_inner_loops_local_variable.grad(x, arr, loss)

    assert loss[None] == 36.0
    assert x.grad[None] == 38.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack, ad_stack_size=0)
def test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable_adaptive_stack_size_kernel_grad():
    x = ti.ndarray(dtype=float, shape=(), needs_grad=True)
    arr = ti.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = ti.ndarray(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable(
        x: ti.types.ndarray(), arr: ti.types.ndarray(), loss: ti.types.ndarray()
    ):
        for i in arr:
            loss[None] += ti.sin(x[None])
            for j in range(3):
                for k in range(3):
                    loss[None] += ti.sin(x[None]) + 1.0
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                loss[None] += s
            for j in range(3):
                for k in range(3):
                    loss[None] += ti.sin(x[None]) + 1.0

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable(x, arr, loss)
    test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable.grad(x, arr, loss)

    assert loss[None] == 54.0
    assert x.grad[None] == 56.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack, ad_stack_size=0)
def test_large_for_loops_adaptive_stack_size():
    x = ti.ndarray(dtype=float, shape=(), needs_grad=True)
    arr = ti.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = ti.ndarray(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_large_loop(x: ti.types.ndarray(), arr: ti.types.ndarray(), loss: ti.types.ndarray()):
        for i in range(5):
            for j in range(2000):
                for k in range(1000):
                    loss[None] += ti.sin(x[None]) + 1.0

    with ti.ad.Tape(loss=loss):
        test_large_loop(x, arr, loss)

    assert loss[None] == 1e7
    assert x.grad[None] == 1e7


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack, ad_stack_size=1)
def test_large_for_loops_fixed_stack_size():
    x = ti.ndarray(dtype=float, shape=(), needs_grad=True)
    arr = ti.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = ti.ndarray(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_large_loop(x: ti.types.ndarray(), arr: ti.types.ndarray(), loss: ti.types.ndarray()):
        for i in range(5):
            for j in range(2000):
                for k in range(1000):
                    loss[None] += ti.sin(x[None]) + 1.0

    with ti.ad.Tape(loss=loss):
        test_large_loop(x, arr, loss)

    assert loss[None] == 1e7
    assert x.grad[None] == 1e7


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_multiple_ib():
    x = ti.ndarray(float, (), needs_grad=True)
    y = ti.ndarray(float, (), needs_grad=True)

    @ti.kernel
    def compute_y(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]

    x[None] = 1.0
    with ti.ad.Tape(y):
        compute_y(x, y)

    assert y[None] == 12.0
    assert x.grad[None] == 12.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_multiple_ib_multiple_outermost():
    x = ti.ndarray(float, (), needs_grad=True)
    y = ti.ndarray(float, (), needs_grad=True)

    @ti.kernel
    def compute_y(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]

    x[None] = 1.0
    with ti.ad.Tape(y):
        compute_y(x, y)

    assert y[None] == 24.0
    assert x.grad[None] == 24.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_multiple_ib_multiple_outermost_mixed():
    x = ti.ndarray(float, (), needs_grad=True)
    y = ti.ndarray(float, (), needs_grad=True)

    @ti.kernel
    def compute_y(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]
                for ii in range(3):
                    y[None] += x[None]

    x[None] = 1.0
    with ti.ad.Tape(y):
        compute_y(x, y)

    assert y[None] == 42.0
    assert x.grad[None] == 42.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_multiple_ib_mixed():
    x = ti.ndarray(float, (), needs_grad=True)
    y = ti.ndarray(float, (), needs_grad=True)

    @ti.kernel
    def compute_y(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]
                for k in range(2):
                    y[None] += x[None]
            for i in range(3):
                y[None] += x[None]

    x[None] = 1.0
    with ti.ad.Tape(y):
        compute_y(x, y)

    assert y[None] == 30.0
    assert x.grad[None] == 30.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_multiple_ib_deeper():
    x = ti.ndarray(float, (), needs_grad=True)
    y = ti.ndarray(float, (), needs_grad=True)

    @ti.kernel
    def compute_y(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                for ii in range(2):
                    y[None] += x[None]
            for i in range(3):
                for ii in range(2):
                    for iii in range(2):
                        y[None] += x[None]

    x[None] = 1.0
    with ti.ad.Tape(y):
        compute_y(x, y)

    assert y[None] == 42.0
    assert x.grad[None] == 42.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_multiple_ib_deeper_non_scalar():
    N = 10
    x = ti.ndarray(float, shape=N, needs_grad=True)
    y = ti.ndarray(float, shape=N, needs_grad=True)

    @ti.kernel
    def compute_y(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for j in range(N):
            for i in range(j):
                y[j] += x[j]
            for i in range(3):
                for ii in range(j):
                    y[j] += x[j]
            for i in range(3):
                for ii in range(2):
                    for iii in range(j):
                        y[j] += x[j]

    x.fill(1.0)
    for i in range(N):
        y.grad[i] = 1.0
    compute_y(x, y)
    compute_y.grad(x, y)
    for i in range(N):
        assert y[i] == i * 10.0
        assert x.grad[i] == i * 10.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_multiple_ib_inner_mixed():
    x = ti.ndarray(float, (), needs_grad=True)
    y = ti.ndarray(float, (), needs_grad=True)

    @ti.kernel
    def compute_y(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                for ii in range(2):
                    y[None] += x[None]
                for iii in range(2):
                    y[None] += x[None]
                    for iiii in range(2):
                        y[None] += x[None]
            for i in range(3):
                for ii in range(2):
                    for iii in range(2):
                        y[None] += x[None]

    x[None] = 1.0
    with ti.ad.Tape(y):
        compute_y(x, y)

    assert y[None] == 78.0
    assert x.grad[None] == 78.0


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_ib_global_load():
    N = 10
    a = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    b = ti.ndarray(ti.i32, shape=N)
    p = ti.ndarray(ti.f32, shape=N, needs_grad=True)

    @ti.kernel
    def compute(a: ti.types.ndarray(), b: ti.types.ndarray(), p: ti.types.ndarray()):
        for i in range(N):
            val = a[i]
            for j in range(b[i]):
                p[i] += i
            p[i] = val * i

    for i in range(N):
        a[i] = i
        b[i] = 2

    compute(a, b, p)

    for i in range(N):
        assert p[i] == i * i
        p.grad[i] = 1

    compute.grad(a, b, p)
    for i in range(N):
        assert a.grad[i] == i


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_ad_if_simple():
    x = ti.ndarray(ti.f32, shape=(), needs_grad=True)
    y = ti.ndarray(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def func(x: ti.types.ndarray(), y: ti.types.ndarray()):
        if x[None] > 0.0:
            y[None] = x[None]

    x[None] = 1
    y.grad[None] = 1

    func(x, y)
    func.grad(x, y)

    assert x.grad[None] == 1


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_ad_if():
    x = ti.ndarray(ti.f32, shape=2, needs_grad=True)
    y = ti.ndarray(ti.f32, shape=2, needs_grad=True)

    @ti.kernel
    def func(i: ti.i32, x: ti.types.ndarray(), y: ti.types.ndarray()):
        if x[i] > 0:
            y[i] = x[i]
        else:
            y[i] = 2 * x[i]

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func(0, x, y)
    func.grad(0, x, y)
    func(1, x, y)
    func.grad(1, x, y)

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_ad_if_nested():
    n = 20
    x = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    y = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    z = ti.ndarray(ti.f32, shape=n, needs_grad=True)

    @ti.kernel
    def func(x: ti.types.ndarray(), y: ti.types.ndarray(), z: ti.types.ndarray()):
        for i in x:
            if x[i] < 2:
                if x[i] == 0:
                    y[i] = 0
                else:
                    y[i] = z[i] * 1
            else:
                if x[i] == 2:
                    y[i] = z[i] * 2
                else:
                    y[i] = z[i] * 3

    z.fill(1)

    for i in range(n):
        x[i] = i % 4

    func(x, y, z)
    for i in range(n):
        assert y[i] == i % 4
        y.grad[i] = 1
    func.grad(x, y, z)

    for i in range(n):
        assert z.grad[i] == i % 4


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_ad_if_mutable():
    x = ti.ndarray(ti.f32, shape=2, needs_grad=True)
    y = ti.ndarray(ti.f32, shape=2, needs_grad=True)

    @ti.kernel
    def func(i: ti.i32, x: ti.types.ndarray(), y: ti.types.ndarray()):
        t = x[i]
        if t > 0:
            y[i] = t
        else:
            y[i] = 2 * t

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func(0, x, y)
    func.grad(0, x, y)
    func(1, x, y)
    func.grad(1, x, y)

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_ad_if_parallel():
    x = ti.ndarray(ti.f32, shape=2, needs_grad=True)
    y = ti.ndarray(ti.f32, shape=2, needs_grad=True)

    @ti.kernel
    def func(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in range(2):
            t = x[i]
            if t > 0:
                y[i] = t
            else:
                y[i] = 2 * t

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func(x, y)
    func.grad(x, y)

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_ad_if_parallel_f64():
    x = ti.ndarray(ti.f64, shape=2, needs_grad=True)
    y = ti.ndarray(ti.f64, shape=2, needs_grad=True)

    @ti.kernel
    def func(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in range(2):
            t = x[i]
            if t > 0:
                y[i] = t
            else:
                y[i] = 2 * t

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func(x, y)
    func.grad(x, y)

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@test_utils.test(arch=archs_support_ndarray_ad, require=ti.extension.adstack)
def test_ad_if_parallel_complex():
    x = ti.ndarray(ti.f32, shape=2, needs_grad=True)
    y = ti.ndarray(ti.f32, shape=2, needs_grad=True)

    @ti.kernel
    def func(x: ti.types.ndarray(), y: ti.types.ndarray()):
        ti.loop_config(parallelize=1)
        for i in range(2):
            t = 0.0
            if x[i] > 0:
                t = 1 / x[i]
            y[i] = t

    x[0] = 0
    x[1] = 2
    y.grad[0] = 1
    y.grad[1] = 1

    func(x, y)
    func.grad(x, y)

    assert x.grad[0] == 0
    assert x.grad[1] == -0.25


@test_utils.test(arch=archs_support_ndarray_ad)
def test_ad_ndarray_i32():
    with pytest.raises(TaichiRuntimeError, match=r"i32 is not supported for ndarray"):
        ti.ndarray(ti.i32, shape=3, needs_grad=True)


@test_utils.test(arch=archs_support_ndarray_ad)
def test_ad_sum_vector():
    N = 10

    @ti.kernel
    def compute_sum(a: ti.types.ndarray(), p: ti.types.ndarray()):
        for i in p:
            p[i] = a[i] * 2

    a = ti.ndarray(ti.math.vec2, shape=N, needs_grad=True)
    p = ti.ndarray(ti.math.vec2, shape=N, needs_grad=True)
    for i in range(N):
        a[i] = [3, 3]

    compute_sum(a, p)

    for i in range(N):
        assert p[i] == [a[i] * 2, a[i] * 3]
        p.grad[i] = [1, 1]

    compute_sum.grad(a, p)

    for i in range(N):
        for j in range(2):
            assert a.grad[i][j] == 2


@test_utils.test(arch=archs_support_ndarray_ad)
def test_ad_multiple_tapes():
    N = 10

    @ti.kernel
    def compute_sum(a: ti.types.ndarray(), p: ti.types.ndarray()):
        for i in a:
            p[None] += a[i][0] * 2 + a[i][1] * 3

    a = ti.ndarray(ti.math.vec2, shape=N, needs_grad=True)
    p = ti.ndarray(ti.f32, shape=(), needs_grad=True)

    init_val = 3
    for i in range(N):
        a[i] = [init_val, init_val]

    with ti.ad.Tape(loss=p):
        compute_sum(a, p)

    assert p[None] == N * (2 + 3) * init_val

    for i in range(N):
        assert a.grad[i][0] == 2
        assert a.grad[i][1] == 3

    # second run
    a.grad.fill(0)
    with ti.ad.Tape(loss=p):
        compute_sum(a, p)

    assert p[None] == N * (2 + 3) * init_val

    for i in range(N):
        assert a.grad[i][0] == 2
        assert a.grad[i][1] == 3


@test_utils.test(arch=archs_support_ndarray_ad)
def test_ad_set_loss_grad():
    x = ti.ndarray(dtype=ti.f32, shape=(), needs_grad=True)
    loss = ti.ndarray(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def eval_x(x: ti.types.ndarray()):
        x[None] = 1.0

    @ti.kernel
    def compute_1(x: ti.types.ndarray(), loss: ti.types.ndarray()):
        loss[None] = x[None]

    @ti.kernel
    def compute_2(x: ti.types.ndarray(), loss: ti.types.ndarray()):
        loss[None] = 2 * x[None]

    @ti.kernel
    def compute_3(x: ti.types.ndarray(), loss: ti.types.ndarray()):
        loss[None] = 4 * x[None]

    eval_x(x)
    with ti.ad.Tape(loss=loss):
        compute_1(x, loss)
        compute_2(x, loss)
        compute_3(x, loss)

    assert loss[None] == 4
    assert x.grad[None] == 4


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=archs_support_ndarray_ad)
def test_ad_mixed_with_torch():
    @test_utils.torch_op(output_shapes=[(1,)])
    @ti.kernel
    def compute_sum(a: ti.types.ndarray(), p: ti.types.ndarray()):
        for i in a:
            p[0] += a[i] * 2

    N = 4
    a = torch.ones(N, requires_grad=True)
    b = a * 2
    c = compute_sum(b)
    c[0].sum().backward()

    for i in range(4):
        assert a.grad[i] == 4


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=archs_support_ndarray_ad)
def test_ad_tape_throw():
    N = 4

    @ti.kernel
    def compute_sum(a: ti.types.ndarray(), p: ti.types.ndarray()):
        for i in a:
            p[0] += a[i] * 2

    a = torch.ones(N, requires_grad=True)
    p = torch.ones(2, requires_grad=True)

    with pytest.raises(RuntimeError, match=r"he loss of `Tape` must be a tensor only contains one element"):
        with ti.ad.Tape(loss=p):
            compute_sum(a, p)

    b = ti.ndarray(ti.f32, shape=(N), needs_grad=True)
    q = ti.ndarray(ti.f32, shape=(2), needs_grad=True)

    with pytest.raises(RuntimeError, match=r"The loss of `Tape` must be an ndarray with only one element"):
        with ti.ad.Tape(loss=q):
            compute_sum(b, q)

    m = torch.ones(1, requires_grad=False)
    with pytest.raises(
        RuntimeError,
        match=r"Gradients of loss are not allocated, please set requires_grad=True for all tensors that are required by autodiff.",
    ):
        with ti.ad.Tape(loss=m):
            compute_sum(a, m)

    n = ti.ndarray(ti.f32, shape=(1), needs_grad=False)
    with pytest.raises(
        RuntimeError,
        match=r"Gradients of loss are not allocated, please set needs_grad=True for all ndarrays that are required by autodiff.",
    ):
        with ti.ad.Tape(loss=n):
            compute_sum(b, n)


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=archs_support_ndarray_ad)
def test_tape_torch_tensor_grad_none():
    N = 3

    @ti.kernel
    def test(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in x:
            a = 2.0
            for j in range(N):
                a += x[i] / 3
            y[0] += a

    device = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"

    a = torch.zeros((N,), device=device, requires_grad=True)
    loss = torch.zeros((1,), device=device, requires_grad=True)

    with ti.ad.Tape(loss=loss):
        test(a, loss)

    for i in range(N):
        assert a.grad[i] == 1.0


@test_utils.test(arch=archs_support_ndarray_ad)
def test_grad_tensor_in_kernel():
    N = 10

    a = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    b = ti.ndarray(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def test(x: ti.types.ndarray(), b: ti.types.ndarray()):
        for i in x:
            b[None] += x.grad[i]

    a.grad.fill(2.0)
    test(a, b)
    assert b[None] == N * 2.0

    with pytest.raises(RuntimeError, match=r"Cannot automatically differentiate through a grad tensor"):
        test.grad(a, b)


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=archs_support_ndarray_ad)
def test_tensor_shape():
    N = 3

    @ti.kernel
    def test(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in range(N):
            a = 2.0
            for j in range(N):
                a += x[i] / x.shape[0]
            y[0] += a

    device = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"

    a = torch.zeros((N,), device=device, requires_grad=True)
    loss = torch.zeros((1,), device=device, requires_grad=True)

    with ti.ad.Tape(loss=loss):
        test(a, loss)

    for i in range(N):
        assert a.grad[i] == 1.0


@test_utils.test(arch=archs_support_ndarray_ad)
def test_ndarray_needs_grad_false():
    N = 3

    @ti.kernel
    def test(x: ti.types.ndarray(needs_grad=False), y: ti.types.ndarray()):
        for i in range(N):
            a = 2.0
            for j in range(N):
                a += x[i] / x.shape[0]
            y[0] += a

    x = ti.ndarray(ti.f32, shape=N, needs_grad=True)
    y = ti.ndarray(ti.f32, shape=1, needs_grad=True)

    test(x, y)

    y.grad.fill(1.0)
    test.grad(x, y)
    for i in range(N):
        assert x.grad[i] == 0.0


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=archs_support_ndarray_ad)
def test_torch_needs_grad_false():
    N = 3

    @ti.kernel
    def test(x: ti.types.ndarray(needs_grad=False), y: ti.types.ndarray()):
        for i in range(N):
            a = 2.0
            for j in range(N):
                a += x[i] / x.shape[0]
            y[0] += a

    x = torch.rand((N,), dtype=torch.float, requires_grad=True)
    y = torch.rand((1,), dtype=torch.float, requires_grad=True)

    test(x, y)

    y.grad.fill_(1.0)
    test.grad(x, y)
    for i in range(N):
        assert x.grad[i] == 0.0


@test_utils.test(arch=archs_support_ndarray_ad)
def test_ad_vector_arg():
    N = 10

    @ti.kernel
    def compute_sum(a: ti.types.ndarray(), p: ti.types.ndarray(), z: ti.math.vec2):
        for i in p:
            p[i] = a[i] * z[0]

    a = ti.ndarray(ti.math.vec2, shape=N, needs_grad=True)
    p = ti.ndarray(ti.math.vec2, shape=N, needs_grad=True)
    z = ti.math.vec2([2.0, 3.0])
    for i in range(N):
        a[i] = [3, 3]

    compute_sum(a, p, z)

    for i in range(N):
        assert p[i] == [a[i] * 2, a[i] * 2]
        p.grad[i] = [1, 1]

    compute_sum.grad(a, p, z)

    for i in range(N):
        for j in range(2):
            assert a.grad[i][j] == 2


@test_utils.test(arch=archs_support_ndarray_ad)
def test_hash_encoder_simple():
    @ti.kernel
    def hash_encoder_kernel(
        table: ti.types.ndarray(),
        output_embedding: ti.types.ndarray(),
    ):
        ti.loop_config(block_dim=256)
        for level in range(1):
            local_features = ti.Vector([0.0])

            tmp0 = local_features[0] + table[0]
            local_features[0] = tmp0

            if level < 0:
                # To keep this IfStmt
                print(1111)

            tmp1 = local_features[0] + table[0]
            local_features[0] = tmp1

            output_embedding[0, 0] = local_features[0]

    table = ti.ndarray(shape=(1), dtype=ti.f32, needs_grad=True)
    output_embedding = ti.ndarray(shape=(1, 1), dtype=ti.f32, needs_grad=True)

    table[0] = 0.2924
    table.grad[0] = 0.0
    output_embedding[0, 0] = 0.7515
    output_embedding.grad[0, 0] = 2.8942e-06

    hash_encoder_kernel.grad(table, output_embedding)

    assert table.grad[0] > 5.788399e-06 and table.grad[0] < 5.7884e-06
