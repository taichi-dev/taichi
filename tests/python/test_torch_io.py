import numpy as np

import taichi as ti

if ti.has_pytorch():
    import torch


@ti.torch_test
def test_io_devices():
    n = 32
    x = ti.field(dtype=ti.i32, shape=n)

    @ti.kernel
    def load(y: ti.ext_arr()):
        for i in x:
            x[i] = y[i] + 10

    @ti.kernel
    def inc():
        for i in x:
            x[i] += i

    @ti.kernel
    def store(y: ti.ext_arr()):
        for i in x:
            y[i] = x[i] * 2

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda:0')
    for device in devices:
        y = torch.Tensor(np.ones(shape=n, dtype=np.int32)).to(device)

        load(y)
        inc()
        store(y)

        y = y.cpu().numpy()

        for i in range(n):
            assert y[i] == (11 + i) * 2


@ti.torch_test
def test_io():
    n = 32

    @ti.kernel
    def torch_kernel(t: ti.ext_arr(), o: ti.ext_arr()):
        for i in range(n):
            o[i] = t[i] * t[i]

    @ti.kernel
    def torch_kernel_2(t_grad: ti.ext_arr(), t: ti.ext_arr(),
                       o_grad: ti.ext_arr()):
        for i in range(n):
            t_grad[i] = 2 * t[i] * o_grad[i]

    class Sqr(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            outp = torch.zeros_like(inp)
            ctx.save_for_backward(inp)
            torch_kernel(inp, outp)
            return outp

        @staticmethod
        def backward(ctx, outp_grad):
            outp_grad = outp_grad.contiguous()
            inp_grad = torch.zeros_like(outp_grad)
            inp, = ctx.saved_tensors
            torch_kernel_2(inp_grad, inp, outp_grad)
            return inp_grad

    sqr = Sqr.apply
    X = torch.tensor(2 * np.ones((n, ), dtype=np.float32), requires_grad=True)
    sqr(X).sum().backward()
    ret = X.grad.cpu()
    for i in range(n):
        assert ret[i] == 4


@ti.torch_test
def test_io_2d():
    n = 32

    @ti.kernel
    def torch_kernel(t: ti.ext_arr(), o: ti.ext_arr()):
        for i in range(n):
            for j in range(n):
                o[i, j] = t[i, j] * t[i, j]

    class Sqr(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            outp = torch.zeros_like(inp)
            torch_kernel(inp, outp)
            return outp

    sqr = Sqr.apply
    X = torch.tensor(2 * np.ones((n, n), dtype=np.float32), requires_grad=True)
    val = sqr(X).sum()
    assert val == 2 * 2 * n * n


@ti.torch_test
def test_io_3d():
    n = 16

    @ti.kernel
    def torch_kernel(t: ti.ext_arr(), o: ti.ext_arr()):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    o[i, j, k] = t[i, j, k] * t[i, j, k]

    class Sqr(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            outp = torch.zeros_like(inp)
            torch_kernel(inp, outp)
            return outp

    sqr = Sqr.apply
    X = torch.tensor(2 * np.ones((n, n, n), dtype=np.float32),
                     requires_grad=True)
    val = sqr(X).sum()
    assert val == 2 * 2 * n * n * n


@ti.torch_test
def test_io_simple():
    n = 32

    x1 = ti.field(ti.f32, shape=(n, n))
    t1 = torch.tensor(2 * np.ones((n, n), dtype=np.float32))

    x2 = ti.Matrix.field(2, 3, ti.f32, shape=(n, n))
    t2 = torch.tensor(2 * np.ones((n, n, 2, 3), dtype=np.float32))

    x1.from_torch(t1)
    for i in range(n):
        for j in range(n):
            assert x1[i, j] == 2

    x2.from_torch(t2)
    for i in range(n):
        for j in range(n):
            for k in range(2):
                for l in range(3):
                    assert x2[i, j][k, l] == 2

    t3 = x2.to_torch()
    assert (t2 == t3).all()


@ti.torch_test
def test_io_zeros():
    mat = ti.Matrix.field(2, 6, dtype=ti.f32, shape=(), needs_grad=True)
    zeros = torch.zeros((2, 6))
    zeros[1, 2] = 3
    mat.from_torch(zeros + 1)

    assert mat[None][1, 2] == 4

    zeros = mat.to_torch()
    assert zeros[1, 2] == 4


@ti.torch_test
def test_fused_kernels():
    n = 12
    X = ti.Matrix.field(3, 2, ti.f32, shape=(n, n, n))
    s = ti.get_runtime().get_num_compiled_functions()
    t = X.to_torch()
    assert ti.get_runtime().get_num_compiled_functions() == s + 1
    X.from_torch(t)
    assert ti.get_runtime().get_num_compiled_functions() == s + 2


@ti.torch_test
def test_device():
    n = 12
    X = ti.Matrix.field(3, 2, ti.f32, shape=(n, n, n))
    assert X.to_torch(device='cpu').device == torch.device('cpu')

    if torch.cuda.is_available():
        assert X.to_torch(device='cuda:0').device == torch.device('cuda:0')


@ti.torch_test
def test_shape_matrix():
    n = 12
    x = ti.Matrix.field(3, 2, ti.f32, shape=(n, n))
    X = x.to_torch()
    for i in range(n):
        for j in range(n):
            for k in range(3):
                for l in range(2):
                    X[i, j, k, l] = i * 10 + j + k * 100 + l * 1000

    x.from_torch(X)
    X1 = x.to_torch()
    x.from_torch(X1)
    X1 = x.to_torch()

    assert (X == X1).all()


@ti.torch_test
def test_shape_vector():
    n = 12
    x = ti.Vector.field(3, ti.f32, shape=(n, n))
    X = x.to_torch()
    for i in range(n):
        for j in range(n):
            for k in range(3):
                X[i, j, k] = i * 10 + j + k * 100

    x.from_torch(X)
    X1 = x.to_torch()
    x.from_torch(X1)
    X1 = x.to_torch()

    assert (X == X1).all()


@ti.torch_test
def test_torch_zero():
    @ti.kernel
    def test_torch(arr: ti.ext_arr()):
        pass

    test_torch(torch.zeros((0), dtype=torch.int32))
    test_torch(torch.zeros((0, 5), dtype=torch.int32))
    test_torch(torch.zeros((5, 0, 5), dtype=torch.int32))
