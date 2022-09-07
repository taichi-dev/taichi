import sys

import numpy as np
import pytest
from taichi.lang.util import has_pytorch

import taichi as ti
from tests import test_utils

if has_pytorch():
    import torch


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=ti.cuda)
def test_torch_cuda_context():
    device = torch.device("cuda:0")
    x = torch.tensor([2.], requires_grad=True, device=device)
    assert torch._C._cuda_hasPrimaryContext(0)
    loss = x**2
    loss.backward()


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(exclude=ti.opengl)
def test_torch_ad():
    n = 32

    x = ti.field(ti.f32, shape=n, needs_grad=True)
    y = ti.field(ti.f32, shape=n, needs_grad=True)

    @ti.kernel
    def torch_kernel():
        for i in range(n):
            # Do whatever complex operations here
            y[n - i - 1] = x[i] * x[i]

    class Sqr(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            x.from_torch(inp)
            torch_kernel()
            outp = y.to_torch()
            return outp

        @staticmethod
        def backward(ctx, outp_grad):
            ti.clear_all_gradients()
            y.grad.from_torch(outp_grad)
            torch_kernel.grad()
            inp_grad = x.grad.to_torch()
            return inp_grad

    sqr = Sqr.apply
    for i in range(10):
        X = torch.tensor(2 * np.ones((n, ), dtype=np.float32),
                         requires_grad=True)
        sqr(X).sum().backward()
        ret = X.grad.cpu().numpy()
        for j in range(n):
            assert ret[j] == 4


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@pytest.mark.skipif(sys.platform == 'win32', reason='not working on Windows.')
@test_utils.test(exclude=ti.opengl)
def test_torch_ad_gpu():
    if not torch.cuda.is_available():
        return

    device = torch.device('cuda:0')
    n = 32

    x = ti.field(ti.f32, shape=n, needs_grad=True)
    y = ti.field(ti.f32, shape=n, needs_grad=True)

    @ti.kernel
    def torch_kernel():
        for i in range(n):
            # Do whatever complex operations here
            y[n - i - 1] = x[i] * x[i]

    class Sqr(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            x.from_torch(inp)
            torch_kernel()
            outp = y.to_torch(device=device)
            return outp

        @staticmethod
        def backward(ctx, outp_grad):
            ti.clear_all_gradients()
            y.grad.from_torch(outp_grad)
            torch_kernel.grad()
            inp_grad = x.grad.to_torch(device=device)
            return inp_grad

    sqr = Sqr.apply
    for i in range(10):
        X = torch.tensor(2 * np.ones((n, ), dtype=np.float32),
                         requires_grad=True,
                         device=device)
        sqr(X).sum().backward()
        ret = X.grad.cpu().numpy()
        for j in range(n):
            assert ret[j] == 4
