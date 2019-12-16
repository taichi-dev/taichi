import taichi as ti
import numpy as np


@ti.host_arch
def test_torch_ad():
  if not ti.has_pytorch():
    return
  import torch
  n = 32

  x = ti.var(ti.f32, shape=n, needs_grad=True)
  y = ti.var(ti.f32, shape=n, needs_grad=True)

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
    X = torch.tensor(2 * np.ones((n,), dtype=np.float32), requires_grad=True)
    sqr(X).sum().backward()
    ret = X.grad.cpu().numpy()
    for j in range(n):
      assert ret[j] == 4
