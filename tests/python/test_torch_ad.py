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
      # Do whatever complex operations here a little bit fancier
      y[n - i - 1] = x[i] * x[i]

  # https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

  class Sqr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
      outp = torch.zeros_like(inp)
      ti.from_torch(x, inp)
      torch_kernel()
      ti.to_torch(y, outp)
      return outp

    @staticmethod
    def backward(ctx, outp_grad):
      inp_grad = torch.zeros_like(outp_grad)

      ti.clear_all_gradients()
      ti.from_torch(y.grad, outp_grad)
      torch_kernel.grad()
      ti.to_torch(x.grad, inp_grad)

      return inp_grad

  sqr = Sqr.apply
  for i in range(10):
    X = torch.tensor(2 * np.ones((n, ), dtype=np.float32), requires_grad=True)
    sqr(X).sum().backward()
    ret = X.grad.cpu().numpy()
    for j in range(n):
      assert ret[j] == 4

