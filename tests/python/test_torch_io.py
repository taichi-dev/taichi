import taichi as ti
import numpy as np

@ti.host_arch
def test_io():
  if not ti.has_pytorch():
    return
  import torch

  n = 32

  @ti.kernel
  def torch_kernel(t: ti.ext_arr(), o: ti.ext_arr()):
    for i in range(n):
      o[i] = t[i] * t[i]

  @ti.kernel
  def torch_kernel_2(t_grad: ti.ext_arr(), t: ti.ext_arr(), o_grad: ti.ext_arr()):
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

  #, device=torch.device('cuda:0')

  sqr = Sqr.apply
  X = torch.tensor(2 * np.ones((n, ), dtype=np.float32), requires_grad=True)
  sqr(X).sum().backward()
  ret = X.grad.cpu()
  for i in range(n):
    assert ret[i] == 4


@ti.host_arch
def test_io_2d():
  return
  if not ti.has_pytorch():
    return
  import torch
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
  X = torch.tensor(2 * np.ones((n, ), dtype=np.float32), requires_grad=True)
  print(sqr(X).sum())

# test_io_2d()

