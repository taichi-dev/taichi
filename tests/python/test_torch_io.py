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

  #, device=torch.device('cuda:0')

  sqr = Sqr.apply
  X = torch.tensor(2 * np.ones((n,), dtype=np.float32), requires_grad=True)
  sqr(X).sum().backward()
  ret = X.grad.cpu()
  for i in range(n):
    assert ret[i] == 4


@ti.host_arch
def test_io_2d():
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
  X = torch.tensor(2 * np.ones((n, n), dtype=np.float32), requires_grad=True)
  val = sqr(X).sum()
  assert val == 2 * 2 * n * n


@ti.host_arch
def test_io_3d():
  if not ti.has_pytorch():
    return
  import torch
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
  X = torch.tensor(2 * np.ones((n, n, n), dtype=np.float32), requires_grad=True)
  val = sqr(X).sum()
  assert val == 2 * 2 * n * n * n


@ti.host_arch
def test_io_simple():
  if not ti.has_pytorch():
    return
  import torch
  n = 32

  x1 = ti.var(ti.f32, shape=(n, n))
  t1 = torch.tensor(2 * np.ones((n, n), dtype=np.float32))

  x2 = ti.Matrix(2, 3, ti.f32, shape=(n, n))
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


@ti.host_arch
def test_io_simple():
  if not ti.has_pytorch():
    return
  import torch
  mat = ti.Matrix(2, 6, dt=ti.f32, shape=(), needs_grad=True)
  zeros = torch.zeros((2, 6))
  zeros[1, 2] = 3
  mat.from_torch(zeros + 1)

  assert mat[None][1, 2] == 4

  zeros = mat.to_torch()
  assert zeros[1, 2] == 4
