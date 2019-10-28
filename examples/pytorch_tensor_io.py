import taichi as ti
import numpy as np
import torch

# ti.set_gdb_trigger(True)
ti.cfg.arch = ti.cuda

# n = 1024 * 1024
n = 32

y = ti.var(ti.f32)

# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

@ti.layout
def values():
  # actually useless in thie example
  ti.root.dense(ti.i, n).place(y)

@ti.kernel
def torch_kernel(t: ti.ext_arr(), o: ti.ext_arr()):
  for i in range(n):
    o[i] = t[i] * t[i]

@ti.kernel
def torch_kernel_2(t_grad: ti.ext_arr(), t: ti.ext_arr(), o_grad: ti.ext_arr()):
  for i in range(n):
    ti.print(o_grad[i])
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
X = torch.tensor(2 * np.ones((n, ), dtype=np.float32), device=torch.device('cuda:0'), requires_grad=True)
sqr(X).sum().backward()
print(X.grad.cpu())

