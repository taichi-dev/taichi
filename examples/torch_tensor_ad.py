import taichi as ti
import numpy as np
import torch

# ti.set_gdb_trigger(True)
ti.cfg.arch = ti.cuda

# n = 1024 * 1024
n = 32

x = ti.var(ti.f32)
y = ti.var(ti.f32)

# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

@ti.layout
def values():
  # actually useless in thie example
  ti.root.dense(ti.i, n).place(x, y)
  ti.root.lazy_grad()

@ti.kernel
def torch_kernel():
  for i in range(n):
    # Do whatever complex operations here
    # a little bit fancier
    y[n - i - 1] = x[i] * x[i]
    
    
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
    ti.clear_all_gradients()
    inp_grad = torch.zeros_like(outp_grad)
    
    ti.from_torch(y.grad, outp_grad)
    torch_kernel.grad()
    ti.to_torch(x.grad, inp_grad)
    
    return inp_grad

sqr = Sqr.apply
for i in range(10):
  X = torch.tensor(2 * np.ones((n, ), dtype=np.float32), device=torch.device('cuda:0'), requires_grad=True)
  sqr(X).sum().backward()
  print(X.grad.cpu())

