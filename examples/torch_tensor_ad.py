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
  ti.root.dense(ti.i, n).place(x)
  ti.root.dense(ti.i, n).place(y)
  ti.root.lazy_grad()

@ti.kernel
def torch_kernel():
  for i in range(n):
    y[i] = x[i] * x[i]
    
    
def copy_from(taichi_tensor):
  @ti.kernel
  def ker(torch_tensor: np.ndarray):
    for i in taichi_tensor:
      taichi_tensor[i] = torch_tensor[i]
      
  ker.materialize()
  return lambda x: ker(x.contiguous())

def copy_to(taichi_tensor):
  @ti.kernel
  def ker(torch_tensor: np.ndarray):
    for i in taichi_tensor:
      torch_tensor[i] = taichi_tensor[i]
  
  ker.materialize()
  return lambda x: ker(x.contiguous())
  
x_copy_from = copy_from(x)
y_copy_to = copy_to(y)

y_grad_copy_from = copy_from(y.grad)
x_grad_copy_to = copy_to(x.grad)

class Sqr(torch.autograd.Function):
  @staticmethod
  def forward(ctx, inp):
    outp = torch.zeros_like(inp)
    x_copy_from(inp)
    torch_kernel()
    y_copy_to(outp)
    return outp
  
  @staticmethod
  def backward(ctx, outp_grad):
    ti.clear_all_gradients()
    inp_grad = torch.zeros_like(outp_grad)
    
    y_grad_copy_from(outp_grad)
    torch_kernel.grad()
    x_grad_copy_to(inp_grad)
    
    return inp_grad

sqr = Sqr.apply
X = torch.tensor(2 * np.ones((n, ), dtype=np.float32), device=torch.device('cuda:0'), requires_grad=True)
sqr(X).sum().backward()
print(X.grad.cpu())

