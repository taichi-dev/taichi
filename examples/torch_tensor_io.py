import taichi as ti
import numpy as np
import torch

# ti.set_gdb_trigger(True)
ti.cfg.arch = ti.cuda

# n = 1024 * 1024
n = 32

y = ti.var(ti.i32)

# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

z = np.array((n,), dtype=np.float32)

@ti.layout
def values():
  ti.root.place(y)


@ti.kernel
def torch_kernel(t: np.ndarray, o: np.ndarray):
  for i in range(n):
    o[i] = t[i] * t[i]
    
@ti.kernel
def torch_kernel_2(t_grad: np.ndarray, t:np.ndarray, o_grad: np.ndarray):
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
    print(outp_grad.cpu())
    inp_grad = torch.zeros_like(outp_grad)
    inp, = ctx.saved_tensors
    torch_kernel_2(inp_grad, inp, outp_grad)
    return inp_grad

sqr = Sqr.apply
x = torch.tensor(1 * np.ones((n, ), dtype=np.float32), device=torch.device('cuda:0'), requires_grad=True)
sqr(x).sum().backward()
# print(sqr(x).sum())#.backward()
print(x.grad.cpu())


