import taichi as ti
import numpy as np
import torch

# ti.set_gdb_trigger(True)
ti.cfg.arch = ti.cuda

n = 1024 * 1024

y = ti.var(ti.i32)

# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

z = np.array((n,), dtype=np.float32)

@ti.layout
def values():
  # ti.root.dense(ti.i, n).place(y)
  ti.root.place(y)


@ti.kernel
def torch_kernel(t: np.ndarray, o: np.ndarray):
  for i in range(n):
    o[i] = t[i] * t[i]
    
@ti.kernel
def torch_kernel_2(t: np.ndarray, o: np.ndarray):
  for i in range(n):
    t[i] = 2 * o[i]
  
torch_kernel(z, z)
# torch_kernel_2(z, z)


class Sqr(torch.autograd.Function):
  @staticmethod
  def forward(ctx, inp):
    outp = torch.zeros_like(inp)
    print("here")
    torch_kernel(inp, outp)
    print("here1")
    return outp
  
  @staticmethod
  def backward(ctx, outp_grad):
    inp_grad = torch.zeros_like(outp_grad)
    print("here2")
    torch_kernel_2(inp_grad, outp_grad)
    print("here3")
    return inp_grad

sqr = Sqr.apply
x = torch.tensor(np.ones((n, ), dtype=np.float32), device=torch.device('cuda:0'), requires_grad=True)
sqr(x).sum().backward()
print(x.grad.cpu())


