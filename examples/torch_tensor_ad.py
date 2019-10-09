import taichi as ti
import numpy as np
import torch

# ti.set_gdb_trigger(True)
ti.cfg.arch = ti.cuda

# n = 1024 * 1024
n = 32

x = ti.var(ti.f32)
y = ti.var(ti.f32)

def from_torch(expr, torch_tensor):
  if not expr.from_torch_:
    import taichi as ti
    import numpy as np
    @ti.kernel
    def ker(torch_tensor: np.ndarray):
      for i in expr:
        expr[i] = torch_tensor[i]
      
    ker.materialize()
    expr.from_torch_ = lambda x: ker(x.contiguous())
  expr.from_torch_(torch_tensor)
  
def to_torch(expr, torch_tensor):
  if not expr.to_torch_:
    import taichi as ti
    import numpy as np
    @ti.kernel
    def ker(torch_tensor: np.ndarray):
      for i in expr:
         torch_tensor[i] = expr[i]
    
    ker.materialize()
    expr.to_torch_ = lambda x: ker(x.contiguous())
  expr.to_torch_(torch_tensor)

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
    from_torch(x, inp)
    torch_kernel()
    to_torch(y, outp)
    return outp
  
  @staticmethod
  def backward(ctx, outp_grad):
    ti.clear_all_gradients()
    inp_grad = torch.zeros_like(outp_grad)
    
    from_torch(y.grad, outp_grad)
    
    torch_kernel.grad()
    
    to_torch(x.grad, inp_grad)
    
    return inp_grad

sqr = Sqr.apply
for i in range(10):
  X = torch.tensor(2 * np.ones((n, ), dtype=np.float32), device=torch.device('cuda:0'), requires_grad=True)
  sqr(X).sum().backward()
  print(X.grad.cpu())

