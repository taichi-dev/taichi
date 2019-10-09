import taichi.lang as ti

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

