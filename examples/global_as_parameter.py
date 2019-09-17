import taichi_lang as ti

ti.runtime.print_preprocessed = True
x = ti.global_var(ti.f32)
y = ti.global_var(ti.f32)
z = ti.global_var(ti.f32)
loss = ti.global_var(ti.f32)

@ti.layout
def tensors():
  ti.root.dense(ti.i, 16).place(x, y, z)
  ti.root.place(loss)
  ti.root.lazy_grad()
  

def double(a, b):
  @ti.kernel
  def kernel():
    for i in range(16):
      b[i] = a[i] * 2 + 1
  kernel.materialize()
  return kernel
      
      
for i in range(16):
  x[i] = i
  
double(x, y)()
double(y, z)()

for i in range(16):
  print(z[i])
