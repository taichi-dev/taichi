import taichi_lang as ti

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
  # Make sure you materialize the kernels immediately (by default they are initialized on first invocation)
  kernel.materialize()
  kernel.grad.materialize() # If you need the gradients
  return kernel
  
@ti.kernel
def compute_loss():
  for i in range(16):
    ti.atomic_add(loss, z[i])
      
for i in range(16):
  x[i] = i
  
# Instantiate your kernels here with different global variables
double1 = double(x, y)
double2 = double(y, z)
with ti.Tape(loss):
  double1()
  double2()
  compute_loss()
  
ti.clear_all_gradients()

for i in range(16):
  print(z[i], x.grad[i])

with ti.Tape(loss):
  double1()
  double2()
  compute_loss()

for i in range(16):
  print(z[i], x.grad[i])
