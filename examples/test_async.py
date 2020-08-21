import taichi as ti
ti.init(arch=ti.gpu, async_mode=True, kernel_profiler=True, print_ir=True) # Try to run on GPU
n_grid = 10000000
grid_m = ti.field(dtype=float, shape=n_grid)

@ti.kernel
def substep():
  for i in grid_m:
    grid_m[i] = 0

for i in range(10000):
  substep()
  
ti.sync()

ti.print_profile_info()
ti.kernel_profiler_print()
