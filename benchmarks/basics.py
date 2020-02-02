import taichi as ti

def benchmark_memset():
  a = ti.var(dt=ti.i32, shape=1024 ** 2)
  
  @ti.kernel
  def memset():
    for i in a:
      a[i] = 1.0
  
  return ti.benchmark(memset)

