import taichi as ti
import time

# originally by @KLozes

def benchmark_range():
  a = ti.var(dt=ti.f32)
  N = 512

  @ti.layout
  def place():
    ti.root.dense(ti.ij, [N,N]).dense(ti.ij, [8, 8]).place(a)

  @ti.kernel
  def fill():
    for i,j in a:
      a[i, j] = 2.0

  fill()

  ti.get_runtime().sync()
  t = time.time()
  for n in range(100):
    fill()
  elapsed = time.time() - t
  ti.get_runtime().sync()
  return elapsed / 100
  
def benchmark_struct():
  a = ti.var(dt=ti.f32)
  N = 512
  
  @ti.layout
  def place():
    ti.root.dense(ti.ij, [N,N]).dense(ti.ij, [8, 8]).place(a)
    
  @ti.kernel
  def fill():
    for j in range(N * 8):
      for i in range(N * 8):
        a[i, j] = 2.0
  
  ti.get_runtime().sync()
  t = time.time()
  for n in range(100):
    fill()
  ti.get_runtime().sync()
  elapsed = time.time() - t
  return elapsed / 100
