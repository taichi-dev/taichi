import taichi as ti
import time

# originally by @KLozes

def benchmark_flat_struct():
  N = 4096
  a = ti.var(dt=ti.f32, shape=(N, N))
  
  @ti.kernel
  def fill():
    for i, j in a:
      a[i, j] = 2.0
  
  return ti.benchmark(fill)

def benchmark_flat_range():
  N = 4096
  a = ti.var(dt=ti.f32, shape=(N, N))
  
  @ti.kernel
  def fill():
    for j in range(N):
      for i in range(N):
        a[i, j] = 2.0
  
  return ti.benchmark(fill)

def benchmark_nested_struct():
  a = ti.var(dt=ti.f32)
  N = 512

  @ti.layout
  def place():
    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

  @ti.kernel
  def fill():
    for i, j in a:
      a[i, j] = 2.0

  fill()

  return ti.benchmark(fill)

def benchmark_nested_range_blocked():
  a = ti.var(dt=ti.f32)
  N = 512
  
  @ti.layout
  def place():
    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)
  
  @ti.kernel
  def fill():
    for X in range(N * N):
      for Y in range(64):
        a[X // N * 8 + Y // 8, X % N * 8 + Y % 8] = 2.0
  
  fill()
  
  return ti.benchmark(fill)
  
def benchmark_nested_range():
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
  
  return ti.benchmark(fill)
