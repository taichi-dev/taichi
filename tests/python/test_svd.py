import taichi as ti
from pytest import approx

def svd(A, iters=5):
  assert A.n == 3 and A.m == 3
  inputs = tuple([e.ptr for e in A.entries])
  rets = ti.core.sifakis_svd_f32(*inputs, iters)
  assert len(rets) == 21
  U_entries = rets[:9]
  V_entries = rets[9:18]
  sig_entries = rets[18:]
  U = ti.expr_init(ti.Matrix.zero(ti.f32, 3, 3))
  V = ti.expr_init(ti.Matrix.zero(ti.f32, 3, 3))
  sigma = ti.expr_init(ti.Matrix.zero(ti.f32, 3, 3))
  for i in range(3):
    for j in range(3):
      U(i, j).assign(U_entries[i * 3 + j])
      V(i, j).assign(V_entries[i * 3 + j])
    sigma(i, i).assign(sig_entries[i])
  return U, sigma, V

@ti.all_archs
def test_transpose():
  A = ti.Matrix(3, 3, dt=ti.f32, shape=())
  U = ti.Matrix(3, 3, dt=ti.f32, shape=())
  sigma = ti.Matrix(3, 3, dt=ti.f32, shape=())
  V = ti.Matrix(3, 3, dt=ti.f32, shape=())
  
  @ti.kernel
  def run():
    U[None], sigma[None], V[None] = svd(A[None])
    
  run()

