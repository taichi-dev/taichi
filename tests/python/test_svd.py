import taichi as ti
import numpy as np
from pytest import approx

def svd(A, dt, iters=5):
  assert A.n == 3 and A.m == 3
  inputs = tuple([e.ptr for e in A.entries])
  assert dt in [ti.f32, ti.f64]
  if dt == ti.f32:
    rets = ti.core.sifakis_svd_f32(*inputs, iters)
  else:
    rets = ti.core.sifakis_svd_f64(*inputs, iters)
  assert len(rets) == 21
  U_entries = rets[:9]
  V_entries = rets[9:18]
  sig_entries = rets[18:]
  U = ti.expr_init(ti.Matrix.zero(dt, 3, 3))
  V = ti.expr_init(ti.Matrix.zero(dt, 3, 3))
  sigma = ti.expr_init(ti.Matrix.zero(dt, 3, 3))
  for i in range(3):
    for j in range(3):
      U(i, j).assign(U_entries[i * 3 + j])
      V(i, j).assign(V_entries[i * 3 + j])
    sigma(i, i).assign(sig_entries[i])
  return U, sigma, V

def mat_equal(A, B):
  return np.max(np.abs(A - B)) < 1e-5

@ti.all_archs
def _test_svd(dt):
  ti.get_runtime().set_default_fp(dt)
  
  A = ti.Matrix(3, 3, dt=dt, shape=())
  A_reconstructed = ti.Matrix(3, 3, dt=dt, shape=())
  U = ti.Matrix(3, 3, dt=dt, shape=())
  UtU = ti.Matrix(3, 3, dt=dt, shape=())
  sigma = ti.Matrix(3, 3, dt=dt, shape=())
  V = ti.Matrix(3, 3, dt=dt, shape=())
  VtV = ti.Matrix(3, 3, dt=dt, shape=())
  
  @ti.kernel
  def run():
    for i in range(1):
      U[None], sigma[None], V[None] = svd(A[None], dt)
      UtU[None] = ti.transposed(U[None]) @ U[None]
      VtV[None] = ti.transposed(V[None]) @ V[None]
      A_reconstructed[None] = U[None] @ sigma[None] @ ti.transposed(V[None])
    
  A[None] = [[1, 1, 3], [9, -3, 2], [-3, 4, 2]]
  
  run()
  
  assert mat_equal(UtU.to_numpy(), np.eye(3))
  assert mat_equal(VtV.to_numpy(), np.eye(3))
  assert mat_equal(A_reconstructed.to_numpy(), A.to_numpy())
  for i in range(3):
    for j in range(3):
      if i != j:
        assert sigma[None][i, j] == approx(0)
        
def test_svd():
  _test_svd(ti.f32)
  _test_svd(ti.f64)


@ti.all_archs
def test_transpose_no_loop():
  # TODO: fix this
  return
  A = ti.Matrix(3, 3, dt=ti.f32, shape=())
  U = ti.Matrix(3, 3, dt=ti.f32, shape=())
  sigma = ti.Matrix(3, 3, dt=ti.f32, shape=())
  V = ti.Matrix(3, 3, dt=ti.f32, shape=())
  
  @ti.kernel
  def run():
    U[None], sigma[None], V[None] = svd(A[None])
  
  run()
