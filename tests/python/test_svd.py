import taichi as ti
import numpy as np
from pytest import approx

@ti.all_archs
def test_precision():
  ti.cfg.fast_math = False
  u = ti.var(ti.f64, shape=())
  v = ti.var(ti.f64, shape=())
  w = ti.var(ti.f64, shape=())
  
  @ti.kernel
  def forward():
    v[None] = ti.sqrt(ti.cast(u[None] + 3.25, ti.f64))
    w[None] = ti.cast(u[None] + 7, ti.f64) / ti.cast(u[None] + 3, ti.f64)
  
  forward()
  assert v[None] ** 2 == approx(3.25, abs=1e-12)
  assert w[None] * 3 == approx(7, abs=1e-12)

def mat_equal(A, B, tol=1e-6):
  return np.max(np.abs(A - B)) < tol

@ti.all_archs
def _test_svd(n, dt):
  ti.get_runtime().set_default_fp(dt)
  ti.get_runtime().set_verbose(False)
  ti.cfg.fast_math = False
  A = ti.Matrix(n, n, dt=dt, shape=())
  A_reconstructed = ti.Matrix(n, n, dt=dt, shape=())
  U = ti.Matrix(n, n, dt=dt, shape=())
  UtU = ti.Matrix(n, n, dt=dt, shape=())
  sigma = ti.Matrix(n, n, dt=dt, shape=())
  V = ti.Matrix(n, n, dt=dt, shape=())
  VtV = ti.Matrix(n, n, dt=dt, shape=())
  
  @ti.kernel
  def run():
    for i in range(1):
      U[None], sigma[None], V[None] = ti.svd(A[None], dt)
      UtU[None] = ti.transposed(U[None]) @ U[None]
      VtV[None] = ti.transposed(V[None]) @ V[None]
      A_reconstructed[None] = U[None] @ sigma[None] @ ti.transposed(V[None])
    
  if n == 3:
    A[None] = [[1, 1, 3], [9, -3, 2], [-3, 4, 2]]
  else:
    A[None] = [[1, 1], [2, 3]]
  
  run()
  
  tol = 1e-5 if dt == ti.f32 else 1e-12
  
  assert mat_equal(UtU.to_numpy(), np.eye(n), tol=tol)
  assert mat_equal(VtV.to_numpy(), np.eye(n), tol=tol)
  assert mat_equal(A_reconstructed.to_numpy(), A.to_numpy(), tol=tol)
  for i in range(n):
    for j in range(n):
      if i != j:
        assert sigma[None][i, j] == approx(0)
        
def test_svd():
  _test_svd(2, ti.f32)
  _test_svd(2, ti.f64)
  _test_svd(3, ti.f32)
  _test_svd(3, ti.f64)


@ti.all_archs
def test_transpose_no_loop():
  A = ti.Matrix(3, 3, dt=ti.f32, shape=())
  U = ti.Matrix(3, 3, dt=ti.f32, shape=())
  sigma = ti.Matrix(3, 3, dt=ti.f32, shape=())
  V = ti.Matrix(3, 3, dt=ti.f32, shape=())
  
  @ti.kernel
  def run():
    U[None], sigma[None], V[None] = ti.svd(A[None])
  
  run()
  # As long as it passes compilation we are good
  
