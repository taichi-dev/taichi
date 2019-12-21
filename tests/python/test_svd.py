import taichi as ti
from pytest import approx

@ti.all_archs
def test_transpose():
  print(ti.core.test_tuple())
