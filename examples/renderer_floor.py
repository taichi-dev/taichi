import taichi as ti

ret = ti.var(dt=ti.f32, shape=())

ti.cfg.arch = ti.cuda

@ti.kernel
def test():
  if True:
    d = 1e-3
    n = ti.Vector([0.0, 0.0, 0.0])
    p = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
      inc = p
      dec = p
      inc[i] += d
      dec[i] -= d
      n[i] = (0.5 / d) * (inc[1] - dec[1])
    ret[None] = n[1]

test()
print(ret[None])
