import taichi_lang as ti

real = ti.f32
dim = 2
n_particles = 1024
n_grid = 32

def vec():
  return ti.Vector(dim, dt=real)

def mat():
  return ti.Matrix(dim, dim, dt=real)

x = vec()
v = vec()

@ti.layout
def place():
  ti.root.dense(ti.index(2), n_particles).place(x, v)

@ti.kernel
def p2g():
  for i in x(0):
    x[i] = x[i] + v[i] * 0.01


'''
@ti.kernel
def grid_op():
  pass


@ti.kernel
def g2p():
  pass
'''


def main():
  pass


if __name__ == '__main__':
  v(0)[1] = 1
  for f in range(100):
    for s in range(20):
      p2g()
      #grid_op()
      #g2p()
  print(x(0)[1])

