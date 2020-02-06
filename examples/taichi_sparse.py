import taichi as ti

n = 512
x = ti.var(ti.f32)
img = ti.var(ti.f32, shape=(n, n))

block1 = ti.root.dense(ti.ij, n // 64).pointer()
block2 = block1.dense(ti.ij, n // 8).pointer()
block2.dense(ti.ij, 8).place(x)

@ti.func
def Vector2(x, y):
  return ti.Vector([x, y])


@ti.func
def inside(p, c, r):
  return (p - c).norm_sqr() <= r * r


@ti.func
def inside_taichi(p):
  p = Vector2(0.5, 0.5) + (p - Vector2(0.5, 0.5)) * 1.11
  ret = -1
  if not inside(p, Vector2(0.50, 0.50), 0.55):
    if ret == -1:
      ret = 0
  if not inside(p, Vector2(0.50, 0.50), 0.50):
    if ret == -1:
      ret = 1
  if inside(p, Vector2(0.50, 0.25), 0.09):
    if ret == -1:
      ret = 1
  if inside(p, Vector2(0.50, 0.75), 0.09):
    if ret == -1:
      ret = 0
  if inside(p, Vector2(0.50, 0.25), 0.25):
    if ret == -1:
      ret = 0
  if inside(p, Vector2(0.50, 0.75), 0.25):
    if ret == -1:
      ret = 1
  if p[0] < 0.5:
    if ret == -1:
      ret = 1
  else:
    if ret == -1:
      ret = 0
  return ret

@ti.kernel
def activate():
  for i, j in ti.ndrange(n, n):
      t = inside_taichi(Vector2(i / n, j / n))
      if t != 0:
        x[i, j] = t


@ti.kernel
def paint():
  for i, j in x:
    t = x[i, j]
    img[i, j] = t * 0.5 + 0.5
    
activate()
paint()

gui = ti.GUI('Logo', (512, 512))
while True:
  gui.set_image(1 - img.to_numpy())
  gui.show()
