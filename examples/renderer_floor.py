import taichi as ti

res = 1280, 720
ret = ti.var(dt=ti.f32, shape=())

dist_limit = 100

# ti.runtime.print_preprocessed = True
# ti.cfg.print_ir = True
ti.cfg.arch = ti.cuda

@ti.func
def sdf(o):
  return o[1] - 0.027

inf = 1e10

@ti.func
def ray_march(p, d):
  j = 0
  dist = 0.0
  limit = 200
  while j < limit and sdf(p + dist * d) > 1e-8 and dist < dist_limit:
    dist += sdf(p + dist * d)
    j += 1
  if dist > dist_limit:
    dist = inf
  return dist


@ti.func
def sdf_normal(p):
  d = 1e-3
  n = ti.Vector([0.0, 0.0, 0.0])
  for i in ti.static(range(3)):
    inc = p
    dec = p
    inc[i] += d
    dec[i] -= d
    n[i] = (0.5 / d) * (sdf(inc) - sdf(dec))
  return ti.Matrix.normalized(n)


@ti.func
def next_hit(pos, d):
  closest = inf
  normal = ti.Vector([0.0, 0.0, 0.0])
  c = ti.Vector([0.0, 0.0, 0.0])

  ray_march_dist = ray_march(pos, d)
  if ray_march_dist < dist_limit and ray_march_dist < closest:
    closest = ray_march_dist
    normal = sdf_normal(ti.Vector([0.0, 0.0, 0.0]))
    c = [1, 0.5, 0.5]

  return closest, normal, c


@ti.kernel
def render():
  pos = ti.Vector([0.5, 0.27, 2.7])
  closest, normal, c = next_hit(pos, ti.Vector([0.0, 0.0, -1.0]))
  n = normal.norm()
  ret[None] = n

render()
print(ret[None])
