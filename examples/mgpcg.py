import numpy as np
import taichi as ti

real = ti.f32
ti.init(default_fp=real, arch=ti.x64, enable_profiler=True)

# grid parameters
N = 128
N_gui = 512  # gui resolution

n_mg_levels = 4
pre_and_post_smoothing = 2
bottom_smoothing = 50

N_ext = N // 2  # number of ext cells set so that that total grid size is still power of 2
N_tot = 2 * N

# setup sparse simulation data arrays
r = ti.Vector(n_mg_levels, dt=real)  # residual
z = ti.Vector(n_mg_levels, dt=real)  # M^-1 r
x = ti.var(dt=real)  # solution
p = ti.var(dt=real)  # conjugate gradient
Ap = ti.var(dt=real)  # matrix-vector product
alpha = ti.var(dt=real)  # step size
beta = ti.var(dt=real)  # step size
sum = ti.var(dt=real)  # storage for reductions
pixels = ti.var(dt=real, shape=(N_gui, N_gui))  # image buffer

grid = ti.root.pointer(ti.ijk, [N_tot // 4]).dense(ti.ijk, 4)
grid.place(x)
grid.place(p)
grid.place(Ap)

for l in range(n_mg_levels):
  grid = ti.root.pointer(ti.ijk, [N_tot // (4 * 2**l)]).dense(ti.ijk, 4)
  grid.place(r(l))
  grid.place(z(l))

ti.root.place(alpha, beta, sum)


@ti.kernel
def init():
  for i, j, k in ti.ndrange((N_ext, N_tot - N_ext), (N_ext, N_tot - N_ext),
                            (N_ext, N_tot - N_ext)):
    xl = (i - N_ext) * 2.0 / N_tot
    yl = (j - N_ext) * 2.0 / N_tot
    zl = (k - N_ext) * 2.0 / N_tot
    r(0)[i, j, k] = ti.sin(2.0 * np.pi * xl) * ti.sin(
        2.0 * np.pi * yl) * ti.sin(2.0 * np.pi * zl)
    z(0)[i, j, k] = 0.0
    Ap[i, j, k] = 0.0
    p[i, j, k] = 0.0
    x[i, j, k] = 0.0


@ti.kernel
def compute_Ap():
  for i, j, k in Ap:
    Ap[i,j,k] = 6.0 * p[i,j,k] - p[i+1,j,k] - p[i-1,j,k] \
                               - p[i,j+1,k] - p[i,j-1,k] \
                               - p[i,j,k+1] - p[i,j,k-1]


@ti.kernel
def reduce_rTr():
  for i, j, k in r(0):
    sum[None] += r(0)[i, j, k] * r(0)[i, j, k]


@ti.kernel
def reduce_zTr():
  for i, j, k in z(0):
    sum[None] += z(0)[i, j, k] * r(0)[i, j, k]


@ti.kernel
def reduce_pAp():
  for i, j, k in p:
    sum[None] += p[i, j, k] * Ap[i, j, k]


@ti.kernel
def update_x():
  for i, j, k in p:
    x[i, j, k] += alpha[None] * p[i, j, k]


@ti.kernel
def update_r():
  for i, j, k in p:
    r(0)[i, j, k] -= alpha[None] * Ap[i, j, k]


@ti.kernel
def update_p():
  for i, j, k in p:
    p[i, j, k] = z(0)[i, j, k] + beta[None] * p[i, j, k]


@ti.kernel
def restrict(l: ti.template()):
  for i, j, k in r(l):
    res = r(l)[i,j,k] - (6.0*z(l)[i,j,k] \
               - z(l)[i+1,j,k] - z(l)[i-1,j,k] \
               - z(l)[i,j+1,k] - z(l)[i,j-1,k] \
               - z(l)[i,j,k+1] - z(l)[i,j,k-1])
    r(l + 1)[i // 2, j // 2, k // 2] += res * 0.5


@ti.kernel
def prolongate(l: ti.template()):
  for i, j, k in z(l):
    z(l)[i, j, k] = z(l + 1)[i // 2, j // 2, k // 2]


@ti.kernel
def smooth(l: ti.template(), phase: ti.template()):
  # phase = red/black Gauss-Seidel phase
  for i, j, k in r(l):
    if (i + j + k) & 1 == phase:
      z(l)[i,j,k] = (r(l)[i,j,k] + z(l)[i+1,j,k] + z(l)[i-1,j,k] \
                                 + z(l)[i,j+1,k] + z(l)[i,j-1,k] \
                                 + z(l)[i,j,k+1] + z(l)[i,j,k-1])/6.0


@ti.kernel
def identity():
  for i, j, k in z(0):
    z(0)[i, j, k] = r(0)[i, j, k]


def apply_preconditioner():
  z(0).fill(0)
  for l in range(n_mg_levels - 1):
    for i in range(pre_and_post_smoothing << l):
      smooth(l, 0)
      smooth(l, 1)
    z(l + 1).fill(0)
    r(l + 1).fill(0)
    restrict(l)

  for i in range(bottom_smoothing):
    smooth(n_mg_levels - 1, 0)
    smooth(n_mg_levels - 1, 1)

  for l in reversed(range(n_mg_levels - 1)):
    prolongate(l)
    for i in range(pre_and_post_smoothing << l):
      smooth(l, 1)
      smooth(l, 0)


@ti.kernel
def paint():
  kk = N_tot * 3 // 8
  for i, j in pixels:
    ii = int(i * N / N_gui) + N_ext
    jj = int(j * N / N_gui) + N_ext
    pixels[i, j] = x[ii, jj, kk] / N_tot


gui = ti.GUI("mgpcg", res=(N_gui, N_gui))

init()

sum[None] = 0.0
reduce_rTr()
initial_rTr = sum[None]

# r = b - Ax = b    since x = 0
# p = r = r + 0 p
apply_preconditioner()
#identity()
update_p()

sum[None] = 0.0
reduce_zTr()
old_zTr = sum[None]

# CG
for i in range(400):
  # alpha = rTr / pTAp
  compute_Ap()
  sum[None] = 0.0
  reduce_pAp()
  pAp = sum[None]
  alpha[None] = old_zTr / pAp

  # x = x + alpha p
  update_x()

  # r = r - alpha Ap
  update_r()

  # check for convergence
  sum[None] = 0.0
  reduce_rTr()
  rTr = sum[None]
  if rTr < initial_rTr * 1.0e-12:
    break

  # z = M^-1 r
  apply_preconditioner()
  #identity()

  # beta = new_rTr / old_rTr
  sum[None] = 0.0
  reduce_zTr()
  new_zTr = sum[None]
  beta[None] = new_zTr / old_zTr

  # p = z + beta p
  update_p()
  old_zTr = new_zTr

  print(' ')
  print(i)
  print(rTr)
  paint()
  gui.set_image(pixels)
  gui.show()

ti.profiler_print()
