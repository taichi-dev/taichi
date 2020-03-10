import numpy as np
import taichi as ti

real = ti.f32
ti.init(default_fp=real, arch=ti.x64, enable_profiler=True)

# grid parameters
N = 128
N_gui = 512 # gui resolution

n_mg_levels = 4
pre_and_post_smoothing = 2
bottom_smoothing = 50

N_ext = N//2 # number of ext cells set so that that total grid size is still power of 2
N_tot = 2*N;

# setup sparse simulation data arrays
r     = ti.Vector(n_mg_levels, dt=real) # residual
z     = ti.Vector(n_mg_levels, dt=real) # M^-1 r
x     = ti.var(dt=real)                 # solution
p     = ti.var(dt=real)                 # conjugate gradient
Ap    = ti.var(dt=real)                 # matrix-vector product
alpha = ti.var(dt=real)                 # step size
beta  = ti.var(dt=real)                 # step size
sum   = ti.var(dt=real)                 # storage for reductions
phase = ti.var(dt=real)                 # red/black Gauss-Seidel phase
pixels = ti.var(dt=real, shape=(N_gui,N_gui))   # image buffer

@ti.layout
def place():
  grid = ti.root.pointer(ti.ijk, [N_tot//4]).dense(ti.ijk,4)
  grid.place(x)
  grid.place(p)
  grid.place(Ap)

  for l in range(n_mg_levels):
    grid = ti.root.pointer(ti.ijk, [N_tot//(4*2**l)]).dense(ti.ijk,4)
    grid.place(r(l))
    grid.place(z(l))

  ti.root.place(alpha, beta, sum, phase)


@ti.kernel
def init():
  for i,j,k in ti.ndrange((N_ext,N_tot-N_ext),(N_ext,N_tot-N_ext),(N_ext,N_tot-N_ext)):
    xl = (i-N_ext)*2.0/N_tot
    yl = (j-N_ext)*2.0/N_tot
    zl = (k-N_ext)*2.0/N_tot
    r(0)[i,j,k] = ti.sin(2.0*np.pi*xl) * ti.sin(2.0*np.pi*yl) * ti.sin(2.0*np.pi*zl)
    z(0)[i,j,k] = 0.0
    Ap[i,j,k] = 0.0
    p[i,j,k] = 0.0
    x[i,j,k] = 0.0


@ti.kernel
def compute_Ap():
  for i,j,k in Ap:
    Ap[i,j,k] = 6.0 * p[i,j,k] - p[i+1,j,k] - p[i-1,j,k] \
                               - p[i,j+1,k] - p[i,j-1,k] \
                               - p[i,j,k+1] - p[i,j,k-1]

@ti.kernel
def reduce_rTr():
  for i,j,k in r(0):
    sum[None] += r(0)[i,j,k]*r(0)[i,j,k]

@ti.kernel
def reduce_zTr():
  for i,j,k in z(0):
    sum[None] += z(0)[i,j,k]*r(0)[i,j,k]

@ti.kernel
def reduce_pAp():
  for i,j,k in p:
    sum[None] += p[i,j,k]*Ap[i,j,k]

@ti.kernel
def update_x():
  for i,j,k in p:
    x[i,j,k] += alpha[None]*p[i,j,k]

@ti.kernel
def update_r():
  for i,j,k in p:
    r(0)[i,j,k] -= alpha[None]*Ap[i,j,k]

@ti.kernel
def update_p():
  for i,j,k in p:
    p[i,j,k] = z(0)[i,j,k] + beta[None]*p[i,j,k]

def make_restrict(l):
  @ti.kernel
  def kernel():
    for i,j,k in r(l):
      res = r(l)[i,j,k] - (6.0*z(l)[i,j,k] \
                 - z(l)[i+1,j,k] - z(l)[i-1,j,k] \
                 - z(l)[i,j+1,k] - z(l)[i,j-1,k] \
                 - z(l)[i,j,k+1] - z(l)[i,j,k-1])
      r(l+1)[i//2,j//2,k//2] += res*0.5
  return kernel

def make_prolongate(l):
  @ti.kernel
  def kernel():
    for i,j,k in z(l):
      z(l)[i,j,k] = z(l+1)[i//2,j//2,k//2]
  return kernel

def make_smooth(l):
  @ti.kernel
  def kernel():
    for i,j,k in r(l):
      ret = 0.0
      if (i+j+k)&1 == phase[None]:
        z(l)[i,j,k] = (r(l)[i,j,k] + z(l)[i+1,j,k] + z(l)[i-1,j,k] \
                                   + z(l)[i,j+1,k] + z(l)[i,j-1,k] \
                                   + z(l)[i,j,k+1] + z(l)[i,j,k-1])/6.0
  return kernel

def make_clear_r(l):
  @ti.kernel
  def kernel():
    for i,j,k in r(l):
      r(l)[i,j,k] = 0.0
  return kernel

def make_clear_z(l):
  @ti.kernel
  def kernel():
    for i,j,k in z(l):
      z(l)[i,j,k] = 0.0
  return kernel

# make kernels for each multigrid level
restrict = np.zeros(n_mg_levels-1, dtype=ti.Kernel)
prolongate = np.zeros(n_mg_levels-1, dtype=ti.Kernel)
for l in range(n_mg_levels-1):
  restrict[l] = make_restrict(l)
  prolongate[l] = make_prolongate(l)

smooth = np.zeros(n_mg_levels, dtype=ti.Kernel)
clear_r = np.zeros(n_mg_levels, dtype=ti.Kernel)
clear_z = np.zeros(n_mg_levels, dtype=ti.Kernel)
for l in range(n_mg_levels):
  smooth[l] = make_smooth(l)
  clear_r[l] = make_clear_r(l)
  clear_z[l] = make_clear_z(l)

@ti.kernel
def identity():
  for i,j,k in z(0):
    z(0)[i,j,k] = r(0)[i,j,k]

def apply_preconditioner():
  clear_z[0]()
  for l in range(n_mg_levels-1):
    for i in range(pre_and_post_smoothing << l):
      phase[None] = 0
      smooth[l]()
      phase[None] = 1
      smooth[l]()
    clear_z[l+1]()
    clear_r[l+1]()
    restrict[l]()

  for i in range(bottom_smoothing):
    phase[None] = 0
    smooth[n_mg_levels-1]()
    phase[None] = 1
    smooth[n_mg_levels-1]()

  for l in reversed(range(n_mg_levels-1)):
    prolongate[l]()
    for i in range(pre_and_post_smoothing << l):
      phase[None] = 1
      smooth[l]()
      phase[None] = 0
      smooth[l]()

@ti.kernel
def paint():
  kk = N_tot*3//8
  for i,j in pixels:
    ii = int(i*N/N_gui) + N_ext
    jj = int(j*N/N_gui) + N_ext
    pixels[i,j] = x[ii,jj,kk]/N_tot

gui = ti.GUI("mgpcg", res=(N_gui,N_gui))

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
  alpha[None] = old_zTr/pAp

  # x = x + alpha p
  update_x()

  # r = r - alpha Ap
  update_r()

  # check for convergence
  sum[None] = 0.0
  reduce_rTr()
  rTr = sum[None]
  if rTr < initial_rTr*1.0e-12:
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
