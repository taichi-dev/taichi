import numpy as np
import taichi as ti

real = ti.f32
ti.init(default_fp=real, arch=ti.x64, async_mode=True, async_opt_listgen=False, async_opt_dse=True, async_opt_fusion=False, kernel_profiler=False
#, async_opt_intermediate_file="mgpcg"
)

# grid parameters
N = 128

n_mg_levels = 4

use_multigrid = True

N_ext = N // 2  # number of ext cells set so that that total grid size is still power of 2
N_tot = 2 * N

# setup sparse simulation data arrays
r = [ti.field(dtype=real) for _ in range(n_mg_levels)]  # residual
z = [ti.field(dtype=real) for _ in range(n_mg_levels)]  # M^-1 r
x = ti.field(dtype=real)  # solution

grid = ti.root.pointer(ti.ijk, [N_tot // 4]).dense(ti.ijk, 4).place(x)

for l in range(n_mg_levels):
    grid = ti.root.pointer(ti.ijk,
                           [N_tot // (4 * 2**l)]).dense(ti.ijk,
                                                        4).place(r[l], z[l])


@ti.kernel
def init():
    for i, j, k in ti.ndrange((N_ext, N_tot - N_ext), (N_ext, N_tot - N_ext),
                              (N_ext, N_tot - N_ext)):
        xl = (i - N_ext) * 2.0 / N_tot
        yl = (j - N_ext) * 2.0 / N_tot
        zl = (k - N_ext) * 2.0 / N_tot
        r[0][i, j, k] = ti.sin(2.0 * np.pi * xl) * ti.sin(
            2.0 * np.pi * yl) * ti.sin(2.0 * np.pi * zl)
        z[0][i, j, k] = 0.0


@ti.kernel
def restrict(l: ti.template()):
    for i, j, k in r[l]:
        res = r[l][i, j, k] - (6.0 * z[l][i, j, k] - z[l][i + 1, j, k] -
                               z[l][i - 1, j, k] - z[l][i, j + 1, k] -
                               z[l][i, j - 1, k] - z[l][i, j, k + 1] -
                               z[l][i, j, k - 1])
        r[l + 1][i // 2, j // 2, k // 2] += res * 0.5


@ti.kernel
def prolongate(l: ti.template()):
    for I in ti.grouped(z[l]):
        z[l][I] = z[l + 1][I // 2]



def apply_preconditioner():
    z[0].fill(0)
    for l in range(n_mg_levels - 1):
        z[l + 1].fill(0)
        r[l + 1].fill(0)
        restrict(l)

    for l in reversed(range(n_mg_levels - 1)):
        prolongate(l)

init()

# CG
for i in range(5):
    if use_multigrid:
        apply_preconditioner()
    else:
        z[0].copy_from(r[0])


ti.sync()
