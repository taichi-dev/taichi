import time

import numpy as np

import taichi as ti

real = ti.f32
ti.init(default_fp=real,
        arch=ti.cuda,
        async_mode=True,
        async_opt_listgen=True,
        async_opt_dse=True,
        async_opt_activation_demotion=True,
        async_opt_fusion=True,
        kernel_profiler=True)

# grid parameters
N = 256

n_mg_levels = 5
pre_and_post_smoothing = 2
bottom_smoothing = 150

use_multigrid = True

N_ext = N // 2  # number of ext cells set so that that total grid size is still power of 2
N_tot = 2 * N

# setup sparse simulation data arrays
r = [ti.field(dtype=real) for _ in range(n_mg_levels)]  # residual
z = [ti.field(dtype=real) for _ in range(n_mg_levels)]  # z = M^-1 r
x = ti.field(dtype=real)
p = ti.field(dtype=real)
Ap = ti.field(dtype=real)
alpha = ti.field(dtype=real)
beta = ti.field(dtype=real)
sum = ti.field(dtype=real)
rTr = ti.field(dtype=real, shape=())
old_zTr = ti.field(dtype=real, shape=())
new_zTr = ti.field(dtype=real, shape=())
pAp = ti.field(dtype=real, shape=())

leaf_size = 8

grid = ti.root.pointer(ti.ijk,
                       [N_tot // leaf_size]).dense(ti.ijk,
                                                   leaf_size).place(x, p, Ap)

for l in range(n_mg_levels):
    grid = ti.root.pointer(ti.ijk, [N_tot // (leaf_size * 2**l)]).dense(
        ti.ijk, leaf_size).place(r[l], z[l])

ti.root.place(alpha, beta, sum)


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
def reduce(p: ti.template(), q: ti.template(), s: ti.template()):
    s[None] = 0
    for I in ti.grouped(p):
        s[None] += p[I] * q[I]


@ti.kernel
def update_x():
    for I in ti.grouped(p):
        x[I] += alpha[None] * p[I]


@ti.kernel
def update_r():
    for I in ti.grouped(p):
        r[0][I] -= alpha[None] * Ap[I]


@ti.kernel
def update_p():
    for I in ti.grouped(p):
        p[I] = z[0][I] + beta[None] * p[I]


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


@ti.kernel
def smooth(l: ti.template(), phase: ti.template()):
    # phase = red/black Gauss-Seidel phase
    for i, j, k in r[l]:
        if (i + j + k) & 1 == phase:
            z[l][i,j,k] = (r[l][i,j,k] + z[l][i+1,j,k] + z[l][i-1,j,k] \
                                       + z[l][i,j+1,k] + z[l][i,j-1,k] \
                                       + z[l][i,j,k+1] + z[l][i,j,k-1])/6.0


def apply_preconditioner():
    z[0].fill(0)
    for l in range(n_mg_levels - 1):
        for i in range(pre_and_post_smoothing << l):
            smooth(l, 0)
            smooth(l, 1)
        z[l + 1].fill(0)
        r[l + 1].fill(0)
        restrict(l)

    for i in range(bottom_smoothing):
        smooth(n_mg_levels - 1, 0)
        smooth(n_mg_levels - 1, 1)

    for l in reversed(range(n_mg_levels - 1)):
        prolongate(l)
        for i in range(pre_and_post_smoothing << l):
            smooth(l, 1)
            smooth(l, 0)


init()

sum[None] = 0.0
reduce(r[0], r[0], rTr)
initial_rTr = rTr[None]

# r = b - Ax = b    since x = 0
# p = r = r + 0 p
if use_multigrid:
    apply_preconditioner()
else:
    z[0].copy_from(r[0])

update_p()

sum[None] = 0.0
reduce(z[0], r[0], old_zTr)
print('old_zTr', old_zTr)


@ti.kernel
def print_rTr():
    print('rTr', rTr[None])


@ti.kernel
def update_alpha():
    alpha[None] = old_zTr[None] / pAp[None]


@ti.kernel
def update_beta():
    beta[None] = new_zTr[None] / old_zTr[None]
    old_zTr[None] = new_zTr[None]


def iterate():
    # alpha = rTr / pTAp
    compute_Ap()
    reduce(p, Ap, pAp)
    update_alpha()

    # ti.sync()
    # x = x + alpha p
    update_x()

    # r = r - alpha Ap
    update_r()

    # check for convergence
    reduce(r[0], r[0], rTr)
    print_rTr()

    # z = M^-1 r
    if use_multigrid:
        apply_preconditioner()
    else:
        z[0].copy_from(r[0])

    # beta = new_rTr / old_rTr
    reduce(z[0], r[0], new_zTr)
    update_beta()

    # p = z + beta p
    update_p()


def loud_sync():
    t = time.time()
    ti.sync()
    print(f'{time.time() - t:.3f} s (compilation + execution)')


ti.sync()
for _ in range(3):
    for i in range(10):
        iterate()
    loud_sync()

ti.print_kernel_profile_info()
ti.core.print_stat()
ti.print_profile_info()
