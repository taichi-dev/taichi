# Authored by Erqi Chen.
# This script shows the optimization process of differential evolution.
# The black points are the search agents, and they finally find the minimum solution.


import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ti.init(arch=ti.cpu)


@ti.func
def clip(_pop: ti.template(), _lb: ti.template(), _ub: ti.template()):
    _search_num, _dim = _pop.shape
    for ii, j in ti.ndrange(_search_num, _dim):
        if _pop[ii, j] > _ub[j]:
            _pop[ii, j] = _ub[j]
        elif _pop[ii, j] < _lb[j]:
            _pop[ii, j] = _lb[j]


@ti.func
def clip_only(_trial: ti.template(), _lb: ti.template(), _ub: ti.template()):
    _dim = _trial.shape[0]
    for j in range(_dim):
        if _trial[j] > _ub[j]:
            _trial[j] = _ub[j]
        elif _trial[j] < _lb[j]:
            _trial[j] = _lb[j]


@ti.func
def f1(_fit: ti.template(), _pop: ti.template()):
    _search_num, _dim = _pop.shape
    for ii in range(_search_num):
        cur = 0.0
        for j in range(_dim):
            cur += ti.pow(_pop[ii, j], 2)

        _fit[ii] = cur


@ti.func
def f1_only(_trial: ti.template()) -> ti.float32:
    _dim = _trial.shape[0]
    _res = 0.0
    for j in range(_dim):
        _res += ti.pow(_trial[j], 2)

    return _res


@ti.func
def find_min(_fit: ti.template()) -> ti.i32:
    _search_num = _fit.shape[0]
    min_fit = _fit[0]
    min_pos = 0
    for _ in ti.ndrange(1):
        for ii in ti.ndrange(_search_num):
            if min_fit < _fit[ii]:
                min_fit = _fit[ii]
                min_pos = ii
    return min_pos


@ti.func
def rand_int(low: ti.i32, high: ti.i32) -> ti.i32:
    r = ti.random(float)
    _res = r * (high - low) + low

    return ti.round(_res, dtype=ti.i32)


@ti.func
def copy_pop_to_field(_pop: ti.template(), _trial: ti.template(), ind: ti.i32):
    _, _dim = _pop.shape
    for j in range(_dim):
        _trial[j] = _pop[ind, j]


@ti.func
def copy_field_to_pop(_pop: ti.template(), _trial: ti.template(), ind: ti.i32):
    _, _dim = _pop.shape
    for j in range(dim):
        _pop[ind, j] = _trial[j]


@ti.func
def copy_2d_to_3d(a: ti.template(), b: ti.template(), _iter: ti.i32):
    r, c = b.shape
    for ii, j in ti.ndrange(r, c):
        a[_iter, ii, j] = b[ii, j]


@ti.func
def copy_field_a_to_b(a: ti.template(), b: ti.template()):
    _dim = a.shape[0]
    for j in range(_dim):
        b[j] = a[j]


@ti.func
def de_crossover(_pop: ti.template(), _trial: ti.template(), a: ti.i32, b: ti.i32, c: ti.i32):
    _, _dim = _pop.shape
    CR = 0.5
    para_F = 0.7
    for k in range(_dim):
        r = ti.random(float)
        if r < CR or k == _dim - 1:
            _trial[k] = _pop[c, k] + para_F * (_pop[a, k] - pop[b, k])


@ti.func
def de_loop(
    _pop: ti.template(),
    all_best: ti.float32,
    _fit: ti.template(),
    _trial: ti.template(),
    _lb: ti.template(),
    _ub: ti.template(),
) -> ti.float32:
    _search_num, _ = _pop.shape
    for ii in range(_search_num):
        copy_pop_to_field(_pop=_pop, _trial=_trial, ind=ii)

        a = rand_int(low=0, high=_search_num)
        while a == ii:
            a = rand_int(low=0, high=_search_num)

        b = rand_int(low=0, high=_search_num)
        while b == ii or a == b:
            b = rand_int(low=0, high=_search_num)

        c = rand_int(low=0, high=_search_num)
        while c == ii or c == a or c == b:
            c = rand_int(low=0, high=_search_num)

        de_crossover(_pop=_pop, _trial=_trial, a=a, b=b, c=c)
        clip_only(_trial=_trial, _lb=_lb, _ub=_ub)
        next_fit = f1_only(_trial=_trial)
        if next_fit < _fit[ii]:
            copy_field_to_pop(_pop=_pop, _trial=_trial, ind=ii)
            _fit[ii] = next_fit
            if next_fit < all_best:
                all_best = next_fit
                copy_field_a_to_b(a=_trial, b=best_pop)

    return all_best


@ti.kernel
def DE(
    _pop: ti.template(),
    _max_iter: ti.i32,
    _lb: ti.template(),
    _ub: ti.template(),
    _fit: ti.template(),
    _best_fit: ti.template(),
    _trial: ti.template(),
):
    f1(_fit=_fit, _pop=_pop)
    min_pos = find_min(_fit=_fit)
    all_best = _fit[min_pos]
    _best_fit[0] = all_best
    copy_2d_to_3d(a=all_pop, b=_pop, _iter=0)

    for _ in range(1):
        for cur_iter in range(1, _max_iter + 1):
            all_best = de_loop(_pop=_pop, _fit=_fit, all_best=all_best, _trial=_trial, _lb=_lb, _ub=_ub)
            _best_fit[cur_iter] = all_best
            copy_2d_to_3d(a=all_pop, b=_pop, _iter=cur_iter)


search_num = 20
dim = 2
max_iter = 50

_lb = np.ones(dim).astype(np.int32) * (-100)
lb = ti.field(ti.i32, shape=dim)
lb.from_numpy(_lb)

_ub = np.ones(dim).astype(np.int32) * 100
ub = ti.field(ti.i32, shape=dim)
ub.from_numpy(_ub)

pop = ti.field(ti.float32, shape=(search_num, dim))
pop.from_numpy((np.random.random((search_num, dim)) * (_ub - _lb) + _lb).astype(np.float32))

fit = ti.field(ti.float32, shape=(search_num,))
best_fit = ti.field(ti.float32, shape=(max_iter,))
best_pop = ti.field(ti.float32, shape=(search_num,))
all_pop = ti.field(ti.float32, shape=(max_iter, search_num, dim))

trial = ti.field(ti.float32, shape=(search_num,))

DE(_pop=pop, _max_iter=max_iter, _lb=lb, _ub=ub, _fit=fit, _best_fit=best_fit, _trial=trial)

res = best_fit.to_numpy()


@ti.kernel
def draw_contour():
    for ii, j in ti.ndrange(201, 201):
        z[ii, j] = x[ii] ** 2 + y[j] ** 2


_x = np.arange(-100, 101, 1)
x = ti.field(ti.float32, shape=201)
x.from_numpy(_x)
_y = np.arange(-100, 101, 1)
y = ti.field(ti.float32, shape=201)
y.from_numpy(_y)
z = ti.field(ti.float32, shape=(201, 201))

draw_contour()

_z = z.to_numpy()
_pop = all_pop.to_numpy()

plt.ion()

"""2d visualization"""
plt.contourf(_x, _y, _z)
plt.colorbar()

for i in range(max_iter):
    plt.cla()
    plt.contourf(_x, _y, _z)
    plt.scatter(_pop[i, :, 0], _pop[i, :, 1], color="black")
    plt.title(f"cur_iter: {i}, best_fit: {best_fit[i]:.2f}")
    # plt.savefig(f"./2dimg/iter-{i}.png")
    plt.pause(0.5)


# import imageio.v2 as imageio
# import os
#
# png_ls = os.listdir("./img")
# f = []
# for i in png_ls:
#     f.append(imageio.imread("./img/" + i))
#
# imageio.mimsave("res.gif", f, "GIF", duration=0.5)


"""3d visualization"""
mesh_x, mesh_y = np.meshgrid(_x, _y)

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
ax.view_init(elev=51, azim=-70)
fig.add_axes(ax)
ax.plot_surface(mesh_x, mesh_y, _z, cmap="viridis", alpha=0.7)

for i in range(max_iter):
    ax.cla()
    ax.plot_surface(mesh_x, mesh_y, _z, cmap="viridis", alpha=0.7)

    row = []
    col = []
    val = []
    nr, _ = _pop[i, :, :].shape
    for _i in range(nr - 1):
        row.append(np.round(_pop[i, _i, 0]).astype(int))
        col.append(np.round(_pop[i, _i, 1]).astype(int))
        val.append(_z[np.round(_pop[i, _i, 0]).astype(int) + 100, np.round(_pop[i, _i, 1]).astype(int) + 100])

    ax.scatter3D(row, col, val, color="black")
    # plt.savefig(f"./3dimg/iter-{i}.png")
    plt.pause(0.5)
#
#
# import imageio.v2 as imageio
# import os
#
# png_ls = os.listdir("./3dimg")
# f = []
# for i in png_ls:
#     f.append(imageio.imread("./3dimg/" + i))
# imageio.mimsave("3dres.gif", f, "GIF", duration=0.5)
