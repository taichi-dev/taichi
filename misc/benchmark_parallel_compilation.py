# This file has a kernel with 16 equal offloaded tasks.

import taichi as ti
ti.init(arch=ti.x64)
quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters
x = ti.Vector(2, dt=ti.f32, shape=n_particles)  # position
v = ti.Vector(2, dt=ti.f32, shape=n_particles)  # velocity
C = ti.Matrix(2, 2, dt=ti.f32, shape=n_particles)  # affine velocity field
F = ti.Matrix(2, 2, dt=ti.f32, shape=n_particles)  # deformation gradient
material = ti.var(dt=ti.i32, shape=n_particles)  # material id
Jp = ti.var(dt=ti.f32, shape=n_particles)  # plastic deformation
grid_v = ti.Vector(2, dt=ti.f32,
                   shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid))  # grid node mass


@ti.kernel
def substep():
    for K in ti.static(range(4)):
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p]
            h = ti.exp(10 * (1.0 - Jp[p]))
            if material[p] == 1:
                h = 0.3
            mu, la = mu_0 * h, lambda_0 * h
            if material[p] == 0:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(F[p])
            J = 1.0
            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                if material[p] == 2:  # Snow
                    new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
                Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if material[p] == 0:
                F[p] = ti.Matrix.identity(ti.f32, 2) * ti.sqrt(J)
            elif material[p] == 2:
                F[p] = U @ sig @ V.T()
            stress = 2 * mu * (F[p] - U @ V.T()) @ F[p].T(
            ) + ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
            stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
            affine = stress + p_mass * C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v[base +
                       offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass


for i in range(32):
    substep()

ti.profiler_print()
ti.core.print_profile_info()
