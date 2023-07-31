import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_implicit_fem():
    E, nu = 5e4, 0.0
    mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # lambda = 0
    density = 1000.0
    dt = 2e-4

    use_sparse = 0

    n_cube = np.array([5] * 3)
    n_verts = np.product(n_cube)
    n_cells = 5 * np.product(n_cube - 1)
    dx = 1 / (n_cube.max() - 1)

    F_vertices = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)

    F_x = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    F_ox = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    F_v = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    F_f = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    F_mul_ans = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    F_m = ti.field(dtype=ti.f32, shape=n_verts)

    n_cells = (n_cube - 1).prod() * 5
    F_B = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_cells)
    F_W = ti.field(dtype=ti.f32, shape=n_cells)

    @ti.func
    def i2p(I):
        return (I.x * n_cube[1] + I.y) * n_cube[2] + I.z

    @ti.func
    def set_element(e, I, verts):
        for i in ti.static(range(3 + 1)):
            F_vertices[e][i] = i2p(I + (([verts[i] >> k for k in range(3)] ^ I) & 1))

    @ti.kernel
    def get_vertices():
        """
        This kernel partitions the cube into tetrahedrons.
        Each unit cube is divided into 5 tetrahedrons.
        """
        for I in ti.grouped(ti.ndrange(*(n_cube - 1))):
            e = ((I.x * (n_cube[1] - 1) + I.y) * (n_cube[2] - 1) + I.z) * 5
            for i, j in ti.static(enumerate([0, 3, 5, 6])):
                set_element(e + i, I, (j, j ^ 1, j ^ 2, j ^ 4))
            set_element(e + 4, I, (1, 2, 4, 7))
        for I in ti.grouped(ti.ndrange(*(n_cube))):
            F_ox[i2p(I)] = I * dx

    @ti.func
    def Ds(verts):
        return ti.Matrix.cols([F_x[verts[i]] - F_x[verts[3]] for i in range(3)])

    @ti.func
    def ssvd(F):
        U, sig, V = ti.svd(F)
        if U.determinant() < 0:
            for i in ti.static(range(3)):
                U[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        if V.determinant() < 0:
            for i in ti.static(range(3)):
                V[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        return U, sig, V

    @ti.func
    def get_force_func(c, verts):
        F = Ds(verts) @ F_B[c]
        P = ti.Matrix.zero(ti.f32, 3, 3)
        U, sig, V = ssvd(F)
        P = 2 * mu * (F - U @ V.transpose())
        H = -F_W[c] * P @ F_B[c].transpose()
        for i in ti.static(range(3)):
            force = ti.Vector([H[j, i] for j in range(3)])
            F_f[verts[i]] += force
            F_f[verts[3]] -= force

    @ti.kernel
    def get_force():
        for c in F_vertices:
            get_force_func(c, F_vertices[c])
        for u in F_f:
            F_f[u].y -= 9.8 * F_m[u]

    @ti.kernel
    def matmul_cell(ret: ti.template(), vel: ti.template()):
        for i in ret:
            ret[i] = vel[i] * F_m[i]
        for c in F_vertices:
            verts = F_vertices[c]
            W_c = F_W[c]
            B_c = F_B[c]
            for u in range(4):
                for d in range(3):
                    dD = ti.Matrix.zero(ti.f32, 3, 3)
                    if u == 3:
                        for j in range(3):
                            dD[d, j] = -1
                    else:
                        dD[d, u] = 1
                    dF = dD @ B_c
                    dP = 2.0 * mu * dF
                    dH = -W_c * dP @ B_c.transpose()
                    for i in range(3):
                        for j in range(3):
                            tmp = vel[verts[i]][j] - vel[verts[3]][j]
                            ret[verts[u]][d] += -(dt**2) * dH[j, i] * tmp

    @ti.kernel
    def add(ans: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
        for i in ans:
            ans[i] = a[i] + k * b[i]

    @ti.kernel
    def dot(a: ti.template(), b: ti.template()) -> ti.f32:
        ans = 0.0
        for i in a:
            ans += a[i].dot(b[i])
        return ans

    F_b = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    F_r0 = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    F_p0 = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

    # ndarray version of F_b
    F_b_ndarr = ti.ndarray(dtype=ti.f32, shape=3 * n_verts)
    # stiffness matrix
    A_builder = ti.linalg.SparseMatrixBuilder(3 * n_verts, 3 * n_verts, max_num_triplets=50000)
    solver = ti.linalg.SparseSolver(ti.f32, "LLT")

    @ti.kernel
    def get_b():
        for i in F_b:
            F_b[i] = F_m[i] * F_v[i] + dt * F_f[i]

    def cg():
        def mul(x):
            matmul_cell(F_mul_ans, x)
            return F_mul_ans

        get_force()
        get_b()
        mul(F_v)
        add(F_r0, F_b, -1, mul(F_v))

        d = F_p0
        d.copy_from(F_r0)
        r_2 = dot(F_r0, F_r0)
        n_iter = 50
        epsilon = 1e-6
        r_2_init = r_2
        r_2_new = r_2
        for _ in range(n_iter):
            q = mul(d)
            alpha = r_2_new / dot(d, q)
            add(F_v, F_v, alpha, d)
            add(F_r0, F_r0, -alpha, q)
            r_2 = r_2_new
            r_2_new = dot(F_r0, F_r0)
            if r_2_new <= r_2_init * epsilon**2:
                break
            beta = r_2_new / r_2
            add(d, F_r0, beta, d)
        F_f.fill(0)
        add(F_x, F_x, dt, F_v)

    @ti.kernel
    def compute_A(A: ti.types.sparse_matrix_builder()):
        # A = M - dt * dt * K
        for i in range(n_verts):
            for j in range(3):
                A[3 * i + j, 3 * i + j] += F_m[i]
        for c in F_vertices:
            verts = F_vertices[c]
            W_c = F_W[c]
            B_c = F_B[c]
            for u in range(4):
                for d in range(3):
                    dD = ti.Matrix.zero(ti.f32, 3, 3)
                    if u == 3:
                        for j in range(3):
                            dD[d, j] = -1
                    else:
                        dD[d, u] = 1
                    dF = dD @ B_c
                    dP = 2.0 * mu * dF
                    dH = -W_c * dP @ B_c.transpose()
                    for i in range(3):
                        for j in range(3):
                            A[3 * verts[u] + d, 3 * verts[i] + j] += -(dt**2) * dH[j, i]
                    for i in range(3):
                        A[3 * verts[u] + d, 3 * verts[3] + i] += -(dt**2) * (-dH[i, 0] - dH[i, 1] - dH[i, 2])

    @ti.kernel
    def flatten(dest: ti.types.ndarray(), src: ti.template()):
        for i in range(n_verts):
            for j in range(3):
                dest[3 * i + j] = src[i][j]

    @ti.kernel
    def aggragate(dest: ti.template(), src: ti.types.ndarray()):
        for i in range(n_verts):
            for j in range(3):
                dest[i][j] = src[3 * i + j]

    def direct():
        get_force()
        get_b()
        flatten(F_b_ndarr, F_b)
        v = solver.solve(F_b_ndarr)
        aggragate(F_v, v)
        F_f.fill(0)
        add(F_x, F_x, dt, F_v)

    @ti.kernel
    def advect():
        for p in F_x:
            F_v[p] += dt * (F_f[p] / F_m[p])
            F_x[p] += dt * F_v[p]
            F_f[p] = ti.Vector([0, 0, 0])

    @ti.kernel
    def init():
        for u in F_x:
            F_x[u] = F_ox[u]
            F_v[u] = [0.0] * 3
            F_f[u] = [0.0] * 3
            F_m[u] = 0.0
        for c in F_vertices:
            F = Ds(F_vertices[c])
            F_B[c] = F.inverse()
            F_W[c] = ti.abs(F.determinant()) / 6
            for i in range(4):
                F_m[F_vertices[c][i]] += F_W[c] / 4 * density
        # for u in F_x:
        #    F_x[u].y += 1.0

    def init_A():
        compute_A(A_builder)
        A = A_builder.build()
        solver.analyze_pattern(A)
        solver.factorize(A)

    @ti.kernel
    def floor_bound():
        for u in F_x:
            if F_x[u].y < 0:
                F_x[u].y = 0
                if F_v[u].y < 0:
                    F_v[u].y = 0

    @ti.func
    def check(u):
        ans = 0
        rest = u
        for i in ti.static(range(3)):
            k = rest % n_cube[2 - i]
            rest = rest // n_cube[2 - i]
            if k == 0:
                ans |= 1 << (i * 2)
            if k == n_cube[2 - i] - 1:
                ans |= 1 << (i * 2 + 1)
        return ans

    def gen_indices():
        su = 0
        for i in range(3):
            su += (n_cube[i] - 1) * (n_cube[(i + 1) % 3] - 1)
        return ti.field(ti.i32, shape=2 * su * 2 * 3)

    indices = gen_indices()

    @ti.kernel
    def get_indices():
        # calculate all the meshes on surface
        cnt = 0
        for c in F_vertices:
            if c % 5 != 4:
                for i in ti.static([0, 2, 3]):
                    verts = [F_vertices[c][(i + j) % 4] for j in range(3)]
                    sum_ = check(verts[0]) & check(verts[1]) & check(verts[2])
                    if sum_:
                        m = ti.atomic_add(cnt, 1)
                        det = ti.Matrix.rows([F_x[verts[i]] - [0.5, 1.5, 0.5] for i in range(3)]).determinant()
                        if det < 0:
                            tmp = verts[1]
                            verts[1] = verts[2]
                            verts[2] = tmp
                        indices[m * 3] = verts[0]
                        indices[m * 3 + 1] = verts[1]
                        indices[m * 3 + 2] = verts[2]

    def substep():
        cg()
        floor_bound()

    get_vertices()
    init()
    get_indices()
    substep()
