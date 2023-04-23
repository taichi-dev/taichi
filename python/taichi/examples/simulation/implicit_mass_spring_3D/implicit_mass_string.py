import numpy as np

from node import *
import taichi as ti

ti.init(arch=ti.vulkan, debug=True)
ti.init(device_memory_fraction=1.0)


mesh = NODE("armadillo_4k")

NV = mesh.vn
NS = mesh.sn
ks = 30000
kd = 0.95  # damping
dt = 0.02
x = ti.Vector.field(3, float, NV)
pos = ti.field(float, 3 * NV)
force = ti.field(float, 3 * NV)
velocity = ti.field(float, 3 * NV)
b_field = ti.field(float, 3 * NV)


# ======================= init I =======================
@ti.kernel
def fill(A: ti.types.sparse_matrix_builder()):
    for i in range(3 * NV):
        A[i, i] += 1


def fill3(A: ti.types.sparse_matrix_builder()):
    for i in range(3):
        A[i, i] += 1


I_builder = ti.linalg.SparseMatrixBuilder(3 * NV, 3 * NV, max_num_triplets=1000000)
fill(I_builder)
I = I_builder.build()

I3 = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
# ======================= init/update pos =======================
@ti.kernel
def init_pos():
    for i in range(NV):
        for j in range(3):
            pos[3 * i + j] = mesh.vertices[i][j]


@ti.kernel
def update_pos():
    for i in range(NV):
        for j in range(3):
            pos[3 * i + j] = pos[3 * i + j] + dt * velocity[3 * i + j]


@ti.kernel
def update_x():
    for i in range(NV):
        for j in range(3):
            x[i][j] = pos[3 * i + j]


init_pos()


# ======================= init/update Force =======================
@ti.kernel
def init_force():
    for i in range(NV):
        force[3 * i] = 0.
        force[3 * i + 1] = - 9.8
        force[3 * i + 2] = 0.


@ti.kernel
def update_force():
    for i in range(NS):
        p1_index = mesh.springs[i][0]
        p2_index = mesh.springs[i][1]
        rest_p1 = mesh.vertices[p1_index]
        rest_p2 = mesh.vertices[p2_index]
        p1 = ti.Vector([pos[3 * p1_index + j] for j in range(3)])
        p2 = ti.Vector([pos[3 * p2_index + j] for j in range(3)])
        v1 = ti.Vector([velocity[3 * p1_index + j] for j in range(3)])
        v2 = ti.Vector([velocity[3 * p2_index + j] for j in range(3)])
        # part_f is p1 -> p2
        l0 = (rest_p1 - rest_p2).norm()
        l = (p1 - p2).norm()
        direction = 0 if l == 0 else (p1 - p2) / l

        # fi = f_p2
        f2 = ks * direction * (l - l0)
        # fj = f_p1
        f1 = -f2
        for j in range(3):
            force[3 * p1_index + j] += f1[j]
            force[3 * p2_index + j] += f2[j]

        fd1 = kd * ((v2 - v1).dot(direction)) * direction
        fd2 = -fd1
        for j in range(3):
            force[3 * p1_index + j] += fd1[j]
            force[3 * p2_index + j] += fd2[j]


@ti.kernel
def update_J(jx_b: ti.types.sparse_matrix_builder(), jv_b: ti.types.sparse_matrix_builder()):
    for i in range(NS):
        p1_index = mesh.springs[i][0]
        p2_index = mesh.springs[i][1]
        rest_p1 = mesh.vertices[p1_index]
        rest_p2 = mesh.vertices[p2_index]
        p1 = ti.Vector([pos[3 * p1_index+j] for j in range(3)])
        p2 = ti.Vector([pos[3 * p2_index+j] for j in range(3)])
        v1 = ti.Vector([velocity[3 * p1_index + j] for j in range(3)])
        v2 = ti.Vector([velocity[3 * p2_index + j] for j in range(3)])
        # part_f is p1 -> p2
        l0 = (rest_p1 - rest_p2).norm()
        l = (p1 - p2).norm()
        direction = 0 if l == 0 else (p1 - p2) / l
        v1_2_norm = 0 if((v1 - v2).norm()) == 0 else (v1 - v2)/((v1 - v2).norm())

        d_tenser = direction.outer_product(direction)

        # Jacobian
        inter_co = direction.dot(v1_2_norm)
        inter_mat = direction.outer_product(v1_2_norm)

        ratio = 0 if l == 0 else (1 - l0 / l)

        jfsx = ks * ((d_tenser-I3) * ratio - d_tenser)
        if l == 0:
            print("!!!!!L==0!!!!!")
        jdx = - kd * ((inter_co * I3 + inter_mat) @ ((d_tenser - I3) / l))
        jdv = kd * d_tenser

        for j in range(3):
            for k in range(3):
                # for Fs
                jx_b[3 * p1_index + j, 3 * p1_index + k] += jfsx[j, k]
                jx_b[3 * p2_index + j, 3 * p1_index + k] -= jfsx[j, k]
                jx_b[3 * p1_index + j, 3 * p2_index + k] -= jfsx[j, k]
                jx_b[3 * p2_index + j, 3 * p2_index + k] += jfsx[j, k]
                # for damping
                jx_b[3 * p1_index + j, 3 * p1_index + k] += jdx[j, k]
                jx_b[3 * p2_index + j, 3 * p1_index + k] -= jdx[j, k]
                jx_b[3 * p1_index + j, 3 * p2_index + k] -= jdx[j, k]
                jx_b[3 * p2_index + j, 3 * p2_index + k] += jdx[j, k]
                # for jdv
                jv_b[3 * p1_index + j, 3 * p1_index + k] += jdv[j, k]
                jv_b[3 * p2_index + j, 3 * p1_index + k] -= jdv[j, k]
                jv_b[3 * p1_index + j, 3 * p2_index + k] -= jdv[j, k]
                jv_b[3 * p2_index + j, 3 * p2_index + k] += jdv[j, k]


init_force()


# ======================= init/update velocity =======================
@ti.kernel
def init_velocity():
    for i in range(NV):
        for j in range(3):
            velocity[i * 3 + j] = 0


@ti.kernel
def update_v_mat(v_b: ti.types.sparse_matrix_builder()):
    for i in range(NV):
        for j in range(3):
            v_b[3 * i + j, 0] += velocity[3 * i + j]


init_velocity()


@ti.kernel
def update_b(Kv: ti.types.ndarray()):
    for i in range(3 * NV):
        b_field[i] = dt * force[i] + dt * dt * Kv[i]

@ti.kernel
def update_v(dv: ti.types.ndarray()):
    for i in range(3 * NV):
        # Pin points
        if i // 3 == 0 or i//3 == 1:
            velocity[i] = 0.0
        else:
            velocity[i] += dv[i]

def substep():
    init_force()

    Jx_builder = ti.linalg.SparseMatrixBuilder(3 * NV, 3 * NV, max_num_triplets=1000000)
    Jv_builder = ti.linalg.SparseMatrixBuilder(3 * NV, 3 * NV, max_num_triplets=1000000)
    update_J(Jx_builder, Jv_builder)
    Jx = Jx_builder.build()
    Jv = Jv_builder.build()

    update_force()

    A = I - dt * Jv - dt * dt * Jx

    vt = velocity.to_numpy()
    Kv = Jx @ vt
    update_b(Kv)
    b = b_field.to_numpy()

    solver = ti.linalg.SparseSolver(solver_type="LDLT")
    solver.analyze_pattern(A)
    solver.factorize(A)
    new_v = solver.solve(b)
    update_v(new_v)
    # isSuccess = solver.info()
    # print(f">>>> Computation was successful?: {isSuccess}")
    # velocity.from_numpy(new_v)
    # print(velocity)

    update_pos()
    update_x()




