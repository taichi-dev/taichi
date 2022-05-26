import taichi as ti

ti.init(arch=ti.cpu)

N = 10

# memory_layout: [x,y,z],[x,y,z],...,[x,y,z]
st0 = ti.ndarray(shape=(N, 3), dtype=ti.f32)  #(10, 3)
# (not works) memory_layout: [x,x,..,x],[y,y,..,y],[z,z,..,z]
st1 = ti.ndarray(shape=(3, N), dtype=ti.f32)  # not works (3, 10)

A = ti.linalg.SparseMatrix(n=10, m=10, dtype=ti.f32)


@ti.kernel
def fill_st0(ts: ti.types.ndarray()):
    for i in range(N):
        ts[i, 0] = i
        ts[i, 1] = i
        ts[i, 2] = i


@ti.kernel
def fill_st1(ts: ti.types.ndarray()):
    for i in range(N):
        ts[0, N] = i
        ts[1, N] = i
        ts[2, N] = i


fill_st0(st0)
A.build_from_ndarray(st0)
print(f"A built from st0:\n{A}")

# Not works as we expected
fill_st1(st1)
A.build_from_ndarray(st1)
print(f"A built from st1:\n{A}")

# memory_layout (AOS): [x,y,z],[x,y,z]....[x,y,z]
vt0_aos = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=N,
                            layout=ti.Layout.AOS)  #(10,3)
# (not works) memory_layout (SOA): [x,x,..,x],[y,y,..,y],[z,z,..,z]
vt0_soa = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=N,
                            layout=ti.Layout.SOA)  #(3,10)
# memory_layout (AOS/SOA): [x,y,z],[x,y,z]....[x,y,z]
vt1_aos = ti.Vector.ndarray(n=1,
                            dtype=ti.f32,
                            shape=3 * N,
                            layout=ti.Layout.AOS)  #(30,1)
vt1_soa = ti.Vector.ndarray(n=1,
                            dtype=ti.f32,
                            shape=3 * N,
                            layout=ti.Layout.SOA)  #(1,30)
vt2_aos = ti.Vector.ndarray(n=3 * N,
                            dtype=ti.f32,
                            shape=(),
                            layout=ti.Layout.AOS)  #(30,)
vt2_soa = ti.Vector.ndarray(n=3 * N,
                            dtype=ti.f32,
                            shape=(),
                            layout=ti.Layout.SOA)  #(30,)

B = ti.linalg.SparseMatrix(n=10, m=10, dtype=ti.f32)


@ti.kernel
def fill_vt0_aos(triplets: ti.types.ndarray()):
    for i in range(N):
        triplet = ti.Vector([i, (i + 1) % N, i], dt=ti.f32)
        triplets[i] = triplet


@ti.kernel
def fill_vt1(triplets: ti.types.ndarray()):
    for i in range(N):
        triplets[3 * i + 0][0] = i
        triplets[3 * i + 1][0] = (i + 1) % N
        triplets[3 * i + 2][0] = i


@ti.kernel
def fill_vt2(triplets: ti.types.ndarray()):
    for i in range(N):
        triplets[None][3 * i] = i
        triplets[None][3 * i + 1] = (i + 1) % N
        triplets[None][3 * i + 2] = i


fill_vt0_aos(vt0_aos)
B.build_from_ndarray(vt0_aos)
print(f"B built from vt0_aos:\n{B}")
fill_vt1(vt1_aos)
B.build_from_ndarray(vt1_aos)
print(f"B built from vt1 aos:\n{B}")
fill_vt1(vt1_soa)
B.build_from_ndarray(vt1_soa)
print(f"B built from vt1 soa:\n{B}")
fill_vt2(vt2_aos)
B.build_from_ndarray(vt2_aos)
print(f"B built from vt2 aos:\n{B}")
fill_vt2(vt2_soa)
B.build_from_ndarray(vt2_soa)
print(f"B built from vt2 soa:\n{B}")

# memory_layout (AOS/SOA): [x,y,z],[x,y,z]....[x,y,z]
mt0_aos = ti.Matrix.ndarray(n=10,
                            m=3,
                            dtype=ti.f32,
                            shape=(),
                            layout=ti.Layout.AOS)  #(10,3)
mt0_soa = ti.Matrix.ndarray(n=10,
                            m=3,
                            dtype=ti.f32,
                            shape=(),
                            layout=ti.Layout.SOA)  #(10,3)
mt1_aos = ti.Matrix.ndarray(n=1,
                            m=3,
                            dtype=ti.f32,
                            shape=(10, ),
                            layout=ti.Layout.AOS)  #(10,1,3)
mt3_aos = ti.Matrix.ndarray(n=3,
                            m=1,
                            dtype=ti.f32,
                            shape=(10, ),
                            layout=ti.Layout.AOS)  #(10,3,1)

# (not works) memory_layout (AOS/SOA): [x,x,..,x],[y,y,..,y],[z,z,..,z]
mt1_soa = ti.Matrix.ndarray(n=1,
                            m=3,
                            dtype=ti.f32,
                            shape=(10, ),
                            layout=ti.Layout.SOA)  #(1,3,10)
mt3_soa = ti.Matrix.ndarray(n=3,
                            m=1,
                            dtype=ti.f32,
                            shape=(10, ),
                            layout=ti.Layout.SOA)  #(3,1,10)
mt2_aos = ti.Matrix.ndarray(n=3,
                            m=10,
                            dtype=ti.f32,
                            shape=(),
                            layout=ti.Layout.AOS)  #(3,10)
mt2_soa = ti.Matrix.ndarray(n=3,
                            m=10,
                            dtype=ti.f32,
                            shape=(),
                            layout=ti.Layout.SOA)  #(3,10)

C = ti.linalg.SparseMatrix(n=10, m=10, dtype=ti.f32)


@ti.kernel
def fill_mt0(mts: ti.types.ndarray()):
    for i in range(N):
        mts[None][i, 0] = i
        mts[None][i, 1] = (i + 2) % N
        mts[None][i, 2] = i


@ti.kernel
def fill_mt1(mts: ti.types.ndarray()):
    for i in range(N):
        mts[i][0, 0] = i
        mts[i][0, 1] = (i + 2) % N
        mts[i][0, 2] = i


@ti.kernel
def fill_mt2(mts: ti.types.ndarray()):
    for i in range(N):
        mts[None][0, i] = i
        mts[None][1, i] = (i + 2) % N
        mts[None][2, i] = i


@ti.kernel
def fill_mt3(mts: ti.types.ndarray()):
    for i in range(N):
        mts[i][0, 0] = i
        mts[i][1, 0] = (i + 2) % N
        mts[i][2, 0] = i


fill_mt0(mt0_aos)
C.build_from_ndarray(mt0_aos)
print(f"C built from mt0 aos:\n{C}")
fill_mt0(mt0_soa)
C.build_from_ndarray(mt0_soa)
print(f"C built from mt0 soa:\n{C}")

fill_mt1(mt1_aos)
C.build_from_ndarray(mt1_aos)
print(f"C built from mt1 aos:\n{C}")
fill_mt3(mt3_aos)
C.build_from_ndarray(mt3_aos)
print(f"C built from mt3 soa:\n{C}")

# Not works as we expected
fill_mt1(mt1_soa)
C.build_from_ndarray(mt1_soa)
print(f"C built from mt1 soa:\n{C}")
fill_mt3(mt3_soa)
C.build_from_ndarray(mt3_soa)
print(f"C built from mt3 soa:\n{C}")
