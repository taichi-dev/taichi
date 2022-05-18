import taichi as ti

ti.init(arch=ti.cpu, gdb_trigger=True)

N = 10
triplets = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=N)
A = ti.linalg.SparseMatrix(n=10, m=10, dtype=ti.f32)


@ti.kernel
def fill(triplets: ti.types.ndarray()):
    for i in range(N):
        triplet = ti.Vector([i, i, i], dt=ti.f32)
        triplets[i] = triplet


@ti.kernel
def print_triplets(triplets: ti.types.ndarray()):
    for i in range(N):
        print(triplets[i])


fill(triplets)
print_triplets(triplets)

A.build_from_ndarray(triplets)
print(A)
print(A.shape)
