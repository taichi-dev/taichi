import taichi as ti

ti.init(arch=ti.x64, debug=True, offline_cache=False)

n = 8
triplets = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=n)
A = ti.linalg.SparseMatrix(n=10,
                           m=10,
                           dtype=ti.f32,
                           storage_format='col_major')


@ti.kernel
def fill(triplets: ti.types.ndarray()):
    for i in range(n):
        triplet = ti.Vector([i, i, i], dt=ti.f32)
        triplets[i] = triplet


fill(triplets)
A.build_from_ndarray(triplets)
for i in range(n):
    assert A[i, i] == i
