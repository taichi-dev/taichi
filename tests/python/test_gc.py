import taichi as ti
from tests import test_utils


def _test_block_gc():
    N = 100000

    dx = 1 / 128
    inv_dx = 1.0 / dx

    x = ti.Vector.field(2, dtype=ti.f32)

    indices = ti.ij

    grid_m = ti.field(dtype=ti.i32)

    grid = ti.root.pointer(indices, 64)
    grid.pointer(indices, 32).dense(indices, 8).place(grid_m)

    ti.root.dense(ti.i, N).place(x)

    @ti.kernel
    def init():
        for i in x:
            x[i] = ti.Vector([ti.random() * 0.1 + 0.5, ti.random() * 0.1 + 0.5])

    init()

    @ti.kernel
    def build_grid():
        for p in x:
            base = int(ti.floor(x[p] * inv_dx - 0.5))
            grid_m[base] += 1

    @ti.kernel
    def move():
        for p in x:
            x[p] += ti.Vector([0.0, 0.1])

    assert grid._num_dynamically_allocated == 0
    for _ in range(100):
        grid.deactivate_all()
        # Scatter the particles to the sparse grid
        build_grid()
        # Move the block of particles
        move()

    ti.sync()
    # The block of particles can occupy at most two blocks on the sparse grid.
    # It's fine to run 100 times and do just one final check, because
    # num_dynamically_allocated stores the number of slots *ever* allocated.
    assert 1 <= grid._num_dynamically_allocated <= 2, grid._num_dynamically_allocated


@test_utils.test(require=ti.extension.sparse)
def test_block():
    _test_block_gc()


@test_utils.test(require=ti.extension.sparse, exclude=ti.metal)
def test_dynamic_gc():
    x = ti.field(dtype=ti.i32)

    L = ti.root.dynamic(ti.i, 1024 * 1024, chunk_size=1024)
    L.place(x)

    assert L._num_dynamically_allocated == 0

    for i in range(100):
        x[1024] = 1
        L.deactivate_all()
        assert L._num_dynamically_allocated <= 2


@test_utils.test(require=ti.extension.sparse)
def test_pointer_gc():
    x = ti.field(dtype=ti.i32)

    L = ti.root.pointer(ti.ij, 32)
    L.pointer(ti.ij, 32).dense(ti.ij, 8).place(x)

    assert L._num_dynamically_allocated == 0

    for i in range(1024):
        x[i * 8, i * 8] = 1
        assert L._num_dynamically_allocated == 1
        L.deactivate_all()

        # Note that being inactive doesn't mean it's not allocated.
        assert L._num_dynamically_allocated == 1
