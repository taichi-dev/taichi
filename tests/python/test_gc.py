import taichi as ti


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
            x[i] = [ti.random() * 0.1 + 0.5, ti.random() * 0.1 + 0.5]

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

    assert grid.num_dynamically_allocated == 0
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
    assert 1 <= grid.num_dynamically_allocated <= 2, grid.num_dynamically_allocated


@ti.test(require=ti.extension.sparse)
def test_block():
    _test_block_gc()


@ti.test(require=[ti.extension.sparse, ti.extension.async_mode],
         async_mode=True)
def test_block_async():
    _test_block_gc()


@ti.test(require=ti.extension.sparse)
def test_dynamic_gc():
    x = ti.field(dtype=ti.i32)

    L = ti.root.dynamic(ti.i, 1024 * 1024, chunk_size=1024)
    L.place(x)

    assert L.num_dynamically_allocated == 0

    for i in range(100):
        x[1024] = 1
        L.deactivate_all()
        assert L.num_dynamically_allocated <= 2


@ti.test(require=ti.extension.sparse)
def test_pointer_gc():
    x = ti.field(dtype=ti.i32)

    L = ti.root.pointer(ti.ij, 32)
    L.pointer(ti.ij, 32).dense(ti.ij, 8).place(x)

    assert L.num_dynamically_allocated == 0

    for i in range(1024):
        x[i * 8, i * 8] = 1
        assert L.num_dynamically_allocated == 1
        L.deactivate_all()

        # Note that being inactive doesn't mean it's not allocated.
        assert L.num_dynamically_allocated == 1


@ti.test(require=[ti.extension.sparse, ti.extension.async_mode],
         async_mode=True)
def test_fuse_allocator_state():
    N = 16
    x = ti.field(dtype=ti.i32, shape=N)
    y = ti.field(dtype=ti.i32)

    y_parent = ti.root.pointer(ti.i, N * 2)
    y_parent.place(y)

    # https://github.com/taichi-dev/taichi/pull/1973#pullrequestreview-511154376

    @ti.kernel
    def activate_y():
        for i in x:
            idx = i + 1
            y[idx] = idx

    @ti.kernel
    def deactivate_y():
        for i in x:
            ti.deactivate(y_parent, i)

    activate_y()
    deactivate_y()
    ti.sync()

    # TODO: assert that activate_y and deactivate_y are not fused.
    assert y_parent.num_dynamically_allocated == N
    ys = y.to_numpy()
    for i, y in enumerate(ys):
        expected = N if i == N else 0
        assert y == expected
