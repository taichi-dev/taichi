import taichi as ti


@ti.host_arch_only
def test_indices():
    a = ti.var(ti.f32, shape=(128, 32, 8))

    b = ti.var(ti.f32)
    ti.root.dense(ti.j, 32).dense(ti.i, 16).place(b)

    ti.get_runtime().materialize()

    mapping_a = a.snode().physical_index_position()

    assert mapping_a == {0: 0, 1: 1, 2: 2}

    mapping_b = b.snode().physical_index_position()

    assert mapping_b == {0: 1, 1: 0}
    # Note that b is column-major:
    # the virtual first index exposed to the user comes second in memory layout.
