import taichi as ti


@ti.all_archs
def _test_dimensionality(d):
    x = ti.Vector.field(2, dtype=ti.i32, shape=(2, ) * d)

    @ti.kernel
    def fill():
        for I in ti.grouped(x):
            x[I] += [I.sum(), I[0]]

    for i in range(2**d):
        indices = []
        for j in range(d):
            indices.append(i // (2**j) % 2)
        x.__getitem__(tuple(indices))[0] = sum(indices) * 2
    fill()
    # FIXME(yuanming-hu): snode_writer needs 9 arguments actually..
    if ti.cfg.arch == ti.cc and d >= 8:
        return
    for i in range(2**d):
        indices = []
        for j in range(d):
            indices.append(i // (2**j) % 2)
        assert x.__getitem__(tuple(indices))[0] == sum(indices) * 3


def test_dimensionality():
    for i in range(2, ti.core.get_max_num_indices() + 1):
        _test_dimensionality(i)
