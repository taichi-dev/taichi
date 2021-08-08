import taichi as ti


@ti.require(ti.extension.bls)
@ti.all_archs
def test_simple_1d():
    x, y = ti.field(ti.f32), ti.field(ti.f32)

    N = 64
    bs = 16

    ti.root.pointer(ti.i, N // bs).dense(ti.i, bs).place(x, y)

    @ti.kernel
    def populate():
        for i in range(N):
            x[i] = i

    @ti.kernel
    def copy():
        ti.block_local(x)
        for i in x:
            y[i] = x[i]

    populate()
    copy()

    for i in range(N):
        assert y[i] == i


@ti.require(ti.extension.bls)
@ti.all_archs
def test_simple_2d():
    x, y = ti.field(ti.f32), ti.field(ti.f32)

    N = 16
    bs = 16

    ti.root.pointer(ti.ij, N // bs).dense(ti.ij, bs).place(x, y)

    @ti.kernel
    def populate():
        for i, j in ti.ndrange(N, N):
            x[i, j] = i - j

    @ti.kernel
    def copy():
        ti.block_local(x)
        for i, j in x:
            y[i, j] = x[i, j]

    populate()
    copy()

    for i in range(N):
        for j in range(N):
            assert y[i, j] == i - j


@ti.require(ti.extension.bls)
@ti.all_archs
def _test_bls_stencil(*args, **kwargs):
    from .bls_test_template import bls_test_template
    bls_test_template(*args, **kwargs)


def test_gather_1d_trivial():
    # y[i] = x[i]
    _test_bls_stencil(1, 128, bs=32, stencil=((0, ), ))


def test_gather_1d():
    # y[i] = x[i - 1] + x[i]
    _test_bls_stencil(1, 128, bs=32, stencil=((-1, ), (0, )))


def test_gather_2d():
    stencil = [(0, 0), (0, -1), (0, 1), (1, 0)]
    _test_bls_stencil(2, 128, bs=16, stencil=stencil)


def test_gather_2d_nonsquare():
    stencil = [(0, 0), (0, -1), (0, 1), (1, 0)]
    _test_bls_stencil(2, 128, bs=(4, 16), stencil=stencil)


def test_gather_3d():
    stencil = [(-1, -1, -1), (2, 0, 1)]
    _test_bls_stencil(3, 64, bs=(4, 8, 16), stencil=stencil)


def test_scatter_1d_trivial():
    # y[i] = x[i]
    _test_bls_stencil(1, 128, bs=32, stencil=((0, ), ), scatter=True)


def test_scatter_1d():
    _test_bls_stencil(1, 128, bs=32, stencil=(
        (1, ),
        (0, ),
    ), scatter=True)


def test_scatter_2d():
    stencil = [(0, 0), (0, -1), (0, 1), (1, 0)]
    _test_bls_stencil(2, 128, bs=16, stencil=stencil, scatter=True)


@ti.require(ti.extension.bls)
@ti.all_archs
def test_multiple_inputs():
    x, y, z, w, w2 = ti.field(ti.i32), ti.field(ti.i32), ti.field(
        ti.i32), ti.field(ti.i32), ti.field(ti.i32)

    N = 128
    bs = 8

    ti.root.pointer(ti.ij, N // bs).dense(ti.ij, bs).place(x, y, z, w, w2)

    @ti.kernel
    def populate():
        for i, j in ti.ndrange((bs, N - bs), (bs, N - bs)):
            x[i, j] = i - j
            y[i, j] = i + j * j
            z[i, j] = i * i - j

    @ti.kernel
    def copy(bls: ti.template(), w: ti.template()):
        if ti.static(bls):
            ti.block_local(x, y, z)
        for i, j in x:
            w[i,
              j] = x[i, j - 2] + y[i + 2, j -
                                   1] + y[i - 1, j] + z[i - 1, j] + z[i + 1, j]

    populate()
    copy(False, w2)
    copy(True, w)

    for i in range(N):
        for j in range(N):
            assert w[i, j] == w2[i, j]


@ti.require(ti.extension.bls)
@ti.all_archs
def test_bls_large_block():
    n = 2**10
    block_size = 32
    stencil_length = 28  # uses 60 * 60 * 4B = 14.0625KiB shared memory

    a = ti.field(dtype=ti.f32)
    b = ti.field(dtype=ti.f32)
    block = ti.root.pointer(ti.ij, n // block_size)
    block.dense(ti.ij, block_size).place(a)
    block.dense(ti.ij, block_size).place(b)

    @ti.kernel
    def foo():
        ti.block_dim(512)
        ti.block_local(a)
        for i, j in a:
            for k in range(stencil_length):
                b[i, j] += a[i + k, j]
                b[i, j] += a[i, j + k]

    foo()


# TODO: BLS on CPU
# TODO: BLS boundary out of bound
# TODO: BLS with TLS
