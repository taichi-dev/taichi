import taichi as ti

@ti.require(ti.extension.bls)
@ti.all_archs
def test_simple_1d():
    x, y = ti.var(ti.f32), ti.var(ti.f32)

    N = 64
    bs = 16

    ti.root.pointer(ti.i, N // bs).dense(ti.i, bs).place(x, y)

    @ti.kernel
    def populate():
        for i in range(N):
            x[i] = i

    @ti.kernel
    def copy():
        ti.cache_shared(x)
        for i in x:
            y[i] = x[i]


    populate()
    copy()

    for i in range(N):
        assert y[i] == i

@ti.require(ti.extension.bls)
@ti.all_archs
def test_simple_2d():
    x, y = ti.var(ti.f32), ti.var(ti.f32)

    N = 16
    bs = 16

    ti.root.pointer(ti.ij, N // bs).dense(ti.ij, bs).place(x, y)

    @ti.kernel
    def populate():
        for i,j in ti.ndrange(N, N):
            x[i, j] = i - j

    @ti.kernel
    def copy():
        ti.cache_shared(x)
        for i, j in x:
            y[i, j] = x[i, j]


    populate()
    copy()

    for i in range(N):
        for j in range(N):
            assert y[i, j] == i - j


@ti.require(ti.extension.bls)
@ti.all_archs
def _test_bls_stencil(dim, N, bs, stencil, block_dim=None):
    x, y, y2 = ti.var(ti.i32), ti.var(ti.i32), ti.var(ti.i32)
    
    index = ti.indices(*range(dim))
    mismatch = ti.var(ti.i32, shape=())
    
    if not isinstance(bs, (tuple, list)):
        bs = [bs for _ in range(dim)]
    
    block = ti.root.pointer(index, [N // bs[i] for i in range(dim)])

    block.dense(index, bs).place(x)
    block.dense(index, bs).place(y)
    block.dense(index, bs).place(y2)

    ndrange = ((bs[i], N - bs[i]) for i in range(dim))
    
    if block_dim is None:
        block_dim = 1
        for i in range(dim):
            block_dim *= bs[i]
    
    @ti.kernel
    def populate():
        for I in ti.grouped(ti.ndrange(*ndrange)):
            s = 0
            for i in ti.static(range(dim)):
                s += I[i] ** (i + 1)
            x[I] = s

    @ti.kernel
    def apply(use_bls: ti.template(), y: ti.template()):
        if ti.static(use_bls):
            ti.cache_shared(x)
        ti.block_dim(block_dim)
        for I in ti.grouped(x):
            s = 0
            for offset in ti.static(stencil):
                s = s + x[I + ti.Vector(offset)]
            y[I] = s


    populate()
    apply(False, y2)
    apply(True, y)
    
    @ti.kernel
    def check():
        for I in ti.grouped(y2):
            if y[I] != y2[I]:
                mismatch[None] = 1
                
    check()
    
    assert mismatch[None] == 0

def test_stencil_1d_trivial():
    # y[i] = x[i]
    _test_bls_stencil(1, 128, bs=32, stencil=((0,)))

def test_stencil_1d():
    # y[i] = x[i - 1] + x[i]
    _test_bls_stencil(1, 128, bs=32, stencil=((-1,), (0, )))
    
def test_stencil_2d():
    stencil = [(0, 0), (0, -1), (0, 1), (1, 0)]
    _test_bls_stencil(2, 128, bs=16, stencil=stencil)
    
def test_stencil_3d():
    stencil = [(-1, -1, -1), (2, 0, 1)]
    _test_bls_stencil(3, 64, bs=4, stencil=stencil)
    
# TODO: multiple-variable BLS
# TODO: BLS epilogues
# TODO: BLS on CPU
# TODO: BLS with TLS
