import taichi as ti

ti.init(arch=ti.gpu, print_ir=True)

def _test_bls_stencil(dim, N, bs, stencil, block_dim=None, scatter=False):
    x, y, y2 = ti.var(ti.i32), ti.var(ti.i32), ti.var(ti.i32)
    
    index = ti.indices(*range(dim))
    mismatch = ti.var(ti.i32, shape=())
    
    if not isinstance(bs, (tuple, list)):
        bs = [bs for _ in range(dim)]


    grid_size = [N // bs[i] for i in range(dim)]
    
    if scatter:
        block = ti.root.pointer(index, grid_size)
    
        block.dense(index, bs).place(x)
        block.dense(index, bs).place(y)
        block.dense(index, bs).place(y2)
    else:
        ti.root.pointer(index, grid_size).dense(index, bs).place(x)
        ti.root.pointer(index, grid_size).dense(index, bs).place(y)
        ti.root.pointer(index, grid_size).dense(index, bs).place(y2)
    
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
                s += I[i]**(i + 1)
            x[I] = s
    
    @ti.kernel
    def apply(use_bls: ti.template(), y: ti.template()):
        if ti.static(use_bls and not scatter):
            ti.cache_shared(x)
        if ti.static(use_bls and scatter):
            ti.cache_shared(y)
        
        ti.block_dim(block_dim)
        for I in ti.grouped(x):
            if ti.static(scatter):
                for offset in ti.static(stencil):
                    y[I + ti.Vector(offset)] += x[I]
            else:
                # gather
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
            print('check', I, y[I], y2[I])
            if y[I] != y2[I]:
                mismatch[None] = 1
    
    check()
    
    assert mismatch[None] == 0

_test_bls_stencil(1, 128, bs=32, stencil=((1, ), (0, ), ), scatter=True)


'''
def test_stencil_1d():
    # y[i] = x[i - 1] + x[i]
    _test_bls_stencil(1, 128, bs=32, stencil=((-1, ), (0, )))
'''
