import taichi as ti
import numpy as np
import random


def bls_test_template(dim,
                      N,
                      bs,
                      stencil,
                      block_dim=None,
                      scatter=False,
                      benchmark=0,
                      dense=False):
    x, y, y2 = ti.var(ti.i32), ti.var(ti.i32), ti.var(ti.i32)

    index = ti.indices(*range(dim))
    mismatch = ti.var(ti.i32, shape=())

    if not isinstance(bs, (tuple, list)):
        bs = [bs for _ in range(dim)]

    grid_size = [N // bs[i] for i in range(dim)]

    if dense:
        create_block = lambda: ti.root.dense(index, grid_size)
    else:
        create_block = lambda: ti.root.pointer(index, grid_size)

    if scatter:
        block = create_block()

        block.dense(index, bs).place(x)
        block.dense(index, bs).place(y)
        block.dense(index, bs).place(y2)
    else:
        create_block().dense(index, bs).place(x)
        create_block().dense(index, bs).place(y)
        create_block().dense(index, bs).place(y2)

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

    if benchmark:
        for i in range(benchmark):
            x.snode().parent().deactivate_all()
            if not scatter:
                populate()
            y.snode().parent().deactivate_all()
            y2.snode().parent().deactivate_all()
            apply(False, y2)
            apply(True, y)
    else:
        # Simply test
        apply(False, y2)
        apply(True, y)

    @ti.kernel
    def check():
        for I in ti.grouped(y2):
            # print('check', I, y[I], y2[I])
            if y[I] != y2[I]:
                mismatch[None] = 1

    check()

    ti.kernel_profiler_print()

    assert mismatch[None] == 0


def bls_scatter(N, ppc=8, block_size=16, benchmark=0, pointer_level=1, sort_points=True):
    M = N * N * ppc

    m1 = ti.var(ti.f32)
    m2 = ti.var(ti.f32)
    m3 = ti.var(ti.f32)
    pid = ti.var(ti.i32)
    err = ti.var(ti.i32, shape=())

    max_num_particles_per_block = block_size**2 * 4096

    x = ti.Vector(2, dt=ti.f32)

    ti.root.dense(ti.i, M).place(x)

    if pointer_level == 1:
        block = ti.root.pointer(ti.ij, N // block_size)
    elif pointer_level == 2:
        block = ti.root.pointer(ti.ij, N // block_size // 4).pointer(ti.ij, 4)
    else:
        raise ValueError('pointer_level must be 1 or 2')
        
    block.dense(ti.ij, block_size).place(m1)
    block.dense(ti.ij, block_size).place(m2)
    block.dense(ti.ij, block_size).place(m3)

    block.dynamic(ti.l, max_num_particles_per_block,
                  chunk_size=block_size ** 2 * ppc * 4).place(pid)

    bound = 0.1

    extend = 4
    
    x_ = [(random.random() * (1 - 2 * bound) + bound, random.random() * (1 - 2 * bound) + bound) for _ in range(M)]
    if sort_points:
        x_.sort(key=lambda q: int(q[0] * N) // block_size * N + int(q[1] * N) // block_size)
    
    x.from_numpy(np.array(x_, dtype=np.float32))

    @ti.kernel
    def insert():
        ti.block_dim(256)
        for i in x:
            x[i] = ti.Vector([
                ti.random() * (1 - 2 * bound) + bound,
                ti.random() * (1 - 2 * bound) + bound
            ])
            ti.append(pid.parent(), [int(x[i][0] * N), int(x[i][1] * N)], i)

    @ti.kernel
    def p2g(use_shared: ti.template(), m: ti.template()):
        ti.block_dim(256)
        if ti.static(use_shared):
            ti.cache_shared(m)
        for i, j, l in pid:
            p = pid[i, j, l]

            u_ = (x[p] * N).cast(ti.i32)

            u0 = ti.assume_in_range(u_[0], i, 0, 1)
            u1 = ti.assume_in_range(u_[1], j, 0, 1)

            u = ti.Vector([u0, u1])

            for offset in ti.static(ti.grouped(ti.ndrange(extend, extend))):
                m[u + offset] += (N * N / M) * 0.01

    @ti.kernel
    def p2g_naive():
        ti.block_dim(256)
        for p in x:
            u = (x[p] * N).cast(ti.i32)

            for offset in ti.static(ti.grouped(ti.ndrange(extend, extend))):
                m3[u + offset] += (N * N / M) * 0.01

    insert()

    for i in range(max(benchmark, 1)):
        p2g(True, m1)
        p2g(False, m2)
        p2g_naive()
        
    @ti.kernel
    def check():
        for i in range(N):
            for j in range(N):
                if abs(m1[i, j] - m3[i, j]) > 1e-4:
                    err[None] = True
                if abs(m2[i, j] - m3[i, j]) > 1e-4:
                    err[None] = True
    check()
    
    assert not err[None]
