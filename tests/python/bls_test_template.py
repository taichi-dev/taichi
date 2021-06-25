import random

import numpy as np

import taichi as ti


def bls_test_template(dim,
                      N,
                      bs,
                      stencil,
                      block_dim=None,
                      scatter=False,
                      benchmark=0,
                      dense=False):
    x, y, y2 = ti.field(ti.i32), ti.field(ti.i32), ti.field(ti.i32)

    index = ti.indices(*range(dim))
    mismatch = ti.field(ti.i32, shape=())

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
            ti.block_local(x)
        if ti.static(use_bls and scatter):
            ti.block_local(y)

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
            x.snode.parent().deactivate_all()
            if not scatter:
                populate()
            y.snode.parent().deactivate_all()
            y2.snode.parent().deactivate_all()
            apply(False, y2)
            apply(True, y)
    else:
        # Simply test
        apply(False, y2)
        apply(True, y)

    @ti.kernel
    def check():
        for I in ti.grouped(y2):
            if y[I] != y2[I]:
                print('check failed', I, y[I], y2[I])
                mismatch[None] = 1

    check()

    ti.kernel_profiler_print()

    assert mismatch[None] == 0


def bls_particle_grid(N,
                      ppc=8,
                      block_size=16,
                      scatter=True,
                      benchmark=0,
                      pointer_level=1,
                      sort_points=True,
                      use_offset=True):
    M = N * N * ppc

    m1 = ti.field(ti.f32)
    m2 = ti.field(ti.f32)
    m3 = ti.field(ti.f32)
    pid = ti.field(ti.i32)
    err = ti.field(ti.i32, shape=())

    max_num_particles_per_block = block_size**2 * 4096

    x = ti.Vector.field(2, dtype=ti.f32)

    s1 = ti.field(dtype=ti.f32)
    s2 = ti.field(dtype=ti.f32)
    s3 = ti.field(dtype=ti.f32)

    ti.root.dense(ti.i, M).place(x)
    ti.root.dense(ti.i, M).place(s1, s2, s3)

    if pointer_level == 1:
        block = ti.root.pointer(ti.ij, N // block_size)
    elif pointer_level == 2:
        block = ti.root.pointer(ti.ij, N // block_size // 4).pointer(ti.ij, 4)
    else:
        raise ValueError('pointer_level must be 1 or 2')

    if use_offset:
        grid_offset = (-N // 2, -N // 2)
        grid_offset_block = (-N // 2 // block_size, -N // 2 // block_size)
        world_offset = -0.5
    else:
        grid_offset = (0, 0)
        grid_offset_block = (0, 0)
        world_offset = 0

    block.dense(ti.ij, block_size).place(m1, offset=grid_offset)
    block.dense(ti.ij, block_size).place(m2, offset=grid_offset)
    block.dense(ti.ij, block_size).place(m3, offset=grid_offset)

    block.dynamic(ti.l,
                  max_num_particles_per_block,
                  chunk_size=block_size**2 * ppc * 4).place(
                      pid, offset=grid_offset_block + (0, ))

    bound = 0.1

    extend = 4

    x_ = [(random.random() * (1 - 2 * bound) + bound + world_offset,
           random.random() * (1 - 2 * bound) + bound + world_offset)
          for _ in range(M)]
    if sort_points:
        x_.sort(key=lambda q: int(q[0] * N) // block_size * N + int(q[1] * N)
                // block_size)

    x.from_numpy(np.array(x_, dtype=np.float32))

    @ti.kernel
    def insert():
        ti.block_dim(256)
        for i in x:
            # It is important to ensure insert and p2g uses the exact same way to compute the base
            # coordinates. Otherwise there might be coordinate mismatch due to float-point errors.
            base = ti.Vector([
                int(ti.floor(x[i][0] * N) - grid_offset[0]),
                int(ti.floor(x[i][1] * N) - grid_offset[1])
            ])
            base_p = ti.rescale_index(m1, pid, base)
            ti.append(pid.parent(), base_p, i)

    scatter_weight = (N * N / M) * 0.01

    @ti.kernel
    def p2g(use_shared: ti.template(), m: ti.template()):
        ti.block_dim(256)
        if ti.static(use_shared):
            ti.block_local(m)
        for I in ti.grouped(pid):
            p = pid[I]

            u_ = ti.floor(x[p] * N).cast(ti.i32)
            Im = ti.rescale_index(pid, m, I)
            u0 = ti.assume_in_range(u_[0], Im[0], 0, 1)
            u1 = ti.assume_in_range(u_[1], Im[1], 0, 1)

            u = ti.Vector([u0, u1])

            for offset in ti.static(ti.grouped(ti.ndrange(extend, extend))):
                m[u + offset] += scatter_weight

    @ti.kernel
    def p2g_naive():
        ti.block_dim(256)
        for p in x:
            u = ti.floor(x[p] * N).cast(ti.i32)

            for offset in ti.static(ti.grouped(ti.ndrange(extend, extend))):
                m3[u + offset] += scatter_weight

    @ti.kernel
    def fill_m1():
        for i, j in ti.ndrange(N, N):
            m1[i, j] = ti.random()

    @ti.kernel
    def g2p(use_shared: ti.template(), s: ti.template()):
        ti.block_dim(256)
        if ti.static(use_shared):
            ti.block_local(m1)
        for I in ti.grouped(pid):
            p = pid[I]

            u_ = ti.floor(x[p] * N).cast(ti.i32)

            Im = ti.rescale_index(pid, m1, I)
            u0 = ti.assume_in_range(u_[0], Im[0], 0, 1)
            u1 = ti.assume_in_range(u_[1], Im[1], 0, 1)

            u = ti.Vector([u0, u1])

            tot = 0.0

            for offset in ti.static(ti.grouped(ti.ndrange(extend, extend))):
                tot += m1[u + offset]

            s[p] = tot

    @ti.kernel
    def g2p_naive(s: ti.template()):
        ti.block_dim(256)
        for p in x:
            u = ti.floor(x[p] * N).cast(ti.i32)

            tot = 0.0
            for offset in ti.static(ti.grouped(ti.ndrange(extend, extend))):
                tot += m1[u + offset]
            s[p] = tot

    insert()

    for i in range(benchmark):
        pid.parent(2).snode.deactivate_all()
        insert()

    @ti.kernel
    def check_m():
        for i in range(grid_offset[0], grid_offset[0] + N):
            for j in range(grid_offset[1], grid_offset[1] + N):
                if abs(m1[i, j] - m3[i, j]) > 1e-4:
                    err[None] = 1
                if abs(m2[i, j] - m3[i, j]) > 1e-4:
                    err[None] = 1

    @ti.kernel
    def check_s():
        for i in range(M):
            if abs(s1[i] - s2[i]) > 1e-4:
                err[None] = 1
            if abs(s1[i] - s3[i]) > 1e-4:
                err[None] = 1

    if scatter:
        for i in range(max(benchmark, 1)):
            p2g(True, m1)
            p2g(False, m2)
            p2g_naive()
        check_m()
    else:
        for i in range(max(benchmark, 1)):
            g2p(True, s1)
            g2p(False, s2)
            g2p_naive(s3)
        check_s()

    assert not err[None]
