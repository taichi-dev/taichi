import taichi as ti

ti.init(arch=ti.cuda, kernel_profiler=True, print_ir=True)

N = 256
# M = 1024 * 1024 * 4
M = 1
block_size = 16

m = ti.var(ti.f32)
pid = ti.var(ti.i32)

max_num_particles_per_block = 1024 * 32

x = ti.Vector(2, dt=ti.f32, shape=M)

block = ti.root.pointer(ti.ij, N // block_size)
block.dense(ti.ij, block_size).place(m)
block.dynamic(ti.l, max_num_particles_per_block, chunk_size=1024).place(pid)


@ti.kernel
def insert():
    for i in x:
        x[i] = ti.Vector([ti.random() * 0.8 + 0.1, ti.random() * 0.8 + 0.1])
        ti.append(pid.parent(), [int(x[i][0] * N), int(x[i][1] * N)], i)

@ti.kernel
def p2g():
    ti.block_dim(256)
    ti.cache_shared(m)
    for i, j, l in pid:
        p = pid[i, j, l]
        
        u_ = (x[p] * N).cast(ti.i32)
        
        u0 = ti.assume_in_range(u_[0], i, 0, 1)
        u1 = ti.assume_in_range(u_[1], j, 0, 1)
        print(u0, u1)
        
        u = ti.Vector([u0, u1])
        
        for offset in ti.static(ti.grouped(ti.ndrange(2, 2))):
            # m[u + offset] += ((N * N) / M) * 0.002
            m[u + offset] += 1
        

insert()

for i in range(1):
    p2g()

ti.kernel_profiler_print()
exit(0)

ti.imshow(m, 'density')

# TODO: debug mode behavior of assume_in_range
