import taichi as ti

ti.init(arch=ti.cuda, kernel_profiler=True)

N = 512
M = 1024 * 1024 * 1
block_size = 16

m1 = ti.var(ti.f32)
m2 = ti.var(ti.f32)
m3 = ti.var(ti.f32)
pid = ti.var(ti.i32)

max_num_particles_per_block = block_size ** 2 * 32

x = ti.Vector(2, dt=ti.f32, shape=M)


block = ti.root.pointer(ti.ij, N // block_size // 4).pointer(ti.ij, 4)
block.dense(ti.ij, block_size).place(m1)
block.dense(ti.ij, block_size).place(m2)
block.dense(ti.ij, block_size).place(m3)

block.dynamic(ti.l, max_num_particles_per_block, chunk_size=1024).place(pid)

bound = 0.1

print(f'ppc {M / ((1 - bound * 2) ** 2 * N ** 2)}')

@ti.kernel
def insert():
    for i in x:
        x[i] = ti.Vector([ti.random() * (1 - 2 * bound) + bound, ti.random() * (1 - 2 * bound) + bound])
        ti.append(pid.parent(), [int(x[i][0] * N), int(x[i][1] * N)], i)
        
@ti.kernel
def p2g(use_shared: ti.template(), m: ti.template()):
    ti.block_dim(128)
    if ti.static(use_shared):
        ti.cache_shared(m)
    for i, j, l in pid:
        p = pid[i, j, l]

        u_ = (x[p] * N).cast(ti.i32)

        u0 = ti.assume_in_range(u_[0], i, 0, 1)
        u1 = ti.assume_in_range(u_[1], j, 0, 1)

        u = ti.Vector([u0, u1])

        for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
            m[u + offset] += (N * N / M) * 0.01
        

@ti.kernel
def p2g_naive():
    ti.block_dim(128)
    for p in x:
        u = (x[p] * N).cast(ti.i32)
        
        for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
            m3[u + offset] += (N * N / M) * 0.01
        
        
insert()

for i in range(10):
    p2g(True, m1)
    p2g(False, m2)
    p2g_naive()
    
for i in range(N):
    for j in range(N):
        assert abs(m1[i, j] - m3[i, j]) < 1e-4
        assert abs(m2[i, j] - m3[i, j]) < 1e-4

ti.kernel_profiler_print()
# print(m.to_numpy())
# exit(0)

ti.imshow(m1, 'density')

# TODO: debug dynamic: prologues/epilogues threads should not be limited to dynamic length!!

# TODO: debug mode behavior of assume_in_range
