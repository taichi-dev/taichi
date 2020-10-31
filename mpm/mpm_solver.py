import taichi as ti

@ti.data_oriented
class MPMSolver:
    grid_size = 1024

    def __init__(
            self,
            res,
            max_num_particles=2 ** 27):
        self.dim = 2

        self.res = res
        self.n_particles = ti.var(ti.i32, shape=())
        self.max_num_particles = max_num_particles
        self.gravity = ti.Vector(self.dim, dt=ti.f32, shape=())
        self.source_bound = ti.Vector(self.dim, dt=ti.f32, shape=2)
        self.source_velocity = ti.Vector(self.dim, dt=ti.f32, shape=())
        self.pid = ti.var(ti.i32)
        # position
        self.x = ti.Vector(self.dim, dt=ti.f32)
        # velocity

        indices = ti.ij

        offset = tuple(-self.grid_size // 2 for _ in range(self.dim))
        self.offset = offset

        # grid node momentum/velocity
        self.grid_v = ti.Vector(self.dim, dt=ti.f32)
        # grid node mass
        self.grid_m = ti.var(dt=ti.f32)

        grid_block_size = 128
        self.grid1 = ti.root.pointer(indices, self.grid_size // grid_block_size)
        
        use2 = True
        if use2:
            self.grid2 = ti.root.dense(indices, self.grid_size // grid_block_size)

        self.leaf_block_size = 16

        block = self.grid1.pointer(indices,
                                  grid_block_size // self.leaf_block_size)
        if use2:
            block2 = self.grid2.dense(indices,
                                       grid_block_size // self.leaf_block_size)

        def block_component(blk, c):
            blk.dense(indices, self.leaf_block_size).place(c, offset=offset)

        block_component(block, self.grid_m)
        for v in self.grid_v.entries:
            block_component(block, v)


        if use2:
            self.grid_m2 = ti.var(dt=ti.f32)

            block_component(block2, self.grid_m2)


        block.dynamic(ti.indices(self.dim),
                      1024 * 1024,
                      chunk_size=self.leaf_block_size ** self.dim * 8).place(
            self.pid, offset=offset + (0,))
        
        self.particle = ti.root.dynamic(ti.i, max_num_particles, 2 ** 20)
        self.particle.place(self.x)


ti.init(arch=ti.cpu, async_mode=False, debug=True)

mpm = MPMSolver(res=(128, 128))
mpm.grid1.deactivate_all()
ti.sync()
print('successfully finishes')
