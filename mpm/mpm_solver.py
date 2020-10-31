import taichi as ti

@ti.data_oriented
class MPMSolver:
    grid_size = 1024

    def __init__(
            self):
        self.dim = 2

        self.pid = ti.var(ti.i32)
        # position

        indices = ti.ij

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
            blk.dense(indices, self.leaf_block_size).place(c)

        block_component(block, self.grid_m)

        if use2:
            self.grid_m2 = ti.var(dt=ti.f32)

            block_component(block2, self.grid_m2)


        block.dynamic(ti.indices(self.dim),
                      1024 * 1024,
                      chunk_size=self.leaf_block_size ** self.dim * 8).place(
            self.pid)
        

ti.init(arch=ti.cpu, async_mode=False, debug=True)

mpm = MPMSolver()
mpm.grid1.deactivate_all()
ti.sync()
print('successfully finishes')
