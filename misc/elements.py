import taichi as ti
import random

@ti.data_oriented
class MPMSolver:
    def __init__(self,
                 res,
                 size=1,
                 max_num_particles=2**25):
        dim = len(res)
        self.dx = size / res[0]
        self.inv_dx = 1.0 / self.dx
        self.pid = ti.var(ti.i32)
        self.x = ti.Vector(dim, dt=ti.f32)
        self.grid_m = ti.var(dt=ti.f32)
        
        indices = ti.ij
        
        self.grid = ti.root.pointer(indices, 32)
        block = self.grid.pointer(indices, 16)
        voxel = block.dense(indices, 8)
        
        voxel.place(self.grid_m)
        block.dynamic(ti.indices(dim), 1024 * 1024, chunk_size=4096).place(self.pid)

        ti.root.dynamic(ti.i, max_num_particles, 2**20).place(self.x)
        self.substeps = 0
        
        for i in range(10000):
            self.x[i] = [random.random() * 0.5, random.random() * 0.5]
            
        self.step()

    @ti.kernel
    def build_pid(self):
        ti.block_dim(256)
        for p in self.x:
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            ti.append(self.pid.parent(), base, p)

    def step(self):
        for i in range(1000):
            self.substeps += 1
            self.grid.deactivate_all()
            self.build_pid()
            print(self.substeps)

ti.init(arch=ti.gpu, use_unified_memory=False, kernel_profiler=True, device_memory_GB=0.5)

mpm = MPMSolver(res=(128, 128))
