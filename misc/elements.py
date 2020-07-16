import taichi as ti
import random
import numpy as np


@ti.data_oriented
class MPMSolver:
    
    def __init__(self,
                 res,
                 size=1,
                 max_num_particles=2**25):
        self.dim = len(res)
        assert self.dim in (
            2, 3), "MPM solver supports only 2D and 3D simulations."

        self.res = res
        self.n_particles = ti.var(ti.i32, shape=())
        self.dx = size / res[0]
        self.inv_dx = 1.0 / self.dx
        self.max_num_particles = max_num_particles
        self.gravity = ti.Vector(self.dim, dt=ti.f32, shape=())
        self.source_bound = ti.Vector(self.dim, dt=ti.f32, shape=2)
        self.source_velocity = ti.Vector(self.dim, dt=ti.f32, shape=())
        self.pid = ti.var(ti.i32)
        # position
        self.x = ti.Vector(self.dim, dt=ti.f32)
        self.grid_m = ti.var(dt=ti.f32)
        
        indices = ti.ij
        
        offset = tuple(-2048 for _ in range(self.dim))
        self.offset = offset

        # grid node momentum/velocity
        self.grid = ti.root.pointer(indices, 32)
        block = self.grid.pointer(indices, 16)
        voxel = block.dense(indices, 8)
        
        voxel.place(self.grid_m, offset=offset)
        block.dynamic(ti.indices(self.dim), 1024 * 1024, chunk_size=4096).place(self.pid, offset=offset + (0,))

        ti.root.dynamic(ti.i, max_num_particles, 2**20).place(self.x)
        self.substeps = 0
        
        self.n_particles[None] = 10000
        for i in range(10000):
            self.x[i] = [random.random() * 0.5, random.random() * 0.5]
        
        

    @ti.kernel
    def build_pid(self):
        ti.block_dim(256)
        for p in self.x:
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            ti.append(self.pid.parent(), base - ti.Vector(list(self.offset)), p)

    def step(self):
        for i in range(1000000):
            self.substeps += 1
            print(self.substeps)
            self.grid.deactivate_all()
            self.build_pid()

    @ti.func
    def seed_particle(self, i, x):
        self.x[i] = x

    @ti.kernel
    def seed(self, new_particles: ti.i32):
        for i in range(self.n_particles[None],
                       self.n_particles[None] + new_particles):
            x = ti.Vector.zero(ti.f32, self.dim)
            for k in ti.static(range(self.dim)):
                x[k] = self.source_bound[0][k] + ti.random(
                ) * self.source_bound[1][k]
            self.seed_particle(i, x)
            
    def add_cube(self,
                 lower_corner,
                 cube_size):
        sample_density = 2**self.dim
        vol = 1
        for i in range(self.dim):
            vol = vol * cube_size[i]
        num_new_particles = int(sample_density * vol / self.dx**self.dim + 1)
        assert self.n_particles[
            None] + num_new_particles <= self.max_num_particles

        for i in range(self.dim):
            self.source_bound[0][i] = lower_corner[i]
            self.source_bound[1][i] = cube_size[i]

        self.seed(num_new_particles)
        self.n_particles[None] += num_new_particles

    @ti.kernel
    def copy_dynamic_nd(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in self.x:
            for j in ti.static(range(self.dim)):
                np_x[i, j] = input_x[i][j]

    @ti.kernel
    def copy_dynamic(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in self.x:
            np_x[i] = input_x[i]

    def particle_info(self):
        np_x = np.ndarray((self.n_particles[None], self.dim), dtype=np.float32)
        self.copy_dynamic_nd(np_x, self.x)
        return {
            'position': np_x,
        }
    

write_to_disk = False

ti.init(arch=ti.cuda, use_unified_memory=False, kernel_profiler=True, device_memory_GB=0.5, debug=True)  # Try to run on GPU

gui = ti.GUI("Taichi MLS-MPM", res=512, background_color=0x112F41)

mpm = MPMSolver(res=(128, 128))

for i in range(5):
    mpm.add_cube(lower_corner=[0.2 + i * 0.1, 0.3 + i * 0.1],
                 cube_size=[0.1, 0.1])

for frame in range(500):
    mpm.step()
    particles = mpm.particle_info()
    pos = particles['position'] * 0.4 + 0.3
    gui.circles(pos)
    gui.show(f'{frame:06d}.png' if write_to_disk else None)
    ti.kernel_profiler_print()
