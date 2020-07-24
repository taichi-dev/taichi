import taichi as ti
import numpy as np
import random


@ti.data_oriented
class MPMSolver:
    material_water = 0
    material_elastic = 1
    material_snow = 2
    material_sand = 3
    materials = {
        'WATER': material_water,
        'ELASTIC': material_elastic,
        'SNOW': material_snow,
        'SAND': material_sand
    }
    
    # Surface boundary conditions
    
    # Stick to the boundary
    surface_sticky = 0
    # Slippy boundary
    surface_slip = 1
    # Slippy and free to separate
    surface_separate = 2
    
    surfaces = {
        'STICKY': surface_sticky,
        'SLIP': surface_slip,
        'SEPARATE': surface_separate
    }
    
    grid_size = 4096
    
    def __init__(
            self,
            res,
            size=1,
            max_num_particles=2**27):
        self.dim = len(res)
        assert self.dim in (
            2, 3), "MPM solver supports only 2D and 3D simulations."
        
        self.res = res
        self.n_particles = ti.var(ti.i32, shape=())
        self.dx = size / res[0]
        self.inv_dx = 1.0 / self.dx
        self.max_num_particles = max_num_particles
        # position
        self.x = ti.Vector(self.dim, dt=ti.f32)
        
        if self.dim == 2:
            indices = ti.ij
        else:
            indices = ti.ijk
        
        # grid node momentum/velocity
        self.grid_v = ti.Vector(self.dim, dt=ti.f32)
        # grid node mass
        self.grid_m = ti.var(dt=ti.f32)
        
        grid_block_size = 128
        self.grid = ti.root.pointer(indices, self.grid_size // grid_block_size)
        
        self.leaf_block_size = 16
        
        block = self.grid.pointer(indices,
                                  grid_block_size // self.leaf_block_size)
        
        def block_component(c):
            block.dense(indices, self.leaf_block_size).place(c)
        
        block_component(self.grid_m)
        
        self.particle = ti.root.dynamic(ti.i, max_num_particles, 2**20)
        self.particle.place(self.x)
    
    @ti.kernel
    def build_pid(self):
        ti.block_dim(64)
        for p in self.x:
            base = int(ti.floor(self.x[p] * self.inv_dx - 0.5))
            self.grid_m[base] += 1


    @ti.kernel
    def move(self):
        for p in self.x:
            self.x[p] += ti.Vector([0.0, 0.001])

    def step(self):
        self.grid.deactivate_all()
        self.build_pid()
        self.move()
        
solver = MPMSolver(res=(128, 128))

for i in range(1000):
    solver.x[i] = [random.random() * 0.1 + 0.5, random.random() * 0.1 + 0.5]

while True:
    solver.step()
    ti.memory_profiler_print()
