import taichi as ti
import numpy as np
import time
import numbers
import math
import multiprocessing as mp

USE_IN_BLENDER = False

ti.require_version(0, 6, 22)


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

    grid_size = 1024

    def __init__(
            self,
            res,
            size=1,
            max_num_particles=2 ** 27,
            # Max 128 MB particles
            padding=3,
            unbounded=False,
            dt_scale=1,
            E_scale=1,
            voxelizer_super_sample=2):
        self.dim = len(res)
        assert self.dim in (
            2, 3), "MPM solver supports only 2D and 3D simulations."

        self.res = res
        self.n_particles = ti.var(ti.i32, shape=())
        self.dx = size / res[0]
        self.inv_dx = 1.0 / self.dx
        self.default_dt = 2e-2 * self.dx / size * dt_scale
        self.p_vol = self.dx ** self.dim
        self.p_rho = 1000
        self.p_mass = self.p_vol * self.p_rho
        self.max_num_particles = max_num_particles
        self.gravity = ti.Vector(self.dim, dt=ti.f32, shape=())
        self.source_bound = ti.Vector(self.dim, dt=ti.f32, shape=2)
        self.source_velocity = ti.Vector(self.dim, dt=ti.f32, shape=())
        self.pid = ti.var(ti.i32)
        # position
        self.x = ti.Vector(self.dim, dt=ti.f32)
        # velocity
        self.v = ti.Vector(self.dim, dt=ti.f32)
        # affine velocity field
        self.C = ti.Matrix(self.dim, self.dim, dt=ti.f32)
        # deformation gradient
        self.F = ti.Matrix(self.dim, self.dim, dt=ti.f32)
        # material id
        self.material = ti.var(dt=ti.i32)
        self.color = ti.var(dt=ti.i32)
        # plastic deformation volume ratio
        self.Jp = ti.var(dt=ti.f32)

        if self.dim == 2:
            indices = ti.ij
        else:
            indices = ti.ijk

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

        if self.dim == 2:
            self.leaf_block_size = 16
        else:
            self.leaf_block_size = 8

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
            # grid node momentum/velocity
            # self.grid_v2 = ti.Vector(self.dim, dt=ti.f32)
            # grid node mass
            self.grid_m2 = ti.var(dt=ti.f32)

            # self.pid2 = ti.var(ti.i32)


            block_component(block2, self.grid_m2)
            # for v in self.grid_v2.entries:
            #     block_component(block2, v)


        block.dynamic(ti.indices(self.dim),
                      1024 * 1024,
                      chunk_size=self.leaf_block_size ** self.dim * 8).place(
            self.pid, offset=offset + (0,))
        if use2:
            pass
            # block2.dynamic(ti.indices(self.dim),
            #               1024 * 1024,
            #               chunk_size=self.leaf_block_size ** self.dim * 8).place(
            #     self.pid2, offset=offset + (0,))

        self.padding = padding

        # Young's modulus and Poisson's ratio
        self.E, self.nu = 1e6 * size * E_scale, 0.2
        # Lame parameters
        self.mu_0, self.lambda_0 = self.E / (
                2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) *
                                                        (1 - 2 * self.nu))

        # Sand parameters
        friction_angle = math.radians(45)
        sin_phi = math.sin(friction_angle)
        self.alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

        self.particle = ti.root.dynamic(ti.i, max_num_particles, 2 ** 20)
        self.particle.place(self.x, self.v, self.C, self.F, self.material,
                            self.color, self.Jp)

        self.total_substeps = 0
        self.unbounded = unbounded

        self.writers = []


    def step(self, frame_dt, print_stat=False):
        print(111)
        self.grid1.deactivate_all()
        ti.sync()
        print('successfully finishes')
        exit()
