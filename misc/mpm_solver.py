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

    grid_size = 4096

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

        if self.dim == 2:
            self.leaf_block_size = 16
        else:
            self.leaf_block_size = 8

        block = self.grid1.pointer(indices,
                                  grid_block_size // self.leaf_block_size)

        def block_component(blk, c):
            blk.dense(indices, self.leaf_block_size).place(c, offset=offset)

        block_component(block, self.grid_m)
        for v in self.grid_v.entries:
            block_component(block, v)

        block.dynamic(ti.indices(self.dim),
                      1024 * 1024,
                      chunk_size=self.leaf_block_size ** self.dim * 8).place(
            self.pid, offset=offset + (0,))

        # grid node momentum/velocity
        self.grid_v2 = ti.Vector(self.dim, dt=ti.f32)
        # grid node mass
        self.grid_m2 = ti.var(dt=ti.f32)

        # self.pid2 = ti.var(ti.i32)

        self.grid2 = ti.root.pointer(indices, self.grid_size // grid_block_size)
        block2 = self.grid2.pointer(indices,
                                    grid_block_size // self.leaf_block_size)
        block_component(block2, self.grid_m2)
        for v in self.grid_v2.entries:
            block_component(block2, v)

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

        if self.dim == 2:
            self.voxelizer = None
            self.set_gravity((0, -9.8))
        else:
            if USE_IN_BLENDER:
                from .voxelizer import Voxelizer
            else:
                from engine.voxelizer import Voxelizer
            self.voxelizer = Voxelizer(res=self.res,
                                       dx=self.dx,
                                       padding=self.padding,
                                       super_sample=voxelizer_super_sample)
            self.set_gravity((0, -9.8, 0))

        self.voxelizer_super_sample = voxelizer_super_sample

        self.grid_postprocess = []

        self.add_bounding_box(self.unbounded)

        self.writers = []

    def stencil_range(self):
        return ti.ndrange(*((3,) * self.dim))

    def set_gravity(self, g):
        assert isinstance(g, (tuple, list))
        assert len(g) == self.dim
        self.gravity[None] = g

    @ti.func
    def sand_projection(self, sigma, p):
        sigma_out = ti.Matrix.zero(ti.f32, self.dim, self.dim)
        epsilon = ti.Vector.zero(ti.f32, self.dim)
        for i in ti.static(range(self.dim)):
            epsilon[i] = ti.log(max(abs(sigma[i, i]), 1e-4))
            sigma_out[i, i] = 1
        tr = epsilon.sum() + self.Jp[p]
        epsilon_hat = epsilon - tr / self.dim
        epsilon_hat_norm = epsilon_hat.norm() + 1e-20
        if tr >= 0.0:
            self.Jp[p] = tr
        else:
            self.Jp[p] = 0.0
            delta_gamma = epsilon_hat_norm + (
                    self.dim * self.lambda_0 +
                    2 * self.mu_0) / (2 * self.mu_0) * tr * self.alpha
            for i in ti.static(range(self.dim)):
                sigma_out[i, i] = ti.exp(epsilon[i] - max(0, delta_gamma) /
                                         epsilon_hat_norm * epsilon_hat[i])

        return sigma_out

    @ti.kernel
    def build_pid(self):
        ti.block_dim(64)
        for p in self.x:
            base = int(ti.floor(self.x[p] * self.inv_dx - 0.5))
            ti.append(self.pid.parent(), base - ti.Vector(list(self.offset)),
                      p)

    @ti.kernel
    def build_pid2(self):
        ti.block_dim(64)
        for p in self.x:
            base = int(ti.floor(self.x[p] * self.inv_dx - 0.5))
            ti.append(self.pid2.parent(), base - ti.Vector(list(self.offset)),
                      p)

    @ti.kernel
    def clear_next_grid(self):
        for I in ti.grouped(self.grid_m2):
            self.grid_m2[I] = 0
            self.grid_v2[I] = [0,] * self.dim


    @ti.kernel
    def clear_this_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0
            self.grid_v[I] = [0,] * self.dim

    @ti.kernel
    def clear_pid(self):
        for I in ti.grouped(self.pid):
            self.pid[I] = 0


    @ti.kernel
    def p2g(self, dt: ti.template()):
        ti.no_activate(self.particle)
        ti.block_dim(256)
        ti.cache_shared(*self.grid_v.entries)
        ti.cache_shared(self.grid_m)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            p = ti.loop_unique(p)
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)

            fx = self.x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2,
                 0.5 * (fx - 0.5) ** 2]
            # deformation gradient update
            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) +
                         dt * self.C[p]) @ self.F[p]
            # Hardening coefficient: snow gets harder when compressed
            h = ti.exp(10 * (1.0 - self.Jp[p]))
            if self.material[
                p] == self.material_elastic:  # jelly, make it softer
                h = 0.3
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.material[p] == self.material_water:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(self.F[p])
            J = 1.0
            if self.material[p] != self.material_sand:
                for d in ti.static(range(self.dim)):
                    new_sig = sig[d, d]
                    if self.material[p] == self.material_snow:  # Snow
                        new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                      1 + 4.5e-3)  # Plasticity
                    self.Jp[p] *= sig[d, d] / new_sig
                    sig[d, d] = new_sig
                    J *= new_sig
            if self.material[p] == self.material_water:
                # Reset deformation gradient to avoid numerical instability
                new_F = ti.Matrix.identity(ti.f32, self.dim)
                new_F[0, 0] = J
                self.F[p] = new_F
            elif self.material[p] == self.material_snow:
                # Reconstruct elastic deformation gradient after plasticity
                self.F[p] = U @ sig @ V.T()

            stress = ti.Matrix.zero(ti.f32, self.dim, self.dim)

            if self.material[p] != self.material_sand:
                stress = 2 * mu * (self.F[p] - U @ V.T()) @ self.F[p].T(
                ) + ti.Matrix.identity(ti.f32, self.dim) * la * J * (J - 1)
            else:
                sig = self.sand_projection(sig, p)
                self.F[p] = U @ sig @ V.T()
                log_sig_sum = 0.0
                center = ti.Matrix.zero(ti.f32, self.dim, self.dim)
                for i in ti.static(range(self.dim)):
                    log_sig_sum += ti.log(sig[i, i])
                    center[i, i] = 2.0 * self.mu_0 * ti.log(
                        sig[i, i]) * (1 / sig[i, i])
                for i in ti.static(range(self.dim)):
                    center[i,
                           i] += self.lambda_0 * log_sig_sum * (1 / sig[i, i])
                stress = U @ center @ V.T() @ self.F[p].T()

            stress = (-dt * self.p_vol * 4 * self.inv_dx ** 2) * stress
            affine = stress + self.p_mass * self.C[p]

            # Loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(float) - fx) * self.dx
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                self.grid_v[base +
                            offset] += weight * (self.p_mass * self.v[p] +
                                                 affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

    @ti.kernel
    def grid_normalization_and_gravity(self, dt: ti.f32):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:  # No need for epsilon here
                self.grid_v[I] = (1 / self.grid_m[I]
                                  ) * self.grid_v[I]  # Momentum to velocity
                self.grid_v[I] += dt * self.gravity[None]

    @ti.kernel
    def grid_bounding_box(self, dt: ti.f32, unbounded: ti.template()):
        for I in ti.grouped(self.grid_m):
            for d in ti.static(range(self.dim)):
                if ti.static(unbounded):
                    if I[d] < -self.grid_size // 2 + self.padding and \
                            self.grid_v[
                                I][d] < 0:
                        self.grid_v[I][d] = 0  # Boundary conditions
                    if I[d] >= self.grid_size // 2 - self.padding and \
                            self.grid_v[
                                I][d] > 0:
                        self.grid_v[I][d] = 0
                else:
                    if I[d] < self.padding and self.grid_v[I][d] < 0:
                        self.grid_v[I][d] = 0  # Boundary conditions
                    if I[d] >= self.res[d] - self.padding and self.grid_v[I][
                        d] > 0:
                        self.grid_v[I][d] = 0

    def add_sphere_collider(self, center, radius, surface=surface_sticky):
        center = list(center)

        @ti.kernel
        def collide(dt: ti.f32):
            for I in ti.grouped(self.grid_m):
                offset = I * self.dx - ti.Vector(center)
                if offset.norm_sqr() < radius * radius:
                    if ti.static(surface == self.surface_sticky):
                        self.grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
                    else:
                        v = self.grid_v[I]
                        normal = offset.normalized(1e-5)
                        normal_component = normal.dot(v)

                        if ti.static(surface == self.surface_slip):
                            # Project out all normal component
                            v = v - normal * normal_component
                        else:
                            # Project out only inward normal component
                            v = v - normal * min(normal_component, 0)

                        self.grid_v[I] = v

        self.grid_postprocess.append(collide)

    def add_surface_collider(self,
                             point,
                             normal,
                             surface=surface_sticky,
                             friction=0.0):
        point = list(point)
        # normalize normal
        normal_scale = 1.0 / math.sqrt(sum(x ** 2 for x in normal))
        normal = list(normal_scale * x for x in normal)

        if surface == self.surface_sticky and friction != 0:
            raise ValueError('friction must be 0 on sticky surfaces.')

        @ti.kernel
        def collide(dt: ti.f32):
            for I in ti.grouped(self.grid_m):
                offset = I * self.dx - ti.Vector(point)
                n = ti.Vector(normal)
                if offset.dot(n) < 0:
                    if ti.static(surface == self.surface_sticky):
                        self.grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
                    else:
                        v = self.grid_v[I]
                        normal_component = n.dot(v)

                        if ti.static(surface == self.surface_slip):
                            # Project out all normal component
                            v = v - n * normal_component
                        else:
                            # Project out only inward normal component
                            v = v - n * min(normal_component, 0)

                        if normal_component < 0 and v.norm() > 1e-30:
                            # apply friction here
                            v = v.normalized() * max(
                                0,
                                v.norm() + normal_component * friction)

                        self.grid_v[I] = v

        self.grid_postprocess.append(collide)

    def add_bounding_box(self, unbounded):
        self.grid_postprocess.append(
            lambda dt: self.grid_bounding_box(dt, unbounded))

    @ti.kernel
    def g2p(self, dt: ti.template()):
        ti.block_dim(256)
        ti.cache_shared(*self.grid_v.entries)
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            p = ti.loop_unique(p)
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], I[D], 0, 1)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2
            ]
            new_v = ti.Vector.zero(ti.f32, self.dim)
            new_C = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            # loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = offset.cast(float) - fx
                g_v = self.grid_v[base + offset]
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.v[p], self.C[p] = new_v, new_C
            self.x[p] += dt * self.v[p]  # advection

    def step(self, frame_dt, print_stat=False):
        begin_t = time.time()
        begin_substep = self.total_substeps

        substeps = int(frame_dt / self.default_dt) + 1

        dt = frame_dt / substeps

        if False:  # before
            for i in range(substeps):
                self.total_substeps += 1

            self.grid1.deactivate_all()
            self.build_pid()
            self.p2g(dt)
            self.grid_normalization_and_gravity(dt)
            for p in self.grid_postprocess:
                p(dt)
            self.g2p(dt)
        else:  # after
            self.grid1.deactivate_all()
            self.grid2.deactivate_all()

            for i in range(substeps):
                self.total_substeps += 1

                self.grid_m, self.grid_m2 = self.grid_m2, self.grid_m
                self.grid_v, self.grid_v2 = self.grid_v2, self.grid_v
                # self.pid, self.pid2 = self.pid2, self.pid
                if i == 0:
                    self.build_pid()

                self.p2g(dt)
                self.grid2.deactivate_all()
                self.pid.snode.deactivate_all()
                self.build_pid()
                # self.build_pid2()
                self.grid_normalization_and_gravity(dt)
                for p in self.grid_postprocess:
                    p(dt)
                self.g2p(dt)

        if print_stat:
            ti.kernel_profiler_print()
            try:
                ti.memory_profiler_print()
            except:
                pass
            print(f'num particles={self.n_particles[None]}')
            print(f'  frame time {time.time() - begin_t:.3f} s')
            print(
                f'  substep time {1000 * (time.time() - begin_t) / (self.total_substeps - begin_substep):.3f} ms'
            )


    @ti.func
    def seed_particle(self, i, x, material, color, velocity):
        self.x[i] = x
        self.v[i] = velocity
        self.F[i] = ti.Matrix.identity(ti.f32, self.dim)
        self.color[i] = color
        self.material[i] = material

        if material == self.material_sand:
            self.Jp[i] = 0
        else:
            self.Jp[i] = 1

    @ti.kernel
    def seed(self, new_particles: ti.i32, new_material: ti.i32, color: ti.i32):
        for i in range(self.n_particles[None],
                       self.n_particles[None] + new_particles):
            self.material[i] = new_material
            x = ti.Vector.zero(ti.f32, self.dim)
            for k in ti.static(range(self.dim)):
                x[k] = self.source_bound[0][k] + ti.random(
                ) * self.source_bound[1][k]
            self.seed_particle(i, x, new_material, color,
                               self.source_velocity[None])

    def set_source_velocity(self, velocity):
        if velocity is not None:
            velocity = list(velocity)
            assert len(velocity) == self.dim
            self.source_velocity[None] = velocity
        else:
            for i in range(self.dim):
                self.source_velocity[None][i] = 0

    def add_cube(self,
                 lower_corner,
                 cube_size,
                 material,
                 color=0xFFFFFF,
                 sample_density=None,
                 velocity=None):
        if sample_density is None:
            sample_density = 2 ** self.dim
        vol = 1
        for i in range(self.dim):
            vol = vol * cube_size[i]
        num_new_particles = int(sample_density * vol / self.dx ** self.dim + 1)

        assert self.n_particles[
                   None] + num_new_particles <= self.max_num_particles

        for i in range(self.dim):
            self.source_bound[0][i] = lower_corner[i]
            self.source_bound[1][i] = cube_size[i]

        self.set_source_velocity(velocity=velocity)

        self.seed(num_new_particles, material, color)
        self.n_particles[None] += num_new_particles

    @ti.func
    def random_point_in_unit_sphere(self):
        ret = ti.Vector.zero(dt=ti.f32, n=self.dim)
        while True:
            for i in ti.static(range(self.dim)):
                ret[i] = ti.random(ti.f32) * 2 - 1
            if ret.norm_sqr() <= 1:
                break
        return ret

    @ti.kernel
    def seed_ellipsoid(self, new_particles: ti.i32, new_material: ti.i32,
                       color: ti.i32):

        for i in range(self.n_particles[None],
                       self.n_particles[None] + new_particles):
            x = self.source_bound[0] + self.random_point_in_unit_sphere(
            ) * self.source_bound[1]
            self.seed_particle(i, x, new_material, color,
                               self.source_velocity[None])

    def add_ellipsoid(self,
                      center,
                      radius,
                      material,
                      color=0xFFFFFF,
                      sample_density=None,
                      velocity=None):
        if sample_density is None:
            sample_density = 2 ** self.dim

        if isinstance(radius, numbers.Number):
            radius = [
                         radius,
                     ] * self.dim

        radius = list(radius)

        if self.dim == 2:
            num_particles = math.pi
        else:
            num_particles = 4 / 3 * math.pi

        for i in range(self.dim):
            num_particles *= radius[i] * self.inv_dx

        num_particles = int(math.ceil(num_particles * sample_density))

        self.source_bound[0] = center
        self.source_bound[1] = radius

        self.set_source_velocity(velocity=velocity)

        assert self.n_particles[None] + num_particles <= self.max_num_particles

        self.seed_ellipsoid(num_particles, material, color)
        self.n_particles[None] += num_particles

    @ti.kernel
    def seed_from_voxels(self, material: ti.i32, color: ti.i32,
                         sample_density: ti.i32):
        for i, j, k in self.voxelizer.voxels:
            inside = 1
            for d in ti.static(range(3)):
                inside = inside and -self.grid_size // 2 + self.padding <= i and i < self.grid_size // 2 - self.padding
            if inside and self.voxelizer.voxels[i, j, k] > 0:
                s = sample_density / self.voxelizer_super_sample ** self.dim
                for l in range(sample_density + 1):
                    if ti.random() + l < s:
                        x = ti.Vector([
                            ti.random() + i,
                            ti.random() + j,
                            ti.random() + k
                        ]) * (self.dx / self.voxelizer_super_sample
                              ) + self.source_bound[0]
                        p = ti.atomic_add(self.n_particles[None], 1)
                        self.seed_particle(p, x, material, color,
                                           self.source_velocity[None])

    def add_mesh(self,
                 triangles,
                 material,
                 color=0xFFFFFF,
                 sample_density=None,
                 velocity=None,
                 translation=None):
        assert self.dim == 3
        if sample_density is None:
            sample_density = 2 ** self.dim

        self.set_source_velocity(velocity=velocity)

        for i in range(self.dim):
            if translation:
                self.source_bound[0][i] = translation[i]
            else:
                self.source_bound[0][i] = 0

        self.voxelizer.voxelize(triangles)
        t = time.time()
        self.seed_from_voxels(material, color, sample_density)
        ti.sync()
        print('Voxelization time:', (time.time() - t) * 1000, 'ms')

    @ti.kernel
    def seed_from_external_array(self, num_particles: ti.i32,
                                 pos: ti.ext_arr(), new_material: ti.i32,
                                 color: ti.i32):

        for i in range(num_particles):
            x = ti.Vector.zero(dt=ti.f32, n=self.dim)
            if ti.static(self.dim == 3):
                x = ti.Vector([pos[i, 0], pos[i, 1], pos[i, 2]])
            else:
                x = ti.Vector([pos[i, 0], pos[i, 1]])
            self.seed_particle(self.n_particles[None] + i, x, new_material,
                               color, self.source_velocity[None])

        self.n_particles[None] += num_particles

    def add_particles(self,
                      particles,
                      material,
                      color=0xFFFFFF,
                      velocity=None):
        self.set_source_velocity(velocity=velocity)
        self.seed_from_external_array(len(particles), particles, material,
                                      color)

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
        np_v = np.ndarray((self.n_particles[None], self.dim), dtype=np.float32)
        self.copy_dynamic_nd(np_v, self.v)
        np_material = np.ndarray((self.n_particles[None],), dtype=np.int32)
        self.copy_dynamic(np_material, self.material)
        np_color = np.ndarray((self.n_particles[None],), dtype=np.int32)
        self.copy_dynamic(np_color, self.color)
        return {
            'position': np_x,
            'velocity': np_v,
            'material': np_material,
            'color': np_color
        }

    @ti.kernel
    def clear_particles(self):
        self.n_particles[None] = 0
        ti.deactivate(self.x.loop_range().parent().snode(), [])

    def dump(self, fn, particles):
        t = time.time()
        output_fn = fn

        np.savez_compressed(output_fn,
                            x=particles['position'],
                            v=particles['velocity'],
                            c=particles['color'])

        print(f'Writing to disk: {time.time() - t:.3f} s')

    def write_particles(self, fn):
        particles = self.particle_info()

        p = mp.Process(target=self.dump, args=(fn, particles))
        p.start()

        self.writers.append(p)

    def flush(self):
        for p in self.writers:
            p.join()
        self.writers = []
