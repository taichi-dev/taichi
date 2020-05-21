# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)

import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

screen_res = (800, 400)
screen_to_world_ratio = 10.0
boundary = (screen_res[0] / screen_to_world_ratio,
            screen_res[1] / screen_to_world_ratio)
cell_size = 2.51
cell_recpr = 1.0 / cell_size


def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s


grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1))

dim = 2
bg_color = 0x112f41
particle_color = 0x068587
boundary_color = 0xebaca2
num_particles_x = 60
num_particles = num_particles_x * 20
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 20.0
epsilon = 1e-5
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# PBF params
h = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.3
corrK = 0.001
# Need ti.pow()
# corrN = 4.0
neighbor_radius = h * 1.05

poly6_factor = 315.0 / 64.0 / np.pi
spiky_grad_factor = -45.0 / np.pi

old_positions = ti.Vector(dim, dt=ti.f32)
positions = ti.Vector(dim, dt=ti.f32)
velocities = ti.Vector(dim, dt=ti.f32)
# Once taichi supports clear(), we can get rid of grid_num_particles
grid_num_particles = ti.var(ti.i32)
grid2particles = ti.var(ti.i32)
particle_num_neighbors = ti.var(ti.i32)
particle_neighbors = ti.var(ti.i32)
lambdas = ti.var(ti.f32)
position_deltas = ti.Vector(dim, dt=ti.f32)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector(2, dt=ti.f32)


@ti.layout
def layout():
    ti.root.dense(ti.i, num_particles).place(old_positions, positions,
                                             velocities)
    grid_snode = ti.root.dense(ti.ij, grid_size)
    grid_snode.place(grid_num_particles)
    grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)
    nb_node = ti.root.dense(ti.i, num_particles)
    nb_node.place(particle_num_neighbors)
    nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
    ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas)
    ti.root.place(board_states)


@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h) / poly6_value(corr_deltaQ_coeff * h, h)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x


@ti.func
def get_cell(pos):
    return (pos * cell_recpr).cast(int)


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[
        1] < grid_size[1]


@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1]
                      ]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p


@ti.kernel
def blit_buffers(f: ti.template(), t: ti.template()):
    for i in f:
        t[i] = f[i]


@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 90
    vel_strength = 8.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b


@ti.kernel
def apply_gravity_within_boundary():
    for i in positions:
        g = ti.Vector([0.0, -9.8])
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)


@ti.kernel
def confine_to_boundary():
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)


@ti.kernel
def update_grid():
    for p_i in positions:
        cell = get_cell(positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = grid_num_particles[cell].atomic_add(1)
        grid2particles[cell, offs] = p_i


@ti.kernel
def find_particle_neighbors():
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i


@ti.kernel
def compute_lambdas():
    # Eq (8) ~ (11)
    for p_i in positions:
        pos_i = positions[p_i]

        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            # TODO: does taichi supports break?
            if p_j >= 0:
                pos_ji = pos_i - positions[p_j]
                grad_j = spiky_gradient(pos_ji, h)
                grad_i += grad_j
                sum_gradient_sqr += grad_j.dot(grad_j)
                # Eq(2)
                density_constraint += poly6_value(pos_ji.norm(), h)

        # Eq(1)
        density_constraint = (mass * density_constraint / rho0) - 1.0

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr +
                                                lambda_epsilon)


@ti.kernel
def compute_position_deltas():
    # Eq(12), (14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.Vector([0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            # TODO: does taichi supports break?
            if p_j >= 0:
                lambda_j = lambdas[p_j]
                pos_ji = pos_i - positions[p_j]
                scorr_ij = compute_scorr(pos_ji)
                pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                    spiky_gradient(pos_ji, h)

        pos_delta_i /= rho0
        position_deltas[p_i] = pos_delta_i


@ti.kernel
def apply_position_deltas():
    for i in positions:
        positions[i] += position_deltas[i]


@ti.kernel
def update_velocities():
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta


def run_pbf():
    blit_buffers(positions, old_positions)
    apply_gravity_within_boundary()

    grid_num_particles.fill(0)
    particle_neighbors.fill(-1)
    update_grid()
    find_particle_neighbors()
    for _ in range(pbf_num_iters):
        compute_lambdas()
        compute_position_deltas()
        apply_position_deltas()

    confine_to_boundary()
    update_velocities()
    # no vorticity/xsph because we cannot do cross product in 2D...


def render(gui):
    canvas = gui.canvas
    canvas.clear(bg_color)
    pos_np = positions.to_numpy()
    for pos in pos_np:
        for j in range(dim):
            pos[j] *= screen_to_world_ratio / screen_res[j]
    gui.circles(pos_np, radius=particle_radius, color=particle_color)
    canvas.rect(ti.vec(
        0, 0), ti.vec(board_states[None][0] / boundary[0],
                      1.0)).radius(1.5).color(boundary_color).close().finish()
    gui.show()


def init_particles():
    np_positions = np.zeros((num_particles, dim), dtype=np.float32)
    delta = h * 0.8
    num_x = num_particles_x
    num_y = num_particles // num_x
    assert num_x * num_y == num_particles
    offs = np.array([(boundary[0] - delta * num_x) * 0.5,
                     (boundary[1] * 0.02)],
                    dtype=np.float32)

    for i in range(num_particles):
        np_positions[i] = np.array([i % num_x, i // num_x]) * delta + offs
    np_velocities = (np.random.rand(num_particles, dim).astype(np.float32) -
                     0.5) * 4.0

    @ti.kernel
    def init(p: ti.ext_arr(), v: ti.ext_arr()):
        for i in range(num_particles):
            for c in ti.static(range(dim)):
                positions[i][c] = p[i, c]
                velocities[i][c] = v[i, c]

    @ti.kernel
    def init2():
        board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])

    init(np_positions, np_velocities)
    init2()


def print_stats():
    print('PBF stats:')
    num = grid_num_particles.to_numpy()
    avg, max = np.mean(num), np.max(num)
    print(f'  #particles per cell: avg={avg:.2f} max={max}')
    num = particle_num_neighbors.to_numpy()
    avg, max = np.mean(num), np.max(num)
    print(f'  #neighbors per particle: avg={avg:.2f} max={max}')


def main():
    init_particles()
    print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')
    gui = ti.GUI('PBF2D', screen_res)
    print_counter = 0
    while True:
        move_board()
        run_pbf()
        print_counter += 1
        if print_counter == 20:
            print_stats()
            print_counter = 0
        render(gui)


if __name__ == '__main__':
    main()
