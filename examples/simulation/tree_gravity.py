# N-body gravity simulation in 300 lines of Taichi, tree method, no multipole, O(N log N)
# Author: archibate <1931127624@qq.com>, all left reserved
from math_utils import (clamp, rand_disk_2d, rand_unit_3d, rand_vector,
                        reflect_boundary)

import taichi as ti

ti.init(arch=ti.cpu)

kUseTree = True
#kDisplay = 'tree mouse pixels cmap save_result'
kDisplay = 'pixels'
kResolution = 512
kShapeFactor = 1
kMaxParticles = 8192
kMaxDepth = kMaxParticles * 1
kMaxNodes = kMaxParticles * 4
kDim = 2

dt = 0.00005
LEAF = -1
TREE = -2

particle_mass = ti.field(ti.f32)
particle_pos = ti.Vector.field(kDim, ti.f32)
particle_vel = ti.Vector.field(kDim, ti.f32)
particle_table = ti.root.dense(ti.i, kMaxParticles)
particle_table.place(particle_pos).place(particle_vel).place(particle_mass)
particle_table_len = ti.field(ti.i32, ())

if kUseTree:
    trash_particle_id = ti.field(ti.i32)
    trash_base_parent = ti.field(ti.i32)
    trash_base_geo_center = ti.Vector.field(kDim, ti.f32)
    trash_base_geo_size = ti.field(ti.f32)
    trash_table = ti.root.dense(ti.i, kMaxDepth)
    trash_table.place(trash_particle_id)
    trash_table.place(trash_base_parent, trash_base_geo_size)
    trash_table.place(trash_base_geo_center)
    trash_table_len = ti.field(ti.i32, ())

    node_mass = ti.field(ti.f32)
    node_weighted_pos = ti.Vector.field(kDim, ti.f32)
    node_particle_id = ti.field(ti.i32)
    node_children = ti.field(ti.i32)
    node_table = ti.root.dense(ti.i, kMaxNodes)
    node_table.place(node_mass, node_particle_id, node_weighted_pos)
    node_table.dense(ti.indices(*list(range(1, 1 + kDim))),
                     2).place(node_children)
    node_table_len = ti.field(ti.i32, ())

if 'mouse' in kDisplay:
    display_image = ti.Vector.field(3, ti.f32, (kResolution, kResolution))
elif len(kDisplay):
    display_image = ti.field(ti.f32, (kResolution, kResolution))


@ti.func
def alloc_node():
    ret = ti.atomic_add(node_table_len[None], 1)
    assert ret < kMaxNodes
    node_mass[ret] = 0
    node_weighted_pos[ret] = particle_pos[0] * 0
    node_particle_id[ret] = LEAF
    for which in ti.grouped(ti.ndrange(*([2] * kDim))):
        node_children[ret, which] = LEAF
    return ret


@ti.func
def alloc_particle():
    ret = ti.atomic_add(particle_table_len[None], 1)
    assert ret < kMaxParticles
    particle_mass[ret] = 0
    particle_pos[ret] = particle_pos[0] * 0
    particle_vel[ret] = particle_pos[0] * 0
    return ret


@ti.func
def alloc_trash():
    ret = ti.atomic_add(trash_table_len[None], 1)
    assert ret < kMaxDepth
    return ret


@ti.func
def alloc_a_node_for_particle(particle_id, parent, parent_geo_center,
                              parent_geo_size):
    position = particle_pos[particle_id]
    mass = particle_mass[particle_id]

    depth = 0
    while depth < kMaxDepth:
        already_particle_id = node_particle_id[parent]
        if already_particle_id == LEAF:
            break
        if already_particle_id != TREE:
            node_particle_id[parent] = TREE
            trash_id = alloc_trash()
            trash_particle_id[trash_id] = already_particle_id
            trash_base_parent[trash_id] = parent
            trash_base_geo_center[trash_id] = parent_geo_center
            trash_base_geo_size[trash_id] = parent_geo_size
            already_pos = particle_pos[already_particle_id]
            already_mass = particle_mass[already_particle_id]
            node_weighted_pos[parent] -= already_pos * already_mass
            node_mass[parent] -= already_mass

        node_weighted_pos[parent] += position * mass
        node_mass[parent] += mass

        which = abs(position > parent_geo_center)
        child = node_children[parent, which]
        if child == LEAF:
            child = alloc_node()
            node_children[parent, which] = child
        child_geo_size = parent_geo_size * 0.5
        child_geo_center = parent_geo_center + (which - 0.5) * child_geo_size

        parent_geo_center = child_geo_center
        parent_geo_size = child_geo_size
        parent = child

        depth = depth + 1

    node_particle_id[parent] = particle_id
    node_weighted_pos[parent] = position * mass
    node_mass[parent] = mass


@ti.kernel
def add_particle_at(mx: ti.f32, my: ti.f32, mass: ti.f32):
    mouse_pos = ti.Vector([mx, my]) + rand_vector(2) * (0.05 / kResolution)

    particle_id = alloc_particle()
    if ti.static(kDim == 2):
        particle_pos[particle_id] = mouse_pos
    else:
        particle_pos[particle_id] = ti.Vector(
            [mouse_pos[0], mouse_pos[1], 0.0])
    particle_mass[particle_id] = mass


@ti.kernel
def add_random_particles(angular_velocity: ti.f32):
    num = ti.static(1)
    particle_id = alloc_particle()
    if ti.static(kDim == 2):
        particle_pos[particle_id] = rand_disk_2d() * 0.2 + 0.5
        velocity = (particle_pos[particle_id] - 0.5) * angular_velocity * 250
        particle_vel[particle_id] = ti.Vector([-velocity.y, velocity.x])
    else:
        particle_pos[particle_id] = rand_unit_3d() * 0.2 + 0.5
        velocity = (ti.Vector(
            [particle_pos[particle_id].x, particle_pos[particle_id].y]) -
                    0.5) * angular_velocity * 180
        particle_vel[particle_id] = ti.Vector([-velocity.y, velocity.x, 0.0])
    particle_mass[particle_id] = 1.5 * ti.random()


@ti.kernel
def build_tree():
    node_table_len[None] = 0
    trash_table_len[None] = 0
    alloc_node()

    particle_id = 0
    while particle_id < particle_table_len[None]:
        alloc_a_node_for_particle(particle_id, 0, particle_pos[0] * 0 + 0.5,
                                  1.0)

        trash_id = 0
        while trash_id < trash_table_len[None]:
            alloc_a_node_for_particle(trash_particle_id[trash_id],
                                      trash_base_parent[trash_id],
                                      trash_base_geo_center[trash_id],
                                      trash_base_geo_size[trash_id])
            trash_id = trash_id + 1

        trash_table_len[None] = 0
        particle_id = particle_id + 1


@ti.func
def gravity_func(distance):
    norm_sqr = distance.norm_sqr() + 1e-3
    return distance / norm_sqr**(3 / 2)


@ti.func
def get_tree_gravity_at(position):
    acc = particle_pos[0] * 0

    trash_table_len[None] = 0
    trash_id = alloc_trash()
    assert trash_id == 0
    trash_base_parent[trash_id] = 0
    trash_base_geo_size[trash_id] = 1.0

    trash_id = 0
    while trash_id < trash_table_len[None]:
        parent = trash_base_parent[trash_id]
        parent_geo_size = trash_base_geo_size[trash_id]

        particle_id = node_particle_id[parent]
        if particle_id >= 0:
            distance = particle_pos[particle_id] - position
            acc += particle_mass[particle_id] * gravity_func(distance)

        else:  # TREE or LEAF
            for which in ti.grouped(ti.ndrange(*([2] * kDim))):
                child = node_children[parent, which]
                if child == LEAF:
                    continue
                node_center = node_weighted_pos[child] / node_mass[child]
                distance = node_center - position
                if distance.norm_sqr() > kShapeFactor**2 * parent_geo_size**2:
                    acc += node_mass[child] * gravity_func(distance)
                else:
                    new_trash_id = alloc_trash()
                    child_geo_size = parent_geo_size * 0.5
                    trash_base_parent[new_trash_id] = child
                    trash_base_geo_size[new_trash_id] = child_geo_size

        trash_id = trash_id + 1

    return acc


@ti.func
def get_raw_gravity_at(pos):
    acc = particle_pos[0] * 0
    for i in range(particle_table_len[None]):
        acc += particle_mass[i] * gravity_func(particle_pos[i] - pos)
    return acc


@ti.kernel
def substep_raw():
    for i in range(particle_table_len[None]):
        acceleration = get_raw_gravity_at(particle_pos[i])
        particle_vel[i] += acceleration * dt
    for i in range(particle_table_len[None]):
        particle_pos[i] += particle_vel[i] * dt


@ti.kernel
def substep_tree():
    particle_id = 0
    while particle_id < particle_table_len[None]:
        acceleration = get_tree_gravity_at(particle_pos[particle_id])
        particle_vel[particle_id] += acceleration * dt
        # well... seems our tree inserter will break if a particle is out-of-bound:
        particle_vel[particle_id] = reflect_boundary(particle_pos[particle_id],
                                                     particle_vel[particle_id],
                                                     0, 1)
        particle_id = particle_id + 1
    for i in range(particle_table_len[None]):
        particle_pos[i] += particle_vel[i] * dt


@ti.kernel
def render_pixels():
    for i in range(particle_table_len[None]):
        position = ti.Vector([particle_pos[i].x, particle_pos[i].y])
        pix = int(position * kResolution)
        display_image[clamp(pix, 0, kResolution - 1)] += 0.3


def render_tree(gui,
                parent=0,
                parent_geo_center=ti.Vector([0.5, 0.5]),
                parent_geo_size=1.0):
    child_geo_size = parent_geo_size * 0.5
    if node_particle_id[parent] >= 0:
        tl = parent_geo_center - child_geo_size
        br = parent_geo_center + child_geo_size
        gui.rect(tl, br, radius=1, color=0xff0000)
    for which in map(ti.Vector, [[0, 0], [0, 1], [1, 0], [1, 1]]):
        child = node_children[(parent, which[0], which[1])]
        if child < 0:
            continue
        a = parent_geo_center + (which - 1) * child_geo_size
        b = parent_geo_center + which * child_geo_size
        child_geo_center = parent_geo_center + (which - 0.5) * child_geo_size
        gui.rect(a, b, radius=1, color=0xff0000)
        render_tree(gui, child, child_geo_center, child_geo_size)


if 'cmap' in kDisplay:
    import matplotlib.cm as cm
    cmap = cm.get_cmap('magma')

print('[Hint] Press `r` to add 512 random particles')
print('[Hint] Press `t` to add 512 random particles with angular velocity')
print('[Hint] Drag with mouse left button to add a series of particles')
print('[Hint] Drag with mouse middle button to add zero-mass particles')
print('[Hint] Click mouse right button to add a single particle')
gui = ti.GUI('Tree-code', kResolution)
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.RMB:
            add_particle_at(*gui.get_cursor_pos(), 1.0)
        elif e.key in 'rt':
            if particle_table_len[None] + 512 < kMaxParticles:
                for i in range(512):
                    add_random_particles(e.key == 't')
    if gui.is_pressed(gui.MMB, gui.LMB):
        add_particle_at(*gui.get_cursor_pos(), gui.is_pressed(gui.LMB))

    if kUseTree:
        build_tree()
        substep_tree()
    else:
        substep_raw()
    if len(kDisplay) and 'trace' not in kDisplay:
        display_image.fill(0)
    if 'pixels' in kDisplay:
        render_pixels()
    if 'cmap' in kDisplay:
        gui.set_image(cmap(display_image.to_numpy()))
    elif len(kDisplay):
        gui.set_image(display_image)
    if 'tree' in kDisplay:
        render_tree(gui)
    if 'pixels' not in kDisplay:
        gui.circles(particle_pos.to_numpy()[:particle_table_len[None]])
    if 'save_result' in kDisplay:
        gui.show(f'{gui.frame:06d}.png')
    else:
        gui.show()
