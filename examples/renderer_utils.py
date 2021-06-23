import math

import taichi as ti

eps = 1e-4
inf = 1e10


@ti.func
def out_dir(n):
    u = ti.Vector([1.0, 0.0, 0.0])
    if ti.abs(n[1]) < 1 - 1e-3:
        u = n.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
    v = n.cross(u)
    phi = 2 * math.pi * ti.random(ti.f32)
    r = ti.random(ti.f32)
    ay = ti.sqrt(r)
    ax = ti.sqrt(1 - r)
    return ax * (ti.cos(phi) * u + ti.sin(phi) * v) + ay * n


@ti.func
def reflect(d, n):
    # Assuming |d| and |n| are normalized
    return d - 2.0 * d.dot(n) * n


@ti.func
def refract(d, n, ni_over_nt):
    # Assuming |d| and |n| are normalized
    has_r, rd = 0, d
    dt = d.dot(n)
    discr = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt)
    if discr > 0.0:
        has_r = 1
        rd = (ni_over_nt * (d - n * dt) - n * ti.sqrt(discr)).normalized()
    else:
        rd *= 0.0
    return has_r, rd


@ti.func
def ray_aabb_intersection(box_min, box_max, o, d):
    intersect = 1

    near_int = -inf
    far_int = inf

    for i in ti.static(range(3)):
        if d[i] == 0:
            if o[i] < box_min[i] or o[i] > box_max[i]:
                intersect = 0
        else:
            i1 = (box_min[i] - o[i]) / d[i]
            i2 = (box_max[i] - o[i]) / d[i]

            new_far_int = ti.max(i1, i2)
            new_near_int = ti.min(i1, i2)

            far_int = ti.min(new_far_int, far_int)
            near_int = ti.max(new_near_int, near_int)

    if near_int > far_int:
        intersect = 0
    return intersect, near_int, far_int


# (T + x d)(T + x d) = r * r
# T*T + 2Td x + x^2 = r * r
# x^2 + 2Td x + (T * T - r * r) = 0

refine = True


@ti.func
def intersect_sphere(pos, d, center, radius):
    T = pos - center
    A = 1.0
    B = 2.0 * T.dot(d)
    C = T.dot(T) - radius * radius
    delta = B * B - 4.0 * A * C
    dist = inf
    hit_pos = ti.Vector([0.0, 0.0, 0.0])

    if delta > -1e-4:
        delta = ti.max(delta, 0)
        sdelta = ti.sqrt(delta)
        ratio = 0.5 / A
        ret1 = ratio * (-B - sdelta)
        dist = ret1
        if ti.static(refine):
            if dist < inf:
                # refinement
                old_dist = dist
                new_pos = pos + d * dist
                T = new_pos - center
                A = 1.0
                B = 2.0 * T.dot(d)
                C = T.dot(T) - radius * radius
                delta = B * B - 4 * A * C
                if delta > 0:
                    sdelta = ti.sqrt(delta)
                    ratio = 0.5 / A
                    ret1 = ratio * (-B - sdelta) + old_dist
                    if ret1 > 0:
                        dist = ret1
                        hit_pos = new_pos + ratio * (-B - sdelta) * d
                    else:
                        pass
                        #ret2 = ratio * (-B + sdelta) + old_dist
                        #if ret2 > 0:
                        #  dist = ret2
                        #  hit_pos = new_pos + ratio * (-B + sdelta) * d
                else:
                    dist = inf

    return dist, hit_pos


@ti.func
def ray_plane_intersect(pos, d, pt_on_plane, norm):
    dist = inf
    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    denom = d.dot(norm)
    if abs(denom) > eps:
        dist = norm.dot(pt_on_plane - pos) / denom
        hit_pos = pos + d * dist
    return dist, hit_pos


@ti.func
def point_aabb_distance2(box_min, box_max, o):
    p = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        p[i] = ti.max(ti.min(o[i], box_max[i]), box_min[i])
    return (p - o).norm_sqr()


@ti.func
def sphere_aabb_intersect(box_min, box_max, o, radius):
    return point_aabb_distance2(box_min, box_max, o) < radius * radius


@ti.func
def sphere_aabb_intersect_motion(box_min, box_max, o1, o2, radius):
    lo = 0.0
    hi = 1.0
    while lo + 1e-5 < hi:
        m1 = 2 * lo / 3 + hi / 3
        m2 = lo / 3 + 2 * hi / 3
        d1 = point_aabb_distance2(box_min, box_max, (1 - m1) * o1 + m1 * o2)
        d2 = point_aabb_distance2(box_min, box_max, (1 - m2) * o1 + m2 * o2)
        if d2 > d1:
            hi = m2
        else:
            lo = m1

    return point_aabb_distance2(box_min, box_max,
                                (1 - lo) * o1 + lo * o2) < radius * radius


@ti.func
def inside(p, c, r):
    return (p - c).norm_sqr() <= r * r


@ti.func
def inside_left(p, c, r):
    return inside(p, c, r) and p[0] < c[0]


@ti.func
def inside_right(p, c, r):
    return inside(p, c, r) and p[0] > c[0]


def Vector2(x, y):
    return ti.Vector([x, y])


inside_taichi = ti.taichi_logo
