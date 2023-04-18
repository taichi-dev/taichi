"""
Marching squares algorithm in Taichi.
See "https://en.wikipedia.org/wiki/Marching_squares"
"""
import time

import numpy as np

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cpu)

W, H = 800, 600
resolution = (W, H)
grid_size = 8
level = 0.15  # Draw contour of isofunc=level
pixels = ti.Vector.field(3, float, shape=resolution)

# Cases 0-15 in the wikipedia page
_edges_np = np.array(
    [
        [[-1, -1], [-1, -1]],
        [[3, 0], [-1, -1]],
        [[0, 1], [-1, -1]],
        [[1, 3], [-1, -1]],
        [[1, 2], [-1, -1]],
        [[0, 1], [2, 3]],
        [[0, 2], [-1, -1]],
        [[2, 3], [-1, -1]],
        [[2, 3], [-1, -1]],
        [[0, 2], [-1, -1]],
        [[0, 3], [1, 2]],
        [[1, 2], [-1, -1]],
        [[1, 3], [-1, -1]],
        [[0, 1], [-1, -1]],
        [[3, 0], [-1, -1]],
        [[-1, -1], [-1, -1]],
    ],
    dtype=np.int32,
)
edge_table = ti.Matrix.field(2, 2, int, 16)
edge_table.from_numpy(_edges_np)

Edge = ti.types.struct(p0=tm.vec2, p1=tm.vec2)
edges = Edge.field()
ti.root.dynamic(ti.i, 1024, chunk_size=32).place(edges)
iTime = ti.field(float, shape=())


@ti.func
def hash22(p):
    n = tm.sin(tm.dot(p, tm.vec2(41, 289)))
    p = tm.fract(tm.vec2(262144, 32768) * n)
    return tm.sin(p * 6.28 + iTime[None])


@ti.func
def noise(p):
    ip = tm.floor(p)
    p -= ip
    v = tm.vec4(
        tm.dot(hash22(ip), p),
        tm.dot(hash22(ip + tm.vec2(1, 0)), p - tm.vec2(1, 0)),
        tm.dot(hash22(ip + tm.vec2(0, 1)), p - tm.vec2(0, 1)),
        tm.dot(hash22(ip + tm.vec2(1, 1)), p - tm.vec2(1, 1)),
    )
    p = p * p * p * (p * (p * 6 - 15) + 10)
    return tm.mix(tm.mix(v.x, v.y, p.x), tm.mix(v.z, v.w, p.x), p.y)


@ti.func
def isofunc(p):
    return noise(p / 4 + 1)


@ti.func
def interp(p1: tm.vec2, p2: tm.vec2, v1: float, v2: float, isovalue: float) -> tm.vec2:
    return tm.mix(p1, p2, (isovalue - v1) / (v2 - v1))


@ti.func
def get_vertex(vertex_id, values, isovalue):
    v = tm.vec2(0)
    square = [tm.vec2(0), tm.vec2(1, 0), tm.vec2(1, 1), tm.vec2(0, 1)]
    if vertex_id == 0:
        v = interp(square[0], square[1], values.x, values.y, isovalue)
    elif vertex_id == 1:
        v = interp(square[1], square[2], values.y, values.z, isovalue)
    elif vertex_id == 2:
        v = interp(square[2], square[3], values.z, values.w, isovalue)
    else:
        v = interp(square[3], square[0], values.w, values.x, isovalue)
    return v


@ti.kernel
def march_squares() -> int:
    edges.deactivate()
    x_range = tm.ceil(grid_size * W / H, int)
    y_range = grid_size
    for i, j in ti.ndrange((-x_range, x_range + 1), (-y_range, y_range + 1)):
        case_id = 0
        values = tm.vec4(
            isofunc(tm.vec2(i, j)),
            isofunc(tm.vec2(i + 1, j)),
            isofunc(tm.vec2(i + 1, j + 1)),
            isofunc(tm.vec2(i, j + 1)),
        )
        if values.x > level:
            case_id |= 1
        if values.y > level:
            case_id |= 2
        if values.z > level:
            case_id |= 4
        if values.w > level:
            case_id |= 8

        # Fix the ambiguity for case 5 and 10
        if case_id == 5 or case_id == 10:
            center = isofunc(tm.vec2(i + 0.5, j + 0.5))
            if center < level:  # A valley, switch the case_id
                case_id = 15 - case_id

        for k in ti.static(range(2)):
            if edge_table[case_id][k, 0] != -1:
                ind1 = edge_table[case_id][k, 0]
                ind2 = edge_table[case_id][k, 1]
                p0 = tm.vec2(i, j) + get_vertex(ind1, values, level)
                p1 = tm.vec2(i, j) + get_vertex(ind2, values, level)
                edges.append(Edge(p0, p1))

    return edges.length()


@ti.func
def dseg(p, a, b):
    p -= a
    b -= a
    h = tm.clamp(tm.dot(p, b) / tm.dot(b, b), 0, 1)
    return tm.length(p - h * b)


@ti.kernel
def render():
    for i, j in pixels:
        p = (2 * tm.vec2(i, j) - tm.vec2(resolution)) / H
        p *= grid_size
        p.y += 1.2

        q = tm.fract(p) - 0.5
        d2 = 0.5 - abs(q)
        dgrid = ti.min(d2.x, d2.y)
        dedge = 1e5
        dv = 1e5
        for k in range(edges.length()):
            s, e = edges[k].p0, edges[k].p1
            dedge = ti.min(dedge, dseg(p, s, e))
            dv = ti.min(dv, tm.length(p - s), tm.length(p - e))

        col = tm.vec3(0.3, 0.6, 0.8)
        if isofunc(p) > level:
            col = tm.vec3(1, 0.8, 0.3)

        # Draw background grid
        col = tm.mix(col, tm.vec3(0), 1 - tm.smoothstep(0, 0.04, dgrid - 0.02))
        # Draw edges
        col = tm.mix(col, tm.vec3(1, 0, 0), 1 - tm.smoothstep(0, 0.05, dedge - 0.02))
        # Draw small circles at the vertices
        col = tm.mix(col, tm.vec3(0), 1 - tm.smoothstep(0, 0.05, dv - 0.1))
        col = tm.mix(col, tm.vec3(1), 1 - tm.smoothstep(0, 0.05, dv - 0.08))
        pixels[i, j] = tm.sqrt(tm.clamp(col, 0, 1))


t0 = time.perf_counter()
gui = ti.GUI("2D Marching Squares", res=resolution, fast_gui=True)
while gui.running and not gui.get_event(gui.ESCAPE):
    iTime[None] = time.perf_counter() - t0
    march_squares()
    render()
    gui.set_image(pixels)
    gui.show()
