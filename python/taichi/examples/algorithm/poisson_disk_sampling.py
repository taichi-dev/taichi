"""
Poisson disk sampling in Taichi, a fancy version.
Based on Yuanming Hu's code: https://github.com/taichi-dev/poisson_disk_sampling

User interface:

1. Click on the window to restart the animation.
2. Press `p` to save screenshot.
"""
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

grid_n = 20
dx = 1 / grid_n
radius = dx * tm.sqrt(2)
desired_samples = 200
grid = ti.field(dtype=int, shape=(grid_n, grid_n))
samples = ti.Vector.field(2, dtype=float, shape=desired_samples)
window_size = 800
dfield = ti.Vector.field(4, dtype=float, shape=(window_size, window_size))
img = ti.Vector.field(3, dtype=float, shape=(window_size, window_size))
iMouse = ti.Vector.field(2, dtype=float, shape=())
iMouse[None] = [0.5, 0.5]
iResolution = tm.vec2(window_size)
head = ti.field(int, shape=())
tail = ti.field(int, shape=())
sample_count = ti.field(int, shape=())


@ti.func
def coord_to_index(p):
    return int(p * tm.vec2(grid_n))


@ti.kernel
def refresh_scene():
    head[None] = 0
    tail[None] = 1
    sample_count[None] = 1

    samples[0] = (p0 := iMouse[None])
    grid[coord_to_index(p0)] = 0

    for i, j in grid:
        grid[i, j] = -1

    for i, j in dfield:
        dfield[i, j] = tm.vec4(1e5)
        img[i, j] = tm.vec3(1)


@ti.func
def find_nearest_point(p):
    x, y = coord_to_index(p)
    dmin = 1e5
    nearest = iMouse[None]
    for i in range(ti.max(0, x - 2), ti.min(grid_n, x + 3)):
        for j in range(ti.max(0, y - 2), ti.min(grid_n, y + 3)):
            ind = grid[i, j]
            if ind != -1:
                q = samples[ind]
                d = (q - p).norm()
                if d < dmin:
                    dmin = d
                    nearest = q
    return dmin, nearest


@ti.kernel
def poisson_disk_sample(num_samples: int) -> int:
    while head[None] < tail[None] and head[None] < ti.min(num_samples, desired_samples):
        source_x = samples[head[None]]
        head[None] += 1

        for _ in range(100):
            theta = ti.random() * 2 * tm.pi
            offset = tm.vec2(tm.cos(theta), tm.sin(theta)) * (1 + ti.random()) * radius
            new_x = source_x + offset
            new_index = coord_to_index(new_x)

            if 0 <= new_x[0] < 1 and 0 <= new_x[1] < 1:
                collision = find_nearest_point(new_x)[0] < radius - 1e-6
                if not collision and tail[None] < desired_samples:
                    samples[tail[None]] = new_x
                    grid[new_index] = tail[None]
                    tail[None] += 1
    return tail[None]


@ti.func
def hash21(p):
    return tm.fract(tm.sin(tm.dot(p, tm.vec2(127.619, 157.583))) * 43758.5453)


@ti.func
def sample_dist(uv):
    uv = uv * iResolution
    x, y = tm.clamp(0, iResolution - 1, uv).cast(int)
    return dfield[x, y]


@ti.kernel
def compute_distance_field():
    for i, j in dfield:
        uv = tm.vec2(i, j) / iResolution
        d, p = find_nearest_point(uv)
        d = (uv - p).norm() - radius / 2.0
        dfield[i, j] = tm.vec4(d, p.x, p.y, radius / 2.0)


@ti.kernel
def render():
    for i, j in img:
        uv = tm.vec2(i, j) / iResolution.y
        st = tm.fract(uv * grid_n) - 0.5
        dg = 0.5 - abs(st)
        d1 = ti.min(dg.x, dg.y)
        d1 = tm.smoothstep(0.05, 0.0, d1)
        col = (1 - tm.vec3(d1)) * 0.7
        sf = 2 / iResolution.y
        buf = sample_dist(uv)
        bufSh = sample_dist(uv + tm.vec2(0.005, 0.015))
        cCol = tm.vec3(hash21(buf.yz + 0.3), hash21(buf.yz), hash21(buf.yz + 0.09))
        pat = (abs(tm.fract(-buf.x * 150) - 0.5) * 2) / 300
        col = tm.mix(col, tm.vec3(0), (1 - tm.smoothstep(0, 3 * sf, pat)) * 0.25)
        ew, ew2 = 0.005, 0.008
        cCol2 = tm.mix(cCol, tm.vec3(1), 0.9)
        col = tm.mix(col, tm.vec3(0), (1 - tm.smoothstep(0, sf * 2, bufSh.x)) * 0.4)
        col = tm.mix(col, tm.vec3(0), 1 - tm.smoothstep(sf, 0, -buf.x))
        col = tm.mix(col, cCol2, 1 - tm.smoothstep(sf, 0, -buf.x - ew))
        col = tm.mix(col, tm.vec3(0), 1 - tm.smoothstep(sf, 0, -buf.x - ew2 - ew))
        col = tm.mix(col, cCol, 1 - tm.smoothstep(sf, 0.0, -buf.x - ew2 - ew * 2))
        col = tm.sqrt(ti.max(col, 0))
        img[i, j] = col


def main():
    refresh_scene()
    gui = ti.ui.Window("Poisson Disk Sampling", res=(window_size, window_size))
    canvas = gui.get_canvas()
    gui.fps_limit = 10
    while gui.running:
        gui.get_event(ti.ui.PRESS)
        if gui.is_pressed(ti.ui.ESCAPE):
            gui.running = False

        if gui.is_pressed(ti.ui.LMB):
            iMouse[None] = gui.get_cursor_pos()
            refresh_scene()

        if gui.is_pressed("p"):
            canvas.set_image(img)
            gui.save_image("screenshot.png")

        poisson_disk_sample(sample_count[None])
        sample_count[None] += 1
        compute_distance_field()
        render()
        canvas.set_image(img)
        gui.show()


if __name__ == "__main__":
    main()
