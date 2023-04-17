import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

N = 12
dt = 5e-5
dx = 1 / N
rho = 4e1
NF = 2 * N**2  # number of faces
NV = (N + 1) ** 2  # number of vertices
E, nu = 4e4, 0.2  # Young's modulus and Poisson's ratio
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)  # Lame parameters
ball_pos, ball_radius = ti.Vector([0.5, 0.0]), 0.31
damping = 14.5

pos = ti.Vector.field(2, float, NV, needs_grad=True)
vel = ti.Vector.field(2, float, NV)
f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, float, NF)
F = ti.Matrix.field(2, 2, float, NF, needs_grad=True)
V = ti.field(float, NF)
phi = ti.field(float, NF)  # potential energy of each face (Neo-Hookean)
U = ti.field(float, (), needs_grad=True)  # total potential energy

gravity = ti.Vector.field(2, float, ())
attractor_pos = ti.Vector.field(2, float, ())
attractor_strength = ti.field(float, ())


@ti.kernel
def update_U():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        V[i] = abs((a - c).cross(b - c))
        D_i = ti.Matrix.cols([a - c, b - c])
        F[i] = D_i @ B[i]
    for i in range(NF):
        F_i = F[i]
        log_J_i = ti.log(F_i.determinant())
        phi_i = mu / 2 * ((F_i.transpose() @ F_i).trace() - 2)
        phi_i -= mu * log_J_i
        phi_i += lam / 2 * log_J_i**2
        phi[i] = phi_i
        U[None] += V[i] * phi_i


@ti.kernel
def advance():
    for i in range(NV):
        acc = -pos.grad[i] / (rho * dx**2)
        g = gravity[None] * 0.8 + attractor_strength[None] * (attractor_pos[None] - pos[i]).normalized(1e-5)
        vel[i] += dt * (acc + g * 40)
        vel[i] *= ti.exp(-dt * damping)
    for i in range(NV):
        # ball boundary condition:
        disp = pos[i] - ball_pos
        disp2 = disp.norm_sqr()
        if disp2 <= ball_radius**2:
            NoV = vel[i].dot(disp)
            if NoV < 0:
                vel[i] -= NoV * disp / disp2
        cond = (pos[i] < 0) & (vel[i] < 0) | (pos[i] > 1) & (vel[i] > 0)
        # rect boundary condition:
        for j in ti.static(range(pos.n)):
            if cond[j]:
                vel[i][j] = 0
        pos[i] += dt * vel[i]


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos[k] = ti.Vector([i, j]) / N * 0.25 + ti.Vector([0.45, 0.45])
        vel[k] = ti.Vector([0, 0])
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        B_i_inv = ti.Matrix.cols([a - c, b - c])
        B[i] = B_i_inv.inverse()


@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2
        a = i * (N + 1) + j
        b = a + 1
        c = a + N + 2
        d = a + N + 1
        f2v[k + 0] = [a, b, c]
        f2v[k + 1] = [c, d, a]


window = ti.ui.Window("FEM128", (512, 512))
canvas = window.get_canvas()

# rendering related fields
vertexColors = ti.Vector.field(3, float, NV)
vertexPositions = ti.Vector.field(2, float, NV)
triangleIndices = ti.field(int, NF * 3)
mouse_circle = ti.Vector.field(2, dtype=float, shape=(1,))
ball_circle = ti.Vector.field(2, dtype=float, shape=(1,))


@ti.kernel
def paint_triangles():
    for i in range(NF):
        k = phi[i] * (10 / E)
        gb = (1 - k) * 0.5
        color = ti.Vector([k + gb, gb, gb])

        ia, ib, ic = f2v[i]
        vertexColors[ia] = color
        vertexColors[ib] = color
        vertexColors[ic] = color
        vertexPositions[ia] = pos[ia]
        vertexPositions[ib] = pos[ib]
        vertexPositions[ic] = pos[ic]
        triangleIndices[i * 3 + 0] = ia
        triangleIndices[i * 3 + 1] = ib
        triangleIndices[i * 3 + 2] = ic


def paint_mouse_ball():
    mouse = window.get_cursor_pos()
    mouse_circle[0] = ti.Vector([mouse[0], mouse[1]])
    ball_circle[0] = ball_pos


def render():
    paint_triangles()
    paint_mouse_ball()

    canvas.circles(mouse_circle, color=(0.2, 0.4, 0.6), radius=0.02)
    canvas.circles(ball_circle, color=(0.4, 0.4, 0.4), radius=ball_radius)

    canvas.triangles(vertexPositions, indices=triangleIndices, per_vertex_color=vertexColors)
    canvas.circles(vertexPositions, radius=0.003, color=(1, 0.6, 0.2))


def main():
    init_mesh()
    init_pos()
    gravity[None] = [0, -1]

    print(
        "[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse buttons to attract/repel. Press R to reset."
    )
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == "r":
                init_pos()
            elif e.key in ("a", ti.ui.LEFT):
                gravity[None] = [-1, 0]
            elif e.key in ("d", ti.ui.RIGHT):
                gravity[None] = [+1, 0]
            elif e.key in ("s", ti.ui.DOWN):
                gravity[None] = [0, -1]
            elif e.key in ("w", ti.ui.UP):
                gravity[None] = [0, +1]

        mouse_pos = window.get_cursor_pos()
        attractor_pos[None] = mouse_pos
        attractor_strength[None] = window.is_pressed(ti.ui.LMB) - window.is_pressed(ti.ui.RMB)
        for i in range(50):
            with ti.ad.Tape(loss=U):
                update_U()
            advance()
        render()
        window.show()


if __name__ == "__main__":
    main()
