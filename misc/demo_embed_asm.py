import taichi as ti

ti.init(arch=ti.opengl, log_level=ti.DEBUG)

n = 512

x = ti.var(ti.f32, (n, n))

@ti.kernel
def render():
    for i, j in x:
        ret = 0.0
        # outputs must be alloca's, i.e. outputs=[x[i, j]] is not allowed
        ti.asm('$0 = round(length(vec2(%0, %1) - 0.5) * 20) * 0.1',
                inputs=[i / n, j / n], outputs=[ret])
        x[i, j] = ret

render()
ti.imshow(x)
