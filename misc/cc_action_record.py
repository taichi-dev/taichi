import taichi as ti
ti.core.start_recording('record.yml')
ti.init(arch=ti.cc)

n = 512
x = ti.Vector(3, ti.f32, (n, n))


@ti.kernel
def render():
    for i, j in x:
        x[i, j] = [i / x.shape[0], j / x.shape[1], 0]


@ti.kernel
def dump_ppm(tensor: ti.template()):
    if ti.static(isinstance(tensor, ti.Matrix)):
        print('P3')
    else:
        print('P2')
    print(tensor.shape[0], tensor.shape[1], 255)
    for _ in range(1):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                c = min(255,
                        max(0, int(tensor[j, x.shape[1] - 1 - i] * 255 + 0.5)))
                if ti.static(isinstance(tensor, ti.Matrix)):
                    r, g, b = c
                    print(r, g, b)
                else:
                    print(c)


render()
dump_ppm(x)
ti.imshow(x)
