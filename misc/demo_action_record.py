import taichi as ti
ti.core.start_recording('record.yml')
ti.init(arch=ti.cc)

x = ti.Vector(3, ti.f32, (512, 512))

@ti.kernel
def render():
    for i, j in x:
        x[i, j] = [i / 512, j / 512, 0]

@ti.kernel
def dump_ppm(tensor: ti.template()):
    if ti.static(isinstance(tensor, ti.Matrix)):
        print('P2')
    else:
        print('P3')
        print(tensor.shape[0], tensor.shape[1])
    print(255)
    for _ in range(1):
        for i in ti.grouped(ti.ndrange(*tensor.shape)):
            c = min(255, max(0, int(tensor[i] * 255 + 0.5)))
            if ti.static(isinstance(tensor, ti.Matrix)):
                r, g, b = c
                print(r, g, b)
            else:
                print(c)

render()
dump_ppm(x)
