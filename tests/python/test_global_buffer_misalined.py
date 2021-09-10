import taichi as ti


@ti.test(require=ti.extension.data64)
def test_global_buffer_misalignment():
    @ti.kernel
    def test(x: ti.f32):
        a = x
        b = ti.cast(0.12, ti.f64)
        for i in range(8):
            b += a

    for i in range(8):
        test(0.1)
