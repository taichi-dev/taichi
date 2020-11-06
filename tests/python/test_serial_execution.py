import taichi as ti


@ti.test(arch=ti.cpu, cpu_max_num_threads=1)
def test_serial_range_for():
    n = 1024 * 32
    s = ti.field(dtype=ti.i32, shape=n)
    counter = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def fill_range():
        counter[None] = 0
        for i in range(n):
            s[ti.atomic_add(counter[None], 1)] = i

    fill_range()

    for i in range(n):
        assert s[i] == i


@ti.test(arch=ti.cpu, cpu_max_num_threads=1)
def test_serial_struct_for():
    n = 1024 * 32
    s = ti.field(dtype=ti.i32, shape=n)
    counter = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def fill_struct():
        counter[None] = 0
        for i in s:
            s[ti.atomic_add(counter[None], 1)] = i

    fill_struct()

    for i in range(n):
        assert s[i] == i
