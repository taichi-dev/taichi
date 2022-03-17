import taichi as ti
from tests import test_utils


# such small block_dim will cause grid_dim too large for OpenGL...
@test_utils.test(exclude=ti.opengl)
def test_parallel_range_for():
    n = 1024 * 1024
    val = ti.field(ti.i32, shape=(n))

    @ti.kernel
    def fill():
        ti.loop_config(parallelize=8)
        ti.loop_config(block_dim=8)
        for i in range(n):
            val[i] = i

    fill()
    # To speed up
    val_np = val.to_numpy()
    for i in range(n):
        assert val_np[i] == i


@test_utils.test()
def test_serial_for():
    @ti.kernel
    def foo() -> ti.i32:
        a = 0
        ti.loop_config(serialize=True)
        for i in range(100):
            a = a + 1
            if a == 50:
                break

        return a

    assert foo() == 50


@test_utils.test(exclude=ti.opengl)
def test_loop_config_parallel_range_for():
    n = 1024 * 1024
    val = ti.field(ti.i32, shape=(n))

    @ti.kernel
    def fill():
        ti.loop_config(parallelize=8, block_dim=8)
        for i in range(n):
            val[i] = i

    fill()
    # To speed up
    val_np = val.to_numpy()
    for i in range(n):
        assert val_np[i] == i


@test_utils.test()
def test_loop_config_serial_for():
    @ti.kernel
    def foo() -> ti.i32:
        a = 0
        ti.loop_config(serialize=True)
        for i in range(100):
            a = a + 1
            if a == 50:
                break

        return a

    assert foo() == 50
