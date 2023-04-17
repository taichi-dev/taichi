import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.sparse)
def test_sparse_grid():
    # create a 2D sparse grid
    grid = ti.sparse.grid(
        {
            "pos": ti.math.vec2,
            "mass": ti.f32,
            "grid2particles": ti.types.vector(20, ti.i32),
        },
        shape=(10, 10),
    )

    # access
    grid[0, 0].pos = ti.math.vec2(1, 2)
    grid[0, 0].mass = 1.0
    grid[0, 0].grid2particles[2] = 123

    # print the usage of the sparse grid, which is in [0,1]
    assert ti.sparse.usage(grid) == test_utils.approx(0.01)
