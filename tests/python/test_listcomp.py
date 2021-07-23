import taichi as ti


@ti.all_archs
def test_listcomp_multiple_ifs():
    x = ti.field(ti.i32, shape=(4,))

    @ti.kernel
    def test() -> ti.i32:
        # Taichi doesn't support global fields appearing anywhere after "for"
        # here.
        a = [x[0] for j in range(100) if j > 2 if j < 5]
        return sum(a)

    for i in range(6):
        x[0] = i
        assert test() == i * 2
