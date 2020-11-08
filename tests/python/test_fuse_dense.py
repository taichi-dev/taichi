import taichi as ti
from .fuse_test_template import template_fuse_dense_x2y2z, \
    template_fuse_reduction


# @ti.test(require=ti.extension.async_mode, async_mode=True)
# def test_fuse_dense_x2y2z():
#     template_fuse_dense_x2y2z(size=100 * 1024**2)
#
#
# @ti.test(require=ti.extension.async_mode, async_mode=True)
# def test_fuse_reduction():
#     template_fuse_reduction(size=10 * 1024**2)


@ti.test(require=ti.extension.async_mode, async_mode=True,
         async_opt_fusion=False,
         async_opt_listgen=False,
         async_opt_dse=False,
         async_opt_activation_demotion=False,
         arch=ti.cpu)
def test_no_fuse_sigs_mismatch():
    n = 1
    x = ti.field(ti.i32, shape=(n, ))

    @ti.kernel
    def inc_i():
        for i in x:
            x[i] += i

    @ti.kernel
    def inc_by(k: ti.i32):
        for i in x:
            x[i] += k

    inc_i()
    inc_by(0)

    x = x.to_numpy()
    assert x[0] == 0