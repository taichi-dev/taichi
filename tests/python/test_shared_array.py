import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda)
def test_shared_array_save():
    block_dim = 128
    pad_num = 16
    a = ti.field(dtype=ti.f32, shape=(block_dim * pad_num, ))

    @ti.kernel
    def func():
        ti.loop_config(block_dim=block_dim)
        for i in range(block_dim * pad_num):
            g_tid = ti.global_thread_idx()
            tid = g_tid % block_dim
            pad = ti.simt.block.SharedArray((block_dim, ), ti.f32)
            pad[tid] = tid * 2.0
            ti.simt.block.sync()
            a[i] = pad[tid]
            ti.simt.block.sync()

    func()
    for i in range(pad_num):
        assert a[i * block_dim + 7] == 14.0
        assert a[i * block_dim + 29] == 58.0
        assert a[i * block_dim + 127] == 254.0
