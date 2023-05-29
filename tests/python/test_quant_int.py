import taichi as ti
from tests import test_utils

# TODO: validation layer support on macos vulkan backend is not working.
vk_on_mac = (ti.vulkan, "Darwin")

# TODO: capfd doesn't function well on CUDA backend on Windows
cuda_on_windows = (ti.cuda, "Windows")


@test_utils.test(require=ti.extension.quant_basic)
def test_quant_int_implicit_cast():
    qi13 = ti.types.quant.int(13, True)
    x = ti.field(dtype=qi13)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    ti.root.place(bitpack)

    @ti.kernel
    def foo():
        x[None] = 10.3

    foo()
    assert x[None] == 10


@test_utils.test(
    require=ti.extension.quant_basic,
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[vk_on_mac, cuda_on_windows],
    debug=True,
)
def test_quant_store_fusion(capfd):
    x = ti.field(dtype=ti.types.quant.int(16, True))
    y = ti.field(dtype=ti.types.quant.int(16, True))
    v = ti.BitpackedFields(max_num_bits=32)
    v.place(x, y)
    ti.root.dense(ti.i, 10).place(v)

    # should fuse store
    @ti.kernel
    def store():
        ti.loop_config(serialize=True)
        for i in range(10):
            x[i] = i
            y[i] = i + 1
            print(x[i], y[i])

    store()
    ti.sync()

    out, err = capfd.readouterr()
    expected_out = """0 1
1 2
2 3
3 4
4 5
5 6
6 7
7 8
8 9
9 10
"""
    assert out == expected_out and err == ""


@test_utils.test(
    require=ti.extension.quant_basic,
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[vk_on_mac, cuda_on_windows],
    debug=True,
)
def test_quant_store_no_fusion(capfd):
    x = ti.field(dtype=ti.types.quant.int(16, True))
    y = ti.field(dtype=ti.types.quant.int(16, True))
    v = ti.BitpackedFields(max_num_bits=32)
    v.place(x, y)
    ti.root.dense(ti.i, 10).place(v)

    @ti.kernel
    def store():
        ti.loop_config(serialize=True)
        for i in range(10):
            x[i] = i
            print(x[i])
            y[i] = i + 1
            print(y[i])

    store()
    ti.sync()

    out, err = capfd.readouterr()
    expected_out = """0
1
1
2
2
3
3
4
4
5
5
6
6
7
7
8
8
9
9
10
"""
    assert out == expected_out and err == ""
