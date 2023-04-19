import taichi as ti
from tests import test_utils

# The walrus operator is not supported until python 3.8,
# and pytest cannot handle files containing walrus operators when python version is below 3.8.
# So, we moved this test to the directory "python38".
# Tests in this directory will not be executed when python version is below 3.8.
# See https://github.com/taichi-dev/taichi/issues/3425 for more information.


@test_utils.test()
def test_namedexpr():
    @ti.kernel
    def foo() -> ti.i32:
        b = 2 + (a := 5)
        b += a
        return b

    assert foo() == 12


# TODO: validation layer support on macos vulkan backend is not working.
vk_on_mac = (ti.vulkan, "Darwin")

# TODO: capfd doesn't function well on CUDA backend on Windows
cuda_on_windows = (ti.cuda, "Windows")


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_docs_scalar_self_documenting_exp(capfd):
    a = ti.field(ti.f32, 4)

    @ti.kernel
    def func():
        a[0] = 1.0

        # with self-documenting expressions
        print(f"{a[0] = :.1f}")

    func()
    ti.sync()

    out, err = capfd.readouterr()
    expected_out = """a[0] = 1.0
"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_docs_matrix_self_documenting_exp(capfd):
    @ti.kernel
    def func():
        m = ti.Matrix([[2e1, 3e2, 4e3], [5e4, 6e5, 7e6]], ti.f32)

        # with self-documenting expressions
        print(f"{m = :g}")

    func()
    ti.sync()

    out, err = capfd.readouterr()
    expected_out = """m = [[20, 300, 4000], [50000, 600000, 7e+06]]
"""
    assert out == expected_out and err == ""
