import taichi as ti

# The walrus operator is not supported until python 3.8,
# and pytest cannot handle files containing walrus operators when python version is below 3.8.
# So, we moved this test to the directory "python38".
# Tests in this directory will not be executed when python version is below 3.8.
# See https://github.com/taichi-dev/taichi/issues/3425 for more information.


@ti.test()
def test_namedexpr():

    @ti.kernel
    def foo() -> ti.i32:
        b = 2 + (a := 5)
        b += a
        return b

    assert foo() == 12
