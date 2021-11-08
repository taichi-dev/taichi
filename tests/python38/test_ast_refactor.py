from sys import version_info

import pytest

import taichi as ti


@ti.test(experimental_ast_refactor=True)
def test_namedexpr():
    @ti.kernel
    def foo() -> ti.i32:
        b = 2 + (a := 5)
        b += a
        return b

    assert foo() == 12
