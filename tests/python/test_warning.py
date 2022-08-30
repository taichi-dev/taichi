import taichi as ti
from tests import test_utils
import pytest


@test_utils.test()
def test_continue_in_static_for_in_non_static_if():
    @ti.kernel
    def test_static_loop():
        for i in ti.static(range(5)):
            x = 0.1
            if x == 0.0:
                continue

    with pytest.warns(SyntaxWarning, match="You are trying to continue in a static `for` loop"):
        test_static_loop()


@test_utils.test()
def test_break_in_static_for_in_non_static_if():
    @ti.kernel
    def test_static_loop():
        for i in ti.static(range(5)):
            x = 0.1
            if x == 0.0:
                break

    with pytest.warns(SyntaxWarning, match="You are trying to break in a static `for` loop"):
        test_static_loop()
