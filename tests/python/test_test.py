'''
This file tests if Taichi's testing utilities are functional.

TODO: Skips these tests after all tests are using @ti.test
'''
import os

import pytest

import taichi as ti
from tests import test_utils

### `ti.test`


@test_utils.test()
def test_all_archs():
    assert ti.lang.impl.current_cfg().arch in test_utils.expected_archs()


@test_utils.test(arch=ti.cpu)
def test_arch_cpu():
    assert ti.lang.impl.current_cfg().arch in [ti.cpu]


@test_utils.test(arch=[ti.cpu])
def test_arch_list_cpu():
    assert ti.lang.impl.current_cfg().arch in [ti.cpu]


@test_utils.test(exclude=ti.cpu)
def test_exclude_cpu():
    assert ti.lang.impl.current_cfg().arch not in [ti.cpu]


@test_utils.test(exclude=[ti.cpu])
def test_exclude_list_cpu():
    assert ti.lang.impl.current_cfg().arch not in [ti.cpu]


@test_utils.test(arch=ti.opengl)
def test_arch_opengl():
    assert ti.lang.impl.current_cfg().arch in [ti.opengl]


@test_utils.test(arch=[ti.cpu, ti.opengl, ti.metal])
def test_multiple_archs():
    assert ti.lang.impl.current_cfg().arch in [ti.cpu, ti.opengl, ti.metal]


@test_utils.test(arch=ti.cpu, debug=True, advanced_optimization=False)
def test_init_args():
    assert ti.lang.impl.current_cfg().debug == True
    assert ti.lang.impl.current_cfg().advanced_optimization == False


@test_utils.test(require=ti.extension.sparse)
def test_require_extensions_1():
    assert ti.lang.impl.current_cfg().arch in [ti.cpu, ti.cuda, ti.metal]


@test_utils.test(arch=[ti.cpu, ti.opengl], require=ti.extension.sparse)
def test_require_extensions_2():
    assert ti.lang.impl.current_cfg().arch in [ti.cpu]


@test_utils.test(arch=[ti.cpu, ti.opengl],
                 require=[ti.extension.sparse, ti.extension.bls])
def test_require_extensions_2():
    assert ti.lang.impl.current_cfg().arch in [ti.cuda]


### `test_utils.approx` and `test_utils.allclose`


@pytest.mark.parametrize('x', [0.1, 3])
@pytest.mark.parametrize(
    'allclose', [test_utils.allclose, lambda x, y: x == test_utils.approx(y)])
@test_utils.test()
def test_allclose_rel(x, allclose):
    rel = test_utils.get_rel_eps()
    assert not allclose(x + x * rel * 3.0, x)
    assert not allclose(x + x * rel * 1.2, x)
    assert allclose(x + x * rel * 0.9, x)
    assert allclose(x + x * rel * 0.5, x)
    assert allclose(x, x)
    assert allclose(x - x * rel * 0.5, x)
    assert allclose(x - x * rel * 0.9, x)
    assert not allclose(x - x * rel * 1.2, x)
    assert not allclose(x - x * rel * 3.0, x)


@pytest.mark.parametrize('x', [0.1, 3])
@pytest.mark.parametrize(
    'allclose', [test_utils.allclose, lambda x, y: x == test_utils.approx(y)])
@test_utils.test()
def test_allclose_rel_reordered1(x, allclose):
    rel = test_utils.get_rel_eps()
    assert not allclose(x + x * rel * 3.0, x)
    assert not allclose(x + x * rel * 1.2, x)
    assert allclose(x + x * rel * 0.9, x)
    assert allclose(x + x * rel * 0.5, x)
    assert allclose(x, x)
    assert allclose(x - x * rel * 0.5, x)
    assert allclose(x - x * rel * 0.9, x)
    assert not allclose(x - x * rel * 1.2, x)
    assert not allclose(x - x * rel * 3.0, x)


@pytest.mark.parametrize('x', [0.1, 3])
@pytest.mark.parametrize(
    'allclose', [test_utils.allclose, lambda x, y: x == test_utils.approx(y)])
@test_utils.test()
def test_allclose_rel_reordered2(x, allclose):
    rel = test_utils.get_rel_eps()
    assert not allclose(x + x * rel * 3.0, x)
    assert not allclose(x + x * rel * 1.2, x)
    assert allclose(x + x * rel * 0.9, x)
    assert allclose(x + x * rel * 0.5, x)
    assert allclose(x, x)
    assert allclose(x - x * rel * 0.5, x)
    assert allclose(x - x * rel * 0.9, x)
    assert not allclose(x - x * rel * 1.2, x)
    assert not allclose(x - x * rel * 3.0, x)


@pytest.mark.skipif(ti._lib.core.with_metal(),
                    reason="Skip metal because metal is used as the example")
def test_disable_fallback():
    old_environ = os.environ.get('TI_WANTED_ARCHS', '')
    os.environ['TI_WANTED_ARCHS'] = "metal"

    with pytest.raises(RuntimeError):

        @test_utils.test(ti.metal)
        def test():
            pass

        test()
        os.environ['TI_WANTED_ARCHS'] = old_environ
    os.environ['TI_WANTED_ARCHS'] = old_environ
