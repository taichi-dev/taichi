'''
This file tests if Taichi's testing utilities are functional.

TODO: Skips these tests after all tests are using @ti.test
'''

import taichi as ti
import pytest

### `ti.test`


@ti.test()
def test_all_archs():
    assert ti.cfg.arch in ti.supported_archs()


@ti.test(ti.cpu)
def test_arch_cpu():
    assert ti.cfg.arch in [ti.cpu]


@ti.test(ti.opengl)
def test_arch_opengl():
    assert ti.cfg.arch in [ti.opengl]


@ti.test(ti.cpu, ti.opengl, ti.metal)
def test_multiple_archs():
    assert ti.cfg.arch in [ti.cpu, ti.opengl, ti.metal]


@ti.test(ti.cpu, debug=True, advanced_optimization=False)
def test_init_args():
    assert ti.cfg.debug == True
    assert ti.cfg.advanced_optimization == False


@ti.test(ti.cpu, ti.opengl, extensions=[ti.extension.sparse])
def test_require_extensions():
    assert ti.cfg.arch in [ti.cpu]


### `ti.approx` and `ti.allclose`


@ti.test()
@pytest.mark.parametrize('x', [0.1, 3])
@pytest.mark.parametrize('allclose',
        [ti.allclose, lambda x, y: x == ti.approx(y)])
def test_allclose_rel(x, allclose):
    rel = 1e-3 if ti.cfg.arch == ti.opengl else 1e-6
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
@ti.test()
@pytest.mark.parametrize('allclose',
        [ti.allclose, lambda x, y: x == ti.approx(y)])
def test_allclose_rel_reordered1(x, allclose):
    rel = 1e-3 if ti.cfg.arch == ti.opengl else 1e-6
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
@pytest.mark.parametrize('allclose',
        [ti.allclose, lambda x, y: x == ti.approx(y)])
@ti.test()
def test_allclose_rel_reordered2(x, allclose):
    rel = 1e-3 if ti.cfg.arch == ti.opengl else 1e-6
    assert not allclose(x + x * rel * 3.0, x)
    assert not allclose(x + x * rel * 1.2, x)
    assert allclose(x + x * rel * 0.9, x)
    assert allclose(x + x * rel * 0.5, x)
    assert allclose(x, x)
    assert allclose(x - x * rel * 0.5, x)
    assert allclose(x - x * rel * 0.9, x)
    assert not allclose(x - x * rel * 1.2, x)
    assert not allclose(x - x * rel * 3.0, x)
