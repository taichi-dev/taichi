'''
This file tests if Taichi's testing utilities are functional.

TODO: Skips these tests after all tests are using @ti.test
'''

import pytest

import taichi as ti

### `ti.test`


@ti.test()
def test_all_archs():
    assert ti.cfg.arch in ti.supported_archs()


@ti.test(arch=ti.cpu)
def test_arch_cpu():
    assert ti.cfg.arch in [ti.cpu]


@ti.test(arch=[ti.cpu])
def test_arch_list_cpu():
    assert ti.cfg.arch in [ti.cpu]


@ti.test(exclude=ti.cpu)
def test_exclude_cpu():
    assert ti.cfg.arch not in [ti.cpu]


@ti.test(exclude=[ti.cpu])
def test_exclude_list_cpu():
    assert ti.cfg.arch not in [ti.cpu]


@ti.test(arch=ti.opengl)
def test_arch_opengl():
    assert ti.cfg.arch in [ti.opengl]


@ti.test(arch=[ti.cpu, ti.opengl, ti.metal])
def test_multiple_archs():
    assert ti.cfg.arch in [ti.cpu, ti.opengl, ti.metal]


@ti.test(arch=ti.cpu, debug=True, advanced_optimization=False)
def test_init_args():
    assert ti.cfg.debug == True
    assert ti.cfg.advanced_optimization == False


@ti.test(require=ti.extension.sparse)
def test_require_extensions_1():
    assert ti.cfg.arch in [ti.cpu, ti.cuda, ti.metal]


@ti.test(arch=[ti.cpu, ti.opengl], require=ti.extension.sparse)
def test_require_extensions_2():
    assert ti.cfg.arch in [ti.cpu]


@ti.test(arch=[ti.cpu, ti.opengl],
         require=[ti.extension.sparse, ti.extension.bls])
def test_require_extensions_2():
    assert ti.cfg.arch in [ti.cuda]


### `ti.approx` and `ti.allclose`


@ti.test()
@pytest.mark.parametrize('x', [0.1, 3])
@pytest.mark.parametrize('allclose',
                         [ti.allclose, lambda x, y: x == ti.approx(y)])
def test_allclose_rel(x, allclose):
    rel = ti.get_rel_eps()
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
    rel = ti.get_rel_eps()
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
    rel = ti.get_rel_eps()
    assert not allclose(x + x * rel * 3.0, x)
    assert not allclose(x + x * rel * 1.2, x)
    assert allclose(x + x * rel * 0.9, x)
    assert allclose(x + x * rel * 0.5, x)
    assert allclose(x, x)
    assert allclose(x - x * rel * 0.5, x)
    assert allclose(x - x * rel * 0.9, x)
    assert not allclose(x - x * rel * 1.2, x)
    assert not allclose(x - x * rel * 3.0, x)
