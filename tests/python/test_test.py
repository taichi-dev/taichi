'''
To test if Taichi testing utils are functional.

TODO: Skips these tests after all tests adapted to @ti.test
'''

import taichi as ti
import pytest

# ti.test


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


### ti.approx


@ti.test(ti.cpu)
@pytest.mark.parametrize('x', [0.1, 3])
@pytest.mark.parametrize('rel', [1e-6])
def test_approx_fine_rel(x, rel):
    assert x + x * rel * 0.9 == ti.approx(x)
    assert x + x * rel * 0.5 == ti.approx(x)
    assert x == ti.approx(x)
    assert x - x * rel * 0.5 == ti.approx(x)
    assert x - x * rel * 0.9 == ti.approx(x)


@ti.test(ti.opengl)
@pytest.mark.parametrize('x', [0.1, 3])
@pytest.mark.parametrize('rel', [1e-3])
def test_approx_poor_rel(x, rel):
    assert x + x * rel * 0.9 == ti.approx(x)
    assert x + x * rel * 0.5 == ti.approx(x)
    assert x == ti.approx(x)
    assert x - x * rel * 0.5 == ti.approx(x)
    assert x - x * rel * 0.9 == ti.approx(x)


@pytest.mark.parametrize('x', [0.1, 3])
@ti.test(ti.opengl)
@pytest.mark.parametrize('rel', [1e-3])
def test_approx_poor_rel_reordered1(x, rel):
    assert x + x * rel * 0.9 == ti.approx(x)
    assert x + x * rel * 0.5 == ti.approx(x)
    assert x == ti.approx(x)
    assert x - x * rel * 0.5 == ti.approx(x)
    assert x - x * rel * 0.9 == ti.approx(x)


@pytest.mark.parametrize('x', [0.1, 3])
@pytest.mark.parametrize('rel', [1e-3])
@ti.test(ti.opengl)
def test_approx_poor_rel_reordered2(x, rel):
    assert x + x * rel * 0.9 == ti.approx(x)
    assert x + x * rel * 0.5 == ti.approx(x)
    assert x == ti.approx(x)
    assert x - x * rel * 0.5 == ti.approx(x)
    assert x - x * rel * 0.9 == ti.approx(x)


@ti.test(ti.cpu)
@pytest.mark.parametrize('x', [0.1, 3])
@pytest.mark.parametrize('rel', [1e-6])
def test_allclose_fine_rel(x, rel):
    assert ti.allclose(x + x * rel * 0.9, x)
    assert ti.allclose(x + x * rel * 0.5, x)
    assert ti.allclose(x, x)
    assert ti.allclose(x - x * rel * 0.5, x)
    assert ti.allclose(x - x * rel * 0.9, x)


@ti.test(ti.opengl)
@pytest.mark.parametrize('x', [0.1, 3])
@pytest.mark.parametrize('rel', [1e-3])
def test_allclose_poor_rel(x, rel):
    assert ti.allclose(x + x * rel * 0.9, x)
    assert ti.allclose(x + x * rel * 0.5, x)
    assert ti.allclose(x, x)
    assert ti.allclose(x - x * rel * 0.5, x)
    assert ti.allclose(x - x * rel * 0.9, x)
