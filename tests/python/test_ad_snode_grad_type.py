from taichi.lang.enums import SNodeGradType

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_snode_grad_type():
    x = ti.field(float, shape=(), needs_grad=True, needs_dual=True)
    assert x.snode.ptr.get_snode_grad_type() == SNodeGradType.PRIMAL
    assert x.grad.snode.ptr.get_snode_grad_type() == SNodeGradType.ADJOINT
    assert x.dual.snode.ptr.get_snode_grad_type() == SNodeGradType.DUAL


@test_utils.test()
def test_snode_grad_type_lazy():
    x = ti.field(float, shape=())
    ti.root.lazy_grad()
    ti.root.lazy_dual()
    assert x.snode.ptr.get_snode_grad_type() == SNodeGradType.PRIMAL
    assert x.grad.snode.ptr.get_snode_grad_type() == SNodeGradType.ADJOINT
    assert x.dual.snode.ptr.get_snode_grad_type() == SNodeGradType.DUAL


@test_utils.test()
def test_snode_clear_gradient():
    x = ti.field(float, shape=(), needs_grad=True, needs_dual=True)
    y = ti.field(float, shape=(), needs_grad=True, needs_dual=True)

    x[None] = 1.0

    @ti.kernel
    def compute():
        y[None] += x[None]**2

    with ti.ad.Tape(loss=y):
        compute()

    with ti.ad.FwdMode(loss=y, param=x):
        compute()

    assert x.grad[None] == 2.0

    with ti.ad.Tape(loss=y):
        compute()

    assert y.dual[None] == 2.0


#TODO: Add test for `adjoint_checkbit` after #5801 merged.
