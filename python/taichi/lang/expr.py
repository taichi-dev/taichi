from .core import taichi_lang_core
from .util import *
from . import impl
from .common_ops import TaichiOperations
import traceback


# Scalar, basic data type
class Expr(TaichiOperations):
    materialize_layout_callback = None
    layout_materialized = False

    def __init__(self, *args, tb=None):
        self.getter = None
        self.setter = None
        self.tb = tb
        if len(args) == 1:
            if isinstance(args[0], taichi_lang_core.Expr):
                self.ptr = args[0]
            elif isinstance(args[0], Expr):
                self.ptr = args[0].ptr
                self.tb = args[0].tb
            else:
                arg = args[0]
                try:
                    import numpy as np
                    if isinstance(arg, np.ndarray):
                        arg = arg.dtype(arg)
                except:
                    pass
                from .impl import make_constant_expr
                self.ptr = make_constant_expr(arg).ptr
        else:
            assert False
        if self.tb:
            self.ptr.set_tb(self.tb)
        self.grad = None
        self.val = self

    @python_scope
    def __setitem__(self, key, value):
        if not Expr.layout_materialized:
            self.materialize_layout_callback()
        self.initialize_accessor()
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key, )
        assert len(key) == self.dim()
        key = key + ((0, ) *
                     (taichi_lang_core.get_max_num_indices() - len(key)))
        self.setter(value, *key)

    @python_scope
    def __getitem__(self, key):
        if not Expr.layout_materialized:
            self.materialize_layout_callback()
        self.initialize_accessor()
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key, )
        key = key + ((0, ) *
                     (taichi_lang_core.get_max_num_indices() - len(key)))
        return self.getter(*key)

    def get_tensor_members(self):
        return [self]

    def variable(self):
        return impl.expr_init(self)

    def is_global(self):
        return self.ptr.is_global_var()

    @global_scope
    @taichi_scope
    def loop_range(self):
        return self

    def serialize(self):
        return self.ptr.serialize()

    @python_scope
    def initialize_accessor(self):
        if self.getter:
            return
        snode = self.ptr.snode()

        if self.snode().data_type() == f32 or self.snode().data_type() == f64:

            def getter(*key):
                assert len(key) == taichi_lang_core.get_max_num_indices()
                return snode.read_float(key)

            def setter(value, *key):
                assert len(key) == taichi_lang_core.get_max_num_indices()
                snode.write_float(key, value)
        else:
            if taichi_lang_core.is_signed(self.snode().data_type()):

                def getter(*key):
                    assert len(key) == taichi_lang_core.get_max_num_indices()
                    return snode.read_int(key)
            else:

                def getter(*key):
                    assert len(key) == taichi_lang_core.get_max_num_indices()
                    return snode.read_uint(key)

            def setter(value, *key):
                assert len(key) == taichi_lang_core.get_max_num_indices()
                snode.write_int(key, value)

        self.getter = getter
        self.setter = setter

    @python_scope
    def set_grad(self, grad):
        self.grad = grad
        self.ptr.set_grad(grad.ptr)

    @python_scope
    def clear(self, deactivate=False):
        assert not deactivate
        node = self.ptr.snode().parent
        assert node
        node.clear_data()

    @python_scope
    def fill(self, val):
        # TODO: avoid too many template instantiations
        from .meta import fill_tensor
        fill_tensor(self, val)

    def parent(self, n=1):
        import taichi as ti
        p = self.ptr.snode()
        for i in range(n):
            p = p.parent
        return Expr(ti.core.global_var_expr_from_snode(p))

    def snode(self):
        from .snode import SNode
        return SNode(self.ptr.snode())

    def __hash__(self):
        return self.ptr.get_raw_address()

    def dim(self):
        if not Expr.layout_materialized:
            self.materialize_layout_callback()
        return self.snode().dim()

    def shape(self):
        if not Expr.layout_materialized:
            self.materialize_layout_callback()
        return self.snode().shape()

    def data_type(self):
        return self.snode().data_type()

    @python_scope
    def to_numpy(self):
        from .meta import tensor_to_ext_arr
        import numpy as np
        arr = np.zeros(shape=self.shape(),
                       dtype=to_numpy_type(self.snode().data_type()))
        tensor_to_ext_arr(self, arr)
        import taichi as ti
        ti.sync()
        return arr

    @python_scope
    def to_torch(self, device=None):
        from .meta import tensor_to_ext_arr
        import torch
        arr = torch.zeros(size=self.shape(),
                          dtype=to_pytorch_type(self.snode().data_type()),
                          device=device)
        tensor_to_ext_arr(self, arr)
        import taichi as ti
        ti.sync()
        return arr

    @python_scope
    def from_numpy(self, arr):
        assert self.dim() == len(arr.shape)
        s = self.shape()
        for i in range(self.dim()):
            assert s[i] == arr.shape[i]
        from .meta import ext_arr_to_tensor
        if hasattr(arr, 'contiguous'):
            arr = arr.contiguous()
        ext_arr_to_tensor(arr, self)
        import taichi as ti
        ti.sync()

    @python_scope
    def from_torch(self, arr):
        self.from_numpy(arr.contiguous())

    @python_scope
    def copy_from(self, other):
        assert isinstance(other, Expr)
        from .meta import tensor_to_tensor
        assert self.dim() == other.dim()
        tensor_to_tensor(self, other)


def make_var_vector(size):
    import taichi as ti
    exprs = []
    for i in range(size):
        exprs.append(taichi_lang_core.make_id_expr(''))
    return ti.Vector(exprs)


def make_expr_group(*exprs):
    if len(exprs) == 1:
        from .matrix import Matrix
        if isinstance(exprs[0], (list, tuple)):
            exprs = exprs[0]
        elif isinstance(exprs[0], Matrix):
            mat = exprs[0]
            assert mat.m == 1
            exprs = mat.entries
    expr_group = taichi_lang_core.ExprGroup()
    for i in exprs:
        expr_group.push_back(Expr(i).ptr)
    return expr_group
