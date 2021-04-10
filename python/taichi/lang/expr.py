from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl
from taichi.lang.common_ops import TaichiOperations
from taichi.lang.util import (is_taichi_class, python_scope, to_numpy_type,
                              to_pytorch_type)
from taichi.misc.util import deprecated

import taichi as ti


# Scalar, basic data type
class Expr(TaichiOperations):
    def __init__(self, *args, tb=None):
        _taichi_skip_traceback = 1
        self.getter = None
        self.setter = None
        self.tb = tb
        if len(args) == 1:
            if isinstance(args[0], _ti_core.Expr):
                self.ptr = args[0]
            elif isinstance(args[0], Expr):
                self.ptr = args[0].ptr
                self.tb = args[0].tb
            elif is_taichi_class(args[0]):
                raise ValueError('cannot initialize scalar expression from '
                                 f'taichi class: {type(args[0])}')
            else:
                # assume to be constant
                arg = args[0]
                try:
                    import numpy as np
                    if isinstance(arg, np.ndarray):
                        arg = arg.dtype(arg)
                except:
                    pass
                self.ptr = impl.make_constant_expr(arg).ptr
        else:
            assert False
        if self.tb:
            self.ptr.set_tb(self.tb)
        self.grad = None
        self.val = self

    @python_scope
    def __setitem__(self, key, value):
        impl.get_runtime().materialize()
        self.initialize_accessor()
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key, )
        assert len(key) == len(self.shape)
        key = key + ((0, ) * (_ti_core.get_max_num_indices() - len(key)))
        self.setter(value, *key)

    @python_scope
    def __getitem__(self, key):
        impl.get_runtime().materialize()
        self.initialize_accessor()
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key, )
        key = key + ((0, ) * (_ti_core.get_max_num_indices() - len(key)))
        return self.getter(*key)

    def loop_range(self):
        return self

    def get_field_members(self):
        return [self]

    @deprecated('x.get_tensor_members()', 'x.get_field_members()')
    def get_tensor_members(self):
        return self.get_field_members()

    @python_scope
    def initialize_accessor(self):
        if self.getter:
            return
        snode = self.ptr.snode()

        if _ti_core.is_real(self.dtype):

            def getter(*key):
                assert len(key) == _ti_core.get_max_num_indices()
                return snode.read_float(key)

            def setter(value, *key):
                assert len(key) == _ti_core.get_max_num_indices()
                snode.write_float(key, value)
        else:
            if _ti_core.is_signed(self.dtype):

                def getter(*key):
                    assert len(key) == _ti_core.get_max_num_indices()
                    return snode.read_int(key)
            else:

                def getter(*key):
                    assert len(key) == _ti_core.get_max_num_indices()
                    return snode.read_uint(key)

            def setter(value, *key):
                assert len(key) == _ti_core.get_max_num_indices()
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
        from taichi.lang.meta import fill_tensor
        fill_tensor(self, val)

    def parent(self, n=1):
        p = self.snode.parent(n)
        return Expr(_ti_core.global_var_expr_from_snode(p.ptr))

    def is_global(self):
        return self.ptr.is_global_var() or self.ptr.is_external_var()

    @property
    def snode(self):
        from taichi.lang.snode import SNode
        return SNode(self.ptr.snode())

    def __hash__(self):
        return self.ptr.get_raw_address()

    @property
    def shape(self):
        if self.ptr.is_external_var():
            dim = impl.get_external_tensor_dim(self.ptr)
            ret = [
                Expr(impl.get_external_tensor_shape_along_axis(self.ptr, i))
                for i in range(dim)
            ]
            return ret
        return self.snode.shape

    @deprecated('x.dim()', 'len(x.shape)')
    def dim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self.snode.dtype

    @deprecated('x.data_type()', 'x.dtype')
    def data_type(self):
        return self.snode.dtype

    @python_scope
    def to_numpy(self):
        import numpy as np
        from taichi.lang.meta import tensor_to_ext_arr
        arr = np.zeros(shape=self.shape, dtype=to_numpy_type(self.dtype))
        tensor_to_ext_arr(self, arr)
        ti.sync()
        return arr

    @python_scope
    def to_torch(self, device=None):
        import torch
        from taichi.lang.meta import tensor_to_ext_arr
        arr = torch.zeros(size=self.shape,
                          dtype=to_pytorch_type(self.dtype),
                          device=device)
        tensor_to_ext_arr(self, arr)
        ti.sync()
        return arr

    @python_scope
    def from_numpy(self, arr):
        assert len(self.shape) == len(arr.shape)
        s = self.shape
        for i in range(len(self.shape)):
            assert s[i] == arr.shape[i]
        from taichi.lang.meta import ext_arr_to_tensor
        if hasattr(arr, 'contiguous'):
            arr = arr.contiguous()
        ext_arr_to_tensor(arr, self)
        ti.sync()

    @python_scope
    def from_torch(self, arr):
        self.from_numpy(arr.contiguous())

    @python_scope
    def copy_from(self, other):
        assert isinstance(other, Expr)
        from taichi.lang.meta import tensor_to_tensor
        assert len(self.shape) == len(other.shape)
        tensor_to_tensor(self, other)

    def __str__(self):
        """Python scope field print support."""
        if impl.inside_kernel():
            return '<ti.Expr>'  # make pybind11 happy, see Matrix.__str__
        else:
            return str(self.to_numpy())

    def __repr__(self):
        # make interactive shell happy, prevent materialization
        if self.is_global():
            # make interactive shell happy, prevent materialization
            return '<ti.field>'
        else:
            return '<ti.Expr>'


def make_var_vector(size):
    exprs = []
    for _ in range(size):
        exprs.append(_ti_core.make_id_expr(''))
    return ti.Vector(exprs)


def make_expr_group(*exprs):
    if len(exprs) == 1:
        if isinstance(exprs[0], (list, tuple)):
            exprs = exprs[0]
        elif isinstance(exprs[0], ti.Matrix):
            mat = exprs[0]
            assert mat.m == 1
            exprs = mat.entries
    expr_group = _ti_core.ExprGroup()
    for i in exprs:
        expr_group.push_back(Expr(i).ptr)
    return expr_group
