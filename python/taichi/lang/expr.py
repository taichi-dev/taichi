from .core import taichi_lang_core
from .util import *
from . import impl
from .common_ops import TaichiOperations
import traceback


# Scalar, basic data type
class Expr(TaichiOperations):
    def __init__(self, *args, tb=None):
        _taichi_skip_traceback = 1
        self.getter = None
        self.setter = None
        self.tb = tb
        if len(args) == 1:
            if isinstance(args[0], taichi_lang_core.Expr):
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

    def __setitem__(self, key, value):
        assert not impl.inside_kernel()
        self.initialize_accessor()
        if key is None:
            key = ()
        elif not isinstance(key, (tuple, list)):
            key = (key, )
        assert len(key) == len(
            self.shape
        ), f"Field of dim {len(self.shape)} accessed with indices of dim {len(key)}"
        key = key + (0, ) * (taichi_lang_core.get_max_num_indices() - len(key))
        self.setter(key, value)

    def __getitem__(self, key):
        assert not impl.inside_kernel()
        self.initialize_accessor()
        if key is None:
            key = ()
        elif not isinstance(key, (tuple, list)):
            key = (key, )
        assert len(key) == len(
            self.shape
        ), f"Field of dim {len(self.shape)} accessed with indices of dim {len(key)}"
        key = key + (0, ) * (taichi_lang_core.get_max_num_indices() - len(key))
        return self.getter(key)

    def loop_range(self):
        return self

    def get_field_members(self):
        return [self]

    @deprecated('x.get_tensor_members()', 'x.get_field_members()')
    def get_tensor_members(self):
        return self.get_field_members()

    @python_scope
    def initialize_accessor(self):
        impl.get_runtime().materialize()
        snode = self.snode.ptr

        if not taichi_lang_core.is_integral(self.dtype):
            self.getter = snode.read_float
            self.setter = snode.write_float

        else:
            if taichi_lang_core.is_signed(self.dtype):
                self.getter = snode.read_int
            else:
                self.getter = snode.read_uint
            self.setter = snode.write_int

        self.initialize_accessor = lambda: None

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

    #@deprecated('tensor.parent()', 'tensor.snode.parent()')
    def parent(self, n=1):
        import taichi as ti
        p = self.snode.parent(n)
        return Expr(ti.core.global_var_expr_from_snode(p.ptr))

    def is_global(self):
        return self.ptr.is_global_var() or self.ptr.is_external_var()

    @property
    def snode(self):
        from .snode import SNode
        return SNode(self.ptr.snode())

    def __hash__(self):
        return self.ptr.get_raw_address()

    @property
    # The invocation into snode.shape is relatively slow.
    # We need to cache it for better performance in SNode accessors, where
    # `assert len(key) == len(self.shape)` is used.
    # Discussion: https://github.com/taichi-dev/taichi/pull/1719#issuecomment-675284385
    @cached_property
    def shape(self):
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
        from .meta import tensor_to_ext_arr
        import numpy as np
        arr = np.zeros(shape=self.shape, dtype=to_numpy_type(self.dtype))
        tensor_to_ext_arr(self, arr)
        import taichi as ti
        ti.sync()
        return arr

    @python_scope
    def to_torch(self, device=None):
        from .meta import tensor_to_ext_arr
        import torch
        arr = torch.zeros(size=self.shape,
                          dtype=to_pytorch_type(self.dtype),
                          device=device)
        tensor_to_ext_arr(self, arr)
        import taichi as ti
        ti.sync()
        return arr

    @python_scope
    def from_numpy(self, arr):
        assert len(self.shape) == len(arr.shape)
        s = self.shape
        for i in range(len(self.shape)):
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
        assert len(self.shape) == len(other.shape)
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
