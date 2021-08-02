from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl
from taichi.lang.common_ops import TaichiOperations
from taichi.lang.util import (is_taichi_class, python_scope, to_numpy_type,
                              to_pytorch_type)

import taichi as ti


# Scalar, basic data type
class Expr(TaichiOperations):
    """A Python-side Expr wrapper, whose member variable `ptr` is an instance of C++ Expr class. A C++ Expr object contains member variable `expr` which holds an instance of C++ Expression class."""
    def __init__(self, *args, tb=None):
        _taichi_skip_traceback = 1
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
    def set_grad(self, grad):
        self.grad = grad
        self.ptr.set_grad(grad.ptr)

    @python_scope
    def fill(self, val):
        """Fill the whole field with value `val` when the class itself represents GlobalVariableExpression (field) or ExternalTensorExpression internally.

        This is an unified interface to match :func:`taichi.lang.Matrix.fill`.

        Args:
            val (Union[int, float]): value to fill
        """
        # TODO: avoid too many template instantiations
        from taichi.lang.meta import fill_tensor
        fill_tensor(self, val)



    def is_global(self):
        """Check whether the class itself represents GlobalVariableExpression (field) or ExternalTensorExpression internally.

        Returns:
            True or False depending on whether the class itself represents GlobalVariableExpression (field) or ExternalTensorExpression internally.
        """
        return self.ptr.is_global_var() or self.ptr.is_external_var()

    def __hash__(self):
        return self.ptr.get_raw_address()

    @property
    def name(self):
        return self.snode.name

    @property
    def shape(self):
        """A list containing sizes for each dimension when the class itself represents GlobalVariableExpression (field) or ExternalTensorExpression internally.

        Returns:
            The list containing sizes for each dimension when the class itself represents GlobalVariableExpression (field) or ExternalTensorExpression internally.
        """
        if self.ptr.is_external_var():
            dim = impl.get_external_tensor_dim(self.ptr)
            ret = [
                Expr(impl.get_external_tensor_shape_along_axis(self.ptr, i))
                for i in range(dim)
            ]
            return ret
        return self.snode.shape

    @python_scope
    def to_numpy(self):
        """Create a numpy array containing the same elements when the class itself represents GlobalVariableExpression (field) or ExternalTensorExpression internally.

        This is an unified interface to match :func:`taichi.lang.Matrix.to_numpy`.

        Returns:
            The numpy array containing the same elements when the class itself represents GlobalVariableExpression (field) or ExternalTensorExpression internally.
        """
        import numpy as np
        from taichi.lang.meta import tensor_to_ext_arr
        arr = np.zeros(shape=self.shape, dtype=to_numpy_type(self.dtype))
        tensor_to_ext_arr(self, arr)
        ti.sync()
        return arr

    @python_scope
    def to_torch(self, device=None):
        """Create a torch array containing the same elements when the class itself represents GlobalVariableExpression (field) or ExternalTensorExpression internally.

        This is an unified interface to match :func:`taichi.lang.Matrix.to_torch`.

        Args:
            device (DeviceType): The device type as a parameter passed into torch.zeros().

        Returns:
            The torch array containing the same elements when the class itself represents GlobalVariableExpression (field) or ExternalTensorExpression internally.
        """
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
        """Load all elements from a numpy array when the class itself represents GlobalVariableExpression (field) or ExternalTensorExpression internally.

        This is an unified interface to match :func:`taichi.lang.Matrix.from_numpy`.
        The numpy array's shape need to be the same as the internal data structure.

        Args:
            arr (NumpyArray): The numpy array containing the elements to load.
        """
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
        """Load all elements from a torch array when the class itself represents GlobalVariableExpression (field) or ExternalTensorExpression internally.

        This is an unified interface to match :func:`taichi.lang.Matrix.from_torch`.
        The torch array's shape need to be the same as the internal data structure.

        Args:
            arr (TorchArray): The torch array containing the elements to load.
        """
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
