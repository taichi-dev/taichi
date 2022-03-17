import numbers
from types import FunctionType, MethodType
from typing import Iterable

import numpy as np
from taichi._lib import core as _ti_core
from taichi._snode.fields_builder import FieldsBuilder
from taichi.lang._ndarray import ScalarNdarray
from taichi.lang._ndrange import GroupedNDRange, _Ndrange
from taichi.lang.any_array import AnyArray, AnyArrayAccess
from taichi.lang.exception import TaichiRuntimeError
from taichi.lang.expr import Expr, make_expr_group
from taichi.lang.field import Field, ScalarField
from taichi.lang.kernel_arguments import SparseMatrixProxy
from taichi.lang.matrix import (Matrix, MatrixField, _IntermediateMatrix,
                                _MatrixFieldElement)
from taichi.lang.mesh import (ConvType, MeshElementFieldProxy, MeshInstance,
                              MeshRelationAccessProxy,
                              MeshReorderedMatrixFieldProxy,
                              MeshReorderedScalarFieldProxy, element_type_name)
from taichi.lang.snode import SNode
from taichi.lang.struct import Struct, StructField, _IntermediateStruct
from taichi.lang.tape import TapeImpl
from taichi.lang.util import (cook_dtype, get_traceback, is_taichi_class,
                              python_scope, taichi_scope, warning)
from taichi.types.primitive_types import f16, f32, f64, i32, i64


@taichi_scope
def expr_init_local_tensor(shape, element_type, elements):
    return get_runtime().prog.current_ast_builder().expr_alloca_local_tensor(
        shape, element_type, elements)


@taichi_scope
def expr_init(rhs):
    if rhs is None:
        return Expr(get_runtime().prog.current_ast_builder().expr_alloca())
    if isinstance(rhs, Matrix):
        return Matrix(rhs.to_list())
    if isinstance(rhs, Struct):
        return Struct(rhs.to_dict())
    if isinstance(rhs, list):
        return [expr_init(e) for e in rhs]
    if isinstance(rhs, tuple):
        return tuple(expr_init(e) for e in rhs)
    if isinstance(rhs, dict):
        return dict((key, expr_init(val)) for key, val in rhs.items())
    if isinstance(rhs, _ti_core.DataType):
        return rhs
    if isinstance(rhs, _ti_core.Arch):
        return rhs
    if isinstance(rhs, _Ndrange):
        return rhs
    if isinstance(rhs, MeshElementFieldProxy):
        return rhs
    if isinstance(rhs, MeshRelationAccessProxy):
        return rhs
    if hasattr(rhs, '_data_oriented'):
        return rhs
    return Expr(get_runtime().prog.current_ast_builder().expr_var(
        Expr(rhs).ptr))


@taichi_scope
def expr_init_list(xs, expected):
    if not isinstance(xs, (list, tuple, Matrix)):
        raise TypeError(f'Cannot unpack type: {type(xs)}')
    if isinstance(xs, Matrix):
        if not xs.m == 1:
            raise ValueError(
                'Matrices with more than one columns cannot be unpacked')
        xs = xs.entries
    if expected != len(xs):
        raise ValueError(
            f'Tuple assignment size mismatch: {expected} != {len(xs)}')
    if isinstance(xs, list):
        return [expr_init(e) for e in xs]
    if isinstance(xs, tuple):
        return tuple(expr_init(e) for e in xs)
    raise ValueError(f'Cannot unpack from {type(xs)}')


@taichi_scope
def expr_init_func(
        rhs):  # temporary solution to allow passing in fields as arguments
    if isinstance(rhs, Field):
        return rhs
    return expr_init(rhs)


def begin_frontend_struct_for(ast_builder, group, loop_range):
    if not isinstance(loop_range, (AnyArray, Field, SNode, _Root)):
        raise TypeError(
            'Can only iterate through Taichi fields/snodes (via template) or dense arrays (via any_arr)'
        )
    if group.size() != len(loop_range.shape):
        raise IndexError(
            'Number of struct-for indices does not match loop variable dimensionality '
            f'({group.size()} != {len(loop_range.shape)}). Maybe you wanted to '
            'use "for I in ti.grouped(x)" to group all indices into a single vector I?'
        )
    ast_builder.begin_frontend_struct_for(group, loop_range._loop_range())


def begin_frontend_if(ast_builder, cond):
    assert ast_builder is not None
    if is_taichi_class(cond):
        raise ValueError(
            'The truth value of vectors/matrices is ambiguous.\n'
            'Consider using `any` or `all` when comparing vectors/matrices:\n'
            '    if all(x == y):\n'
            'or\n'
            '    if any(x != y):\n')
    ast_builder.begin_frontend_if(Expr(cond).ptr)


@taichi_scope
def subscript(value, *_indices, skip_reordered=False):
    if isinstance(value, np.ndarray):
        return value.__getitem__(_indices)

    if isinstance(value, (tuple, list, dict)):
        assert len(_indices) == 1
        return value[_indices[0]]

    has_slice = False
    flattened_indices = []
    for _index in _indices:
        if is_taichi_class(_index):
            ind = _index.entries
        elif isinstance(_index, slice):
            ind = [_index]
            has_slice = True
        else:
            ind = [_index]
        flattened_indices += ind
    _indices = tuple(flattened_indices)
    if isinstance(_indices,
                  tuple) and len(_indices) == 1 and _indices[0] is None:
        _indices = ()

    if has_slice:
        if not isinstance(value, Matrix):
            raise SyntaxError(
                f"The type {type(value)} do not support index of slice type")
    else:
        indices_expr_group = make_expr_group(*_indices)
        index_dim = indices_expr_group.size()

    if is_taichi_class(value):
        return value._subscript(*_indices)
    if isinstance(value, MeshElementFieldProxy):
        return value.subscript(*_indices)
    if isinstance(value, MeshRelationAccessProxy):
        return value.subscript(*_indices)
    if isinstance(value,
                  (MeshReorderedScalarFieldProxy,
                   MeshReorderedMatrixFieldProxy)) and not skip_reordered:
        assert index_dim == 1
        reordered_index = tuple([
            Expr(
                _ti_core.get_index_conversion(value.mesh_ptr,
                                              value.element_type,
                                              Expr(_indices[0]).ptr,
                                              ConvType.g2r))
        ])
        return subscript(value, *reordered_index, skip_reordered=True)
    if isinstance(value, SparseMatrixProxy):
        return value.subscript(*_indices)
    if isinstance(value, Field):
        _var = value._get_field_members()[0].ptr
        if _var.snode() is None:
            if _var.is_primal():
                raise RuntimeError(
                    f"{_var.get_expr_name()} has not been placed.")
            else:
                raise RuntimeError(
                    f"Gradient {_var.get_expr_name()} has not been placed, check whether `needs_grad=True`"
                )
        field_dim = int(_var.get_attribute("dim"))
        if field_dim != index_dim:
            raise IndexError(
                f'Field with dim {field_dim} accessed with indices of dim {index_dim}'
            )
        if isinstance(value, MatrixField):
            return _MatrixFieldElement(value, indices_expr_group)
        if isinstance(value, StructField):
            return _IntermediateStruct(
                {k: subscript(v, *_indices)
                 for k, v in value._items})
        return Expr(_ti_core.subscript(_var, indices_expr_group))
    if isinstance(value, AnyArray):
        # TODO: deprecate using get_attribute to get dim
        field_dim = int(value.ptr.get_attribute("dim"))
        element_dim = len(value.element_shape)
        if field_dim != index_dim + element_dim:
            raise IndexError(
                f'Field with dim {field_dim - element_dim} accessed with indices of dim {index_dim}'
            )
        if element_dim == 0:
            return Expr(_ti_core.subscript(value.ptr, indices_expr_group))
        n = value.element_shape[0]
        m = 1 if element_dim == 1 else value.element_shape[1]
        any_array_access = AnyArrayAccess(value, _indices)
        ret = _IntermediateMatrix(n, m, [
            any_array_access.subscript(i, j) for i in range(n)
            for j in range(m)
        ])
        ret.any_array_access = any_array_access
        return ret
    if isinstance(value, SNode):
        # When reading bit structure we only support the 0-D case for now.
        field_dim = 0
        if field_dim != index_dim:
            raise IndexError(
                f'Field with dim {field_dim} accessed with indices of dim {index_dim}'
            )
        return Expr(_ti_core.subscript(value.ptr, indices_expr_group))
    # Directly evaluate in Python for non-Taichi types
    return value.__getitem__(*_indices)


@taichi_scope
def make_tensor_element_expr(_var, _indices, shape, stride):
    return Expr(
        _ti_core.make_tensor_element_expr(_var, make_expr_group(*_indices),
                                          shape, stride))


class PyTaichi:
    def __init__(self, kernels=None):
        self.materialized = False
        self.prog = None
        self.compiled_functions = {}
        self.compiled_grad_functions = {}
        self.scope_stack = []
        self.inside_kernel = False
        self.current_kernel = None
        self.global_vars = []
        self.matrix_fields = []
        self.default_fp = f32
        self.default_ip = i32
        self.target_tape = None
        self.grad_replaced = False
        self.kernels = kernels or []
        self._signal_handler_registry = None

    def get_num_compiled_functions(self):
        return len(self.compiled_functions) + len(self.compiled_grad_functions)

    def set_default_fp(self, fp):
        assert fp in [f16, f32, f64]
        self.default_fp = fp
        default_cfg().default_fp = self.default_fp

    def set_default_ip(self, ip):
        assert ip in [i32, i64]
        self.default_ip = ip
        default_cfg().default_ip = self.default_ip

    def create_program(self):
        if self.prog is None:
            self.prog = _ti_core.Program()

    @staticmethod
    def materialize_root_fb(is_first_call):
        if root.finalized:
            return
        if not is_first_call and root.empty:
            # We have to forcefully finalize when `is_first_call` is True (even
            # if the root itself is empty), so that there is a valid struct
            # llvm::Module, if no field has been declared before the first kernel
            # invocation. Example case:
            # https://github.com/taichi-dev/taichi/blob/27bb1dc3227d9273a79fcb318fdb06fd053068f5/tests/python/test_ad_basics.py#L260-L266
            return
        root.finalize(raise_warning=not is_first_call)
        global _root_fb
        _root_fb = FieldsBuilder()

    @staticmethod
    def _finalize_root_fb_for_aot():
        if _root_fb.finalized:
            raise RuntimeError(
                'AOT: can only finalize the root FieldsBuilder once')
        _root_fb._finalize_for_aot()

    @staticmethod
    def _get_tb(_var):
        return getattr(_var, 'declaration_tb', str(_var.ptr))

    def _check_field_not_placed(self):
        not_placed = []
        for _var in self.global_vars:
            if _var.ptr.snode() is None:
                not_placed.append(self._get_tb(_var))

        if len(not_placed):
            bar = '=' * 44 + '\n'
            raise RuntimeError(
                f'These field(s) are not placed:\n{bar}' +
                f'{bar}'.join(not_placed) +
                f'{bar}Please consider specifying a shape for them. E.g.,' +
                '\n\n  x = ti.field(float, shape=(2, 3))')

    def _check_matrix_field_member_shape(self):
        for _field in self.matrix_fields:
            shapes = [
                _field.get_scalar_field(i, j).shape for i in range(_field.n)
                for j in range(_field.m)
            ]
            if any(shape != shapes[0] for shape in shapes):
                raise RuntimeError(
                    'Members of the following field have different shapes ' +
                    f'{shapes}:\n{self._get_tb(_field._get_field_members()[0])}'
                )

    def _calc_matrix_field_dynamic_index_stride(self):
        for _field in self.matrix_fields:
            _field._calc_dynamic_index_stride()

    def materialize(self):
        self.materialize_root_fb(not self.materialized)
        self.materialized = True

        self._check_field_not_placed()
        self._check_matrix_field_member_shape()
        self._calc_matrix_field_dynamic_index_stride()
        self.global_vars = []
        self.matrix_fields = []

    def _register_signal_handlers(self):
        if self._signal_handler_registry is None:
            self._signal_handler_registry = _ti_core.HackedSignalRegister()

    def clear(self):
        if self.prog:
            self.prog.finalize()
            self.prog = None
        self._signal_handler_registry = None
        self.materialized = False

    def get_tape(self, loss=None):
        return TapeImpl(self, loss)

    def sync(self):
        self.materialize()
        self.prog.synchronize()


pytaichi = PyTaichi()


def get_runtime():
    return pytaichi


def reset():
    global pytaichi
    old_kernels = pytaichi.kernels
    pytaichi.clear()
    pytaichi = PyTaichi(old_kernels)
    for k in old_kernels:
        k.reset()
    _ti_core.reset_default_compile_config()


@taichi_scope
def static_print(*args, __p=print, **kwargs):
    __p(*args, **kwargs)


# we don't add @taichi_scope decorator for @ti.pyfunc to work
def static_assert(cond, msg=None):
    """Throw AssertionError when `cond` is False.
    The bool value in `cond` must can be determined at compile time.

    Args:
        cond (bool): an expression with a bool value.
        msg (str): assertion message.

    Example::

        >>> year = 2001
        >>> @ti.kernel
        >>> def test():
        >>>     ti.static_assert(year % 4 == 0, "the year must be a lunar year")
        AssertionError: the year must be a lunar year
    """
    if msg is not None:
        assert cond, msg
    else:
        assert cond


def inside_kernel():
    return pytaichi.inside_kernel


def index_nd(dim):
    return axes(*range(dim))


class _UninitializedRootFieldsBuilder:
    def __getattr__(self, item):
        if item == '__qualname__':
            # For sphinx docstring extraction.
            return '_UninitializedRootFieldsBuilder'
        raise TaichiRuntimeError('Please call init() first')


# `root` initialization must be delayed until after the program is
# created. Unfortunately, `root` exists in both taichi.lang.impl module and
# the top-level taichi module at this point; so if `root` itself is written, we
# would have to make sure that `root` in all the modules get updated to the same
# instance. This is an error-prone process.
#
# To avoid this situation, we create `root` once during the import time, and
# never write to it. The core part, `_root_fb`, is the one whose initialization
# gets delayed. `_root_fb` will only exist in the taichi.lang.impl module, so
# writing to it is would result in less for maintenance cost.
#
# `_root_fb` will be overriden inside :func:`taichi.lang.init`.
_root_fb = _UninitializedRootFieldsBuilder()


def deactivate_all_snodes():
    """Recursively deactivate all SNodes."""
    for root_fb in FieldsBuilder._finalized_roots():
        root_fb.deactivate_all()


class _Root:
    """Wrapper around the default root FieldsBuilder instance."""
    @staticmethod
    def parent(n=1):
        """Same as :func:`taichi.SNode.parent`"""
        return _root_fb.root.parent(n)

    @staticmethod
    def _loop_range():
        """Same as :func:`taichi.SNode.loop_range`"""
        return _root_fb.root._loop_range()

    @staticmethod
    def _get_children():
        """Same as :func:`taichi.SNode.get_children`"""
        return _root_fb.root._get_children()

    # TODO: Record all of the SNodeTrees that finalized under 'ti.root'
    @staticmethod
    def deactivate_all():
        warning(
            """'ti.root.deactivate_all()' would deactivate all finalized snodes."""
        )
        deactivate_all_snodes()

    @property
    def shape(self):
        """Same as :func:`taichi.SNode.shape`"""
        return _root_fb.root.shape

    @property
    def _id(self):
        return _root_fb.root._id

    def __getattr__(self, item):
        return getattr(_root_fb, item)

    def __repr__(self):
        return 'ti.root'


root = _Root()
"""Root of the declared Taichi :func:`~taichi.lang.impl.field`s.

See also https://docs.taichi.graphics/lang/articles/advanced/layout

Example::

    >>> x = ti.field(ti.f32)
    >>> ti.root.pointer(ti.ij, 4).dense(ti.ij, 8).place(x)
"""


@python_scope
def create_field_member(dtype, name):
    dtype = cook_dtype(dtype)

    # primal
    x = Expr(_ti_core.make_id_expr(""))
    x.declaration_tb = get_traceback(stacklevel=4)
    x.ptr = _ti_core.global_new(x.ptr, dtype)
    x.ptr.set_name(name)
    x.ptr.set_is_primal(True)
    pytaichi.global_vars.append(x)

    x_grad = None
    if _ti_core.needs_grad(dtype):
        # adjoint
        x_grad = Expr(_ti_core.make_id_expr(""))
        x_grad.ptr = _ti_core.global_new(x_grad.ptr, dtype)
        x_grad.ptr.set_name(name + ".grad")
        x_grad.ptr.set_is_primal(False)
        x.ptr.set_grad(x_grad.ptr)

    return x, x_grad


@python_scope
def field(dtype, shape=None, name="", offset=None, needs_grad=False):
    """Defines a Taichi field.

    A Taichi field can be viewed as an abstract N-dimensional array, hiding away
    the complexity of how its underlying :class:`~taichi.lang.snode.SNode` are
    actually defined. The data in a Taichi field can be directly accessed by
    a Taichi :func:`~taichi.lang.kernel_impl.kernel`.

    See also https://docs.taichi.graphics/lang/articles/basic/field

    Args:
        dtype (DataType): data type of the field.
        shape (Union[int, tuple[int]], optional): shape of the field.
        name (str, optional): name of the field.
        offset (Union[int, tuple[int]], optional): offset of the field domain.
        needs_grad (bool, optional): whether this field participates in autodiff
            and thus needs an adjoint field to store the gradients.

    Example::

        The code below shows how a Taichi field can be declared and defined::

            >>> x1 = ti.field(ti.f32, shape=(16, 8))
            >>>
            >>> # Equivalently
            >>> x2 = ti.field(ti.f32)
            >>> ti.root.dense(ti.ij, shape=(16, 8)).place(x2)
    """

    if isinstance(shape, numbers.Number):
        shape = (shape, )

    if isinstance(offset, numbers.Number):
        offset = (offset, )

    if shape is not None and offset is not None:
        assert len(shape) == len(
            offset
        ), f'The dimensionality of shape and offset must be the same  ({len(shape)} != {len(offset)})'

    assert (offset is None or shape
            is not None), 'The shape cannot be None when offset is being set'

    x, x_grad = create_field_member(dtype, name)
    x, x_grad = ScalarField(x), ScalarField(x_grad)
    x._set_grad(x_grad)

    if shape is not None:
        dim = len(shape)
        root.dense(index_nd(dim), shape).place(x, offset=offset)
        if needs_grad:
            root.dense(index_nd(dim), shape).place(x_grad)
    return x


@python_scope
def ndarray(dtype, shape):
    """Defines a Taichi ndarray with scalar elements.

    Args:
        dtype (DataType): Data type of each value.
        shape (Union[int, tuple[int]]): Shape of the ndarray.

    Example:
        The code below shows how a Taichi ndarray with scalar elements can be declared and defined::

            >>> x = ti.ndarray(ti.f32, shape=(16, 8))
    """
    if isinstance(shape, numbers.Number):
        shape = (shape, )
    return ScalarNdarray(dtype, shape)


@taichi_scope
def ti_print(*_vars, sep=' ', end='\n'):
    def entry2content(_var):
        if isinstance(_var, str):
            return _var
        return Expr(_var).ptr

    def list_ti_repr(_var):
        yield '['  # distinguishing tuple & list will increase maintainance cost
        for i, v in enumerate(_var):
            if i:
                yield ', '
            yield v
        yield ']'

    def vars2entries(_vars):
        for _var in _vars:
            if hasattr(_var, '__ti_repr__'):
                res = _var.__ti_repr__()
            elif isinstance(_var, (list, tuple)):
                # If the first element is '__ti_format__', this list is the result of ti_format.
                if len(_var) > 0 and isinstance(
                        _var[0], str) and _var[0] == '__ti_format__':
                    res = _var[1:]
                else:
                    res = list_ti_repr(_var)
            else:
                yield _var
                continue

            for v in vars2entries(res):
                yield v

    def add_separators(_vars):
        for i, _var in enumerate(_vars):
            if i:
                yield sep
            yield _var
        yield end

    def fused_string(entries):
        accumated = ''
        for entry in entries:
            if isinstance(entry, str):
                accumated += entry
            else:
                if accumated:
                    yield accumated
                    accumated = ''
                yield entry
        if accumated:
            yield accumated

    _vars = add_separators(_vars)
    entries = vars2entries(_vars)
    entries = fused_string(entries)
    contentries = [entry2content(entry) for entry in entries]
    get_runtime().prog.current_ast_builder().create_print(contentries)


@taichi_scope
def ti_format(*args, **kwargs):
    content = args[0]
    mixed = args[1:]
    new_mixed = []
    new_mixed_kwargs = {}
    args = []
    for x in mixed:
        if isinstance(x, Expr):
            new_mixed.append('{}')
            args.append(x)
        else:
            new_mixed.append(x)
    for k, v in kwargs.items():
        if isinstance(v, Expr):
            new_mixed_kwargs[k] = '{}'
            args.append(v)
        else:
            new_mixed_kwargs[k] = v
    try:
        content = content.format(*new_mixed, **new_mixed_kwargs)
    except ValueError:
        print('Number formatting is not supported with Taichi fields')
        exit(1)
    res = content.split('{}')
    assert len(res) == len(
        args
    ) + 1, 'Number of args is different from number of positions provided in string'

    for i, arg in enumerate(args):
        res.insert(i * 2 + 1, arg)
    res.insert(0, '__ti_format__')
    return res


@taichi_scope
def ti_assert(cond, msg, extra_args):
    # Mostly a wrapper to help us convert from Expr (defined in Python) to
    # _ti_core.Expr (defined in C++)
    get_runtime().prog.current_ast_builder().create_assert_stmt(
        Expr(cond).ptr, msg, [Expr(x).ptr for x in extra_args])


@taichi_scope
def ti_int(_var):
    if hasattr(_var, '__ti_int__'):
        return _var.__ti_int__()
    return int(_var)


@taichi_scope
def ti_float(_var):
    if hasattr(_var, '__ti_float__'):
        return _var.__ti_float__()
    return float(_var)


@taichi_scope
def zero(x):
    # TODO: get dtype from Expr and Matrix:
    """Return an array of zeros with the same shape and type as the input. It's also a scalar
    if the input is a scalar.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): The input.

    Returns:
        A new copy of the input but filled with zeros.

    Example::

        >>> x = ti.Vector([1, 1])
        >>> @ti.kernel
        >>> def test():
        >>>     y = ti.zero(x)
        >>>     print(y)
        [0, 0]
    """
    return x * 0


@taichi_scope
def one(x):
    """Return an array of ones with the same shape and type as the input. It's also a scalar
    if the input is a scalar.

    Args:
        x (Union[:mod:`~taichi.types.primitive_types`, :class:`~taichi.Matrix`]): The input.

    Returns:
        A new copy of the input but filled with ones.

    Example::

        >>> x = ti.Vector([0, 0])
        >>> @ti.kernel
        >>> def test():
        >>>     y = ti.one(x)
        >>>     print(y)
        [1, 1]
    """
    return zero(x) + 1


def axes(*x: Iterable[int]):
    """Defines a list of axes to be used by a field.

    Args:
        *x: A list of axes to be activated

    Note that Taichi has already provided a set of commonly used axes. For example,
    `ti.ij` is just `axes(0, 1)` under the hood.
    """
    return [_ti_core.Axis(i) for i in x]


Axis = _ti_core.Axis


def static(x, *xs):
    """Evaluates a Taichi-scope expression at compile time.

    `static()` is what enables the so-called metaprogramming in Taichi. It is
    in many ways similar to ``constexpr`` in C++.

    See also https://docs.taichi.graphics/lang/articles/advanced/meta.

    Args:
        x (Any): an expression to be evaluated
        *xs (Any): for Python-ish swapping assignment

    Example:
        The most common usage of `static()` is for compile-time evaluation::

            >>> cond = False
            >>>
            >>> @ti.kernel
            >>> def run():
            >>>     if ti.static(cond):
            >>>         do_a()
            >>>     else:
            >>>         do_b()

        Depending on the value of ``cond``, ``run()`` will be directly compiled
        into either ``do_a()`` or ``do_b()``. Thus there won't be a runtime
        condition check.

        Another common usage is for compile-time loop unrolling::

            >>> @ti.kernel
            >>> def run():
            >>>     for i in ti.static(range(3)):
            >>>         print(i)
            >>>
            >>> # The above will be unrolled to:
            >>> @ti.kernel
            >>> def run():
            >>>     print(0)
            >>>     print(1)
            >>>     print(2)
    """
    if len(xs):  # for python-ish pointer assign: x, y = ti.static(y, x)
        return [static(x)] + [static(x) for x in xs]

    if isinstance(x,
                  (bool, int, float, range, list, tuple, enumerate, _Ndrange,
                   GroupedNDRange, zip, filter, map)) or x is None:
        return x
    if isinstance(x, AnyArray):
        return x
    if isinstance(x, Field):
        return x
    if isinstance(x, (FunctionType, MethodType)):
        return x
    raise ValueError(
        f'Input to ti.static must be compile-time constants or global pointers, instead of {type(x)}'
    )


@taichi_scope
def grouped(x):
    """Groups the indices in the iterator returned by `ndrange()` into a 1-D vector.

    This is often used when you want to iterate over all indices returned by `ndrange()`
    in one `for` loop and a single index.

    Args:
        x (:func:`~taichi.ndrange`): an iterator object returned by `ti.ndrange`.

    Example::
        >>> # without ti.grouped
        >>> for I in ti.ndrange(2, 3):
        >>>     print(I)
        prints 0, 1, 2, 3, 4, 5

        >>> # with ti.grouped
        >>> for I in ti.grouped(ndrange(2, 3)):
        >>>     print(I)
        prints [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]
    """
    if isinstance(x, _Ndrange):
        return x.grouped()
    return x


def stop_grad(x):
    get_runtime().prog.current_ast_builder().stop_grad(x.snode.ptr)


def current_cfg():
    return get_runtime().prog.config


def default_cfg():
    return _ti_core.default_compile_config()


def call_internal(name, *args):
    return expr_init(
        _ti_core.insert_internal_func_call(name, make_expr_group(args)))


@taichi_scope
def mesh_relation_access(mesh, from_index, to_element_type):
    # to support ti.mesh_local and access mesh attribute as field
    if isinstance(from_index, MeshInstance):
        return getattr(from_index, element_type_name(to_element_type))
    if isinstance(mesh, MeshInstance):
        return MeshRelationAccessProxy(mesh, from_index, to_element_type)
    raise RuntimeError("Relation access should be with a mesh instance!")


__all__ = [
    'axes', 'deactivate_all_snodes', 'field', 'grouped', 'ndarray', 'one',
    'root', 'static', 'static_assert', 'static_print', 'stop_grad', 'zero'
]
