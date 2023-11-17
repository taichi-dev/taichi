import numbers
from types import FunctionType, MethodType
from typing import Any, Iterable, Sequence

import numpy as np
from taichi._lib import core as _ti_core
from taichi._snode.fields_builder import FieldsBuilder
from taichi.lang._ndarray import ScalarNdarray
from taichi.lang._ndrange import GroupedNDRange, _Ndrange
from taichi.lang._texture import RWTextureAccessor
from taichi.lang.any_array import AnyArray
from taichi.lang.enums import SNodeGradType
from taichi.lang.exception import (
    TaichiCompilationError,
    TaichiRuntimeError,
    TaichiSyntaxError,
    TaichiTypeError,
)
from taichi.lang.expr import Expr, make_expr_group
from taichi.lang.field import Field, ScalarField
from taichi.lang.kernel_arguments import SparseMatrixProxy
from taichi.lang.matrix import (
    Matrix,
    MatrixField,
    MatrixNdarray,
    MatrixType,
    Vector,
    VectorNdarray,
    make_matrix,
)
from taichi.lang.mesh import (
    ConvType,
    MeshElementFieldProxy,
    MeshInstance,
    MeshRelationAccessProxy,
    MeshReorderedMatrixFieldProxy,
    MeshReorderedScalarFieldProxy,
    element_type_name,
)
from taichi.lang.simt.block import SharedArray
from taichi.lang.snode import SNode
from taichi.lang.struct import Struct, StructField, _IntermediateStruct
from taichi.lang.util import (
    cook_dtype,
    get_traceback,
    is_taichi_class,
    python_scope,
    taichi_scope,
    warning,
)
from taichi.types.primitive_types import (
    all_types,
    f16,
    f32,
    f64,
    i32,
    i64,
    u8,
    u32,
    u64,
)


@taichi_scope
def expr_init_shared_array(shape, element_type):
    return (
        get_runtime()
        .compiling_callable.ast_builder()
        .expr_alloca_shared_array(shape, element_type, _ti_core.DebugInfo(get_runtime().get_current_src_info()))
    )


@taichi_scope
def expr_init(rhs):
    if rhs is None:
        return Expr(
            get_runtime()
            .compiling_callable.ast_builder()
            .expr_alloca(_ti_core.DebugInfo(get_runtime().get_current_src_info()))
        )
    if isinstance(rhs, Matrix) and (hasattr(rhs, "_DIM")):
        return Matrix(*rhs.to_list(), ndim=rhs.ndim)
    if isinstance(rhs, Matrix):
        return make_matrix(rhs.to_list())
    if isinstance(rhs, SharedArray):
        return rhs
    if isinstance(rhs, Struct):
        return Struct(rhs.to_dict(include_methods=True, include_ndim=True))
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
    if hasattr(rhs, "_data_oriented"):
        return rhs
    return Expr(
        get_runtime()
        .compiling_callable.ast_builder()
        .expr_var(Expr(rhs).ptr, _ti_core.DebugInfo(get_runtime().get_current_src_info()))
    )


@taichi_scope
def expr_init_func(rhs):  # temporary solution to allow passing in fields as arguments
    if isinstance(rhs, Field):
        return rhs
    return expr_init(rhs)


def begin_frontend_struct_for(ast_builder, group, loop_range):
    if not isinstance(loop_range, (AnyArray, Field, SNode, RWTextureAccessor, _Root)):
        raise TypeError(
            f"Cannot loop over the object {type(loop_range)} in Taichi scope. Only Taichi fields (via template) or dense arrays (via types.ndarray) are supported."
        )
    if group.size() != len(loop_range.shape):
        raise IndexError(
            "Number of struct-for indices does not match loop variable dimensionality "
            f"({group.size()} != {len(loop_range.shape)}). Maybe you wanted to "
            'use "for I in ti.grouped(x)" to group all indices into a single vector I?'
        )
    dbg_info = _ti_core.DebugInfo(get_runtime().get_current_src_info())
    if isinstance(loop_range, (AnyArray, RWTextureAccessor)):
        ast_builder.begin_frontend_struct_for_on_external_tensor(group, loop_range._loop_range(), dbg_info)
    else:
        ast_builder.begin_frontend_struct_for_on_snode(group, loop_range._loop_range(), dbg_info)


def begin_frontend_if(ast_builder, cond, stmt_dbg_info):
    assert ast_builder is not None
    if is_taichi_class(cond):
        raise ValueError(
            "The truth value of vectors/matrices is ambiguous.\n"
            "Consider using `any` or `all` when comparing vectors/matrices:\n"
            "    if all(x == y):\n"
            "or\n"
            "    if any(x != y):\n"
        )
    ast_builder.begin_frontend_if(Expr(cond).ptr, stmt_dbg_info)


@taichi_scope
def _calc_slice(index, default_stop):
    start, stop, step = index.start or 0, index.stop or default_stop, index.step or 1

    def check_validity(x):
        #  TODO(mzmzm): support variable in slice
        if isinstance(x, Expr):
            raise TaichiCompilationError(
                "Taichi does not support variables in slice now, please use constant instead of it."
            )

    check_validity(start), check_validity(stop), check_validity(step)
    return [_ for _ in range(start, stop, step)]


def validate_subscript_index(value, index):
    if isinstance(value, Field):
        # field supports negative indices
        return

    if isinstance(index, Expr):
        return

    if isinstance(index, Iterable):
        for ind in index:
            validate_subscript_index(value, ind)

    if isinstance(index, slice):
        validate_subscript_index(value, index.start)
        validate_subscript_index(value, index.stop)

    if isinstance(index, numbers.Number) and index < 0:
        raise TaichiSyntaxError("Negative indices are not supported in Taichi kernels.")


@taichi_scope
def subscript(ast_builder, value, *_indices, skip_reordered=False):
    dbg_info = _ti_core.DebugInfo(get_runtime().get_current_src_info())
    ast_builder = get_runtime().compiling_callable.ast_builder()
    # Directly evaluate in Python for non-Taichi types
    if not isinstance(
        value,
        (
            Expr,
            Field,
            AnyArray,
            SparseMatrixProxy,
            MeshElementFieldProxy,
            MeshRelationAccessProxy,
            SharedArray,
        ),
    ):
        if len(_indices) == 1:
            _indices = _indices[0]
        return value.__getitem__(_indices)

    has_slice = False

    flattened_indices = []
    for _index in _indices:
        if isinstance(_index, Matrix):
            ind = _index.to_list()
        elif isinstance(_index, slice):
            ind = [_index]
            has_slice = True
        else:
            ind = [_index]
        flattened_indices += ind
    indices = tuple(flattened_indices)
    validate_subscript_index(value, indices)

    if len(indices) == 1 and indices[0] is None:
        indices = ()

    if has_slice:
        if not (isinstance(value, Expr) and value.is_tensor()):
            raise TaichiSyntaxError(f"The type {type(value)} do not support index of slice type")
    else:
        indices_expr_group = make_expr_group(*indices)

    if isinstance(value, SharedArray):
        return value.subscript(*indices)
    if isinstance(value, MeshElementFieldProxy):
        return value.subscript(*indices)
    if isinstance(value, MeshRelationAccessProxy):
        return value.subscript(*indices)
    if isinstance(value, (MeshReorderedScalarFieldProxy, MeshReorderedMatrixFieldProxy)) and not skip_reordered:
        reordered_index = tuple(
            [
                Expr(
                    ast_builder.mesh_index_conversion(
                        value.mesh_ptr, value.element_type, Expr(indices[0]).ptr, ConvType.g2r, dbg_info
                    )
                )
            ]
        )
        return subscript(ast_builder, value, *reordered_index, skip_reordered=True)
    if isinstance(value, SparseMatrixProxy):
        return value.subscript(*indices)
    if isinstance(value, Field):
        _var = value._get_field_members()[0].ptr
        snode = _var.snode()
        if snode is None:
            if _var.is_primal():
                raise RuntimeError(f"{_var.get_expr_name()} has not been placed.")
            else:
                raise RuntimeError(
                    f"Gradient {_var.get_expr_name()} has not been placed, check whether `needs_grad=True`"
                )

        if isinstance(value, MatrixField):
            return Expr(ast_builder.expr_subscript(value.ptr, indices_expr_group, dbg_info))
        if isinstance(value, StructField):
            entries = {k: subscript(ast_builder, v, *indices) for k, v in value._items}
            entries["__struct_methods"] = value.struct_methods
            return _IntermediateStruct(entries)
        return Expr(ast_builder.expr_subscript(_var, indices_expr_group, dbg_info))
    if isinstance(value, AnyArray):
        return Expr(ast_builder.expr_subscript(value.ptr, indices_expr_group, dbg_info))
    assert isinstance(value, Expr)
    # Index into TensorType
    # value: IndexExpression with ret_type = TensorType
    assert value.is_tensor()

    if has_slice:
        shape = value.get_shape()
        dim = len(shape)
        assert dim == len(indices)
        indices = [
            _calc_slice(index, shape[i]) if isinstance(index, slice) else index for i, index in enumerate(indices)
        ]
        if dim == 1:
            assert isinstance(indices[0], list)
            multiple_indices = [make_expr_group(i) for i in indices[0]]
            return_shape = (len(indices[0]),)
        else:
            assert dim == 2
            if isinstance(indices[0], list) and isinstance(indices[1], list):
                multiple_indices = [make_expr_group(i, j) for i in indices[0] for j in indices[1]]
                return_shape = (len(indices[0]), len(indices[1]))
            elif isinstance(indices[0], list):  # indices[1] is not list
                multiple_indices = [make_expr_group(i, indices[1]) for i in indices[0]]
                return_shape = (len(indices[0]),)
            else:  # indices[0] is not list while indices[1] is list
                multiple_indices = [make_expr_group(indices[0], j) for j in indices[1]]
                return_shape = (len(indices[1]),)
        return Expr(
            _ti_core.subscript_with_multiple_indices(
                value.ptr,
                multiple_indices,
                return_shape,
                dbg_info,
            )
        )
    return Expr(ast_builder.expr_subscript(value.ptr, indices_expr_group, dbg_info))


class SrcInfoGuard:
    def __init__(self, info_stack, info):
        self.info_stack = info_stack
        self.info = info

    def __enter__(self):
        self.info_stack.append(self.info)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.info_stack.pop()


class PyTaichi:
    def __init__(self, kernels=None):
        self.materialized = False
        self.prog = None
        self.src_info_stack = []
        self.inside_kernel = False
        self.compiling_callable = None  # pointer to instance of lang::Kernel/Function
        self.current_kernel = None
        self.global_vars = []
        self.grad_vars = []
        self.dual_vars = []
        self.matrix_fields = []
        self.default_fp = f32
        self.default_ip = i32
        self.default_up = u32
        self.target_tape = None
        self.fwd_mode_manager = None
        self.grad_replaced = False
        self.kernels = kernels or []
        self._signal_handler_registry = None
        self.unfinalized_fields_builder = {}

    def initialize_fields_builder(self, builder):
        self.unfinalized_fields_builder[builder] = get_traceback(2)

    def clear_compiled_functions(self):
        for k in self.kernels:
            k.compiled_kernels.clear()

    def finalize_fields_builder(self, builder):
        self.unfinalized_fields_builder.pop(builder)

    def validate_fields_builder(self):
        for builder, tb in self.unfinalized_fields_builder.items():
            if builder == _root_fb:
                continue

            raise TaichiRuntimeError(
                f"Field builder {builder} is not finalized. " f"Please call finalize() on it. Traceback:\n{tb}"
            )

    def get_num_compiled_functions(self):
        count = 0
        for k in self.kernels:
            count += len(k.compiled_kernels)
        return count

    def src_info_guard(self, info):
        return SrcInfoGuard(self.src_info_stack, info)

    def get_current_src_info(self):
        return self.src_info_stack[-1]

    def set_default_fp(self, fp):
        assert fp in [f16, f32, f64]
        self.default_fp = fp
        default_cfg().default_fp = self.default_fp

    def set_default_ip(self, ip):
        assert ip in [i32, i64]
        self.default_ip = ip
        self.default_up = u32 if ip == i32 else u64
        default_cfg().default_ip = self.default_ip
        default_cfg().default_up = self.default_up

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

        if get_runtime().prog.config().debug:
            if not root.finalized:
                root._allocate_adjoint_checkbit()

        root.finalize(raise_warning=not is_first_call)
        global _root_fb
        _root_fb = FieldsBuilder()

    @staticmethod
    def _finalize_root_fb_for_aot():
        if _root_fb.finalized:
            raise RuntimeError("AOT: can only finalize the root FieldsBuilder once")
        _root_fb._finalize_for_aot()

    @staticmethod
    def _get_tb(_var):
        return getattr(_var, "declaration_tb", str(_var.ptr))

    def _check_field_not_placed(self):
        not_placed = []
        for _var in self.global_vars:
            if _var.ptr.snode() is None:
                not_placed.append(self._get_tb(_var))

        if len(not_placed):
            bar = "=" * 44 + "\n"
            raise RuntimeError(
                f"These field(s) are not placed:\n{bar}"
                + f"{bar}".join(not_placed)
                + f"{bar}Please consider specifying a shape for them. E.g.,"
                + "\n\n  x = ti.field(float, shape=(2, 3))"
            )

    def _check_gradient_field_not_placed(self, gradient_type):
        not_placed = set()
        gradient_vars = []
        if gradient_type == "grad":
            gradient_vars = self.grad_vars
        elif gradient_type == "dual":
            gradient_vars = self.dual_vars
        for _var in gradient_vars:
            if _var.ptr.snode() is None:
                not_placed.add(self._get_tb(_var))

        if len(not_placed):
            bar = "=" * 44 + "\n"
            raise RuntimeError(
                f"These field(s) requrie `needs_{gradient_type}=True`, however their {gradient_type} field(s) are not placed:\n{bar}"
                + f"{bar}".join(not_placed)
                + f"{bar}Please consider place the {gradient_type} field(s). E.g.,"
                + "\n\n  ti.root.dense(ti.i, 1).place(x.{gradient_type})"
                + "\n\n Or specify a shape for the field(s). E.g.,"
                + "\n\n  x = ti.field(float, shape=(2, 3), needs_{gradient_type}=True)"
            )

    def _check_matrix_field_member_shape(self):
        for _field in self.matrix_fields:
            shapes = [_field.get_scalar_field(i, j).shape for i in range(_field.n) for j in range(_field.m)]
            if any(shape != shapes[0] for shape in shapes):
                raise RuntimeError(
                    "Members of the following field have different shapes "
                    + f"{shapes}:\n{self._get_tb(_field._get_field_members()[0])}"
                )

    def _calc_matrix_field_dynamic_index_stride(self):
        for _field in self.matrix_fields:
            _field._calc_dynamic_index_stride()

    def materialize(self):
        self.materialize_root_fb(not self.materialized)
        self.materialized = True

        self.validate_fields_builder()

        self._check_field_not_placed()
        self._check_gradient_field_not_placed("grad")
        self._check_gradient_field_not_placed("dual")
        self._check_matrix_field_member_shape()
        self._calc_matrix_field_dynamic_index_stride()
        self.global_vars = []
        self.grad_vars = []
        self.dual_vars = []
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
    """The print function in Taichi scope.

    This function is called at compile time and has no runtime overhead.
    """
    __p(*args, **kwargs)


# we don't add @taichi_scope decorator for @ti.pyfunc to work
def static_assert(cond, msg=None):
    """Throw AssertionError when `cond` is False.

    This function is called at compile time and has no runtime overhead.
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
    if isinstance(cond, Expr):
        raise TaichiTypeError("Static assert with non-static condition")
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
        if item == "__qualname__":
            # For sphinx docstring extraction.
            return "_UninitializedRootFieldsBuilder"
        raise TaichiRuntimeError("Please call init() first")


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
# `_root_fb` will be overridden inside :func:`taichi.lang.init`.
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
        warning("""'ti.root.deactivate_all()' would deactivate all finalized snodes.""")
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
        return "ti.root"


root = _Root()
"""Root of the declared Taichi :func:`~taichi.lang.impl.field`s.

See also https://docs.taichi-lang.org/docs/layout

Example::

    >>> x = ti.field(ti.f32)
    >>> ti.root.pointer(ti.ij, 4).dense(ti.ij, 8).place(x)
"""


def _create_snode(axis_seq: Sequence[int], shape_seq: Sequence[numbers.Number], same_level: bool):
    dim = len(axis_seq)
    assert dim == len(shape_seq)
    snode = root
    if same_level:
        snode = snode.dense(axes(*axis_seq), shape_seq)
    else:
        for i in range(dim):
            snode = snode.dense(axes(axis_seq[i]), (shape_seq[i],))
    return snode


@python_scope
def create_field_member(dtype, name, needs_grad, needs_dual):
    dtype = cook_dtype(dtype)

    # primal
    prog = get_runtime().prog
    if prog is None:
        raise TaichiRuntimeError("Cannont create field, maybe you forgot to call `ti.init()` first?")

    x = Expr(prog.make_id_expr(""))
    x.declaration_tb = get_traceback(stacklevel=4)
    x.ptr = _ti_core.expr_field(x.ptr, dtype)
    x.ptr.set_name(name)
    x.ptr.set_grad_type(SNodeGradType.PRIMAL)
    pytaichi.global_vars.append(x)

    x_grad = None
    x_dual = None
    # The x_grad_checkbit is used for global data access rule checker
    x_grad_checkbit = None
    if _ti_core.is_real(dtype):
        # adjoint
        x_grad = Expr(get_runtime().prog.make_id_expr(""))
        x_grad.declaration_tb = get_traceback(stacklevel=4)
        x_grad.ptr = _ti_core.expr_field(x_grad.ptr, dtype)
        x_grad.ptr.set_name(name + ".grad")
        x_grad.ptr.set_grad_type(SNodeGradType.ADJOINT)
        x.ptr.set_adjoint(x_grad.ptr)
        if needs_grad:
            pytaichi.grad_vars.append(x_grad)

        if prog.config().debug:
            # adjoint checkbit
            x_grad_checkbit = Expr(get_runtime().prog.make_id_expr(""))
            dtype = u8
            if prog.config().arch in (_ti_core.opengl, _ti_core.vulkan, _ti_core.gles):
                dtype = i32
            x_grad_checkbit.ptr = _ti_core.expr_field(x_grad_checkbit.ptr, cook_dtype(dtype))
            x_grad_checkbit.ptr.set_name(name + ".grad_checkbit")
            x_grad_checkbit.ptr.set_grad_type(SNodeGradType.ADJOINT_CHECKBIT)
            x.ptr.set_adjoint_checkbit(x_grad_checkbit.ptr)

        # dual
        x_dual = Expr(get_runtime().prog.make_id_expr(""))
        x_dual.ptr = _ti_core.expr_field(x_dual.ptr, dtype)
        x_dual.ptr.set_name(name + ".dual")
        x_dual.ptr.set_grad_type(SNodeGradType.DUAL)
        x.ptr.set_dual(x_dual.ptr)
        if needs_dual:
            pytaichi.dual_vars.append(x_dual)
    elif needs_grad or needs_dual:
        raise TaichiRuntimeError(f"{dtype} is not supported for field with `needs_grad=True` or `needs_dual=True`.")

    return x, x_grad, x_dual


@python_scope
def _field(
    dtype,
    shape=None,
    order=None,
    name="",
    offset=None,
    needs_grad=False,
    needs_dual=False,
):
    x, x_grad, x_dual = create_field_member(dtype, name, needs_grad, needs_dual)
    x = ScalarField(x)
    if x_grad:
        x_grad = ScalarField(x_grad)
        x._set_grad(x_grad)
    if x_dual:
        x_dual = ScalarField(x_dual)
        x._set_dual(x_dual)

    if shape is None:
        if offset is not None:
            raise TaichiSyntaxError("shape cannot be None when offset is set")
        if order is not None:
            raise TaichiSyntaxError("shape cannot be None when order is set")
    else:
        if isinstance(shape, numbers.Number):
            shape = (shape,)
        if isinstance(offset, numbers.Number):
            offset = (offset,)
        dim = len(shape)
        if offset is not None and dim != len(offset):
            raise TaichiSyntaxError(f"The dimensionality of shape and offset must be the same ({dim} != {len(offset)})")
        axis_seq = []
        shape_seq = []
        if order is not None:
            if dim != len(order):
                raise TaichiSyntaxError(
                    f"The dimensionality of shape and order must be the same ({dim} != {len(order)})"
                )
            if dim != len(set(order)):
                raise TaichiSyntaxError("The axes in order must be different")
            for ch in order:
                axis = ord(ch) - ord("i")
                if axis < 0 or axis >= dim:
                    raise TaichiSyntaxError(f"Invalid axis {ch}")
                axis_seq.append(axis)
                shape_seq.append(shape[axis])
        else:
            axis_seq = list(range(dim))
            shape_seq = list(shape)
        same_level = order is None
        _create_snode(axis_seq, shape_seq, same_level).place(x, offset=offset)
        if needs_grad:
            _create_snode(axis_seq, shape_seq, same_level).place(x_grad, offset=offset)
        if needs_dual:
            _create_snode(axis_seq, shape_seq, same_level).place(x_dual, offset=offset)
    return x


@python_scope
def field(dtype, *args, **kwargs):
    """Defines a Taichi field.

    A Taichi field can be viewed as an abstract N-dimensional array, hiding away
    the complexity of how its underlying :class:`~taichi.lang.snode.SNode` are
    actually defined. The data in a Taichi field can be directly accessed by
    a Taichi :func:`~taichi.lang.kernel_impl.kernel`.

    See also https://docs.taichi-lang.org/docs/field

    Args:
        dtype (DataType): data type of the field. Note it can be vector or matrix types as well.
        shape (Union[int, tuple[int]], optional): shape of the field.
        order (str, optional): order of the shape laid out in memory.
        name (str, optional): name of the field.
        offset (Union[int, tuple[int]], optional): offset of the field domain.
        needs_grad (bool, optional): whether this field participates in autodiff (reverse mode)
            and thus needs an adjoint field to store the gradients.
        needs_dual (bool, optional): whether this field participates in autodiff (forward mode)
            and thus needs an dual field to store the gradients.

    Example::

        The code below shows how a Taichi field can be declared and defined::

            >>> x1 = ti.field(ti.f32, shape=(16, 8))
            >>> # Equivalently
            >>> x2 = ti.field(ti.f32)
            >>> ti.root.dense(ti.ij, shape=(16, 8)).place(x2)
            >>>
            >>> x3 = ti.field(ti.f32, shape=(16, 8), order='ji')
            >>> # Equivalently
            >>> x4 = ti.field(ti.f32)
            >>> ti.root.dense(ti.j, shape=8).dense(ti.i, shape=16).place(x4)
            >>>
            >>> x5 = ti.field(ti.math.vec3, shape=(16, 8))

    """
    if isinstance(dtype, MatrixType):
        if dtype.ndim == 1:
            return Vector.field(dtype.n, dtype.dtype, *args, **kwargs)
        return Matrix.field(dtype.n, dtype.m, dtype.dtype, *args, **kwargs)
    return _field(dtype, *args, **kwargs)


@python_scope
def ndarray(dtype, shape, needs_grad=False):
    """Defines a Taichi ndarray with scalar elements.

    Args:
        dtype (Union[DataType, MatrixType]): Data type of each element. This can be either a scalar type like ti.f32 or a compound type like ti.types.vector(3, ti.i32).
        shape (Union[int, tuple[int]]): Shape of the ndarray.

    Example:
        The code below shows how a Taichi ndarray with scalar elements can be declared and defined::

            >>> x = ti.ndarray(ti.f32, shape=(16, 8))  # ndarray of shape (16, 8), each element is ti.f32 scalar.
            >>> vec3 = ti.types.vector(3, ti.i32)
            >>> y = ti.ndarray(vec3, shape=(10, 2))  # ndarray of shape (10, 2), each element is a vector of 3 ti.i32 scalars.
            >>> matrix_ty = ti.types.matrix(3, 4, float)
            >>> z = ti.ndarray(matrix_ty, shape=(4, 5))  # ndarray of shape (4, 5), each element is a matrix of (3, 4) ti.float scalars.
    """
    # primal
    prog = get_runtime().prog
    if prog is None:
        raise TaichiRuntimeError("Cannont create ndarray, maybe you forgot to call `ti.init()` first?")

    if isinstance(shape, numbers.Number):
        shape = (shape,)
    if not all((isinstance(x, int) or isinstance(x, np.integer)) and x > 0 and x <= 2**31 - 1 for x in shape):
        raise TaichiRuntimeError(f"{shape} is not a valid shape for ndarray")
    if dtype in all_types:
        dt = cook_dtype(dtype)
        x = ScalarNdarray(dt, shape)
    elif isinstance(dtype, MatrixType):
        if dtype.ndim == 1:
            x = VectorNdarray(dtype.n, dtype.dtype, shape)
        else:
            x = MatrixNdarray(dtype.n, dtype.m, dtype.dtype, shape)
        dt = dtype.dtype
    else:
        raise TaichiRuntimeError(f"{dtype} is not supported as ndarray element type")
    if needs_grad:
        if not _ti_core.is_real(dt):
            raise TaichiRuntimeError(f"{dt} is not supported for ndarray with `needs_grad=True` or `needs_dual=True`.")
        x_grad = ndarray(dtype, shape, needs_grad=False)
        x._set_grad(x_grad)
    return x


@taichi_scope
def ti_format_list_to_content_entries(raw):
    # return a pair of [content, format]
    def entry2content(_var):
        if isinstance(_var, str):
            return [_var, None]
        if isinstance(_var, list):
            assert len(_var) == 2 and (isinstance(_var[1], str) or _var[1] is None)
            _var[0] = Expr(_var[0]).ptr
            return _var
        return [Expr(_var).ptr, None]

    def list_ti_repr(_var):
        yield "["  # distinguishing tuple & list will increase maintenance cost
        for i, v in enumerate(_var):
            if i:
                yield ", "
            yield v
        yield "]"

    def vars2entries(_vars):
        for _var in _vars:
            # If the first element is '__ti_fmt_value__', this list is an Expr and its format.
            if isinstance(_var, list) and len(_var) == 3 and isinstance(_var[0], str) and _var[0] == "__ti_fmt_value__":
                # yield [Expr, format] as a whole and don't pass it to vars2entries() again
                yield _var[1:]
                continue
            elif hasattr(_var, "__ti_repr__"):
                res = _var.__ti_repr__()
            elif isinstance(_var, (list, tuple)):
                # If the first element is '__ti_format__', this list is the result of ti_format.
                if len(_var) > 0 and isinstance(_var[0], str) and _var[0] == "__ti_format__":
                    res = _var[1:]
                else:
                    res = list_ti_repr(_var)
            else:
                yield _var
                continue

            for v in vars2entries(res):
                yield v

    def fused_string(entries):
        accumated = ""
        for entry in entries:
            if isinstance(entry, str):
                accumated += entry
            else:
                if accumated:
                    yield accumated
                    accumated = ""
                yield entry
        if accumated:
            yield accumated

    def extract_formats(entries):
        contents, formats = zip(*entries)
        return list(contents), list(formats)

    entries = vars2entries(raw)
    entries = fused_string(entries)
    entries = [entry2content(entry) for entry in entries]
    return extract_formats(entries)


@taichi_scope
def ti_print(*_vars, sep=" ", end="\n"):
    def add_separators(_vars):
        for i, _var in enumerate(_vars):
            if i:
                yield sep
            yield _var
        yield end

    _vars = add_separators(_vars)
    contents, formats = ti_format_list_to_content_entries(_vars)
    get_runtime().compiling_callable.ast_builder().create_print(
        contents, formats, _ti_core.DebugInfo(get_runtime().get_current_src_info())
    )


@taichi_scope
def ti_format(*args):
    content = args[0]
    mixed = args[1:]
    new_mixed = []
    args = []
    for x in mixed:
        # x is a (formatted) Expr
        if isinstance(x, Expr) or (isinstance(x, list) and len(x) == 3 and x[0] == "__ti_fmt_value__"):
            new_mixed.append("{}")
            args.append(x)
        else:
            new_mixed.append(x)
    content = content.format(*new_mixed)
    res = content.split("{}")
    assert len(res) == len(args) + 1, "Number of args is different from number of positions provided in string"

    for i, arg in enumerate(args):
        res.insert(i * 2 + 1, arg)
    res.insert(0, "__ti_format__")
    return res


@taichi_scope
def ti_assert(cond, msg, extra_args, dbg_info):
    # Mostly a wrapper to help us convert from Expr (defined in Python) to
    # _ti_core.Expr (defined in C++)
    get_runtime().compiling_callable.ast_builder().create_assert_stmt(Expr(cond).ptr, msg, extra_args, dbg_info)


@taichi_scope
def ti_int(_var):
    if hasattr(_var, "__ti_int__"):
        return _var.__ti_int__()
    return int(_var)


@taichi_scope
def ti_bool(_var):
    if hasattr(_var, "__ti_bool__"):
        return _var.__ti_bool__()
    return bool(_var)


@taichi_scope
def ti_float(_var):
    if hasattr(_var, "__ti_float__"):
        return _var.__ti_float__()
    return float(_var)


@taichi_scope
def zero(x):
    # TODO: get dtype from Expr and Matrix:
    """Returns an array of zeros with the same shape and type as the input. It's also a scalar
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
    """Returns an array of ones with the same shape and type as the input. It's also a scalar
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


def static(x, *xs) -> Any:
    """Evaluates a Taichi-scope expression at compile time.

    `static()` is what enables the so-called metaprogramming in Taichi. It is
    in many ways similar to ``constexpr`` in C++.

    See also https://docs.taichi-lang.org/docs/meta.

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

    if (
        isinstance(
            x,
            (
                bool,
                int,
                float,
                range,
                list,
                tuple,
                enumerate,
                GroupedNDRange,
                _Ndrange,
                zip,
                filter,
                map,
            ),
        )
        or x is None
    ):
        return x
    if isinstance(x, (np.bool_, np.integer, np.floating)):
        return x

    if isinstance(x, AnyArray):
        return x
    if isinstance(x, Field):
        return x
    if isinstance(x, (FunctionType, MethodType)):
        return x
    raise ValueError(f"Input to ti.static must be compile-time constants or global pointers, instead of {type(x)}")


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
        >>> for I in ti.grouped(ti.ndrange(2, 3)):
        >>>     print(I)
        prints [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]
    """
    if isinstance(x, _Ndrange):
        return x.grouped()
    return x


def stop_grad(x):
    """Stops computing gradients during back propagation.

    Args:
        x (:class:`~taichi.Field`): A field.
    """
    get_runtime().compiling_callable.ast_builder().stop_grad(x.snode.ptr)


def current_cfg():
    return get_runtime().prog.config()


def default_cfg():
    return _ti_core.default_compile_config()


def call_internal(name, *args, with_runtime_context=True):
    return expr_init(_ti_core.insert_internal_func_call(getattr(_ti_core.InternalOp, name), make_expr_group(args)))


def get_cuda_compute_capability():
    return _ti_core.query_int64("cuda_compute_capability")


@taichi_scope
def mesh_relation_access(mesh, from_index, to_element_type):
    # to support ti.mesh_local and access mesh attribute as field
    if isinstance(from_index, MeshInstance):
        return getattr(from_index, element_type_name(to_element_type))
    if isinstance(mesh, MeshInstance):
        return MeshRelationAccessProxy(mesh, from_index, to_element_type)
    raise RuntimeError("Relation access should be with a mesh instance!")


__all__ = [
    "axes",
    "deactivate_all_snodes",
    "field",
    "grouped",
    "ndarray",
    "one",
    "root",
    "static",
    "static_assert",
    "static_print",
    "stop_grad",
    "zero",
]
