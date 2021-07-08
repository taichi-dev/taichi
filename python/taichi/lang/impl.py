import numbers
import types
import warnings

import numpy as np
from taichi.core.util import ti_core as _ti_core
from taichi.lang.exception import InvalidOperationError, TaichiSyntaxError
from taichi.lang.expr import Expr, make_expr_group
from taichi.lang.snode import SNode
from taichi.lang.tape import TapeImpl
from taichi.lang.util import (cook_dtype, is_taichi_class, python_scope,
                              taichi_scope)
from taichi.misc.util import deprecated, get_traceback, warning
from taichi.snode.fields_builder import FieldsBuilder

import taichi as ti


@taichi_scope
def expr_init(rhs):
    if rhs is None:
        return Expr(_ti_core.expr_alloca())
    if is_taichi_class(rhs):
        return rhs.variable()
    else:
        if isinstance(rhs, list):
            return [expr_init(e) for e in rhs]
        elif isinstance(rhs, tuple):
            return tuple(expr_init(e) for e in rhs)
        elif isinstance(rhs, dict):
            return dict((key, expr_init(val)) for key, val in rhs.items())
        elif isinstance(rhs, _ti_core.DataType):
            return rhs
        elif isinstance(rhs, _ti_core.Arch):
            return rhs
        elif isinstance(rhs, ti.ndrange):
            return rhs
        elif hasattr(rhs, '_data_oriented'):
            return rhs
        else:
            return Expr(_ti_core.expr_var(Expr(rhs).ptr))


@taichi_scope
def expr_init_list(xs, expected):
    if not isinstance(xs, (list, tuple, ti.Matrix)):
        raise TypeError(f'Cannot unpack type: {type(xs)}')
    if isinstance(xs, ti.Matrix):
        if not xs.m == 1:
            raise ValueError(
                f'Matrices with more than one columns cannot be unpacked')
        xs = xs.entries
    if expected != len(xs):
        raise ValueError(
            f'Tuple assignment size mismatch: {expected} != {len(xs)}')
    if isinstance(xs, list):
        return [expr_init(e) for e in xs]
    elif isinstance(xs, tuple):
        return tuple(expr_init(e) for e in xs)
    else:
        raise ValueError(f'Cannot unpack from {type(xs)}')


@taichi_scope
def expr_init_func(
        rhs):  # temporary solution to allow passing in fields as arguments
    if isinstance(rhs, Expr) and rhs.ptr.is_global_var():
        return rhs
    if isinstance(rhs, ti.Matrix) and rhs.is_global():
        return rhs
    return expr_init(rhs)


def begin_frontend_struct_for(group, loop_range):
    if not isinstance(loop_range, Expr) or not loop_range.is_global():
        raise TypeError('Can only iterate through global variables/fields')
    if group.size() != len(loop_range.shape):
        raise IndexError(
            'Number of struct-for indices does not match loop variable dimensionality '
            f'({group.size()} != {len(loop_range.shape)}). Maybe you wanted to '
            'use "for I in ti.grouped(x)" to group all indices into a single vector I?'
        )
    _ti_core.begin_frontend_struct_for(group, loop_range.ptr)


def begin_frontend_if(cond):
    if is_taichi_class(cond):
        raise ValueError(
            'The truth value of vectors/matrices is ambiguous.\n'
            'Consider using `any` or `all` when comparing vectors/matrices:\n'
            '    if all(x == y):\n'
            'or\n'
            '    if any(x != y):\n')
    _ti_core.begin_frontend_if(Expr(cond).ptr)


def wrap_scalar(x):
    if type(x) in [int, float]:
        return Expr(x)
    else:
        return x


@taichi_scope
def subscript(value, *indices):
    _taichi_skip_traceback = 1
    if isinstance(value, np.ndarray):
        return value.__getitem__(*indices)

    if isinstance(value, (tuple, list, dict)):
        assert len(indices) == 1
        return value[indices[0]]

    flattened_indices = []
    for i in range(len(indices)):
        if is_taichi_class(indices[i]):
            ind = indices[i].entries
        else:
            ind = [indices[i]]
        flattened_indices += ind
    indices = tuple(flattened_indices)

    if is_taichi_class(value):
        return value.subscript(*indices)
    elif isinstance(value, (Expr, SNode)):
        if isinstance(value, Expr):
            if not value.is_global():
                raise TypeError(
                    'Subscription (e.g., "a[i, j]") only works on fields or external arrays.'
                )
            field_dim = int(value.ptr.get_attribute("dim"))
        else:
            # When reading bit structure we only support the 0-D case for now.
            field_dim = 0
        if isinstance(indices,
                      tuple) and len(indices) == 1 and indices[0] is None:
            indices = []
        indices_expr_group = make_expr_group(*indices)
        index_dim = indices_expr_group.size()
        if field_dim != index_dim:
            raise IndexError(
                f'Field with dim {field_dim} accessed with indices of dim {index_dim}'
            )
        return Expr(_ti_core.subscript(value.ptr, indices_expr_group))
    else:
        return value[indices]


@taichi_scope
def chain_compare(comparators, ops):
    _taichi_skip_traceback = 1
    assert len(comparators) == len(ops) + 1, \
      f'Chain comparison invoked with {len(comparators)} comparators but {len(ops)} operators'
    ret = True
    for i in range(len(ops)):
        lhs = comparators[i]
        rhs = comparators[i + 1]
        if ops[i] == 'Lt':
            now = lhs < rhs
        elif ops[i] == 'LtE':
            now = lhs <= rhs
        elif ops[i] == 'Gt':
            now = lhs > rhs
        elif ops[i] == 'GtE':
            now = lhs >= rhs
        elif ops[i] == 'Eq':
            now = lhs == rhs
        elif ops[i] == 'NotEq':
            now = lhs != rhs
        else:
            assert False, f'Unknown operator {ops[i]}'
        ret = ti.logical_and(ret, now)
    return ret


@taichi_scope
def maybe_transform_ti_func_call_to_stmt(ti_func, *args, **kwargs):
    _taichi_skip_traceback = 1
    if '_sitebuiltins' == getattr(ti_func, '__module__', '') and getattr(
            getattr(ti_func, '__class__', ''), '__name__', '') == 'Quitter':
        raise TaichiSyntaxError(f'exit or quit not supported in Taichi-scope')
    if getattr(ti_func, '__module__',
               '') == '__main__' and not getattr(ti_func, '__wrapped__', ''):
        warnings.warn(
            f'Calling into non-Taichi function {ti_func.__name__}.'
            ' This means that scope inside that function will not be processed'
            ' by the Taichi transformer. Proceed with caution! '
            ' Maybe you want to decorate it with @ti.func?',
            UserWarning,
            stacklevel=2)

    is_taichi_function = getattr(ti_func, '_is_taichi_function', False)
    # If is_taichi_function is true: call a decorated Taichi function
    # in a Taichi kernel/function.

    if is_taichi_function and get_runtime().experimental_real_function:
        # Compiles the function here.
        # Invokes Func.__call__.
        func_call_result = ti_func(*args, **kwargs)
        return _ti_core.insert_expr_stmt(func_call_result.ptr)
    else:
        return ti_func(*args, **kwargs)


class PyTaichi:
    def __init__(self, kernels=None):
        self.materialized = False
        self.prog = None
        self.materialize_callbacks = []
        self.compiled_functions = {}
        self.compiled_grad_functions = {}
        self.scope_stack = []
        self.inside_kernel = False
        self.current_kernel = None
        self.global_vars = []
        self.print_preprocessed = False
        self.experimental_real_function = False
        self.default_fp = ti.f32
        self.default_ip = ti.i32
        self.target_tape = None
        self.inside_complex_kernel = False
        self.kernels = kernels or []

    def get_num_compiled_functions(self):
        return len(self.compiled_functions) + len(self.compiled_grad_functions)

    def set_default_fp(self, fp):
        assert fp in [ti.f32, ti.f64]
        self.default_fp = fp
        default_cfg().default_fp = self.default_fp

    def set_default_ip(self, ip):
        assert ip in [ti.i32, ti.i64]
        self.default_ip = ip
        default_cfg().default_ip = self.default_ip

    def create_program(self):
        if self.prog is None:
            self.prog = _ti_core.Program()

    def materialize_root_fb(self, first):
        if (not root.finalized and not root.empty) or first:
            root.finalize()

        if root.finalized:
            global _root_fb
            _root_fb = FieldsBuilder()

    def materialize(self):
        self.materialize_root_fb(not self.materialized)

        if self.materialized:
            return

        self.materialized = True
        not_placed = []
        for var in self.global_vars:
            if var.ptr.snode() is None:
                tb = getattr(var, 'declaration_tb', str(var.ptr))
                not_placed.append(tb)

        if len(not_placed):
            bar = '=' * 44 + '\n'
            raise RuntimeError(
                f'These field(s) are not placed:\n{bar}' +
                f'{bar}'.join(not_placed) +
                f'{bar}Please consider specifying a shape for them. E.g.,' +
                '\n\n  x = ti.field(float, shape=(2, 3))')

        for callback in self.materialize_callbacks:
            callback()
        self.materialize_callbacks = []

    def clear(self):
        if self.prog:
            self.prog.finalize()
            self.prog = None
        self.materialized = False

    def get_tape(self, loss=None):
        return TapeImpl(self, loss)

    def sync(self):
        self.materialize()
        self.prog.synchronize()


pytaichi = PyTaichi()


def get_runtime():
    return pytaichi


def materialize_callback(foo):
    get_runtime().materialize_callbacks.append(foo)


def _clamp_unsigned_to_range(npty, val):
    # npty: np.int32 or np.int64
    iif = np.iinfo(npty)
    if iif.min <= val <= iif.max:
        return val
    cap = (1 << iif.bits)
    if not (0 <= val < cap):
        # We let pybind11 fail intentionally, because this isn't the case we want
        # to deal with: |val| does't fall into the valid range of either
        # the signed or the unsigned type.
        return val
    new_val = val - cap
    ti.warn(
        f'Constant {val} has exceeded the range of {iif.bits} int, clamped to {new_val}'
    )
    return new_val


@taichi_scope
def make_constant_expr(val):
    _taichi_skip_traceback = 1
    if isinstance(val, (int, np.integer)):
        if pytaichi.default_ip in {ti.i32, ti.u32}:
            # It is not always correct to do such clamp without the type info on
            # the LHS, but at least this makes assigning constant to unsigned
            # int work. See https://github.com/taichi-dev/taichi/issues/2060
            return Expr(
                _ti_core.make_const_expr_i32(
                    _clamp_unsigned_to_range(np.int32, val)))
        elif pytaichi.default_ip in {ti.i64, ti.u64}:
            return Expr(
                _ti_core.make_const_expr_i64(
                    _clamp_unsigned_to_range(np.int64, val)))
        else:
            assert False
    elif isinstance(val, (float, np.floating, np.ndarray)):
        if pytaichi.default_fp == ti.f32:
            return Expr(_ti_core.make_const_expr_f32(val))
        elif pytaichi.default_fp == ti.f64:
            return Expr(_ti_core.make_const_expr_f64(val))
        else:
            assert False
    else:
        raise ValueError(f'Invalid constant scalar expression: {type(val)}')


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
    _taichi_skip_traceback = 1
    if msg is not None:
        assert cond, msg
    else:
        assert cond


def inside_kernel():
    return pytaichi.inside_kernel


def index_nd(dim):
    return indices(*range(dim))


class _UninitializedRootFieldsBuilder:
    def __getattr__(self, item):
        raise InvalidOperationError('Please call init() first')


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


class _Root:
    """Wrapper around the default root FieldsBuilder instance."""
    def parent(self, n=1):
        """Same as :func:`taichi.SNode.parent`"""
        return _root_fb.root.parent(n)

    def loop_range(self, n=1):
        """Same as :func:`taichi.SNode.loop_range`"""
        return _root_fb.root.loop_range()

    def get_children(self):
        """Same as :func:`taichi.SNode.get_children`"""
        return _root_fb.root.get_children()

    @property
    def id(self):
        return _root_fb.root.id

    def __getattr__(self, item):
        return getattr(_root_fb, item)

    def __repr__(self):
        return 'ti.root'


root = _Root()


@deprecated('ti.var', 'ti.field')
def var(dt, shape=None, offset=None, needs_grad=False):
    _taichi_skip_traceback = 1
    return field(dt, shape, offset, needs_grad)


@python_scope
def field(dtype, shape=None, offset=None, needs_grad=False):
    _taichi_skip_traceback = 1

    dtype = cook_dtype(dtype)

    if isinstance(shape, numbers.Number):
        shape = (shape, )

    if isinstance(offset, numbers.Number):
        offset = (offset, )

    if shape is not None and offset is not None:
        assert len(shape) == len(
            offset
        ), f'The dimensionality of shape and offset must be the same  ({len(shape)} != {len(offset)})'

    assert (offset is not None and shape is None
            ) == False, f'The shape cannot be None when offset is being set'

    del _taichi_skip_traceback

    # primal
    x = Expr(_ti_core.make_id_expr(""))
    x.declaration_tb = get_traceback(stacklevel=2)
    x.ptr = _ti_core.global_new(x.ptr, dtype)
    x.ptr.set_is_primal(True)
    pytaichi.global_vars.append(x)

    if _ti_core.needs_grad(dtype):
        # adjoint
        x_grad = Expr(_ti_core.make_id_expr(""))
        x_grad.ptr = _ti_core.global_new(x_grad.ptr, dtype)
        x_grad.ptr.set_is_primal(False)
        x.set_grad(x_grad)

    if shape is not None:
        dim = len(shape)
        root.dense(index_nd(dim), shape).place(x, offset=offset)
        if needs_grad:
            root.dense(index_nd(dim), shape).place(x.grad)
    return x


class Layout:
    def __init__(self, soa=False):
        self.soa = soa


SOA = Layout(soa=True)
AOS = Layout(soa=False)


@python_scope
def layout(func):
    raise InvalidOperationError('layout(): Deprecated')


@taichi_scope
def ti_print(*vars, sep=' ', end='\n'):
    def entry2content(var):
        if isinstance(var, str):
            return var
        else:
            return Expr(var).ptr

    def list_ti_repr(var):
        yield '['  # distinguishing tuple & list will increase maintainance cost
        for i, v in enumerate(var):
            if i:
                yield ', '
            yield v
        yield ']'

    def vars2entries(vars):
        for var in vars:
            if hasattr(var, '__ti_repr__'):
                res = var.__ti_repr__()
            elif isinstance(var, (list, tuple)):
                res = list_ti_repr(var)
            else:
                yield var
                continue

            for v in vars2entries(res):
                yield v

    def add_separators(vars):
        for i, var in enumerate(vars):
            if i:
                yield sep
            yield var
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

    vars = add_separators(vars)
    entries = vars2entries(vars)
    entries = fused_string(entries)
    contentries = [entry2content(entry) for entry in entries]
    _ti_core.create_print(contentries)


@taichi_scope
def ti_assert(cond, msg, extra_args):
    # Mostly a wrapper to help us convert from Expr (defined in Python) to
    # _ti_core.Expr (defined in C++)
    _ti_core.create_assert_stmt(
        Expr(cond).ptr, msg, [Expr(x).ptr for x in extra_args])


@taichi_scope
def ti_int(var):
    _taichi_skip_traceback = 1
    if hasattr(var, '__ti_int__'):
        return var.__ti_int__()
    else:
        return int(var)


@taichi_scope
def ti_float(var):
    _taichi_skip_traceback = 1
    if hasattr(var, '__ti_float__'):
        return var.__ti_float__()
    else:
        return float(var)


@taichi_scope
def zero(x):
    # TODO: get dtype from Expr and Matrix:
    return x * 0


@taichi_scope
def one(x):
    return zero(x) + 1


@taichi_scope
def get_external_tensor_dim(var):
    return _ti_core.get_external_tensor_dim(var)


@taichi_scope
def get_external_tensor_shape_along_axis(var, i):
    return _ti_core.get_external_tensor_shape_along_axis(var, i)


def indices(*x):
    return [_ti_core.Index(i) for i in x]


index = indices


def static(x, *xs):
    _taichi_skip_traceback = 1
    if len(xs):  # for python-ish pointer assign: x, y = ti.static(y, x)
        return [static(x)] + [static(x) for x in xs]

    if isinstance(x,
                  (bool, int, float, range, list, tuple, enumerate, ti.ndrange,
                   ti.GroupedNDRange, zip, filter, map)) or x is None:
        return x
    elif isinstance(x, (Expr, ti.Matrix)) and x.is_global():
        return x
    elif isinstance(x, (types.FunctionType, types.MethodType)):
        return x
    else:
        raise ValueError(
            f'Input to ti.static must be compile-time constants or global pointers, instead of {type(x)}'
        )


@taichi_scope
def grouped(x):
    if isinstance(x, ti.ndrange):
        return x.grouped()
    else:
        return x


def stop_grad(x):
    _ti_core.stop_grad(x.snode.ptr)


def current_cfg():
    return _ti_core.current_compile_config()


def default_cfg():
    return _ti_core.default_compile_config()


def call_internal(name):
    _ti_core.create_internal_func_stmt(name)
