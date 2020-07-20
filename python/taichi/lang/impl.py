import inspect
from .core import taichi_lang_core
from .expr import Expr
from .snode import SNode
from .util import *
from .exception import TaichiSyntaxError


@taichi_scope
def expr_init(rhs):
    import taichi as ti
    if rhs is None:
        return Expr(taichi_lang_core.expr_alloca())
    if is_taichi_class(rhs):
        return rhs.variable()
    else:
        if isinstance(rhs, list):
            return [expr_init(e) for e in rhs]
        elif isinstance(rhs, tuple):
            return tuple(expr_init(e) for e in rhs)
        elif isinstance(rhs, taichi_lang_core.DataType):
            return rhs
        elif isinstance(rhs, ti.ndrange):
            return rhs
        elif hasattr(rhs, '_data_oriented'):
            return rhs
        else:
            return Expr(taichi_lang_core.expr_var(Expr(rhs).ptr))


@taichi_scope
def expr_init_list(xs, expected):
    import taichi as ti
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
def expr_init_func(rhs):  # temporary solution to allow passing in tensors as
    import taichi as ti
    if isinstance(rhs, Expr) and rhs.ptr.is_global_var():
        return rhs
    if isinstance(rhs, ti.Matrix) and rhs.is_global():
        return rhs
    return expr_init(rhs)


def begin_frontend_struct_for(group, loop_range):
    if group.size() != len(loop_range.shape):
        raise IndexError(
            'Number of struct-for indices does not match loop variable dimensionality '
            f'({group.size()} != {len(loop_range.shape)}). Maybe you wanted to '
            'use "for I in ti.grouped(x)" to group all indices into a single vector I?'
        )
    taichi_lang_core.begin_frontend_struct_for(group, loop_range.ptr)


def wrap_scalar(x):
    if type(x) in [int, float]:
        return Expr(x)
    else:
        return x


@taichi_scope
def subscript(value, *indices):
    import numpy as np
    _taichi_skip_traceback = 1
    if isinstance(value, np.ndarray):
        return value.__getitem__(*indices)

    if isinstance(value, tuple) or isinstance(value, list):
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
    else:
        if isinstance(indices,
                      tuple) and len(indices) == 1 and indices[0] is None:
            indices = []
        indices_expr_group = make_expr_group(*indices)
        tensor_dim = int(value.ptr.get_attribute("dim"))
        index_dim = indices_expr_group.size()
        if tensor_dim != index_dim:
            raise IndexError(
                f'Tensor with dim {tensor_dim} accessed with indices of dim {index_dim}'
            )
        return Expr(taichi_lang_core.subscript(value.ptr, indices_expr_group))


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
        ret = logical_and(ret, now)
    return ret


@taichi_scope
def func_call_with_check(func, *args, **kwargs):
    _taichi_skip_traceback = 1
    if '_sitebuiltins' == getattr(func, '__module__', '') and getattr(
            getattr(func, '__class__', ''), '__name__', '') == 'Quitter':
        raise TaichiSyntaxError(f'exit or quit not supported in Taichi-scope')
    if getattr(func, '__module__',
               '') == '__main__' and not getattr(func, '__wrapped__', ''):
        import warnings
        warnings.warn(
            f'Calling into non-Taichi function {func.__name__}.'
            ' This means that scope inside that function will not be processed'
            ' by the Taichi transformer. Proceed with caution! '
            ' Maybe you want to decorate it with @ti.func?',
            UserWarning,
            stacklevel=2)

    return func(*args, **kwargs)


class PyTaichi:
    def __init__(self, kernels=None):
        self.materialized = False
        self.prog = None
        self.layout_functions = []
        self.compiled_functions = {}
        self.compiled_grad_functions = {}
        self.scope_stack = []
        self.inside_kernel = False
        self.global_vars = []
        self.print_preprocessed = False
        self.default_fp = f32
        self.default_ip = i32
        self.target_tape = None
        self.inside_complex_kernel = False
        self.kernels = kernels or []

    def get_num_compiled_functions(self):
        return len(self.compiled_functions) + len(self.compiled_grad_functions)

    def set_default_fp(self, fp):
        assert fp in [f32, f64]
        self.default_fp = fp
        default_cfg().default_fp = self.default_fp

    def set_default_ip(self, ip):
        assert ip in [i32, i64]
        self.default_ip = ip
        default_cfg().default_ip = self.default_ip

    def create_program(self):
        if self.prog is None:
            self.prog = taichi_lang_core.Program()

    def materialize(self):
        if self.materialized:
            return
        self.create_program()

        def layout():
            for func in self.layout_functions:
                func()

        import taichi as ti
        ti.trace("Materializing layout...".format())
        taichi_lang_core.layout(layout)
        self.materialized = True
        for var in self.global_vars:
            assert var.ptr.snode(
            ) is not None, 'Some variable(s) are not placed'

    def clear(self):
        if self.prog:
            self.prog.finalize()
            self.prog = None
        self.materialized = False

    def get_tape(self, loss=None):
        from .tape import Tape
        return Tape(self, loss)

    def sync(self):
        self.materialize()
        self.prog.synchronize()
        # print's in kernel won't take effect until ti.sync(), discussion:
        # https://github.com/taichi-dev/taichi/pull/1303#discussion_r444897102
        print(taichi_lang_core.pop_python_print_buffer(), end='')


pytaichi = PyTaichi()


def get_runtime():
    return pytaichi


@taichi_scope
def make_constant_expr(val):
    import numpy as np
    _taichi_skip_traceback = 1
    if isinstance(val, (int, np.integer)):
        if pytaichi.default_ip == i32:
            return Expr(taichi_lang_core.make_const_expr_i32(val))
        elif pytaichi.default_ip == i64:
            return Expr(taichi_lang_core.make_const_expr_i64(val))
        else:
            assert False
    elif isinstance(val, (float, np.floating, np.ndarray)):
        if pytaichi.default_fp == f32:
            return Expr(taichi_lang_core.make_const_expr_f32(val))
        elif pytaichi.default_fp == f64:
            return Expr(taichi_lang_core.make_const_expr_f64(val))
        else:
            assert False
    else:
        raise ValueError(f'Bad constant scalar expression: {type(val)}')


def reset():
    global pytaichi
    old_kernels = pytaichi.kernels
    pytaichi.clear()
    pytaichi = PyTaichi(old_kernels)
    for k in old_kernels:
        k.reset()
    taichi_lang_core.reset_default_compile_config()


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


class Root:
    def __init__(self):
        pass

    def __getattribute__(self, item):
        import taichi as ti
        ti.get_runtime().create_program()
        root = SNode(ti.get_runtime().prog.get_root())
        return getattr(root, item)

    def __repr__(self):
        return 'ti.root'


root = Root()


#@deprecated('ti.var', 'ti.field')
def var(dt, shape=None, offset=None, needs_grad=False):
    _taichi_skip_traceback = 1
    return field(dt, shape, offset, needs_grad)


@python_scope
def field(dtype, shape=None, offset=None, needs_grad=False):
    _taichi_skip_traceback = 1
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

    if get_runtime().materialized:
        raise RuntimeError(
            "No new variables can be declared after materialization, i.e. kernel invocations "
            "or Python-scope tensor accesses. I.e., data layouts must be specified before "
            "any computation. Try appending ti.init() or ti.reset() "
            "right after 'import taichi as ti' if you are using Jupyter notebook."
        )

    del _taichi_skip_traceback

    # primal
    x = Expr(taichi_lang_core.make_id_expr(""))
    x.ptr = taichi_lang_core.global_new(x.ptr, dtype)
    x.ptr.set_is_primal(True)
    pytaichi.global_vars.append(x)

    if taichi_lang_core.needs_grad(dtype):
        # adjoint
        x_grad = Expr(taichi_lang_core.make_id_expr(""))
        x_grad.ptr = taichi_lang_core.global_new(x_grad.ptr, dtype)
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
    assert not pytaichi.materialized, "All layout must be specified before the first kernel launch / data access."
    warning(
        f"@ti.layout will be deprecated in the future, use ti.root directly to specify data layout anytime before the data structure materializes.",
        PendingDeprecationWarning,
        stacklevel=3)
    pytaichi.layout_functions.append(func)


@taichi_scope
def ti_print(*vars, sep=' ', end='\n'):
    def entry2content(var):
        if isinstance(var, str):
            return var
        else:
            return Expr(var).ptr

    def vars2entries(vars):
        for var in vars:
            if hasattr(var, '__ti_repr__'):
                repr = var.__ti_repr__()
                for v in vars2entries(repr):
                    yield v
            else:
                yield var

    def add_separators(vars):
        for i, var in enumerate(vars):
            if i: yield sep
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
    taichi_lang_core.create_print(contentries)


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


def indices(*x):
    return [taichi_lang_core.Index(i) for i in x]


index = indices


def static(x, *xs):
    _taichi_skip_traceback = 1
    if len(xs):  # for python-ish pointer assign: x, y = ti.static(y, x)
        return [static(x)] + [static(x) for x in xs]
    import types
    import taichi as ti
    if isinstance(x, (bool, int, float, range, list, tuple, enumerate,
                      ti.ndrange, ti.GroupedNDRange)) or x is None:
        return x
    elif isinstance(x, ti.lang.expr.Expr) and x.ptr.is_global_var():
        return x
    elif isinstance(x, ti.Matrix) and x.is_global():
        return x
    elif isinstance(x, (types.FunctionType, types.MethodType)):
        return x
    else:
        raise ValueError(
            f'Input to ti.static must be compile-time constants or global pointers, instead of {type(x)}'
        )


@taichi_scope
def grouped(x):
    import taichi as ti
    if isinstance(x, ti.ndrange):
        return x.grouped()
    else:
        return x


def stop_grad(x):
    taichi_lang_core.stop_grad(x.snode().ptr)


def current_cfg():
    return taichi_lang_core.current_compile_config()


def default_cfg():
    return taichi_lang_core.default_compile_config()


from .kernel import *
from .ops import *
from .kernel_arguments import *


def call_internal(name):
    taichi_lang_core.create_internal_func_stmt(name)
