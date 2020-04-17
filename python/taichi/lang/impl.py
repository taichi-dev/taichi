import inspect
from .core import taichi_lang_core
from .expr import Expr
from .snode import SNode
from .util import *


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


def expr_init_func(rhs):  # temporary solution to allow passing in tensors as
    import taichi as ti
    if isinstance(rhs, Expr) and rhs.ptr.is_global_var():
        return rhs
    if isinstance(rhs, ti.Matrix) and rhs.is_global():
        return rhs
    return expr_init(rhs)


def wrap_scalar(x):
    if type(x) in [int, float]:
        return Expr(x)
    else:
        return x


def atomic_add(a, b):
    return a.atomic_add(b)


def atomic_sub(a, b):
    return a.atomic_sub(b)


def atomic_min(a, b):
    return a.atomic_min(b)


def atomic_max(a, b):
    return a.atomic_max(b)


def atomic_and(a, b):
    return a.atomic_and(b)


def atomic_or(a, b):
    return a.atomic_or(b)


def atomic_xor(a, b):
    return a.atomic_xor(b)


def subscript(value, *indices):
    import numpy as np
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
            indices_expr_group = make_expr_group()
        else:
            indices_expr_group = make_expr_group(*indices)
        tensor_dim = int(value.ptr.get_attribute("dim"))
        index_dim = indices_expr_group.size()
        assert tensor_dim == index_dim, f'Tensor with dim {tensor_dim} accessed with indices of dim {index_dim}'
        return Expr(taichi_lang_core.subscript(value.ptr, indices_expr_group))


def chain_compare(comparators, ops):
    assert len(comparators) == len(ops) + 1, \
      f'Chain comparison invoked with {len(comparators)} comparators but {len(ops)} operators'
    evaluated_comparators = []
    for i in range(len(comparators)):
        evaluated_comparators += [expr_init(comparators[i])]
    ret = expr_init(True)
    for i in range(len(ops)):
        lhs = evaluated_comparators[i]
        rhs = evaluated_comparators[i + 1]
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
        ret = ret.logical_and(now)
    return ret


class PyTaichi:
    def __init__(self, kernels=[]):
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
        self.kernels = kernels
        Expr.materialize_layout_callback = self.materialize

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
        Expr.layout_materialized = True

        def layout():
            for func in self.layout_functions:
                func()

        import taichi as ti
        ti.trace("Materializing layout...".format())
        taichi_lang_core.layout(layout)
        self.materialized = True
        for var in self.global_vars:
            assert var.ptr.snode() is not None, 'Some variable(s) not placed'

    def clear(self):
        if self.prog:
            self.prog.finalize()
            self.prog = None
        Expr.materialize_layout_callback = None
        Expr.layout_materialized = False

    def get_tape(self, loss=None):
        from .tape import Tape
        return Tape(self, loss)

    def sync(self):
        self.materialize()
        self.prog.synchronize()


pytaichi = PyTaichi()


def get_runtime():
    return pytaichi


def make_constant_expr(val):
    if isinstance(val, int):
        if pytaichi.default_ip == i32:
            return Expr(taichi_lang_core.make_const_expr_i32(val))
        elif pytaichi.default_ip == i64:
            return Expr(taichi_lang_core.make_const_expr_i64(val))
        else:
            assert False
    else:
        if pytaichi.default_fp == f32:
            return Expr(taichi_lang_core.make_const_expr_f32(val))
        elif pytaichi.default_fp == f64:
            return Expr(taichi_lang_core.make_const_expr_f64(val))
        else:
            assert False


def reset():
    global pytaichi
    old_kernels = pytaichi.kernels
    pytaichi.clear()
    pytaichi = PyTaichi(old_kernels)
    for k in old_kernels:
        k.reset()
    taichi_lang_core.reset_default_compile_config()


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


root = Root()


def var(dt, shape=None, needs_grad=False):
    if isinstance(shape, numbers.Number):
        shape = (shape, )

    assert not get_runtime(
    ).materialized, 'No new variables can be declared after kernel invocations or Python-scope tensor accesses.'

    # primal
    x = Expr(taichi_lang_core.make_id_expr(""))
    x.ptr = taichi_lang_core.global_new(x.ptr, dt)
    x.ptr.set_is_primal(True)
    pytaichi.global_vars.append(x)

    if taichi_lang_core.needs_grad(dt):
        # adjoint
        x_grad = Expr(taichi_lang_core.make_id_expr(""))
        x_grad.ptr = taichi_lang_core.global_new(x_grad.ptr, dt)
        x_grad.ptr.set_is_primal(False)
        x.set_grad(x_grad)

    if shape is not None:
        dim = len(shape)
        root.dense(index_nd(dim), shape).place(x)
        if needs_grad:
            root.dense(index_nd(dim), shape).place(x.grad)

    return x


class Layout:
    def __init__(self, soa=False):
        self.soa = soa


SOA = Layout(soa=True)
AOS = Layout(soa=False)


def layout(func):
    assert not pytaichi.materialized, "All layout must be specified before the first kernel launch / data access."
    pytaichi.layout_functions.append(func)


def ti_print(var):
    code = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
    arg_name = code[code.index('(') + 1:code.rindex(')')]
    taichi_lang_core.print_(Expr(var).ptr, arg_name)


def ti_int(var):
    if hasattr(var, '__ti_int__'):
        return var.__ti_int__()
    else:
        return int(var)


def ti_float(var):
    if hasattr(var, '__ti_float__'):
        return var.__ti_float__()
    else:
        return float(var)


def indices(*x):
    return [taichi_lang_core.Index(i) for i in x]


index = indices


def static(x, *xs):
    if len(xs):  # for python-ish pointer assign: x, y = ti.static(y, x)
        return [static(x)] + [static(x) for x in xs]
    import types
    import taichi as ti
    assert get_runtime(
    ).inside_kernel, 'ti.static can only be used inside Taichi kernels'
    if isinstance(x, (bool, int, float, range, list, tuple, enumerate,
                      ti.ndrange, ti.GroupedNDRange)):
        return x
    elif isinstance(x, ti.lang.expr.Expr) and x.ptr.is_global_var():
        return x
    elif isinstance(x, ti.Matrix) and x.is_global():
        return x
    elif isinstance(x, ti.Func) or isinstance(x, types.MethodType):
        return x
    else:
        raise ValueError(
            f'Input to ti.static must be compile-time constants or global pointers, instead of {type(x)}'
        )


def grouped(x):
    import taichi as ti
    assert get_runtime(
    ).inside_kernel, 'ti.grouped can only be used inside Taichi kernels'
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
