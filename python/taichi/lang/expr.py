from .core import taichi_lang_core
from .util import *
import traceback


# Scalar, basic data type
class Expr:
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

    @staticmethod
    def stack_info():
        s = traceback.extract_stack()[3:-1]
        for i, l in enumerate(s):
            if 'taichi_ast_generator' in l:
                s = s[i + 1:]
                break
        raw = ''.join(traceback.format_list(s))
        # remove the confusing last line
        return '\n'.join(raw.split('\n')[:-3]) + '\n'

    def __add__(self, other):
        other = Expr(other)
        return Expr(taichi_lang_core.expr_add(self.ptr, other.ptr),
                    tb=self.stack_info())

    __radd__ = __add__

    def __neg__(self):
        return Expr(taichi_lang_core.expr_neg(self.ptr), tb=self.stack_info())

    def __sub__(self, other):
        other = Expr(other)
        return Expr(taichi_lang_core.expr_sub(self.ptr, other.ptr),
                    tb=self.stack_info())

    def __rsub__(self, other):
        other = Expr(other)
        return Expr(taichi_lang_core.expr_sub(other.ptr, self.ptr))

    def __mul__(self, other):
        if is_taichi_class(other) and hasattr(other, '__rmul__'):
            return other.__rmul__(self)
        else:
            other = Expr(other)
            return Expr(taichi_lang_core.expr_mul(self.ptr, other.ptr))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return Expr(taichi_lang_core.expr_truediv(self.ptr, Expr(other).ptr))

    def __rtruediv__(self, other):
        return Expr(taichi_lang_core.expr_truediv(Expr(other).ptr, self.ptr))

    def __floordiv__(self, other):
        return Expr(taichi_lang_core.expr_floordiv(self.ptr, Expr(other).ptr))

    def __rfloordiv__(self, other):
        return Expr(taichi_lang_core.expr_floordiv(Expr(other).ptr, self.ptr))

    def __mod__(self, other):
        other = Expr(other)
        quotient = Expr(taichi_lang_core.expr_floordiv(self.ptr, other.ptr))
        multiply = Expr(taichi_lang_core.expr_mul(other.ptr, quotient.ptr))
        return Expr(taichi_lang_core.expr_sub(self.ptr, multiply.ptr))

    def __iadd__(self, other):
        self.atomic_add(other)

    def __isub__(self, other):
        self.atomic_sub(other)

    def __imul__(self, other):
        self.assign(Expr(taichi_lang_core.expr_mul(self.ptr, other.ptr)))

    def __itruediv__(self, other):
        self.assign(
            Expr(taichi_lang_core.expr_truediv(self.ptr,
                                               Expr(other).ptr)))

    def __ifloordiv__(self, other):
        self.assign(
            Expr(taichi_lang_core.expr_floordiv(self.ptr,
                                                Expr(other).ptr)))

    def __iand__(self, other):
        self.atomic_and(other)

    def __ior__(self, other):
        self.atomic_or(other)

    def __ixor__(self, other):
        self.atomic_xor(other)

    def __le__(self, other):
        other = Expr(other)
        return Expr(taichi_lang_core.expr_cmp_le(self.ptr, other.ptr))

    def __lt__(self, other):
        other = Expr(other)
        return Expr(taichi_lang_core.expr_cmp_lt(self.ptr, other.ptr))

    def __ge__(self, other):
        other = Expr(other)
        return Expr(taichi_lang_core.expr_cmp_ge(self.ptr, other.ptr))

    def __gt__(self, other):
        other = Expr(other)
        return Expr(taichi_lang_core.expr_cmp_gt(self.ptr, other.ptr))

    def __eq__(self, other):
        other = Expr(other)
        return Expr(taichi_lang_core.expr_cmp_eq(self.ptr, other.ptr))

    def __ne__(self, other):
        other = Expr(other)
        return Expr(taichi_lang_core.expr_cmp_ne(self.ptr, other.ptr))

    def __and__(self, item):
        item = Expr(item)
        return Expr(taichi_lang_core.expr_bit_and(self.ptr, item.ptr))

    def __or__(self, item):
        item = Expr(item)
        return Expr(taichi_lang_core.expr_bit_or(self.ptr, item.ptr))

    def __xor__(self, item):
        item = Expr(item)
        return Expr(taichi_lang_core.expr_bit_xor(self.ptr, item.ptr))

    def logical_and(self, item):
        return self & item

    def logical_or(self, item):
        return self | item

    def logical_not(self):
        return Expr(taichi_lang_core.expr_bit_not(self.ptr),
                    tb=self.stack_info())

    def assign(self, other):
        taichi_lang_core.expr_assign(self.ptr,
                                     Expr(other).ptr, self.stack_info())

    def __setitem__(self, key, value):
        if not Expr.layout_materialized:
            self.materialize_layout_callback()
        self.initialize_accessor()
        if key is None:
            key = ()
        if not isinstance(key, tuple):
            key = (key, )
        assert len(key) == self.dim()
        key = key + ((0, ) *
                     (taichi_lang_core.get_max_num_indices() - len(key)))
        self.setter(value, *key)

    def __getitem__(self, key):
        import taichi as ti
        assert not ti.get_runtime().inside_kernel
        if not Expr.layout_materialized:
            self.materialize_layout_callback()
        self.initialize_accessor()
        if key is None:
            key = ()
        if not isinstance(key, tuple):
            key = (key, )
        key = key + ((0, ) *
                     (taichi_lang_core.get_max_num_indices() - len(key)))
        return self.getter(*key)

    def loop_range(self):
        return self

    def augassign(self, x, op):
        x = Expr(x)
        if op == 'Add':
            self += x
        elif op == 'Sub':
            self -= x
        elif op == 'Mult':
            self *= x
        elif op == 'Div':
            self /= x
        elif op == 'FloorDiv':
            self //= x
        elif op == 'BitAnd':
            self &= x
        elif op == 'BitOr':
            self |= x
        elif op == 'BitXor':
            self ^= x
        else:
            assert False, op

    def atomic_add(self, other):
        import taichi as ti
        other_ptr = ti.wrap_scalar(other).ptr
        return ti.expr_init(
            taichi_lang_core.expr_atomic_add(self.ptr, other_ptr))

    def atomic_sub(self, other):
        import taichi as ti
        other_ptr = ti.wrap_scalar(other).ptr
        return ti.expr_init(
            taichi_lang_core.expr_atomic_sub(self.ptr, other_ptr))

    def atomic_min(self, other):
        import taichi as ti
        other_ptr = ti.wrap_scalar(other).ptr
        return ti.expr_init(
            taichi_lang_core.expr_atomic_min(self.ptr, other_ptr))

    def atomic_max(self, other):
        import taichi as ti
        other_ptr = ti.wrap_scalar(other).ptr
        return ti.expr_init(
            taichi_lang_core.expr_atomic_max(self.ptr, other_ptr))

    def atomic_and(self, other):
        import taichi as ti
        other_ptr = ti.wrap_scalar(other).ptr
        return ti.expr_init(
            taichi_lang_core.expr_atomic_bit_and(self.ptr, other_ptr))

    def atomic_or(self, other):
        import taichi as ti
        other_ptr = ti.wrap_scalar(other).ptr
        return ti.expr_init(
            taichi_lang_core.expr_atomic_bit_or(self.ptr, other_ptr))

    def atomic_xor(self, other):
        import taichi as ti
        other_ptr = ti.wrap_scalar(other).ptr
        return ti.expr_init(
            taichi_lang_core.expr_atomic_bit_xor(self.ptr, other_ptr))

    def serialize(self):
        return self.ptr.serialize()

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

    def set_grad(self, grad):
        self.grad = grad
        self.ptr.set_grad(grad.ptr)

    def clear(self, deactivate=False):
        assert not deactivate
        node = self.ptr.snode().parent
        assert node
        node.clear_data()

    def fill(self, val):
        # TODO: avoid too many template instantiations
        from .meta import fill_tensor
        fill_tensor(self, val)

    def __rpow__(self, power, modulo=None):
        return Expr(power).__pow__(self, modulo)

    def __pow__(self, power, modulo=None):
        import taichi as ti
        if not isinstance(power, int) or abs(power) > 100:
            return Expr(taichi_lang_core.expr_pow(self.ptr, Expr(power).ptr))
        if power == 0:
            return Expr(1)
        negative = power < 0
        power = abs(power)
        tmp = self
        ret = None
        while power:
            if power & 1:
                if ret is None:
                    ret = tmp
                else:
                    ret = ti.expr_init(ret * tmp)
            tmp = ti.expr_init(tmp * tmp)
            power >>= 1
        if negative:
            return 1 / ret
        else:
            return ret

    def __abs__(self):
        import taichi as ti
        return ti.abs(self)

    def __ti_int__(self):
        import taichi as ti
        return ti.cast(self, ti.get_runtime().default_ip)

    def __ti_float__(self):
        import taichi as ti
        return ti.cast(self, ti.get_runtime().default_fp)

    def parent(self):
        import taichi as ti
        return Expr(ti.core.global_var_expr_from_snode(
            self.ptr.snode().parent))

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
        dim = self.dim()
        s = []
        for i in range(dim):
            s.append(self.snode().get_shape(i))
        return tuple(s)

    def to_numpy(self):
        from .meta import tensor_to_ext_arr
        import numpy as np
        arr = np.empty(shape=self.shape(),
                       dtype=to_numpy_type(self.snode().data_type()))
        tensor_to_ext_arr(self, arr)
        import taichi as ti
        ti.sync()
        return arr

    def to_torch(self, device=None):
        from .meta import tensor_to_ext_arr
        import torch
        arr = torch.empty(size=self.shape(),
                          dtype=to_pytorch_type(self.snode().data_type()),
                          device=device)
        tensor_to_ext_arr(self, arr)
        import taichi as ti
        ti.sync()
        return arr

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

    def from_torch(self, arr):
        self.from_numpy(arr.contiguous())

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
