import taichi as ti


class TaichiOperations:
    def __neg__(self):
        _taichi_skip_traceback = 1
        return ti.neg(self)

    def __abs__(self):
        _taichi_skip_traceback = 1
        return ti.abs(self)

    def __add__(self, other):
        _taichi_skip_traceback = 1
        return ti.add(self, other)

    def __radd__(self, other):
        _taichi_skip_traceback = 1
        return ti.add(other, self)

    def __sub__(self, other):
        _taichi_skip_traceback = 1
        return ti.sub(self, other)

    def __rsub__(self, other):
        _taichi_skip_traceback = 1
        return ti.sub(other, self)

    def __mul__(self, other):
        _taichi_skip_traceback = 1
        return ti.mul(self, other)

    def __rmul__(self, other):
        _taichi_skip_traceback = 1
        return ti.mul(other, self)

    def __truediv__(self, other):
        _taichi_skip_traceback = 1
        return ti.truediv(self, other)

    def __rtruediv__(self, other):
        _taichi_skip_traceback = 1
        return ti.truediv(other, self)

    def __floordiv__(self, other):
        _taichi_skip_traceback = 1
        return ti.floordiv(self, other)

    def __rfloordiv__(self, other):
        _taichi_skip_traceback = 1
        return ti.floordiv(other, self)

    def __mod__(self, other):
        _taichi_skip_traceback = 1
        return ti.mod(self, other)

    def __rmod__(self, other):
        _taichi_skip_traceback = 1
        return ti.mod(other, self)

    def __pow__(self, other, modulo=None):
        _taichi_skip_traceback = 1
        return ti.pow(self, other)

    def __rpow__(self, other, modulo=None):
        _taichi_skip_traceback = 1
        return ti.pow(other, self)

    def __le__(self, other):
        _taichi_skip_traceback = 1
        return ti.cmp_le(self, other)

    def __lt__(self, other):
        _taichi_skip_traceback = 1
        return ti.cmp_lt(self, other)

    def __ge__(self, other):
        _taichi_skip_traceback = 1
        return ti.cmp_ge(self, other)

    def __gt__(self, other):
        _taichi_skip_traceback = 1
        return ti.cmp_gt(self, other)

    def __eq__(self, other):
        _taichi_skip_traceback = 1
        return ti.cmp_eq(self, other)

    def __ne__(self, other):
        _taichi_skip_traceback = 1
        return ti.cmp_ne(self, other)

    def __and__(self, other):
        _taichi_skip_traceback = 1
        return ti.bit_and(self, other)

    def __rand__(self, other):
        _taichi_skip_traceback = 1
        return ti.bit_and(other, self)

    def __or__(self, other):
        _taichi_skip_traceback = 1
        return ti.bit_or(self, other)

    def __ror__(self, other):
        _taichi_skip_traceback = 1
        return ti.bit_or(other, self)

    def __xor__(self, other):
        _taichi_skip_traceback = 1
        return ti.bit_xor(self, other)

    def __rxor__(self, other):
        _taichi_skip_traceback = 1
        return ti.bit_xor(other, self)

    def __lshift__(self, other):
        _taichi_skip_traceback = 1
        return ti.bit_shl(self, other)

    def __rlshift__(self, other):
        _taichi_skip_traceback = 1
        return ti.bit_shl(other, self)

    def __rshift__(self, other):
        _taichi_skip_traceback = 1
        return ti.bit_sar(self, other)

    def __rrshift__(self, other):
        _taichi_skip_traceback = 1
        return ti.bit_sar(other, self)

    def logical_and(self, other):
        _taichi_skip_traceback = 1
        return ti.logical_and(self, other)

    def logical_or(self, other):
        _taichi_skip_traceback = 1
        return ti.logical_or(self, other)

    def __invert__(self):  # ~a => a.__invert__()
        _taichi_skip_traceback = 1
        return ti.bit_not(self)

    def __not__(self):  # not a => a.__not__()
        _taichi_skip_traceback = 1
        return ti.logical_not(self)

    def atomic_add(self, other):
        _taichi_skip_traceback = 1
        return ti.atomic_add(self, other)

    def atomic_sub(self, other):
        _taichi_skip_traceback = 1
        return ti.atomic_sub(self, other)

    def atomic_and(self, other):
        _taichi_skip_traceback = 1
        return ti.atomic_and(self, other)

    def atomic_xor(self, other):
        _taichi_skip_traceback = 1
        return ti.atomic_xor(self, other)

    def atomic_or(self, other):
        _taichi_skip_traceback = 1
        return ti.atomic_or(self, other)

    def __iadd__(self, other):
        _taichi_skip_traceback = 1
        self.atomic_add(other)
        return self

    def __isub__(self, other):
        _taichi_skip_traceback = 1
        self.atomic_sub(other)
        return self

    def __iand__(self, other):
        _taichi_skip_traceback = 1
        self.atomic_and(other)
        return self

    def __ixor__(self, other):
        _taichi_skip_traceback = 1
        self.atomic_xor(other)
        return self

    def __ior__(self, other):
        _taichi_skip_traceback = 1
        self.atomic_or(other)
        return self

    # we don't support atomic_mul/truediv/floordiv/mod yet:
    def __imul__(self, other):
        _taichi_skip_traceback = 1
        self.assign(ti.mul(self, other))
        return self

    def __itruediv__(self, other):
        _taichi_skip_traceback = 1
        self.assign(ti.truediv(self, other))
        return self

    def __ifloordiv__(self, other):
        _taichi_skip_traceback = 1
        self.assign(ti.floordiv(self, other))
        return self

    def __imod__(self, other):
        _taichi_skip_traceback = 1
        self.assign(ti.mod(self, other))
        return self

    def __ilshift__(self, other):
        _taichi_skip_traceback = 1
        self.assign(ti.bit_shl(self, other))
        return self

    def __irshift__(self, other):
        _taichi_skip_traceback = 1
        self.assign(ti.bit_shr(self, other))
        return self

    def assign(self, other):
        _taichi_skip_traceback = 1
        return ti.assign(self, other)

    def augassign(self, x, op):
        _taichi_skip_traceback = 1
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
        elif op == 'Mod':
            self %= x
        elif op == 'BitAnd':
            self &= x
        elif op == 'BitOr':
            self |= x
        elif op == 'BitXor':
            self ^= x
        elif op == 'RShift':
            self >>= x
        elif op == 'LShift':
            self <<= x
        else:
            assert False, op

    def __ti_int__(self):
        _taichi_skip_traceback = 1
        return ti.cast(self, int)

    def __ti_float__(self):
        _taichi_skip_traceback = 1
        return ti.cast(self, float)
