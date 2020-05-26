class TaichiOperations:
    def __neg__(self):
        import taichi as ti
        return ti.neg(self)

    def __abs__(self):
        import taichi as ti
        return ti.abs(self)

    def __add__(self, other):
        import taichi as ti
        return ti.add(self, other)

    def __radd__(self, other):
        import taichi as ti
        return ti.add(other, self)

    def __sub__(self, other):
        import taichi as ti
        return ti.sub(self, other)

    def __rsub__(self, other):
        import taichi as ti
        return ti.sub(other, self)

    def __mul__(self, other):
        import taichi as ti
        return ti.mul(self, other)

    def __rmul__(self, other):
        import taichi as ti
        return ti.mul(other, self)

    def __truediv__(self, other):
        import taichi as ti
        return ti.truediv(self, other)

    def __rtruediv__(self, other):
        import taichi as ti
        return ti.truediv(other, self)

    def __floordiv__(self, other):
        import taichi as ti
        return ti.floordiv(self, other)

    def __rfloordiv__(self, other):
        import taichi as ti
        return ti.floordiv(other, self)

    def __mod__(self, other):
        import taichi as ti
        return ti.mod(self, other)

    def __pow__(self, other, modulo=None):
        import taichi as ti
        return ti.pow(self, other)

    def __rpow__(self, other, modulo=None):
        import taichi as ti
        return ti.pow(other, self)

    def __le__(self, other):
        import taichi as ti
        return ti.cmp_le(self, other)

    def __lt__(self, other):
        import taichi as ti
        return ti.cmp_lt(self, other)

    def __ge__(self, other):
        import taichi as ti
        return ti.cmp_ge(self, other)

    def __gt__(self, other):
        import taichi as ti
        return ti.cmp_gt(self, other)

    def __eq__(self, other):
        import taichi as ti
        return ti.cmp_eq(self, other)

    def __ne__(self, other):
        import taichi as ti
        return ti.cmp_ne(self, other)

    def __and__(self, other):
        import taichi as ti
        return ti.bit_and(self, other)

    def __or__(self, other):
        import taichi as ti
        return ti.bit_or(self, other)

    def __xor__(self, other):
        import taichi as ti
        return ti.bit_xor(self, other)

    def logical_and(self, other):
        import taichi as ti
        return ti.logical_and(self, other)

    def logical_or(self, other):
        import taichi as ti
        return ti.logical_or(self, other)

    def __invert__(self):  # ~a => a.__invert__()
        import taichi as ti
        return ti.bit_not(self)

    def __not__(self):  # not a => a.__not__()
        import taichi as ti
        return ti.logical_not(self)
