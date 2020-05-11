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
