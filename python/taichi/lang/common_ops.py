from taichi.lang import ops


class TaichiOperations:
    """The base class of taichi operations of expressions. Subclasses: :class:`~taichi.lang.expr.Expr`, :class:`~taichi.lang.matrix.Matrix`"""
    def __neg__(self):
        _taichi_skip_traceback = 1
        return ops.neg(self)

    def __abs__(self):
        _taichi_skip_traceback = 1
        return ops.abs(self)

    def __add__(self, other):
        _taichi_skip_traceback = 1
        return ops.add(self, other)

    def __radd__(self, other):
        _taichi_skip_traceback = 1
        return ops.add(other, self)

    def __sub__(self, other):
        _taichi_skip_traceback = 1
        return ops.sub(self, other)

    def __rsub__(self, other):
        _taichi_skip_traceback = 1
        return ops.sub(other, self)

    def __mul__(self, other):
        _taichi_skip_traceback = 1
        return ops.mul(self, other)

    def __rmul__(self, other):
        _taichi_skip_traceback = 1
        return ops.mul(other, self)

    def __truediv__(self, other):
        _taichi_skip_traceback = 1
        return ops.truediv(self, other)

    def __rtruediv__(self, other):
        _taichi_skip_traceback = 1
        return ops.truediv(other, self)

    def __floordiv__(self, other):
        _taichi_skip_traceback = 1
        return ops.floordiv(self, other)

    def __rfloordiv__(self, other):
        _taichi_skip_traceback = 1
        return ops.floordiv(other, self)

    def __mod__(self, other):
        _taichi_skip_traceback = 1
        return ops.mod(self, other)

    def __rmod__(self, other):
        _taichi_skip_traceback = 1
        return ops.mod(other, self)

    def __pow__(self, other, modulo=None):
        _taichi_skip_traceback = 1
        return ops.pow(self, other)

    def __rpow__(self, other, modulo=None):
        _taichi_skip_traceback = 1
        return ops.pow(other, self)

    def __le__(self, other):
        _taichi_skip_traceback = 1
        return ops.cmp_le(self, other)

    def __lt__(self, other):
        _taichi_skip_traceback = 1
        return ops.cmp_lt(self, other)

    def __ge__(self, other):
        _taichi_skip_traceback = 1
        return ops.cmp_ge(self, other)

    def __gt__(self, other):
        _taichi_skip_traceback = 1
        return ops.cmp_gt(self, other)

    def __eq__(self, other):
        _taichi_skip_traceback = 1
        return ops.cmp_eq(self, other)

    def __ne__(self, other):
        _taichi_skip_traceback = 1
        return ops.cmp_ne(self, other)

    def __and__(self, other):
        _taichi_skip_traceback = 1
        return ops.bit_and(self, other)

    def __rand__(self, other):
        _taichi_skip_traceback = 1
        return ops.bit_and(other, self)

    def __or__(self, other):
        _taichi_skip_traceback = 1
        return ops.bit_or(self, other)

    def __ror__(self, other):
        _taichi_skip_traceback = 1
        return ops.bit_or(other, self)

    def __xor__(self, other):
        _taichi_skip_traceback = 1
        return ops.bit_xor(self, other)

    def __rxor__(self, other):
        _taichi_skip_traceback = 1
        return ops.bit_xor(other, self)

    def __lshift__(self, other):
        _taichi_skip_traceback = 1
        return ops.bit_shl(self, other)

    def __rlshift__(self, other):
        _taichi_skip_traceback = 1
        return ops.bit_shl(other, self)

    def __rshift__(self, other):
        _taichi_skip_traceback = 1
        return ops.bit_sar(self, other)

    def __rrshift__(self, other):
        _taichi_skip_traceback = 1
        return ops.bit_sar(other, self)

    def logical_and(self, other):
        """Return the new expression of computing logical and between self and a given operand.

        Args:
            other (Any): Given operand.

        Returns:
            :class:`~taichi.lang.expr.Expr`: The computing expression of logical and."""
        _taichi_skip_traceback = 1
        return ops.logical_and(self, other)

    def logical_or(self, other):
        """Return the new expression of computing logical or between self and a given operand.

        Args:
            other (Any): Given operand.

        Returns:
            :class:`~taichi.lang.expr.Expr`: The computing expression of logical or."""
        _taichi_skip_traceback = 1
        return ops.logical_or(self, other)

    def __invert__(self):  # ~a => a.__invert__()
        _taichi_skip_traceback = 1
        return ops.bit_not(self)

    def __not__(self):  # not a => a.__not__()
        _taichi_skip_traceback = 1
        return ops.logical_not(self)

    def atomic_add(self, other):
        """Return the new expression of computing atomic add between self and a given operand.

        Args:
            other (Any): Given operand.

        Returns:
            :class:`~taichi.lang.expr.Expr`: The computing expression of atomic add."""
        _taichi_skip_traceback = 1
        return ops.atomic_add(self, other)

    def atomic_sub(self, other):
        """Return the new expression of computing atomic sub between self and a given operand.

        Args:
            other (Any): Given operand.

        Returns:
            :class:`~taichi.lang.expr.Expr`: The computing expression of atomic sub."""
        _taichi_skip_traceback = 1
        return ops.atomic_sub(self, other)

    def atomic_and(self, other):
        """Return the new expression of computing atomic and between self and a given operand.

        Args:
            other (Any): Given operand.

        Returns:
            :class:`~taichi.lang.expr.Expr`: The computing expression of atomic and."""
        _taichi_skip_traceback = 1
        return ops.atomic_and(self, other)

    def atomic_xor(self, other):
        """Return the new expression of computing atomic xor between self and a given operand.

        Args:
            other (Any): Given operand.

        Returns:
            :class:`~taichi.lang.expr.Expr`: The computing expression of atomic xor."""
        _taichi_skip_traceback = 1
        return ops.atomic_xor(self, other)

    def atomic_or(self, other):
        """Return the new expression of computing atomic or between self and a given operand.

        Args:
            other (Any): Given operand.

        Returns:
            :class:`~taichi.lang.expr.Expr`: The computing expression of atomic or."""
        _taichi_skip_traceback = 1
        return ops.atomic_or(self, other)

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
        self.assign(ops.mul(self, other))
        return self

    def __itruediv__(self, other):
        _taichi_skip_traceback = 1
        self.assign(ops.truediv(self, other))
        return self

    def __ifloordiv__(self, other):
        _taichi_skip_traceback = 1
        self.assign(ops.floordiv(self, other))
        return self

    def __imod__(self, other):
        _taichi_skip_traceback = 1
        self.assign(ops.mod(self, other))
        return self

    def __ilshift__(self, other):
        _taichi_skip_traceback = 1
        self.assign(ops.bit_shl(self, other))
        return self

    def __irshift__(self, other):
        _taichi_skip_traceback = 1
        self.assign(ops.bit_shr(self, other))
        return self

    def __ipow__(self, other):
        _taichi_skip_traceback = 1
        self.assign(ops.pow(self, other))
        return self

    def assign(self, other):
        """Assign the expression of the given operand to self.

        Args:
            other (Any): Given operand.

        Returns:
            :class:`~taichi.lang.expr.Expr`: The expression after assigning."""
        _taichi_skip_traceback = 1
        return ops.assign(self, other)

    # pylint: disable=R0201
    def augassign(self, x, op):
        """Generate the computing expression between self and the given operand of given operator and assigned to self.

        Args:
            x (Any): Given operand.
            op (str): The name of operator."""
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
        elif op == 'Pow':
            self **= x
        else:
            assert False, op

    def __ti_int__(self):
        _taichi_skip_traceback = 1
        return ops.cast(self, int)

    def __ti_float__(self):
        _taichi_skip_traceback = 1
        return ops.cast(self, float)
