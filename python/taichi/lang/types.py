import numbers

import taichi.lang.matrix
from taichi.lang.exception import TaichiSyntaxError


class CompoundType:
    def empty(self):
        """
        Create an empty instance of the given compound type.
        """
        raise NotImplementedError

    def scalar_filled(self, value):
        instance = self.empty()
        return instance.broadcast_copy(value)

    def field(self, **kwargs):
        raise NotImplementedError


def matrix(m, n, dtype=None):
    return taichi.lang.matrix.MatrixType(m, n, dtype=dtype)


def vector(m, dtype=None):
    return taichi.lang.matrix.MatrixType(m, 1, dtype=dtype)


def struct(**kwargs):
    return taichi.lang.struct.StructType(**kwargs)
