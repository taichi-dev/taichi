import numbers

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
    from taichi.lang.matrix import MatrixType
    return MatrixType(m, n, dtype=dtype)


def vector(m, dtype=None):
    from taichi.lang.matrix import MatrixType
    return MatrixType(m, 1, dtype=dtype)


def struct(**kwargs):
    from taichi.lang.struct import StructType
    return StructType(**kwargs)
