import taichi


class CompoundType:
    pass


# TODO: maybe move MatrixType, StructType here to avoid the circular import?
def matrix(n, m, dtype):
    return taichi.lang.matrix.MatrixType(n, m, dtype)


def vector(n, dtype):
    return taichi.lang.matrix.MatrixType(n, 1, dtype)


def struct(**kwargs):
    return taichi.lang.struct.StructType(**kwargs)


__all__ = ['matrix', 'vector', 'struct']
