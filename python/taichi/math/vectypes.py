import collections
import sys

from taichi.lang.matrix import Matrix
from taichi.types.primitive_types import f32, i32


class _VectorType(Matrix):
    """Abstract class to cook glsl-style vectors.
    """

    _KEYMAP_SET = ["xyzw", "rgba", "uvw"]
    _DIM = 3
    _DTYPE = f32

    def __init__(self, *data, is_ref=False):
        x = data[0]
        assert not isinstance(
            x, collections.abc.Sequence), "Matrix is not accepted"

        assert len(data) in [1, self._DIM], "Dimension not match"

        if len(data) == 1:
            data = [x] * self._DIM

        super().__init__(data, self._DTYPE, is_ref=is_ref)

    def to_list(self):
        """Return this matrix as a 1D `list`.

        This is similar to `numpy.ndarray`'s `flatten` and `ravel` methods,
        but this function always returns a new list.
        """
        return [self(i) for i in range(self.n)]


def _wrap_ops(cls):
    def wrapped(func):
        return lambda *args, **kwargs: cls(*func(*args, **kwargs))

    methods = [
        '__neg__', '__abs__', '__add__', '__radd__', '__sub__', '__rsub__',
        '__mul__', '__rmul__', '__truediv__', '__rtruediv__', '__floordiv__',
        '__rfloordiv__', '__mod__', '__rmod__', '__pow__', '__rpow__',
        '_atomic_add', '_atomic_sub', '_atomic_and', '_atomic_xor',
        '_atomic_or', '__iadd__', '__isub__', '__iand__', '__ixor__',
        '__ior__', '__imul__', '__itruediv__', '__ifloordiv__', '__imod__',
        '__ilshift__', '__irshift__', '__ipow__'
    ]
    for op in methods:
        func = wrapped(getattr(cls, op))
        setattr(cls, op, func)


def _generate_vectorND_classes():
    module = sys.modules[__name__]
    for dim in [2, 3, 4]:
        for dt, prefix in zip([f32, i32], ["", "i"]):
            vec_class_name = f"{prefix}vec{dim}"
            vec_class = type(vec_class_name, (_VectorType, ), {
                "_DIM": dim,
                "_DTYPE": dt
            })
            _wrap_ops(vec_class)
            setattr(module, vec_class_name, vec_class)
            globals()[vec_class_name] = vec_class


_generate_vectorND_classes()

__all__ = ['ivec2', 'ivec3', 'ivec4', 'vec2', 'vec3', 'vec4']
