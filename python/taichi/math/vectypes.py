import collections
import functools
from taichi.lang.matrix import Matrix
from taichi.lang.util import python_scope, in_python_scope
from taichi.types.primitive_types import i32, f32


class _VectorType(Matrix):
    """Abstract class to cook glsl-style vectors.
    """

    _KEYMAP_SET = ["xyzw", "rgba", "uvw"]
    _DIM = 3
    _DTYPE = f32

    def __init__(self, *data):
        x = data[0]
        assert not isinstance(
            x, collections.abc.Sequence), "Matrix is not accepted"
        
        assert len(data) in [1, self._DIM], "Dimension not match"
        
        if len(data) == 1:
            data = [x] * self._DIM
        
        super().__init__(data, self._DTYPE)
        
        self._add_swizzle_attrs()

    def _add_swizzle_attrs(self):
        """Create and bind properties for vector swizzles.
        """
        def getter_template(index, instance):
            return instance(index)

        @python_scope
        def setter_template(index, instance, value):
            instance[index] = value

        for key_group in _VectorType._KEYMAP_SET:
            for index, key in enumerate(key_group):
                prop = property(functools.partial(getter_template, index),
                                functools.partial(setter_template, index))
                setattr(type(self), key, prop)

    def __getattr__(self, attr_name):
        for key_group in _VectorType._KEYMAP_SET:
            if any(x not in key_group for x in attr_name):
                continue

            result = []
            for key in attr_name:
                result.append(self(key_group.index(key)))
            if result:
                if self._DTYPE == f32:
                    return globals()[f"vec{len(result)}"](*result)
                else:
                    return globals()[f"ivec{len(result)}"](*result)

        raise AttributeError(f"Cannot get attribute: {attr_name}")

    def __setattr__(self, attr_name, values):
        if len(attr_name) > 1:
            for key_group in _VectorType._KEYMAP_SET:
                if any(x not in key_group for x in attr_name):
                    continue

                if len(attr_name) != len(values):
                    raise Exception("values does not match the attribute")

                was_valid = False
                for key, value in zip(attr_name, values):
                    if in_python_scope():
                        self[key_group.index(key)] = value
                    else:
                        self(key_group.index(key))._assign(value)
                    was_valid = True

                if was_valid:
                    return

        super().__setattr__(attr_name, values)

    def to_list(self):
        """Return this matrix as a 1D `list`.

        This is similar to `numpy.ndarray`'s `flatten` and `ravel` methods,
        but this function always returns a new list.
        """
        return [self(i) for i in range(self.n)]


def _wrap_ops(cls):
    def wrapped(parent):
        return lambda self, other: cls(*parent(self, other))

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
    import sys
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