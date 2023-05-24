import numbers

from taichi._lib import core as _ti_core
from taichi.lang import expr, impl, ops
from taichi.lang.enums import Layout
from taichi.lang.exception import (
    TaichiRuntimeTypeError,
    TaichiSyntaxError,
    TaichiTypeError,
)
from taichi.lang.matrix import Matrix, MatrixType
from taichi.lang.struct import StructType
from taichi.lang.util import cook_dtype, in_python_scope, python_scope, taichi_scope
from taichi.types import primitive_types, template, ndarray_type, texture_type
from taichi.types.compound_types import CompoundType


class ArgPack:
    """The ArgPack type class."""

    def __init__(self, *args, **kwargs):
        # converts dicts to argument packs
        if len(args) == 1 and kwargs == {} and isinstance(args[0], dict):
            self.entries = args[0]
        elif len(args) == 0:
            self.entries = kwargs
        else:
            raise TaichiSyntaxError(
                "Custom argument packs need to be initialized using either dictionary or keyword arguments"
            )
        for k, v in self.entries.items():
            self.entries[k] = v if in_python_scope() else impl.expr_init(v)

    @property
    def keys(self):
        """Returns the list of member names in string format.

        Example::

           >>> vec3 = ti.types.vector(3, ti.f32)
           >>> sphere = ti.Struct(center=vec3([0, 0, 0]), radius=1.0)
           >>> sphere.keys
           ['center', 'radius']
        """
        return list(self.entries.keys())

    @property
    def _members(self):
        return list(self.entries.values())

    @property
    def items(self):
        """Returns the items in this argument pack.

        Example::

            >>> vec3 = ti.types.vector(3, ti.f32)
            >>> sphere = ti.ArgPack(center=vec3([0, 0, 0]), radius=1.0)
            >>> sphere.items
            dict_items([('center', 2), ('radius', 1.0)])
        """
        return self.entries.items()

    def __getitem__(self, key):
        ret = self.entries[key]
        return ret

    def __setitem__(self, key, value):
        self.entries[key] = value

    def _set_entries(self, value):
        if isinstance(value, dict):
            value = ArgPack(value)
        for k in self.keys:
            self[k] = value[k]

    @staticmethod
    def _make_getter(key):
        def getter(self):
            """Get an entry from custom argument pack by name."""
            return self[key]

        return getter

    @staticmethod
    def _make_setter(key):
        @python_scope
        def setter(self, value):
            self[key] = value

        return setter

    @taichi_scope
    def _assign(self, other):
        if not isinstance(other, (dict, ArgPack)):
            raise TaichiTypeError("Only dict or ArgPack can be assigned to a ArgPack")
        if isinstance(other, dict):
            other = ArgPack(other)
        if self.entries.keys() != other.entries.keys():
            raise TaichiTypeError(f"Member mismatch between argument packs {self.keys}, {other.keys}")
        for k, v in self.items:
            v._assign(other.entries[k])
        return self

    def __len__(self):
        """Get the number of entries in a custom argument pack."""
        return len(self.entries)

    def __iter__(self):
        return self.entries.values()

    def __str__(self):
        """Python scope argument pack array print support."""
        if impl.inside_kernel():
            item_str = ", ".join([str(k) + "=" + str(v) for k, v in self.items])
            return f"<ti.ArgPack {item_str}>"
        return str(self.to_dict())

    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self):
        """Converts the ArgPack to a dictionary.

        Returns:
            Dict: The result dictionary.
        """
        res_dict = {
            k: v.to_dict() if isinstance(v, ArgPack) else v.to_list() if isinstance(v, Matrix) else v
            for k, v in self.entries.items()
        }
        return res_dict


class ArgPackType(CompoundType):
    def __init__(self, **kwargs):
        self.members = {}
        for k, dtype in kwargs.items():
            default = None
            if isinstance(dtype, tuple):
                if len(dtype) != 2:
                    raise TaichiSyntaxError("ArgPack with default values should provide tuple with exactly 2 elements.")
                default = dtype[1]
                dtype = dtype[0]
            if isinstance(dtype, StructType):
                self.members[k] = dtype, default
            elif isinstance(dtype, ArgPackType):
                self.members[k] = dtype, default
            elif isinstance(dtype, MatrixType):
                self.members[k] = dtype, default
            elif isinstance(dtype, ndarray_type.NdarrayType):
                self.members[k] = dtype, default
            elif isinstance(dtype, texture_type.RWTextureType):
                self.members[k] = dtype, default
            elif isinstance(dtype, texture_type.TextureType):
                self.members[k] = dtype, default
            elif isinstance(dtype, template):
                if default is not None:
                    raise TaichiSyntaxError("ArgPack does not support template with default values.")
                self.members[k] = dtype, None
            else:
                dtype = cook_dtype(dtype)
                self.members[k] = dtype, default

    def __call__(self, *args, **kwargs):
        """Create an instance of this argument pack type."""
        d = {}
        items = self.members.items()
        # iterate over the members of this argument pack
        for index, pair in enumerate(items):
            name, (dtype, default) = pair  # (member name, (member type, default value=None))
            if index < len(args):  # set from args
                data = args[index]
            else:  # set from kwargs
                data = kwargs.get(name, default)

            # If dtype is CompoundType and data is a scalar, it cannot be
            # casted in the self.cast call later. We need an initialization here.
            if isinstance(dtype, CompoundType) and not isinstance(data, (dict, ArgPack)):
                data = dtype(data)

            d[name] = data

        entries = ArgPack(d)
        pack = self.cast(entries)
        return pack

    def cast(self, pack):
        # sanity check members
        if self.members.keys() != pack.entries.keys():
            raise TaichiSyntaxError("Incompatible arguments for custom argument pack members!")
        entries = {}
        for k, (dtype, default) in self.members.items():
            if isinstance(dtype, MatrixType):
                entries[k] = dtype(pack.entries[k])
            elif isinstance(dtype, CompoundType):
                entries[k] = dtype.cast(pack.entries[k])
            else:
                if in_python_scope():
                    v = pack.entries[k]
                    entries[k] = int(v) if dtype in primitive_types.integer_types else float(v)
                else:
                    entries[k] = ops.cast(pack.entries[k], dtype)
        return ArgPack(entries)


__all__ = ["ArgPack"]
