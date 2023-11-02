from taichi.lang.matrix import Matrix
from taichi.lang.util import in_python_scope, python_scope
import numpy as np
import taichi.lang
from taichi._lib import core as _ti_core
from taichi.lang import impl, ops
from taichi.lang.exception import (
    TaichiRuntimeTypeError,
    TaichiSyntaxError,
)
from taichi.lang.matrix import MatrixType
from taichi.lang.struct import StructType, Struct
from taichi.lang.util import cook_dtype
from taichi.types import (
    ndarray_type,
    primitive_types,
    sparse_matrix_builder,
    texture_type,
)
from taichi.types.compound_types import CompoundType
from taichi.types.utils import is_signed


class ArgPack:
    """ The `ArgPack` Type Class.

    The `ArgPack` operates as a dictionary-like data pack, storing members as (key, value) pairs. Members stored can
    range from scalars and matrices to other dictionary-like structures. Distinguished from structs, `ArgPack` can
    accommodate buffer types such as `NdarrayType` and `TextureType` from Taichi. However, unlike `ti.Struct` which
    serves as a data container, `ArgPack` functions as a reference container. It's important to note that `ArgPack`
    cannot be nested within other types except for another `ArgPack`, and can only be utilized as kernel parameters.

    Args:
        annotations (Dict[str, Union[Dict, Matrix, Struct]]): \
            The keys and types for `ArgPack` members.
        dtype (ArgPackType): \
            The ArgPackType class of this ArgPack object.
        entries (Dict[str, Union[Dict, Matrix, Struct]]): \
            The keys and corresponding values for `ArgPack` members.

    Returns:
        An instance of this `ArgPack`.

    Example::

        >>> vec3 = ti.types.vector(3, ti.f32)
        >>> pack_type = ti.ArgPackType(v=vec3, t=ti.f32)
        >>> a = pack_type(v=vec3([0, 0, 0]), t=1.0)
        >>> print(a.items)
        dict_items([('v', [0. 0. 0.]), ('t', 1.0)])
    """

    _instance_count = 0

    def __init__(self, annotations, dtype, *args, **kwargs):
        # converts dicts to argument packs
        if len(args) == 1 and kwargs == {} and isinstance(args[0], dict):
            self.__entries = args[0]
        elif len(args) == 0:
            self.__entries = kwargs
        else:
            raise TaichiSyntaxError(
                "Custom argument packs need to be initialized using either dictionary or keyword arguments"
            )
        if annotations.keys() != self.__entries.keys():
            raise TaichiSyntaxError("ArgPack annotations keys not equals to entries keys.")
        self.__annotations = annotations
        for k, v in self.__entries.items():
            self.__entries[k] = v if in_python_scope() else impl.expr_init(v)
        self._register_members()
        self.__dtype = dtype
        self.__argpack = impl.get_runtime().prog.create_argpack(self.__dtype)
        for i, (k, v) in enumerate(self.__entries.items()):
            self._write_to_device(self.__annotations[k], type(v), v, self._calc_element_true_index(i))

    def __del__(self):
        if impl is not None and impl.get_runtime() is not None and impl.get_runtime().prog is not None:
            impl.get_runtime().prog.delete_argpack(self.__argpack)

    @property
    def keys(self):
        """Returns the list of member names in string format.

        Example::

           >>> vec3 = ti.types.vector(3, ti.f32)
           >>> sphere_pack = ti.ArgPackType(center=vec3, radius=ti.f32)
           >>> sphere = sphere_pack(center=vec3([0, 0, 0]), radius=1.0)
           >>> sphere.keys
           ['center', 'radius']
        """
        return list(self.__entries.keys())

    @property
    def _members(self):
        return list(self.__entries.values())

    @property
    def _annotations(self):
        return list(self.__annotations.values())

    @property
    def items(self):
        """Returns the items in this argument pack.

        Example::

           >>> vec3 = ti.types.vector(3, ti.f32)
           >>> sphere_pack = ti.ArgPackType(center=vec3, radius=ti.f32)
           >>> sphere = sphere_pack(center=vec3([0, 0, 0]), radius=1.0)
           >>> sphere.items
            dict_items([('center', 2), ('radius', 1.0)])
        """
        return self.__entries.items()

    def __getitem__(self, key):
        ret = self.__entries[key]
        return ret

    def __setitem__(self, key, value):
        self.__entries[key] = value
        index = self._calc_element_true_index(list(self.__annotations).index(key))
        self._write_to_device(self.__annotations[key], type(value), value, index)

    def _set_entries(self, value):
        if isinstance(value, dict):
            value = ArgPack(self.__annotations, value)
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

    def _register_members(self):
        # https://stackoverflow.com/questions/48448074/adding-a-property-to-an-existing-object-instance
        cls = self.__class__
        new_cls_name = cls.__name__ + str(cls._instance_count)
        cls._instance_count += 1
        properties = {k: property(cls._make_getter(k), cls._make_setter(k)) for k in self.keys}
        self.__class__ = type(new_cls_name, (cls,), properties)

    def __len__(self):
        """Get the number of entries in a custom argument pack."""
        return len(self.__entries)

    def __iter__(self):
        return self.__entries.values()

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
            for k, v in self.__entries.items()
        }
        return res_dict

    def _calc_element_true_index(self, old_index):
        for i in range(old_index):
            anno = list(self.__annotations.values())[i]
            if (
                isinstance(anno, sparse_matrix_builder)
                or isinstance(anno, ndarray_type.NdarrayType)
                or isinstance(anno, texture_type.TextureType)
                or isinstance(anno, texture_type.RWTextureType)
                or isinstance(anno, ndarray_type.NdarrayType)
            ):
                old_index -= 1
        return old_index

    def _write_to_device(self, needed, provided, v, index):
        if isinstance(needed, ArgPackType):
            if not isinstance(v, ArgPack):
                raise TaichiRuntimeTypeError.get(index, str(needed), str(provided))
            self.__argpack.set_arg_nested_argpack(index, v.__argpack)
        else:
            # Note: do not use sth like "needed == f32". That would be slow.
            if id(needed) in primitive_types.real_type_ids:
                if not isinstance(v, (float, int, np.floating, np.integer)):
                    raise TaichiRuntimeTypeError.get(index, needed.to_string(), provided)
                self.__argpack.set_arg_float((index,), float(v))
            elif id(needed) in primitive_types.integer_type_ids:
                if not isinstance(v, (int, np.integer)):
                    raise TaichiRuntimeTypeError.get(index, needed.to_string(), provided)
                if is_signed(cook_dtype(needed)):
                    self.__argpack.set_arg_int((index,), int(v))
                else:
                    self.__argpack.set_arg_uint((index,), int(v))
            elif isinstance(needed, sparse_matrix_builder):
                pass
            elif isinstance(needed, ndarray_type.NdarrayType) and isinstance(v, taichi.lang._ndarray.Ndarray):
                pass
            elif isinstance(needed, texture_type.TextureType) and isinstance(v, taichi.lang._texture.Texture):
                pass
            elif isinstance(needed, texture_type.RWTextureType) and isinstance(v, taichi.lang._texture.Texture):
                pass
            elif isinstance(needed, ndarray_type.NdarrayType):
                pass
            elif isinstance(needed, MatrixType):
                if needed.dtype in primitive_types.real_types:

                    def cast_func(x):
                        if not isinstance(x, (int, float, np.integer, np.floating)):
                            raise TaichiRuntimeTypeError.get(index, needed.dtype.to_string(), type(x))
                        return float(x)

                elif needed.dtype in primitive_types.integer_types:

                    def cast_func(x):
                        if not isinstance(x, (int, np.integer)):
                            raise TaichiRuntimeTypeError.get(index, needed.dtype.to_string(), type(x))
                        return int(x)

                else:
                    raise ValueError(f"Matrix dtype {needed.dtype} is not integer type or real type.")

                if needed.ndim == 2:
                    v = [cast_func(v[i, j]) for i in range(needed.n) for j in range(needed.m)]
                else:
                    v = [cast_func(v[i]) for i in range(needed.n)]
                v = needed(*v)
                needed.set_argpack_struct_args(v, self.__argpack, (index,))
            elif isinstance(needed, StructType):
                if not isinstance(v, needed):
                    raise TaichiRuntimeTypeError.get(index, str(needed), provided)
                needed.set_argpack_struct_args(v, self.__argpack, (index,))
            else:
                raise ValueError(f"Argument type mismatch. Expecting {needed}, got {type(v)}.")


class _IntermediateArgPack(ArgPack):
    """Intermediate argument pack class for compiler internal use only.

    Args:
        annotations (Dict[str, Union[Expr, Matrix, Struct]]): keys and types for struct members.
        entries (Dict[str, Union[Expr, Matrix, Struct]]): keys and values for struct members.
    """

    def __init__(self, annotations, dtype, *args, **kwargs):
        # converts dicts to argument packs
        if len(args) == 1 and kwargs == {} and isinstance(args[0], dict):
            self._ArgPack__entries = args[0]
        elif len(args) == 0:
            self._ArgPack__entries = kwargs
        else:
            raise TaichiSyntaxError(
                "Custom argument packs need to be initialized using either dictionary or keyword arguments"
            )
        if annotations.keys() != self._ArgPack__entries.keys():
            raise TaichiSyntaxError("ArgPack annotations keys not equals to entries keys.")
        self._ArgPack__annotations = annotations
        self._register_members()
        self._ArgPack__dtype = dtype
        self._ArgPack__argpack = impl.get_runtime().prog.create_argpack(dtype)

    def __del__(self):
        pass


class ArgPackType(CompoundType):
    def __init__(self, **kwargs):
        self.members = {}
        elements = []
        for k, dtype in kwargs.items():
            if isinstance(dtype, StructType):
                self.members[k] = dtype
                elements.append([dtype.dtype, k])
            elif isinstance(dtype, ArgPackType):
                self.members[k] = dtype
                elements.append(
                    [
                        _ti_core.DataType(
                            _ti_core.get_type_factory_instance().get_struct_type_for_argpack_ptr(dtype.dtype)
                        ),
                        k,
                    ]
                )
            elif isinstance(dtype, MatrixType):
                # Convert MatrixType to StructType
                if dtype.ndim == 1:
                    elements_ = [(dtype.dtype, f"{k}_{i}") for i in range(dtype.n)]
                else:
                    elements_ = [(dtype.dtype, f"{k}_{i}_{j}") for i in range(dtype.n) for j in range(dtype.m)]
                self.members[k] = dtype
                elements.append([_ti_core.get_type_factory_instance().get_struct_type(elements_), k])
            elif isinstance(dtype, sparse_matrix_builder):
                self.members[k] = dtype
            elif isinstance(dtype, ndarray_type.NdarrayType):
                self.members[k] = dtype
            elif isinstance(dtype, texture_type.RWTextureType):
                self.members[k] = dtype
            elif isinstance(dtype, texture_type.TextureType):
                self.members[k] = dtype
            else:
                dtype = cook_dtype(dtype)
                self.members[k] = dtype
                elements.append([dtype, k])
        if len(elements) == 0:
            # Use i32 as a placeholder for empty argpacks
            elements.append([primitive_types.i32, k])
        self.dtype = _ti_core.get_type_factory_instance().get_argpack_type(elements)

    def __call__(self, *args, **kwargs):
        """Create an instance of this argument pack type."""
        d = {}
        items = self.members.items()
        # iterate over the members of this argument pack
        for index, pair in enumerate(items):
            name, dtype = pair  # (member name, member type))
            if index < len(args):  # set from args
                data = args[index]
            else:  # set from kwargs
                data = kwargs.get(name, None)

            # If dtype is CompoundType and data is a scalar, it cannot be
            # casted in the self.cast call later. We need an initialization here.
            if isinstance(dtype, CompoundType) and not isinstance(data, (dict, ArgPack, Struct)):
                data = dtype(data)

            d[name] = data

        entries = ArgPack(self.members, self.dtype, d)
        pack = self.cast(entries)
        return pack

    def __instancecheck__(self, instance):
        if not isinstance(instance, ArgPack):
            return False
        if list(self.members.keys()) != list(instance._ArgPack__entries.keys()):
            return False
        for k, v in self.members.items():
            if isinstance(v, ArgPackType):
                if not isinstance(instance._ArgPack__entries[k], v):
                    return False
            elif instance._ArgPack__annotations[k] != v:
                return False
        return True

    def cast(self, pack):
        # sanity check members
        if self.members.keys() != pack._ArgPack__entries.keys():
            raise TaichiSyntaxError("Incompatible arguments for custom argument pack members!")
        entries = {}
        for k, dtype in self.members.items():
            if isinstance(dtype, MatrixType):
                entries[k] = dtype(pack._ArgPack__entries[k])
            elif isinstance(dtype, CompoundType):
                entries[k] = dtype.cast(pack._ArgPack__entries[k])
            elif isinstance(dtype, ArgPackType):
                entries[k] = dtype.cast(pack._ArgPack__entries[k])
            elif isinstance(dtype, ndarray_type.NdarrayType):
                entries[k] = pack._ArgPack__entries[k]
            elif isinstance(dtype, texture_type.RWTextureType):
                entries[k] = pack._ArgPack__entries[k]
            elif isinstance(dtype, texture_type.TextureType):
                entries[k] = pack._ArgPack__entries[k]
            elif isinstance(dtype, sparse_matrix_builder):
                entries[k] = pack._ArgPack__entries[k]
            else:
                if in_python_scope():
                    v = pack._ArgPack__entries[k]
                    entries[k] = int(v) if dtype in primitive_types.integer_types else float(v)
                else:
                    entries[k] = ops.cast(pack._ArgPack__entries[k], dtype)
        pack = ArgPack(self.members, self.dtype, entries)
        return pack

    def from_taichi_object(self, arg_load_dict: dict):
        d = {}
        items = self.members.items()
        for index, pair in enumerate(items):
            name, dtype = pair
            d[name] = arg_load_dict[name]
        pack = _IntermediateArgPack(self.members, self.dtype, d)
        pack._ArgPack__dtype = self.dtype
        return pack

    def __str__(self):
        """Python scope argpack type print support."""
        item_str = ", ".join([str(k) + "=" + str(v) for k, v in self.members.items()])
        return f"<ti.ArgPackType {item_str}>"


__all__ = ["ArgPack"]
