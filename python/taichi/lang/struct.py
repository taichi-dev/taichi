import copy
import numbers

from numpy import broadcast
from taichi.lang import expr, impl
from taichi.lang.common_ops import TaichiOperations
from taichi.lang.enums import Layout
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.field import Field, ScalarField, SNodeHostAccess
from taichi.lang.matrix import Matrix
from taichi.lang.ops import cast
from taichi.lang.types import CompoundType
from taichi.lang.util import (cook_dtype, in_python_scope, is_taichi_class,
                              python_scope, taichi_scope)

import taichi as ti


class Struct(TaichiOperations):
    """The Struct type class.
    Args:
        entries (Dict[str, Union[Dict, Expr, Matrix, Struct]]): keys and values for struct members.
    """
    is_taichi_class = True

    def __init__(self, *args, **kwargs):
        # converts lists to matrices and dicts to structs
        if len(args) == 1 and kwargs == {} and isinstance(args[0], dict):
            self.entries = args[0]
        elif len(args) == 0:
            self.entries = kwargs
        else:
            raise TaichiSyntaxError(
                "Custom structs need to be initialized using either dictionary or keyword arguments"
            )
        for k, v in self.entries.items():
            if isinstance(v, (list, tuple)):
                v = Matrix(v)
            if isinstance(v, dict):
                v = Struct(v)
            self.entries[k] = v
        self.register_members()
        self.local_tensor_proxy = None
        self.any_array_access = None

    @property
    def keys(self):
        return list(self.entries.keys())

    @property
    def members(self):
        return list(self.entries.values())

    @property
    def items(self):
        return self.entries.items()

    def register_members(self):
        for k in self.keys:
            setattr(Struct, k,
                    property(
                        Struct.make_getter(k),
                        Struct.make_setter(k),
                    ))

    def __getitem__(self, key):
        _taichi_skip_traceback = 1
        ret = self.entries[key]
        if isinstance(ret, SNodeHostAccess):
            ret = ret.accessor.getter(*ret.key)
        return ret

    def __setitem__(self, key, value):
        _taichi_skip_traceback = 1
        if isinstance(self.entries[key], SNodeHostAccess):
            self.entries[key].accessor.setter(value, *self.entries[key].key)
        else:
            if in_python_scope():
                self.entries[key].set_entries(value)
            else:
                self.entries[key] = value

    def set_entries(self, value):
        if isinstance(value, dict):
            value = Struct(value)
        for k in self.keys:
            self[k] = value[k]

    @staticmethod
    def make_getter(key):
        def getter(self):
            """Get an entry from custom struct by name."""
            _taichi_skip_traceback = 1
            return self[key]

        return getter

    @staticmethod
    def make_setter(key):
        @python_scope
        def setter(self, value):
            _taichi_skip_traceback = 1
            self[key] = value

        return setter

    def element_wise_unary(self, foo):
        _taichi_skip_traceback = 1
        ret = self.empty_copy()
        for k, v in self.items:
            if isinstance(v, expr.Expr):
                ret.entries[k] = foo(v)
            else:
                ret.entries[k] = v.element_wise_unary(foo)
        return ret

    def element_wise_binary(self, foo, other):
        _taichi_skip_traceback = 1
        ret = self.empty_copy()
        if isinstance(other, (dict)):
            other = Struct(other)
        if isinstance(other, Struct):
            if self.entries.keys() != other.entries.keys():
                raise TypeError(
                    f"Member mismatch between structs {self.keys}, {other.keys}"
                )
            for k, v in self.items:
                if isinstance(v, expr.Expr):
                    ret.entries[k] = foo(v, other.entries[k])
                else:
                    ret.entries[k] = v.element_wise_binary(
                        foo, other.entries[k])
        else:  # assumed to be scalar
            for k, v in self.items:
                if isinstance(v, expr.Expr):
                    ret.entries[k] = foo(v, other)
                else:
                    ret.entries[k] = v.element_wise_binary(foo, other)
        return ret

    def broadcast_copy(self, other):
        if isinstance(other, dict):
            other = Struct(other)
        if not isinstance(other, Struct):
            ret = self.empty_copy()
            for k, v in ret.items:
                if isinstance(v, (Matrix, Struct)):
                    ret.entries[k] = v.broadcast_copy(other)
                else:
                    ret.entries[k] = other
            other = ret
        if self.entries.keys() != other.entries.keys():
            raise TypeError(
                f"Member mismatch between structs {self.keys}, {other.keys}")
        return other

    def element_wise_writeback_binary(self, foo, other):
        ret = self.empty_copy()
        if isinstance(other, (dict)):
            other = Struct(other)
        if is_taichi_class(other):
            other = other.variable()
        if foo.__name__ == 'assign' and not isinstance(other, Struct):
            raise TaichiSyntaxError(
                'cannot assign scalar expr to '
                f'taichi class {type(self)}, maybe you want to use `a.fill(b)` instead?'
            )
        if isinstance(other, Struct):
            if self.entries.keys() != other.entries.keys():
                raise TypeError(
                    f"Member mismatch between structs {self.keys}, {other.keys}"
                )
            for k, v in self.items:
                if isinstance(v, expr.Expr):
                    ret.entries[k] = foo(v, other.entries[k])
                else:
                    ret.entries[k] = v.element_wise_binary(
                        foo, other.entries[k])
        else:  # assumed to be scalar
            for k, v in self.items:
                if isinstance(v, expr.Expr):
                    ret.entries[k] = foo(v, other)
                else:
                    ret.entries[k] = v.element_wise_binary(foo, other)
        return ret

    def element_wise_ternary(self, foo, other, extra):
        ret = self.empty_copy()
        other = self.broadcast_copy(other)
        extra = self.broadcast_copy(extra)
        for k, v in self.items:
            if isinstance(v, expr.Expr):
                ret.entries[k] = foo(v, other.entries[k], extra.entries[k])
            else:
                ret.entries[k] = v.element_wise_ternary(
                    foo, other.entries[k], extra.entries[k])
        return ret

    @taichi_scope
    def fill(self, val):
        """Fills the Struct with a specific value in Taichi scope.

        Args:
            val (Union[int, float]): Value to fill.
        """
        def assign_renamed(x, y):
            return ti.assign(x, y)

        return self.element_wise_writeback_binary(assign_renamed, val)

    def empty_copy(self):
        """
        Nested structs and matrices need to be recursively handled.
        """
        struct = Struct.empty(self.keys)
        for k, v in self.items:
            if isinstance(v, (Struct, Matrix)):
                struct.entries[k] = v.empty_copy()
        return struct

    def copy(self):
        ret = self.empty_copy()
        ret.entries = copy.copy(self.entries)
        return ret

    @taichi_scope
    def variable(self):
        ret = self.copy()
        ret.entries = {
            k: impl.expr_init(v) if isinstance(v,
                                               (numbers.Number,
                                                expr.Expr)) else v.variable()
            for k, v in ret.items
        }
        return ret

    def __len__(self):
        """Get the number of entries in a custom struct"""
        return len(self.entries)

    def __iter__(self):
        return self.entries.values()

    def __str__(self):
        """Python scope struct array print support."""
        if impl.inside_kernel():
            item_str = ", ".join(
                [str(k) + "=" + str(v) for k, v in self.items])
            return f'<ti.Struct {item_str}>'
        else:
            return str(self.to_dict())

    def __repr__(self):
        return str(self.to_dict())

    @python_scope
    def to_dict(self):
        """Converts the Struct to a dictionary.

        Args:

        Returns:
            Dict: The result dictionary.
        """
        return self.entries

    @classmethod
    def empty(cls, entries):
        """Clear the struct and fill None.

        Args:
            members (Dict[str, DataType]): the names and data types for struct members.
        Returns:
            :class:`~taichi.lang.struct.Struct`: A :class:`~taichi.lang.struct.Struct` instance filled with None.

        """
        return cls({k: None for k in entries})

    @classmethod
    @python_scope
    def field(cls,
              members,
              shape=None,
              name="<Struct>",
              offset=None,
              needs_grad=False,
              layout=Layout.AOS):

        if shape is None and offset is not None:
            raise TaichiSyntaxError(
                "shape cannot be None when offset is being set")

        field_dict = {}

        for key, dtype in members.items():
            field_name = name + '.' + key
            if isinstance(dtype, CompoundType):
                field_dict[key] = dtype.field(shape=None,
                                              name=field_name,
                                              offset=offset,
                                              needs_grad=needs_grad)
            else:
                field_dict[key] = impl.field(dtype,
                                             shape=None,
                                             name=field_name,
                                             offset=offset,
                                             needs_grad=needs_grad)

        if shape is not None:
            if isinstance(shape, numbers.Number):
                shape = (shape, )
            if isinstance(offset, numbers.Number):
                offset = (offset, )

            if offset is not None and len(shape) != len(offset):
                raise TaichiSyntaxError(
                    f'The dimensionality of shape and offset must be the same ({len(shape)} != {len(offset)})'
                )
            dim = len(shape)
            if layout == Layout.SOA:
                for e in field_dict.values():
                    ti.root.dense(impl.index_nd(dim),
                                  shape).place(e, offset=offset)
                if needs_grad:
                    for e in field_dict.values():
                        ti.root.dense(impl.index_nd(dim),
                                      shape).place(e.grad, offset=offset)
            else:
                ti.root.dense(impl.index_nd(dim),
                              shape).place(*tuple(field_dict.values()),
                                           offset=offset)
                if needs_grad:
                    grads = tuple(e.grad for e in field_dict.values())
                    ti.root.dense(impl.index_nd(dim),
                                  shape).place(*grads, offset=offset)
        return StructField(field_dict, name=name)


class StructField(Field):
    """Taichi struct field with SNode implementation.
       Instead of directly contraining Expr entries, the StructField object
       directly hosts members as `Field` instances to support nested structs.

    Args:
        field_dict (Dict[str, Field]): Struct field members.
        name (string, optional): The custom name of the field.
    """
    def __init__(self, field_dict, name=None):
        # will not call Field initializer
        self.field_dict = field_dict
        self._name = name
        self.register_fields()

    @property
    def name(self):
        return self._name

    @property
    def keys(self):
        return list(self.field_dict.keys())

    @property
    def members(self):
        return list(self.field_dict.values())

    @property
    def items(self):
        return self.field_dict.items()

    @staticmethod
    def make_getter(key):
        def getter(self):
            """Get an entry from custom struct by name."""
            _taichi_skip_traceback = 1
            return self.field_dict[key]

        return getter

    @staticmethod
    def make_setter(key):
        @python_scope
        def setter(self, value):
            _taichi_skip_traceback = 1
            self.field_dict[key] = value

        return setter

    def register_fields(self):
        for k in self.keys:
            setattr(
                StructField, k,
                property(
                    StructField.make_getter(k),
                    StructField.make_setter(k),
                ))

    def get_field_members(self):
        """Get A flattened list of all struct elements.

        Returns:
            A list of struct elements.
        """
        field_members = []
        for m in self.members:
            assert isinstance(m, Field)
            field_members += m.get_field_members()
        return field_members

    @property
    def snode(self):
        """Gets representative SNode for info purposes.

        Returns:
            SNode: Representative SNode (SNode of first field member).
        """
        return self.members[0].snode

    def loop_range(self):
        """Gets representative field member for loop range info.

        Returns:
            taichi_core.Expr: Representative (first) field member.
        """
        return self.members[0].loop_range()

    @python_scope
    def copy_from(self, other):
        """Copies all elements from another field.

        The shape of the other field needs to be the same as `self`.

        Args:
            other (Field): The source field.
        """
        assert isinstance(other, Field)
        assert set(self.keys) == set(other.keys)
        for k in self.keys:
            self.field_dict[k].copy_from(other[k])

    @python_scope
    def fill(self, val):
        """Fills `self` with a specific value.

        Args:
            val (Union[int, float]): Value to fill.
        """
        for v in self.members:
            v.fill(val)

    def initialize_host_accessors(self):
        for v in self.members:
            v.initialize_host_accessors()

    def get_member_field(self, key):
        """Creates a ScalarField using a specific field member. Only used for quant.

        Args:
            key (str): Specified key of the field member.

        Returns:
            ScalarField: The result ScalarField.
        """
        return self.field_dict[key]

    @python_scope
    def from_numpy(self, array_dict):
        for k, v in self.items:
            v.from_numpy(array_dict[k])

    @python_scope
    def from_torch(self, array_dict):
        for k, v in self.items:
            v.from_torch(array_dict[k])

    @python_scope
    def to_numpy(self):
        """Converts the Struct field instance to a dictionary of NumPy arrays. The dictionary may be nested when converting
           nested structs.

        Args:
        Returns:
            Dict[str, Union[numpy.ndarray, Dict]]: The result NumPy array.
        """
        return {k: v.to_numpy() for k, v in self.items}

    @python_scope
    def to_torch(self, device=None):
        """Converts the Struct field instance to a dictionary of PyTorch tensors. The dictionary may be nested when converting
           nested structs.

        Args:
            device (torch.device, optional): The desired device of returned tensor.
        Returns:
            Dict[str, Union[torch.Tensor, Dict]]: The result PyTorch tensor.
        """
        return {k: v.to_torch(device=device) for k, v in self.items}

    @python_scope
    def __setitem__(self, indices, element):
        self.initialize_host_accessors()
        self[indices].set_entries(element)

    @python_scope
    def __getitem__(self, indices):
        self.initialize_host_accessors()
        # scalar fields does not instantiate SNodeHostAccess by default
        entries = {
            k: v.host_access(self.pad_key(indices))[0] if isinstance(
                v, ScalarField) else v[indices]
            for k, v in self.items
        }
        return Struct(entries)


class StructType(CompoundType):
    def __init__(self, **kwargs):
        self.members = {}
        for k, dtype in kwargs.items():
            if isinstance(dtype, CompoundType):
                self.members[k] = dtype
            else:
                self.members[k] = cook_dtype(dtype)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            if kwargs == {}:
                raise TaichiSyntaxError(
                    "Custom type instances need to be created with an initial value."
                )
            else:
                # initialize struct members by keywords
                entries = Struct(kwargs)
        elif len(args) == 1:
            # fill a single scalar
            if isinstance(args[0], (numbers.Number, expr.Expr)):
                entries = self.scalar_filled(args[0])
            else:
                # fill a single vector or matrix
                # initialize struct members by dictionary
                entries = Struct(args[0])
        struct = self.cast(entries)
        return struct

    def cast(self, struct, in_place=False):
        if not in_place:
            struct = struct.copy()
        # sanity check members
        if self.members.keys() != struct.entries.keys():
            raise TaichiSyntaxError(
                "Incompatible arguments for custom struct members!")
        for k, dtype in self.members.items():
            if isinstance(dtype, CompoundType):
                struct.entries[k] = dtype.cast(struct.entries[k])
            else:
                if in_python_scope():
                    v = struct.entries[k]
                    struct.entries[k] = int(
                        v) if dtype in ti.integer_types else float(v)
                else:
                    struct.entries[k] = cast(struct.entries[k], dtype)
        return struct

    def empty(self):
        """
        Create an empty instance of the given compound type.
        Nested structs and matrices need to be recursively handled.
        """
        struct = Struct.empty(self.members.keys())
        for k, dtype in self.members.items():
            if isinstance(dtype, CompoundType):
                struct.entries[k] = dtype.empty()
        return struct

    def field(self, **kwargs):
        return Struct.field(self.members, **kwargs)
