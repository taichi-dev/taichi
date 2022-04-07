import numbers

from taichi.lang import expr, impl, ops
from taichi.lang.common_ops import TaichiOperations
from taichi.lang.enums import Layout
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.field import Field, ScalarField, SNodeHostAccess
from taichi.lang.matrix import Matrix
from taichi.lang.util import (cook_dtype, in_python_scope, is_taichi_class,
                              python_scope, taichi_scope)
from taichi.types import primitive_types
from taichi.types.compound_types import CompoundType


class Struct(TaichiOperations):
    """The Struct type class.

    A struct is a dictionary-like data structure that stores members as
    (key, value) pairs. Valid data members of a struct can be scalars,
    matrices or other dictionary-like stuctures.

    Args:
        entries (Dict[str, Union[Dict, Expr, Matrix, Struct]]): \
            keys and values for struct members.

    Returns:
        An instance of this struct.

    Example::

        >>> vec3 = ti.types.vector(3, ti.f32)
        >>> a = ti.Struct(v=vec3([0, 0, 0]), t=1.0)
        >>> print(a.items)
        dict_items([('v', [0. 0. 0.]), ('t', 1.0)])
        >>>
        >>> B = ti.Struct(v=vec3([0., 0., 0.]), t=1.0, A=a)
        >>> print(B.items)
        dict_items([('v', [0. 0. 0.]), ('t', 1.0), ('A', {'v': [[0.], [0.], [0.]], 't': 1.0})])
    """
    _is_taichi_class = True

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
            self.entries[k] = v if in_python_scope() else impl.expr_init(v)
        self._register_members()

    @property
    def keys(self):
        """Returns the list of member names in string format.

        Example::

           >>> vec3 = ti.types.vector(3, ti.f32)
           >>> sphere = ti.Struct(center=vec3([0, 0, 0]), radius=1.0)
           >>> a.keys
           ['center', 'radius']
        """
        return list(self.entries.keys())

    @property
    def _members(self):
        return list(self.entries.values())

    @property
    def items(self):
        """Returns the items in this struct.

        Example::

            >>> vec3 = ti.types.vector(3, ti.f32)
            >>> sphere = ti.Struct(center=vec3([0, 0, 0]), radius=1.0)
            >>> sphere.items
            dict_items([('center', 2), ('radius', 1.0)])
        """
        return self.entries.items()

    def _register_members(self):
        for k in self.keys:
            setattr(Struct, k,
                    property(
                        Struct._make_getter(k),
                        Struct._make_setter(k),
                    ))

    def __getitem__(self, key):
        ret = self.entries[key]
        if isinstance(ret, SNodeHostAccess):
            ret = ret.accessor.getter(*ret.key)
        return ret

    def __setitem__(self, key, value):
        if isinstance(self.entries[key], SNodeHostAccess):
            self.entries[key].accessor.setter(value, *self.entries[key].key)
        else:
            if in_python_scope():
                if isinstance(self.entries[key], Struct) or isinstance(
                        self.entries[key], Matrix):
                    self.entries[key]._set_entries(value)
                else:
                    if isinstance(value, numbers.Number):
                        self.entries[key] = value
                    else:
                        raise TypeError(
                            "A number is expected when assigning struct members"
                        )
            else:
                self.entries[key] = value

    def _set_entries(self, value):
        if isinstance(value, dict):
            value = Struct(value)
        for k in self.keys:
            self[k] = value[k]

    @staticmethod
    def _make_getter(key):
        def getter(self):
            """Get an entry from custom struct by name."""
            return self[key]

        return getter

    @staticmethod
    def _make_setter(key):
        @python_scope
        def setter(self, value):
            self[key] = value

        return setter

    def _element_wise_unary(self, foo):
        entries = {}
        for k, v in self.items:
            if is_taichi_class(v):
                entries[k] = v._element_wise_unary(foo)
            else:
                entries[k] = foo(v)
        return Struct(entries)

    def _element_wise_binary(self, foo, other):
        other = self._broadcast_copy(other)
        entries = {}
        for k, v in self.items:
            if is_taichi_class(v):
                entries[k] = v._element_wise_binary(foo, other.entries[k])
            else:
                entries[k] = foo(v, other.entries[k])
        return Struct(entries)

    def _broadcast_copy(self, other):
        if isinstance(other, dict):
            other = Struct(other)
        if not isinstance(other, Struct):
            entries = {}
            for k, v in self.items:
                if is_taichi_class(v):
                    entries[k] = v._broadcast_copy(other)
                else:
                    entries[k] = other
            other = Struct(entries)
        if self.entries.keys() != other.entries.keys():
            raise TypeError(
                f"Member mismatch between structs {self.keys}, {other.keys}")
        return other

    def _element_wise_writeback_binary(self, foo, other):
        if foo.__name__ == 'assign' and not isinstance(other, (dict, Struct)):
            raise TaichiSyntaxError(
                'cannot assign scalar expr to '
                f'taichi class {type(self)}, maybe you want to use `a.fill(b)` instead?'
            )
        other = self._broadcast_copy(other)
        entries = {}
        for k, v in self.items:
            if is_taichi_class(v):
                entries[k] = v._element_wise_binary(foo, other.entries[k])
            else:
                entries[k] = foo(v, other.entries[k])
        return self if foo.__name__ == 'assign' else Struct(entries)

    def _element_wise_ternary(self, foo, other, extra):
        other = self._broadcast_copy(other)
        extra = self._broadcast_copy(extra)
        entries = {}
        for k, v in self.items:
            if is_taichi_class(v):
                entries[k] = v._element_wise_ternary(foo, other.entries[k],
                                                     extra.entries[k])
            else:
                entries[k] = foo(v, other.entries[k], extra.entries[k])
        return Struct(entries)

    @taichi_scope
    def fill(self, val):
        """Fills the Struct with a specific value in Taichi scope.

        Args:
            val (Union[int, float]): Value to fill.
        """
        def assign_renamed(x, y):
            return ops.assign(x, y)

        return self._element_wise_writeback_binary(assign_renamed, val)

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
        return str(self.to_dict())

    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self):
        """Converts the Struct to a dictionary.

        Args:

        Returns:
            Dict: The result dictionary.
        """
        return {
            k: v.to_dict() if isinstance(v, Struct) else
            v.to_list() if isinstance(v, Matrix) else v
            for k, v in self.entries.items()
        }

    @classmethod
    @python_scope
    def field(cls,
              members,
              shape=None,
              name="<Struct>",
              offset=None,
              needs_grad=False,
              layout=Layout.AOS):
        """Creates a :class:`~taichi.StructField` with each element
        has this struct as its type.

        Args:
            members (dict): a dict, each item is like `name: type`.
            shape (Tuple[int]): width and height of the field.
            offset (Tuple[int]): offset of the indices of the created field.
                For example if `offset=(-10, -10)` the indices of the field
                will start at `(-10, -10)`, not `(0, 0)`.
            needs_grad (bool): enabling gradient field or not.
            layout: AOS or SOA.

        Example:

            >>> vec3 = ti.types.vector(3, ti.f32)
            >>> sphere = {"center": vec3, "radius": float}
            >>> F = ti.Struct.field(sphere, shape=(3, 3))
            >>> F
            {'center': array([[[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]],

               [[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]],

               [[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]]], dtype=float32), 'radius': array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]], dtype=float32)}
        """

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
                    impl.root.dense(impl.index_nd(dim),
                                    shape).place(e, offset=offset)
                if needs_grad:
                    for e in field_dict.values():
                        impl.root.dense(impl.index_nd(dim),
                                        shape).place(e.grad, offset=offset)
            else:
                impl.root.dense(impl.index_nd(dim),
                                shape).place(*tuple(field_dict.values()),
                                             offset=offset)
                if needs_grad:
                    grads = tuple(e.grad for e in field_dict.values())
                    impl.root.dense(impl.index_nd(dim),
                                    shape).place(*grads, offset=offset)
        return StructField(field_dict, name=name)


class _IntermediateStruct(Struct):
    """Intermediate struct class for compiler internal use only.

    Args:
        entries (Dict[str, Union[Expr, Matrix, Struct]]): keys and values for struct members.
    """
    def __init__(self, entries):
        assert isinstance(entries, dict)
        self.entries = entries
        self._register_members()


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
        self.name = name
        self._register_fields()

    @property
    def keys(self):
        """Returns the list of names of the field members.

        Example::

            >>> f1 = ti.Vector.field(3, ti.f32, shape=(3, 3))
            >>> f2 = ti.field(ti.f32, shape=(3, 3))
            >>> F = ti.StructField({"center": f1, "radius": f2})
            >>> F.keys
            ['center', 'radius']
        """
        return list(self.field_dict.keys())

    @property
    def _members(self):
        return list(self.field_dict.values())

    @property
    def _items(self):
        return self.field_dict.items()

    @staticmethod
    def _make_getter(key):
        def getter(self):
            """Get an entry from custom struct by name."""
            return self.field_dict[key]

        return getter

    @staticmethod
    def _make_setter(key):
        @python_scope
        def setter(self, value):
            self.field_dict[key] = value

        return setter

    def _register_fields(self):
        for k in self.keys:
            setattr(
                StructField, k,
                property(
                    StructField._make_getter(k),
                    StructField._make_setter(k),
                ))

    def _get_field_members(self):
        """Gets A flattened list of all struct elements.

        Returns:
            A list of struct elements.
        """
        field_members = []
        for m in self._members:
            assert isinstance(m, Field)
            field_members += m._get_field_members()
        return field_members

    @property
    def _snode(self):
        """Gets representative SNode for info purposes.

        Returns:
            SNode: Representative SNode (SNode of first field member).
        """
        return self._members[0]._snode

    def _loop_range(self):
        """Gets representative field member for loop range info.

        Returns:
            taichi_core.Expr: Representative (first) field member.
        """
        return self._members[0]._loop_range()

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
        """Fills this struct field with a specified value.

        Args:
            val (Union[int, float]): Value to fill.
        """
        for v in self._members:
            v.fill(val)

    def _initialize_host_accessors(self):
        for v in self._members:
            v._initialize_host_accessors()

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
        """Copies the data from a set of `numpy.array` into this field.

        The argument `array_dict` must be a dictionay-like object, it
        contains all the keys in this field and the copying process
        between corresponding items can be performed.
        """
        for k, v in self._items:
            v.from_numpy(array_dict[k])

    @python_scope
    def from_torch(self, array_dict):
        """Copies the data from a set of `torch.tensor` into this field.

        The argument `array_dict` must be a dictionay-like object, it
        contains all the keys in this field and the copying process
        between corresponding items can be performed.
        """
        for k, v in self._items:
            v.from_torch(array_dict[k])

    @python_scope
    def to_numpy(self):
        """Converts the Struct field instance to a dictionary of NumPy arrays.

        The dictionary may be nested when converting nested structs.

        Returns:
            Dict[str, Union[numpy.ndarray, Dict]]: The result NumPy array.
        """
        return {k: v.to_numpy() for k, v in self._items}

    @python_scope
    def to_torch(self, device=None):
        """Converts the Struct field instance to a dictionary of PyTorch tensors.

        The dictionary may be nested when converting nested structs.

        Args:
            device (torch.device, optional): The
                desired device of returned tensor.

        Returns:
            Dict[str, Union[torch.Tensor, Dict]]: The result
                PyTorch tensor.
        """
        return {k: v.to_torch(device=device) for k, v in self._items}

    @python_scope
    def __setitem__(self, indices, element):
        self._initialize_host_accessors()
        self[indices]._set_entries(element)

    @python_scope
    def __getitem__(self, indices):
        self._initialize_host_accessors()
        # scalar fields does not instantiate SNodeHostAccess by default
        entries = {
            k: v._host_access(self._pad_key(indices))[0] if isinstance(
                v, ScalarField) else v[indices]
            for k, v in self._items
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
                entries = self.filled_with_scalar(args[0])
            else:
                # initialize struct members by dictionary
                entries = Struct(args[0])
        struct = self.cast(entries)
        return struct

    def cast(self, struct):
        # sanity check members
        if self.members.keys() != struct.entries.keys():
            raise TaichiSyntaxError(
                "Incompatible arguments for custom struct members!")
        entries = {}
        for k, dtype in self.members.items():
            if isinstance(dtype, CompoundType):
                entries[k] = dtype.cast(struct.entries[k])
            else:
                if in_python_scope():
                    v = struct.entries[k]
                    entries[k] = int(
                        v
                    ) if dtype in primitive_types.integer_types else float(v)
                else:
                    entries[k] = ops.cast(struct.entries[k], dtype)
        return Struct(entries)

    def filled_with_scalar(self, value):
        entries = {}
        for k, dtype in self.members.items():
            if isinstance(dtype, CompoundType):
                entries[k] = dtype.filled_with_scalar(value)
            else:
                entries[k] = value
        return Struct(entries)

    def field(self, **kwargs):
        return Struct.field(self.members, **kwargs)


__all__ = ["Struct", "StructField"]
