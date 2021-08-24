import copy
import numbers

from taichi.lang import expr, impl
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.field import Field, SNodeHostAccess
from taichi.lang.matrix import Matrix
from taichi.lang.types import CompoundType
from taichi.lang.util import python_scope, taichi_scope


import taichi as ti


class Struct:
    """The Struct type class.
    Args:
        entries (Dict[str, Union[Dict, Expr, Matrix, Struct]]): keys and values for struct members.
    """
    is_taichi_class = True

    def __init__(self, entries):
        # converts lists to matrices and dicts to structs
        self.entries = {}
        for k, v in entries.items:
            if isinstance(v, (list, tuple)):
                v = Matrix(v)
            if isinstance(v, dict):
                v = Struct(v)
            self.entries[k] = v
        self.register_members()

    @property
    def keys(self):
        return list(self.entries.keys())

    @property
    def members(self):
        return list(self.entries.values())

    def items(self):
        return self.entries.items()

    def empty_copy(self):
        return Struct.empty(self.members)

    def copy(self):
        ret = self.empty_copy()
        ret.entries = copy.copy(self.entries)
        return ret

    def register_members(self):
        for k in self.keys:
            setattr(
                Struct, k,
                property(
                    Struct.make_getter(k),
                    Struct.make_setter(k),
                ))

    def __call__(self, key, **kwargs):
        _taichi_skip_traceback = 1
        assert kwargs == {}
        ret = self.entries[key]
        if isinstance(ret, SNodeHostAccess):
            ret = ret.accessor.getter(*ret.key)
        return ret


    @staticmethod
    def make_getter(key):
        def getter(self):
            """Get an entry from custom struct by name."""
            _taichi_skip_traceback = 1
            return self.entries[key]
        return getter

    @staticmethod
    def make_setter(key):
        @python_scope
        def setter(self, value):
            _taichi_skip_traceback = 1
            self.entries[key] = value
        return setter


    def __len__(self):
        """Get the number of entries in a custom struct"""
        return len(self.entries)

    def __iter__(self):
        return self.entries.values()

    def __str__(self):
        """Python scope struct array print support."""
        if impl.inside_kernel():
            return f'<ti.Struct {", ".join([str(k) + "=" + str(v) for k, v in self.entries])}>'
        else:
            return str(self.to_numpy())

    def __repr__(self):
        return str(self.to_numpy())

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
              struct_name="<Struct>",
              offset=None,
              needs_grad=False,
              layout=None):       

        if layout is not None:
            assert shape is not None, 'layout is useless without shape'
        if shape is None:
            assert offset is None, "shape cannot be None when offset is being set"

        field_dict = {}

        for key, dtype in members.items():
            name = struct_name + '.' + key
            if isinstance(dtype, CompoundType):
                field_dict[key] = dtype.field(shape=None, name=name, offset=offset, needs_grad=needs_grad)
            else:
                field_dict[key] = impl.field(dtype, shape=None, name=name, offset=offset, needs_grad=needs_grad)

        if shape is not None:
            if isinstance(shape, numbers.Number):
                shape = (shape, )
            if isinstance(offset, numbers.Number):
                offset = (offset, )

            if offset is not None:
                assert len(shape) == len(
                    offset
                ), f'The dimensionality of shape and offset must be the same  ({len(shape)} != {len(offset)})'

            if layout is None:
                layout = ti.AOS

            dim = len(shape)
            if layout.soa:
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
                                  shape).place(*grads,
                                               offset=offset)
        return StructField(field_dict, name=struct_name)


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
        self.register_fields()

    @property
    def keys(self):
        return list(self.field_dict.keys())

    @property
    def members(self):
        return list(self.field_dict.values())

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
            self[k].copy_from(other[k])

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
        for k in self.keys:
            self(k).from_numpy(array_dict[k])

    @python_scope
    def from_torch(self, array_dict):
        for k in self.keys:
            self(k).from_torch(array_dict[k])

    @python_scope
    def to_numpy(self):
        """Converts the Struct field instance to a dictionary of NumPy arrays. The dictionary may be nested when converting
           nested structs.

        Args:
        Returns:
            Dict[str, Union[numpy.ndarray, Dict]]: The result NumPy array.
        """
        return {k: v.to_numpy() for k, v in self.field_dict.items()}

    @python_scope
    def to_torch(self, device):
        """Converts the Struct field instance to a dictionary of PyTorch tensors. The dictionary may be nested when converting
           nested structs.

        Args:
            device (torch.device, optional): The desired device of returned tensor.
        Returns:
            Dict[str, Union[torch.Tensor, Dict]]: The result PyTorch tensor.
        """
        return {k: v.to_torch(device=device) for k, v in self.field_dict.items()}

    
    @python_scope
    def __setitem__(self, indices, element):
        self.initialize_host_accessors()
        for k, v in element.values():
            self[indices][k] = v
    
    @python_scope
    def __getitem__(self, indices):
        self.initialize_host_accessors()
        indices = self.pad_key(indices)
        entries = {
            k: v[indices] for k, v in self.field_dict.items()
        }
        return Struct(entries)
    

class StructType(CompoundType):
    
    def __init__(self, **kwargs):
        self.members = kwargs

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            if kwargs == {}:
                raise TaichiSyntaxError("Custom type instances need to be created with an initial value.")
            else:
                # initialize struct members by keywords
                entries = kwargs
        elif len(args) == 1:
            # fill a single scalar
            if isinstance(args[0], numbers.Number):
                return self.scalar_filled(args[0])
            # fill a single vector or matrix
            # initialize struct members by dictionary
            entries = args[0]
        struct = self.cast(Struct(entries))
        return struct

    def cast(self, struct, in_place=False):
        if not in_place:
            struct = struct.copy()
        # sanity check members
        if self.members.keys() != struct.entries.keys():
            raise TaichiSyntaxError("Incompatible arguments for custom struct members!")
        for k, dtype in self.members.items():
            if isinstance(dtype, CompoundType):
                struct.entries[k] = dtype.cast(struct.entries[k])
            else:
                struct.entries[k] = cast(struct.entries[k], dtype)
        return struct
        
    def empty(self):
        """
        Create an empty instance of the given compound type.
        """
        return Struct.empty(self.members)
    
    def field(self, **kwargs):
        return Struct.field(self.m, self.n, **kwargs)
