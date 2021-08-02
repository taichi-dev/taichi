
import numbers

from taichi.lang import expr, impl
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.util import (python_scope, taichi_scope)

import taichi as ti

class Struct:
    """The Struct type class.
    Args:
        entries (Dict[str, Any]): the names and data types for struct members.
    """
    is_taichi_class = True

    def __init__(self, entries):
        self.entries = entries

    def members(self):
        return self.entries.keys()

    def empty_copy(self):
        return Struct.empty(self.members())


    def is_global(self):
        results = [False for _ in self.entries.values()]
        for i, e in enumerate(self.entries.values()):
            if isinstance(e, expr.Expr):
                if e.is_global():
                    results[i] = True
            assert results[i] == results[0], \
                "Structs with mixed global/local entries are not allowed"
        return results[0]

    def get_field_members(self):
        """Get struct elements list.

        Returns:
            A list of struct elements.
        """
        return list(self.entries.values())

    def register_member(self, member_name, dtype):
        self.entries[member_name] = impl.field(dtype, name=self.name + '.' + member_name)
        setattr(Struct, member_name, 
            property(
                Struct.make_getter(member_name),
                Struct.make_setter(member_name),
            )
        )

    def __call__(self, name):
        _taichi_skip_traceback = 1
        return self.entries[name]

    @taichi_scope
    def subscript(self, *indices):
        _taichi_skip_traceback = 1
        if self.is_global():
            ret = self.empty_copy()
            for k, e in self.entries.items():
                ret.entries[k] = impl.subscript(e, *indices)
            return ret
        else:
            raise TaichiSyntaxError("Custom struct members cannot be locally subscripted")

    def make_grad(self):
        ret = self.empty_copy()
        for k in ret.members():
            ret.entries[k] = self.entries[k].grad
        return ret

    @staticmethod
    def make_getter(member_name):
        def getter(self):
            """Get an entry from custom struct by name."""
            _taichi_skip_traceback = 1
            return self.entries[member_name]
        return getter

    @staticmethod
    def make_setter(member_name):
        @python_scope
        def setter(self, value):
            _taichi_skip_traceback = 1
            self.entries[member_name] = value
        return setter

    class Proxy:
        def __init__(self, struct, index):
            """Proxy when a tensor of Structs is accessed by host."""
            self.struct = struct
            self.index = index
            for member_name in self.struct.members():
                setattr(Struct.Proxy, member_name, 
                    property(
                        Struct.Proxy.make_getter(member_name),
                        Struct.Proxy.make_setter(member_name),
                    )
                )

        @python_scope
        def _get(self, name):
            return self.struct(name)[self.index]

        @python_scope
        def _set(self, name, value):
            self.struct(name)[self.index] = value

        @staticmethod
        def make_getter(member_name):
            @python_scope
            def getter(self):
                return self.struct(member_name)[self.index]
            return getter

        @staticmethod
        def make_setter(member_name):
            @python_scope
            def setter(self, value):
                self.struct(member_name)[self.index] = value
            return setter
        
        @property
        def value(self):
            ret = self.struct.empty_copy()
            for k in self.struct.members():
                ret.entries[k] = self.struct(k)[self.index]
            return ret
    
    # host access & python scope operation
    @python_scope
    def __getitem__(self, indices):
        """Access to the element at the given indices in a struct array.

        Args:
            indices (Sequence[Expr]): the indices of the element.

        Returns:
            The value of the element at a specific position of a struct array.

        """
        if self.is_global():
            return Struct.Proxy(self, indices)
        else:
            raise TaichiSyntaxError("Custom struct members cannot be locally subscripted")
        
    @python_scope
    def __setitem__(self, indices, item):
        raise NotImplementedError("Cannot assign the whole struct in Python scope")

    def __len__(self):
        """Get the number of entries in a custom struct"""
        return len(self.entries)

    def __iter__(self):
        return self.entries.values()

    def loop_range(self):
        return list(self.entries.values())[0]

    @property
    def shape(self):
        return self.loop_range().shape

    @property
    def snode(self):
        return self.loop_range().snode

    def __str__(self):
        """Python scope struct array print support."""
        if impl.inside_kernel():
            return f'<ti.Struct {", ".join([str(k) + "=" + str(v.dtype) for k, v in self.entries])}>'
        else:
            return str(self.to_numpy())

    def __repr__(self):
        if self.is_global():
            # make interactive shell happy, prevent materialization
            return f'<ti.Struct {", ".join([str(k) + "=" + str(v.dtype) for k, v in self.entries])}>'
        else:
            return str(self.to_numpy())

    @python_scope
    def from_numpy(self, array_dict):
        for k in self.members():
            self(k).from_numpy(array_dict[k])
    
    @python_scope
    def from_torch(self, array_dict):
        for k in self.members():
            self(k).from_torch(array_dict[k])

    @python_scope
    def to_numpy(self):
        return {k: v.to_numpy() for k, v in self.entries.items()}
    
    @python_scope
    def to_torch(self):
        return {k: v.to_torch() for k, v in self.entries.items()}

    @classmethod
    def empty(cls, members):
        """Clear the struct and fill None.

        Args:
            members (Dict[str, DataType]): the names and data types for struct members.
        Returns:
            :class:`~taichi.lang.struct.Struct`: A :class:`~taichi.lang.struct.Struct` instance filled with None.

        """
        return cls({k: None for k in members})

    @classmethod
    @python_scope
    def field(cls,
              members,
              shape=None,
              struct_name="<Struct>",
              offset=None,
              needs_grad=False,
              layout=None):

        self = cls.empty(members.keys())
        self.name = struct_name

        if layout is not None:
            assert shape is not None, 'layout is useless without shape'
        if shape is None:
            assert offset is None, "shape cannot be None when offset is being set"

        for member_name, dtype in members.items():
            self.register_member(member_name, dtype)
        self.grad = self.make_grad()

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
                for e in self.entries.values():
                    ti.root.dense(impl.index_nd(dim),
                                  shape).place(e, offset=offset)
                if needs_grad:
                    for e in self.entries.values():
                        ti.root.dense(impl.index_nd(dim),
                                      shape).place(e.grad, offset=offset)
            else:
                ti.root.dense(impl.index_nd(dim),
                              shape).place(*tuple(self.entries.values()), offset=offset)
                if needs_grad:
                    ti.root.dense(impl.index_nd(dim),
                                  shape).place(*tuple(self.entries.values()),
                                               offset=offset)
        return self