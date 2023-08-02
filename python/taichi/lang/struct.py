import numbers
from types import MethodType

import numpy as np
from taichi._lib import core as _ti_core
from taichi.lang import expr, impl, ops
from taichi.lang.enums import Layout
from taichi.lang.exception import (
    TaichiRuntimeTypeError,
    TaichiSyntaxError,
    TaichiTypeError,
)
from taichi.lang.expr import Expr
from taichi.lang.field import Field, ScalarField, SNodeHostAccess
from taichi.lang.matrix import Matrix, MatrixType
from taichi.lang.util import cook_dtype, in_python_scope, python_scope, taichi_scope
from taichi.types import primitive_types
from taichi.types.compound_types import CompoundType
from taichi.types.utils import is_signed


class Struct:
    """The Struct type class.

    A struct is a dictionary-like data structure that stores members as
    (key, value) pairs. Valid data members of a struct can be scalars,
    matrices or other dictionary-like structures.

    Args:
        entries (Dict[str, Union[Dict, Expr, Matrix, Struct]]): \
            keys and values for struct members. Entries can optionally
            include a dictionary of functions with the key '__struct_methods'
            which will be attached to the struct for executing on the struct data.

    Returns:
        An instance of this struct.

    Example::
_
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
    _instance_count = 0

    def __init__(self, *args, **kwargs):
        # converts lists to matrices and dicts to structs
        if len(args) == 1 and kwargs == {} and isinstance(args[0], dict):
            self.__entries = args[0]
        elif len(args) == 0:
            self.__entries = kwargs
        else:
            raise TaichiSyntaxError(
                "Custom structs need to be initialized using either dictionary or keyword arguments"
            )
        self.__methods = self.__entries.pop("__struct_methods", {})
        matrix_ndim = self.__entries.pop("__matrix_ndim", {})
        self._register_methods()

        for k, v in self.__entries.items():
            if isinstance(v, (list, tuple)):
                v = Matrix(v)
            if isinstance(v, dict):
                v = Struct(v)
            self.__entries[k] = v if in_python_scope() else impl.expr_init(v)
        self._register_members()
        self.__dtype = None

    @property
    def keys(self):
        """Returns the list of member names in string format.

        Example::

           >>> vec3 = ti.types.vector(3, ti.f32)
           >>> sphere = ti.Struct(center=vec3([0, 0, 0]), radius=1.0)
           >>> a.keys
           ['center', 'radius']
        """
        return list(self.__entries.keys())

    @property
    def _members(self):
        return list(self.__entries.values())

    @property
    def entries(self):
        return self.__entries

    @property
    def methods(self):
        return self.__methods

    @property
    def items(self):
        """Returns the items in this struct.

        Example::

            >>> vec3 = ti.types.vector(3, ti.f32)
            >>> sphere = ti.Struct(center=vec3([0, 0, 0]), radius=1.0)
            >>> sphere.items
            dict_items([('center', 2), ('radius', 1.0)])
        """
        return self.__entries.items()

    def _register_members(self):
        # https://stackoverflow.com/questions/48448074/adding-a-property-to-an-existing-object-instance
        cls = self.__class__
        new_cls_name = cls.__name__ + str(cls._instance_count)
        cls._instance_count += 1
        properties = {k: property(cls._make_getter(k), cls._make_setter(k)) for k in self.keys}
        self.__class__ = type(new_cls_name, (cls,), properties)

    def _register_methods(self):
        for name, method in self.__methods.items():
            # use MethodType to pass self (this object) to the method
            setattr(self, name, MethodType(method, self))

    def __getitem__(self, key):
        ret = self.__entries[key]
        if isinstance(ret, SNodeHostAccess):
            ret = ret.accessor.getter(*ret.key)
        return ret

    def __setitem__(self, key, value):
        if isinstance(self.__entries[key], SNodeHostAccess):
            self.__entries[key].accessor.setter(value, *self.__entries[key].key)
        else:
            if in_python_scope():
                if isinstance(self.__entries[key], Struct) or isinstance(self.__entries[key], Matrix):
                    self.__entries[key]._set_entries(value)
                else:
                    if isinstance(value, numbers.Number):
                        self.__entries[key] = value
                    else:
                        raise TypeError("A number is expected when assigning struct members")
            else:
                self.__entries[key] = value

    def _set_entries(self, value):
        if isinstance(value, dict):
            value = Struct(value)
        for k in self.keys:
            self[k] = value[k]
        self.__dtype = value.__dtype

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

    @taichi_scope
    def _assign(self, other):
        if not isinstance(other, (dict, Struct)):
            raise TaichiTypeError("Only dict or Struct can be assigned to a Struct")
        if isinstance(other, dict):
            other = Struct(other)
        if self.__entries.keys() != other.__entries.keys():
            raise TaichiTypeError(f"Member mismatch between structs {self.keys}, {other.keys}")
        for k, v in self.items:
            v._assign(other.__entries[k])
        self.__dtype = other.__dtype
        return self

    def __len__(self):
        """Get the number of entries in a custom struct"""
        return len(self.__entries)

    def __iter__(self):
        return self.__entries.values()

    def __str__(self):
        """Python scope struct array print support."""
        if impl.inside_kernel():
            item_str = ", ".join([str(k) + "=" + str(v) for k, v in self.items])
            item_str += f", struct_methods={self.__methods}"
            return f"<ti.Struct {item_str}>"
        return str(self.to_dict())

    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self, include_methods=False, include_ndim=False):
        """Converts the Struct to a dictionary.

        Args:
            include_methods (bool): Whether any struct methods should be included
                in the result dictionary under the key '__struct_methods'.

        Returns:
            Dict: The result dictionary.
        """
        res_dict = {
            k: v.to_dict(include_methods=include_methods, include_ndim=include_ndim)
            if isinstance(v, Struct)
            else v.to_list()
            if isinstance(v, Matrix)
            else v
            for k, v in self.__entries.items()
        }
        if include_methods:
            res_dict["__struct_methods"] = self.__methods
        if include_ndim:
            res_dict["__matrix_ndim"] = dict()
            for k, v in self.__entries.items():
                if isinstance(v, Matrix):
                    res_dict["__matrix_ndim"][k] = v.ndim
        return res_dict

    @classmethod
    @python_scope
    def field(
        cls,
        members,
        methods={},
        shape=None,
        name="<Struct>",
        offset=None,
        needs_grad=False,
        needs_dual=False,
        layout=Layout.AOS,
    ):
        """Creates a :class:`~taichi.StructField` with each element
        has this struct as its type.

        Args:
            members (dict): a dict, each item is like `name: type`.
            methods (dict): a dict of methods that should be included with
                the field.  Each struct item of the field will have the
                methods as instance functions.
            shape (Tuple[int]): width and height of the field.
            offset (Tuple[int]): offset of the indices of the created field.
                For example if `offset=(-10, -10)` the indices of the field
                will start at `(-10, -10)`, not `(0, 0)`.
            needs_grad (bool): enabling grad field (reverse mode autodiff) or not.
            needs_dual (bool): enabling dual field (forward mode autodiff) or not.
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
            raise TaichiSyntaxError("shape cannot be None when offset is being set")

        field_dict = {}

        for key, dtype in members.items():
            field_name = name + "." + key
            if isinstance(dtype, CompoundType):
                if isinstance(dtype, StructType):
                    field_dict[key] = dtype.field(
                        shape=None,
                        name=field_name,
                        offset=offset,
                        needs_grad=needs_grad,
                        needs_dual=needs_dual,
                    )
                else:
                    field_dict[key] = dtype.field(
                        shape=None,
                        name=field_name,
                        offset=offset,
                        needs_grad=needs_grad,
                        needs_dual=needs_dual,
                        ndim=getattr(dtype, "ndim", 2),
                    )
            else:
                field_dict[key] = impl.field(
                    dtype,
                    shape=None,
                    name=field_name,
                    offset=offset,
                    needs_grad=needs_grad,
                    needs_dual=needs_dual,
                )

        if shape is not None:
            if isinstance(shape, numbers.Number):
                shape = (shape,)
            if isinstance(offset, numbers.Number):
                offset = (offset,)

            if offset is not None and len(shape) != len(offset):
                raise TaichiSyntaxError(
                    f"The dimensionality of shape and offset must be the same ({len(shape)} != {len(offset)})"
                )
            dim = len(shape)
            if layout == Layout.SOA:
                for e in field_dict.values():
                    impl.root.dense(impl.index_nd(dim), shape).place(e, offset=offset)
                if needs_grad:
                    for e in field_dict.values():
                        impl.root.dense(impl.index_nd(dim), shape).place(e.grad, offset=offset)
                if needs_dual:
                    for e in field_dict.values():
                        impl.root.dense(impl.index_nd(dim), shape).place(e.dual, offset=offset)
            else:
                impl.root.dense(impl.index_nd(dim), shape).place(*tuple(field_dict.values()), offset=offset)
                if needs_grad:
                    grads = tuple(e.grad for e in field_dict.values())
                    impl.root.dense(impl.index_nd(dim), shape).place(*grads, offset=offset)

                if needs_dual:
                    duals = tuple(e.dual for e in field_dict.values())
                    impl.root.dense(impl.index_nd(dim), shape).place(*duals, offset=offset)

        return StructField(field_dict, methods, name=name)


class _IntermediateStruct(Struct):
    """Intermediate struct class for compiler internal use only.

    Args:
        entries (Dict[str, Union[Expr, Matrix, Struct]]): keys and values for struct members.
            Any methods included under the key '__struct_methods' will be applied to each
            struct instance.
    """

    def __init__(self, entries):
        assert isinstance(entries, dict)
        self._Struct__methods = entries.pop("__struct_methods", {})
        self._register_methods()
        self._Struct__entries = entries
        self._register_members()


class StructField(Field):
    """Taichi struct field with SNode implementation.

       Instead of directly constraining Expr entries, the StructField object
       directly hosts members as `Field` instances to support nested structs.

    Args:
        field_dict (Dict[str, Field]): Struct field members.
        struct_methods (Dict[str, callable]): Dictionary of functions to apply
            to each struct instance in the field.
        name (string, optional): The custom name of the field.
    """

    def __init__(self, field_dict, struct_methods, name=None, is_primal=True):
        # will not call Field initializer
        self.field_dict = field_dict
        self.struct_methods = struct_methods
        self.name = name
        self.grad = None
        self.dual = None
        if is_primal:
            grad_field_dict = {}
            for k, v in self.field_dict.items():
                grad_field_dict[k] = v.grad
            self.grad = StructField(grad_field_dict, struct_methods, name + ".grad", is_primal=False)

            dual_field_dict = {}
            for k, v in self.field_dict.items():
                dual_field_dict[k] = v.dual
            self.dual = StructField(dual_field_dict, struct_methods, name + ".dual", is_primal=False)
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
            setattr(self, k, self.field_dict[k])

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
        """Gets SNode of representative field member for loop range info.

        Returns:
            taichi_python.SNode: SNode of representative (first) field member.
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
            self.field_dict[k].copy_from(other.get_member_field(k))

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
        """Creates a ScalarField using a specific field member.

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
    def from_paddle(self, array_dict):
        """Copies the data from a set of `paddle.Tensor` into this field.

        The argument `array_dict` must be a dictionay-like object, it
        contains all the keys in this field and the copying process
        between corresponding items can be performed.
        """
        for k, v in self._items:
            v.from_paddle(array_dict[k])

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
    def to_paddle(self, place=None):
        """Converts the Struct field instance to a dictionary of Paddle tensors.

        The dictionary may be nested when converting nested structs.

        Args:
            place (paddle.CPUPlace()/CUDAPlace(n), optional): The
                desired place of returned tensor.

        Returns:
            Dict[str, Union[paddle.Tensor, Dict]]: The result
                Paddle tensor.
        """
        return {k: v.to_paddle(place=place) for k, v in self._items}

    @python_scope
    def __setitem__(self, indices, element):
        self._initialize_host_accessors()
        self[indices]._set_entries(element)

    @python_scope
    def __getitem__(self, indices):
        self._initialize_host_accessors()
        # scalar fields does not instantiate SNodeHostAccess by default
        entries = {
            k: v._host_access(self._pad_key(indices))[0] if isinstance(v, ScalarField) else v[indices]
            for k, v in self._items
        }
        entries["__struct_methods"] = self.struct_methods
        return Struct(entries)


class StructType(CompoundType):
    def __init__(self, **kwargs):
        self.members = {}
        self.methods = {}
        elements = []
        for k, dtype in kwargs.items():
            if k == "__struct_methods":
                self.methods = dtype
            elif isinstance(dtype, StructType):
                self.members[k] = dtype
                elements.append([dtype.dtype, k])
            elif isinstance(dtype, MatrixType):
                self.members[k] = dtype
                elements.append([dtype.tensor_type, k])
            else:
                dtype = cook_dtype(dtype)
                self.members[k] = dtype
                elements.append([dtype, k])
        self.dtype = _ti_core.get_type_factory_instance().get_struct_type(elements)

    def __call__(self, *args, **kwargs):
        """Create an instance of this struct type."""
        d = {}
        items = self.members.items()
        # iterate over the members of this struct
        for index, pair in enumerate(items):
            name, dtype = pair  # (member name, member type)
            if index < len(args):  # set from args
                data = args[index]
            else:  # set from kwargs
                data = kwargs.get(name, 0)

            # If dtype is CompoundType and data is a scalar, it cannot be
            # casted in the self.cast call later. We need an initialization here.
            if isinstance(dtype, CompoundType) and not isinstance(data, (dict, Struct)):
                data = dtype(data)

            d[name] = data

        entries = Struct(d)
        entries._Struct__dtype = self.dtype
        struct = self.cast(entries)
        struct._Struct__dtype = self.dtype
        return struct

    def __instancecheck__(self, instance):
        if not isinstance(instance, Struct):
            return False
        if list(self.members.keys()) != list(instance._Struct__entries.keys()):
            return False
        if (
            hasattr(instance, "_Struct__dtype")
            and instance._Struct__dtype is not None
            and instance._Struct__dtype != self.dtype
        ):
            return False
        for index, (name, dtype) in enumerate(self.members.items()):
            val = instance._members[index]
            if isinstance(dtype, StructType):
                if not isinstance(val, dtype):
                    return False
            elif isinstance(dtype, MatrixType):
                if isinstance(val, Expr):
                    if not val.is_tensor():
                        return False
                if val.get_shape() != dtype.get_shape():
                    return False
            elif dtype in primitive_types.integer_types:
                if isinstance(val, Expr):
                    if val.is_tensor() or val.is_struct() or val.element_type() not in primitive_types.integer_types:
                        return False
                elif not isinstance(val, (int, np.integer)):
                    return False
            elif dtype in primitive_types.real_types:
                if isinstance(val, Expr):
                    if val.is_tensor() or val.is_struct() or val.element_type() not in primitive_types.real_types:
                        return False
                elif not isinstance(val, (float, np.floating)):
                    return False
        return True

    def from_taichi_object(self, func_ret, ret_index=()):
        d = {}
        items = self.members.items()
        for index, pair in enumerate(items):
            name, dtype = pair
            if isinstance(dtype, CompoundType):
                d[name] = dtype.from_taichi_object(func_ret, ret_index + (index,))
            else:
                d[name] = expr.Expr(
                    _ti_core.make_get_element_expr(
                        func_ret.ptr,
                        ret_index + (index,),
                        _ti_core.DebugInfo(impl.get_runtime().get_current_src_info()),
                    )
                )
        d["__struct_methods"] = self.methods

        struct = Struct(d)
        struct._Struct__dtype = self.dtype
        return struct

    def from_kernel_struct_ret(self, launch_ctx, ret_index=()):
        d = {}
        items = self.members.items()
        for index, pair in enumerate(items):
            name, dtype = pair
            if isinstance(dtype, CompoundType):
                d[name] = dtype.from_kernel_struct_ret(launch_ctx, ret_index + (index,))
            else:
                if dtype in primitive_types.integer_types:
                    if is_signed(cook_dtype(dtype)):
                        d[name] = launch_ctx.get_struct_ret_int(ret_index + (index,))
                    else:
                        d[name] = launch_ctx.get_struct_ret_uint(ret_index + (index,))
                elif dtype in primitive_types.real_types:
                    d[name] = launch_ctx.get_struct_ret_float(ret_index + (index,))
                else:
                    raise TaichiRuntimeTypeError(f"Invalid return type on index={ret_index + (index, )}")
        d["__struct_methods"] = self.methods

        struct = Struct(d)
        struct._Struct__dtype = self.dtype
        return struct

    def set_kernel_struct_args(self, struct, launch_ctx, ret_index=()):
        # TODO: move this to class Struct after we add dtype to Struct
        items = self.members.items()
        for index, pair in enumerate(items):
            name, dtype = pair
            if isinstance(dtype, CompoundType):
                dtype.set_kernel_struct_args(struct[name], launch_ctx, ret_index + (index,))
            else:
                if dtype in primitive_types.integer_types:
                    if is_signed(cook_dtype(dtype)):
                        launch_ctx.set_struct_arg_int(ret_index + (index,), struct[name])
                    else:
                        launch_ctx.set_struct_arg_uint(ret_index + (index,), struct[name])
                elif dtype in primitive_types.real_types:
                    launch_ctx.set_struct_arg_float(ret_index + (index,), struct[name])
                else:
                    raise TaichiRuntimeTypeError(f"Invalid argument type on index={ret_index + (index, )}")

    def set_argpack_struct_args(self, struct, argpack, ret_index=()):
        # TODO: move this to class Struct after we add dtype to Struct
        items = self.members.items()
        for index, pair in enumerate(items):
            name, dtype = pair
            if isinstance(dtype, CompoundType):
                dtype.set_kernel_struct_args(struct[name], argpack, ret_index + (index,))
            else:
                if dtype in primitive_types.integer_types:
                    if is_signed(cook_dtype(dtype)):
                        argpack.set_arg_int(ret_index + (index,), struct[name])
                    else:
                        argpack.set_arg_uint(ret_index + (index,), struct[name])
                elif dtype in primitive_types.real_types:
                    argpack.set_arg_float(ret_index + (index,), struct[name])
                else:
                    raise TaichiRuntimeTypeError(f"Invalid argument type on index={ret_index + (index, )}")

    def cast(self, struct):
        # sanity check members
        if self.members.keys() != struct._Struct__entries.keys():
            raise TaichiSyntaxError("Incompatible arguments for custom struct members!")
        entries = {}
        for k, dtype in self.members.items():
            if isinstance(dtype, MatrixType):
                entries[k] = dtype(struct._Struct__entries[k])
            elif isinstance(dtype, CompoundType):
                entries[k] = dtype.cast(struct._Struct__entries[k])
            else:
                if in_python_scope():
                    v = struct._Struct__entries[k]
                    entries[k] = int(v) if dtype in primitive_types.integer_types else float(v)
                else:
                    entries[k] = ops.cast(struct._Struct__entries[k], dtype)
        entries["__struct_methods"] = self.methods
        struct = Struct(entries)
        struct._Struct__dtype = self.dtype
        return struct

    def filled_with_scalar(self, value):
        entries = {}
        for k, dtype in self.members.items():
            if isinstance(dtype, MatrixType):
                entries[k] = dtype(value)
            elif isinstance(dtype, CompoundType):
                entries[k] = dtype.filled_with_scalar(value)
            else:
                entries[k] = value
        entries["__struct_methods"] = self.methods
        struct = Struct(entries)
        struct._Struct__dtype = self.dtype
        return struct

    def field(self, **kwargs):
        return Struct.field(self.members, self.methods, **kwargs)

    def __str__(self):
        """Python scope struct type print support."""
        item_str = ", ".join([str(k) + "=" + str(v) for k, v in self.members.items()])
        item_str += f", struct_methods={self.methods}"
        return f"<ti.StructType {item_str}>"


def dataclass(cls):
    """Converts a class with field annotations and methods into a taichi struct type.

    This will return a normal custom struct type, with the functions added to it.
    Struct fields can be generated in the normal way from the struct type.
    Functions in the class can be run on the struct instance.

    This class decorator inspects the class for annotations and methods and
        1.  Sets the annotations as fields for the struct
        2.  Attaches the methods to the struct type

    Example::

        >>> @ti.dataclass
        >>> class Sphere:
        >>>     center: vec3
        >>>     radius: ti.f32
        >>>
        >>>     @ti.func
        >>>     def area(self):
        >>>         return 4 * 3.14 * self.radius * self.radius
        >>>
        >>> my_spheres = Sphere.field(shape=(n, ))
        >>> my_sphere[2].area()

    Args:
        cls (Class): the class with annotations and methods to convert to a struct

    Returns:
        A taichi struct with the annotations as fields
            and methods from the class attached.
    """
    # save the annotation fields for the struct
    fields = getattr(cls, "__annotations__", {})
    # raise error if there are default values
    for k in fields.keys():
        if hasattr(cls, k):
            raise TaichiSyntaxError("Default value in @dataclass is not supported.")
    # get the class methods to be attached to the struct types
    fields["__struct_methods"] = {
        attribute: getattr(cls, attribute)
        for attribute in dir(cls)
        if callable(getattr(cls, attribute)) and not attribute.startswith("__")
    }
    return StructType(**fields)


__all__ = ["Struct", "StructField", "dataclass"]
