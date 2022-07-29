import numbers

from taichi._lib import core as _ti_core
from taichi.lang import expr, impl, matrix
from taichi.lang.field import BitpackedFields, Field


class SNode:
    """A Python-side SNode wrapper.

    For more information on Taichi's SNode system, please check out
    these references:

    * https://docs.taichi-lang.org/docs/sparse
    * https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf

    Arg:
        ptr (pointer): The C++ side SNode pointer.
    """
    def __init__(self, ptr):
        self.ptr = ptr

    def dense(self, axes, dimensions):
        """Adds a dense SNode as a child component of `self`.

        Args:
            axes (List[Axis]): Axes to activate.
            dimensions (Union[List[int], int]): Shape of each axis.

        Returns:
            The added :class:`~taichi.lang.SNode` instance.
        """
        if isinstance(dimensions, int):
            dimensions = [dimensions] * len(axes)
        return SNode(
            self.ptr.dense(axes, dimensions,
                           impl.current_cfg().packed))

    def pointer(self, axes, dimensions):
        """Adds a pointer SNode as a child component of `self`.

        Args:
            axes (List[Axis]): Axes to activate.
            dimensions (Union[List[int], int]): Shape of each axis.

        Returns:
            The added :class:`~taichi.lang.SNode` instance.
        """
        if isinstance(dimensions, int):
            dimensions = [dimensions] * len(axes)
        return SNode(
            self.ptr.pointer(axes, dimensions,
                             impl.current_cfg().packed))

    @staticmethod
    def _hash(axes, dimensions):
        # original code is #def hash(self,axes, dimensions) without #@staticmethod   before fix pylint R0201
        """Not supported."""
        raise RuntimeError('hash not yet supported')
        # if isinstance(dimensions, int):
        #     dimensions = [dimensions] * len(axes)
        # return SNode(self.ptr.hash(axes, dimensions,
        #                            impl.current_cfg().packed))

    def dynamic(self, axis, dimension, chunk_size=None):
        """Adds a dynamic SNode as a child component of `self`.

        Args:
            axis (List[Axis]): Axis to activate, must be 1.
            dimension (int): Shape of the axis.
            chunk_size (int): Chunk size.

        Returns:
            The added :class:`~taichi.lang.SNode` instance.
        """
        assert len(axis) == 1
        if chunk_size is None:
            chunk_size = dimension
        return SNode(
            self.ptr.dynamic(axis[0], dimension, chunk_size,
                             impl.current_cfg().packed))

    def bitmasked(self, axes, dimensions):
        """Adds a bitmasked SNode as a child component of `self`.

        Args:
            axes (List[Axis]): Axes to activate.
            dimensions (Union[List[int], int]): Shape of each axis.

        Returns:
            The added :class:`~taichi.lang.SNode` instance.
        """
        if isinstance(dimensions, int):
            dimensions = [dimensions] * len(axes)
        return SNode(
            self.ptr.bitmasked(axes, dimensions,
                               impl.current_cfg().packed))

    def quant_array(self, axes, dimensions, max_num_bits):
        """Adds a quant_array SNode as a child component of `self`.

        Args:
            axes (List[Axis]): Axes to activate.
            dimensions (Union[List[int], int]): Shape of each axis.
            max_num_bits (int): Maximum number of bits it can hold.

        Returns:
            The added :class:`~taichi.lang.SNode` instance.
        """
        if isinstance(dimensions, int):
            dimensions = [dimensions] * len(axes)
        return SNode(
            self.ptr.quant_array(axes, dimensions, max_num_bits,
                                 impl.current_cfg().packed))

    def place(self, *args, offset=None):
        """Places a list of Taichi fields under the `self` container.

        Args:
            *args (List[ti.field]): A list of Taichi fields to place.
            offset (Union[Number, tuple[Number]]): Offset of the field domain.

        Returns:
            The `self` container.
        """
        if offset is None:
            offset = ()
        if isinstance(offset, numbers.Number):
            offset = (offset, )

        for arg in args:
            if isinstance(arg, BitpackedFields):
                bit_struct_type = arg.bit_struct_type_builder.build()
                bit_struct_snode = self.ptr.bit_struct(
                    bit_struct_type,
                    impl.current_cfg().packed)
                for (field, id_in_bit_struct) in arg.fields:
                    bit_struct_snode.place(field, offset, id_in_bit_struct)
            elif isinstance(arg, Field):
                for var in arg._get_field_members():
                    self.ptr.place(var.ptr, offset, -1)
            elif isinstance(arg, list):
                for x in arg:
                    self.place(x, offset=offset)
            else:
                raise ValueError(f'{arg} cannot be placed')
        return self

    def lazy_grad(self):
        """Automatically place the adjoint fields following the layout of their primal fields.

        Users don't need to specify ``needs_grad`` when they define scalar/vector/matrix fields (primal fields) using autodiff.
        When all the primal fields are defined, using ``taichi.root.lazy_grad()`` could automatically generate
        their corresponding adjoint fields (gradient field).

        To know more details about primal, adjoint fields and ``lazy_grad()``,
        please see Page 4 and Page 13-14 of DiffTaichi Paper: https://arxiv.org/pdf/1910.00935.pdf
        """
        self.ptr.lazy_grad(True, False)

    def lazy_dual(self):
        """Automatically place the dual fields following the layout of their primal fields.
        """
        self.ptr.lazy_grad(False, True)

    def parent(self, n=1):
        """Gets an ancestor of `self` in the SNode tree.

        Args:
            n (int): the number of levels going up from `self`.

        Returns:
            Union[None, _Root, SNode]: The n-th parent of `self`.
        """
        p = self.ptr
        while p and n > 0:
            p = p.parent
            n -= 1
        if p is None:
            return None

        if p.type == _ti_core.SNodeType.root:
            return impl.root

        return SNode(p)

    def _path_from_root(self):
        """Gets the path from root to `self` in the SNode tree.

        Returns:
            List[Union[_Root, SNode]]: The list of SNodes on the path from root to `self`.
        """
        p = self
        res = [p]
        while p != impl.root:
            p = p.parent()
            res.append(p)
        res.reverse()
        return res

    @property
    def _dtype(self):
        """Gets the data type of `self`.

        Returns:
            DataType: The data type of `self`.
        """
        return self.ptr.data_type()

    @property
    def _id(self):
        """Gets the id of `self`.

        Returns:
            int: The id of `self`.
        """
        return self.ptr.id

    @property
    def shape(self):
        """Gets the number of elements from root in each axis of `self`.

        Returns:
            Tuple[int]: The number of elements from root in each axis of `self`.
        """
        dim = self.ptr.num_active_indices()
        ret = tuple(self.ptr.get_shape_along_axis(i) for i in range(dim))

        return ret

    def _loop_range(self):
        """Gets the taichi_python.Expr wrapping the taichi_python.GlobalVariableExpression corresponding to `self` to serve as loop range.

        Returns:
            taichi_python.Expr: See above.
        """
        return impl.get_runtime().prog.global_var_expr_from_snode(self.ptr)

    @property
    def _name(self):
        """Gets the name of `self`.

        Returns:
            str: The name of `self`.
        """
        return self.ptr.name()

    @property
    def _snode(self):
        """Gets `self`.
        Returns:
            SNode: `self`.
        """
        return self

    def _get_children(self):
        """Gets all children components of `self`.

        Returns:
            List[SNode]: All children components of `self`.
        """
        children = []
        for i in range(self.ptr.get_num_ch()):
            children.append(SNode(self.ptr.get_ch(i)))
        return children

    @property
    def _num_dynamically_allocated(self):
        runtime = impl.get_runtime()
        runtime.materialize_root_fb(False)
        return runtime.prog.get_snode_num_dynamically_allocated(self.ptr)

    @property
    def _cell_size_bytes(self):
        impl.get_runtime().materialize_root_fb(False)
        return self.ptr.cell_size_bytes

    @property
    def _offset_bytes_in_parent_cell(self):
        impl.get_runtime().materialize_root_fb(False)
        return self.ptr.offset_bytes_in_parent_cell

    def deactivate_all(self):
        """Recursively deactivate all children components of `self`."""
        ch = self._get_children()
        for c in ch:
            c.deactivate_all()
        SNodeType = _ti_core.SNodeType
        if self.ptr.type == SNodeType.pointer or self.ptr.type == SNodeType.bitmasked:
            from taichi._kernels import \
                snode_deactivate  # pylint: disable=C0415
            snode_deactivate(self)
        if self.ptr.type == SNodeType.dynamic:
            # Note that dynamic nodes are different from other sparse nodes:
            # instead of deactivating each element, we only need to deactivate
            # its parent, whose linked list of chunks of elements will be deleted.
            from taichi._kernels import \
                snode_deactivate_dynamic  # pylint: disable=C0415
            snode_deactivate_dynamic(self)

    def __repr__(self):
        type_ = str(self.ptr.type)[len('SNodeType.'):]
        return f'<ti.SNode of type {type_}>'

    def __str__(self):
        # ti.root.dense(ti.i, 3).dense(ti.jk, (4, 5)).place(x)
        # ti.root => dense [3] => dense [3, 4, 5] => place [3, 4, 5]
        type_ = str(self.ptr.type)[len('SNodeType.'):]
        shape = str(list(self.shape))
        parent = str(self.parent())
        return f'{parent} => {type_} {shape}'

    def __eq__(self, other):
        return self.ptr == other.ptr

    def _physical_index_position(self):
        """Gets mappings from virtual axes to physical axes.

        Returns:
            Dict[int, int]: Mappings from virtual axes to physical axes.
        """
        ret = {}
        for virtual, physical in enumerate(
                self.ptr.get_physical_index_position()):
            if physical != -1:
                ret[virtual] = physical
        return ret


def rescale_index(a, b, I):
    """Rescales the index 'I' of field (or SNode) 'a' to match the shape of SNode 'b'.

    Args:

        a, b (Union[:class:`~taichi.Field`, :class:`~taichi.MatrixField`): Input taichi fields or snodes.
        I (Union[list, :class:`~taichi.Vector`]): grouped loop index.

    Returns:
        Ib (:class:`~taichi.Vector`): rescaled grouped loop index
    """
    assert isinstance(
        a, (Field, SNode)), "The first argument must be a field or an SNode"
    assert isinstance(
        b, (Field, SNode)), "The second argument must be a field or an SNode"
    if isinstance(I, list):
        I = matrix.Vector(I)
    else:
        assert isinstance(
            I, matrix.Matrix
        ), "The third argument must be an index (list or ti.Vector)"
    entries = [I(i) for i in range(I.n)]
    for n in range(min(I.n, min(len(a.shape), len(b.shape)))):
        if a.shape[n] > b.shape[n]:
            entries[n] = I(n) // (a.shape[n] // b.shape[n])
        if a.shape[n] < b.shape[n]:
            entries[n] = I(n) * (b.shape[n] // a.shape[n])
    return matrix.Vector(entries)


def append(node, indices, val):
    """Append a value `val` to a SNode `node` at index `indices`.

    Args:
        node (:class:`~taichi.SNode`): Input SNode.
        indices (Union[int, :class:`~taichi.Vector`]): the indices to visit.
        val (:mod:`~taichi.types`): the data to be appended.
    """
    a = impl.expr_init(
        _ti_core.insert_append(node._snode.ptr, expr.make_expr_group(indices),
                               expr.Expr(val).ptr))
    return a


def is_active(node, indices):
    """Explicitly query whether a cell in a SNode `node` at location
    `indices` is active or not.

    Args:
        node (:class:`~taichi.SNode`): Must be a pointer, hash or bitmasked node.
        indices (Union[int, list, :class:`~taichi.Vector`]): the indices to visit.

    Returns:
        bool: the cell `node[indices]` is active or not.
    """
    return expr.Expr(
        _ti_core.insert_is_active(node._snode.ptr,
                                  expr.make_expr_group(indices)))


def activate(node, indices):
    """Explicitly activate a cell of `node` at location `indices`.

    Args:
        node (:class:`~taichi.SNode`): Must be a pointer, hash or bitmasked node.
        indices (Union[int, :class:`~taichi.Vector`]): the indices to activate.
    """
    impl.get_runtime().prog.current_ast_builder().insert_activate(
        node._snode.ptr, expr.make_expr_group(indices))


def deactivate(node, indices):
    """Explicitly deactivate a cell of `node` at location `indices`.

    After deactivation, the Taichi runtime automatically recycles and zero-fills
    the memory of the deactivated cell.

    Args:
        node (:class:`~taichi.SNode`): Must be a pointer, hash or bitmasked node.
        indices (Union[int, :class:`~taichi.Vector`]): the indices to deactivate.
    """
    impl.get_runtime().prog.current_ast_builder().insert_deactivate(
        node._snode.ptr, expr.make_expr_group(indices))


def length(node, indices):
    """Return the length of the dynamic SNode `node` at index `indices`.

    Args:
        node (:class:`~taichi.SNode`): a dynamic SNode.
        indices (Union[int, :class:`~taichi.Vector`]): the indices to query.

    Returns:
        int: the length of cell `node[indices]`.
    """
    return expr.Expr(
        _ti_core.insert_len(node._snode.ptr, expr.make_expr_group(indices)))


def get_addr(f, indices):
    """Query the memory address (on CUDA/x64) of field `f` at index `indices`.

    Currently, this function can only be called inside a taichi kernel.

    Args:
        f (Union[:class:`~taichi.Field`, :class:`~taichi.MatrixField`]): Input taichi field for memory address query.
        indices (Union[int, :class:`~taichi.Vector`]): The specified field indices of the query.

    Returns:
        ti.u64: The memory address of `f[indices]`.
    """
    return expr.Expr(
        _ti_core.expr_get_addr(f._snode.ptr, expr.make_expr_group(indices)))


__all__ = [
    'activate', 'append', 'deactivate', 'get_addr', 'is_active', 'length',
    'rescale_index', "SNode"
]
