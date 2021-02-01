from . import impl
from .util import deprecated
import numbers


class SNode:
    def __init__(self, ptr):
        self.ptr = ptr

    def dense(self, indices, dimensions):
        if isinstance(dimensions, int):
            dimensions = [dimensions] * len(indices)
        return SNode(self.ptr.dense(indices, dimensions))

    def pointer(self, indices, dimensions):
        if isinstance(dimensions, int):
            dimensions = [dimensions] * len(indices)
        return SNode(self.ptr.pointer(indices, dimensions))

    def hash(self, indices, dimensions):
        if isinstance(dimensions, int):
            dimensions = [dimensions] * len(indices)
        return SNode(self.ptr.hash(indices, dimensions))

    def dynamic(self, index, dimension, chunk_size=None):
        assert len(index) == 1
        if chunk_size is None:
            chunk_size = dimension
        return SNode(self.ptr.dynamic(index[0], dimension, chunk_size))

    def bitmasked(self, indices, dimensions):
        if isinstance(dimensions, int):
            dimensions = [dimensions] * len(indices)
        return SNode(self.ptr.bitmasked(indices, dimensions))

    @deprecated('_bit_struct', 'bit_struct')
    def _bit_struct(self, num_bits):
        return self.bit_struct(num_bits)

    def bit_struct(self, num_bits):
        return SNode(self.ptr.bit_struct(num_bits))

    @deprecated('_bit_array', 'bit_array')
    def _bit_array(self, indices, dimensions, num_bits):
        return self.bit_array(indices, dimensions, num_bits)

    def bit_array(self, indices, dimensions, num_bits):
        if isinstance(dimensions, int):
            dimensions = [dimensions] * len(indices)
        return SNode(self.ptr.bit_array(indices, dimensions, num_bits))

    def place(self, *args, offset=None, shared_exponent=False):
        from .expr import Expr
        from .util import is_taichi_class
        if offset is None:
            offset = []
        if isinstance(offset, numbers.Number):
            offset = (offset, )
        if shared_exponent:
            self.ptr.begin_shared_exp_placement()
        for arg in args:
            if isinstance(arg, Expr):
                self.ptr.place(Expr(arg).ptr, offset)
            elif isinstance(arg, list):
                for x in arg:
                    self.place(x, offset=offset)
            elif is_taichi_class(arg):
                self.place(arg.get_field_members(), offset=offset)
            else:
                raise ValueError(f'{arg} cannot be placed')
        if shared_exponent:
            self.ptr.end_shared_exp_placement()
        return self

    def lazy_grad(self):
        self.ptr.lazy_grad()

    def parent(self, n=1):
        impl.get_runtime().materialize()
        p = self.ptr
        while p and n > 0:
            p = p.parent
            n -= 1
        if p is None:
            return None
        if p.type == impl.taichi_lang_core.SNodeType.root:
            return impl.root
        return SNode(p)

    @property
    def dtype(self):
        return self.ptr.data_type()

    @deprecated('x.data_type()', 'x.dtype')
    def data_type(self):
        return self.dtype

    @deprecated('x.dim()', 'len(x.shape)')
    def dim(self):
        return len(self.shape)

    @property
    def id(self):
        return self.ptr.id

    @property
    def shape(self):
        impl.get_runtime().materialize()
        dim = self.ptr.num_active_indices()
        ret = [self.ptr.get_shape_along_axis(i) for i in range(dim)]

        class callable_tuple(tuple):
            @deprecated('x.shape()', 'x.shape')
            def __call__(self):
                return self

        ret = callable_tuple(ret)
        return ret

    @deprecated('x.get_shape(i)', 'x.shape[i]')
    def get_shape(self, i):
        return self.shape[i]

    def loop_range(self):
        import taichi as ti
        return ti.Expr(ti.core.global_var_expr_from_snode(self.ptr))

    @deprecated('x.snode()', 'x.snode')
    def __call__(self):  # TODO: remove this after v0.7.0
        return self

    @property
    def snode(self):
        return self

    def get_children(self):
        children = []
        for i in range(self.ptr.get_num_ch()):
            children.append(SNode(self.ptr.get_ch(i)))
        return children

    @property
    def num_dynamically_allocated(self):
        runtime = impl.get_runtime()
        runtime.materialize()
        return runtime.prog.get_snode_num_dynamically_allocated(self.ptr)

    @property
    def cell_size_bytes(self):
        runtime = impl.get_runtime()
        runtime.materialize()
        return self.ptr.cell_size_bytes

    def deactivate_all(self):
        ch = self.get_children()
        for c in ch:
            c.deactivate_all()
        import taichi as ti
        if self.ptr.type == ti.core.SNodeType.pointer or self.ptr.type == ti.core.SNodeType.bitmasked:
            from .meta import snode_deactivate
            snode_deactivate(self)
        if self.ptr.type == ti.core.SNodeType.dynamic:
            from .meta import snode_deactivate_dynamic
            # Note that dynamic nodes are different from other sparse nodes:
            # instead of deactivating each element, we only need to deactivate
            # its parent, whose linked list of chunks of elements will be deleted.
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

    def physical_index_position(self):
        ret = {}
        for virtual, physical in enumerate(
                self.ptr.get_physical_index_position()):
            if physical != -1:
                ret[virtual] = physical
        return ret
