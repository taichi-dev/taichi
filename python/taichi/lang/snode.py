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

    def place(self, *args, offset=None):
        from .expr import Expr
        from .util import is_taichi_class
        if offset is None:
            offset = []
        for arg in args:
            if isinstance(arg, Expr):
                self.ptr.place(Expr(arg).ptr, offset)
            elif isinstance(arg, list):
                for x in arg:
                    self.place(x, offset=offset)
            elif is_taichi_class(arg):
                self.place(arg.get_tensor_members(), offset=offset)
            else:
                raise ValueError(f'{arg} cannot be placed')
        return self

    def lazy_grad(self):
        self.ptr.lazy_grad()

    def parent(self):
        return SNode(self.ptr.snode().parent)

    def data_type(self):
        return self.ptr.data_type()

    def dim(self):
        return self.ptr.num_active_indices()

    def shape(self):
        return tuple(self.get_shape(i) for i in range(self.dim()))

    def get_shape(self, i):
        return self.ptr.get_num_elements_along_axis(i)

    def loop_range(self):
        import taichi as ti
        return ti.Expr(ti.core.global_var_expr_from_snode(self.ptr))

    def snode(self):
        return self

    def get_children(self):
        children = []
        for i in range(self.ptr.get_num_ch()):
            children.append(SNode(self.ptr.get_ch(i)))
        return children

    def deactivate_all(self):
        ch = self.get_children()
        for c in ch:
            c.deactivate_all()
        import taichi as ti
        if self.ptr.type == ti.core.SNodeType.pointer or self.ptr.type == ti.core.SNodeType.bitmasked:
            from .meta import snode_deactivate
            snode_deactivate(self)
