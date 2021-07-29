# The reason we import just the taichi.core.util module, instead of the ti_core
# object within it, is that ti_core is stateful. While in practice ti_core is
# loaded during the import procedure, it's probably still good to delay the
# access to it.

from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl
from taichi.lang.exception import InvalidOperationError


class SNodeTree:
    def __init__(self, ptr):
        self.ptr = ptr
        self.destroyed = False

    def destroy(self):
        if self.destroyed:
            raise InvalidOperationError('SNode tree has been destroyed')
        self.ptr.destroy_snode_tree(impl.get_runtime().prog)
        self.destroyed = True

    @property
    def id(self):
        if self.destroyed:
            raise InvalidOperationError('SNode tree has been destroyed')
        return self.ptr.id()
