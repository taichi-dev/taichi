from taichi.lang import impl
from taichi.lang.util import taichi_scope


def sync():
    return impl.call_internal("block_barrier", with_runtime_context=False)


class SharedArray:
    _is_taichi_class = True

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.shared_array_proxy = impl.expr_init_shared_array(shape, dtype)

    @taichi_scope
    def _subscript(self, indices, get_ref=False):
        return impl.make_index_expr(self.shared_array_proxy, (indices, ))
