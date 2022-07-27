from taichi.lang import impl
from taichi.lang.util import taichi_scope
from taichi._lib import core as _ti_core


def sync():
    if impl.get_runtime().prog.config.arch == _ti_core.cuda:
        return impl.call_internal("block_barrier", with_runtime_context=False)
    elif impl.get_runtime().prog.config.arch == _ti_core.vulkan:
        print("CALLING VULKAN SYNC")
        return impl.call_internal("workgroupBarrier", with_runtime_context=False)

def mem_sync():
    if impl.get_runtime().prog.config.arch == _ti_core.cuda:
        return impl.call_internal("block_barrier", with_runtime_context=False)
    elif impl.get_runtime().prog.config.arch == _ti_core.vulkan:
        return impl.call_internal("workgroupMemoryBarrier", with_runtime_context=False)

def global_thread_idx():
    if impl.get_runtime().prog.config.arch == _ti_core.vulkan:
        return impl.call_internal("vkGlobalThreadIdx", with_runtime_context=False)


class SharedArray:
    _is_taichi_class = True

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.shared_array_proxy = impl.expr_init_shared_array(shape, dtype)

    @taichi_scope
    def _subscript(self, *indices, get_ref=False):
        return impl.make_index_expr(self.shared_array_proxy, (indices, ))
