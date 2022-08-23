from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang.util import taichi_scope


def sync():
    arch = impl.get_runtime().prog.config.arch
    if arch == _ti_core.cuda:
        return impl.call_internal("block_barrier", with_runtime_context=False)
    if arch == _ti_core.vulkan:
        return impl.call_internal("workgroupBarrier",
                                  with_runtime_context=False)
    raise ValueError(f'ti.block.shared_array is not supported for arch {arch}')


def mem_sync():
    arch = impl.get_runtime().prog.config.arch
    if arch == _ti_core.cuda:
        return impl.call_internal("block_barrier", with_runtime_context=False)
    if arch == _ti_core.vulkan:
        return impl.call_internal("workgroupMemoryBarrier",
                                  with_runtime_context=False)
    raise ValueError(f'ti.block.mem_sync is not supported for arch {arch}')


def thread_idx():
    arch = impl.get_runtime().prog.config.arch
    if arch is _ti_core.vulkan:
        return impl.call_internal("localInvocationId",
                                  with_runtime_context=False)
    raise ValueError(f'ti.block.thread_idx is not supported for arch {arch}')


def global_thread_idx():
    arch = impl.get_runtime().prog.config.arch
    if arch == _ti_core.cuda:
        return impl.get_runtime().prog.current_ast_builder(
        ).insert_thread_idx_expr()
    if impl.get_runtime().prog.config.arch == _ti_core.vulkan:
        return impl.call_internal("vkGlobalThreadIdx",
                                  with_runtime_context=False)
    raise ValueError(
        f'ti.block.global_thread_idx is not supported for arch {arch}')


class SharedArray:
    _is_taichi_class = True

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.shared_array_proxy = impl.expr_init_shared_array(shape, dtype)

    @taichi_scope
    def _subscript(self, *indices, get_ref=False):
        return impl.make_index_expr(self.shared_array_proxy, (indices, ))
