from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang.expr import make_expr_group
from taichi.lang.util import taichi_scope


def arch_uses_spv(arch):
    return arch == _ti_core.vulkan or arch == _ti_core.metal or arch == _ti_core.opengl or arch == _ti_core.dx11


def sync():
    arch = impl.get_runtime().prog.config().arch
    if arch == _ti_core.cuda or arch == _ti_core.amdgpu:
        return impl.call_internal("block_barrier", with_runtime_context=False)
    if arch_uses_spv(arch):
        return impl.call_internal("workgroupBarrier",
                                  with_runtime_context=False)
    raise ValueError(f'ti.block.shared_array is not supported for arch {arch}')


def mem_sync():
    arch = impl.get_runtime().prog.config().arch
    if arch == _ti_core.cuda:
        return impl.call_internal("block_barrier", with_runtime_context=False)
    if arch_uses_spv(arch):
        return impl.call_internal("workgroupMemoryBarrier",
                                  with_runtime_context=False)
    raise ValueError(f'ti.block.mem_sync is not supported for arch {arch}')


def thread_idx():
    arch = impl.get_runtime().prog.config().arch
    if arch_uses_spv(arch):
        return impl.call_internal("localInvocationId",
                                  with_runtime_context=False)
    raise ValueError(f'ti.block.thread_idx is not supported for arch {arch}')


def global_thread_idx():
    arch = impl.get_runtime().prog.config().arch
    if arch == _ti_core.cuda or _ti_core.amdgpu:
        return impl.get_runtime().compiling_callable.ast_builder(
        ).insert_thread_idx_expr()
    if arch_uses_spv(arch):
        return impl.call_internal("globalInvocationId",
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
    def subscript(self, *indices):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()
        return impl.Expr(
            ast_builder.expr_subscript(
                self.shared_array_proxy, make_expr_group(*indices),
                impl.get_runtime().get_current_src_info()))
