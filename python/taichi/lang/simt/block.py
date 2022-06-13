from taichi.lang import impl


def sync():
    return impl.call_internal("block_barrier", with_runtime_context=False)
