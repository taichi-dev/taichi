from taichi.lang import impl

def sample_texture(uv):
    return impl.call_internal("sample_texture",
                              uv.x, uv.y,
                              with_runtime_context=False)
