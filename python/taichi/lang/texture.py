from taichi._lib import core as _ti_core
from taichi.lang import expr, impl
from taichi.lang.util import taichi_scope

import taichi as ti

class TextureSampler:
    def __init__(self, ptr_expr) -> None:
        self.ptr_expr = ptr_expr

    @taichi_scope
    def sample_lod(self, uv, lod):
        v = _ti_core.make_texture_op_expr(
        _ti_core.TextureOpType.sample_lod, self.ptr_expr,
        impl.make_expr_group(uv.x, uv.y, lod))
        r = impl.call_internal("composite_extract_0",
                            v,
                            with_runtime_context=False)
        g = impl.call_internal("composite_extract_1",
                            v,
                            with_runtime_context=False)
        b = impl.call_internal("composite_extract_2",
                            v,
                            with_runtime_context=False)
        a = impl.call_internal("composite_extract_3",
                            v,
                            with_runtime_context=False)
        return ti.Vector([r, g, b, a])

    @taichi_scope
    def fetch(self, index, lod):
        v = _ti_core.make_texture_op_expr(
            _ti_core.TextureOpType.fetch_texel, self.ptr_expr,
            impl.make_expr_group(index.x, index.y, lod))
        r = impl.call_internal("composite_extract_0",
                            v,
                            with_runtime_context=False)
        g = impl.call_internal("composite_extract_1",
                            v,
                            with_runtime_context=False)
        b = impl.call_internal("composite_extract_2",
                            v,
                            with_runtime_context=False)
        a = impl.call_internal("composite_extract_3",
                            v,
                            with_runtime_context=False)
        return ti.Vector([r, g, b, a])

class Texture:
    """Taichi Texture class.

    Args:
        dtype (DataType): Data type of each value.
        num_channels (int): Number of channels in texture
        shape (Tuple[int]): Shape of the Texture.
    """
    def __init__(self, dtype, num_channels, arr_shape):
        self.tex = impl.get_runtime().prog.create_texture(
            dtype, num_channels, arr_shape)

    def from_ndarray(self, ndarray):
        self.tex.from_ndarray(ndarray.arr)

    def device_allocation_ptr(self):
        return self.tex.device_allocation_ptr()
