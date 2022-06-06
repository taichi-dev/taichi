from taichi.lang import impl, expr
import taichi as ti

def sample_texture(tex, uv):
    t = impl.call_internal("global_texture_ptr", expr.make_constant_expr(tex.device_allocation_ptr(), ti.u64),
                              with_runtime_context=False)
    v = impl.call_internal("sample_texture", t, uv.x, uv.y,
                              with_runtime_context=False)
    r = impl.call_internal("composite_extract_0",
                              v, with_runtime_context=False)
    g = impl.call_internal("composite_extract_1",
                              v, with_runtime_context=False)
    b = impl.call_internal("composite_extract_2",
                              v, with_runtime_context=False)
    a = impl.call_internal("composite_extract_3",
                              v, with_runtime_context=False)
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