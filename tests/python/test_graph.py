import platform

import numpy as np
import pytest
from taichi.lang.exception import TaichiCompilationError

import taichi as ti
from tests import test_utils

supported_floating_types = [ti.f32] if platform.system() == 'Darwin' else [
    ti.f32, ti.f64
]


@test_utils.test(arch=ti.vulkan)
def test_ndarray_int():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(dtype=ti.i32, field_dim=1)):
        for i in range(n):
            pos[i] = 1

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                           'pos',
                           ti.i32,
                           field_dim=1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.i32, shape=(n, ))
    g.run({'pos': a})
    assert (a.to_numpy() == np.ones(4)).all()


@test_utils.test(arch=ti.vulkan)
def test_ndarray_0dim():
    @ti.kernel
    def test(pos: ti.types.ndarray(dtype=ti.i32, field_dim=0)):
        pos[None] = 1

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                           'pos',
                           ti.i32,
                           field_dim=0)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.i32, shape=())
    g.run({'pos': a})
    assert a.to_numpy() == 1


@test_utils.test(arch=ti.vulkan)
def test_ndarray_float():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(field_dim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                           'pos',
                           ti.f32,
                           field_dim=1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.f32, shape=(n, ))
    g.run({'pos': a})
    assert (a.to_numpy() == (np.ones(4) * 2.5)).all()


@test_utils.test(arch=ti.vulkan)
def test_arg_mismatched_field_dim():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(field_dim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                           'pos',
                           ti.f32,
                           field_dim=2)
    g_init = ti.graph.GraphBuilder()
    with pytest.raises(TaichiCompilationError,
                       match="doesn't match kernel's annotated field_dim"):
        g_init.dispatch(test, sym_pos)


@test_utils.test(arch=ti.vulkan)
def test_arg_mismatched_field_dim_ndarray():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(field_dim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'pos', ti.f32, 1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.f32, shape=(n, n))
    with pytest.raises(RuntimeError, match="Dispatch node is compiled for"):
        g.run({'pos': a})


@test_utils.test(arch=ti.vulkan)
def test_repeated_arg_name():
    n = 4

    @ti.kernel
    def test1(pos: ti.types.ndarray(field_dim=1)):
        for i in range(n):
            pos[i] = 2.5

    @ti.kernel
    def test2(v: ti.f32):
        for i in range(n):
            print(v)

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                           'pos',
                           ti.f32,
                           field_dim=1)
    sym_pos1 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'pos', ti.f32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(test1, sym_pos)

    with pytest.raises(RuntimeError):
        builder.dispatch(test2, sym_pos1)


@test_utils.test(arch=ti.vulkan)
def test_arg_mismatched_scalar_dtype():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(field_dim=1), val: ti.f32):
        for i in range(n):
            pos[i] = val

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'pos', ti.f32, 1)
    sym_val = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'val', ti.i32)
    g_init = ti.graph.GraphBuilder()
    with pytest.raises(TaichiCompilationError,
                       match="doesn't match kernel's annotated dtype"):
        g_init.dispatch(test, sym_pos, sym_val)


@test_utils.test(arch=ti.vulkan)
def test_arg_mismatched_ndarray_dtype():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(dtype=ti.f32, field_dim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'pos', ti.i32, 1)
    g_init = ti.graph.GraphBuilder()
    with pytest.raises(TaichiCompilationError,
                       match="doesn't match kernel's annotated dtype"):
        g_init.dispatch(test, sym_pos)


@test_utils.test(arch=ti.vulkan)
def test_ndarray_dtype_mismatch_runtime():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(field_dim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                           'pos',
                           ti.f32,
                           field_dim=1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.i32, shape=(n, ))
    with pytest.raises(RuntimeError, match="but got an ndarray with dtype="):
        g.run({'pos': a})


def build_graph_vector(N, dtype):
    @ti.kernel
    def vector_sum(mat: ti.types.vector(N, dtype),
                   res: ti.types.ndarray(dtype=dtype, field_dim=1)):
        res[0] = mat.sum() + mat[2]

    sym_A = ti.graph.Arg(ti.graph.ArgKind.MATRIX, 'mat',
                         ti.types.vector(N, dtype))
    sym_res = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'res', dtype, field_dim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(vector_sum, sym_A, sym_res)
    graph = builder.compile()
    return graph


def build_graph_matrix(N, dtype):
    @ti.kernel
    def matrix_sum(mat: ti.types.matrix(N, 2, dtype),
                   res: ti.types.ndarray(dtype=dtype, field_dim=1)):
        res[0] = mat.sum()

    sym_A = ti.graph.Arg(ti.graph.ArgKind.MATRIX, 'mat',
                         ti.types.matrix(N, 2, dtype))
    sym_res = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'res', dtype, field_dim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(matrix_sum, sym_A, sym_res)
    graph = builder.compile()
    return graph


@test_utils.test(arch=ti.vulkan)
def test_matrix_int():
    n = 4
    A = ti.Matrix([4, 5] * n)
    res = ti.ndarray(ti.i32, shape=(1, ))
    graph = build_graph_matrix(n, dtype=ti.i32)
    graph.run({"mat": A, "res": res})
    assert (res.to_numpy()[0] == 36)


@test_utils.test(arch=ti.vulkan)
def test_matrix_float():
    n = 4
    A = ti.Matrix([4.2, 5.7] * n)
    res = ti.ndarray(ti.f32, shape=(1))
    graph = build_graph_matrix(n, dtype=ti.f32)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == test_utils.approx(39.6, rel=1e-5)


@test_utils.test(arch=ti.vulkan)
def test_vector_int():
    n = 12
    A = ti.Vector([1, 3, 13, 4, 5, 6, 7, 2, 3, 4, 1, 25])
    res = ti.ndarray(ti.i32, shape=(1, ))
    graph = build_graph_vector(n, dtype=ti.i32)
    graph.run({"mat": A, "res": res})
    assert (res.to_numpy()[0] == 87)


@test_utils.test(arch=ti.vulkan)
def test_vector_float():
    n = 8
    A = ti.Vector([1.4, 3.7, 13.2, 4.5, 5.6, 6.1, 7.2, 2.6])
    res = ti.ndarray(ti.f32, shape=(1, ))
    graph = build_graph_vector(n, dtype=ti.f32)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == test_utils.approx(57.5, rel=1e-5)


@pytest.mark.parametrize('dt', supported_floating_types)
@test_utils.test(arch=ti.vulkan)
def test_arg_float(dt):
    @ti.kernel
    def foo(a: dt, b: ti.types.ndarray(dtype=dt, field_dim=1)):
        b[0] = a

    k = ti.ndarray(dt, shape=(1, ))

    sym_A = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'mat', dt)
    sym_B = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                         'b',
                         dt,
                         field_dim=1,
                         element_shape=())
    builder = ti.graph.GraphBuilder()
    builder.dispatch(foo, sym_A, sym_B)
    graph = builder.compile()
    graph.run({"mat": 3.12, 'b': k})
    assert k.to_numpy()[0] == test_utils.approx(3.12, rel=1e-5)


@pytest.mark.parametrize('dt',
                         [ti.i32, ti.i64, ti.i16, ti.u16, ti.u32, ti.u64])
@test_utils.test(arch=ti.vulkan)
def test_arg_int(dt):
    @ti.kernel
    def foo(a: dt, b: ti.types.ndarray(dtype=dt, field_dim=1)):
        b[0] = a

    k = ti.ndarray(dt, shape=(1, ))

    sym_A = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'mat', dt)
    sym_B = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                         'b',
                         dt,
                         field_dim=1,
                         element_shape=())
    builder = ti.graph.GraphBuilder()
    builder.dispatch(foo, sym_A, sym_B)
    graph = builder.compile()
    graph.run({"mat": 1234, 'b': k})
    assert k.to_numpy()[0] == 1234


@test_utils.test(arch=ti.vulkan)
def test_texture():
    res = (256, 256)

    @ti.kernel
    def make_texture(tex: ti.types.rw_texture(num_dimensions=2,
                                              num_channels=1,
                                              channel_format=ti.f32,
                                              lod=0)):
        for i, j in ti.ndrange(128, 128):
            tex.store(ti.Vector([i, j]), ti.Vector([0.1, 0.0, 0.0, 0.0]))

    @ti.kernel
    def paint(t: ti.f32, pixels: ti.types.ndarray(field_dim=2),
              tex: ti.types.texture(num_dimensions=2)):
        for i, j in pixels:
            uv = ti.Vector([i / res[0], j / res[1]])
            warp_uv = uv + ti.Vector(
                [ti.cos(t + uv.x * 5.0),
                 ti.sin(t + uv.y * 5.0)]) * 0.1
            c = ti.math.vec4(0.0)
            if uv.x > 0.5:
                c = tex.sample_lod(warp_uv, 0.0)
            else:
                c = tex.fetch(ti.cast(warp_uv * 128, ti.i32), 0)
            pixels[i, j] = [c.r, c.r, c.r, 1.0]

    _t = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 't', ti.f32)
    _pixels_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                               'pixels_arr',
                               ti.f32,
                               field_dim=2,
                               element_shape=(4, ))

    _rw_tex = ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE,
                           'rw_tex',
                           channel_format=ti.f32,
                           shape=(128, 128),
                           num_channels=1)
    _tex = ti.graph.Arg(ti.graph.ArgKind.TEXTURE,
                        'tex',
                        channel_format=ti.f32,
                        shape=(128, 128),
                        num_channels=1)

    g_builder = ti.graph.GraphBuilder()
    g_builder.dispatch(make_texture, _rw_tex)
    g_builder.dispatch(paint, _t, _pixels_arr, _tex)
    g = g_builder.compile()

    pixels_arr = ti.Vector.ndarray(4, dtype=float, shape=res)
    texture = ti.Texture(ti.f32, 1, (128, 128))
    t = 1

    g.run({
        'rw_tex': texture,
        't': t,
        'pixels_arr': pixels_arr,
        'tex': texture
    })
    pixels = pixels_arr.to_numpy()
    for i in range(res[0]):
        for j in range(res[1]):
            assert test_utils.allclose(pixels[i, j], [0.1, 0.1, 0.1, 1.])
