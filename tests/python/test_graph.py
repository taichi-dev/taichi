import platform

import numpy as np
import pytest
from taichi.lang.exception import TaichiCompilationError

import taichi as ti
from tests import test_utils

supported_floating_types = [ti.f32] if platform.system() == "Darwin" else [ti.f32, ti.f64]

supported_archs_cgraph = [ti.vulkan, ti.opengl]


@test_utils.test(arch=supported_archs_cgraph)
def test_ndarray_int():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(n):
            pos[i] = 1

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.i32, ndim=1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.i32, shape=(n,))
    g.run({"pos": a})
    assert (a.to_numpy() == np.ones(4)).all()


@test_utils.test(arch=supported_archs_cgraph)
def test_ndarray_1dim_scalar():
    @ti.kernel
    def ti_test_debug(arr: ti.types.ndarray(ndim=1)):
        arr[0] = 0

    debug_arr = ti.ndarray(ti.i32, shape=5)
    sym_debug_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "debug_arr", ti.types.vector(1, ti.f32), ndim=1)

    g_builder = ti.graph.GraphBuilder()
    g_builder.dispatch(ti_test_debug, sym_debug_arr)


@test_utils.test(arch=supported_archs_cgraph)
def test_ndarray_0dim():
    @ti.kernel
    def test(pos: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        pos[None] = 1

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.i32, ndim=0)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.i32, shape=())
    g.run({"pos": a})
    assert a.to_numpy() == 1


@test_utils.test(arch=supported_archs_cgraph)
def test_ndarray_float():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(ndim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.f32, ndim=1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.f32, shape=(n,))
    g.run({"pos": a})
    assert (a.to_numpy() == (np.ones(4) * 2.5)).all()


@test_utils.test(arch=supported_archs_cgraph)
def test_arg_mismatched_ndim():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(ndim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.f32, ndim=2)
    g_init = ti.graph.GraphBuilder()
    with pytest.raises(TaichiCompilationError, match="doesn't match kernel's annotated ndim"):
        g_init.dispatch(test, sym_pos)


@test_utils.test(arch=supported_archs_cgraph)
def test_arg_mismatched_ndim_ndarray():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(ndim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.f32, 1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.f32, shape=(n, n))
    with pytest.raises(RuntimeError, match="Dispatch node is compiled for"):
        g.run({"pos": a})


@test_utils.test(arch=supported_archs_cgraph)
def test_repeated_arg_name():
    n = 4

    @ti.kernel
    def test1(pos: ti.types.ndarray(ndim=1)):
        for i in range(n):
            pos[i] = 2.5

    @ti.kernel
    def test2(v: ti.f32):
        for i in range(n):
            print(v)

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.f32, ndim=1)
    sym_pos1 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "pos", ti.f32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(test1, sym_pos)

    with pytest.raises(RuntimeError):
        builder.dispatch(test2, sym_pos1)


@test_utils.test(arch=supported_archs_cgraph)
def test_arg_mismatched_scalar_dtype():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(ndim=1), val: ti.f32):
        for i in range(n):
            pos[i] = val

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.f32, 1)
    sym_val = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "val", ti.i32)
    g_init = ti.graph.GraphBuilder()
    with pytest.raises(TaichiCompilationError, match="doesn't match kernel's annotated dtype"):
        g_init.dispatch(test, sym_pos, sym_val)


@test_utils.test(arch=supported_archs_cgraph)
def test_arg_mismatched_ndarray_dtype():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.i32, 1)
    g_init = ti.graph.GraphBuilder()
    with pytest.raises(TaichiCompilationError, match="doesn't match kernel's annotated dtype"):
        g_init.dispatch(test, sym_pos)


@test_utils.test(arch=supported_archs_cgraph)
def test_ndarray_dtype_mismatch_runtime():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(ndim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.f32, ndim=1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.i32, shape=(n,))
    with pytest.raises(RuntimeError, match="but got an ndarray with dtype="):
        g.run({"pos": a})


def build_graph_vector(N, dtype):
    @ti.kernel
    def vector_sum(mat: ti.types.vector(N, dtype), res: ti.types.ndarray(dtype=dtype, ndim=1)):
        res[0] = mat.sum() + mat[2]

    sym_A = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "mat", ti.types.vector(N, dtype))
    sym_res = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "res", dtype, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(vector_sum, sym_A, sym_res)
    graph = builder.compile()
    return graph


def build_graph_matrix(N, dtype):
    @ti.kernel
    def matrix_sum(mat: ti.types.matrix(N, 2, dtype), res: ti.types.ndarray(dtype=dtype, ndim=1)):
        res[0] = mat.sum()

    sym_A = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "mat", ti.types.matrix(N, 2, dtype))
    sym_res = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "res", dtype, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(matrix_sum, sym_A, sym_res)
    graph = builder.compile()
    return graph


@pytest.mark.sm70
@pytest.mark.parametrize("dt", [ti.u8, ti.u16, ti.u32, ti.u64, ti.i8, ti.i16, ti.i32, ti.i64])
@test_utils.test(arch=supported_archs_cgraph)
def test_matrix_int(dt):
    if ti.lang.impl.current_cfg().arch == ti.opengl and dt not in [ti.u32, ti.i32]:
        return
    n = 4
    A = ti.Matrix([4, 5] * n, dt)
    res = ti.ndarray(dt, shape=(1,))
    graph = build_graph_matrix(n, dtype=dt)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == 36


@pytest.mark.parametrize("dt", supported_floating_types)
@test_utils.test(arch=supported_archs_cgraph)
def test_matrix_float(dt):
    if ti.lang.impl.current_cfg().arch == ti.opengl and dt not in [ti.f32]:
        return
    n = 4
    A = ti.Matrix([4.2, 5.7] * n, dt)
    res = ti.ndarray(dt, shape=(1,))
    graph = build_graph_matrix(n, dtype=dt)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == test_utils.approx(39.6, rel=1e-5)


@pytest.mark.sm70
@test_utils.test(arch=[ti.vulkan])
def test_matrix_float16():
    n = 4
    A = ti.Matrix([4.0, 5.0] * n, ti.f16)
    res = ti.ndarray(ti.f16, shape=(1,))
    graph = build_graph_matrix(n, dtype=ti.f16)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == test_utils.approx(36.0, rel=1e-5)


@pytest.mark.sm70
@pytest.mark.parametrize("dt", [ti.u8, ti.u16, ti.u32, ti.u64, ti.i8, ti.i16, ti.i32, ti.i64])
@test_utils.test(arch=supported_archs_cgraph)
def test_vector_int(dt):
    if ti.lang.impl.current_cfg().arch == ti.opengl and dt not in [ti.u32, ti.i32]:
        return
    n = 12
    A = ti.Vector([1, 3, 13, 4, 5, 6, 7, 2, 3, 4, 1, 25], dt)
    res = ti.ndarray(dt, shape=(1,))
    graph = build_graph_vector(n, dtype=dt)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == 87


@pytest.mark.parametrize("dt", supported_floating_types)
@test_utils.test(arch=supported_archs_cgraph)
def test_vector_float(dt):
    if ti.lang.impl.current_cfg().arch == ti.opengl and dt not in [ti.f32]:
        return
    n = 8
    A = ti.Vector([1.4, 3.7, 13.2, 4.5, 5.6, 6.1, 7.2, 2.6], dt)
    res = ti.ndarray(dt, shape=(1,))
    graph = build_graph_vector(n, dtype=dt)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == test_utils.approx(57.5, rel=1e-5)


@pytest.mark.sm70
@test_utils.test(arch=[ti.vulkan])
def test_vector_float16():
    n = 4
    A = ti.Vector([1.4, 3.7, 13.2, 4.5], ti.f16)
    res = ti.ndarray(ti.f16, shape=(1,))
    graph = build_graph_vector(n, dtype=ti.f16)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == test_utils.approx(36.0, rel=1e-2)


@pytest.mark.parametrize("dt", supported_floating_types)
@test_utils.test(arch=supported_archs_cgraph)
def test_arg_float(dt):
    @ti.kernel
    def foo(a: dt, b: ti.types.ndarray(dtype=dt, ndim=1)):
        b[0] = a

    k = ti.ndarray(dt, shape=(1,))

    sym_A = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "mat", dt)
    sym_B = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "b", dt, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(foo, sym_A, sym_B)
    graph = builder.compile()
    graph.run({"mat": 3.12, "b": k})
    assert k.to_numpy()[0] == test_utils.approx(3.12, rel=1e-5)


@pytest.mark.parametrize("dt", [ti.i32, ti.i64, ti.u32, ti.u64])
@test_utils.test(arch=supported_archs_cgraph, exclude=[(ti.vulkan, "Darwin")])
def test_arg_int(dt):
    @ti.kernel
    def foo(a: dt, b: ti.types.ndarray(dtype=dt, ndim=1)):
        b[0] = a

    k = ti.ndarray(dt, shape=(1,))

    sym_A = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "mat", dt)
    sym_B = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "b", dt, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(foo, sym_A, sym_B)
    graph = builder.compile()
    graph.run({"mat": 1234, "b": k})
    assert k.to_numpy()[0] == 1234


@pytest.mark.parametrize("dt", [ti.i16, ti.u16, ti.u8, ti.i8])
@test_utils.test(arch=ti.vulkan)
def test_arg_short(dt):
    @ti.kernel
    def foo(a: dt, b: ti.types.ndarray(dtype=dt, ndim=1)):
        b[0] = a

    k = ti.ndarray(dt, shape=(1,))

    sym_A = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "mat", dt)
    sym_B = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "b", dt, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(foo, sym_A, sym_B)
    graph = builder.compile()
    graph.run({"mat": 123, "b": k})
    assert k.to_numpy()[0] == 123


@test_utils.test(arch=ti.vulkan)
def test_texture():
    res = (256, 256)

    @ti.kernel
    def make_texture(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in ti.ndrange(128, 128):
            tex.store(ti.Vector([i, j]), ti.Vector([0.1, 0.0, 0.0, 0.0]))

    @ti.kernel
    def paint(
        t: ti.f32,
        pixels: ti.types.ndarray(ndim=2),
        tex: ti.types.texture(num_dimensions=2),
    ):
        for i, j in pixels:
            uv = ti.Vector([i / res[0], j / res[1]])
            warp_uv = uv + ti.Vector([ti.cos(t + uv.x * 5.0), ti.sin(t + uv.y * 5.0)]) * 0.1
            c = ti.math.vec4(0.0)
            if uv.x > 0.5:
                c = tex.sample_lod(warp_uv, 0.0)
            else:
                c = tex.fetch(ti.cast(warp_uv * 128, ti.i32), 0)
            pixels[i, j] = [c.r, c.r, c.r, 1.0]

    _t = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "t", ti.f32)
    _pixels_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pixels_arr", ti.math.vec4, ndim=2)

    _rw_tex = ti.graph.Arg(
        ti.graph.ArgKind.RWTEXTURE,
        "rw_tex",
        ndim=2,
        fmt=ti.Format.r32f,
    )
    _tex = ti.graph.Arg(
        ti.graph.ArgKind.TEXTURE,
        "tex",
        ndim=2,
    )

    g_builder = ti.graph.GraphBuilder()
    g_builder.dispatch(make_texture, _rw_tex)
    g_builder.dispatch(paint, _t, _pixels_arr, _tex)
    g = g_builder.compile()

    pixels_arr = ti.Vector.ndarray(4, dtype=float, shape=res)
    texture = ti.Texture(ti.Format.r32f, (128, 128))
    t = 1

    g.run({"rw_tex": texture, "t": t, "pixels_arr": pixels_arr, "tex": texture})
    pixels = pixels_arr.to_numpy()
    for i in range(res[0]):
        for j in range(res[1]):
            assert test_utils.allclose(pixels[i, j], [0.1, 0.1, 0.1, 1.0])


@test_utils.test(arch=supported_archs_cgraph)
def test_ti_func_with_template_args():
    MyStruct = ti.types.struct(
        id=ti.i32,
        val=ti.f32,
        center=ti.types.vector(3, ti.f32),
        color=ti.types.vector(4, ti.i32),
    )

    arr = ti.ndarray(ti.i32, shape=())

    @ti.func
    def test_func(x: ti.template()):
        x.id = 0
        x.val = 1.0
        x.center = ti.Vector([0.0, 0.0, 0.0])
        x.color = ti.Vector([1, 1, 0, 0])

    @ti.kernel
    def test_kernel(arr: ti.types.ndarray()):
        x = MyStruct()
        test_func(x)
        arr[None] = x.color[1]

    sym_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "arr", ti.i32, ndim=0)
    g_builder = ti.graph.GraphBuilder()
    g_builder.dispatch(test_kernel, sym_arr)
    g = g_builder.compile()
    g.run({"arr": arr})
    assert arr.to_numpy() == 1


@test_utils.test(arch=[ti.vulkan])
def test_texture_struct_for():
    res = (128, 128)
    tex = ti.Texture(ti.Format.r32f, res)
    arr = ti.ndarray(ti.f32, res)

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    @ti.kernel
    def read(tex: ti.types.texture(num_dimensions=2), arr: ti.types.ndarray()):
        for i, j in arr:
            arr[i, j] = tex.fetch(ti.Vector([i, j]), 0).x

    sym_tex = ti.graph.Arg(
        ti.graph.ArgKind.RWTEXTURE,
        "tex",
        fmt=ti.Format.r32f,
        ndim=2,
    )
    sym_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "arr", ti.f32, ndim=2)

    gb = ti.graph.GraphBuilder()
    gb.dispatch(write, sym_tex)
    gb.dispatch(read, sym_tex, sym_arr)
    graph = gb.compile()

    graph.run({"tex": tex, "arr": arr})
    assert arr.to_numpy().sum() == 128 * 128
