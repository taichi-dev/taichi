import sys

import pytest

import taichi as ti
from tests import test_utils


def _get_matrix_swizzle_apis():
    swizzle_gen = ti.lang.swizzle_generator.SwizzleGenerator()
    KEMAP_SET = ['xyzw', 'rgba', 'stpq']
    res = []
    for key_group in KEMAP_SET:
        sw_patterns = swizzle_gen.generate(key_group, required_length=4)
        sw_patterns = map(lambda p: ''.join(p), sw_patterns)
        res += sw_patterns
    return sorted(res)


def _get_expected_matrix_apis():
    base = [
        'all',
        'any',
        'any_array_access',
        'cast',
        'cols',
        'cross',
        'determinant',
        'diag',
        'dot',
        'dynamic_index_stride',
        'entries',
        'field',
        'fill',
        'identity',
        'inverse',
        'local_tensor_proxy',
        'max',
        'min',
        'ndarray',
        'norm',
        'norm_inv',
        'norm_sqr',
        'normalized',
        'one',
        'outer_product',
        'rotation2d',
        'rows',
        'sum',
        'to_list',
        'to_numpy',
        'trace',
        'transpose',
        'unit',
        'zero',
    ]
    res = base + _get_matrix_swizzle_apis()
    return sorted(res)


user_api = {}
user_api[ti] = [
    'CRITICAL', 'DEBUG', 'ERROR', 'Field', 'FieldsBuilder', 'GUI', 'INFO',
    'Layout', 'Matrix', 'MatrixField', 'MatrixNdarray', 'Mesh', 'Ndarray',
    'SNode', 'ScalarField', 'ScalarNdarray', 'Struct', 'StructField', 'TRACE',
    'TaichiAssertionError', 'TaichiCompilationError', 'TaichiNameError',
    'TaichiRuntimeError', 'TaichiRuntimeTypeError', 'TaichiSyntaxError',
    'TaichiTypeError', 'Tape', 'TetMesh', 'TriMesh', 'Vector', 'VectorNdarray',
    'WARN', 'abs', 'acos', 'activate', 'ad', 'aot', 'append', 'arm64', 'asin',
    'assume_in_range', 'atan2', 'atomic_add', 'atomic_and', 'atomic_max',
    'atomic_min', 'atomic_or', 'atomic_sub', 'atomic_xor', 'axes', 'bit_cast',
    'bit_shr', 'block_local', 'cache_read_only', 'cast', 'cc', 'ceil',
    'clear_all_gradients', 'cos', 'cpu', 'cuda', 'data_oriented', 'deactivate',
    'deactivate_all_snodes', 'dx11', 'eig', 'exp', 'experimental', 'extension',
    'f16', 'f32', 'f64', 'field', 'float16', 'float32', 'float64', 'floor',
    'func', 'get_addr', 'global_thread_idx', 'gpu', 'grouped', 'hex_to_rgb',
    'i', 'i16', 'i32', 'i64', 'i8', 'ij', 'ijk', 'ijkl', 'ijl', 'ik', 'ikl',
    'il', 'init', 'int16', 'int32', 'int64', 'int8', 'is_active',
    'is_logging_effective', 'j', 'jk', 'jkl', 'jl', 'k', 'kernel', 'kl', 'l',
    'lang', 'length', 'linalg', 'log', 'loop_config', 'math', 'max',
    'mesh_local', 'mesh_patch_idx', 'metal', 'min', 'ndarray', 'ndrange',
    'no_activate', 'one', 'opengl', 'polar_decompose', 'pow', 'profiler',
    'randn', 'random', 'raw_div', 'raw_mod', 'rescale_index', 'reset',
    'rgb_to_hex', 'root', 'round', 'rsqrt', 'select', 'set_logging_level',
    'simt', 'sin', 'solve', 'sparse_matrix_builder', 'sqrt', 'static',
    'static_assert', 'static_print', 'stop_grad', 'svd', 'swizzle_generator',
    'sym_eig', 'sync', 'tan', 'tanh', 'template', 'tools', 'types', 'u16',
    'u32', 'u64', 'u8', 'ui', 'uint16', 'uint32', 'uint64', 'uint8', 'vulkan',
    'wasm', 'x64', 'x86_64', 'zero'
]
user_api[ti.Field] = [
    'copy_from', 'dtype', 'fill', 'from_numpy', 'from_torch', 'parent',
    'shape', 'snode', 'to_numpy', 'to_torch'
]
user_api[ti.FieldsBuilder] = [
    'bit_array', 'bit_struct', 'bitmasked', 'deactivate_all', 'dense',
    'dynamic', 'finalize', 'lazy_grad', 'place', 'pointer'
]
user_api[ti.math] = [
    'cconj', 'cdiv', 'cexp', 'cinv', 'clamp', 'clog', 'cmul', 'cpow', 'cross',
    'csqrt', 'degrees', 'distance', 'dot', 'e', 'eye', 'fract', 'ivec2',
    'ivec3', 'ivec4', 'log2', 'mat2', 'mat3', 'mat4', 'mix', 'mod',
    'normalize', 'pi', 'radians', 'reflect', 'refract', 'rot2', 'rot3',
    'rotate2d', 'rotate3d', 'sign', 'smoothstep', 'step', 'uvec2', 'uvec3',
    'uvec4', 'vec2', 'vec3', 'vec4'
]
user_api[ti.Matrix] = _get_expected_matrix_apis()
user_api[ti.MatrixField] = [
    'copy_from', 'dtype', 'fill', 'from_numpy', 'from_torch',
    'get_scalar_field', 'parent', 'shape', 'snode', 'to_numpy', 'to_torch'
]
user_api[ti.MatrixNdarray] = [
    'copy_from', 'element_shape', 'fill', 'from_numpy', 'to_numpy'
]
user_api[ti.Ndarray] = ['copy_from', 'element_shape', 'fill']
user_api[ti.SNode] = [
    'bit_array', 'bit_struct', 'bitmasked', 'deactivate_all', 'dense',
    'dynamic', 'lazy_grad', 'parent', 'place', 'pointer', 'shape'
]
user_api[ti.ScalarField] = [
    'copy_from', 'dtype', 'fill', 'from_numpy', 'from_torch', 'parent',
    'shape', 'snode', 'to_numpy', 'to_torch'
]
user_api[ti.ScalarNdarray] = [
    'copy_from', 'element_shape', 'fill', 'from_numpy', 'to_numpy'
]
user_api[ti.Struct] = ['field', 'fill', 'items', 'keys', 'to_dict']
user_api[ti.StructField] = [
    'copy_from', 'dtype', 'fill', 'from_numpy', 'from_torch',
    'get_member_field', 'keys', 'parent', 'shape', 'snode', 'to_numpy',
    'to_torch'
]
user_api[ti.VectorNdarray] = [
    'copy_from', 'element_shape', 'fill', 'from_numpy', 'to_numpy'
]


@pytest.mark.parametrize('src', user_api.keys())
@test_utils.test(arch=ti.cpu)
def test_api(src):
    # When Python version is below 3.7, deprecated names are
    # handled as normal names, which will fail this test.
    expected = user_api[src]
    actual = [s for s in dir(src) if not s.startswith('_')]
    assert sys.version_info < (
        3, 7
    ) or actual == expected, f'Failed for API={src}:\n  expected={expected}\n  actual={actual}'
