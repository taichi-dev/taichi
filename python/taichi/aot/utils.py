from taichi.lang._ndarray import ScalarNdarray
from taichi.lang._texture import Texture
from taichi.lang.enums import Layout
from taichi.lang.exception import TaichiCompilationError
from taichi.lang.matrix import Matrix, MatrixNdarray, MatrixType, VectorNdarray
from taichi.lang.util import cook_dtype
from taichi.types.annotations import template
from taichi.types.ndarray_type import NdarrayType
from taichi.types.texture_type import RWTextureType, TextureType

template_types = (NdarrayType, template)


def check_type_match(lhs, rhs):
    if cook_dtype(lhs) == cook_dtype(rhs):
        return True
    return False


def produce_injected_args_from_template(kernel, template_args):
    injected_args = []
    num_template_args = len([
        arg.annotation for arg in kernel.arguments
        if isinstance(arg.annotation, template_types)
    ])
    assert num_template_args == len(
        template_args
    ), f'Need {num_template_args} inputs to instantiate the template parameters, got {len(template_args)}'
    for arg in kernel.arguments:
        anno = arg.annotation
        if isinstance(anno, template_types):
            injected_args.append(template_args[arg.name])
        else:
            injected_args.append(0)
    return injected_args


def produce_injected_args(kernel, symbolic_args=None):
    injected_args = []
    for i, arg in enumerate(kernel.arguments):
        anno = arg.annotation
        if isinstance(anno, template_types):
            if not isinstance(anno, NdarrayType):
                raise TaichiCompilationError(
                    f'Expected Ndaray type, got {anno}')
            if symbolic_args is not None:
                element_shape = tuple(symbolic_args[i].element_shape)
                element_dim = len(element_shape)
                field_dim = symbolic_args[i].field_dim
                dtype = symbolic_args[i].dtype()
            else:
                element_shape = anno.element_shape
                element_dim = anno.element_dim
                field_dim = anno.field_dim
                dtype = anno.dtype

            if element_shape is None or field_dim is None:
                raise TaichiCompilationError(
                    'Please either specify both `element_shape` and `field_dim` '
                    'in the param annotation, or provide an example '
                    f'ndarray for param={arg.name}')
            if anno.field_dim is not None and field_dim != anno.field_dim:
                raise TaichiCompilationError(
                    f'{field_dim} from Arg {arg.name} doesn\'t match kernel\'s annotated field_dim={anno.field_dim}'
                )

            if anno.dtype is not None and not check_type_match(
                    dtype, anno.dtype):
                raise TaichiCompilationError(
                    f' Arg {arg.name}\'s dtype {dtype.to_string()} doesn\'t match kernel\'s annotated dtype={anno.dtype.to_string()}'
                )

            if element_dim is None or element_dim == 0:
                injected_args.append(ScalarNdarray(dtype, (2, ) * field_dim))
            elif element_dim == 1:
                injected_args.append(
                    VectorNdarray(element_shape[0],
                                  dtype=dtype,
                                  shape=(2, ) * field_dim,
                                  layout=Layout.AOS))
            elif element_dim == 2:
                injected_args.append(
                    MatrixNdarray(element_shape[0],
                                  element_shape[1],
                                  dtype=dtype,
                                  shape=(2, ) * field_dim,
                                  layout=Layout.AOS))
            else:
                raise RuntimeError('')
        elif isinstance(anno, (TextureType, RWTextureType)):
            if symbolic_args is None:
                raise RuntimeError(
                    'Texture type annotation doesn\'t have enough information for aot. Please either specify the channel_format, shape and num_channels in the graph arg declaration.'
                )
            texture_shape = tuple(symbolic_args[i].texture_shape)
            channel_format = symbolic_args[i].channel_format()
            num_channels = symbolic_args[i].num_channels
            injected_args.append(
                Texture(channel_format, num_channels, texture_shape))
        elif isinstance(anno, MatrixType):
            if not isinstance(symbolic_args[i], list):
                raise RuntimeError('Expected a symbolic arg with Matrix type.')

            symbolic_mat_n = len(symbolic_args[i])
            symbolic_mat_m = len(symbolic_args[i][0])

            if symbolic_mat_m != anno.m or symbolic_mat_n != anno.n:
                raise RuntimeError(
                    f'Matrix dimension mismatch, expected ({anno.n}, {anno.m}) '
                    f'but dispatched shape ({symbolic_mat_n}, {symbolic_mat_m}).'
                )
            injected_args.append(Matrix([0] * anno.n * anno.m, dt=anno.dtype))
        else:
            if symbolic_args is not None:
                dtype = symbolic_args[i].dtype()
            else:
                dtype = anno

            if not check_type_match(dtype, anno):
                raise TaichiCompilationError(
                    f' Arg {arg.name}\'s dtype {dtype.to_string()} doesn\'t match kernel\'s annotated dtype={anno.to_string()}'
                )
            # For primitive types, we can just inject a dummy value.
            injected_args.append(0)
    return injected_args
