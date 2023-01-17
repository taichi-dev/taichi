from taichi.lang._ndarray import ScalarNdarray
from taichi.lang._texture import Texture
from taichi.lang.exception import TaichiCompilationError
from taichi.lang.matrix import Matrix, MatrixNdarray, MatrixType, VectorNdarray
from taichi.lang.util import cook_dtype
from taichi.types.annotations import template
from taichi.types.ndarray_type import NdarrayType
from taichi.types.texture_type import TY_CH2FORMAT, RWTextureType, TextureType

template_types = (NdarrayType, TextureType, template)


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
        elif isinstance(anno, RWTextureType):
            texture_shape = (2, ) * anno.num_dimensions
            fmt = anno.fmt
            injected_args.append(Texture(fmt, texture_shape))
        else:
            injected_args.append(0)
    return injected_args


def produce_injected_args(kernel, symbolic_args=None):
    injected_args = []
    for i, arg in enumerate(kernel.arguments):
        anno = arg.annotation
        if isinstance(anno, NdarrayType):
            # TODO(Haidong) we should always use MatrixType and get rid of the element shapes
            if symbolic_args is not None:
                arg_element_shape = tuple(symbolic_args[i].element_shape)
                arg_element_dim = len(arg_element_shape)
                arg_ndarray_dim = symbolic_args[i].field_dim
                arg_dtype = symbolic_args[i].dtype()
            elif isinstance(anno.dtype, MatrixType):
                n, m, elem_ndim, prim_dtype = anno.dtype._get_type_info()

                arg_element_shape = [n, m]
                arg_element_dim = elem_ndim
                arg_ndarray_dim = anno.ndim
                arg_dtype = prim_dtype
            else:
                arg_element_shape = []
                arg_element_dim = 0
                arg_ndarray_dim = anno.ndim
                arg_dtype = anno.dtype

            # Checking parameters' consistency between
            # the annotation "anno.xxx" and the actual argument "arg_xxx"
            if arg_element_shape is None or arg_ndarray_dim is None:
                raise TaichiCompilationError(
                    'Please either specify both `element_shape` and `ndim` '
                    'in the param annotation, or provide an example '
                    f'ndarray for param={arg.name}')
            if anno.ndim is not None and arg_ndarray_dim != anno.ndim:
                raise TaichiCompilationError(
                    f'{arg_ndarray_dim} from Arg {arg.name} doesn\'t match kernel\'s annotated ndim={anno.ndim}'
                )
            anno_dtype = anno.dtype
            if isinstance(anno_dtype, MatrixType):
                _, _, _, prim_dtype = anno_dtype._get_type_info()
                anno_dtype = prim_dtype
            if anno_dtype is not None:
                if not check_type_match(arg_dtype, anno_dtype):
                    raise TaichiCompilationError(
                        f' Arg {arg.name}\'s dtype {arg_dtype.to_string()} doesn\'t match kernel\'s annotated dtype={anno_dtype.to_string()}'
                    )

            if arg_element_dim is None or arg_element_dim == 0 or arg_element_shape == (
                    1, ):
                injected_args.append(
                    ScalarNdarray(dtype, (2, ) * arg_ndarray_dim))
            elif arg_element_dim == 1:
                injected_args.append(
                    VectorNdarray(arg_element_shape[0],
                                  dtype=dtype,
                                  shape=(2, ) * arg_ndarray_dim))
            elif arg_element_dim == 2:
                injected_args.append(
                    MatrixNdarray(arg_element_shape[0],
                                  arg_element_shape[1],
                                  dtype=dtype,
                                  shape=(2, ) * arg_ndarray_dim))
            else:
                raise RuntimeError('')
        elif isinstance(anno, RWTextureType):
            texture_shape = (2, ) * anno.num_dimensions
            fmt = anno.fmt
            injected_args.append(Texture(fmt, texture_shape))
        elif isinstance(anno, TextureType):
            if symbolic_args is None:
                raise RuntimeError(
                    'Texture type annotation doesn\'t have enough information for aot. Please either specify the channel_format, shape and num_channels in the graph arg declaration.'
                )
            texture_shape = tuple(symbolic_args[i].texture_shape)
            fmt = TY_CH2FORMAT[(symbolic_args[i].channel_format(),
                                symbolic_args[i].num_channels)]
            injected_args.append(Texture(fmt, texture_shape))
        elif isinstance(anno, MatrixType):
            if not isinstance(symbolic_args[i], list):
                raise RuntimeError('Expected a symbolic arg with Matrix type.')

            n, m, _, prim_dtype = anno._get_type_info()

            symbolic_mat_n = len(symbolic_args[i])
            symbolic_mat_m = len(symbolic_args[i][0])

            if symbolic_mat_m != m or symbolic_mat_n != n:
                raise RuntimeError(
                    f'Matrix dimension mismatch, expected ({n}, {m}) '
                    f'but dispatched shape ({symbolic_mat_n}, {symbolic_mat_m}).'
                )
            injected_args.append(Matrix([0] * n * m, dt=prim_dtype))
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
