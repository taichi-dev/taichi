from taichi.lang._ndarray import ScalarNdarray
from taichi.lang.enums import Layout
from taichi.lang.exception import TaichiCompilationError
from taichi.lang.matrix import Matrix, MatrixNdarray, MatrixType, VectorNdarray
from taichi.types.annotations import template
from taichi.types.ndarray_type import NdarrayType

template_types = (NdarrayType, template)


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
                dtype = symbolic_args[i].dtype()
            else:
                element_shape = anno.element_shape
                element_dim = anno.element_dim
                dtype = anno.dtype

            if element_shape is None or anno.field_dim is None:
                raise TaichiCompilationError(
                    'Please either specify both `element_shape` and `field_dim` '
                    'in the param annotation, or provide an example '
                    f'ndarray for param={arg.name}')
            if element_dim is None or element_dim == 0:
                injected_args.append(
                    ScalarNdarray(dtype, (2, ) * anno.field_dim))
            elif element_dim == 1:
                injected_args.append(
                    VectorNdarray(element_shape[0],
                                  dtype=dtype,
                                  shape=(2, ) * anno.field_dim,
                                  layout=Layout.AOS))
            elif element_dim == 2:
                injected_args.append(
                    MatrixNdarray(element_shape[0],
                                  element_shape[1],
                                  dtype=dtype,
                                  shape=(2, ) * anno.field_dim,
                                  layout=Layout.AOS))
            else:
                raise RuntimeError('')
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
            # For primitive types, we can just inject a dummy value.
            injected_args.append(0)
    return injected_args
