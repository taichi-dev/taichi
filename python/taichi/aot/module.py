from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from taichi.lang import impl, kernel_impl
from taichi.lang._ndarray import ScalarNdarray
from taichi.lang.enums import Layout
from taichi.lang.field import ScalarField
from taichi.lang.matrix import MatrixField, MatrixNdarray, VectorNdarray
from taichi.types.annotations import ArgAnyArray, template
from taichi.types.primitive_types import f32


class KernelTemplate:
    def __init__(self, kernel_fn, aot_module):
        self._kernel_fn = kernel_fn
        self._aot_module = aot_module

    @staticmethod
    def keygen(v, key_p, fields):
        if isinstance(v, (int, float, bool)):
            key_p += '=' + str(v) + ','
            return key_p
        for ky, val in fields:
            if val is v:
                key_p += '=' + ky + ','
                return key_p
        raise RuntimeError('Arg type must be of type int/float/boolean' +
                           'or taichi field. Type ' + str(type(v)) +
                           ' is not supported')

    def instantiate(self, **kwargs):
        name = self._kernel_fn.__name__
        kernel = self._kernel_fn._primal
        assert isinstance(kernel, kernel_impl.Kernel)
        injected_args = []
        key_p = ''
        anno_index = 0
        template_args = {}

        for index, (key, value) in enumerate(kwargs.items()):
            template_args[index] = (key, value)

        for anno in kernel.argument_annotations:
            if isinstance(anno, template):
                (k, v) = template_args[anno_index]
                key_p += k
                key_p = self.keygen(v, key_p, self._aot_module._fields.items())
                injected_args.append(v)
                anno_index += 1
            else:
                injected_args.append(0)
        kernel.ensure_compiled(*injected_args)
        self._aot_module._aot_builder.add_kernel_template(
            name, key_p, kernel.kernel_cpp)

        # kernel AOT
        self._aot_module._kernels.append(kernel)


class Module:
    """An AOT module to save and load Taichi kernels.

    This module serializes the Taichi kernels for a specific arch. The
    serialized module can later be loaded to run on that backend, without the
    Python environment.

    Example:
      Usage::

        m = ti.aot.Module(ti.metal)
        m.add_kernel(foo)
        m.add_kernel(bar)

        m.save('/path/to/module')

        # Now the module file '/path/to/module' contains the Metal kernels
        # for running ``foo`` and ``bar``.
    """
    def __init__(self, arch):
        """Creates a new AOT module instance

        Args:
          arch: Target backend architecture. This is ignored for now. The AOT
            backend still uses the one specified in :func:`~taichi.lang.init`.
        """
        self._arch = arch
        self._kernels = []
        self._fields = {}
        rtm = impl.get_runtime()
        rtm._finalize_root_fb_for_aot()
        self._aot_builder = rtm.prog.make_aot_module_builder(arch)

    def add_field(self, name, field):
        """Add a taichi field to the AOT module.

        Args:
          name: name of taichi field
          field: taichi field

        Example::

            >>> a = ti.field(ti.f32, shape=(4,4))
            >>> b = ti.field("something")
            >>>
            >>> m.add_field(a)
            >>> m.add_field(b)
            >>>
            >>> # Must add in sequence
        """
        is_scalar = True
        self._fields[name] = field
        column_num = 1
        row_num = 1
        if isinstance(field, MatrixField):
            is_scalar = False
            row_num = field.m
            column_num = field.n
        else:
            assert isinstance(field, ScalarField)
        self._aot_builder.add_field(name, field.snode.ptr, is_scalar,
                                    field.dtype, field.snode.shape, row_num,
                                    column_num)

    def add_kernel(self, kernel_fn, example_any_arrays=None, name=None):
        """Add a taichi kernel to the AOT module.

        Args:
          kernel_fn (Function): the function decorated by taichi `kernel`.
          example_any_arrays (Dict[int, ti.ndarray]): a dict where key is arg_id and key is example any_arr input.
          name (str): Name to identify this kernel in the module. If not
            provided, uses the built-in ``__name__`` attribute of `kernel_fn`.

        """
        name = name or kernel_fn.__name__
        kernel = kernel_fn._primal
        assert isinstance(kernel, kernel_impl.Kernel)
        injected_args = []
        num_arr = len([
            anno for anno in kernel.argument_annotations
            if isinstance(anno, ArgAnyArray)
        ])
        assert example_any_arrays is None or num_arr == len(
            example_any_arrays
        ), f'Need {num_arr} example any_arr inputs but got {len(example_any_arrays)}'
        i = 0
        for anno in kernel.argument_annotations:
            if isinstance(anno, ArgAnyArray):
                if example_any_arrays:
                    injected_args.append(example_any_arrays[i])
                else:
                    assert anno.element_shape is not None and anno.field_dim is not None, 'Please either specify element_shape & field_dim in the kernel arg annotation or provide a dict of example ndarrays.'
                    if anno.element_dim == 0:
                        injected_args.append(
                            ScalarNdarray(dtype=f32,
                                          shape=(2, ) * anno.field_dim))
                    elif anno.element_dim == 1:
                        injected_args.append(
                            VectorNdarray(anno.element_shape[0],
                                          dtype=f32,
                                          shape=(2, ) * anno.field_dim,
                                          layout=Layout.AOS))
                    elif anno.element_dim == 2:
                        injected_args.append(
                            MatrixNdarray(anno.element_shape[0],
                                          anno.element_shape[1],
                                          dtype=f32,
                                          shape=(2, ) * anno.field_dim,
                                          layout=Layout.AOS))
                    else:
                        raise RuntimeError('')
            else:
                # For primitive types, we can just inject a dummy value.
                injected_args.append(0)
            i = i + 1
        kernel.ensure_compiled(*injected_args)
        self._aot_builder.add(name, kernel.kernel_cpp)

        # kernel AOT
        self._kernels.append(kernel)

    @contextmanager
    def add_kernel_template(self, kernel_fn):
        """Add a taichi kernel (with template parameters) to the AOT module.

        Args:
          kernel_fn (Function): the function decorated by taichi `kernel`.

        Example::

            >>> @ti.kernel
            >>> def bar_tmpl(a: ti.template()):
            >>>   x = a
            >>>   # or y = a
            >>>   # do something with `x` or `y`
            >>>
            >>> m = ti.aot.Module(arch)
            >>> with m.add_kernel_template(bar_tmpl) as kt:
            >>>   kt.instantiate(a=x)
            >>>   kt.instantiate(a=y)
            >>>
            >>> @ti.kernel
            >>> def bar_tmpl_multiple_args(a: ti.template(), b: ti.template())
            >>>   x = a
            >>>   y = b
            >>>   # do something with `x` and `y`
            >>>
            >>> with m.add_kernel_template(bar_tmpl) as kt:
            >>>   kt.instantiate(a=x, b=y)

        TODO:
          * Support external array
        """
        kt = KernelTemplate(kernel_fn, self)
        yield kt

    def save(self, filepath, filename):
        """
        Args:
          filepath (str): path to a folder to store aot files.
          filename (str): filename prefix for stored aot files.
        """
        filepath = str(PurePosixPath(Path(filepath)))
        self._aot_builder.dump(filepath, filename)
