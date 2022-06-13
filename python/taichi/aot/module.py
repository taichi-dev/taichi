from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from taichi.aot.utils import (produce_injected_args,
                              produce_injected_args_from_template)
from taichi.lang import impl, kernel_impl
from taichi.lang.field import ScalarField
from taichi.lang.matrix import MatrixField
from taichi.types.annotations import template


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
        raise RuntimeError('Arg type must be of type int/float/boolean'
                           f' or taichi field. Type {str(type(v))}'
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

        for arg in kernel.arguments:
            if isinstance(arg.annotation, template):
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

    def add_kernel(self, kernel_fn, template_args=None, name=None):
        """Add a taichi kernel to the AOT module.

        Args:
          kernel_fn (Function): the function decorated by taichi `kernel`.
          template_args (Dict[str, Any]): a dict where key is the template
            parameter name, and value is the instantiating arg. Note that this
            works for both :class:`~taichi.types.template` and for
            `:class:`~taichi.types.ndarray`.
          name (str): Name to identify this kernel in the module. If not
            provided, uses the built-in ``__name__`` attribute of `kernel_fn`.

        """
        kernel_name = name or kernel_fn.__name__
        kernel = kernel_fn._primal
        assert isinstance(kernel, kernel_impl.Kernel)
        if template_args is not None:
            injected_args = produce_injected_args_from_template(
                kernel, template_args)
        else:
            injected_args = produce_injected_args(kernel)
        kernel.ensure_compiled(*injected_args)
        self._aot_builder.add(kernel_name, kernel.kernel_cpp)

        # kernel AOT
        self._kernels.append(kernel)

    def add_graph(self, name, graph):
        self._aot_builder.add_graph(name, graph._compiled_graph)

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
