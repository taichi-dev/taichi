from taichi.lang import impl, kernel_arguments, kernel_impl, expr, matrix
from contextlib import contextmanager
import sys


class KernelTemplate(object):
      def __init__(self, kernel_fn, aot_module):
        self.kernel_fn = kernel_fn
        self.aot_module = aot_module
      
      def instantiate(self, **kwargs):
        name = self.kernel_fn.__name__
        kernel = self.kernel_fn._primal
        assert isinstance(kernel, kernel_impl.Kernel)
        injected_args = []
        key_p = ""
        anno_index = 0
        template_args = {}

        for index, (key, value) in enumerate(kwargs.items()):
          template_args[index] = (key, value)

        for i in range(len(kernel.argument_annotations)):
          anno = kernel.argument_annotations[i]
          if isinstance(anno, kernel_arguments.template):
            (k, v) = template_args[anno_index]
            key_p += k
            for ky, val in self.aot_module.fields.items():
              if (val is v):
                key_p += "=" + ky + "/"
            injected_args.append(v)
            anno_index += 1
          else:
            injected_args.append(0)
        
        kernel.ensure_compiled(*injected_args)
        self.aot_module.aot_builder.add_kernel_template(name, key_p, kernel.kernel_cpp)

        # kernel AOT
        self.aot_module.kernels.append(kernel)


class Module:
    """An AOT module to save and load Taichi kernels.

    This module serializes the Taichi kernels for a specific arch. The
    serialized module can later be loaded to run on that backend, without the
    Python environment.

    Example::

        m = ti.aot.Module(ti.metal)
        m.add_kernel(foo)
        m.add_kernel(bar)

        m.save('/path/to/module')

        # Now the module file '/path/to/module' contains the Metal kernels
        # for running ``foo`` and ``bar``.
    """
    def __init__(self, arch):
        self.arch = arch
        self.kernels = []
        self.fields = {}
        impl.get_runtime().materialize()
        self.aot_builder = impl.get_runtime().prog.make_aot_module_builder(
            arch)
    
    def add_field(self, name, field):
      """Add a taichi field to the AOT module.
      Args: 
        name: name of taichi field
        field: taichi field

        a = ti.field("something")
        b = ti.field("something")

        m.add_field(a)
        m.add_field(b)
        
        Must add in sequence
      """
      # assert isinstance(field, expr.Expr)

      # TODO: pass field.shape, field.dt to aotModuleBuilder
      field_number = len(self.fields)
      # print(field_number)
      is_vector = False
      self.fields[name] = field
      if type(field) is matrix.Matrix:
        assert isinstance(field, matrix.Matrix)
        # print(field.shape)
        # print(type(field.shape))
        # print(field.dtype)
        # print(sys.getsizeof(field))
        is_vector = True
      else:
        assert isinstance(field, expr.Expr)
      self.aot_builder.add_field(name, is_vector, field.dtype)

    def add_kernel(self, kernel_fn, name=None):
        """Add a taichi kernel to the AOT module.

        Args:
          kernel_fn (Function): the function decorated by taichi `kernel`.
          name (str): Name to identify this kernel in the module. If not
            provided, uses the built-in ``__name__`` attribute of `kernel_fn`.

        TODO:
          * Support external array
        """
        name = name or kernel_fn.__name__
        kernel = kernel_fn._primal
        assert isinstance(kernel, kernel_impl.Kernel)
        injected_args = []
        for i in range(len(kernel.argument_annotations)):
            anno = kernel.argument_annotations[i]
            if isinstance(anno, kernel_arguments.ArgExtArray):
                raise RuntimeError('Arg type `ext_arr` not supported yet')
            else:
                # For primitive types, we can just inject a dummy value.
                injected_args.append(0)
        kernel.ensure_compiled(*injected_args)
        self.aot_builder.add(name, kernel.kernel_cpp)

        # kernel AOT
        self.kernels.append(kernel)

    @contextmanager
    def add_kernel_template(self, kernel_fn):
        """Add a taichi kernel (with template parameters) to the AOT module.

        Args:
          kernel_fn (Function): the function decorated by taichi `kernel`.
          name (str): Name to identify this kernel in the module. If not
            provided, uses the built-in ``__name__`` attribute of `kernel_fn`.


        Example:
          Note that if `kernel_fn` contains at least one template parameter, it
          is required that users provide an explicit `name`. In addition, all
          the values of these template parameters must be instantiated via
          `template_args`.

          Usage::

            @ti.kernel
            def bar_tmpl(a: ti.template()):
              x = a
              # or y = a
              # do something with `x` or `y`

            m = ti.aot.Module(arch)
            with m.add_kernel_template(bar_tmpl) as kt:
              kt.instantiate(a=x)
              kt.instantiate(a=y)

            @ti.kernel
            def bar_tmpl_multiple_args(a: ti.template(), b: ti.template())
              x = a
              y = b
              # do something with `x` and `y`
            
            with m.add_kernel_template(bar_tmpl) as kt:
              kt.instantiate(a=x, b=y)

        TODO:
          * Support external array
        """
        kt = KernelTemplate(kernel_fn, self)
        yield kt

    def save(self, filepath, filename):
        self.aot_builder.dump(filepath, filename)
