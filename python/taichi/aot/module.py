from taichi.lang import impl, kernel_arguments, kernel_impl


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
        self._arch = arch
        self._kernels = []
        impl.get_runtime().materialize()
        self._aot_builder = impl.get_runtime().prog.make_aot_module_builder(
            arch)

    def add_kernel(self, kernel_fn, name=None, template_args=None):
        """Add a taichi kernel to the AOT module.

        Args:
          kernel_fn (Function): the function decorated by taichi `kernel`.
          name (str): Name to identify this kernel in the module. If not
            provided, uses the built-in ``__name__`` attribute of `kernel_fn`.
          template_args (dict[str: Any]): Used to instantiate the template
            parameters in the passed in function, this is because the template
            parameters must be known at compile time.

        Example:
          Note that if `kernel_fn` contains at least one template parameter, it
          is required that users provide an explicit `name`. In addition, all
          the values of these template parameters must be instantiated via
          `template_args`.

          Usage::

            @ti.kernel
            def bar(a: ti.template()):
              x = a
              # do something with `x`

            m = ti.aot.Module(arch)
            m.add_kernel(bar, name='bar_a=True', template_args={'a': True})

          Later on, the ``bar`` kernel instantiated with ``a = True`` can be
          found in the module via ``"bar_a=True"``.

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

            if isinstance(anno, kernel_arguments.Template):
                value = template_args[kernel.argument_names[i]]
                injected_args.append(value)
            else:
                # For primitive types, we can just inject a dummy value.
                injected_args.append(0)
        kernel.ensure_compiled(*injected_args)
        self._aot_builder.add(name, kernel.kernel_cpp)

        # kernel AOT
        self._kernels.append(kernel)

    def save(self, filepath, filename):
        self._aot_builder.dump(filepath, filename)
