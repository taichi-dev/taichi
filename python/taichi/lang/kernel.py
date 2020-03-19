import re
import inspect
from .transformer import ASTTransformer
import ast
from .kernel_arguments import *
from .util import *
import functools


def remove_indent(lines):
    lines = lines.split('\n')
    to_remove = 0
    for i in range(len(lines[0])):
        if lines[0][i] == ' ':
            to_remove = i + 1
        else:
            break

    cleaned = []
    for l in lines:
        cleaned.append(l[to_remove:])
        if len(l) >= to_remove:
            for i in range(to_remove):
                assert l[i] == ' '

    return '\n'.join(cleaned)


# The ti.func decorator
def func(foo):
    if _inside_class(level_of_class_stackframe=3):
        func = Func(foo, classfunc=True)

        @functools.wraps(foo)
        def decorated(*args):
            return func.__call__(*args)

        return decorated
    else:
        return Func(foo)


class Func:
    def __init__(self, func, classfunc=False):
        self.func = func
        self.compiled = None
        self.classfunc = classfunc

    def __call__(self, *args):
        if self.compiled is None:
            self.do_compile()
        ret = self.compiled(*args)
        return ret

    def do_compile(self):
        from .impl import get_runtime
        src = remove_indent(inspect.getsource(self.func))
        tree = ast.parse(src)

        func_body = tree.body[0]
        func_body.decorator_list = []

        if get_runtime().print_preprocessed:
            import astor
            print('Before preprocessing:')
            print(astor.to_source(tree.body[0], indent_with='  '))

        visitor = ASTTransformer(is_kernel=False, is_classfunc=self.classfunc)
        visitor.visit(tree)
        ast.fix_missing_locations(tree)

        if get_runtime().print_preprocessed:
            import astor
            print('After preprocessing:')
            print(astor.to_source(tree.body[0], indent_with='  '))

        ast.increment_lineno(tree, inspect.getsourcelines(self.func)[1] - 1)

        local_vars = {}
        #frame = inspect.currentframe().f_back
        #global_vars = dict(frame.f_globals, **frame.f_locals)
        import copy
        global_vars = copy.copy(self.func.__globals__)
        exec(
            compile(tree,
                    filename=inspect.getsourcefile(self.func),
                    mode='exec'), global_vars, local_vars)
        self.compiled = local_vars[self.func.__name__]


def classfunc(foo):
    import warnings
    warnings.warn('@ti.classfunc is deprecated. Please use @ti.func directly.',
                  DeprecationWarning)

    func = Func(foo, classfunc=True)

    @functools.wraps(foo)
    def decorated(*args):
        return func.__call__(*args)

    return decorated


class KernelTemplateMapper:
    def __init__(self, annotations, template_slot_locations):
        self.annotations = annotations
        # Make sure extractors's size is the same as the number of args
        dummy_extract = lambda arg: (type(arg).__name__, )
        self.extractors = tuple((i, getattr(anno, 'extract', dummy_extract))
                                for (i, anno) in enumerate(self.annotations))
        self.num_args = len(annotations)
        self.template_slot_locations = template_slot_locations
        self.mapping = {}

    def extract(self, args):
        extracted = []
        for i, extractor in self.extractors:
            extracted.append(extractor(args[i]))
        return tuple(extracted)

    def lookup(self, args):
        if len(args) != self.num_args:
            raise Exception(
                f'{self.num_args} argument(s) needed but {len(args)} provided.'
            )

        key = self.extract(args)
        if key not in self.mapping:
            count = len(self.mapping)
            self.mapping[key] = count
        return self.mapping[key], key


class KernelDefError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class KernelArgError(Exception):
    def __init__(self, pos, needed, provided):
        self.pos = pos
        self.needed = needed
        self.provided = provided

    def message(self):
        return 'Argument {} (type={}) cannot be converted into required type {}'.format(
            self.pos, str(self.needed), str(self.provided))


class Kernel:
    counter = 0

    def __init__(self, func, is_grad, classkernel=False):
        self.func = func
        self.kernel_counter = Kernel.counter
        Kernel.counter += 1
        self.is_grad = is_grad
        self.arguments = []
        self.argument_names = []
        self.classkernel = classkernel
        self.extract_arguments()
        self.template_slot_locations = []
        for i in range(len(self.arguments)):
            if isinstance(self.arguments[i], template):
                self.template_slot_locations.append(i)
        self.mapper = KernelTemplateMapper(self.arguments,
                                           self.template_slot_locations)
        from .impl import get_runtime
        get_runtime().kernels.append(self)
        self.reset()

    def reset(self):
        from .impl import get_runtime
        self.runtime = get_runtime()
        if self.is_grad:
            self.compiled_functions = self.runtime.compiled_functions
        else:
            self.compiled_functions = self.runtime.compiled_grad_functions

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise KernelDefError(
                    'Taichi kernels do not support variable keyword parameters (i.e., **kwargs)'
                )
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise KernelDefError(
                    'Taichi kernels do not support variable positional parameters (i.e., *args)'
                )
            if param.default is not inspect.Parameter.empty:
                raise KernelDefError(
                    'Taichi kernels do not support default values for arguments'
                )
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise KernelDefError(
                    'Taichi kernels do not support keyword parameters')
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise KernelDefError(
                    'Taichi kernels only support "positional or keyword" parameters'
                )
            annotation = param.annotation
            if param.annotation is inspect.Parameter.empty:
                if i == 0 and self.classkernel:
                    annotation = template()
                else:
                    raise KernelDefError(
                        'Taichi kernels parameters must be type annotated')
            self.arguments.append(annotation)
            self.argument_names.append(param.name)

    def materialize(self, key=None, args=None, arg_features=None):
        if key is None:
            key = (self.func, 0)
        if not self.runtime.materialized:
            self.runtime.materialize()
        if key in self.compiled_functions:
            return
        grad_suffix = ""
        if self.is_grad:
            grad_suffix = "_grad"
        kernel_name = "{}_c{}_{}{}".format(self.func.__name__,
                                           self.kernel_counter, key[1],
                                           grad_suffix)
        import taichi as ti
        ti.trace("Compiling kernel {}...".format(kernel_name))

        src = remove_indent(inspect.getsource(self.func))
        tree = ast.parse(src)
        if self.runtime.print_preprocessed:
            import astor
            print('Before preprocessing:')
            print(astor.to_source(tree.body[0]))

        func_body = tree.body[0]
        func_body.decorator_list = []

        local_vars = {}
        # Discussions: https://github.com/yuanming-hu/taichi/issues/282
        import copy
        global_vars = copy.copy(self.func.__globals__)

        for i, arg in enumerate(func_body.args.args):
            anno = arg.annotation
            if isinstance(anno, ast.Name):
                global_vars[anno.id] = self.arguments[i]

        if self.is_grad:
            from .ast_checker import KernelSimplicityASTChecker
            KernelSimplicityASTChecker(self.func).visit(tree)

        visitor = ASTTransformer(
            excluded_paremeters=self.template_slot_locations,
            func=self,
            arg_features=arg_features)

        visitor.visit(tree)
        ast.fix_missing_locations(tree)

        if self.runtime.print_preprocessed:
            import astor
            print('After preprocessing:')
            print(astor.to_source(tree.body[0], indent_with='  '))

        ast.increment_lineno(tree, inspect.getsourcelines(self.func)[1] - 1)

        freevar_names = self.func.__code__.co_freevars
        closure = self.func.__closure__
        if closure:
            freevar_values = list(map(lambda x: x.cell_contents, closure))
            for name, value in zip(freevar_names, freevar_values):
                global_vars[name] = value

        # inject template parameters into globals
        for i in self.template_slot_locations:
            template_var_name = self.argument_names[i]
            global_vars[template_var_name] = args[i]

        exec(
            compile(tree,
                    filename=inspect.getsourcefile(self.func),
                    mode='exec'), global_vars, local_vars)
        compiled = local_vars[self.func.__name__]

        taichi_kernel = taichi_lang_core.create_kernel(kernel_name,
                                                       self.is_grad)

        # Do not change the name of 'taichi_ast_generator'
        # The warning system needs this identifier to remove unnecessary messages
        def taichi_ast_generator():
            if self.runtime.inside_kernel:
                import taichi as ti
                raise ti.TaichiSyntaxError(
                    "Kernels cannot call other kernels. I.e., nested kernels are not allowed. Please check if you have direct/indirect invocation of kernels within kernels. Note that some methods provided by the Taichi standard library may invoke kernels, and please move their invocations to Python-scope."
                )
            self.runtime.inside_kernel = True
            compiled()
            self.runtime.inside_kernel = False

        taichi_kernel = taichi_kernel.define(taichi_ast_generator)

        assert key not in self.compiled_functions
        self.compiled_functions[key] = self.get_function_body(taichi_kernel)

    def get_function_body(self, t_kernel):
        # The actual function body
        def func__(*args):
            assert len(args) == len(
                self.arguments), '{} arguments needed but {} provided'.format(
                    len(self.arguments), len(args))

            tmps = []
            callbacks = []

            actual_argument_slot = 0
            for i, v in enumerate(args):
                needed = self.arguments[i]
                if isinstance(needed, template):
                    continue
                provided = type(v)
                # Note: do not use sth like needed == f32. That would be slow.
                if id(needed) in real_type_ids:
                    if not isinstance(v, (float, int)):
                        raise KernelArgError(i, needed, provided)
                    t_kernel.set_arg_float(actual_argument_slot, float(v))
                elif id(needed) in integer_type_ids:
                    if not isinstance(v, int):
                        raise KernelArgError(i, needed, provided)
                    t_kernel.set_arg_int(actual_argument_slot, int(v))
                elif self.match_ext_arr(v, needed):
                    dt = to_taichi_type(v.dtype)
                    has_torch = has_pytorch()
                    is_numpy = isinstance(v, np.ndarray)
                    if is_numpy:
                        tmp = np.ascontiguousarray(v)
                        tmps.append(tmp)  # Purpose: do not GC tmp!
                        t_kernel.set_arg_nparray(actual_argument_slot,
                                                 int(tmp.ctypes.data),
                                                 tmp.nbytes)
                    else:

                        def get_call_back(u, v):
                            def call_back():
                                u.copy_(v)

                            return call_back

                        assert has_torch and isinstance(v, torch.Tensor)
                        tmp = v
                        taichi_arch = self.runtime.prog.config.arch

                        if str(v.device).startswith('cuda'):
                            # External tensor on cuda
                            if taichi_arch != taichi_lang_core.Arch.cuda:
                                # copy data back to cpu
                                host_v = v.to(device='cpu', copy=True)
                                tmp = host_v
                                callbacks.append(get_call_back(v, host_v))
                        else:
                            # External tensor on cpu
                            if taichi_arch == taichi_lang_core.Arch.cuda:
                                gpu_v = v.cuda()
                                tmp = gpu_v
                                callbacks.append(get_call_back(v, gpu_v))
                        t_kernel.set_arg_nparray(
                            actual_argument_slot, int(tmp.data_ptr()),
                            tmp.element_size() * tmp.nelement())
                    shape = v.shape
                    max_num_indices = taichi_lang_core.get_max_num_indices()
                    assert len(
                        shape
                    ) <= max_num_indices, "External array cannot have > {} indices".format(
                        max_num_indices)
                    for i, s in enumerate(shape):
                        t_kernel.set_extra_arg_int(actual_argument_slot, i, s)
                else:
                    assert False
                actual_argument_slot += 1
            # Both the class kernels and the plain-function kernels are unified now.
            # In both cases, |self.grad| is another Kernel instance that computes the
            # gradient. For class kerenls, args[0] is always the kernel owner.
            if not self.is_grad and self.runtime.target_tape and not self.runtime.inside_complex_kernel:
                self.runtime.target_tape.insert(self, args)

            t_kernel()

            if callbacks:
                import taichi as ti
                ti.sync()
                for c in callbacks:
                    c()

        return func__

    def match_ext_arr(self, v, needed):
        needs_array = isinstance(
            needed, np.ndarray) or needed == np.ndarray or isinstance(
                needed, ext_arr)
        has_array = isinstance(v, np.ndarray)
        if not has_array and has_pytorch():
            has_array = isinstance(v, torch.Tensor)
        return has_array and needs_array

    # For small kernels (< 3us), the performance can be pretty sensitive to overhead in __call__
    # Thus this part needs to be fast. (i.e. < 3us on a 4 GHz x64 CPU)
    def __call__(self, *args, **kwargs):
        assert len(kwargs) == 0, 'kwargs not supported for Taichi kernels'
        instance_id, arg_features = self.mapper.lookup(args)
        key = (self.func, instance_id)
        self.materialize(key=key, args=args, arg_features=arg_features)
        return self.compiled_functions[key](*args)


# For a Taichi class definition like below:
#
# @ti.data_oriented
# class X:
#   @ti.kernel
#   def foo(self):
#     ...
#
# When ti.kernel runs, the stackframe's |code_context| of Python 3.8(+) is
# different from that of Python 3.7 and below. In 3.8+, it is 'class X:',
# whereas in <=3.7, it is '@ti.data_oriented'. More interestingly, if the class
# inherits, i.e. class X(object):, then in both versions, |code_context| is
# 'class X(object):'...
_KERNEL_CLASS_STACKFRAME_STMT_RES = [
    re.compile(r'@(\w+\.)?data_oriented'),
    re.compile(r'class '),
]


def _inside_class(level_of_class_stackframe):
    import inspect
    frames = inspect.stack()
    try:
        maybe_class_frame = frames[level_of_class_stackframe]
        statement_list = maybe_class_frame[4]
        first_statment = statement_list[0].strip()
        for pat in _KERNEL_CLASS_STACKFRAME_STMT_RES:
            if pat.match(first_statment):
                return True
    except:
        pass
    return False


def _kernel_impl(func, level_of_class_stackframe, verbose=False):
    # Can decorators determine if a function is being defined inside a class?
    # https://stackoverflow.com/a/8793684/12003165
    is_classkernel = _inside_class(level_of_class_stackframe + 1)

    if verbose:
        print(f'kernel={func.__name__} is_classkernel={is_classkernel}')
    primal = Kernel(func, is_grad=False, classkernel=is_classkernel)
    adjoint = Kernel(func, is_grad=True, classkernel=is_classkernel)
    # Having |primal| contains |grad| makes the tape work.
    primal.grad = adjoint

    from functools import wraps
    if is_classkernel:
        # For class kernels, their primal/adjoint callables are constructed when the
        # kernel is accessed via the instance inside BoundedDifferentiableMethod.
        # This is because we need to bind the kernel or |grad| to the instance
        # owning the kernel, which is not known until the kernel is accessed.
        #
        # See also: BoundedDifferentiableMethod, data_oriented.
        @wraps(func)
        def wrapped(*args, **kwargs):
            # If we reach here (we should never), it means the class is not decorated
            # with @data_oriented, otherwise getattr would have intercepted the call.
            clsobj = type(args[0])
            assert not hasattr(clsobj, '_data_oriented')
            raise KernelDefError(
                f'Please decorate class {clsobj.__name__} with @data_oriented')
    else:

        @wraps(func)
        def wrapped(*args, **kwargs):
            primal(*args, **kwargs)

        wrapped.grad = adjoint

    wrapped._is_wrapped_kernel = True
    wrapped._is_classkernel = is_classkernel
    wrapped._primal = primal
    wrapped._adjoint = adjoint
    return wrapped


def kernel(func):
    return _kernel_impl(func, level_of_class_stackframe=3)


def classkernel(func):
    import warnings
    warnings.warn(
        '@ti.classkernel is deprecated. Please use @ti.kernel directly.',
        DeprecationWarning)
    return _kernel_impl(func, level_of_class_stackframe=3)


class BoundedDifferentiableMethod:
    def __init__(self, kernel_owner, wrapped_kernel_func):
        clsobj = type(kernel_owner)
        if not getattr(clsobj, '_data_oriented', False):
            raise KernelDefError(
                f'Please decorate class {clsobj.__name__} with @data_oriented')
        self._kernel_owner = kernel_owner
        self._primal = wrapped_kernel_func._primal
        self._adjoint = wrapped_kernel_func._adjoint

    def __call__(self, *args, **kwargs):
        return self._primal(self._kernel_owner, *args, **kwargs)

    def grad(self, *args, **kwargs):
        return self._adjoint(self._kernel_owner, *args, **kwargs)


def data_oriented(cls):
    def getattr(self, item):
        x = super(cls, self).__getattribute__(item)
        if hasattr(x, '_is_wrapped_kernel'):
            import inspect
            assert inspect.ismethod(x)
            wrapped = x.__func__
            assert inspect.isfunction(wrapped)
            if wrapped._is_classkernel:
                return BoundedDifferentiableMethod(self, wrapped)
        return x

    cls.__getattribute__ = getattr
    cls._data_oriented = True

    return cls
