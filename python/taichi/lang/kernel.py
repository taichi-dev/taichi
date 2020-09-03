import re
import inspect
from .transformer import ASTTransformer
import ast
from .kernel_arguments import *
from .util import *
from .shell import oinspect, _shell_pop_print
from .exception import TaichiSyntaxError
from . import impl
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
    _taichi_skip_traceback = 1
    fun = Func(foo)

    @functools.wraps(foo)
    def decorated(*args):
        _taichi_skip_traceback = 1
        return fun.__call__(*args)

    return decorated


# The ti.pyfunc decorator
def pyfunc(foo):
    '''
    Creates a function that are callable both in Taichi-scope and Python-scope.
    The function should be simple, and not contains Taichi-scope specifc syntax
    including struct-for.
    '''
    fun = Func(foo, pyfunc=True)

    @functools.wraps(foo)
    def decorated(*args):
        _taichi_skip_traceback = 1
        return fun.__call__(*args)

    return decorated


class Func:
    def __init__(self, func, pyfunc=False):
        self.func = func
        self.compiled = None
        self.pyfunc = pyfunc
        self.arguments = []
        self.argument_names = []
        _taichi_skip_traceback = 1
        self.extract_arguments()

    def __call__(self, *args):
        _taichi_skip_traceback = 1
        if not impl.inside_kernel():
            if not self.pyfunc:
                raise TaichiSyntaxError(
                    "Taichi functions cannot be called from Python-scope."
                    " Use @ti.pyfunc if you wish to call Taichi functions "
                    "from both Python-scope and Taichi-scope.")
            return self.func(*args)
        if self.compiled is None:
            self.do_compile()
        ret = self.compiled(*args)
        return ret

    def do_compile(self):
        src = remove_indent(oinspect.getsource(self.func))
        tree = ast.parse(src)

        func_body = tree.body[0]
        func_body.decorator_list = []

        visitor = ASTTransformer(is_kernel=False, func=self)
        visitor.visit(tree)

        ast.increment_lineno(tree, oinspect.getsourcelines(self.func)[1] - 1)

        local_vars = {}
        global_vars = _get_global_vars(self.func)

        exec(
            compile(tree,
                    filename=oinspect.getsourcefile(self.func),
                    mode='exec'), global_vars, local_vars)
        self.compiled = local_vars[self.func.__name__]

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect._empty, None):
            self.return_type = sig.return_annotation
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise KernelDefError(
                    'Taichi functions do not support variable keyword parameters (i.e., **kwargs)'
                )
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise KernelDefError(
                    'Taichi functions do not support variable positional parameters (i.e., *args)'
                )
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise KernelDefError(
                    'Taichi functions do not support keyword parameters')
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise KernelDefError(
                    'Taichi functions only support "positional or keyword" parameters'
                )
            annotation = param.annotation
            if annotation is not inspect.Parameter.empty:
                if id(annotation) in type_ids:
                    warning(
                        'Data type annotations are unnecessary for Taichi'
                        ' functions, consider removing it',
                        stacklevel=4)
                elif not isinstance(annotation, template):
                    raise KernelDefError(
                        f'Invalid type annotation (argument {i}) of Taichi function: {annotation}'
                    )
            self.arguments.append(annotation)
            self.argument_names.append(param.name)


classfunc = obsolete('@ti.classfunc', '@ti.func directly')


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
            _taichi_skip_traceback = 1
            raise TypeError(
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


def _get_global_vars(func):
    # Discussions: https://github.com/taichi-dev/taichi/issues/282
    import copy
    global_vars = copy.copy(func.__globals__)

    freevar_names = func.__code__.co_freevars
    closure = func.__closure__
    if closure:
        freevar_values = list(map(lambda x: x.cell_contents, closure))
        for name, value in zip(freevar_names, freevar_values):
            global_vars[name] = value

    return global_vars


class Kernel:
    counter = 0

    def __init__(self, func, is_grad):
        self.func = func
        self.kernel_counter = Kernel.counter
        Kernel.counter += 1
        self.is_grad = is_grad
        self.arguments = []
        self.argument_names = []
        self.return_type = None
        _taichi_skip_traceback = 1
        self.extract_arguments()
        del _taichi_skip_traceback
        self.template_slot_locations = []
        for i in range(len(self.arguments)):
            if isinstance(self.arguments[i], template):
                self.template_slot_locations.append(i)
        self.mapper = KernelTemplateMapper(self.arguments,
                                           self.template_slot_locations)
        impl.get_runtime().kernels.append(self)
        self.reset()

    def reset(self):
        self.runtime = impl.get_runtime()
        if self.is_grad:
            self.compiled_functions = self.runtime.compiled_functions
        else:
            self.compiled_functions = self.runtime.compiled_grad_functions

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect._empty, None):
            self.return_type = sig.return_annotation
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
                annotation = template()
            elif isinstance(annotation, (template, ext_arr)):
                pass
            elif id(annotation) in type_ids:
                pass
            else:
                _taichi_skip_traceback = 1
                raise KernelDefError(
                    f'Invalid type annotation (argument {i}) of Taichi kernel: {annotation}'
                )
            self.arguments.append(annotation)
            self.argument_names.append(param.name)

    def materialize(self, key=None, args=None, arg_features=None):
        _taichi_skip_traceback = 1
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

        src = remove_indent(oinspect.getsource(self.func))
        tree = ast.parse(src)

        func_body = tree.body[0]
        func_body.decorator_list = []

        local_vars = {}
        global_vars = _get_global_vars(self.func)

        for i, arg in enumerate(func_body.args.args):
            anno = arg.annotation
            if isinstance(anno, ast.Name):
                global_vars[anno.id] = self.arguments[i]

        if isinstance(func_body.returns, ast.Name):
            global_vars[func_body.returns.id] = self.return_type

        if self.is_grad:
            from .ast_checker import KernelSimplicityASTChecker
            KernelSimplicityASTChecker(self.func).visit(tree)

        visitor = ASTTransformer(
            excluded_paremeters=self.template_slot_locations,
            func=self,
            arg_features=arg_features)

        visitor.visit(tree)

        ast.increment_lineno(tree, oinspect.getsourcelines(self.func)[1] - 1)

        # inject template parameters into globals
        for i in self.template_slot_locations:
            template_var_name = self.argument_names[i]
            global_vars[template_var_name] = args[i]

        exec(
            compile(tree,
                    filename=oinspect.getsourcefile(self.func),
                    mode='exec'), global_vars, local_vars)
        compiled = local_vars[self.func.__name__]

        taichi_kernel = taichi_lang_core.create_kernel(kernel_name,
                                                       self.is_grad)

        # Do not change the name of 'taichi_ast_generator'
        # The warning system needs this identifier to remove unnecessary messages
        def taichi_ast_generator():
            _taichi_skip_traceback = 1
            if self.runtime.inside_kernel:
                import taichi as ti
                raise TaichiSyntaxError(
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
            has_external_arrays = False

            actual_argument_slot = 0
            launch_ctx = t_kernel.make_launch_context()
            for i, v in enumerate(args):
                needed = self.arguments[i]
                if isinstance(needed, template):
                    continue
                provided = type(v)
                # Note: do not use sth like "needed == f32". That would be slow.
                if id(needed) in real_type_ids:
                    if not isinstance(v, (float, int)):
                        raise KernelArgError(i, needed, provided)
                    launch_ctx.set_arg_float(actual_argument_slot, float(v))
                elif id(needed) in integer_type_ids:
                    if not isinstance(v, int):
                        raise KernelArgError(i, needed, provided)
                    launch_ctx.set_arg_int(actual_argument_slot, int(v))
                elif self.match_ext_arr(v, needed):
                    has_external_arrays = True
                    has_torch = has_pytorch()
                    is_numpy = isinstance(v, np.ndarray)
                    if is_numpy:
                        tmp = np.ascontiguousarray(v)
                        tmps.append(tmp)  # Purpose: do not GC tmp!
                        launch_ctx.set_arg_nparray(actual_argument_slot,
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
                        launch_ctx.set_arg_nparray(
                            actual_argument_slot, int(tmp.data_ptr()),
                            tmp.element_size() * tmp.nelement())
                    shape = v.shape
                    max_num_indices = taichi_lang_core.get_max_num_indices()
                    assert len(
                        shape
                    ) <= max_num_indices, "External array cannot have > {} indices".format(
                        max_num_indices)
                    for i, s in enumerate(shape):
                        launch_ctx.set_extra_arg_int(actual_argument_slot, i,
                                                     s)
                else:
                    raise ValueError(
                        f'Argument type mismatch. Expecting {needed}, got {type(v)}.'
                    )
                actual_argument_slot += 1
            # Both the class kernels and the plain-function kernels are unified now.
            # In both cases, |self.grad| is another Kernel instance that computes the
            # gradient. For class kernels, args[0] is always the kernel owner.
            if not self.is_grad and self.runtime.target_tape and not self.runtime.inside_complex_kernel:
                self.runtime.target_tape.insert(self, args)

            t_kernel(launch_ctx)

            ret = None
            ret_dt = self.return_type
            has_ret = ret_dt is not None

            if has_external_arrays or has_ret:
                import taichi as ti
                ti.sync()

            if has_ret:
                if id(ret_dt) in integer_type_ids:
                    ret = t_kernel.get_ret_int(0)
                else:
                    ret = t_kernel.get_ret_float(0)

            if callbacks:
                for c in callbacks:
                    c()

            return ret

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
    @_shell_pop_print
    def __call__(self, *args, **kwargs):
        _taichi_skip_traceback = 1
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


def kernel(func):
    _taichi_skip_traceback = 1

    primal = Kernel(func, is_grad=False)
    adjoint = Kernel(func, is_grad=True)
    # Having |primal| contains |grad| makes the tape work.
    primal.grad = adjoint

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        _taichi_skip_traceback = 1
        return primal(*args, **kwargs)

    @functools.wraps(func)
    def wrapped_grad(*args, **kwargs):
        _taichi_skip_traceback = 1
        return adjoint(*args, **kwargs)

    if not _inside_class(level_of_class_stackframe=3):
        wrapped.grad = wrapped_grad
        return wrapped

    class BoundKernelProperty(property):
        @functools.wraps(func)
        def __call__(self, *args, **kwargs):
            _taichi_skip_traceback = 1
            return primal(*args, **kwargs)

        @functools.wraps(func)
        def grad(self, *args, **kwargs):
            _taichi_skip_traceback = 1
            return adjoint(*args, **kwargs)

    @functools.wraps(func)
    @BoundKernelProperty
    def prop_kernel(self):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            _taichi_skip_traceback = 1
            return primal(self, *args, **kwargs)

        @functools.wraps(func)
        def wrapped_grad(*args, **kwargs):
            _taichi_skip_traceback = 1
            return adjoint(self, *args, **kwargs)

        wrapped.grad = wrapped_grad
        return wrapped

    return prop_kernel


def data_oriented(cls):
    cls._data_oriented = True
    return cls


classkernel = obsolete('@ti.classkernel', '@ti.kernel directly')
