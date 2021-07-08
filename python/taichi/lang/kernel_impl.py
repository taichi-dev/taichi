import ast
import copy
import functools
import inspect
import re

import numpy as np
from taichi.core import primitive_types
from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl, util
from taichi.lang.ast_checker import KernelSimplicityASTChecker
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.kernel_arguments import ext_arr, template
from taichi.lang.shell import _shell_pop_print, oinspect
from taichi.lang.transformer import ASTTransformer
from taichi.misc.util import obsolete

import taichi as ti


def _remove_indent(lines):
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
    is_classfunc = _inside_class(level_of_class_stackframe=3)

    _taichi_skip_traceback = 1
    fun = Func(foo, classfunc=is_classfunc)

    @functools.wraps(foo)
    def decorated(*args):
        _taichi_skip_traceback = 1
        return fun.__call__(*args)

    decorated._is_taichi_function = True
    return decorated


# The ti.pyfunc decorator
def pyfunc(foo):
    '''
    Creates a function that are callable both in Taichi-scope and Python-scope.
    The function should be simple, and not contains Taichi-scope specifc syntax
    including struct-for.
    '''
    is_classfunc = _inside_class(level_of_class_stackframe=3)
    fun = Func(foo, classfunc=is_classfunc, pyfunc=True)

    @functools.wraps(foo)
    def decorated(*args):
        _taichi_skip_traceback = 1
        return fun.__call__(*args)

    decorated._is_taichi_function = True
    return decorated


class Func:
    function_counter = 0

    def __init__(self, func, classfunc=False, pyfunc=False):
        self.func = func
        self.func_id = Func.function_counter
        Func.function_counter += 1
        self.compiled = None
        self.classfunc = classfunc
        self.pyfunc = pyfunc
        self.argument_annotations = []
        self.argument_names = []
        _taichi_skip_traceback = 1
        self.extract_arguments()
        self.template_slot_locations = []
        for i in range(len(self.argument_annotations)):
            if isinstance(self.argument_annotations[i], template):
                self.template_slot_locations.append(i)
        self.mapper = TaichiCallableTemplateMapper(
            self.argument_annotations, self.template_slot_locations)
        self.taichi_functions = {}  # The |Function| class in C++

    def __call__(self, *args):
        _taichi_skip_traceback = 1
        if not impl.inside_kernel():
            if not self.pyfunc:
                raise TaichiSyntaxError(
                    "Taichi functions cannot be called from Python-scope."
                    " Use @ti.pyfunc if you wish to call Taichi functions "
                    "from both Python-scope and Taichi-scope.")
            return self.func(*args)

        if impl.get_runtime().experimental_real_function:
            if impl.get_runtime().current_kernel.is_grad:
                raise TaichiSyntaxError(
                    "Real function in gradient kernels unsupported.")
            instance_id, arg_features = self.mapper.lookup(args)
            key = _ti_core.FunctionKey(self.func.__name__, self.func_id,
                                       instance_id)
            if self.compiled is None:
                self.compiled = {}
            if key.instance_id not in self.compiled:
                self.do_compile(key=key, args=args)
            return self.func_call_rvalue(key=key, args=args)
        else:
            if self.compiled is None:
                self.do_compile(key=None, args=args)
            ret = self.compiled(*args)
            return ret

    def func_call_rvalue(self, key, args):
        # Skip the template args, e.g., |self|
        assert impl.get_runtime().experimental_real_function
        non_template_args = []
        for i in range(len(self.argument_annotations)):
            if not isinstance(self.argument_annotations[i], template):
                non_template_args.append(args[i])
        non_template_args = impl.make_expr_group(non_template_args)
        return ti.Expr(
            _ti_core.make_func_call_expr(
                self.taichi_functions[key.instance_id], non_template_args))

    def do_compile(self, key, args):
        src = _remove_indent(oinspect.getsource(self.func))
        tree = ast.parse(src)

        func_body = tree.body[0]
        func_body.decorator_list = []

        visitor = ASTTransformer(is_kernel=False, func=self)
        visitor.visit(tree)

        ast.increment_lineno(tree, oinspect.getsourcelines(self.func)[1] - 1)

        local_vars = {}
        global_vars = _get_global_vars(self.func)

        if impl.get_runtime().experimental_real_function:
            # inject template parameters into globals
            for i in self.template_slot_locations:
                template_var_name = self.argument_names[i]
                global_vars[template_var_name] = args[i]

        exec(
            compile(tree,
                    filename=oinspect.getsourcefile(self.func),
                    mode='exec'), global_vars, local_vars)

        if impl.get_runtime().experimental_real_function:
            self.compiled[key.instance_id] = local_vars[self.func.__name__]
            self.taichi_functions[key.instance_id] = _ti_core.create_function(
                key)
            self.taichi_functions[key.instance_id].set_function_body(
                self.compiled[key.instance_id])
        else:
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
            if annotation is inspect.Parameter.empty:
                if i == 0 and self.classfunc:
                    annotation = template()
            else:
                if not id(annotation
                          ) in primitive_types.type_ids and not isinstance(
                              annotation, template):
                    raise KernelDefError(
                        f'Invalid type annotation (argument {i}) of Taichi function: {annotation}'
                    )
            self.argument_annotations.append(annotation)
            self.argument_names.append(param.name)


class TaichiCallableTemplateMapper:
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
        message = f'Argument {pos} (type={provided}) cannot be converted into required type {needed}'
        super().__init__(message)
        self.pos = pos
        self.needed = needed
        self.provided = provided


def _get_global_vars(func):
    # Discussions: https://github.com/taichi-dev/taichi/issues/282
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

    def __init__(self, func, is_grad, classkernel=False):
        self.func = func
        self.kernel_counter = Kernel.counter
        Kernel.counter += 1
        self.is_grad = is_grad
        self.grad = None
        self.argument_annotations = []
        self.argument_names = []
        self.return_type = None
        self.classkernel = classkernel
        _taichi_skip_traceback = 1
        self.extract_arguments()
        del _taichi_skip_traceback
        self.template_slot_locations = []
        for i in range(len(self.argument_annotations)):
            if isinstance(self.argument_annotations[i], template):
                self.template_slot_locations.append(i)
        self.mapper = TaichiCallableTemplateMapper(
            self.argument_annotations, self.template_slot_locations)
        impl.get_runtime().kernels.append(self)
        self.reset()
        self.kernel_cpp = None

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
                if i == 0 and self.classkernel:  # The |self| parameter
                    annotation = template()
                else:
                    _taichi_skip_traceback = 1
                    raise KernelDefError(
                        'Taichi kernels parameters must be type annotated')
            else:
                if isinstance(annotation, (template, ext_arr)):
                    pass
                elif id(annotation) in primitive_types.type_ids:
                    pass
                else:
                    _taichi_skip_traceback = 1
                    raise KernelDefError(
                        f'Invalid type annotation (argument {i}) of Taichi kernel: {annotation}'
                    )
            self.argument_annotations.append(annotation)
            self.argument_names.append(param.name)

    def materialize(self, key=None, args=None, arg_features=None):
        _taichi_skip_traceback = 1
        if key is None:
            key = (self.func, 0)
        self.runtime.materialize()
        if key in self.compiled_functions:
            return
        grad_suffix = ""
        if self.is_grad:
            grad_suffix = "_grad"
        kernel_name = "{}_c{}_{}{}".format(self.func.__name__,
                                           self.kernel_counter, key[1],
                                           grad_suffix)
        ti.trace("Compiling kernel {}...".format(kernel_name))

        src = _remove_indent(oinspect.getsource(self.func))
        tree = ast.parse(src)

        func_body = tree.body[0]
        func_body.decorator_list = []

        local_vars = {}
        global_vars = _get_global_vars(self.func)

        for i, arg in enumerate(func_body.args.args):
            anno = arg.annotation
            if isinstance(anno, ast.Name):
                global_vars[anno.id] = self.argument_annotations[i]

        if isinstance(func_body.returns, ast.Name):
            global_vars[func_body.returns.id] = self.return_type

        if self.is_grad:
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

        taichi_kernel = _ti_core.create_kernel(kernel_name, self.is_grad)

        # Do not change the name of 'taichi_ast_generator'
        # The warning system needs this identifier to remove unnecessary messages
        def taichi_ast_generator():
            _taichi_skip_traceback = 1
            if self.runtime.inside_kernel:
                raise TaichiSyntaxError(
                    "Kernels cannot call other kernels. I.e., nested kernels are not allowed. Please check if you have direct/indirect invocation of kernels within kernels. Note that some methods provided by the Taichi standard library may invoke kernels, and please move their invocations to Python-scope."
                )
            self.runtime.inside_kernel = True
            self.runtime.current_kernel = self
            compiled()
            self.runtime.inside_kernel = False
            self.runtime.current_kernel = None

        taichi_kernel = taichi_kernel.define(taichi_ast_generator)
        self.kernel_cpp = taichi_kernel

        assert key not in self.compiled_functions
        self.compiled_functions[key] = self.get_function_body(taichi_kernel)

    def get_function_body(self, t_kernel):
        # The actual function body
        def func__(*args):
            assert len(args) == len(
                self.argument_annotations
            ), '{} arguments needed but {} provided'.format(
                len(self.argument_annotations), len(args))

            tmps = []
            callbacks = []
            has_external_arrays = False

            actual_argument_slot = 0
            launch_ctx = t_kernel.make_launch_context()
            for i, v in enumerate(args):
                needed = self.argument_annotations[i]
                if isinstance(needed, template):
                    continue
                provided = type(v)
                # Note: do not use sth like "needed == f32". That would be slow.
                if id(needed) in primitive_types.real_type_ids:
                    if not isinstance(v, (float, int)):
                        raise KernelArgError(i, needed.to_string(), provided)
                    launch_ctx.set_arg_float(actual_argument_slot, float(v))
                elif id(needed) in primitive_types.integer_type_ids:
                    if not isinstance(v, int):
                        raise KernelArgError(i, needed.to_string(), provided)
                    launch_ctx.set_arg_int(actual_argument_slot, int(v))
                elif self.match_ext_arr(v, needed):
                    has_external_arrays = True
                    has_torch = util.has_pytorch()
                    is_numpy = isinstance(v, np.ndarray)
                    if is_numpy:
                        tmp = np.ascontiguousarray(v)
                        # Purpose: DO NOT GC |tmp|!
                        tmps.append(tmp)
                        launch_ctx.set_arg_nparray(actual_argument_slot,
                                                   int(tmp.ctypes.data),
                                                   tmp.nbytes)
                    else:

                        def get_call_back(u, v):
                            def call_back():
                                u.copy_(v)

                            return call_back

                        assert has_torch
                        import torch
                        assert isinstance(v, torch.Tensor)
                        tmp = v
                        taichi_arch = self.runtime.prog.config.arch

                        if str(v.device).startswith('cuda'):
                            # External tensor on cuda
                            if taichi_arch != _ti_core.Arch.cuda:
                                # copy data back to cpu
                                host_v = v.to(device='cpu', copy=True)
                                tmp = host_v
                                callbacks.append(get_call_back(v, host_v))
                        else:
                            # External tensor on cpu
                            if taichi_arch == _ti_core.Arch.cuda:
                                gpu_v = v.cuda()
                                tmp = gpu_v
                                callbacks.append(get_call_back(v, gpu_v))
                        launch_ctx.set_arg_nparray(
                            actual_argument_slot, int(tmp.data_ptr()),
                            tmp.element_size() * tmp.nelement())
                    shape = v.shape
                    max_num_indices = _ti_core.get_max_num_indices()
                    assert len(
                        shape
                    ) <= max_num_indices, "External array cannot have > {} indices".format(
                        max_num_indices)
                    for ii, s in enumerate(shape):
                        launch_ctx.set_extra_arg_int(actual_argument_slot, ii,
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
                ti.sync()

            if has_ret:
                if id(ret_dt) in primitive_types.integer_type_ids:
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
        if not has_array and util.has_pytorch():
            import torch
            has_array = isinstance(v, torch.Tensor)
        return has_array and needs_array

    def ensure_compiled(self, *args):
        instance_id, arg_features = self.mapper.lookup(args)
        key = (self.func, instance_id)
        self.materialize(key=key, args=args, arg_features=arg_features)
        return key

    # For small kernels (< 3us), the performance can be pretty sensitive to overhead in __call__
    # Thus this part needs to be fast. (i.e. < 3us on a 4 GHz x64 CPU)
    @_shell_pop_print
    def __call__(self, *args, **kwargs):
        _taichi_skip_traceback = 1
        assert len(kwargs) == 0, 'kwargs not supported for Taichi kernels'
        key = self.ensure_compiled(*args)
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
    frames = oinspect.stack()
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
    _taichi_skip_traceback = 1

    if verbose:
        print(f'kernel={func.__name__} is_classkernel={is_classkernel}')
    primal = Kernel(func, is_grad=False, classkernel=is_classkernel)
    adjoint = Kernel(func, is_grad=True, classkernel=is_classkernel)
    # Having |primal| contains |grad| makes the tape work.
    primal.grad = adjoint

    if is_classkernel:
        # For class kernels, their primal/adjoint callables are constructed
        # when the kernel is accessed via the instance inside
        # _BoundedDifferentiableMethod.
        # This is because we need to bind the kernel or |grad| to the instance
        # owning the kernel, which is not known until the kernel is accessed.
        #
        # See also: _BoundedDifferentiableMethod, data_oriented.
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            _taichi_skip_traceback = 1
            # If we reach here (we should never), it means the class is not decorated
            # with @ti.data_oriented, otherwise getattr would have intercepted the call.
            clsobj = type(args[0])
            assert not hasattr(clsobj, '_data_oriented')
            raise KernelDefError(
                f'Please decorate class {clsobj.__name__} with @ti.data_oriented'
            )
    else:

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            _taichi_skip_traceback = 1
            return primal(*args, **kwargs)

        wrapped.grad = adjoint

    wrapped._is_wrapped_kernel = True
    wrapped._is_classkernel = is_classkernel
    wrapped._primal = primal
    wrapped._adjoint = adjoint
    return wrapped


def kernel(func):
    _taichi_skip_traceback = 1
    return _kernel_impl(func, level_of_class_stackframe=3)


classfunc = obsolete('@ti.classfunc', '@ti.func directly')
classkernel = obsolete('@ti.classkernel', '@ti.kernel directly')


class _BoundedDifferentiableMethod:
    def __init__(self, kernel_owner, wrapped_kernel_func):
        clsobj = type(kernel_owner)
        if not getattr(clsobj, '_data_oriented', False):
            raise KernelDefError(
                f'Please decorate class {clsobj.__name__} with @ti.data_oriented'
            )
        self._kernel_owner = kernel_owner
        self._primal = wrapped_kernel_func._primal
        self._adjoint = wrapped_kernel_func._adjoint

    def __call__(self, *args, **kwargs):
        _taichi_skip_traceback = 1
        return self._primal(self._kernel_owner, *args, **kwargs)

    def grad(self, *args, **kwargs):
        _taichi_skip_traceback = 1
        return self._adjoint(self._kernel_owner, *args, **kwargs)


def data_oriented(cls):
    def getattr(self, item):
        _taichi_skip_traceback = 1
        x = super(cls, self).__getattribute__(item)
        if hasattr(x, '_is_wrapped_kernel'):
            if inspect.ismethod(x):
                wrapped = x.__func__
            else:
                wrapped = x
            assert inspect.isfunction(wrapped)
            if wrapped._is_classkernel:
                return _BoundedDifferentiableMethod(self, wrapped)
        return x

    cls.__getattribute__ = getattr
    cls._data_oriented = True

    return cls
