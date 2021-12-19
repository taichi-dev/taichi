import ast
import functools
import inspect
import re
import sys
import textwrap

import numpy as np
import taichi.lang
from taichi._lib import core as _ti_core
from taichi.lang import impl, runtime_ops, util
from taichi.lang.ast import (ASTTransformerContext, KernelSimplicityASTChecker,
                             transform_tree)
from taichi.lang.enums import Layout
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.expr import Expr
from taichi.lang.shell import _shell_pop_print, oinspect
from taichi.lang.util import to_taichi_type
from taichi.linalg.sparse_matrix import sparse_matrix_builder
from taichi.tools.util import obsolete
from taichi.types import any_arr, primitive_types, template

from taichi import _logging

if util.has_pytorch():
    import torch


def func(fn):
    """Marks a function as callable in Taichi-scope.

    This decorator transforms a Python function into a Taichi one. Taichi
    will JIT compile it into native instructions.

    Args:
        fn (Callable): The Python function to be decorated

    Returns:
        Callable: The decorated function

    Example::

        >>> @ti.func
        >>> def foo(x):
        >>>     return x + 2
        >>>
        >>> @ti.kernel
        >>> def run():
        >>>     print(foo(40))  # 42
    """
    is_classfunc = _inside_class(level_of_class_stackframe=3)

    _taichi_skip_traceback = 1
    fun = Func(fn, _classfunc=is_classfunc)

    @functools.wraps(fn)
    def decorated(*args):
        _taichi_skip_traceback = 1
        return fun.__call__(*args)

    decorated._is_taichi_function = True
    return decorated


def pyfunc(fn):
    """Marks a function as callable in both Taichi and Python scopes.

    When called inside the Taichi scope, Taichi will JIT compile it into
    native instructions. Otherwise it will be invoked directly as a
    Python function.

    See also :func:`~taichi.lang.kernel_impl.func`.

    Args:
        fn (Callable): The Python function to be decorated

    Returns:
        Callable: The decorated function
    """
    is_classfunc = _inside_class(level_of_class_stackframe=3)
    fun = Func(fn, _classfunc=is_classfunc, _pyfunc=True)

    @functools.wraps(fn)
    def decorated(*args):
        _taichi_skip_traceback = 1
        return fun.__call__(*args)

    decorated._is_taichi_function = True
    return decorated


def _get_tree_and_ctx(self,
                      excluded_parameters=(),
                      is_kernel=True,
                      arg_features=None,
                      args=None):
    file = oinspect.getsourcefile(self.func)
    src, start_lineno = oinspect.getsourcelines(self.func)
    src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
    tree = ast.parse(textwrap.dedent("\n".join(src)))

    func_body = tree.body[0]
    func_body.decorator_list = []

    global_vars = _get_global_vars(self.func)

    for i, arg in enumerate(func_body.args.args):
        anno = arg.annotation
        if isinstance(anno, ast.Name):
            global_vars[anno.id] = self.argument_annotations[i]

    if isinstance(func_body.returns, ast.Name):
        global_vars[func_body.returns.id] = self.return_type

    if is_kernel or impl.get_runtime().experimental_real_function:
        # inject template parameters into globals
        for i in self.template_slot_locations:
            template_var_name = self.argument_names[i]
            global_vars[template_var_name] = args[i]

    return tree, ASTTransformerContext(excluded_parameters=excluded_parameters,
                                       is_kernel=is_kernel,
                                       func=self,
                                       arg_features=arg_features,
                                       global_vars=global_vars,
                                       argument_data=args,
                                       src=src,
                                       start_lineno=start_lineno,
                                       file=file)


class Func:
    function_counter = 0

    def __init__(self, _func, _classfunc=False, _pyfunc=False):
        self.func = _func
        self.func_id = Func.function_counter
        Func.function_counter += 1
        self.compiled = None
        self.classfunc = _classfunc
        self.pyfunc = _pyfunc
        self.argument_annotations = []
        self.argument_names = []
        _taichi_skip_traceback = 1
        self.return_type = None
        self.extract_arguments()
        self.template_slot_locations = []
        for i, anno in enumerate(self.argument_annotations):
            if isinstance(anno, template):
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
            instance_id, _ = self.mapper.lookup(args)
            key = _ti_core.FunctionKey(self.func.__name__, self.func_id,
                                       instance_id)
            if self.compiled is None:
                self.compiled = {}
            if key.instance_id not in self.compiled:
                self.do_compile(key=key, args=args)
            return self.func_call_rvalue(key=key, args=args)
        tree, ctx = _get_tree_and_ctx(self, is_kernel=False, args=args)
        ret = transform_tree(tree, ctx)
        if not impl.get_runtime().experimental_real_function:
            if self.return_type and not ctx.returned:
                raise TaichiSyntaxError(
                    "Function has a return type but does not have a return statement"
                )
        return ret

    def func_call_rvalue(self, key, args):
        # Skip the template args, e.g., |self|
        assert impl.get_runtime().experimental_real_function
        non_template_args = []
        for i, anno in enumerate(self.argument_annotations):
            if not isinstance(anno, template):
                non_template_args.append(args[i])
        non_template_args = impl.make_expr_group(non_template_args)
        return Expr(
            _ti_core.make_func_call_expr(
                self.taichi_functions[key.instance_id], non_template_args))

    def do_compile(self, key, args):
        tree, ctx = _get_tree_and_ctx(self, is_kernel=False, args=args)
        self.compiled[key.instance_id] = lambda: transform_tree(tree, ctx)
        self.taichi_functions[key.instance_id] = _ti_core.create_function(key)
        self.taichi_functions[key.instance_id].set_function_body(
            self.compiled[key.instance_id])

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
                # TODO: pyfunc also need type annotation check when real function is enabled,
                #       but that has to happen at runtime when we know which scope it's called from.
                elif not self.pyfunc and impl.get_runtime(
                ).experimental_real_function:
                    raise KernelDefError(
                        f'Taichi function `{self.func.__name__}` parameter `{arg_name}` must be type annotated'
                    )
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
        self.num_args = len(annotations)
        self.template_slot_locations = template_slot_locations
        self.mapping = {}

    @staticmethod
    def extract_arg(arg, anno):
        if isinstance(anno, template):
            if isinstance(arg, taichi.lang.snode.SNode):
                return arg.ptr
            if isinstance(arg, taichi.lang.expr.Expr):
                return arg.ptr.get_underlying_ptr_address()
            if isinstance(arg, _ti_core.Expr):
                return arg.get_underlying_ptr_address()
            if isinstance(arg, tuple):
                return tuple(
                    TaichiCallableTemplateMapper.extract_arg(item, anno)
                    for item in arg)
            return arg
        if isinstance(anno, any_arr):
            if isinstance(arg, taichi.lang._ndarray.ScalarNdarray):
                anno.check_element_dim(arg, 0)
                anno.check_element_shape(())
                anno.check_field_dim(len(arg.shape))
                return arg.dtype, len(arg.shape), (), Layout.AOS
            if isinstance(arg, taichi.lang.matrix.VectorNdarray):
                anno.check_element_dim(arg, 1)
                anno.check_element_shape((arg.n, ))
                anno.check_field_dim(len(arg.shape))
                anno.check_layout(arg)
                return arg.dtype, len(arg.shape) + 1, (arg.n, ), arg.layout
            if isinstance(arg, taichi.lang.matrix.MatrixNdarray):
                anno.check_element_dim(arg, 2)
                anno.check_element_shape((arg.n, arg.m))
                anno.check_field_dim(len(arg.shape))
                anno.check_layout(arg)
                return arg.dtype, len(arg.shape) + 2, (arg.n,
                                                       arg.m), arg.layout
            # external arrays
            element_dim = 0 if anno.element_dim is None else anno.element_dim
            layout = Layout.AOS if anno.layout is None else anno.layout
            shape = tuple(arg.shape)
            if len(shape) < element_dim:
                raise ValueError(
                    f"Invalid argument into ti.any_arr() - required element_dim={element_dim}, "
                    f"but the argument has only {len(shape)} dimensions")
            element_shape = (
            ) if element_dim == 0 else shape[:
                                             element_dim] if layout == Layout.SOA else shape[
                                                 -element_dim:]
            return to_taichi_type(arg.dtype), len(shape), element_shape, layout
        return type(arg).__name__,

    def extract(self, args):
        extracted = []
        for arg, anno in zip(args, self.annotations):
            extracted.append(self.extract_arg(arg, anno))
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
    pass


class KernelArgError(Exception):
    def __init__(self, pos, needed, provided):
        message = f'Argument {pos} (type={provided}) cannot be converted into required type {needed}'
        super().__init__(message)
        self.pos = pos
        self.needed = needed
        self.provided = provided


def _get_global_vars(_func):
    closure_vars = inspect.getclosurevars(_func)
    return {
        **closure_vars.globals,
        **closure_vars.nonlocals,
        **closure_vars.builtins
    }


class Kernel:
    counter = 0

    def __init__(self, _func, is_grad, _classkernel=False):
        self.func = _func
        self.kernel_counter = Kernel.counter
        Kernel.counter += 1
        self.is_grad = is_grad
        self.grad = None
        self.argument_annotations = []
        self.argument_names = []
        self.return_type = None
        self.classkernel = _classkernel
        _taichi_skip_traceback = 1
        self.extract_arguments()
        del _taichi_skip_traceback
        self.template_slot_locations = []
        for i, anno in enumerate(self.argument_annotations):
            if isinstance(anno, template):
                self.template_slot_locations.append(i)
        self.mapper = TaichiCallableTemplateMapper(
            self.argument_annotations, self.template_slot_locations)
        impl.get_runtime().kernels.append(self)
        self.reset()
        self.kernel_cpp = None

    def reset(self):
        self.runtime = impl.get_runtime()
        if self.is_grad:
            self.compiled_functions = self.runtime.compiled_grad_functions
        else:
            self.compiled_functions = self.runtime.compiled_functions

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
                if isinstance(annotation, (template, any_arr)):
                    pass
                elif id(annotation) in primitive_types.type_ids:
                    pass
                elif isinstance(annotation, sparse_matrix_builder):
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
        kernel_name = f"{self.func.__name__}_c{self.kernel_counter}_{key[1]}{grad_suffix}"
        _logging.trace(f"Compiling kernel {kernel_name}...")

        tree, ctx = _get_tree_and_ctx(
            self,
            args=args,
            excluded_parameters=self.template_slot_locations,
            arg_features=arg_features)

        if self.is_grad:
            KernelSimplicityASTChecker(self.func).visit(tree)

        # Do not change the name of 'taichi_ast_generator'
        # The warning system needs this identifier to remove unnecessary messages
        def taichi_ast_generator():
            _taichi_skip_traceback = 1
            if self.runtime.inside_kernel:
                raise TaichiSyntaxError(
                    "Kernels cannot call other kernels. I.e., nested kernels are not allowed. "
                    "Please check if you have direct/indirect invocation of kernels within kernels. "
                    "Note that some methods provided by the Taichi standard library may invoke kernels, "
                    "and please move their invocations to Python-scope.")
            self.runtime.inside_kernel = True
            self.runtime.current_kernel = self
            try:
                transform_tree(tree, ctx)
                if not impl.get_runtime().experimental_real_function:
                    if self.return_type and not ctx.returned:
                        raise TaichiSyntaxError(
                            "Kernel has a return type but does not have a return statement"
                        )
            finally:
                self.runtime.inside_kernel = False
                self.runtime.current_kernel = None

        taichi_kernel = _ti_core.create_kernel(taichi_ast_generator,
                                               kernel_name, self.is_grad)

        self.kernel_cpp = taichi_kernel

        assert key not in self.compiled_functions
        self.compiled_functions[key] = self.get_function_body(taichi_kernel)

    def get_function_body(self, t_kernel):
        # The actual function body
        def func__(*args):
            assert len(args) == len(
                self.argument_annotations
            ), f'{len(self.argument_annotations)} arguments needed but {len(args)} provided'

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
                elif isinstance(needed, sparse_matrix_builder):
                    # Pass only the base pointer of the ti.linalg.sparse_matrix_builder() argument
                    launch_ctx.set_arg_int(actual_argument_slot, v.get_addr())
                elif isinstance(needed, any_arr) and (
                        self.match_ext_arr(v)
                        or isinstance(v, taichi.lang._ndarray.Ndarray)):
                    is_ndarray = False
                    if isinstance(v, taichi.lang._ndarray.Ndarray):
                        v = v.arr
                        is_ndarray = True
                    has_external_arrays = True
                    ndarray_use_torch = self.runtime.prog.config.ndarray_use_torch
                    has_torch = util.has_pytorch()
                    is_numpy = isinstance(v, np.ndarray)
                    if is_numpy:
                        tmp = np.ascontiguousarray(v)
                        # Purpose: DO NOT GC |tmp|!
                        tmps.append(tmp)
                        launch_ctx.set_arg_external_array(
                            actual_argument_slot, int(tmp.ctypes.data),
                            tmp.nbytes, False)
                    elif is_ndarray and not ndarray_use_torch:
                        # Use ndarray's own memory allocator
                        tmp = v
                        launch_ctx.set_arg_external_array(
                            actual_argument_slot,
                            int(tmp.device_allocation_ptr()),
                            tmp.element_size() * tmp.nelement(), True)
                    else:

                        def get_call_back(u, v):
                            def call_back():
                                u.copy_(v)

                            return call_back

                        assert has_torch
                        assert isinstance(v, torch.Tensor)
                        tmp = v
                        taichi_arch = self.runtime.prog.config.arch
                        # Ndarray means its memory is allocated on the specified taichi arch.
                        # Since torch only supports CPU & CUDA, torch-base ndarray only supports
                        # taichi cpu/cuda backend as well.
                        # Note I put x64/arm64/cuda here to be more specific.
                        assert not is_ndarray or taichi_arch in (
                            _ti_core.Arch.cuda, _ti_core.Arch.x64,
                            _ti_core.Arch.arm64
                        ), "Torch-based ndarray is only supported on taichi x64/arm64/cuda backend."

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
                        launch_ctx.set_arg_external_array(
                            actual_argument_slot, int(tmp.data_ptr()),
                            tmp.element_size() * tmp.nelement(), False)

                    shape = v.shape
                    max_num_indices = _ti_core.get_max_num_indices()
                    assert len(
                        shape
                    ) <= max_num_indices, f"External array cannot have > {max_num_indices} indices"
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
            if not self.is_grad and self.runtime.target_tape and not self.runtime.grad_replaced:
                self.runtime.target_tape.insert(self, args)

            t_kernel(launch_ctx)

            ret = None
            ret_dt = self.return_type
            has_ret = ret_dt is not None

            if has_ret or (impl.current_cfg().async_mode
                           and has_external_arrays):
                runtime_ops.sync()

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

    @staticmethod
    def match_ext_arr(v):
        has_array = isinstance(v, np.ndarray)
        if not has_array and util.has_pytorch():
            has_array = isinstance(v, torch.Tensor)
        return has_array

    def ensure_compiled(self, *args):
        instance_id, arg_features = self.mapper.lookup(args)
        key = (self.func, instance_id)
        self.materialize(key=key, args=args, arg_features=arg_features)
        return key

    # For small kernels (< 3us), the performance can be pretty sensitive to overhead in __call__
    # Thus this part needs to be fast. (i.e. < 3us on a 4 GHz x64 CPU)
    @_shell_pop_print
    def __call__(self, *args, **kwargs):
        if self.is_grad and impl.current_cfg().opt_level == 0:
            _logging.warn(
                """opt_level = 1 is enforced to enable gradient computation."""
            )
            impl.current_cfg().opt_level = 1
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
    try:
        maybe_class_frame = sys._getframe(level_of_class_stackframe)
        statement_list = inspect.getframeinfo(maybe_class_frame)[3]
        first_statment = statement_list[0].strip()
        for pat in _KERNEL_CLASS_STACKFRAME_STMT_RES:
            if pat.match(first_statment):
                return True
    except:
        pass
    return False


def _kernel_impl(_func, level_of_class_stackframe, verbose=False):
    # Can decorators determine if a function is being defined inside a class?
    # https://stackoverflow.com/a/8793684/12003165
    is_classkernel = _inside_class(level_of_class_stackframe + 1)
    _taichi_skip_traceback = 1

    if verbose:
        print(f'kernel={_func.__name__} is_classkernel={is_classkernel}')
    primal = Kernel(_func, is_grad=False, _classkernel=is_classkernel)
    adjoint = Kernel(_func, is_grad=True, _classkernel=is_classkernel)
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
        @functools.wraps(_func)
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

        @functools.wraps(_func)
        def wrapped(*args, **kwargs):
            _taichi_skip_traceback = 1
            return primal(*args, **kwargs)

        wrapped.grad = adjoint

    wrapped._is_wrapped_kernel = True
    wrapped._is_classkernel = is_classkernel
    wrapped._primal = primal
    wrapped._adjoint = adjoint
    return wrapped


def kernel(fn):
    """Marks a function as a Taichi kernel.

    A Taichi kernel is a function written in Python, and gets JIT compiled by
    Taichi into native CPU/GPU instructions (e.g. a series of CUDA kernels).
    The top-level ``for`` loops are automatically parallelized, and distributed
    to either a CPU thread pool or massively parallel GPUs.

    Kernel's gradient kernel would be generated automatically by the AutoDiff system.

    See also https://docs.taichi.graphics/lang/articles/basic/syntax#kernels.

    Args:
        fn (Callable): the Python function to be decorated

    Returns:
        Callable: The decorated function

    Example::

        >>> x = ti.field(ti.i32, shape=(4, 8))
        >>>
        >>> @ti.kernel
        >>> def run():
        >>>     # Assigns all the elements of `x` in parallel.
        >>>     for i in x:
        >>>         x[i] = i
    """
    _taichi_skip_traceback = 1
    return _kernel_impl(fn, level_of_class_stackframe=3)


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
        self._is_staticmethod = wrapped_kernel_func._is_staticmethod
        self.__name__ = None

    def __call__(self, *args, **kwargs):
        _taichi_skip_traceback = 1
        if self._is_staticmethod:
            return self._primal(*args, **kwargs)
        return self._primal(self._kernel_owner, *args, **kwargs)

    def grad(self, *args, **kwargs):
        _taichi_skip_traceback = 1
        return self._adjoint(self._kernel_owner, *args, **kwargs)


def data_oriented(cls):
    """Marks a class as Taichi compatible.

    To allow for modularized code, Taichi provides this decorator so that
    Taichi kernels can be defined inside a class.

    See also https://docs.taichi.graphics/lang/articles/advanced/odop

    Example::

        >>> @ti.data_oriented
        >>> class TiArray:
        >>>     def __init__(self, n):
        >>>         self.x = ti.field(ti.f32, shape=n)
        >>>
        >>>     @ti.kernel
        >>>     def inc(self):
        >>>         for i in self.x:
        >>>             self.x[i] += 1.0
        >>>
        >>> a = TiArray(32)
        >>> a.inc()

    Args:
        cls (Class): the class to be decorated

    Returns:
        The decorated class.
    """
    def _getattr(self, item):
        _taichi_skip_traceback = 1
        method = cls.__dict__.get(item, None)
        is_property = method.__class__ == property
        is_staticmethod = method.__class__ == staticmethod
        if is_property:
            x = method.fget
        else:
            x = super(cls, self).__getattribute__(item)
        if hasattr(x, '_is_wrapped_kernel'):
            if inspect.ismethod(x):
                wrapped = x.__func__
            else:
                wrapped = x
            wrapped._is_staticmethod = is_staticmethod
            assert inspect.isfunction(wrapped)
            if wrapped._is_classkernel:
                ret = _BoundedDifferentiableMethod(self, wrapped)
                ret.__name__ = wrapped.__name__
                if is_property:
                    return ret()
                return ret
        if is_property:
            return x(self)
        return x

    cls.__getattribute__ = _getattr
    cls._data_oriented = True

    return cls
