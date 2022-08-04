import ast
import functools
import inspect
import re
import sys
import textwrap

import numpy as np
import taichi.lang
from taichi._lib import core as _ti_core
from taichi.lang import impl, ops, runtime_ops
from taichi.lang.ast import (ASTTransformerContext, KernelSimplicityASTChecker,
                             transform_tree)
from taichi.lang.ast.ast_transformer_utils import ReturnStatus
from taichi.lang.enums import AutodiffMode, Layout
from taichi.lang.exception import (TaichiCompilationError, TaichiRuntimeError,
                                   TaichiRuntimeTypeError, TaichiSyntaxError,
                                   handle_exception_from_cpp)
from taichi.lang.expr import Expr
from taichi.lang.kernel_arguments import KernelArgument
from taichi.lang.matrix import Matrix, MatrixType
from taichi.lang.shell import _shell_pop_print, oinspect
from taichi.lang.util import has_paddle, has_pytorch, to_taichi_type
from taichi.types import (ndarray_type, primitive_types, sparse_matrix_builder,
                          template, texture_type)

from taichi import _logging

if has_pytorch():
    import torch

if has_paddle():
    import paddle


def func(fn, is_real_function=False):
    """Marks a function as callable in Taichi-scope.

    This decorator transforms a Python function into a Taichi one. Taichi
    will JIT compile it into native instructions.

    Args:
        fn (Callable): The Python function to be decorated
        is_real_function (bool): Whether the function is a real function

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

    fun = Func(fn, _classfunc=is_classfunc, is_real_function=is_real_function)

    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        return fun.__call__(*args, **kwargs)

    decorated._is_taichi_function = True
    decorated._is_real_function = is_real_function
    return decorated


def real_func(fn):
    return func(fn, is_real_function=True)


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
    def decorated(*args, **kwargs):
        return fun.__call__(*args, **kwargs)

    decorated._is_taichi_function = True
    return decorated


def _get_tree_and_ctx(self,
                      excluded_parameters=(),
                      is_kernel=True,
                      arg_features=None,
                      args=None,
                      ast_builder=None,
                      is_real_function=False):
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
            global_vars[anno.id] = self.arguments[i].annotation

    if isinstance(func_body.returns, ast.Name):
        global_vars[func_body.returns.id] = self.return_type

    if is_kernel or is_real_function:
        # inject template parameters into globals
        for i in self.template_slot_locations:
            template_var_name = self.arguments[i].name
            global_vars[template_var_name] = args[i]

    return tree, ASTTransformerContext(excluded_parameters=excluded_parameters,
                                       is_kernel=is_kernel,
                                       func=self,
                                       arg_features=arg_features,
                                       global_vars=global_vars,
                                       argument_data=args,
                                       src=src,
                                       start_lineno=start_lineno,
                                       file=file,
                                       ast_builder=ast_builder,
                                       is_real_function=is_real_function)


def _process_args(self, args, kwargs):
    ret = [argument.default for argument in self.arguments]
    len_args = len(args)

    if len_args > len(ret):
        raise TaichiSyntaxError("Too many arguments.")

    for i, arg in enumerate(args):
        ret[i] = arg

    for key, value in kwargs.items():
        found = False
        for i, arg in enumerate(self.arguments):
            if key == arg.name:
                if i < len_args:
                    raise TaichiSyntaxError(
                        f"Multiple values for argument '{key}'.")
                ret[i] = value
                found = True
                break
        if not found:
            raise TaichiSyntaxError(f"Unexpected argument '{key}'.")

    for i, arg in enumerate(ret):
        if arg is inspect.Parameter.empty:
            raise TaichiSyntaxError(
                f"Parameter '{self.arguments[i].name}' missing.")

    return ret


class Func:
    function_counter = 0

    def __init__(self,
                 _func,
                 _classfunc=False,
                 _pyfunc=False,
                 is_real_function=False):
        self.func = _func
        self.func_id = Func.function_counter
        Func.function_counter += 1
        self.compiled = None
        self.classfunc = _classfunc
        self.pyfunc = _pyfunc
        self.is_real_function = is_real_function
        self.arguments = []
        self.return_type = None
        self.extract_arguments()
        self.template_slot_locations = []
        for i, arg in enumerate(self.arguments):
            if isinstance(arg.annotation, template):
                self.template_slot_locations.append(i)
        self.mapper = TaichiCallableTemplateMapper(
            self.arguments, self.template_slot_locations)
        self.taichi_functions = {}  # The |Function| class in C++

    def __call__(self, *args, **kwargs):
        args = _process_args(self, args, kwargs)

        if not impl.inside_kernel():
            if not self.pyfunc:
                raise TaichiSyntaxError(
                    "Taichi functions cannot be called from Python-scope.")
            return self.func(*args)

        if self.is_real_function:
            if impl.get_runtime(
            ).current_kernel.autodiff_mode != AutodiffMode.NONE:
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
        tree, ctx = _get_tree_and_ctx(
            self,
            is_kernel=False,
            args=args,
            ast_builder=impl.get_runtime().prog.current_ast_builder(),
            is_real_function=self.is_real_function)
        ret = transform_tree(tree, ctx)
        if not self.is_real_function:
            if self.return_type and ctx.returned != ReturnStatus.ReturnedValue:
                raise TaichiSyntaxError(
                    "Function has a return type but does not have a return statement"
                )
        return ret

    def func_call_rvalue(self, key, args):
        # Skip the template args, e.g., |self|
        assert self.is_real_function
        non_template_args = []
        for i, kernel_arg in enumerate(self.arguments):
            anno = kernel_arg.annotation
            if not isinstance(anno, template):
                if id(anno) in primitive_types.type_ids:
                    non_template_args.append(ops.cast(args[i], anno))
                elif isinstance(anno, primitive_types.RefType):
                    non_template_args.append(
                        _ti_core.make_reference(args[i].ptr))
                else:
                    non_template_args.append(args[i])
        non_template_args = impl.make_expr_group(non_template_args)
        return Expr(
            _ti_core.make_func_call_expr(
                self.taichi_functions[key.instance_id], non_template_args))

    def do_compile(self, key, args):
        tree, ctx = _get_tree_and_ctx(self,
                                      is_kernel=False,
                                      args=args,
                                      is_real_function=self.is_real_function)
        fn = impl.get_runtime().prog.create_function(key)

        def func_body():
            ctx.ast_builder = fn.ast_builder()
            transform_tree(tree, ctx)

        self.taichi_functions[key.instance_id] = fn
        self.compiled[key.instance_id] = func_body
        self.taichi_functions[key.instance_id].set_function_body(func_body)

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect.Signature.empty, None):
            self.return_type = sig.return_annotation
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise TaichiSyntaxError(
                    'Taichi functions do not support variable keyword parameters (i.e., **kwargs)'
                )
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TaichiSyntaxError(
                    'Taichi functions do not support variable positional parameters (i.e., *args)'
                )
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise TaichiSyntaxError(
                    'Taichi functions do not support keyword parameters')
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TaichiSyntaxError(
                    'Taichi functions only support "positional or keyword" parameters'
                )
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                if i == 0 and self.classfunc:
                    annotation = template()
                # TODO: pyfunc also need type annotation check when real function is enabled,
                #       but that has to happen at runtime when we know which scope it's called from.
                elif not self.pyfunc and self.is_real_function:
                    raise TaichiSyntaxError(
                        f'Taichi function `{self.func.__name__}` parameter `{arg_name}` must be type annotated'
                    )
            else:
                if not id(annotation
                          ) in primitive_types.type_ids and not isinstance(
                              annotation, template) and not isinstance(
                                  annotation, primitive_types.RefType):
                    raise TaichiSyntaxError(
                        f'Invalid type annotation (argument {i}) of Taichi function: {annotation}'
                    )
            self.arguments.append(
                KernelArgument(annotation, param.name, param.default))


class TaichiCallableTemplateMapper:
    def __init__(self, arguments, template_slot_locations):
        self.arguments = arguments
        self.num_args = len(arguments)
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
            if isinstance(arg, taichi.lang._ndarray.Ndarray):
                raise TaichiRuntimeTypeError(
                    'Ndarray shouldn\'t be passed in via `ti.template()`, please annotate your kernel using `ti.types.ndarray(...)` instead'
                )
            return arg
        if isinstance(anno, texture_type.TextureType):
            return '#'
        if isinstance(anno, ndarray_type.NdarrayType):
            if isinstance(arg, taichi.lang._ndarray.ScalarNdarray):
                anno.check_matched(arg.get_type())
                return arg.dtype, len(arg.shape), (), Layout.AOS
            if isinstance(arg, taichi.lang.matrix.VectorNdarray):
                anno.check_matched(arg.get_type())
                return arg.dtype, len(arg.shape) + 1, (arg.n, ), arg.layout
            if isinstance(arg, taichi.lang.matrix.MatrixNdarray):
                anno.check_matched(arg.get_type())
                return arg.dtype, len(arg.shape) + 2, (arg.n,
                                                       arg.m), arg.layout
            # external arrays
            element_dim = 0 if anno.element_dim is None else anno.element_dim
            layout = Layout.AOS if anno.layout is None else anno.layout
            shape = tuple(arg.shape)
            if len(shape) < element_dim:
                raise ValueError(
                    f"Invalid argument into ti.types.ndarray() - required element_dim={element_dim}, "
                    f"but the argument has only {len(shape)} dimensions")
            element_shape = (
            ) if element_dim == 0 else shape[:
                                             element_dim] if layout == Layout.SOA else shape[
                                                 -element_dim:]
            return to_taichi_type(arg.dtype), len(shape), element_shape, layout
        if isinstance(anno, sparse_matrix_builder):
            return arg.dtype
        # Use '#' as a placeholder because other kinds of arguments are not involved in template instantiation
        return '#'

    def extract(self, args):
        extracted = []
        for arg, kernel_arg in zip(args, self.arguments):
            extracted.append(self.extract_arg(arg, kernel_arg.annotation))
        return tuple(extracted)

    def lookup(self, args):
        if len(args) != self.num_args:
            raise TypeError(
                f'{self.num_args} argument(s) needed but {len(args)} provided.'
            )

        key = self.extract(args)
        if key not in self.mapping:
            count = len(self.mapping)
            self.mapping[key] = count
        return self.mapping[key], key


def _get_global_vars(_func):
    # Discussions: https://github.com/taichi-dev/taichi/issues/282
    global_vars = _func.__globals__.copy()

    freevar_names = _func.__code__.co_freevars
    closure = _func.__closure__
    if closure:
        freevar_values = list(map(lambda x: x.cell_contents, closure))
        for name, value in zip(freevar_names, freevar_values):
            global_vars[name] = value

    return global_vars


class Kernel:
    counter = 0

    def __init__(self, _func, autodiff_mode, _classkernel=False):
        self.func = _func
        self.kernel_counter = Kernel.counter
        Kernel.counter += 1
        assert autodiff_mode in (AutodiffMode.NONE, AutodiffMode.FORWARD,
                                 AutodiffMode.REVERSE)
        self.autodiff_mode = autodiff_mode
        self.grad = None
        self.arguments = []
        self.return_type = None
        self.classkernel = _classkernel
        self.extract_arguments()
        self.template_slot_locations = []
        for i, arg in enumerate(self.arguments):
            if isinstance(arg.annotation, template):
                self.template_slot_locations.append(i)
        self.mapper = TaichiCallableTemplateMapper(
            self.arguments, self.template_slot_locations)
        impl.get_runtime().kernels.append(self)
        self.reset()
        self.kernel_cpp = None
        # TODO[#5114]: get rid of compiled_functions and use compiled_kernels instead.
        # Main motivation is that compiled_kernels can be potentially serialized in the AOT scenario.
        self.compiled_kernels = {}

    def reset(self):
        self.runtime = impl.get_runtime()

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect._empty, None):
            self.return_type = sig.return_annotation
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise TaichiSyntaxError(
                    'Taichi kernels do not support variable keyword parameters (i.e., **kwargs)'
                )
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TaichiSyntaxError(
                    'Taichi kernels do not support variable positional parameters (i.e., *args)'
                )
            if param.default is not inspect.Parameter.empty:
                raise TaichiSyntaxError(
                    'Taichi kernels do not support default values for arguments'
                )
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise TaichiSyntaxError(
                    'Taichi kernels do not support keyword parameters')
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TaichiSyntaxError(
                    'Taichi kernels only support "positional or keyword" parameters'
                )
            annotation = param.annotation
            if param.annotation is inspect.Parameter.empty:
                if i == 0 and self.classkernel:  # The |self| parameter
                    annotation = template()
                else:
                    raise TaichiSyntaxError(
                        'Taichi kernels parameters must be type annotated')
            else:
                if isinstance(
                        annotation,
                    (template, ndarray_type.NdarrayType,
                     texture_type.TextureType, texture_type.RWTextureType)):
                    pass
                elif id(annotation) in primitive_types.type_ids:
                    pass
                elif isinstance(annotation, sparse_matrix_builder):
                    pass
                elif isinstance(annotation, MatrixType):
                    pass
                else:
                    raise TaichiSyntaxError(
                        f'Invalid type annotation (argument {i}) of Taichi kernel: {annotation}'
                    )
            self.arguments.append(
                KernelArgument(annotation, param.name, param.default))

    def materialize(self, key=None, args=None, arg_features=None):
        if key is None:
            key = (self.func, 0, self.autodiff_mode)
        self.runtime.materialize()

        if key in self.runtime.compiled_functions:
            return

        grad_suffix = ""
        if self.autodiff_mode == AutodiffMode.FORWARD:
            grad_suffix = "_forward_grad"
        elif self.autodiff_mode == AutodiffMode.REVERSE:
            grad_suffix = "_reverse_grad"
        kernel_name = f"{self.func.__name__}_c{self.kernel_counter}_{key[1]}{grad_suffix}"
        _logging.trace(f"Compiling kernel {kernel_name}...")

        tree, ctx = _get_tree_and_ctx(
            self,
            args=args,
            excluded_parameters=self.template_slot_locations,
            arg_features=arg_features)

        if self.autodiff_mode != AutodiffMode.NONE:
            KernelSimplicityASTChecker(self.func).visit(tree)

        if impl.current_cfg().use_mesh:
            taichi.lang.Mesh.update_relation(tree, ctx)

        # Do not change the name of 'taichi_ast_generator'
        # The warning system needs this identifier to remove unnecessary messages
        def taichi_ast_generator(kernel_cxx):
            if self.runtime.inside_kernel:
                raise TaichiSyntaxError(
                    "Kernels cannot call other kernels. I.e., nested kernels are not allowed. "
                    "Please check if you have direct/indirect invocation of kernels within kernels. "
                    "Note that some methods provided by the Taichi standard library may invoke kernels, "
                    "and please move their invocations to Python-scope.")
            self.runtime.inside_kernel = True
            self.runtime.current_kernel = self
            try:
                ctx.ast_builder = kernel_cxx.ast_builder()
                transform_tree(tree, ctx)
                if not ctx.is_real_function:
                    if self.return_type and ctx.returned != ReturnStatus.ReturnedValue:
                        raise TaichiSyntaxError(
                            "Kernel has a return type but does not have a return statement"
                        )
            finally:
                self.runtime.inside_kernel = False
                self.runtime.current_kernel = None

        taichi_kernel = impl.get_runtime().prog.create_kernel(
            taichi_ast_generator, kernel_name, self.autodiff_mode)

        self.kernel_cpp = taichi_kernel

        assert key not in self.runtime.compiled_functions
        self.runtime.compiled_functions[key] = self.get_function_body(
            taichi_kernel)
        self.compiled_kernels[key] = taichi_kernel

    def get_torch_callbacks(self, v, has_torch, is_ndarray=True):
        callbacks = []

        def get_call_back(u, v):
            def call_back():
                u.copy_(v)

            return call_back

        assert has_torch
        assert isinstance(v, torch.Tensor)
        if not v.is_contiguous():
            raise ValueError(
                "Non contiguous tensors are not supported, please call tensor.contiguous() before passing it into taichi kernel."
            )
        tmp = v
        taichi_arch = self.runtime.prog.config.arch
        # Ndarray means its memory is allocated on the specified taichi arch.
        # Since torch only supports CPU & CUDA, torch-base ndarray only supports
        # taichi cpu/cuda backend as well.
        # Note I put x64/arm64/cuda here to be more specific.
        assert not is_ndarray or taichi_arch in (
            _ti_core.Arch.cuda, _ti_core.Arch.x64, _ti_core.Arch.arm64
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
        return tmp, callbacks

    def get_paddle_callbacks(self, v, has_pp):
        callbacks = []

        def get_call_back(u, v):
            def call_back():
                u.copy_(v, False)

            return call_back

        assert has_pp
        assert isinstance(v, paddle.Tensor)

        tmp = v.value().get_tensor()
        taichi_arch = self.runtime.prog.config.arch

        if v.place.is_gpu_place():
            # External tensor on cuda
            if taichi_arch != _ti_core.Arch.cuda:
                # copy data back to cpu
                host_v = v.cpu()
                tmp = host_v.value().get_tensor()
                callbacks.append(get_call_back(v, host_v))
        elif v.place.is_cpu_place():
            # External tensor on cpu
            if taichi_arch == _ti_core.Arch.cuda:
                gpu_v = v.cuda()
                tmp = gpu_v.value().get_tensor()
                callbacks.append(get_call_back(v, gpu_v))
        else:
            # Paddle do support many other backends like XPU, NPU, MLU, IPU.
            raise TaichiRuntimeError(
                f"Taichi do not support backend {v.place} that Paddle support."
            )

        return tmp, callbacks

    def get_function_body(self, t_kernel):
        # The actual function body
        def func__(*args):
            assert len(args) == len(
                self.arguments
            ), f'{len(self.arguments)} arguments needed but {len(args)} provided'

            tmps = []
            callbacks = []
            has_external_arrays = False
            has_torch = has_pytorch()
            has_pp = has_paddle()

            actual_argument_slot = 0
            launch_ctx = t_kernel.make_launch_context()
            for i, v in enumerate(args):
                needed = self.arguments[i].annotation
                if isinstance(needed, template):
                    continue
                provided = type(v)
                # Note: do not use sth like "needed == f32". That would be slow.
                if id(needed) in primitive_types.real_type_ids:
                    if not isinstance(v, (float, int)):
                        raise TaichiRuntimeTypeError.get(
                            i, needed.to_string(), provided)
                    launch_ctx.set_arg_float(actual_argument_slot, float(v))
                elif id(needed) in primitive_types.integer_type_ids:
                    if not isinstance(v, int):
                        raise TaichiRuntimeTypeError.get(
                            i, needed.to_string(), provided)
                    launch_ctx.set_arg_int(actual_argument_slot, int(v))
                elif isinstance(needed, sparse_matrix_builder):
                    # Pass only the base pointer of the ti.types.sparse_matrix_builder() argument
                    launch_ctx.set_arg_int(actual_argument_slot, v._get_addr())
                elif isinstance(needed,
                                ndarray_type.NdarrayType) and isinstance(
                                    v, taichi.lang._ndarray.Ndarray):
                    has_external_arrays = True
                    v = v.arr
                    launch_ctx.set_arg_ndarray(actual_argument_slot, v)
                elif isinstance(needed,
                                texture_type.TextureType) and isinstance(
                                    v, taichi.lang._texture.Texture):
                    has_external_arrays = True
                    v = v.tex
                    launch_ctx.set_arg_texture(actual_argument_slot, v)
                elif isinstance(needed,
                                texture_type.RWTextureType) and isinstance(
                                    v, taichi.lang._texture.Texture):
                    has_external_arrays = True
                    v = v.tex
                    launch_ctx.set_arg_rw_texture(actual_argument_slot, v)
                elif isinstance(
                        needed,
                        ndarray_type.NdarrayType) and (self.match_ext_arr(v)):
                    has_external_arrays = True
                    is_numpy = isinstance(v, np.ndarray)
                    is_torch = isinstance(v,
                                          torch.Tensor) if has_torch else False

                    # Element shapes are already spcialized in Taichi codegen.
                    # The shape information for element dims are no longer needed.
                    # Therefore we strip the element shapes from the shape vector,
                    # so that it only holds "real" array shapes.
                    is_soa = needed.layout == Layout.SOA
                    array_shape = v.shape
                    element_dim = needed.element_dim
                    if element_dim:
                        array_shape = v.shape[
                            element_dim:] if is_soa else v.shape[:-element_dim]
                    if is_numpy:
                        tmp = np.ascontiguousarray(v)
                        # Purpose: DO NOT GC |tmp|!
                        tmps.append(tmp)
                        launch_ctx.set_arg_external_array_with_shape(
                            actual_argument_slot, int(tmp.ctypes.data),
                            tmp.nbytes, array_shape)
                    elif is_torch:
                        is_ndarray = False
                        tmp, torch_callbacks = self.get_torch_callbacks(
                            v, has_torch, is_ndarray)
                        callbacks += torch_callbacks
                        launch_ctx.set_arg_external_array_with_shape(
                            actual_argument_slot, int(tmp.data_ptr()),
                            tmp.element_size() * tmp.nelement(), array_shape)
                    else:
                        # For now, paddle.fluid.core.Tensor._ptr() is only available on develop branch
                        tmp, paddle_callbacks = self.get_paddle_callbacks(
                            v, has_pp)
                        callbacks += paddle_callbacks
                        launch_ctx.set_arg_external_array_with_shape(
                            actual_argument_slot, int(tmp._ptr()),
                            v.element_size() * v.size, array_shape)

                elif isinstance(needed, MatrixType):
                    if id(needed.dtype) in primitive_types.real_type_ids:
                        for a in range(needed.n):
                            for b in range(needed.m):
                                if not isinstance(v[a, b], (int, float)):
                                    raise TaichiRuntimeTypeError.get(
                                        i, needed.dtype.to_string(),
                                        type(v[a, b]))
                                launch_ctx.set_arg_float(
                                    actual_argument_slot, float(v[a, b]))
                                actual_argument_slot += 1
                    elif id(needed.dtype) in primitive_types.integer_type_ids:
                        for a in range(needed.n):
                            for b in range(needed.m):
                                if not isinstance(v[a, b], int):
                                    raise TaichiRuntimeTypeError.get(
                                        i, needed.dtype.to_string(),
                                        type(v[a, b]))
                                launch_ctx.set_arg_int(actual_argument_slot,
                                                       int(v[a, b]))
                                actual_argument_slot += 1
                    else:
                        raise ValueError(
                            f'Matrix dtype {needed.dtype} is not integer type or real type.'
                        )
                    continue
                else:
                    raise ValueError(
                        f'Argument type mismatch. Expecting {needed}, got {type(v)}.'
                    )
                actual_argument_slot += 1
            # Both the class kernels and the plain-function kernels are unified now.
            # In both cases, |self.grad| is another Kernel instance that computes the
            # gradient. For class kernels, args[0] is always the kernel owner.
            if self.autodiff_mode == AutodiffMode.NONE and self.runtime.target_tape and not self.runtime.grad_replaced:
                self.runtime.target_tape.insert(self, args)

            if actual_argument_slot > 8 and (
                    impl.current_cfg().arch == _ti_core.opengl
                    or impl.current_cfg().arch == _ti_core.cc):
                raise TaichiRuntimeError(
                    f"The number of elements in kernel arguments is too big! Do not exceed 8 on {_ti_core.arch_name(impl.current_cfg().arch)} backend."
                )

            if actual_argument_slot > 64 and (
                (impl.current_cfg().arch != _ti_core.opengl
                 and impl.current_cfg().arch != _ti_core.cc)):
                raise TaichiRuntimeError(
                    f"The number of elements in kernel arguments is too big! Do not exceed 64 on {_ti_core.arch_name(impl.current_cfg().arch)} backend."
                )

            try:
                t_kernel(launch_ctx)
            except Exception as e:
                e = handle_exception_from_cpp(e)
                raise e from None

            ret = None
            ret_dt = self.return_type
            has_ret = ret_dt is not None

            if has_ret:
                runtime_ops.sync()

            if has_ret:
                if id(ret_dt) in primitive_types.integer_type_ids:
                    ret = t_kernel.get_ret_int(0)
                elif id(ret_dt) in primitive_types.real_type_ids:
                    ret = t_kernel.get_ret_float(0)
                elif id(ret_dt.dtype) in primitive_types.integer_type_ids:
                    it = iter(t_kernel.get_ret_int_tensor(0))
                    ret = Matrix([[next(it) for _ in range(ret_dt.m)]
                                  for _ in range(ret_dt.n)])
                else:
                    it = iter(t_kernel.get_ret_float_tensor(0))
                    ret = Matrix([[next(it) for _ in range(ret_dt.m)]
                                  for _ in range(ret_dt.n)])
            if callbacks:
                for c in callbacks:
                    c()

            return ret

        return func__

    @staticmethod
    def match_ext_arr(v):
        has_array = isinstance(v, np.ndarray)
        if not has_array and has_pytorch():
            has_array = isinstance(v, torch.Tensor)
        if not has_array and has_paddle():
            has_array = isinstance(v, paddle.Tensor)
        return has_array

    def ensure_compiled(self, *args):
        instance_id, arg_features = self.mapper.lookup(args)
        key = (self.func, instance_id, self.autodiff_mode)
        self.materialize(key=key, args=args, arg_features=arg_features)
        return key

    # For small kernels (< 3us), the performance can be pretty sensitive to overhead in __call__
    # Thus this part needs to be fast. (i.e. < 3us on a 4 GHz x64 CPU)
    @_shell_pop_print
    def __call__(self, *args, **kwargs):
        args = _process_args(self, args, kwargs)

        # Transform the primal kernel to forward mode grad kernel
        # then recover to primal when exiting the forward mode manager
        if self.runtime.fwd_mode_manager:
            # TODO: if we would like to compute 2nd-order derivatives by forward-on-reverse in a nested context manager fashion,
            # i.e., a `Tape` nested in the `FwdMode`, we can transform the kernels with `mode_original == AutodiffMode.REVERSE` only,
            # to avoid duplicate computation for 1st-order derivatives
            mode_original = self.autodiff_mode
            self.autodiff_mode = AutodiffMode.FORWARD
            self.runtime.fwd_mode_manager.insert(self, mode_original)

        if self.autodiff_mode != AutodiffMode.NONE and impl.current_cfg(
        ).opt_level == 0:
            _logging.warn(
                """opt_level = 1 is enforced to enable gradient computation."""
            )
            impl.current_cfg().opt_level = 1
        key = self.ensure_compiled(*args)
        return self.runtime.compiled_functions[key](*args)


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

    if verbose:
        print(f'kernel={_func.__name__} is_classkernel={is_classkernel}')
    primal = Kernel(_func,
                    autodiff_mode=AutodiffMode.NONE,
                    _classkernel=is_classkernel)
    adjoint = Kernel(_func,
                     autodiff_mode=AutodiffMode.REVERSE,
                     _classkernel=is_classkernel)
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
            # If we reach here (we should never), it means the class is not decorated
            # with @ti.data_oriented, otherwise getattr would have intercepted the call.
            clsobj = type(args[0])
            assert not hasattr(clsobj, '_data_oriented')
            raise TaichiSyntaxError(
                f'Please decorate class {clsobj.__name__} with @ti.data_oriented'
            )
    else:

        @functools.wraps(_func)
        def wrapped(*args, **kwargs):
            try:
                return primal(*args, **kwargs)
            except (TaichiCompilationError, TaichiRuntimeError) as e:
                raise type(e)('\n' + str(e)) from None

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

    See also https://docs.taichi-lang.org/docs/syntax#kernel.

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
    return _kernel_impl(fn, level_of_class_stackframe=3)


class _BoundedDifferentiableMethod:
    def __init__(self, kernel_owner, wrapped_kernel_func):
        clsobj = type(kernel_owner)
        if not getattr(clsobj, '_data_oriented', False):
            raise TaichiSyntaxError(
                f'Please decorate class {clsobj.__name__} with @ti.data_oriented'
            )
        self._kernel_owner = kernel_owner
        self._primal = wrapped_kernel_func._primal
        self._adjoint = wrapped_kernel_func._adjoint
        self._is_staticmethod = wrapped_kernel_func._is_staticmethod
        self.__name__ = None

    def __call__(self, *args, **kwargs):
        if self._is_staticmethod:
            return self._primal(*args, **kwargs)
        return self._primal(self._kernel_owner, *args, **kwargs)

    def grad(self, *args, **kwargs):
        return self._adjoint(self._kernel_owner, *args, **kwargs)


def data_oriented(cls):
    """Marks a class as Taichi compatible.

    To allow for modularized code, Taichi provides this decorator so that
    Taichi kernels can be defined inside a class.

    See also https://docs.taichi-lang.org/docs/odop

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


__all__ = ["data_oriented", "func", "kernel"]
