import ast
import functools
import inspect
import operator
import re
import sys
import textwrap
import typing
import types
import warnings
import weakref

import numpy as np
import taichi.lang
from taichi._lib import core as _ti_core
from taichi.lang import impl, ops, runtime_ops
from taichi.lang.any_array import AnyArray
from taichi.lang._wrap_inspect import getsourcefile, getsourcelines
from taichi.lang.argpack import ArgPackType, ArgPack
from taichi.lang.ast import (
    ASTTransformerContext,
    KernelSimplicityASTChecker,
    transform_tree,
)
from taichi.lang.ast.ast_transformer_utils import ReturnStatus
from taichi.lang.enums import AutodiffMode, Layout
from taichi.lang.exception import (
    TaichiCompilationError,
    TaichiRuntimeError,
    TaichiRuntimeTypeError,
    TaichiSyntaxError,
    TaichiTypeError,
    handle_exception_from_cpp,
)
from taichi.lang.expr import Expr
from taichi.lang.kernel_arguments import KernelArgument
from taichi.lang.matrix import MatrixType
from taichi.lang.shell import _shell_pop_print
from taichi.lang.struct import StructType
from taichi.lang.util import cook_dtype, has_paddle, has_pytorch, to_taichi_type
from taichi.types import (
    ndarray_type,
    primitive_types,
    sparse_matrix_builder,
    template,
    texture_type,
)
from taichi.types.compound_types import CompoundType
from taichi.types.utils import is_signed

from taichi import _logging


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
    is_classfunc = _inside_class(level_of_class_stackframe=3 + is_real_function)

    fun = Func(fn, _classfunc=is_classfunc, is_real_function=is_real_function)

    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        return fun.__call__(*args, **kwargs)

    decorated._is_taichi_function = True
    decorated._is_real_function = is_real_function
    decorated.func = fun
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
    decorated._is_real_function = False
    decorated.func = fun
    return decorated


def _get_tree_and_ctx(
    self,
    excluded_parameters=(),
    is_kernel=True,
    arg_features=None,
    args=None,
    ast_builder=None,
    is_real_function=False,
):
    file = getsourcefile(self.func)
    src, start_lineno = getsourcelines(self.func)
    src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
    tree = ast.parse(textwrap.dedent("\n".join(src)))

    func_body = tree.body[0]
    func_body.decorator_list = []

    global_vars = _get_global_vars(self.func)

    if is_kernel or is_real_function:
        # inject template parameters into globals
        for i in self.template_slot_locations:
            template_var_name = self.arguments[i].name
            global_vars[template_var_name] = args[i]

    return tree, ASTTransformerContext(
        excluded_parameters=excluded_parameters,
        is_kernel=is_kernel,
        func=self,
        arg_features=arg_features,
        global_vars=global_vars,
        argument_data=args,
        src=src,
        start_lineno=start_lineno,
        file=file,
        ast_builder=ast_builder,
        is_real_function=is_real_function,
    )


def _process_args(self, args, kwargs):
    ret = [argument.default for argument in self.arguments]
    len_args = len(args)

    if len_args > len(ret):
        arg_str = ", ".join([str(arg) for arg in args])
        expected_str = ", ".join([f"{arg.name} : {arg.annotation}" for arg in self.arguments])
        msg = f"Too many arguments. Expected ({expected_str}), got ({arg_str})."
        raise TaichiSyntaxError(msg)

    for i, arg in enumerate(args):
        ret[i] = arg

    for key, value in kwargs.items():
        found = False
        for i, arg in enumerate(self.arguments):
            if key == arg.name:
                if i < len_args:
                    raise TaichiSyntaxError(f"Multiple values for argument '{key}'.")
                ret[i] = value
                found = True
                break
        if not found:
            raise TaichiSyntaxError(f"Unexpected argument '{key}'.")

    for i, arg in enumerate(ret):
        if arg is inspect.Parameter.empty:
            if self.arguments[i].annotation is inspect._empty:
                raise TaichiSyntaxError(f"Parameter `{self.arguments[i].name}` missing.")
            else:
                raise TaichiSyntaxError(
                    f"Parameter `{self.arguments[i].name} : {self.arguments[i].annotation}` missing."
                )

    return ret


class Func:
    function_counter = 0

    def __init__(self, _func, _classfunc=False, _pyfunc=False, is_real_function=False):
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
        self.mapper = TaichiCallableTemplateMapper(self.arguments, self.template_slot_locations)
        self.taichi_functions = {}  # The |Function| class in C++
        self.has_print = False

    def __call__(self, *args, **kwargs):
        args = _process_args(self, args, kwargs)

        if not impl.inside_kernel():
            if not self.pyfunc:
                raise TaichiSyntaxError("Taichi functions cannot be called from Python-scope.")
            return self.func(*args)

        if self.is_real_function:
            if impl.get_runtime().current_kernel.autodiff_mode != AutodiffMode.NONE:
                raise TaichiSyntaxError("Real function in gradient kernels unsupported.")
            instance_id, arg_features = self.mapper.lookup(args)
            key = _ti_core.FunctionKey(self.func.__name__, self.func_id, instance_id)
            if self.compiled is None:
                self.compiled = {}
            if key.instance_id not in self.compiled:
                self.do_compile(key=key, args=args, arg_features=arg_features)
            return self.func_call_rvalue(key=key, args=args)
        tree, ctx = _get_tree_and_ctx(
            self,
            is_kernel=False,
            args=args,
            ast_builder=impl.get_runtime().current_kernel.ast_builder(),
            is_real_function=self.is_real_function,
        )
        ret = transform_tree(tree, ctx)
        if not self.is_real_function:
            if self.return_type and ctx.returned != ReturnStatus.ReturnedValue:
                raise TaichiSyntaxError("Function has a return type but does not have a return statement")
        return ret

    def func_call_rvalue(self, key, args):
        # Skip the template args, e.g., |self|
        assert self.is_real_function
        non_template_args = []
        dbg_info = _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
        for i, kernel_arg in enumerate(self.arguments):
            anno = kernel_arg.annotation
            if not isinstance(anno, template):
                if id(anno) in primitive_types.type_ids:
                    non_template_args.append(ops.cast(args[i], anno))
                elif isinstance(anno, primitive_types.RefType):
                    non_template_args.append(_ti_core.make_reference(args[i].ptr, dbg_info))
                elif isinstance(anno, ndarray_type.NdarrayType):
                    if not isinstance(args[i], AnyArray):
                        raise TaichiTypeError(
                            f"Expected ndarray in the kernel argument for argument {kernel_arg.name}, got {args[i]}"
                        )
                    non_template_args += _ti_core.get_external_tensor_real_func_args(args[i].ptr, dbg_info)
                else:
                    non_template_args.append(args[i])
        non_template_args = impl.make_expr_group(non_template_args)
        func_call = (
            impl.get_runtime()
            .compiling_callable.ast_builder()
            .insert_func_call(self.taichi_functions[key.instance_id], non_template_args, dbg_info)
        )
        if self.return_type is None:
            return None
        func_call = Expr(func_call)
        ret = []

        for i, return_type in enumerate(self.return_type):
            if id(return_type) in primitive_types.type_ids:
                ret.append(
                    Expr(
                        _ti_core.make_get_element_expr(
                            func_call.ptr, (i,), _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
                        )
                    )
                )
            elif isinstance(return_type, (StructType, MatrixType)):
                ret.append(return_type.from_taichi_object(func_call, (i,)))
            else:
                raise TaichiTypeError(f"Unsupported return type for return value {i}: {return_type}")
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)

    def do_compile(self, key, args, arg_features):
        tree, ctx = _get_tree_and_ctx(
            self, is_kernel=False, args=args, arg_features=arg_features, is_real_function=self.is_real_function
        )
        fn = impl.get_runtime().prog.create_function(key)

        def func_body():
            old_callable = impl.get_runtime().compiling_callable
            impl.get_runtime().compiling_callable = fn
            ctx.ast_builder = fn.ast_builder()
            transform_tree(tree, ctx)
            impl.get_runtime().compiling_callable = old_callable

        self.taichi_functions[key.instance_id] = fn
        self.compiled[key.instance_id] = func_body
        self.taichi_functions[key.instance_id].set_function_body(func_body)

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect.Signature.empty, None):
            self.return_type = sig.return_annotation
            if sys.version_info >= (3, 9):
                if (
                    isinstance(self.return_type, (types.GenericAlias, typing._GenericAlias))
                    and self.return_type.__origin__ is tuple
                ):
                    self.return_type = self.return_type.__args__
            else:
                if isinstance(self.return_type, typing._GenericAlias) and self.return_type.__origin__ is tuple:
                    self.return_type = self.return_type.__args__
            if not isinstance(self.return_type, (list, tuple)):
                self.return_type = (self.return_type,)
            for i, return_type in enumerate(self.return_type):
                if return_type is Ellipsis:
                    raise TaichiSyntaxError("Ellipsis is not supported in return type annotations")
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise TaichiSyntaxError("Taichi functions do not support variable keyword parameters (i.e., **kwargs)")
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TaichiSyntaxError("Taichi functions do not support variable positional parameters (i.e., *args)")
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise TaichiSyntaxError("Taichi functions do not support keyword parameters")
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TaichiSyntaxError('Taichi functions only support "positional or keyword" parameters')
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                if i == 0 and self.classfunc:
                    annotation = template()
                # TODO: pyfunc also need type annotation check when real function is enabled,
                #       but that has to happen at runtime when we know which scope it's called from.
                elif not self.pyfunc and self.is_real_function:
                    raise TaichiSyntaxError(
                        f"Taichi function `{self.func.__name__}` parameter `{arg_name}` must be type annotated"
                    )
            else:
                if isinstance(annotation, ndarray_type.NdarrayType):
                    pass
                elif isinstance(annotation, MatrixType):
                    pass
                elif isinstance(annotation, StructType):
                    pass
                elif id(annotation) in primitive_types.type_ids:
                    pass
                elif isinstance(annotation, template):
                    pass
                elif isinstance(annotation, primitive_types.RefType):
                    pass
                else:
                    raise TaichiSyntaxError(f"Invalid type annotation (argument {i}) of Taichi function: {annotation}")
            self.arguments.append(KernelArgument(annotation, param.name, param.default))


class TaichiCallableTemplateMapper:
    def __init__(self, arguments, template_slot_locations):
        self.arguments = arguments
        self.num_args = len(arguments)
        self.template_slot_locations = template_slot_locations
        self.mapping = {}

    @staticmethod
    def extract_arg(arg, anno, arg_name):
        if isinstance(anno, template):
            if isinstance(arg, taichi.lang.snode.SNode):
                return arg.ptr
            if isinstance(arg, taichi.lang.expr.Expr):
                return arg.ptr.get_underlying_ptr_address()
            if isinstance(arg, _ti_core.Expr):
                return arg.get_underlying_ptr_address()
            if isinstance(arg, tuple):
                return tuple(TaichiCallableTemplateMapper.extract_arg(item, anno, arg_name) for item in arg)
            if isinstance(arg, taichi.lang._ndarray.Ndarray):
                raise TaichiRuntimeTypeError(
                    "Ndarray shouldn't be passed in via `ti.template()`, please annotate your kernel using `ti.types.ndarray(...)` instead"
                )

            if isinstance(arg, (list, tuple, dict, set)) or hasattr(arg, "_data_oriented"):
                # [Composite arguments] Return weak reference to the object
                # Taichi kernel will cache the extracted arguments, thus we can't simply return the original argument.
                # Instead, a weak reference to the original value is returned to avoid memory leak.

                # TODO(zhanlue): replacing "tuple(args)" with "hash of argument values"
                # This can resolve the following issues:
                # 1. Invalid weak-ref will leave a dead(dangling) entry in both caches: "self.mapping" and "self.compiled_functions"
                # 2. Different argument instances with same type and same value, will get templatized into seperate kernels.
                return weakref.ref(arg)

            # [Primitive arguments] Return the value
            return arg
        if isinstance(anno, ArgPackType):
            if not isinstance(arg, ArgPack):
                raise TaichiRuntimeTypeError(f"Argument {arg_name} must be a argument pack, got {type(arg)}")
            return tuple(
                TaichiCallableTemplateMapper.extract_arg(arg[name], dtype, arg_name)
                for index, (name, dtype) in enumerate(anno.members.items())
            )
        if isinstance(anno, texture_type.TextureType):
            if not isinstance(arg, taichi.lang._texture.Texture):
                raise TaichiRuntimeTypeError(f"Argument {arg_name} must be a texture, got {type(arg)}")
            if arg.num_dims != anno.num_dimensions:
                raise TaichiRuntimeTypeError(
                    f"TextureType dimension mismatch for argument {arg_name}: expected {anno.num_dimensions}, got {arg.num_dims}"
                )
            return (arg.num_dims,)
        if isinstance(anno, texture_type.RWTextureType):
            if not isinstance(arg, taichi.lang._texture.Texture):
                raise TaichiRuntimeTypeError(f"Argument {arg_name} must be a texture, got {type(arg)}")
            if arg.num_dims != anno.num_dimensions:
                raise TaichiRuntimeTypeError(
                    f"RWTextureType dimension mismatch for argument {arg_name}: expected {anno.num_dimensions}, got {arg.num_dims}"
                )
            if arg.fmt != anno.fmt:
                raise TaichiRuntimeTypeError(
                    f"RWTextureType format mismatch for argument {arg_name}: expected {anno.fmt}, got {arg.fmt}"
                )
            # (penguinliong) '0' is the assumed LOD level. We currently don't
            # support mip-mapping.
            return arg.num_dims, arg.fmt, 0
        if isinstance(anno, ndarray_type.NdarrayType):
            if isinstance(arg, taichi.lang._ndarray.Ndarray):
                anno.check_matched(arg.get_type(), arg_name)
                needs_grad = (arg.grad is not None) if anno.needs_grad is None else anno.needs_grad
                return arg.element_type, len(arg.shape), needs_grad, anno.boundary
            if isinstance(arg, AnyArray):
                ty = arg.get_type()
                anno.check_matched(arg.get_type(), arg_name)
                return ty.element_type, len(arg.shape), ty.needs_grad, anno.boundary
            # external arrays
            shape = getattr(arg, "shape", None)
            if shape is None:
                raise TaichiRuntimeTypeError(f"Invalid type for argument {arg_name}, got {arg}")
            shape = tuple(shape)
            element_shape = ()
            dtype = to_taichi_type(arg.dtype)
            if isinstance(anno.dtype, MatrixType):
                if anno.ndim is not None:
                    if len(shape) != anno.dtype.ndim + anno.ndim:
                        raise ValueError(
                            f"Invalid value for argument {arg_name} - required array has ndim={anno.ndim} element_dim={anno.dtype.ndim}, "
                            f"array with {len(shape)} dimensions is provided"
                        )
                else:
                    if len(shape) < anno.dtype.ndim:
                        raise ValueError(
                            f"Invalid value for argument {arg_name} - required element_dim={anno.dtype.ndim}, "
                            f"array with {len(shape)} dimensions is provided"
                        )
                element_shape = shape[-anno.dtype.ndim :]
                anno_element_shape = anno.dtype.get_shape()
                if None not in anno_element_shape and element_shape != anno_element_shape:
                    raise ValueError(
                        f"Invalid value for argument {arg_name} - required element_shape={anno_element_shape}, "
                        f"array with element shape of {element_shape} is provided"
                    )
            elif anno.dtype is not None:
                # User specified scalar dtype
                if anno.dtype != dtype:
                    raise ValueError(
                        f"Invalid value for argument {arg_name} - required array has dtype={anno.dtype.to_string()}, "
                        f"array with dtype={dtype.to_string()} is provided"
                    )

                if anno.ndim is not None and len(shape) != anno.ndim:
                    raise ValueError(
                        f"Invalid value for argument {arg_name} - required array has ndim={anno.ndim}, "
                        f"array with {len(shape)} dimensions is provided"
                    )
            needs_grad = getattr(arg, "requires_grad", False) if anno.needs_grad is None else anno.needs_grad
            element_type = (
                _ti_core.get_type_factory_instance().get_tensor_type(element_shape, dtype)
                if len(element_shape) != 0
                else arg.dtype
            )
            return element_type, len(shape) - len(element_shape), needs_grad, anno.boundary
        if isinstance(anno, sparse_matrix_builder):
            return arg.dtype
        # Use '#' as a placeholder because other kinds of arguments are not involved in template instantiation
        return "#"

    def extract(self, args):
        extracted = []
        for arg, kernel_arg in zip(args, self.arguments):
            extracted.append(self.extract_arg(arg, kernel_arg.annotation, kernel_arg.name))
        return tuple(extracted)

    def lookup(self, args):
        if len(args) != self.num_args:
            raise TypeError(f"{self.num_args} argument(s) needed but {len(args)} provided.")

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
        assert autodiff_mode in (
            AutodiffMode.NONE,
            AutodiffMode.VALIDATION,
            AutodiffMode.FORWARD,
            AutodiffMode.REVERSE,
        )
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
        self.mapper = TaichiCallableTemplateMapper(self.arguments, self.template_slot_locations)
        impl.get_runtime().kernels.append(self)
        self.reset()
        self.kernel_cpp = None
        self.compiled_kernels = {}
        self.has_print = False

    def ast_builder(self):
        assert self.kernel_cpp is not None
        return self.kernel_cpp.ast_builder()

    def reset(self):
        self.runtime = impl.get_runtime()
        self.compiled_kernels = {}

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect._empty, None):
            self.return_type = sig.return_annotation
            if sys.version_info >= (3, 9):
                if (
                    isinstance(self.return_type, (types.GenericAlias, typing._GenericAlias))
                    and self.return_type.__origin__ is tuple
                ):
                    self.return_type = self.return_type.__args__
            else:
                if isinstance(self.return_type, typing._GenericAlias) and self.return_type.__origin__ is tuple:
                    self.return_type = self.return_type.__args__
            if not isinstance(self.return_type, (list, tuple)):
                self.return_type = (self.return_type,)
            for return_type in self.return_type:
                if return_type is Ellipsis:
                    raise TaichiSyntaxError("Ellipsis is not supported in return type annotations")
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise TaichiSyntaxError("Taichi kernels do not support variable keyword parameters (i.e., **kwargs)")
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TaichiSyntaxError("Taichi kernels do not support variable positional parameters (i.e., *args)")
            if param.default is not inspect.Parameter.empty:
                raise TaichiSyntaxError("Taichi kernels do not support default values for arguments")
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise TaichiSyntaxError("Taichi kernels do not support keyword parameters")
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TaichiSyntaxError('Taichi kernels only support "positional or keyword" parameters')
            annotation = param.annotation
            if param.annotation is inspect.Parameter.empty:
                if i == 0 and self.classkernel:  # The |self| parameter
                    annotation = template()
                else:
                    raise TaichiSyntaxError("Taichi kernels parameters must be type annotated")
            else:
                if isinstance(
                    annotation,
                    (
                        template,
                        ndarray_type.NdarrayType,
                        texture_type.TextureType,
                        texture_type.RWTextureType,
                    ),
                ):
                    pass
                elif id(annotation) in primitive_types.type_ids:
                    pass
                elif isinstance(annotation, sparse_matrix_builder):
                    pass
                elif isinstance(annotation, MatrixType):
                    pass
                elif isinstance(annotation, StructType):
                    pass
                elif isinstance(annotation, ArgPackType):
                    pass
                else:
                    raise TaichiSyntaxError(f"Invalid type annotation (argument {i}) of Taichi kernel: {annotation}")
            self.arguments.append(KernelArgument(annotation, param.name, param.default))

    def materialize(self, key=None, args=None, arg_features=None):
        if key is None:
            key = (self.func, 0, self.autodiff_mode)
        self.runtime.materialize()

        if key in self.compiled_kernels:
            return

        kernel_name = f"{self.func.__name__}_c{self.kernel_counter}_{key[1]}"
        _logging.trace(f"Compiling kernel {kernel_name} in {self.autodiff_mode}...")

        tree, ctx = _get_tree_and_ctx(
            self,
            args=args,
            excluded_parameters=self.template_slot_locations,
            arg_features=arg_features,
        )

        if self.autodiff_mode != AutodiffMode.NONE:
            KernelSimplicityASTChecker(self.func).visit(tree)

        # Do not change the name of 'taichi_ast_generator'
        # The warning system needs this identifier to remove unnecessary messages
        def taichi_ast_generator(kernel_cxx):
            if self.runtime.inside_kernel:
                raise TaichiSyntaxError(
                    "Kernels cannot call other kernels. I.e., nested kernels are not allowed. "
                    "Please check if you have direct/indirect invocation of kernels within kernels. "
                    "Note that some methods provided by the Taichi standard library may invoke kernels, "
                    "and please move their invocations to Python-scope."
                )
            self.kernel_cpp = kernel_cxx
            self.runtime.inside_kernel = True
            self.runtime.current_kernel = self
            assert self.runtime.compiling_callable is None
            self.runtime.compiling_callable = kernel_cxx
            try:
                ctx.ast_builder = kernel_cxx.ast_builder()
                transform_tree(tree, ctx)
                if not ctx.is_real_function:
                    if self.return_type and ctx.returned != ReturnStatus.ReturnedValue:
                        raise TaichiSyntaxError("Kernel has a return type but does not have a return statement")
            finally:
                self.runtime.inside_kernel = False
                self.runtime.current_kernel = None
                self.runtime.compiling_callable = None

        taichi_kernel = impl.get_runtime().prog.create_kernel(taichi_ast_generator, kernel_name, self.autodiff_mode)
        assert key not in self.compiled_kernels
        self.compiled_kernels[key] = taichi_kernel

    def launch_kernel(self, t_kernel, *args):
        assert len(args) == len(self.arguments), f"{len(self.arguments)} arguments needed but {len(args)} provided"

        tmps = []
        callbacks = []

        actual_argument_slot = 0
        launch_ctx = t_kernel.make_launch_context()
        max_arg_num = 64
        exceed_max_arg_num = False

        def set_arg_ndarray(indices, v):
            v_primal = v.arr
            v_grad = v.grad.arr if v.grad else None
            if v_grad is None:
                launch_ctx.set_arg_ndarray(indices, v_primal)
            else:
                launch_ctx.set_arg_ndarray_with_grad(indices, v_primal, v_grad)

        def set_arg_texture(indices, v):
            launch_ctx.set_arg_texture(indices, v.tex)

        def set_arg_rw_texture(indices, v):
            launch_ctx.set_arg_rw_texture(indices, v.tex)

        def set_arg_ext_array(indices, v, needed):
            # Element shapes are already specialized in Taichi codegen.
            # The shape information for element dims are no longer needed.
            # Therefore we strip the element shapes from the shape vector,
            # so that it only holds "real" array shapes.
            is_soa = needed.layout == Layout.SOA
            array_shape = v.shape
            if functools.reduce(operator.mul, array_shape, 1) > np.iinfo(np.int32).max:
                warnings.warn("Ndarray index might be out of int32 boundary but int64 indexing is not supported yet.")
            if needed.dtype is None or id(needed.dtype) in primitive_types.type_ids:
                element_dim = 0
            else:
                element_dim = needed.dtype.ndim
                array_shape = v.shape[element_dim:] if is_soa else v.shape[:-element_dim]
            if isinstance(v, np.ndarray):
                if v.flags.c_contiguous:
                    launch_ctx.set_arg_external_array_with_shape(indices, int(v.ctypes.data), v.nbytes, array_shape, 0)
                elif v.flags.f_contiguous:
                    # TODO: A better way that avoids copying is saving strides info.
                    tmp = np.ascontiguousarray(v)
                    # Purpose: DO NOT GC |tmp|!
                    tmps.append(tmp)

                    def callback(original, updated):
                        np.copyto(original, np.asfortranarray(updated))

                    callbacks.append(functools.partial(callback, v, tmp))
                    launch_ctx.set_arg_external_array_with_shape(
                        indices, int(tmp.ctypes.data), tmp.nbytes, array_shape, 0
                    )
                else:
                    raise ValueError(
                        "Non contiguous numpy arrays are not supported, please call np.ascontiguousarray(arr) "
                        "before passing it into taichi kernel."
                    )
            elif has_pytorch():
                import torch  # pylint: disable=C0415

                if isinstance(v, torch.Tensor):
                    if not v.is_contiguous():
                        raise ValueError(
                            "Non contiguous tensors are not supported, please call tensor.contiguous() before "
                            "passing it into taichi kernel."
                        )
                    taichi_arch = self.runtime.prog.config().arch

                    def get_call_back(u, v):
                        def call_back():
                            u.copy_(v)

                        return call_back

                    # FIXME: only allocate when launching grad kernel
                    if v.requires_grad and v.grad is None:
                        v.grad = torch.zeros_like(v)

                    if v.requires_grad:
                        if not isinstance(v.grad, torch.Tensor):
                            raise ValueError(
                                f"Expecting torch.Tensor for gradient tensor, but getting {v.grad.__class__.__name__} instead"
                            )
                        if not v.grad.is_contiguous():
                            raise ValueError(
                                "Non contiguous gradient tensors are not supported, please call tensor.grad.contiguous() before passing it into taichi kernel."
                            )

                    tmp = v
                    if (str(v.device) != "cpu") and not (
                        str(v.device).startswith("cuda") and taichi_arch == _ti_core.Arch.cuda
                    ):
                        # Getting a torch CUDA tensor on Taichi non-cuda arch:
                        # We just replace it with a CPU tensor and by the end of kernel execution we'll use the
                        # callback to copy the values back to the original CUDA tensor.
                        host_v = v.to(device="cpu", copy=True)
                        tmp = host_v
                        callbacks.append(get_call_back(v, host_v))

                    launch_ctx.set_arg_external_array_with_shape(
                        indices,
                        int(tmp.data_ptr()),
                        tmp.element_size() * tmp.nelement(),
                        array_shape,
                        int(v.grad.data_ptr()) if v.grad is not None else 0,
                    )
                else:
                    raise TaichiRuntimeTypeError(
                        f"Argument {needed.to_string()} cannot be converted into required type {v}"
                    )
            elif has_paddle():
                import paddle  # pylint: disable=C0415

                if isinstance(v, paddle.Tensor):
                    # For now, paddle.fluid.core.Tensor._ptr() is only available on develop branch
                    def get_call_back(u, v):
                        def call_back():
                            u.copy_(v, False)

                        return call_back

                    tmp = v.value().get_tensor()
                    taichi_arch = self.runtime.prog.config().arch
                    if v.place.is_gpu_place():
                        if taichi_arch != _ti_core.Arch.cuda:
                            # Paddle cuda tensor on Taichi non-cuda arch
                            host_v = v.cpu()
                            tmp = host_v.value().get_tensor()
                            callbacks.append(get_call_back(v, host_v))
                    elif v.place.is_cpu_place():
                        if taichi_arch == _ti_core.Arch.cuda:
                            # Paddle cpu tensor on Taichi cuda arch
                            gpu_v = v.cuda()
                            tmp = gpu_v.value().get_tensor()
                            callbacks.append(get_call_back(v, gpu_v))
                    else:
                        # Paddle do support many other backends like XPU, NPU, MLU, IPU
                        raise TaichiRuntimeTypeError(f"Taichi do not support backend {v.place} that Paddle support")
                    launch_ctx.set_arg_external_array_with_shape(
                        indices, int(tmp._ptr()), v.element_size() * v.size, array_shape, 0
                    )
                else:
                    raise TaichiRuntimeTypeError(
                        f"Argument {needed.to_string()} cannot be converted into required type {v}"
                    )
            else:
                raise TaichiRuntimeTypeError(
                    f"Argument {needed.to_string()} cannot be converted into required type {v}"
                )

        def set_arg_matrix(indices, v, needed):
            if needed.dtype in primitive_types.real_types:

                def cast_func(x):
                    if not isinstance(x, (int, float, np.integer, np.floating)):
                        raise TaichiRuntimeTypeError(
                            f"Argument {needed.dtype.to_string()} cannot be converted into required type {type(x)}"
                        )
                    return float(x)

            elif needed.dtype in primitive_types.integer_types:

                def cast_func(x):
                    if not isinstance(x, (int, np.integer)):
                        raise TaichiRuntimeTypeError(
                            f"Argument {needed.dtype.to_string()} cannot be converted into required type {type(x)}"
                        )
                    return int(x)

            else:
                raise ValueError(f"Matrix dtype {needed.dtype} is not integer type or real type.")

            if needed.ndim == 2:
                v = [cast_func(v[i, j]) for i in range(needed.n) for j in range(needed.m)]
            else:
                v = [cast_func(v[i]) for i in range(needed.n)]
            v = needed(*v)
            needed.set_kernel_struct_args(v, launch_ctx, indices)

        def set_arg_sparse_matrix_builder(indices, v):
            # Pass only the base pointer of the ti.types.sparse_matrix_builder() argument
            launch_ctx.set_arg_uint(indices, v._get_ndarray_addr())

        set_later_list = []

        def recursive_set_args(needed, provided, v, indices):
            in_argpack = len(indices) > 1
            nonlocal actual_argument_slot, exceed_max_arg_num, set_later_list
            if actual_argument_slot >= max_arg_num:
                exceed_max_arg_num = True
                return 0
            actual_argument_slot += 1
            if isinstance(needed, ArgPackType):
                if not isinstance(v, ArgPack):
                    raise TaichiRuntimeTypeError.get(indices, str(needed), str(provided))
                idx_new = 0
                for j, (name, anno) in enumerate(needed.members.items()):
                    idx_new += recursive_set_args(anno, type(v[name]), v[name], indices + (idx_new,))
                launch_ctx.set_arg_argpack(indices, v._ArgPack__argpack)
                return 1
            # Note: do not use sth like "needed == f32". That would be slow.
            if id(needed) in primitive_types.real_type_ids:
                if not isinstance(v, (float, int, np.floating, np.integer)):
                    raise TaichiRuntimeTypeError.get(indices, needed.to_string(), provided)
                if in_argpack:
                    return 1
                launch_ctx.set_arg_float(indices, float(v))
                return 1
            if id(needed) in primitive_types.integer_type_ids:
                if not isinstance(v, (int, np.integer)):
                    raise TaichiRuntimeTypeError.get(indices, needed.to_string(), provided)
                if in_argpack:
                    return 1
                if is_signed(cook_dtype(needed)):
                    launch_ctx.set_arg_int(indices, int(v))
                else:
                    launch_ctx.set_arg_uint(indices, int(v))
                return 1
            if isinstance(needed, sparse_matrix_builder):
                if in_argpack:
                    set_later_list.append((set_arg_sparse_matrix_builder, (v,)))
                    return 0
                set_arg_sparse_matrix_builder(indices, v)
                return 1
            if isinstance(needed, ndarray_type.NdarrayType) and isinstance(v, taichi.lang._ndarray.Ndarray):
                if in_argpack:
                    set_later_list.append((set_arg_ndarray, (v,)))
                    return 0
                set_arg_ndarray(indices, v)
                return 1
            if isinstance(needed, texture_type.TextureType) and isinstance(v, taichi.lang._texture.Texture):
                if in_argpack:
                    set_later_list.append((set_arg_texture, (v,)))
                    return 0
                set_arg_texture(indices, v)
                return 1
            if isinstance(needed, texture_type.RWTextureType) and isinstance(v, taichi.lang._texture.Texture):
                if in_argpack:
                    set_later_list.append((set_arg_rw_texture, (v,)))
                    return 0
                set_arg_rw_texture(indices, v)
                return 1
            if isinstance(needed, ndarray_type.NdarrayType):
                if in_argpack:
                    set_later_list.append((set_arg_ext_array, (v, needed)))
                    return 0
                set_arg_ext_array(indices, v, needed)
                return 1
            if isinstance(needed, MatrixType):
                if in_argpack:
                    return 1
                set_arg_matrix(indices, v, needed)
                return 1
            if isinstance(needed, StructType):
                if in_argpack:
                    return 1
                if not isinstance(v, needed):
                    raise TaichiRuntimeTypeError(f"Argument {provided} cannot be converted into required type {needed}")
                needed.set_kernel_struct_args(v, launch_ctx, indices)
                return 1
            raise ValueError(f"Argument type mismatch. Expecting {needed}, got {type(v)}.")

        template_num = 0
        for i, val in enumerate(args):
            needed_ = self.arguments[i].annotation
            if isinstance(needed_, template):
                template_num += 1
                continue
            recursive_set_args(needed_, type(val), val, (i - template_num,))

        for i, (set_arg_func, params) in enumerate(set_later_list):
            set_arg_func((len(args) - template_num + i,), *params)

        if exceed_max_arg_num:
            raise TaichiRuntimeError(
                f"The number of elements in kernel arguments is too big! Do not exceed {max_arg_num} on {_ti_core.arch_name(impl.current_cfg().arch)} backend."
            )

        try:
            prog = impl.get_runtime().prog
            # Compile kernel (& Online Cache & Offline Cache)
            compiled_kernel_data = prog.compile_kernel(prog.config(), prog.get_device_caps(), t_kernel)
            # Launch kernel
            prog.launch_kernel(compiled_kernel_data, launch_ctx)
        except Exception as e:
            e = handle_exception_from_cpp(e)
            if impl.get_runtime().print_full_traceback:
                raise e
            raise e from None

        ret = None
        ret_dt = self.return_type
        has_ret = ret_dt is not None

        if has_ret or self.has_print:
            runtime_ops.sync()

        if has_ret:
            ret = []
            for i, ret_type in enumerate(ret_dt):
                ret.append(self.construct_kernel_ret(launch_ctx, ret_type, (i,)))
            if len(ret_dt) == 1:
                ret = ret[0]
        if callbacks:
            for c in callbacks:
                c()

        return ret

    def construct_kernel_ret(self, launch_ctx, ret_type, index=()):
        if isinstance(ret_type, CompoundType):
            return ret_type.from_kernel_struct_ret(launch_ctx, index)
        if ret_type in primitive_types.integer_types:
            if is_signed(cook_dtype(ret_type)):
                return launch_ctx.get_struct_ret_int(index)
            return launch_ctx.get_struct_ret_uint(index)
        if ret_type in primitive_types.real_types:
            return launch_ctx.get_struct_ret_float(index)
        raise TaichiRuntimeTypeError(f"Invalid return type on index={index}")

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
        if self.runtime.fwd_mode_manager and not self.runtime.grad_replaced:
            # TODO: if we would like to compute 2nd-order derivatives by forward-on-reverse in a nested context manager fashion,
            # i.e., a `Tape` nested in the `FwdMode`, we can transform the kernels with `mode_original == AutodiffMode.REVERSE` only,
            # to avoid duplicate computation for 1st-order derivatives
            self.runtime.fwd_mode_manager.insert(self)

        # Both the class kernels and the plain-function kernels are unified now.
        # In both cases, |self.grad| is another Kernel instance that computes the
        # gradient. For class kernels, args[0] is always the kernel owner.

        # No need to capture grad kernels because they are already bound with their primal kernels
        if (
            self.autodiff_mode in (AutodiffMode.NONE, AutodiffMode.VALIDATION)
            and self.runtime.target_tape
            and not self.runtime.grad_replaced
        ):
            self.runtime.target_tape.insert(self, args)

        if self.autodiff_mode != AutodiffMode.NONE and impl.current_cfg().opt_level == 0:
            _logging.warn("""opt_level = 1 is enforced to enable gradient computation.""")
            impl.current_cfg().opt_level = 1
        key = self.ensure_compiled(*args)
        kernel_cpp = self.compiled_kernels[key]
        return self.launch_kernel(kernel_cpp, *args)


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
    re.compile(r"@(\w+\.)?data_oriented"),
    re.compile(r"class "),
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
        print(f"kernel={_func.__name__} is_classkernel={is_classkernel}")
    primal = Kernel(_func, autodiff_mode=AutodiffMode.NONE, _classkernel=is_classkernel)
    adjoint = Kernel(_func, autodiff_mode=AutodiffMode.REVERSE, _classkernel=is_classkernel)
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
            assert not hasattr(clsobj, "_data_oriented")
            raise TaichiSyntaxError(f"Please decorate class {clsobj.__name__} with @ti.data_oriented")

    else:

        @functools.wraps(_func)
        def wrapped(*args, **kwargs):
            try:
                return primal(*args, **kwargs)
            except (TaichiCompilationError, TaichiRuntimeError) as e:
                if impl.get_runtime().print_full_traceback:
                    raise e
                raise type(e)("\n" + str(e)) from None

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
        if not getattr(clsobj, "_data_oriented", False):
            raise TaichiSyntaxError(f"Please decorate class {clsobj.__name__} with @ti.data_oriented")
        self._kernel_owner = kernel_owner
        self._primal = wrapped_kernel_func._primal
        self._adjoint = wrapped_kernel_func._adjoint
        self._is_staticmethod = wrapped_kernel_func._is_staticmethod
        self.__name__ = None

    def __call__(self, *args, **kwargs):
        try:
            if self._is_staticmethod:
                return self._primal(*args, **kwargs)
            return self._primal(self._kernel_owner, *args, **kwargs)
        except (TaichiCompilationError, TaichiRuntimeError) as e:
            if impl.get_runtime().print_full_traceback:
                raise e
            raise type(e)("\n" + str(e)) from None

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
        if hasattr(x, "_is_wrapped_kernel"):
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


__all__ = ["data_oriented", "func", "kernel", "pyfunc", "real_func"]
