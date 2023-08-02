import ast
import collections.abc
import itertools
import operator
import re
import warnings
from collections import ChainMap
from sys import version_info
import inspect
import math

import numpy as np
from taichi._lib import core as _ti_core
from taichi.lang import _ndarray, any_array, expr, impl, kernel_arguments, matrix, mesh
from taichi.lang import ops as ti_ops
from taichi.lang._ndrange import _Ndrange, ndrange
from taichi.lang.argpack import ArgPackType
from taichi.lang.ast.ast_transformer_utils import Builder, LoopStatus, ReturnStatus
from taichi.lang.ast.symbol_resolver import ASTResolver
from taichi.lang.exception import (
    TaichiIndexError,
    TaichiSyntaxError,
    TaichiTypeError,
    handle_exception_from_cpp,
)
from taichi.lang.expr import Expr, make_expr_group
from taichi.lang.exception import TaichiRuntimeTypeError
from taichi.lang.field import Field
from taichi.lang.matrix import Matrix, MatrixType, Vector
from taichi.lang.snode import append, deactivate, length
from taichi.lang.struct import Struct, StructType
from taichi.lang.util import is_taichi_class, to_taichi_type
from taichi.types import annotations, ndarray_type, primitive_types, texture_type
from taichi.types.utils import is_integral

if version_info < (3, 9):
    from astunparse import unparse
else:
    from ast import unparse


def reshape_list(flat_list, target_shape):
    if len(target_shape) < 2:
        return flat_list

    curr_list = []
    dim = target_shape[-1]
    for i, elem in enumerate(flat_list):
        if i % dim == 0:
            curr_list.append([])
        curr_list[-1].append(elem)

    return reshape_list(curr_list, target_shape[:-1])


def boundary_type_cast_warning(expression):
    expr_dtype = expression.ptr.get_rvalue_type()
    if not is_integral(expr_dtype) or expr_dtype in [
        primitive_types.i64,
        primitive_types.u64,
        primitive_types.u32,
    ]:
        warnings.warn(
            f"Casting range_for boundary values from {expr_dtype} to i32, which may cause numerical issues",
            Warning,
        )


class ASTTransformer(Builder):
    @staticmethod
    def build_Name(ctx, node):
        node.ptr = ctx.get_var_by_name(node.id)
        if isinstance(node, (ast.stmt, ast.expr)) and isinstance(node.ptr, Expr):
            node.ptr.dbg_info = _ti_core.DebugInfo(ctx.get_pos_info(node))
            node.ptr.ptr.set_dbg_info(node.ptr.dbg_info)
        return node.ptr

    @staticmethod
    def build_AnnAssign(ctx, node):
        build_stmt(ctx, node.value)
        build_stmt(ctx, node.annotation)

        is_static_assign = isinstance(node.value, ast.Call) and node.value.func.ptr is impl.static

        node.ptr = ASTTransformer.build_assign_annotated(
            ctx, node.target, node.value.ptr, is_static_assign, node.annotation.ptr
        )
        return node.ptr

    @staticmethod
    def build_assign_annotated(ctx, target, value, is_static_assign, annotation):
        """Build an annotated assignment like this: target: annotation = value.

        Args:
           ctx (ast_builder_utils.BuilderContext): The builder context.
           target (ast.Name): A variable name. `target.id` holds the name as
           a string.
           annotation: A type we hope to assign to the target
           value: A node representing the value.
           is_static_assign: A boolean value indicating whether this is a static assignment
        """
        is_local = isinstance(target, ast.Name)
        if is_local and target.id in ctx.kernel_args:
            raise TaichiSyntaxError(
                f'Kernel argument "{target.id}" is immutable in the kernel. '
                f"If you want to change its value, please create a new variable."
            )
        anno = impl.expr_init(annotation)
        if is_static_assign:
            raise TaichiSyntaxError("Static assign cannot be used on annotated assignment")
        if is_local and not ctx.is_var_declared(target.id):
            var = ti_ops.cast(value, anno)
            var = impl.expr_init(var)
            ctx.create_variable(target.id, var)
        else:
            var = build_stmt(ctx, target)
            if var.ptr.get_rvalue_type() != anno:
                raise TaichiSyntaxError("Static assign cannot have type overloading")
            var._assign(value)
        return var

    @staticmethod
    def build_Assign(ctx, node):
        build_stmt(ctx, node.value)
        is_static_assign = isinstance(node.value, ast.Call) and node.value.func.ptr is impl.static

        # Keep all generated assign statements and compose single one at last.
        # The variable is introduced to support chained assignments.
        # Ref https://github.com/taichi-dev/taichi/issues/2659.
        values = node.value.ptr if is_static_assign else impl.expr_init(node.value.ptr)

        for node_target in node.targets:
            ASTTransformer.build_assign_unpack(ctx, node_target, values, is_static_assign)
        return None

    @staticmethod
    def build_assign_unpack(ctx, node_target, values, is_static_assign):
        """Build the unpack assignments like this: (target1, target2) = (value1, value2).
        The function should be called only if the node target is a tuple.

        Args:
            ctx (ast_builder_utils.BuilderContext): The builder context.
            node_target (ast.Tuple): A list or tuple object. `node_target.elts` holds a
            list of nodes representing the elements.
            values: A node/list representing the values.
            is_static_assign: A boolean value indicating whether this is a static assignment
        """
        if not isinstance(node_target, ast.Tuple):
            return ASTTransformer.build_assign_basic(ctx, node_target, values, is_static_assign)
        targets = node_target.elts

        if isinstance(values, matrix.Matrix):
            if not values.m == 1:
                raise ValueError("Matrices with more than one columns cannot be unpacked")
            values = values.entries

        # Unpack: a, b, c = ti.Vector([1., 2., 3.])
        if isinstance(values, impl.Expr) and values.ptr.is_tensor():
            if len(values.get_shape()) > 1:
                raise ValueError("Matrices with more than one columns cannot be unpacked")

            values = ctx.ast_builder.expand_exprs([values.ptr])
            if len(values) == 1:
                values = values[0]

        if isinstance(values, impl.Expr) and values.ptr.is_struct():
            values = ctx.ast_builder.expand_exprs([values.ptr])
            if len(values) == 1:
                values = values[0]

        if not isinstance(values, collections.abc.Sequence):
            raise TaichiSyntaxError(f"Cannot unpack type: {type(values)}")

        if len(values) != len(targets):
            raise TaichiSyntaxError("The number of targets is not equal to value length")

        for i, target in enumerate(targets):
            ASTTransformer.build_assign_basic(ctx, target, values[i], is_static_assign)

        return None

    @staticmethod
    def build_assign_basic(ctx, target, value, is_static_assign):
        """Build basic assignment like this: target = value.

        Args:
           ctx (ast_builder_utils.BuilderContext): The builder context.
           target (ast.Name): A variable name. `target.id` holds the name as
           a string.
           value: A node representing the value.
           is_static_assign: A boolean value indicating whether this is a static assignment
        """
        is_local = isinstance(target, ast.Name)
        if is_local and target.id in ctx.kernel_args:
            raise TaichiSyntaxError(
                f'Kernel argument "{target.id}" is immutable in the kernel. '
                f"If you want to change its value, please create a new variable."
            )
        if is_static_assign:
            if not is_local:
                raise TaichiSyntaxError("Static assign cannot be used on elements in arrays")
            ctx.create_variable(target.id, value)
            var = value
        elif is_local and not ctx.is_var_declared(target.id):
            var = impl.expr_init(value)
            ctx.create_variable(target.id, var)
        else:
            var = build_stmt(ctx, target)
            try:
                var._assign(value)
            except AttributeError:
                raise TaichiSyntaxError(
                    f"Variable '{unparse(target).strip()}' cannot be assigned. Maybe it is not a Taichi object?"
                )
        return var

    @staticmethod
    def build_NamedExpr(ctx, node):
        build_stmt(ctx, node.value)
        is_static_assign = isinstance(node.value, ast.Call) and node.value.func.ptr is impl.static
        node.ptr = ASTTransformer.build_assign_basic(ctx, node.target, node.value.ptr, is_static_assign)
        return node.ptr

    @staticmethod
    def is_tuple(node):
        if isinstance(node, ast.Tuple):
            return True
        if isinstance(node, ast.Index) and isinstance(node.value.ptr, tuple):
            return True
        if isinstance(node.ptr, tuple):
            return True
        return False

    @staticmethod
    def build_Subscript(ctx, node):
        build_stmt(ctx, node.value)
        build_stmt(ctx, node.slice)
        if not ASTTransformer.is_tuple(node.slice):
            node.slice.ptr = [node.slice.ptr]
        node.ptr = impl.subscript(ctx.ast_builder, node.value.ptr, *node.slice.ptr)
        return node.ptr

    @staticmethod
    def build_Slice(ctx, node):
        if node.lower is not None:
            build_stmt(ctx, node.lower)
        if node.upper is not None:
            build_stmt(ctx, node.upper)
        if node.step is not None:
            build_stmt(ctx, node.step)

        node.ptr = slice(
            node.lower.ptr if node.lower else None,
            node.upper.ptr if node.upper else None,
            node.step.ptr if node.step else None,
        )
        return node.ptr

    @staticmethod
    def build_ExtSlice(ctx, node):
        build_stmts(ctx, node.dims)
        node.ptr = tuple(dim.ptr for dim in node.dims)
        return node.ptr

    @staticmethod
    def build_Tuple(ctx, node):
        build_stmts(ctx, node.elts)
        node.ptr = tuple(elt.ptr for elt in node.elts)
        return node.ptr

    @staticmethod
    def build_List(ctx, node):
        build_stmts(ctx, node.elts)
        node.ptr = [elt.ptr for elt in node.elts]
        return node.ptr

    @staticmethod
    def build_Dict(ctx, node):
        dic = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                dic.update(build_stmt(ctx, value))
            else:
                dic[build_stmt(ctx, key)] = build_stmt(ctx, value)
        node.ptr = dic
        return node.ptr

    @staticmethod
    def process_listcomp(ctx, node, result):
        result.append(build_stmt(ctx, node.elt))

    @staticmethod
    def process_dictcomp(ctx, node, result):
        key = build_stmt(ctx, node.key)
        value = build_stmt(ctx, node.value)
        result[key] = value

    @staticmethod
    def process_generators(ctx, node, now_comp, func, result):
        if now_comp >= len(node.generators):
            return func(ctx, node, result)
        with ctx.static_scope_guard():
            _iter = build_stmt(ctx, node.generators[now_comp].iter)

        if isinstance(_iter, impl.Expr) and _iter.ptr.is_tensor():
            shape = _iter.ptr.get_shape()
            flattened = [Expr(x) for x in ctx.ast_builder.expand_exprs([_iter.ptr])]
            _iter = reshape_list(flattened, shape)

        for value in _iter:
            with ctx.variable_scope_guard():
                ASTTransformer.build_assign_unpack(ctx, node.generators[now_comp].target, value, True)
                with ctx.static_scope_guard():
                    build_stmts(ctx, node.generators[now_comp].ifs)
                ASTTransformer.process_ifs(ctx, node, now_comp, 0, func, result)
        return None

    @staticmethod
    def process_ifs(ctx, node, now_comp, now_if, func, result):
        if now_if >= len(node.generators[now_comp].ifs):
            return ASTTransformer.process_generators(ctx, node, now_comp + 1, func, result)
        cond = node.generators[now_comp].ifs[now_if].ptr
        if cond:
            ASTTransformer.process_ifs(ctx, node, now_comp, now_if + 1, func, result)

        return None

    @staticmethod
    def build_ListComp(ctx, node):
        result = []
        ASTTransformer.process_generators(ctx, node, 0, ASTTransformer.process_listcomp, result)
        node.ptr = result
        return node.ptr

    @staticmethod
    def build_DictComp(ctx, node):
        result = {}
        ASTTransformer.process_generators(ctx, node, 0, ASTTransformer.process_dictcomp, result)
        node.ptr = result
        return node.ptr

    @staticmethod
    def build_Index(ctx, node):
        node.ptr = build_stmt(ctx, node.value)
        return node.ptr

    @staticmethod
    def build_Constant(ctx, node):
        node.ptr = node.value
        return node.ptr

    @staticmethod
    def build_Num(ctx, node):
        node.ptr = node.n
        return node.ptr

    @staticmethod
    def build_Str(ctx, node):
        node.ptr = node.s
        return node.ptr

    @staticmethod
    def build_Bytes(ctx, node):
        node.ptr = node.s
        return node.ptr

    @staticmethod
    def build_NameConstant(ctx, node):
        node.ptr = node.value
        return node.ptr

    @staticmethod
    def build_keyword(ctx, node):
        build_stmt(ctx, node.value)
        if node.arg is None:
            node.ptr = node.value.ptr
        else:
            node.ptr = {node.arg: node.value.ptr}
        return node.ptr

    @staticmethod
    def build_Starred(ctx, node):
        node.ptr = build_stmt(ctx, node.value)
        return node.ptr

    @staticmethod
    def build_FormattedValue(ctx, node):
        node.ptr = build_stmt(ctx, node.value)
        if node.format_spec is None or len(node.format_spec.values) == 0:
            return node.ptr
        values = node.format_spec.values
        assert len(values) == 1
        format_str = values[0].s
        assert format_str is not None
        # distinguished from normal list
        return ["__ti_fmt_value__", node.ptr, format_str]

    @staticmethod
    def build_JoinedStr(ctx, node):
        str_spec = ""
        args = []
        for sub_node in node.values:
            if isinstance(sub_node, ast.FormattedValue):
                str_spec += "{}"
                args.append(build_stmt(ctx, sub_node))
            elif isinstance(sub_node, ast.Constant):
                str_spec += sub_node.value
            elif isinstance(sub_node, ast.Str):
                str_spec += sub_node.s
            else:
                raise TaichiSyntaxError("Invalid value for fstring.")

        args.insert(0, str_spec)
        node.ptr = impl.ti_format(*args)
        return node.ptr

    @staticmethod
    def build_call_if_is_builtin(ctx, node, args, keywords):
        from taichi.lang import matrix_ops  # pylint: disable=C0415

        func = node.func.ptr
        replace_func = {
            id(print): impl.ti_print,
            id(min): ti_ops.min,
            id(max): ti_ops.max,
            id(int): impl.ti_int,
            id(bool): impl.ti_bool,
            id(float): impl.ti_float,
            id(any): matrix_ops.any,
            id(all): matrix_ops.all,
            id(abs): abs,
            id(pow): pow,
            id(operator.matmul): matrix_ops.matmul,
        }

        # Builtin 'len' function on Matrix Expr
        if id(func) == id(len) and len(args) == 1:
            if isinstance(args[0], Expr) and args[0].ptr.is_tensor():
                node.ptr = args[0].get_shape()[0]
                return True

        if id(func) in replace_func:
            node.ptr = replace_func[id(func)](*args, **keywords)
            return True
        return False

    @staticmethod
    def build_call_if_is_type(ctx, node, args, keywords):
        func = node.func.ptr
        if id(func) in primitive_types.type_ids:
            if len(args) != 1 or keywords:
                raise TaichiSyntaxError("A primitive type can only decorate a single expression.")
            if is_taichi_class(args[0]):
                raise TaichiSyntaxError("A primitive type cannot decorate an expression with a compound type.")

            if isinstance(args[0], expr.Expr):
                if args[0].ptr.is_tensor():
                    raise TaichiSyntaxError("A primitive type cannot decorate an expression with a compound type.")
                node.ptr = ti_ops.cast(args[0], func)
            else:
                node.ptr = expr.Expr(args[0], dtype=func)
            return True
        return False

    @staticmethod
    def is_external_func(ctx, func) -> bool:
        if ctx.is_in_static_scope():  # allow external function in static scope
            return False
        if hasattr(func, "_is_taichi_function") or hasattr(func, "_is_wrapped_kernel"):  # taichi func/kernel
            return False
        if hasattr(func, "__module__") and func.__module__ and func.__module__.startswith("taichi."):
            return False
        return True

    @staticmethod
    def warn_if_is_external_func(ctx, node):
        func = node.func.ptr
        if not ASTTransformer.is_external_func(ctx, func):
            return
        name = unparse(node.func).strip()
        warnings.warn_explicit(
            f"\x1b[38;5;226m"  # Yellow
            f'Calling non-taichi function "{name}". '
            f"Scope inside the function is not processed by the Taichi AST transformer. "
            f"The function may not work as expected. Proceed with caution! "
            f"Maybe you can consider turning it into a @ti.func?"
            f"\x1b[0m",  # Reset
            SyntaxWarning,
            ctx.file,
            node.lineno + ctx.lineno_offset,
            module="taichi",
        )

    @staticmethod
    # Parses a formatted string and extracts format specifiers from it, along with positional and keyword arguments.
    # This function produces a canonicalized formatted string that includes solely empty replacement fields, e.g. 'qwerty {} {} {} {} {}'.
    # Note that the arguments can be used multiple times in the string.
    # e.g.:
    # origin input: 'qwerty {1} {} {1:.3f} {k:.4f} {k:}'.format(1.0, 2.0, k=k)
    # raw_string: 'qwerty {1} {} {1:.3f} {k:.4f} {k:}'
    # raw_args: [1.0, 2.0]
    # raw_keywords: {'k': <ti.Expr>}
    # return value: ['qwerty {} {} {} {} {}', 2.0, 1.0, ['__ti_fmt_value__', 2.0, '.3f'], ['__ti_fmt_value__', <ti.Expr>, '.4f'], <ti.Expr>]
    def canonicalize_formatted_string(raw_string, *raw_args, **raw_keywords):
        raw_brackets = re.findall(r"{(.*?)}", raw_string)
        brackets = []
        unnamed = 0
        for bracket in raw_brackets:
            item, spec = bracket.split(":") if ":" in bracket else (bracket, None)
            if item.isdigit():
                item = int(item)
            # handle unnamed positional args
            if item == "":
                item = unnamed
                unnamed += 1
            # handle empty spec
            if spec == "":
                spec = None
            brackets.append((item, spec))

        # check for errors in the arguments
        max_args_index = max([t[0] for t in brackets if isinstance(t[0], int)], default=-1)
        if max_args_index + 1 != len(raw_args):
            raise TaichiSyntaxError(
                f"Expected {max_args_index + 1} positional argument(s), but received {len(raw_args)} instead."
            )
        brackets_keywords = [t[0] for t in brackets if isinstance(t[0], str)]
        for item in brackets_keywords:
            if item not in raw_keywords:
                raise TaichiSyntaxError(f"Keyword '{item}' not found.")
        for item in raw_keywords:
            if item not in brackets_keywords:
                raise TaichiSyntaxError(f"Keyword '{item}' not used.")

        # reorganize the arguments based on their positions, keywords, and format specifiers
        args = []
        for item, spec in brackets:
            new_arg = raw_args[item] if isinstance(item, int) else raw_keywords[item]
            if spec is not None:
                args.append(["__ti_fmt_value__", new_arg, spec])
            else:
                args.append(new_arg)
        # put the formatted string as the first argument to make ti.format() happy
        args.insert(0, re.sub(r"{.*?}", "{}", raw_string))
        return args

    @staticmethod
    def build_Call(ctx, node):
        if ASTTransformer.get_decorator(ctx, node) in ["static", "static_assert"]:
            with ctx.static_scope_guard():
                build_stmt(ctx, node.func)
                build_stmts(ctx, node.args)
                build_stmts(ctx, node.keywords)
        else:
            build_stmt(ctx, node.func)
            build_stmts(ctx, node.args)
            build_stmts(ctx, node.keywords)

        args = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                arg_list = arg.ptr
                if isinstance(arg_list, Expr) and arg_list.is_tensor():
                    # Expand Expr with Matrix-type return into list of Exprs
                    arg_list = [Expr(x) for x in ctx.ast_builder.expand_exprs([arg_list.ptr])]

                for i in arg_list:
                    args.append(i)
            else:
                args.append(arg.ptr)
        keywords = dict(ChainMap(*[keyword.ptr for keyword in node.keywords]))
        func = node.func.ptr

        if id(func) in [id(print), id(impl.ti_print)]:
            ctx.func.has_print = True

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value.ptr, str) and node.func.attr == "format":
            raw_string = node.func.value.ptr
            args = ASTTransformer.canonicalize_formatted_string(raw_string, *args, **keywords)
            node.ptr = impl.ti_format(*args)
            return node.ptr

        if id(func) == id(Matrix) or id(func) == id(Vector):
            node.ptr = matrix.make_matrix(*args, **keywords)
            return node.ptr

        if ASTTransformer.build_call_if_is_builtin(ctx, node, args, keywords):
            return node.ptr

        if ASTTransformer.build_call_if_is_type(ctx, node, args, keywords):
            return node.ptr

        if hasattr(node.func, "caller"):
            node.ptr = func(node.func.caller, *args, **keywords)
            return node.ptr
        ASTTransformer.warn_if_is_external_func(ctx, node)
        try:
            node.ptr = func(*args, **keywords)
        except TypeError as e:
            module = inspect.getmodule(func)
            error_msg = re.sub(r"\bExpr\b", "Taichi Expression", str(e))
            msg = f"TypeError when calling `{func.__name__}`: {error_msg}."
            if ASTTransformer.is_external_func(ctx, node.func.ptr):
                args_has_expr = any([isinstance(arg, Expr) for arg in args])
                if args_has_expr and (module == math or module == np):
                    exec_str = f"from taichi import {func.__name__}"
                    try:
                        exec(exec_str, {})
                    except:
                        pass
                    else:
                        msg += f"\nDid you mean to use `ti.{func.__name__}` instead of `{module.__name__}.{func.__name__}`?"
            raise TaichiTypeError(msg)

        if getattr(func, "_is_taichi_function", False):
            ctx.func.has_print |= func.func.has_print

        return node.ptr

    @staticmethod
    def build_FunctionDef(ctx, node):
        if ctx.visited_funcdef:
            raise TaichiSyntaxError(
                f"Function definition is not allowed in 'ti.{'kernel' if ctx.is_kernel else 'func'}'."
            )
        ctx.visited_funcdef = True

        args = node.args
        assert args.vararg is None
        assert args.kwonlyargs == []
        assert args.kw_defaults == []
        assert args.kwarg is None

        def decl_and_create_variable(annotation, name, arg_features, invoke_later_dict, prefix_name, arg_depth):
            full_name = prefix_name + "_" + name
            if not isinstance(annotation, primitive_types.RefType):
                ctx.kernel_args.append(name)
            if isinstance(annotation, ArgPackType):
                kernel_arguments.push_argpack_arg(name)
                d = {}
                items_to_put_in_dict = []
                for j, (_name, anno) in enumerate(annotation.members.items()):
                    result, obj = decl_and_create_variable(
                        anno, _name, arg_features[j], invoke_later_dict, full_name, arg_depth + 1
                    )
                    if not result:
                        d[_name] = None
                        items_to_put_in_dict.append((full_name + "_" + _name, _name, obj))
                    else:
                        d[_name] = obj
                argpack = kernel_arguments.decl_argpack_arg(annotation, d)
                for item in items_to_put_in_dict:
                    invoke_later_dict[item[0]] = argpack, item[1], *item[2]
                return True, argpack
            if isinstance(annotation, annotations.template):
                return True, ctx.global_vars[name]
            if isinstance(annotation, annotations.sparse_matrix_builder):
                return False, (
                    kernel_arguments.decl_sparse_matrix,
                    (
                        to_taichi_type(arg_features),
                        full_name,
                    ),
                )
            if isinstance(annotation, ndarray_type.NdarrayType):
                return False, (
                    kernel_arguments.decl_ndarray_arg,
                    (
                        to_taichi_type(arg_features[0]),
                        arg_features[1],
                        full_name,
                        arg_features[2],
                        arg_features[3],
                    ),
                )
            if isinstance(annotation, texture_type.TextureType):
                return False, (kernel_arguments.decl_texture_arg, (arg_features[0], full_name))
            if isinstance(annotation, texture_type.RWTextureType):
                return False, (
                    kernel_arguments.decl_rw_texture_arg,
                    (arg_features[0], arg_features[1], arg_features[2], full_name),
                )
            if isinstance(annotation, MatrixType):
                return True, kernel_arguments.decl_matrix_arg(annotation, name, arg_depth)
            if isinstance(annotation, StructType):
                return True, kernel_arguments.decl_struct_arg(annotation, name, arg_depth)
            return True, kernel_arguments.decl_scalar_arg(annotation, name, arg_depth)

        def transform_as_kernel():
            # Treat return type
            if node.returns is not None:
                for return_type in ctx.func.return_type:
                    kernel_arguments.decl_ret(return_type)
            impl.get_runtime().compiling_callable.finalize_rets()

            invoke_later_dict = dict()
            create_variable_later = dict()
            for i, arg in enumerate(args.args):
                if isinstance(ctx.func.arguments[i].annotation, ArgPackType):
                    kernel_arguments.push_argpack_arg(ctx.func.arguments[i].name)
                    d = {}
                    items_to_put_in_dict = []
                    for j, (name, anno) in enumerate(ctx.func.arguments[i].annotation.members.items()):
                        result, obj = decl_and_create_variable(
                            anno, name, ctx.arg_features[i][j], invoke_later_dict, "__argpack_" + name, 1
                        )
                        if not result:
                            d[name] = None
                            items_to_put_in_dict.append(("__argpack_" + name, name, obj))
                        else:
                            d[name] = obj
                    argpack = kernel_arguments.decl_argpack_arg(ctx.func.arguments[i].annotation, d)
                    for item in items_to_put_in_dict:
                        invoke_later_dict[item[0]] = argpack, item[1], *item[2]
                    create_variable_later[arg.arg] = argpack
                else:
                    result, obj = decl_and_create_variable(
                        ctx.func.arguments[i].annotation,
                        ctx.func.arguments[i].name,
                        ctx.arg_features[i] if ctx.arg_features is not None else None,
                        invoke_later_dict,
                        "",
                        0,
                    )
                    ctx.create_variable(arg.arg, obj if result else obj[0](*obj[1]))
            for k, v in invoke_later_dict.items():
                argpack, name, func, params = v
                argpack[name] = func(*params)
            for k, v in create_variable_later.items():
                ctx.create_variable(k, v)

            impl.get_runtime().compiling_callable.finalize_params()
            # remove original args
            node.args.args = []

        if ctx.is_kernel:  # ti.kernel
            transform_as_kernel()

        else:  # ti.func
            if ctx.is_real_function:
                transform_as_kernel()
            else:
                assert len(args.args) == len(ctx.argument_data)
                for i, (arg, data) in enumerate(zip(args.args, ctx.argument_data)):
                    # Template arguments are passed by reference.
                    if isinstance(ctx.func.arguments[i].annotation, annotations.template):
                        ctx.create_variable(ctx.func.arguments[i].name, data)
                        continue

                    # Ndarray arguments are passed by reference.
                    if isinstance(ctx.func.arguments[i].annotation, (ndarray_type.NdarrayType)):
                        if not isinstance(
                            data,
                            (
                                _ndarray.ScalarNdarray,
                                matrix.VectorNdarray,
                                matrix.MatrixNdarray,
                                any_array.AnyArray,
                            ),
                        ):
                            raise TaichiSyntaxError(
                                f"Argument {arg.arg} of type {ctx.func.arguments[i].annotation} is not recognized."
                            )
                        ctx.func.arguments[i].annotation.check_matched(data.get_type(), ctx.func.arguments[i].name)
                        ctx.create_variable(ctx.func.arguments[i].name, data)
                        continue

                    # Matrix arguments are passed by value.
                    if isinstance(ctx.func.arguments[i].annotation, (MatrixType)):
                        # "data" is expected to be an Expr here,
                        # so we simply call "impl.expr_init_func(data)" to perform:
                        #
                        # TensorType* t = alloca()
                        # assign(t, data)
                        #
                        # We created local variable "t" - a copy of the passed-in argument "data"
                        if not isinstance(data, expr.Expr) or not data.ptr.is_tensor():
                            raise TaichiSyntaxError(
                                f"Argument {arg.arg} of type {ctx.func.arguments[i].annotation} is expected to be a Matrix, but got {type(data)}."
                            )

                        element_shape = data.ptr.get_rvalue_type().shape()
                        if len(element_shape) != ctx.func.arguments[i].annotation.ndim:
                            raise TaichiSyntaxError(
                                f"Argument {arg.arg} of type {ctx.func.arguments[i].annotation} is expected to be a Matrix with ndim {ctx.func.arguments[i].annotation.ndim}, but got {len(element_shape)}."
                            )

                        assert ctx.func.arguments[i].annotation.ndim > 0
                        if element_shape[0] != ctx.func.arguments[i].annotation.n:
                            raise TaichiSyntaxError(
                                f"Argument {arg.arg} of type {ctx.func.arguments[i].annotation} is expected to be a Matrix with n {ctx.func.arguments[i].annotation.n}, but got {element_shape[0]}."
                            )

                        if (
                            ctx.func.arguments[i].annotation.ndim == 2
                            and element_shape[1] != ctx.func.arguments[i].annotation.m
                        ):
                            raise TaichiSyntaxError(
                                f"Argument {arg.arg} of type {ctx.func.arguments[i].annotation} is expected to be a Matrix with m {ctx.func.arguments[i].annotation.m}, but got {element_shape[0]}."
                            )

                        ctx.create_variable(arg.arg, impl.expr_init_func(data))
                        continue

                    if id(ctx.func.arguments[i].annotation) in primitive_types.type_ids:
                        ctx.create_variable(
                            arg.arg, impl.expr_init_func(ti_ops.cast(data, ctx.func.arguments[i].annotation))
                        )
                        continue
                    # Create a copy for non-template arguments,
                    # so that they are passed by value.
                    ctx.create_variable(arg.arg, impl.expr_init_func(data))

        with ctx.variable_scope_guard():
            build_stmts(ctx, node.body)

        return None

    @staticmethod
    def build_Return(ctx, node):
        if not ctx.is_real_function:
            if ctx.is_in_non_static_control_flow():
                raise TaichiSyntaxError("Return inside non-static if/for is not supported")
        if node.value is not None:
            build_stmt(ctx, node.value)
        if node.value is None or node.value.ptr is None:
            if not ctx.is_real_function:
                ctx.returned = ReturnStatus.ReturnedVoid
            return None
        if ctx.is_kernel or ctx.is_real_function:
            # TODO: check if it's at the end of a kernel, throw TaichiSyntaxError if not
            if ctx.func.return_type is None:
                raise TaichiSyntaxError(
                    f'A {"kernel" if ctx.is_kernel else "function"} '
                    "with a return value must be annotated "
                    "with a return type, e.g. def func() -> ti.f32"
                )
            return_exprs = []
            if len(ctx.func.return_type) == 1:
                node.value.ptr = [node.value.ptr]
            assert len(ctx.func.return_type) == len(node.value.ptr)
            for return_type, ptr in zip(ctx.func.return_type, node.value.ptr):
                if id(return_type) in primitive_types.type_ids:
                    if isinstance(ptr, Expr):
                        if ptr.is_tensor() or ptr.is_struct() or ptr.element_type() not in primitive_types.all_types:
                            raise TaichiRuntimeTypeError.get_ret(str(return_type), ptr)
                    elif not isinstance(ptr, (float, int, np.floating, np.integer)):
                        raise TaichiRuntimeTypeError.get_ret(str(return_type), ptr)
                    return_exprs += [ti_ops.cast(expr.Expr(ptr), return_type).ptr]
                elif isinstance(return_type, MatrixType):
                    values = ptr
                    if isinstance(values, Matrix):
                        if values.ndim != ctx.func.return_type.ndim:
                            raise TaichiRuntimeTypeError(
                                f"Return matrix ndim mismatch, expecting={return_type.ndim}, got={values.ndim}."
                            )
                        elif return_type.get_shape() != values.get_shape():
                            raise TaichiRuntimeTypeError(
                                f"Return matrix shape mismatch, expecting={return_type.get_shape()}, got={values.get_shape()}."
                            )
                        values = (
                            itertools.chain.from_iterable(values.to_list())
                            if values.ndim == 1
                            else iter(values.to_list())
                        )
                    elif isinstance(values, Expr):
                        if not values.is_tensor():
                            raise TaichiRuntimeTypeError.get_ret(return_type.to_string(), ptr)
                        elif (
                            return_type.dtype in primitive_types.real_types
                            and not values.element_type() in primitive_types.all_types
                        ):
                            raise TaichiRuntimeTypeError.get_ret(return_type.dtype.to_string(), values.element_type())
                        elif (
                            return_type.dtype in primitive_types.integer_types
                            and not values.element_type() in primitive_types.integer_types
                        ):
                            raise TaichiRuntimeTypeError.get_ret(return_type.dtype.to_string(), values.element_type())
                        elif len(values.get_shape()) != return_type.ndim:
                            raise TaichiRuntimeTypeError(
                                f"Return matrix ndim mismatch, expecting={return_type.ndim}, got={len(values.get_shape())}."
                            )
                        elif return_type.get_shape() != values.get_shape():
                            raise TaichiRuntimeTypeError(
                                f"Return matrix shape mismatch, expecting={return_type.get_shape()}, got={values.get_shape()}."
                            )
                        values = [values]
                    else:
                        np_array = np.array(values)
                        dt, shape, ndim = np_array.dtype, np_array.shape, np_array.ndim
                        if return_type.dtype in primitive_types.real_types and dt not in (
                            float,
                            int,
                            np.floating,
                            np.integer,
                        ):
                            raise TaichiRuntimeTypeError.get_ret(return_type.dtype.to_string(), dt)
                        elif return_type.dtype in primitive_types.integer_types and dt not in (int, np.integer):
                            raise TaichiRuntimeTypeError.get_ret(return_type.dtype.to_string(), dt)
                        elif ndim != return_type.ndim:
                            raise TaichiRuntimeTypeError(
                                f"Return matrix ndim mismatch, expecting={return_type.ndim}, got={ndim}."
                            )
                        elif return_type.get_shape() != shape:
                            raise TaichiRuntimeTypeError(
                                f"Return matrix shape mismatch, expecting={return_type.get_shape()}, got={shape}."
                            )
                        values = [values]
                    return_exprs += [ti_ops.cast(exp, return_type.dtype) for exp in values]
                elif isinstance(return_type, StructType):
                    if not isinstance(ptr, Struct) or not isinstance(ptr, return_type):
                        raise TaichiRuntimeTypeError.get_ret(str(return_type), ptr)
                    values = ptr
                    assert isinstance(values, Struct)
                    return_exprs += expr._get_flattened_ptrs(values)
                else:
                    raise TaichiSyntaxError("The return type is not supported now!")
            ctx.ast_builder.create_kernel_exprgroup_return(
                expr.make_expr_group(return_exprs), _ti_core.DebugInfo(ctx.get_pos_info(node))
            )
        else:
            ctx.return_data = node.value.ptr
            if ctx.func.return_type is not None:
                if len(ctx.func.return_type) == 1:
                    ctx.return_data = [ctx.return_data]
                for i, return_type in enumerate(ctx.func.return_type):
                    if id(return_type) in primitive_types.type_ids:
                        ctx.return_data[i] = ti_ops.cast(ctx.return_data[i], return_type)
                if len(ctx.func.return_type) == 1:
                    ctx.return_data = ctx.return_data[0]
        if not ctx.is_real_function:
            ctx.returned = ReturnStatus.ReturnedValue
        return None

    @staticmethod
    def build_Module(ctx, node):
        with ctx.variable_scope_guard():
            # Do NOT use |build_stmts| which inserts 'del' statements to the
            # end and deletes parameters passed into the module
            for stmt in node.body:
                build_stmt(ctx, stmt)
        return None

    @staticmethod
    def build_attribute_if_is_dynamic_snode_method(ctx, node):
        is_subscript = isinstance(node.value, ast.Subscript)
        names = ("append", "deactivate", "length")
        if node.attr not in names:
            return False
        if is_subscript:
            x = node.value.value.ptr
            indices = node.value.slice.ptr
        else:
            x = node.value.ptr
            indices = []
        if not isinstance(x, Field):
            return False
        if not x.parent().ptr.type == _ti_core.SNodeType.dynamic:
            return False
        field_dim = x.snode.ptr.num_active_indices()
        indices_expr_group = make_expr_group(*indices)
        index_dim = indices_expr_group.size()
        if field_dim != index_dim + 1:
            return False
        if node.attr == "append":
            node.ptr = lambda val: append(x.parent(), indices, val)
        elif node.attr == "deactivate":
            node.ptr = lambda: deactivate(x.parent(), indices)
        else:
            node.ptr = lambda: length(x.parent(), indices)
        return True

    @staticmethod
    def build_Attribute(ctx, node):
        # There are two valid cases for the methods of Dynamic SNode:
        #
        # 1. x[i, j].append (where the dimension of the field (3 in this case) is equal to one plus the number of the
        # indices (2 in this case) )
        #
        # 2. x.append (where the dimension of the field is one, equal to x[()].append)
        #
        # For the first case, the AST (simplified) is like node = Attribute(value=Subscript(value=x, slice=[i, j]),
        # attr="append"), when we build_stmt(node.value)(build the expression of the Subscript i.e. x[i, j]),
        # it should build the expression of node.value.value (i.e. x) and node.value.slice (i.e. [i, j]), and raise a
        # TaichiIndexError because the dimension of the field is not equal to the number of the indices. Therefore,
        # when we meet the error, we can detect whether it is a method of Dynamic SNode and build the expression if
        # it is by calling build_attribute_if_is_dynamic_snode_method. If we find that it is not a method of Dynamic
        # SNode, we raise the error again.
        #
        # For the second case, the AST (simplified) is like node = Attribute(value=x, attr="append"), and it does not
        # raise error when we build_stmt(node.value). Therefore, when we do not meet the error, we can also detect
        # whether it is a method of Dynamic SNode and build the expression if it is by calling
        # build_attribute_if_is_dynamic_snode_method. If we find that it is not a method of Dynamic SNode,
        # we continue to process it as a normal attribute node.
        try:
            build_stmt(ctx, node.value)
        except Exception as e:
            e = handle_exception_from_cpp(e)
            if isinstance(e, TaichiIndexError):
                node.value.ptr = None
                if ASTTransformer.build_attribute_if_is_dynamic_snode_method(ctx, node):
                    return node.ptr
            raise e

        if ASTTransformer.build_attribute_if_is_dynamic_snode_method(ctx, node):
            return node.ptr

        if isinstance(node.value.ptr, Expr) and not hasattr(node.value.ptr, node.attr):
            if node.attr in Matrix._swizzle_to_keygroup:
                keygroup = Matrix._swizzle_to_keygroup[node.attr]
                Matrix._keygroup_to_checker[keygroup](node.value.ptr, node.attr)
                attr_len = len(node.attr)
                if attr_len == 1:
                    node.ptr = Expr(
                        impl.get_runtime()
                        .compiling_callable.ast_builder()
                        .expr_subscript(
                            node.value.ptr.ptr,
                            make_expr_group(keygroup.index(node.attr)),
                            _ti_core.DebugInfo(impl.get_runtime().get_current_src_info()),
                        )
                    )
                else:
                    node.ptr = Expr(
                        _ti_core.subscript_with_multiple_indices(
                            node.value.ptr.ptr,
                            [make_expr_group(keygroup.index(ch)) for ch in node.attr],
                            (attr_len,),
                            _ti_core.DebugInfo(impl.get_runtime().get_current_src_info()),
                        )
                    )
            else:
                from taichi.lang import (  # pylint: disable=C0415
                    matrix_ops as tensor_ops,
                )

                node.ptr = getattr(tensor_ops, node.attr)
                setattr(node, "caller", node.value.ptr)
        else:
            node.ptr = getattr(node.value.ptr, node.attr)
        return node.ptr

    @staticmethod
    def build_BinOp(ctx, node):
        build_stmt(ctx, node.left)
        build_stmt(ctx, node.right)
        # pylint: disable-msg=C0415
        from taichi.lang.matrix_ops import matmul

        op = {
            ast.Add: lambda l, r: l + r,
            ast.Sub: lambda l, r: l - r,
            ast.Mult: lambda l, r: l * r,
            ast.Div: lambda l, r: l / r,
            ast.FloorDiv: lambda l, r: l // r,
            ast.Mod: lambda l, r: l % r,
            ast.Pow: lambda l, r: l**r,
            ast.LShift: lambda l, r: l << r,
            ast.RShift: lambda l, r: l >> r,
            ast.BitOr: lambda l, r: l | r,
            ast.BitXor: lambda l, r: l ^ r,
            ast.BitAnd: lambda l, r: l & r,
            ast.MatMult: matmul,
        }.get(type(node.op))
        try:
            node.ptr = op(node.left.ptr, node.right.ptr)
        except TypeError as e:
            raise TaichiTypeError(str(e)) from None
        return node.ptr

    @staticmethod
    def build_AugAssign(ctx, node):
        build_stmt(ctx, node.target)
        build_stmt(ctx, node.value)
        if isinstance(node.target, ast.Name) and node.target.id in ctx.kernel_args:
            raise TaichiSyntaxError(
                f'Kernel argument "{node.target.id}" is immutable in the kernel. '
                f"If you want to change its value, please create a new variable."
            )
        node.ptr = node.target.ptr._augassign(node.value.ptr, type(node.op).__name__)
        return node.ptr

    @staticmethod
    def build_UnaryOp(ctx, node):
        build_stmt(ctx, node.operand)
        op = {
            ast.UAdd: lambda l: l,
            ast.USub: lambda l: -l,
            ast.Not: ti_ops.logical_not,
            ast.Invert: lambda l: ~l,
        }.get(type(node.op))
        node.ptr = op(node.operand.ptr)
        return node.ptr

    @staticmethod
    def build_bool_op(op):
        def inner(operands):
            if len(operands) == 1:
                return operands[0].ptr
            return op(operands[0].ptr, inner(operands[1:]))

        return inner

    @staticmethod
    def build_static_and(operands):
        for operand in operands:
            if not operand.ptr:
                return operand.ptr
        return operands[-1].ptr

    @staticmethod
    def build_static_or(operands):
        for operand in operands:
            if operand.ptr:
                return operand.ptr
        return operands[-1].ptr

    @staticmethod
    def build_BoolOp(ctx, node):
        build_stmts(ctx, node.values)
        if ctx.is_in_static_scope():
            ops = {
                ast.And: ASTTransformer.build_static_and,
                ast.Or: ASTTransformer.build_static_or,
            }
        elif impl.get_runtime().short_circuit_operators:
            ops = {
                ast.And: ASTTransformer.build_bool_op(ti_ops.logical_and),
                ast.Or: ASTTransformer.build_bool_op(ti_ops.logical_or),
            }
        else:
            ops = {
                ast.And: ASTTransformer.build_bool_op(ti_ops.bit_and),
                ast.Or: ASTTransformer.build_bool_op(ti_ops.bit_or),
            }
        op = ops.get(type(node.op))
        node.ptr = op(node.values)
        return node.ptr

    @staticmethod
    def build_Compare(ctx, node):
        build_stmt(ctx, node.left)
        build_stmts(ctx, node.comparators)
        ops = {
            ast.Eq: lambda l, r: l == r,
            ast.NotEq: lambda l, r: l != r,
            ast.Lt: lambda l, r: l < r,
            ast.LtE: lambda l, r: l <= r,
            ast.Gt: lambda l, r: l > r,
            ast.GtE: lambda l, r: l >= r,
        }
        ops_static = {
            ast.In: lambda l, r: l in r,
            ast.NotIn: lambda l, r: l not in r,
        }
        if ctx.is_in_static_scope():
            ops = {**ops, **ops_static}
        operands = [node.left.ptr] + [comparator.ptr for comparator in node.comparators]
        val = True
        for i, node_op in enumerate(node.ops):
            if isinstance(node_op, (ast.Is, ast.IsNot)):
                name = "is" if isinstance(node_op, ast.Is) else "is not"
                raise TaichiSyntaxError(f'Operator "{name}" in Taichi scope is not supported.')
            l = operands[i]
            r = operands[i + 1]
            op = ops.get(type(node_op))

            if op is None:
                if type(node_op) in ops_static:
                    raise TaichiSyntaxError(f'"{type(node_op).__name__}" is only supported inside `ti.static`.')
                else:
                    raise TaichiSyntaxError(f'"{type(node_op).__name__}" is not supported in Taichi kernels.')
            val = ti_ops.logical_and(val, op(l, r))
        if not isinstance(val, (bool, np.bool_)):
            val = ti_ops.cast(val, primitive_types.u1)
        node.ptr = val
        return node.ptr

    @staticmethod
    def get_decorator(ctx, node):
        if not isinstance(node, ast.Call):
            return ""
        for wanted, name in [
            (impl.static, "static"),
            (impl.static_assert, "static_assert"),
            (impl.grouped, "grouped"),
            (ndrange, "ndrange"),
        ]:
            if ASTResolver.resolve_to(node.func, wanted, ctx.global_vars):
                return name
        return ""

    @staticmethod
    def get_for_loop_targets(node):
        """
        Returns the list of indices of the for loop |node|.
        See also: https://docs.python.org/3/library/ast.html#ast.For
        """
        if isinstance(node.target, ast.Name):
            return [node.target.id]
        assert isinstance(node.target, ast.Tuple)
        return [name.id for name in node.target.elts]

    @staticmethod
    def build_static_for(ctx, node, is_grouped):
        ti_unroll_limit = impl.get_runtime().unrolling_limit
        if is_grouped:
            assert len(node.iter.args[0].args) == 1
            ndrange_arg = build_stmt(ctx, node.iter.args[0].args[0])
            if not isinstance(ndrange_arg, _Ndrange):
                raise TaichiSyntaxError("Only 'ti.ndrange' is allowed in 'ti.static(ti.grouped(...))'.")
            targets = ASTTransformer.get_for_loop_targets(node)
            if len(targets) != 1:
                raise TaichiSyntaxError(f"Group for should have 1 loop target, found {len(targets)}")
            target = targets[0]
            iter_time = 0
            alert_already = False

            for value in impl.grouped(ndrange_arg):
                iter_time += 1
                if not alert_already and ti_unroll_limit and iter_time > ti_unroll_limit:
                    alert_already = True
                    warnings.warn_explicit(
                        f"""You are unrolling more than
                        {ti_unroll_limit} iterations, so the compile time may be extremely long.
                        You can use a non-static for loop if you want to decrease the compile time.
                        You can disable this warning by setting ti.init(unrolling_limit=0).""",
                        SyntaxWarning,
                        ctx.file,
                        node.lineno + ctx.lineno_offset,
                        module="taichi",
                    )

                with ctx.variable_scope_guard():
                    ctx.create_variable(target, value)
                    build_stmts(ctx, node.body)
                    status = ctx.loop_status()
                    if status == LoopStatus.Break:
                        break
                    elif status == LoopStatus.Continue:
                        ctx.set_loop_status(LoopStatus.Normal)
        else:
            build_stmt(ctx, node.iter)
            targets = ASTTransformer.get_for_loop_targets(node)

            iter_time = 0
            alert_already = False
            for target_values in node.iter.ptr:
                if not isinstance(target_values, collections.abc.Sequence) or len(targets) == 1:
                    target_values = [target_values]

                iter_time += 1
                if not alert_already and ti_unroll_limit and iter_time > ti_unroll_limit:
                    alert_already = True
                    warnings.warn_explicit(
                        f"""You are unrolling more than
                        {ti_unroll_limit} iterations, so the compile time may be extremely long.
                        You can use a non-static for loop if you want to decrease the compile time.
                        You can disable this warning by setting ti.init(unrolling_limit=0).""",
                        SyntaxWarning,
                        ctx.file,
                        node.lineno + ctx.lineno_offset,
                        module="taichi",
                    )

                with ctx.variable_scope_guard():
                    for target, target_value in zip(targets, target_values):
                        ctx.create_variable(target, target_value)
                    build_stmts(ctx, node.body)
                    status = ctx.loop_status()
                    if status == LoopStatus.Break:
                        break
                    elif status == LoopStatus.Continue:
                        ctx.set_loop_status(LoopStatus.Normal)
        return None

    @staticmethod
    def build_range_for(ctx, node):
        with ctx.variable_scope_guard():
            loop_name = node.target.id
            ctx.check_loop_var(loop_name)
            loop_var = expr.Expr(ctx.ast_builder.make_id_expr(""))
            ctx.create_variable(loop_name, loop_var)
            if len(node.iter.args) not in [1, 2]:
                raise TaichiSyntaxError(f"Range should have 1 or 2 arguments, found {len(node.iter.args)}")
            if len(node.iter.args) == 2:
                begin_expr = expr.Expr(build_stmt(ctx, node.iter.args[0]))
                end_expr = expr.Expr(build_stmt(ctx, node.iter.args[1]))

                # Warning for implicit dtype conversion
                boundary_type_cast_warning(begin_expr)
                boundary_type_cast_warning(end_expr)

                begin = ti_ops.cast(begin_expr, primitive_types.i32)
                end = ti_ops.cast(end_expr, primitive_types.i32)

            else:
                end_expr = expr.Expr(build_stmt(ctx, node.iter.args[0]))

                # Warning for implicit dtype conversion
                boundary_type_cast_warning(end_expr)

                begin = ti_ops.cast(expr.Expr(0), primitive_types.i32)
                end = ti_ops.cast(end_expr, primitive_types.i32)

            for_di = _ti_core.DebugInfo(ctx.get_pos_info(node))
            ctx.ast_builder.begin_frontend_range_for(loop_var.ptr, begin.ptr, end.ptr, for_di)
            build_stmts(ctx, node.body)
            ctx.ast_builder.end_frontend_range_for()
        return None

    @staticmethod
    def build_ndrange_for(ctx, node):
        with ctx.variable_scope_guard():
            ndrange_var = impl.expr_init(build_stmt(ctx, node.iter))
            ndrange_begin = ti_ops.cast(expr.Expr(0), primitive_types.i32)
            ndrange_end = ti_ops.cast(
                expr.Expr(impl.subscript(ctx.ast_builder, ndrange_var.acc_dimensions, 0)),
                primitive_types.i32,
            )
            ndrange_loop_var = expr.Expr(ctx.ast_builder.make_id_expr(""))
            for_di = _ti_core.DebugInfo(ctx.get_pos_info(node))
            ctx.ast_builder.begin_frontend_range_for(ndrange_loop_var.ptr, ndrange_begin.ptr, ndrange_end.ptr, for_di)
            I = impl.expr_init(ndrange_loop_var)
            targets = ASTTransformer.get_for_loop_targets(node)
            if len(targets) != len(ndrange_var.dimensions):
                raise TaichiSyntaxError(
                    "Ndrange for loop with number of the loop variables not equal to "
                    "the dimension of the ndrange is not supported. "
                    "Please check if the number of arguments of ti.ndrange() is equal to "
                    "the number of the loop variables."
                )
            for i, target in enumerate(targets):
                if i + 1 < len(targets):
                    target_tmp = impl.expr_init(I // ndrange_var.acc_dimensions[i + 1])
                else:
                    target_tmp = impl.expr_init(I)
                ctx.create_variable(
                    target,
                    impl.expr_init(
                        target_tmp
                        + impl.subscript(
                            ctx.ast_builder,
                            impl.subscript(ctx.ast_builder, ndrange_var.bounds, i),
                            0,
                        )
                    ),
                )
                if i + 1 < len(targets):
                    I._assign(I - target_tmp * ndrange_var.acc_dimensions[i + 1])
            build_stmts(ctx, node.body)
            ctx.ast_builder.end_frontend_range_for()
        return None

    @staticmethod
    def build_grouped_ndrange_for(ctx, node):
        with ctx.variable_scope_guard():
            ndrange_var = impl.expr_init(build_stmt(ctx, node.iter.args[0]))
            ndrange_begin = ti_ops.cast(expr.Expr(0), primitive_types.i32)
            ndrange_end = ti_ops.cast(
                expr.Expr(impl.subscript(ctx.ast_builder, ndrange_var.acc_dimensions, 0)),
                primitive_types.i32,
            )
            ndrange_loop_var = expr.Expr(ctx.ast_builder.make_id_expr(""))
            for_di = _ti_core.DebugInfo(ctx.get_pos_info(node))
            ctx.ast_builder.begin_frontend_range_for(ndrange_loop_var.ptr, ndrange_begin.ptr, ndrange_end.ptr, for_di)

            targets = ASTTransformer.get_for_loop_targets(node)
            if len(targets) != 1:
                raise TaichiSyntaxError(f"Group for should have 1 loop target, found {len(targets)}")
            target = targets[0]
            mat = matrix.make_matrix([0] * len(ndrange_var.dimensions), dt=primitive_types.i32)
            target_var = impl.expr_init(mat)

            ctx.create_variable(target, target_var)
            I = impl.expr_init(ndrange_loop_var)
            for i in range(len(ndrange_var.dimensions)):
                if i + 1 < len(ndrange_var.dimensions):
                    target_tmp = I // ndrange_var.acc_dimensions[i + 1]
                else:
                    target_tmp = I
                impl.subscript(ctx.ast_builder, target_var, i)._assign(target_tmp + ndrange_var.bounds[i][0])
                if i + 1 < len(ndrange_var.dimensions):
                    I._assign(I - target_tmp * ndrange_var.acc_dimensions[i + 1])
            build_stmts(ctx, node.body)
            ctx.ast_builder.end_frontend_range_for()
        return None

    @staticmethod
    def build_struct_for(ctx, node, is_grouped):
        # for i, j in x
        # for I in ti.grouped(x)
        targets = ASTTransformer.get_for_loop_targets(node)

        for target in targets:
            ctx.check_loop_var(target)

        with ctx.variable_scope_guard():
            if is_grouped:
                if len(targets) != 1:
                    raise TaichiSyntaxError(f"Group for should have 1 loop target, found {len(targets)}")
                target = targets[0]
                loop_var = build_stmt(ctx, node.iter)
                loop_indices = expr.make_var_list(size=len(loop_var.shape), ast_builder=ctx.ast_builder)
                expr_group = expr.make_expr_group(loop_indices)
                impl.begin_frontend_struct_for(ctx.ast_builder, expr_group, loop_var)
                ctx.create_variable(target, matrix.make_matrix(loop_indices, dt=primitive_types.i32))
                build_stmts(ctx, node.body)
                ctx.ast_builder.end_frontend_struct_for()
            else:
                _vars = []
                for name in targets:
                    var = expr.Expr(ctx.ast_builder.make_id_expr(""))
                    _vars.append(var)
                    ctx.create_variable(name, var)
                loop_var = node.iter.ptr
                expr_group = expr.make_expr_group(*_vars)
                impl.begin_frontend_struct_for(ctx.ast_builder, expr_group, loop_var)
                build_stmts(ctx, node.body)
                ctx.ast_builder.end_frontend_struct_for()
        return None

    @staticmethod
    def build_mesh_for(ctx, node):
        targets = ASTTransformer.get_for_loop_targets(node)
        if len(targets) != 1:
            raise TaichiSyntaxError("Mesh for should have 1 loop target, found {len(targets)}")
        target = targets[0]

        with ctx.variable_scope_guard():
            var = expr.Expr(ctx.ast_builder.make_id_expr(""))
            ctx.mesh = node.iter.ptr.mesh
            assert isinstance(ctx.mesh, impl.MeshInstance)
            mesh_idx = mesh.MeshElementFieldProxy(ctx.mesh, node.iter.ptr._type, var.ptr)
            ctx.create_variable(target, mesh_idx)
            ctx.ast_builder.begin_frontend_mesh_for(
                mesh_idx.ptr,
                ctx.mesh.mesh_ptr,
                node.iter.ptr._type,
                _ti_core.DebugInfo(impl.get_runtime().get_current_src_info()),
            )
            build_stmts(ctx, node.body)
            ctx.mesh = None
            ctx.ast_builder.end_frontend_mesh_for()
        return None

    @staticmethod
    def build_nested_mesh_for(ctx, node):
        targets = ASTTransformer.get_for_loop_targets(node)
        if len(targets) != 1:
            raise TaichiSyntaxError("Nested-mesh for should have 1 loop target, found {len(targets)}")
        target = targets[0]

        with ctx.variable_scope_guard():
            ctx.mesh = node.iter.ptr.mesh
            assert isinstance(ctx.mesh, impl.MeshInstance)
            loop_name = node.target.id + "_index__"
            loop_var = expr.Expr(ctx.ast_builder.make_id_expr(""))
            ctx.create_variable(loop_name, loop_var)
            begin = expr.Expr(0)
            end = ti_ops.cast(node.iter.ptr.size, primitive_types.i32)
            for_di = _ti_core.DebugInfo(ctx.get_pos_info(node))
            ctx.ast_builder.begin_frontend_range_for(loop_var.ptr, begin.ptr, end.ptr, for_di)
            entry_expr = _ti_core.get_relation_access(
                ctx.mesh.mesh_ptr,
                node.iter.ptr.from_index.ptr,
                node.iter.ptr.to_element_type,
                loop_var.ptr,
            )
            entry_expr.type_check(impl.get_runtime().prog.config())
            mesh_idx = mesh.MeshElementFieldProxy(ctx.mesh, node.iter.ptr.to_element_type, entry_expr)
            ctx.create_variable(target, mesh_idx)
            build_stmts(ctx, node.body)
            ctx.ast_builder.end_frontend_range_for()

        return None

    @staticmethod
    def build_For(ctx, node):
        if node.orelse:
            raise TaichiSyntaxError("'else' clause for 'for' not supported in Taichi kernels")
        decorator = ASTTransformer.get_decorator(ctx, node.iter)
        double_decorator = ""
        if decorator != "" and len(node.iter.args) == 1:
            double_decorator = ASTTransformer.get_decorator(ctx, node.iter.args[0])

        if decorator == "static":
            if double_decorator == "static":
                raise TaichiSyntaxError("'ti.static' cannot be nested")
            with ctx.loop_scope_guard(is_static=True):
                return ASTTransformer.build_static_for(ctx, node, double_decorator == "grouped")
        with ctx.loop_scope_guard():
            if decorator == "ndrange":
                if double_decorator != "":
                    raise TaichiSyntaxError("No decorator is allowed inside 'ti.ndrange")
                return ASTTransformer.build_ndrange_for(ctx, node)
            if decorator == "grouped":
                if double_decorator == "static":
                    raise TaichiSyntaxError("'ti.static' is not allowed inside 'ti.grouped'")
                elif double_decorator == "ndrange":
                    return ASTTransformer.build_grouped_ndrange_for(ctx, node)
                elif double_decorator == "grouped":
                    raise TaichiSyntaxError("'ti.grouped' cannot be nested")
                else:
                    return ASTTransformer.build_struct_for(ctx, node, is_grouped=True)
            elif (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            ):
                return ASTTransformer.build_range_for(ctx, node)
            else:
                build_stmt(ctx, node.iter)
                if isinstance(node.iter.ptr, mesh.MeshElementField):
                    if not _ti_core.is_extension_supported(impl.default_cfg().arch, _ti_core.Extension.mesh):
                        raise Exception(
                            "Backend " + str(impl.default_cfg().arch) + " doesn't support MeshTaichi extension"
                        )
                    return ASTTransformer.build_mesh_for(ctx, node)
                if isinstance(node.iter.ptr, mesh.MeshRelationAccessProxy):
                    return ASTTransformer.build_nested_mesh_for(ctx, node)
                # Struct for
                return ASTTransformer.build_struct_for(ctx, node, is_grouped=False)

    @staticmethod
    def build_While(ctx, node):
        if node.orelse:
            raise TaichiSyntaxError("'else' clause for 'while' not supported in Taichi kernels")

        with ctx.loop_scope_guard():
            stmt_dbg_info = _ti_core.DebugInfo(ctx.get_pos_info(node))
            ctx.ast_builder.begin_frontend_while(expr.Expr(1, dtype=primitive_types.i32).ptr, stmt_dbg_info)
            while_cond = build_stmt(ctx, node.test)
            impl.begin_frontend_if(ctx.ast_builder, while_cond, stmt_dbg_info)
            ctx.ast_builder.begin_frontend_if_true()
            ctx.ast_builder.pop_scope()
            ctx.ast_builder.begin_frontend_if_false()
            ctx.ast_builder.insert_break_stmt(stmt_dbg_info)
            ctx.ast_builder.pop_scope()
            build_stmts(ctx, node.body)
            ctx.ast_builder.pop_scope()
        return None

    @staticmethod
    def build_If(ctx, node):
        build_stmt(ctx, node.test)
        is_static_if = ASTTransformer.get_decorator(ctx, node.test) == "static"

        if is_static_if:
            if node.test.ptr:
                build_stmts(ctx, node.body)
            else:
                build_stmts(ctx, node.orelse)
            return node

        with ctx.non_static_if_guard(node):
            stmt_dbg_info = _ti_core.DebugInfo(ctx.get_pos_info(node))
            impl.begin_frontend_if(ctx.ast_builder, node.test.ptr, stmt_dbg_info)
            ctx.ast_builder.begin_frontend_if_true()
            build_stmts(ctx, node.body)
            ctx.ast_builder.pop_scope()
            ctx.ast_builder.begin_frontend_if_false()
            build_stmts(ctx, node.orelse)
            ctx.ast_builder.pop_scope()
        return None

    @staticmethod
    def build_Expr(ctx, node):
        build_stmt(ctx, node.value)
        return None

    @staticmethod
    def build_IfExp(ctx, node):
        build_stmt(ctx, node.test)
        build_stmt(ctx, node.body)
        build_stmt(ctx, node.orelse)

        has_tensor_type = False
        if isinstance(node.test.ptr, expr.Expr) and node.test.ptr.is_tensor():
            has_tensor_type = True
        if isinstance(node.body.ptr, expr.Expr) and node.body.ptr.is_tensor():
            has_tensor_type = True
        if isinstance(node.orelse.ptr, expr.Expr) and node.orelse.ptr.is_tensor():
            has_tensor_type = True

        if has_tensor_type:
            if isinstance(node.test.ptr, expr.Expr) and node.test.ptr.is_tensor():
                raise TaichiSyntaxError(
                    "Using conditional expression for element-wise select operation on "
                    "Taichi vectors/matrices is deprecated and removed starting from Taichi v1.5.0 "
                    'Please use "ti.select" instead.'
                )
            node.ptr = ti_ops.select(node.test.ptr, node.body.ptr, node.orelse.ptr)
            return node.ptr

        is_static_if = ASTTransformer.get_decorator(ctx, node.test) == "static"

        if is_static_if:
            if node.test.ptr:
                node.ptr = build_stmt(ctx, node.body)
            else:
                node.ptr = build_stmt(ctx, node.orelse)
            return node.ptr

        node.ptr = ti_ops.ifte(node.test.ptr, node.body.ptr, node.orelse.ptr)
        return node.ptr

    @staticmethod
    def _is_string_mod_args(msg):
        # 1. str % (a, b, c, ...)
        # 2. str % single_item
        # Note that |msg.right| may not be a tuple.
        if not isinstance(msg, ast.BinOp):
            return False
        if not isinstance(msg.op, ast.Mod):
            return False
        if isinstance(msg.left, ast.Str):
            return True
        if isinstance(msg.left, ast.Constant) and isinstance(msg.left.value, str):
            return True
        return False

    @staticmethod
    def _handle_string_mod_args(ctx, node):
        msg = build_stmt(ctx, node.left)
        args = build_stmt(ctx, node.right)
        if not isinstance(args, collections.abc.Sequence):
            args = (args,)
        args = [expr.Expr(x).ptr for x in args]
        return msg, args

    @staticmethod
    def ti_format_list_to_assert_msg(raw):
        # TODO: ignore formats here for now
        entries, _ = impl.ti_format_list_to_content_entries([raw])
        msg = ""
        args = []
        for entry in entries:
            if isinstance(entry, str):
                msg += entry
            elif isinstance(entry, _ti_core.Expr):
                ty = entry.get_rvalue_type()
                if ty in primitive_types.real_types:
                    msg += "%f"
                elif ty in primitive_types.integer_types:
                    msg += "%d"
                else:
                    raise TaichiSyntaxError(f"Unsupported data type: {type(ty)}")
                args.append(entry)
            else:
                raise TaichiSyntaxError(f"Unsupported type: {type(entry)}")
        return msg, args

    @staticmethod
    def build_Assert(ctx, node):
        extra_args = []
        if node.msg is not None:
            if ASTTransformer._is_string_mod_args(node.msg):
                msg, extra_args = ASTTransformer._handle_string_mod_args(ctx, node.msg)
            else:
                msg = build_stmt(ctx, node.msg)
                if isinstance(node.msg, ast.Constant):
                    msg = str(msg)
                elif isinstance(node.msg, ast.Str):
                    pass
                elif isinstance(msg, collections.abc.Sequence) and len(msg) > 0 and msg[0] == "__ti_format__":
                    msg, extra_args = ASTTransformer.ti_format_list_to_assert_msg(msg)
                else:
                    raise TaichiSyntaxError(f"assert info must be constant or formatted string, not {type(msg)}")
        else:
            msg = unparse(node.test)
        test = build_stmt(ctx, node.test)
        impl.ti_assert(test, msg.strip(), extra_args, _ti_core.DebugInfo(ctx.get_pos_info(node)))
        return None

    @staticmethod
    def build_Break(ctx, node):
        if ctx.is_in_static_for():
            nearest_non_static_if: ast.If = ctx.current_loop_scope().nearest_non_static_if
            if nearest_non_static_if:
                msg = ctx.get_pos_info(nearest_non_static_if.test)
                msg += (
                    "You are trying to `break` a static `for` loop, "
                    "but the `break` statement is inside a non-static `if`. "
                )
                raise TaichiSyntaxError(msg)
            ctx.set_loop_status(LoopStatus.Break)
        else:
            ctx.ast_builder.insert_break_stmt(_ti_core.DebugInfo(ctx.get_pos_info(node)))
        return None

    @staticmethod
    def build_Continue(ctx, node):
        if ctx.is_in_static_for():
            nearest_non_static_if: ast.If = ctx.current_loop_scope().nearest_non_static_if
            if nearest_non_static_if:
                msg = ctx.get_pos_info(nearest_non_static_if.test)
                msg += (
                    "You are trying to `continue` a static `for` loop, "
                    "but the `continue` statement is inside a non-static `if`. "
                )
                raise TaichiSyntaxError(msg)
            ctx.set_loop_status(LoopStatus.Continue)
        else:
            ctx.ast_builder.insert_continue_stmt(_ti_core.DebugInfo(ctx.get_pos_info(node)))
        return None

    @staticmethod
    def build_Pass(ctx, node):
        return None


build_stmt = ASTTransformer()


def build_stmts(ctx, stmts):
    with ctx.variable_scope_guard():
        for stmt in stmts:
            if ctx.returned != ReturnStatus.NoReturn or ctx.loop_status() != LoopStatus.Normal:
                break
            else:
                build_stmt(ctx, stmt)
    return stmts
