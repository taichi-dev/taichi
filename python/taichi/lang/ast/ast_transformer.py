import ast
import collections.abc
import itertools
import warnings
from collections import ChainMap
from sys import version_info

from taichi._lib import core as _ti_core
from taichi.lang import (_ndarray, any_array, expr, impl, kernel_arguments,
                         matrix, mesh)
from taichi.lang import ops as ti_ops
from taichi.lang._ndrange import _Ndrange, ndrange
from taichi.lang.ast.ast_transformer_utils import (Builder, LoopStatus,
                                                   ReturnStatus)
from taichi.lang.ast.symbol_resolver import ASTResolver
from taichi.lang.exception import TaichiSyntaxError, TaichiTypeError
from taichi.lang.field import Field
from taichi.lang.matrix import (Matrix, MatrixType, Vector, _PyScopeMatrixImpl,
                                _TiScopeMatrixImpl)
from taichi.lang.snode import append
from taichi.lang.util import in_taichi_scope, is_taichi_class, to_taichi_type
from taichi.types import (annotations, ndarray_type, primitive_types,
                          texture_type)
from taichi.types.utils import is_integral

if version_info < (3, 9):
    from astunparse import unparse
else:
    from ast import unparse


def boundary_type_cast_warning(expression):
    expr_dtype = expression.ptr.get_ret_type()
    if not is_integral(expr_dtype) or expr_dtype in [
            primitive_types.i64, primitive_types.u64, primitive_types.u32
    ]:
        warnings.warn(
            f"Casting range_for boundary values from {expr_dtype} to i32, which may cause numerical issues",
            Warning)


class ASTTransformer(Builder):
    @staticmethod
    def build_Name(ctx, node):
        node.ptr = ctx.get_var_by_name(node.id)
        return node.ptr

    @staticmethod
    def build_AnnAssign(ctx, node):
        build_stmt(ctx, node.value)
        build_stmt(ctx, node.annotation)

        is_static_assign = isinstance(
            node.value, ast.Call) and node.value.func.ptr is impl.static

        node.ptr = ASTTransformer.build_assign_annotated(
            ctx, node.target, node.value.ptr, is_static_assign,
            node.annotation.ptr)
        return node.ptr

    @staticmethod
    def build_assign_annotated(ctx, target, value, is_static_assign,
                               annotation):
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
        anno = impl.expr_init(annotation)
        if is_static_assign:
            raise TaichiSyntaxError(
                "Static assign cannot be used on annotated assignment")
        if is_local and not ctx.is_var_declared(target.id):
            var = ti_ops.cast(value, anno)
            var = impl.expr_init(var)
            ctx.create_variable(target.id, var)
        else:
            var = build_stmt(ctx, target)
            if var.ptr.get_ret_type() != anno:
                raise TaichiSyntaxError(
                    "Static assign cannot have type overloading")
            var._assign(value)
        return var

    @staticmethod
    def build_Assign(ctx, node):
        build_stmt(ctx, node.value)

        is_static_assign = isinstance(
            node.value, ast.Call) and node.value.func.ptr is impl.static

        # Keep all generated assign statements and compose single one at last.
        # The variable is introduced to support chained assignments.
        # Ref https://github.com/taichi-dev/taichi/issues/2659.
        values = node.value.ptr if is_static_assign else impl.expr_init(
            node.value.ptr)

        for node_target in node.targets:
            ASTTransformer.build_assign_unpack(ctx, node_target, values,
                                               is_static_assign)
        return None

    @staticmethod
    def build_assign_slice(ctx, node_target, values, is_static_assign):
        target = ASTTransformer.build_Subscript(ctx, node_target, get_ref=True)
        if isinstance(node_target.value.ptr, Matrix):
            if isinstance(node_target.value.ptr._impl, _TiScopeMatrixImpl):
                target._assign(values)
            elif isinstance(node_target.value.ptr._impl, _PyScopeMatrixImpl):
                if in_taichi_scope():
                    raise TaichiTypeError(
                        'PyScope matrix cannot be assigned in Taichi Scope')
                node_target.ptr._assign(node_target.slice.ptr, values)
            else:
                raise TaichiTypeError(f'{type(target)} cannot be subscripted')
        else:
            ASTTransformer.build_assign_basic(ctx,
                                              target,
                                              values,
                                              is_static_assign,
                                              build_target=False)

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
        if isinstance(node_target, ast.Subscript):
            return ASTTransformer.build_assign_slice(ctx, node_target, values,
                                                     is_static_assign)

        if not isinstance(node_target, ast.Tuple):
            return ASTTransformer.build_assign_basic(ctx, node_target, values,
                                                     is_static_assign)
        targets = node_target.elts

        if isinstance(values, matrix.Matrix):
            if not values.m == 1:
                raise ValueError(
                    'Matrices with more than one columns cannot be unpacked')
            values = values.entries

        if not isinstance(values, collections.abc.Sequence):
            raise TaichiSyntaxError(f'Cannot unpack type: {type(values)}')

        if len(values) != len(targets):
            raise TaichiSyntaxError(
                "The number of targets is not equal to value length")

        for i, target in enumerate(targets):
            ASTTransformer.build_assign_basic(ctx, target, values[i],
                                              is_static_assign)

        return None

    @staticmethod
    def build_assign_basic(ctx,
                           target,
                           value,
                           is_static_assign,
                           build_target=True):
        """Build basic assignment like this: target = value.

         Args:
            ctx (ast_builder_utils.BuilderContext): The builder context.
            target (ast.Name): A variable name. `target.id` holds the name as
            a string.
            value: A node representing the value.
            is_static_assign: A boolean value indicating whether this is a static assignment
        """
        is_local = isinstance(target, ast.Name)
        if is_static_assign:
            if not is_local:
                raise TaichiSyntaxError(
                    "Static assign cannot be used on elements in arrays")
            ctx.create_variable(target.id, value)
            var = value
        elif is_local and not ctx.is_var_declared(target.id):
            var = impl.expr_init(value)
            ctx.create_variable(target.id, var)
        else:
            if build_target:
                var = build_stmt(ctx, target)
            else:
                var = target
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
        is_static_assign = isinstance(
            node.value, ast.Call) and node.value.func.ptr is impl.static
        node.ptr = ASTTransformer.build_assign_basic(ctx, node.target,
                                                     node.value.ptr,
                                                     is_static_assign)
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
    def build_Subscript(ctx, node, get_ref=False):
        build_stmt(ctx, node.value)
        build_stmt(ctx, node.slice)
        if not ASTTransformer.is_tuple(node.slice):
            node.slice.ptr = [node.slice.ptr]
        node.ptr = impl.subscript(node.value.ptr,
                                  *node.slice.ptr,
                                  get_ref=get_ref)
        return node.ptr

    @staticmethod
    def build_Slice(ctx, node):
        if node.lower is not None:
            build_stmt(ctx, node.lower)
        if node.upper is not None:
            build_stmt(ctx, node.upper)
        if node.step is not None:
            build_stmt(ctx, node.step)

        node.ptr = slice(node.lower.ptr if node.lower else None,
                         node.upper.ptr if node.upper else None,
                         node.step.ptr if node.step else None)
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
        for value in _iter:
            with ctx.variable_scope_guard():
                ASTTransformer.build_assign_unpack(
                    ctx, node.generators[now_comp].target, value, True)
                with ctx.static_scope_guard():
                    build_stmts(ctx, node.generators[now_comp].ifs)
                ASTTransformer.process_ifs(ctx, node, now_comp, 0, func,
                                           result)
        return None

    @staticmethod
    def process_ifs(ctx, node, now_comp, now_if, func, result):
        if now_if >= len(node.generators[now_comp].ifs):
            return ASTTransformer.process_generators(ctx, node, now_comp + 1,
                                                     func, result)
        cond = node.generators[now_comp].ifs[now_if].ptr
        if cond:
            ASTTransformer.process_ifs(ctx, node, now_comp, now_if + 1, func,
                                       result)

        return None

    @staticmethod
    def build_ListComp(ctx, node):
        result = []
        ASTTransformer.process_generators(ctx, node, 0,
                                          ASTTransformer.process_listcomp,
                                          result)
        node.ptr = result
        return node.ptr

    @staticmethod
    def build_DictComp(ctx, node):
        result = {}
        ASTTransformer.process_generators(ctx, node, 0,
                                          ASTTransformer.process_dictcomp,
                                          result)
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
    def build_JoinedStr(ctx, node):
        str_spec = ''
        args = []
        for sub_node in node.values:
            if isinstance(sub_node, ast.FormattedValue):
                str_spec += '{}'
                args.append(build_stmt(ctx, sub_node.value))
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
        func = node.func.ptr
        replace_func = {
            id(print): impl.ti_print,
            id(min): ti_ops.min,
            id(max): ti_ops.max,
            id(int): impl.ti_int,
            id(float): impl.ti_float,
            id(any): ti_ops.ti_any,
            id(all): ti_ops.ti_all,
            id(abs): abs,
            id(pow): pow,
        }
        if id(func) in replace_func:
            node.ptr = replace_func[id(func)](*args, **keywords)
            if func is min or func is max:
                name = "min" if func is min else "max"
                warnings.warn_explicit(
                    f'Calling builtin function "{name}" in Taichi scope is deprecated. '
                    f'Please use "ti.{name}" instead.', DeprecationWarning,
                    ctx.file, node.lineno + ctx.lineno_offset)
            return True
        return False

    @staticmethod
    def build_call_if_is_type(ctx, node, args, keywords):
        func = node.func.ptr
        if id(func) in primitive_types.type_ids:
            if len(args) != 1 or keywords:
                raise TaichiSyntaxError(
                    "A primitive type can only decorate a single expression.")
            if is_taichi_class(args[0]):
                raise TaichiSyntaxError(
                    "A primitive type cannot decorate an expression with a compound type."
                )
            if isinstance(args[0], expr.Expr):
                node.ptr = ti_ops.cast(args[0], func)
            else:
                node.ptr = expr.Expr(args[0], dtype=func)
            return True
        return False

    @staticmethod
    def warn_if_is_external_func(ctx, node):
        func = node.func.ptr
        if ctx.is_in_static_scope():  # allow external function in static scope
            return
        if hasattr(func, "_is_taichi_function") or hasattr(
                func, "_is_wrapped_kernel"):  # taichi func/kernel
            return
        if hasattr(
                func, "__module__"
        ) and func.__module__ and func.__module__.startswith("taichi."):
            return
        name = unparse(node.func).strip()
        warnings.warn_explicit(
            f'Calling non-taichi function "{name}". '
            f'Scope inside the function is not processed by the Taichi AST transformer. '
            f'The function may not work as expected. Proceed with caution! '
            f'Maybe you can consider turning it into a @ti.func?', UserWarning,
            ctx.file, node.lineno + ctx.lineno_offset)

    @staticmethod
    def build_Call(ctx, node):
        if ASTTransformer.get_decorator(ctx, node) == 'static':
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
                for i in arg.ptr:
                    args.append(i)
            else:
                args.append(arg.ptr)
        keywords = dict(ChainMap(*[keyword.ptr for keyword in node.keywords]))
        func = node.func.ptr

        if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value.ptr, str) and node.func.attr == 'format':
            args.insert(0, node.func.value.ptr)
            node.ptr = impl.ti_format(*args, **keywords)
            return node.ptr

        if (isinstance(node.func, ast.Attribute) and
            (func == Matrix
             or func == Vector)) and impl.current_cfg().real_matrix:
            node.ptr = matrix.make_matrix(*args, **keywords)
            return node.ptr

        if ASTTransformer.build_call_if_is_builtin(ctx, node, args, keywords):
            return node.ptr

        if ASTTransformer.build_call_if_is_type(ctx, node, args, keywords):
            return node.ptr

        node.ptr = func(*args, **keywords)
        ASTTransformer.warn_if_is_external_func(ctx, node)

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

        def transform_as_kernel():
            # Treat return type
            if node.returns is not None:
                kernel_arguments.decl_ret(ctx.func.return_type)

            for i, arg in enumerate(args.args):
                if isinstance(ctx.func.arguments[i].annotation,
                              annotations.template):
                    ctx.create_variable(arg.arg, ctx.global_vars[arg.arg])
                elif isinstance(ctx.func.arguments[i].annotation,
                                annotations.sparse_matrix_builder):
                    ctx.create_variable(
                        arg.arg,
                        kernel_arguments.decl_sparse_matrix(
                            to_taichi_type(ctx.arg_features[i])))
                elif isinstance(ctx.func.arguments[i].annotation,
                                ndarray_type.NdarrayType):
                    ctx.create_variable(
                        arg.arg,
                        kernel_arguments.decl_ndarray_arg(
                            to_taichi_type(ctx.arg_features[i][0]),
                            ctx.arg_features[i][1], ctx.arg_features[i][2],
                            ctx.arg_features[i][3]))
                elif isinstance(ctx.func.arguments[i].annotation,
                                texture_type.TextureType):
                    ctx.create_variable(
                        arg.arg,
                        kernel_arguments.decl_texture_arg(
                            ctx.func.arguments[i].annotation.num_dimensions))
                elif isinstance(ctx.func.arguments[i].annotation,
                                texture_type.RWTextureType):
                    ctx.create_variable(
                        arg.arg,
                        kernel_arguments.decl_rw_texture_arg(
                            ctx.func.arguments[i].annotation.num_dimensions,
                            ctx.func.arguments[i].annotation.num_channels,
                            ctx.func.arguments[i].annotation.channel_format,
                            ctx.func.arguments[i].annotation.lod))
                elif isinstance(ctx.func.arguments[i].annotation, MatrixType):
                    ctx.create_variable(
                        arg.arg,
                        kernel_arguments.decl_matrix_arg(
                            ctx.func.arguments[i].annotation))
                elif isinstance(ctx.func.arguments[i].annotation,
                                primitive_types.RefType):
                    ctx.create_variable(
                        arg.arg,
                        kernel_arguments.decl_scalar_arg(
                            ctx.func.arguments[i].annotation))
                else:
                    ctx.global_vars[
                        arg.arg] = kernel_arguments.decl_scalar_arg(
                            ctx.func.arguments[i].annotation)
            # remove original args
            node.args.args = []

        if ctx.is_kernel:  # ti.kernel
            transform_as_kernel()

        else:  # ti.func
            if ctx.is_real_function:
                transform_as_kernel()
            else:
                assert len(args.args) == len(ctx.argument_data)
                for i, (arg,
                        data) in enumerate(zip(args.args, ctx.argument_data)):
                    # Template arguments are passed by reference.
                    if isinstance(ctx.func.arguments[i].annotation,
                                  annotations.template):

                        ctx.create_variable(ctx.func.arguments[i].name, data)
                        continue

                    # Ndarray arguments are passed by reference.
                    if isinstance(ctx.func.arguments[i].annotation,
                                  (ndarray_type.NdarrayType)):
                        if not isinstance(
                                data,
                            (_ndarray.ScalarNdarray, matrix.VectorNdarray,
                             matrix.MatrixNdarray, any_array.AnyArray)):
                            raise TaichiSyntaxError(
                                f"Argument {arg.arg} of type {ctx.func.arguments[i].annotation} is not recognized."
                            )
                        ctx.func.arguments[i].annotation.check_matched(
                            data.get_type())
                        ctx.create_variable(ctx.func.arguments[i].name, data)
                        continue

                    # Matrix arguments are passed by value.
                    if isinstance(ctx.func.arguments[i].annotation,
                                  (MatrixType)):
                        if not isinstance(data, Matrix):
                            raise TaichiSyntaxError(
                                f"Argument {arg.arg} of type {ctx.func.arguments[i].annotation} is expected to be a Matrix, but got {type(data)}."
                            )

                        if data.m != ctx.func.arguments[i].annotation.m:
                            raise TaichiSyntaxError(
                                f"Argument {arg.arg} of type {ctx.func.arguments[i].annotation} is expected to be a Matrix with m {ctx.func.arguments[i].annotation.m}, but got {data.m}."
                            )

                        if data.n != ctx.func.arguments[i].annotation.n:
                            raise TaichiSyntaxError(
                                f"Argument {arg.arg} of type {ctx.func.arguments[i].annotation} is expected to be a Matrix with n {ctx.func.arguments[i].annotation.n}, but got {data.n}."
                            )
                        ctx.create_variable(arg.arg, impl.expr_init_func(data))
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
                raise TaichiSyntaxError(
                    "Return inside non-static if/for is not supported")
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
                    'with a return value must be annotated '
                    'with a return type, e.g. def func() -> ti.f32')
            if id(ctx.func.return_type) in primitive_types.type_ids:
                ctx.ast_builder.create_kernel_exprgroup_return(
                    expr.make_expr_group(
                        ti_ops.cast(expr.Expr(node.value.ptr),
                                    ctx.func.return_type).ptr))
            elif isinstance(ctx.func.return_type, MatrixType):
                ctx.ast_builder.create_kernel_exprgroup_return(
                    expr.make_expr_group([
                        ti_ops.cast(exp, ctx.func.return_type.dtype) for exp in
                        itertools.chain.from_iterable(node.value.ptr.to_list())
                    ]))
            else:
                raise TaichiSyntaxError(
                    "The return type is not supported now!")
            # For args[0], it is an ast.Attribute, because it loads the
            # attribute, |ptr|, of the expression |ret_expr|. Therefore we
            # only need to replace the object part, i.e. args[0].value
        else:
            ctx.return_data = node.value.ptr
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
    def build_Attribute(ctx, node):
        if node.attr == "append" and isinstance(node.value, ast.Subscript):
            x = build_stmt(ctx, node.value.value)
            if not isinstance(x, Field) or x.parent(
            ).ptr.type != _ti_core.SNodeType.dynamic:
                raise TaichiSyntaxError(
                    f"In Taichi scope the `append` method is only defined for dynamic SNodes, but {x} is encountered"
                )
            index = build_stmt(ctx, node.value.slice)
            node.value.ptr = None
            node.ptr = lambda val: append(x.parent(), index, val)
        else:
            build_stmt(ctx, node.value)
            node.ptr = getattr(node.value.ptr, node.attr)
        return node.ptr

    @staticmethod
    def build_BinOp(ctx, node):
        build_stmt(ctx, node.left)
        build_stmt(ctx, node.right)
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
            ast.MatMult: lambda l, r: l @ r,
        }.get(type(node.op))
        node.ptr = op(node.left.ptr, node.right.ptr)
        return node.ptr

    @staticmethod
    def build_AugAssign(ctx, node):
        build_stmt(ctx, node.target)
        build_stmt(ctx, node.value)
        node.ptr = node.target.ptr._augassign(node.value.ptr,
                                              type(node.op).__name__)
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
            ast.Is: lambda l, r: l is r,
            ast.IsNot: lambda l, r: l is not r,
        }
        if ctx.is_in_static_scope():
            ops = {**ops, **ops_static}
        operands = [node.left.ptr
                    ] + [comparator.ptr for comparator in node.comparators]
        val = True
        for i, node_op in enumerate(node.ops):
            l = operands[i]
            r = operands[i + 1]
            op = ops.get(type(node_op))
            if isinstance(node_op, (ast.Is, ast.IsNot)):
                name = "is" if isinstance(node_op, ast.Is) else "is not"
                warnings.warn_explicit(
                    f'Operator "{name}" in Taichi scope is deprecated. Please avoid using it.',
                    DeprecationWarning, ctx.file,
                    node.lineno + ctx.lineno_offset)
            if op is None:
                if type(node_op) in ops_static:
                    raise TaichiSyntaxError(
                        f'"{type(node_op).__name__}" is only supported inside `ti.static`.'
                    )
                else:
                    raise TaichiSyntaxError(
                        f'"{type(node_op).__name__}" is not supported in Taichi kernels.'
                    )
            val = ti_ops.bit_and(val, op(l, r))
        if not isinstance(val, bool):
            val = ti_ops.cast(val, primitive_types.i32)
        node.ptr = val
        return node.ptr

    @staticmethod
    def get_decorator(ctx, node):
        if not isinstance(node, ast.Call):
            return ''
        for wanted, name in [
            (impl.static, 'static'),
            (impl.grouped, 'grouped'),
            (ndrange, 'ndrange'),
        ]:
            if ASTResolver.resolve_to(node.func, wanted, ctx.global_vars):
                return name
        return ''

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
        if is_grouped:
            assert len(node.iter.args[0].args) == 1
            ndrange_arg = build_stmt(ctx, node.iter.args[0].args[0])
            if not isinstance(ndrange_arg, _Ndrange):
                raise TaichiSyntaxError(
                    "Only 'ti.ndrange' is allowed in 'ti.static(ti.grouped(...))'."
                )
            targets = ASTTransformer.get_for_loop_targets(node)
            if len(targets) != 1:
                raise TaichiSyntaxError(
                    f"Group for should have 1 loop target, found {len(targets)}"
                )
            target = targets[0]
            for value in impl.grouped(ndrange_arg):
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
            for target_values in node.iter.ptr:
                if not isinstance(
                        target_values,
                        collections.abc.Sequence) or len(targets) == 1:
                    target_values = [target_values]
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
            loop_var = expr.Expr(ctx.ast_builder.make_id_expr(''))
            ctx.create_variable(loop_name, loop_var)
            if len(node.iter.args) not in [1, 2]:
                raise TaichiSyntaxError(
                    f"Range should have 1 or 2 arguments, found {len(node.iter.args)}"
                )
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

            ctx.ast_builder.begin_frontend_range_for(loop_var.ptr, begin.ptr,
                                                     end.ptr)
            build_stmts(ctx, node.body)
            ctx.ast_builder.end_frontend_range_for()
        return None

    @staticmethod
    def build_ndrange_for(ctx, node):
        with ctx.variable_scope_guard():
            ndrange_var = impl.expr_init(build_stmt(ctx, node.iter))
            ndrange_begin = ti_ops.cast(expr.Expr(0), primitive_types.i32)
            ndrange_end = ti_ops.cast(
                expr.Expr(impl.subscript(ndrange_var.acc_dimensions, 0)),
                primitive_types.i32)
            ndrange_loop_var = expr.Expr(ctx.ast_builder.make_id_expr(''))
            ctx.ast_builder.begin_frontend_range_for(ndrange_loop_var.ptr,
                                                     ndrange_begin.ptr,
                                                     ndrange_end.ptr)
            I = impl.expr_init(ndrange_loop_var)
            targets = ASTTransformer.get_for_loop_targets(node)
            for i, target in enumerate(targets):
                if i + 1 < len(targets):
                    target_tmp = impl.expr_init(
                        I // ndrange_var.acc_dimensions[i + 1])
                else:
                    target_tmp = impl.expr_init(I)
                ctx.create_variable(
                    target,
                    impl.expr_init(target_tmp + impl.subscript(
                        impl.subscript(ndrange_var.bounds, i), 0)))
                if i + 1 < len(targets):
                    I._assign(I -
                              target_tmp * ndrange_var.acc_dimensions[i + 1])
            build_stmts(ctx, node.body)
            ctx.ast_builder.end_frontend_range_for()
        return None

    @staticmethod
    def build_grouped_ndrange_for(ctx, node):
        with ctx.variable_scope_guard():
            ndrange_var = impl.expr_init(build_stmt(ctx, node.iter.args[0]))
            ndrange_begin = ti_ops.cast(expr.Expr(0), primitive_types.i32)
            ndrange_end = ti_ops.cast(
                expr.Expr(impl.subscript(ndrange_var.acc_dimensions, 0)),
                primitive_types.i32)
            ndrange_loop_var = expr.Expr(ctx.ast_builder.make_id_expr(''))
            ctx.ast_builder.begin_frontend_range_for(ndrange_loop_var.ptr,
                                                     ndrange_begin.ptr,
                                                     ndrange_end.ptr)

            targets = ASTTransformer.get_for_loop_targets(node)
            if len(targets) != 1:
                raise TaichiSyntaxError(
                    f"Group for should have 1 loop target, found {len(targets)}"
                )
            target = targets[0]
            target_var = impl.expr_init(
                matrix.Vector([0] * len(ndrange_var.dimensions),
                              dt=primitive_types.i32))
            ctx.create_variable(target, target_var)
            I = impl.expr_init(ndrange_loop_var)
            for i in range(len(ndrange_var.dimensions)):
                if i + 1 < len(ndrange_var.dimensions):
                    target_tmp = I // ndrange_var.acc_dimensions[i + 1]
                else:
                    target_tmp = I
                impl.subscript(target_var, i)._assign(target_tmp +
                                                      ndrange_var.bounds[i][0])
                if i + 1 < len(ndrange_var.dimensions):
                    I._assign(I -
                              target_tmp * ndrange_var.acc_dimensions[i + 1])
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
                    raise TaichiSyntaxError(
                        f"Group for should have 1 loop target, found {len(targets)}"
                    )
                target = targets[0]
                loop_var = build_stmt(ctx, node.iter)
                loop_indices = expr.make_var_list(size=len(loop_var.shape),
                                                  ast_builder=ctx.ast_builder)
                expr_group = expr.make_expr_group(loop_indices)
                impl.begin_frontend_struct_for(ctx.ast_builder, expr_group,
                                               loop_var)
                ctx.create_variable(
                    target, matrix.Vector(loop_indices,
                                          dt=primitive_types.i32))
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
                impl.begin_frontend_struct_for(ctx.ast_builder, expr_group,
                                               loop_var)
                build_stmts(ctx, node.body)
                ctx.ast_builder.end_frontend_struct_for()
        return None

    @staticmethod
    def build_mesh_for(ctx, node):
        targets = ASTTransformer.get_for_loop_targets(node)
        if len(targets) != 1:
            raise TaichiSyntaxError(
                "Mesh for should have 1 loop target, found {len(targets)}")
        target = targets[0]

        with ctx.variable_scope_guard():
            var = expr.Expr(ctx.ast_builder.make_id_expr(""))
            ctx.mesh = node.iter.ptr.mesh
            assert isinstance(ctx.mesh, impl.MeshInstance)
            mesh_idx = mesh.MeshElementFieldProxy(ctx.mesh,
                                                  node.iter.ptr._type, var.ptr)
            ctx.create_variable(target, mesh_idx)
            ctx.ast_builder.begin_frontend_mesh_for(mesh_idx.ptr,
                                                    ctx.mesh.mesh_ptr,
                                                    node.iter.ptr._type)
            build_stmts(ctx, node.body)
            ctx.mesh = None
            ctx.ast_builder.end_frontend_mesh_for()
        return None

    @staticmethod
    def build_nested_mesh_for(ctx, node):
        targets = ASTTransformer.get_for_loop_targets(node)
        if len(targets) != 1:
            raise TaichiSyntaxError(
                "Nested-mesh for should have 1 loop target, found {len(targets)}"
            )
        target = targets[0]

        with ctx.variable_scope_guard():
            ctx.mesh = node.iter.ptr.mesh
            assert isinstance(ctx.mesh, impl.MeshInstance)
            loop_name = node.target.id + '_index__'
            loop_var = expr.Expr(ctx.ast_builder.make_id_expr(''))
            ctx.create_variable(loop_name, loop_var)
            begin = expr.Expr(0)
            end = node.iter.ptr.size
            ctx.ast_builder.begin_frontend_range_for(loop_var.ptr, begin.ptr,
                                                     end.ptr)
            entry_expr = _ti_core.get_relation_access(
                ctx.mesh.mesh_ptr, node.iter.ptr.from_index.ptr,
                node.iter.ptr.to_element_type, loop_var.ptr)
            entry_expr.type_check(impl.get_runtime().prog.config)
            mesh_idx = mesh.MeshElementFieldProxy(
                ctx.mesh, node.iter.ptr.to_element_type, entry_expr)
            ctx.create_variable(target, mesh_idx)
            build_stmts(ctx, node.body)
            ctx.ast_builder.end_frontend_range_for()

        return None

    @staticmethod
    def build_For(ctx, node):
        if node.orelse:
            raise TaichiSyntaxError(
                "'else' clause for 'for' not supported in Taichi kernels")
        decorator = ASTTransformer.get_decorator(ctx, node.iter)
        double_decorator = ''
        if decorator != '' and len(node.iter.args) == 1:
            double_decorator = ASTTransformer.get_decorator(
                ctx, node.iter.args[0])

        if decorator == 'static':
            if double_decorator == 'static':
                raise TaichiSyntaxError("'ti.static' cannot be nested")
            with ctx.loop_scope_guard(is_static=True):
                return ASTTransformer.build_static_for(
                    ctx, node, double_decorator == 'grouped')
        with ctx.loop_scope_guard():
            if decorator == 'ndrange':
                if double_decorator != '':
                    raise TaichiSyntaxError(
                        "No decorator is allowed inside 'ti.ndrange")
                return ASTTransformer.build_ndrange_for(ctx, node)
            if decorator == 'grouped':
                if double_decorator == 'static':
                    raise TaichiSyntaxError(
                        "'ti.static' is not allowed inside 'ti.grouped'")
                elif double_decorator == 'ndrange':
                    return ASTTransformer.build_grouped_ndrange_for(ctx, node)
                elif double_decorator == 'grouped':
                    raise TaichiSyntaxError("'ti.grouped' cannot be nested")
                else:
                    return ASTTransformer.build_struct_for(ctx,
                                                           node,
                                                           is_grouped=True)
            elif isinstance(node.iter, ast.Call) and isinstance(
                    node.iter.func, ast.Name) and node.iter.func.id == 'range':
                return ASTTransformer.build_range_for(ctx, node)
            else:
                build_stmt(ctx, node.iter)
                if isinstance(node.iter.ptr, mesh.MeshElementField):
                    if not _ti_core.is_extension_supported(
                            impl.default_cfg().arch, _ti_core.Extension.mesh):
                        raise Exception(
                            'Backend ' + str(impl.default_cfg().arch) +
                            ' doesn\'t support MeshTaichi extension')
                    return ASTTransformer.build_mesh_for(ctx, node)
                if isinstance(node.iter.ptr, mesh.MeshRelationAccessProxy):
                    return ASTTransformer.build_nested_mesh_for(ctx, node)
                # Struct for
                return ASTTransformer.build_struct_for(ctx,
                                                       node,
                                                       is_grouped=False)

    @staticmethod
    def build_While(ctx, node):
        if node.orelse:
            raise TaichiSyntaxError(
                "'else' clause for 'while' not supported in Taichi kernels")

        with ctx.loop_scope_guard():
            ctx.ast_builder.begin_frontend_while(
                expr.Expr(1, dtype=primitive_types.i32).ptr)
            while_cond = build_stmt(ctx, node.test)
            impl.begin_frontend_if(ctx.ast_builder, while_cond)
            ctx.ast_builder.begin_frontend_if_true()
            ctx.ast_builder.pop_scope()
            ctx.ast_builder.begin_frontend_if_false()
            ctx.ast_builder.insert_break_stmt()
            ctx.ast_builder.pop_scope()
            build_stmts(ctx, node.body)
            ctx.ast_builder.pop_scope()
        return None

    @staticmethod
    def build_If(ctx, node):
        build_stmt(ctx, node.test)
        is_static_if = (ASTTransformer.get_decorator(ctx,
                                                     node.test) == "static")

        if is_static_if:
            if node.test.ptr:
                build_stmts(ctx, node.body)
            else:
                build_stmts(ctx, node.orelse)
            return node

        with ctx.non_static_if_guard(node):
            impl.begin_frontend_if(ctx.ast_builder, node.test.ptr)
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
        if not isinstance(node.value, ast.Call):
            return None
        is_taichi_function = getattr(node.value.func.ptr,
                                     '_is_taichi_function', False)
        if is_taichi_function and node.value.func.ptr._is_real_function:
            func_call_result = node.value.ptr
            ctx.ast_builder.insert_expr_stmt(func_call_result.ptr)
        return None

    @staticmethod
    def build_IfExp(ctx, node):
        build_stmt(ctx, node.test)
        build_stmt(ctx, node.body)
        build_stmt(ctx, node.orelse)

        if is_taichi_class(node.test.ptr) or is_taichi_class(
                node.body.ptr) or is_taichi_class(node.orelse.ptr):
            node.ptr = ti_ops.select(node.test.ptr, node.body.ptr,
                                     node.orelse.ptr)
            warnings.warn_explicit(
                'Using conditional expression for element-wise select operation on '
                'Taichi vectors/matrices is deprecated. '
                'Please use "ti.select" instead.', DeprecationWarning,
                ctx.file, node.lineno + ctx.lineno_offset)
            return node.ptr

        is_static_if = (ASTTransformer.get_decorator(ctx,
                                                     node.test) == "static")

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
        if isinstance(msg.left, ast.Constant) and isinstance(
                msg.left.value, str):
            return True
        return False

    @staticmethod
    def _handle_string_mod_args(ctx, node):
        msg = build_stmt(ctx, node.left)
        args = build_stmt(ctx, node.right)
        if not isinstance(args, collections.abc.Sequence):
            args = (args, )
        args = [expr.Expr(x).ptr for x in args]
        return msg, args

    @staticmethod
    def ti_format_list_to_assert_msg(raw):
        entries = impl.ti_format_list_to_content_entries([raw])
        msg = ""
        args = []
        for entry in entries:
            if isinstance(entry, str):
                msg += entry
            elif isinstance(entry, _ti_core.Expr):
                ty = entry.get_ret_type()
                if ty in primitive_types.real_types:
                    msg += "%f"
                elif ty in primitive_types.integer_types:
                    msg += "%d"
                else:
                    raise TaichiSyntaxError(
                        f"Unsupported data type: {type(ty)}")
                args.append(entry)
            else:
                raise TaichiSyntaxError(f"Unsupported type: {type(entry)}")
        return msg, args

    @staticmethod
    def build_Assert(ctx, node):
        extra_args = []
        if node.msg is not None:
            if ASTTransformer._is_string_mod_args(node.msg):
                msg, extra_args = ASTTransformer._handle_string_mod_args(
                    ctx, node.msg)
            else:
                msg = build_stmt(ctx, node.msg)
                if isinstance(node.msg, ast.Constant):
                    msg = str(msg)
                elif isinstance(node.msg, ast.Str):
                    pass
                elif isinstance(msg, collections.abc.Sequence) and len(
                        msg) > 0 and msg[0] == "__ti_format__":
                    msg, extra_args = ASTTransformer.ti_format_list_to_assert_msg(
                        msg)
                else:
                    raise TaichiSyntaxError(
                        f"assert info must be constant or formatted string, not {type(msg)}"
                    )
        else:
            msg = unparse(node.test)
        test = build_stmt(ctx, node.test)
        impl.ti_assert(test, msg.strip(), extra_args)
        return None

    @staticmethod
    def build_Break(ctx, node):
        if ctx.is_in_static_for():
            nearest_non_static_if: ast.If = ctx.current_loop_scope(
            ).nearest_non_static_if
            if nearest_non_static_if:
                msg = ctx.get_pos_info(nearest_non_static_if.test)
                msg += "You are trying to `break` a static `for` loop, " \
                       "but the `break` statement is inside a non-static `if`. "
                raise TaichiSyntaxError(msg)
            ctx.set_loop_status(LoopStatus.Break)
        else:
            ctx.ast_builder.insert_break_stmt()
        return None

    @staticmethod
    def build_Continue(ctx, node):
        if ctx.is_in_static_for():
            nearest_non_static_if: ast.If = ctx.current_loop_scope(
            ).nearest_non_static_if
            if nearest_non_static_if:
                msg = ctx.get_pos_info(nearest_non_static_if.test)
                msg += "You are trying to `continue` a static `for` loop, " \
                       "but the `continue` statement is inside a non-static `if`. "
                raise TaichiSyntaxError(msg)
            ctx.set_loop_status(LoopStatus.Continue)
        else:
            ctx.ast_builder.insert_continue_stmt()
        return None

    @staticmethod
    def build_Pass(ctx, node):
        return None


build_stmt = ASTTransformer()


def build_stmts(ctx, stmts):
    with ctx.variable_scope_guard():
        for stmt in stmts:
            if ctx.returned != ReturnStatus.NoReturn or ctx.loop_status(
            ) != LoopStatus.Normal:
                break
            else:
                build_stmt(ctx, stmt)
    return stmts
