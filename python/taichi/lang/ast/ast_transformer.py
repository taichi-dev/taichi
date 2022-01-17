import ast
import collections.abc
from collections import ChainMap

import astor
from taichi._lib import core as _ti_core
from taichi.lang import expr, impl, kernel_arguments, kernel_impl, matrix, mesh
from taichi.lang import ops as ti_ops
from taichi.lang._ndrange import ndrange
from taichi.lang.ast.ast_transformer_utils import Builder, LoopStatus
from taichi.lang.ast.symbol_resolver import ASTResolver
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.matrix import MatrixType
from taichi.lang.util import is_taichi_class, to_taichi_type
from taichi.types import annotations, primitive_types

from taichi import linalg


class ASTTransformer(Builder):
    @staticmethod
    def build_Name(ast_builder, ctx, node):
        node.ptr = ctx.get_var_by_name(node.id)
        return node.ptr

    @staticmethod
    def build_AnnAssign(ast_builder, ctx, node):
        build_stmt(ast_builder, ctx, node.value)
        build_stmt(ast_builder, ctx, node.annotation)

        is_static_assign = isinstance(
            node.value, ast.Call) and node.value.func.ptr is impl.static

        node.ptr = ASTTransformer.build_assign_annotated(
            ast_builder, ctx, node.target, node.value.ptr, is_static_assign,
            node.annotation.ptr)
        return node.ptr

    @staticmethod
    def build_assign_annotated(ast_builder, ctx, target, value,
                               is_static_assign, annotation):
        """Build an annotated assginment like this: target: annotation = value.

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
            var = build_stmt(ast_builder, ctx, target)
            if var.ptr.get_ret_type() != anno:
                raise TaichiSyntaxError(
                    "Static assign cannot have type overloading")
            var.assign(value)
        return var

    @staticmethod
    def build_Assign(ast_builder, ctx, node):
        build_stmt(ast_builder, ctx, node.value)

        is_static_assign = isinstance(
            node.value, ast.Call) and node.value.func.ptr is impl.static

        # Keep all generated assign statements and compose single one at last.
        # The variable is introduced to support chained assignments.
        # Ref https://github.com/taichi-dev/taichi/issues/2659.
        for node_target in node.targets:
            ASTTransformer.build_assign_unpack(ast_builder, ctx, node_target,
                                               node.value.ptr,
                                               is_static_assign)
        return None

    @staticmethod
    def build_assign_unpack(ast_builder, ctx, node_target, values,
                            is_static_assign):
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
            return ASTTransformer.build_assign_basic(ast_builder, ctx,
                                                     node_target, values,
                                                     is_static_assign)
        targets = node_target.elts
        tmp_tuple = values if is_static_assign else impl.expr_init_list(
            values, len(targets))

        for i, target in enumerate(targets):
            ASTTransformer.build_assign_basic(ast_builder, ctx, target,
                                              tmp_tuple[i], is_static_assign)

        return None

    @staticmethod
    def build_assign_basic(ast_builder, ctx, target, value, is_static_assign):
        """Build basic assginment like this: target = value.

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
            var = build_stmt(ast_builder, ctx, target)
            var.assign(value)
        return var

    @staticmethod
    def build_NamedExpr(ast_builder, ctx, node):
        build_stmt(ast_builder, ctx, node.value)
        is_static_assign = isinstance(
            node.value, ast.Call) and node.value.func.ptr is impl.static
        node.ptr = ASTTransformer.build_assign_basic(ast_builder, ctx,
                                                     node.target,
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
    def build_Subscript(ast_builder, ctx, node):
        build_stmt(ast_builder, ctx, node.value)
        build_stmt(ast_builder, ctx, node.slice)
        if not ASTTransformer.is_tuple(node.slice):
            node.slice.ptr = [node.slice.ptr]
        node.ptr = impl.subscript(node.value.ptr, *node.slice.ptr)
        return node.ptr

    @staticmethod
    def build_Tuple(ast_builder, ctx, node):
        build_stmts(ast_builder, ctx, node.elts)
        node.ptr = tuple(elt.ptr for elt in node.elts)
        return node.ptr

    @staticmethod
    def build_List(ast_builder, ctx, node):
        build_stmts(ast_builder, ctx, node.elts)
        node.ptr = [elt.ptr for elt in node.elts]
        return node.ptr

    @staticmethod
    def build_Dict(ast_builder, ctx, node):
        dic = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                dic.update(build_stmt(ast_builder, ctx, value))
            else:
                dic[build_stmt(ast_builder, ctx,
                               key)] = build_stmt(ast_builder, ctx, value)
        node.ptr = dic
        return node.ptr

    @staticmethod
    def process_listcomp(ast_builder, ctx, node, result):
        result.append(build_stmt(ast_builder, ctx, node.elt))

    @staticmethod
    def process_dictcomp(ast_builder, ctx, node, result):
        key = build_stmt(ast_builder, ctx, node.key)
        value = build_stmt(ast_builder, ctx, node.value)
        result[key] = value

    @staticmethod
    def process_generators(ast_builder, ctx, node, now_comp, func, result):
        if now_comp >= len(node.generators):
            return func(ast_builder, ctx, node, result)
        _iter = build_stmt(ast_builder, ctx, node.generators[now_comp].iter)
        for value in _iter:
            with ctx.variable_scope_guard():
                ASTTransformer.build_assign_unpack(
                    ast_builder, ctx, node.generators[now_comp].target, value,
                    True)
                build_stmts(ast_builder, ctx, node.generators[now_comp].ifs)
                ASTTransformer.process_ifs(ast_builder, ctx, node, now_comp, 0,
                                           func, result)
        return None

    @staticmethod
    def process_ifs(ast_builder, ctx, node, now_comp, now_if, func, result):
        if now_if >= len(node.generators[now_comp].ifs):
            return ASTTransformer.process_generators(ast_builder, ctx, node,
                                                     now_comp + 1, func,
                                                     result)
        cond = node.generators[now_comp].ifs[now_if].ptr
        if cond:
            ASTTransformer.process_ifs(ast_builder, ctx, node, now_comp,
                                       now_if + 1, func, result)

        return None

    @staticmethod
    def build_comprehension(ast_builder, ctx, node):
        build_stmt(ast_builder, ctx, node.target)
        build_stmt(ast_builder, ctx, node.iter)
        build_stmts(ast_builder, ctx, node.ifs)
        return None

    @staticmethod
    def build_ListComp(ast_builder, ctx, node):
        result = []
        ASTTransformer.process_generators(ast_builder, ctx, node, 0,
                                          ASTTransformer.process_listcomp,
                                          result)
        node.ptr = result
        return node.ptr

    @staticmethod
    def build_DictComp(ast_builder, ctx, node):
        result = {}
        ASTTransformer.process_generators(ast_builder, ctx, node, 0,
                                          ASTTransformer.process_dictcomp,
                                          result)
        node.ptr = result
        return node.ptr

    @staticmethod
    def build_Index(ast_builder, ctx, node):

        node.ptr = build_stmt(ast_builder, ctx, node.value)
        return node.ptr

    @staticmethod
    def build_Constant(ast_builder, ctx, node):
        node.ptr = node.value
        return node.ptr

    @staticmethod
    def build_Num(ast_builder, ctx, node):
        node.ptr = node.n
        return node.ptr

    @staticmethod
    def build_Str(ast_builder, ctx, node):
        node.ptr = node.s
        return node.ptr

    @staticmethod
    def build_Bytes(ast_builder, ctx, node):
        node.ptr = node.s
        return node.ptr

    @staticmethod
    def build_NameConstant(ast_builder, ctx, node):
        node.ptr = node.value
        return node.ptr

    @staticmethod
    def build_keyword(ast_builder, ctx, node):
        build_stmt(ast_builder, ctx, node.value)
        if node.arg is None:
            node.ptr = node.value.ptr
        else:
            node.ptr = {node.arg: node.value.ptr}
        return node.ptr

    @staticmethod
    def build_Starred(ast_builder, ctx, node):
        node.ptr = build_stmt(ast_builder, ctx, node.value)
        return node.ptr

    @staticmethod
    def build_JoinedStr(ast_builder, ctx, node):
        str_spec = ''
        args = []
        for sub_node in node.values:
            if isinstance(sub_node, ast.FormattedValue):
                str_spec += '{}'
                args.append(build_stmt(ast_builder, ctx, sub_node.value))
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
    def build_Call(ast_builder, ctx, node):
        is_in_static_scope_prev = ctx.is_in_static_scope
        if ASTTransformer.get_decorator(ctx, node) == 'static':
            ctx.is_in_static_scope = True

        build_stmt(ast_builder, ctx, node.func)
        build_stmts(ast_builder, ctx, node.args)
        build_stmts(ast_builder, ctx, node.keywords)

        ctx.is_in_static_scope = is_in_static_scope_prev

        args = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                args += arg.ptr
            else:
                args.append(arg.ptr)
        keywords = dict(ChainMap(*[keyword.ptr for keyword in node.keywords]))

        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name == 'format' and isinstance(node.func.value.ptr, str):
                args.insert(0, node.func.value.ptr)
                node.ptr = impl.ti_format(*args, **keywords)
            else:
                node.ptr = node.func.ptr(*args, **keywords)
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == 'print':
                node.ptr = impl.ti_print(*args, **keywords)
            elif func_name == 'min':
                node.ptr = ti_ops.ti_min(*args, **keywords)
            elif func_name == 'max':
                node.ptr = ti_ops.ti_max(*args, **keywords)
            elif func_name == 'int':
                node.ptr = impl.ti_int(*args, **keywords)
            elif func_name == 'float':
                node.ptr = impl.ti_float(*args, **keywords)
            elif func_name == 'any':
                node.ptr = ti_ops.ti_any(*args, **keywords)
            elif func_name == 'all':
                node.ptr = ti_ops.ti_all(*args, **keywords)
            else:
                node.ptr = node.func.ptr(*args, **keywords)
        else:
            node.ptr = node.func.ptr(*args, **keywords)

        return node.ptr

    @staticmethod
    def build_FunctionDef(ast_builder, ctx, node):
        args = node.args
        assert args.vararg is None
        assert args.kwonlyargs == []
        assert args.kw_defaults == []
        assert args.kwarg is None

        def transform_as_kernel():
            # Treat return type
            if node.returns is not None:
                kernel_arguments.decl_scalar_ret(ctx.func.return_type)

            for i, arg in enumerate(args.args):
                if isinstance(ctx.func.argument_annotations[i],
                              annotations.template):
                    continue
                elif isinstance(ctx.func.argument_annotations[i],
                                linalg.sparse_matrix_builder):
                    ctx.create_variable(arg.arg,
                                        kernel_arguments.decl_sparse_matrix())
                elif isinstance(ctx.func.argument_annotations[i],
                                annotations.any_arr):
                    ctx.create_variable(
                        arg.arg,
                        kernel_arguments.decl_any_arr_arg(
                            to_taichi_type(ctx.arg_features[i][0]),
                            ctx.arg_features[i][1], ctx.arg_features[i][2],
                            ctx.arg_features[i][3]))
                elif isinstance(ctx.func.argument_annotations[i], MatrixType):
                    ctx.global_vars[
                        arg.arg] = kernel_arguments.decl_matrix_arg(
                            ctx.func.argument_annotations[i])
                else:
                    ctx.global_vars[
                        arg.arg] = kernel_arguments.decl_scalar_arg(
                            ctx.func.argument_annotations[i])
            # remove original args
            node.args.args = []

        build_stmts(ast_builder, ctx, node.decorator_list)
        if ctx.is_kernel:  # ti.kernel
            for decorator in node.decorator_list:
                if decorator.ptr is kernel_impl.func:
                    raise TaichiSyntaxError(
                        "Function definition not allowed in 'ti.kernel'.")
            transform_as_kernel()

        else:  # ti.func
            for decorator in node.decorator_list:
                if decorator.ptr is kernel_impl.func:
                    raise TaichiSyntaxError(
                        "Function definition not allowed in 'ti.func'.")
            if impl.get_runtime().experimental_real_function:
                transform_as_kernel()
            else:
                len_args = len(args.args)
                len_default = len(args.defaults)
                len_provided = len(ctx.argument_data)
                len_minimum = len_args - len_default
                if len_args < len_provided or len_args - len_default > len_provided:
                    if len(args.defaults):
                        raise TaichiSyntaxError(
                            f"Function receives {len_minimum} to {len_args} argument(s) and {len_provided} provided."
                        )
                    else:
                        raise TaichiSyntaxError(
                            f"Function receives {len_args} argument(s) and {len_provided} provided."
                        )
                # Transform as force-inlined func
                default_start = len_provided - len_minimum
                ctx.argument_data = list(ctx.argument_data)
                for arg in args.defaults[default_start:]:
                    ctx.argument_data.append(build_stmt(ast_builder, ctx, arg))
                assert len(args.args) == len(ctx.argument_data)
                for i, (arg,
                        data) in enumerate(zip(args.args, ctx.argument_data)):
                    # Remove annotations because they are not used.
                    args.args[i].annotation = None
                    # Template arguments are passed by reference.
                    if isinstance(ctx.func.argument_annotations[i],
                                  annotations.template):
                        ctx.create_variable(ctx.func.argument_names[i], data)
                        continue
                    # Create a copy for non-template arguments,
                    # so that they are passed by value.
                    ctx.create_variable(arg.arg, impl.expr_init_func(data))

        with ctx.variable_scope_guard():
            build_stmts(ast_builder, ctx, node.body)

        return None

    @staticmethod
    def build_Return(ast_builder, ctx, node):
        if not impl.get_runtime().experimental_real_function:
            if ctx.is_in_non_static():
                raise TaichiSyntaxError(
                    "Return inside non-static if/for is not supported")
        build_stmt(ast_builder, ctx, node.value)
        if ctx.is_kernel or impl.get_runtime().experimental_real_function:
            # TODO: check if it's at the end of a kernel, throw TaichiSyntaxError if not
            if node.value is not None:
                if ctx.func.return_type is None:
                    raise TaichiSyntaxError(
                        f'A {"kernel" if ctx.is_kernel else "function"} '
                        'with a return value must be annotated '
                        'with a return type, e.g. def func() -> ti.f32')
                ast_builder.create_kernel_return(
                    ti_ops.cast(expr.Expr(node.value.ptr),
                                ctx.func.return_type).ptr)
                # For args[0], it is an ast.Attribute, because it loads the
                # attribute, |ptr|, of the expression |ret_expr|. Therefore we
                # only need to replace the object part, i.e. args[0].value
        else:
            ctx.return_data = node.value.ptr
        if not impl.get_runtime().experimental_real_function:
            ctx.returned = True
        return None

    @staticmethod
    def build_Module(ast_builder, ctx, node):
        with ctx.variable_scope_guard():
            # Do NOT use |build_stmts| which inserts 'del' statements to the
            # end and deletes parameters passed into the module
            for stmt in node.body:
                build_stmt(ast_builder, ctx, stmt)
        return None

    @staticmethod
    def build_Attribute(ast_builder, ctx, node):
        build_stmt(ast_builder, ctx, node.value)
        node.ptr = getattr(node.value.ptr, node.attr)
        return node.ptr

    @staticmethod
    def build_BinOp(ast_builder, ctx, node):
        build_stmt(ast_builder, ctx, node.left)
        build_stmt(ast_builder, ctx, node.right)
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
    def build_AugAssign(ast_builder, ctx, node):
        build_stmt(ast_builder, ctx, node.target)
        build_stmt(ast_builder, ctx, node.value)
        node.ptr = node.target.ptr.augassign(node.value.ptr,
                                             type(node.op).__name__)
        return node.ptr

    @staticmethod
    def build_UnaryOp(ast_builder, ctx, node):
        build_stmt(ast_builder, ctx, node.operand)
        op = {
            ast.UAdd: lambda l: l,
            ast.USub: lambda l: -l,
            ast.Not: ti_ops.logical_not,
            ast.Invert: lambda l: ~l,
        }.get(type(node.op))
        node.ptr = op(node.operand.ptr)
        return node.ptr

    @staticmethod
    def build_short_circuit_and(ast_builder, operands):
        if len(operands) == 1:
            return operands[0].ptr

        val = impl.expr_init(None)
        lhs = operands[0].ptr
        impl.begin_frontend_if(ast_builder, lhs)

        ast_builder.begin_frontend_if_true()
        rhs = ASTTransformer.build_short_circuit_and(ast_builder, operands[1:])
        val.assign(rhs)
        ast_builder.pop_scope()

        ast_builder.begin_frontend_if_false()
        val.assign(0)
        ast_builder.pop_scope()

        return val

    @staticmethod
    def build_short_circuit_or(ast_builder, operands):
        if len(operands) == 1:
            return operands[0].ptr

        val = impl.expr_init(None)
        lhs = operands[0].ptr
        impl.begin_frontend_if(ast_builder, lhs)

        ast_builder.begin_frontend_if_true()
        val.assign(1)
        ast_builder.pop_scope()

        ast_builder.begin_frontend_if_false()
        rhs = ASTTransformer.build_short_circuit_or(ast_builder, operands[1:])
        val.assign(rhs)
        ast_builder.pop_scope()

        return val

    @staticmethod
    def build_normal_bool_op(op):
        def inner(ast_builder, operands):
            result = op(operands[0].ptr, operands[1].ptr)
            for i in range(2, len(operands)):
                result = op(result, operands[i].ptr)
            return result

        return inner

    @staticmethod
    def build_static_short_circuit_and(ast_builder, operands):
        for operand in operands:
            if not operand.ptr:
                return operand.ptr
        return operands[-1].ptr

    @staticmethod
    def build_static_short_circuit_or(ast_builder, operands):
        for operand in operands:
            if operand.ptr:
                return operand.ptr
        return operands[-1].ptr

    @staticmethod
    def build_BoolOp(ast_builder, ctx, node):
        build_stmts(ast_builder, ctx, node.values)
        if ctx.is_in_static_scope:
            ops = {
                ast.And: ASTTransformer.build_static_short_circuit_and,
                ast.Or: ASTTransformer.build_static_short_circuit_or,
            }
        elif impl.get_runtime().short_circuit_operators:
            ops = {
                ast.And: ASTTransformer.build_short_circuit_and,
                ast.Or: ASTTransformer.build_short_circuit_or,
            }
        else:
            ops = {
                ast.And:
                ASTTransformer.build_normal_bool_op(ti_ops.logical_and),
                ast.Or: ASTTransformer.build_normal_bool_op(ti_ops.logical_or),
            }
        op = ops.get(type(node.op))
        node.ptr = op(ast_builder, node.values)
        return node.ptr

    @staticmethod
    def build_Compare(ast_builder, ctx, node):
        build_stmt(ast_builder, ctx, node.left)
        build_stmts(ast_builder, ctx, node.comparators)
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
        if ctx.is_in_static_scope:
            ops = {**ops, **ops_static}
        operands = [node.left.ptr
                    ] + [comparator.ptr for comparator in node.comparators]
        val = True
        for i, node_op in enumerate(node.ops):
            l = operands[i]
            r = operands[i + 1]
            op = ops.get(type(node_op))
            if op is None:
                if type(node_op) in ops_static:
                    raise TaichiSyntaxError(
                        f'"{type(node_op).__name__}" is only supported inside `ti.static`.'
                    )
                else:
                    raise TaichiSyntaxError(
                        f'"{type(node_op).__name__}" is not supported in Taichi kernels.'
                    )
            val = ti_ops.logical_and(val, op(l, r))
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
    def build_static_for(ast_builder, ctx, node, is_grouped):
        if is_grouped:
            assert len(node.iter.args[0].args) == 1
            ndrange_arg = build_stmt(ast_builder, ctx,
                                     node.iter.args[0].args[0])
            if not isinstance(ndrange_arg, ndrange):
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
                    build_stmts(ast_builder, ctx, node.body)
                    status = ctx.loop_status()
                    if status == LoopStatus.Break:
                        break
                    elif status == LoopStatus.Continue:
                        ctx.set_loop_status(LoopStatus.Normal)
        else:
            build_stmt(ast_builder, ctx, node.iter)
            targets = ASTTransformer.get_for_loop_targets(node)
            for target_values in node.iter.ptr:
                if not isinstance(
                        target_values,
                        collections.abc.Sequence) or len(targets) == 1:
                    target_values = [target_values]
                with ctx.variable_scope_guard():
                    for target, target_value in zip(targets, target_values):
                        ctx.create_variable(target, target_value)
                    build_stmts(ast_builder, ctx, node.body)
                    status = ctx.loop_status()
                    if status == LoopStatus.Break:
                        break
                    elif status == LoopStatus.Continue:
                        ctx.set_loop_status(LoopStatus.Normal)
        return None

    @staticmethod
    def build_range_for(ast_builder, ctx, node):
        with ctx.variable_scope_guard():
            loop_name = node.target.id
            ctx.check_loop_var(loop_name)
            loop_var = expr.Expr(ast_builder.make_id_expr(''))
            ctx.create_variable(loop_name, loop_var)
            if len(node.iter.args) not in [1, 2]:
                raise TaichiSyntaxError(
                    f"Range should have 1 or 2 arguments, found {len(node.iter.args)}"
                )
            if len(node.iter.args) == 2:
                begin = ti_ops.cast(
                    expr.Expr(build_stmt(ast_builder, ctx, node.iter.args[0])),
                    primitive_types.i32)
                end = ti_ops.cast(
                    expr.Expr(build_stmt(ast_builder, ctx, node.iter.args[1])),
                    primitive_types.i32)
            else:
                begin = ti_ops.cast(expr.Expr(0), primitive_types.i32)
                end = ti_ops.cast(
                    expr.Expr(build_stmt(ast_builder, ctx, node.iter.args[0])),
                    primitive_types.i32)
            ast_builder.begin_frontend_range_for(loop_var.ptr, begin.ptr,
                                                 end.ptr)
            build_stmts(ast_builder, ctx, node.body)
            ast_builder.end_frontend_range_for()
        return None

    @staticmethod
    def build_ndrange_for(ast_builder, ctx, node):
        with ctx.variable_scope_guard():
            ndrange_var = impl.expr_init(
                build_stmt(ast_builder, ctx, node.iter))
            ndrange_begin = ti_ops.cast(expr.Expr(0), primitive_types.i32)
            ndrange_end = ti_ops.cast(
                expr.Expr(impl.subscript(ndrange_var.acc_dimensions, 0)),
                primitive_types.i32)
            ndrange_loop_var = expr.Expr(ast_builder.make_id_expr(''))
            ast_builder.begin_frontend_range_for(ndrange_loop_var.ptr,
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
                    I.assign(I -
                             target_tmp * ndrange_var.acc_dimensions[i + 1])
            build_stmts(ast_builder, ctx, node.body)
            ast_builder.end_frontend_range_for()
        return None

    @staticmethod
    def build_grouped_ndrange_for(ast_builder, ctx, node):
        with ctx.variable_scope_guard():
            ndrange_var = impl.expr_init(
                build_stmt(ast_builder, ctx, node.iter.args[0]))
            ndrange_begin = ti_ops.cast(expr.Expr(0), primitive_types.i32)
            ndrange_end = ti_ops.cast(
                expr.Expr(impl.subscript(ndrange_var.acc_dimensions, 0)),
                primitive_types.i32)
            ndrange_loop_var = expr.Expr(ast_builder.make_id_expr(''))
            ast_builder.begin_frontend_range_for(ndrange_loop_var.ptr,
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
                impl.subscript(target_var,
                               i).assign(target_tmp + ndrange_var.bounds[i][0])
                if i + 1 < len(ndrange_var.dimensions):
                    I.assign(I -
                             target_tmp * ndrange_var.acc_dimensions[i + 1])
            build_stmts(ast_builder, ctx, node.body)
            ast_builder.end_frontend_range_for()
        return None

    @staticmethod
    def build_struct_for(ast_builder, ctx, node, is_grouped):
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
                loop_var = build_stmt(ast_builder, ctx, node.iter)
                loop_indices = expr.make_var_list(size=len(loop_var.shape))
                expr_group = expr.make_expr_group(loop_indices)
                impl.begin_frontend_struct_for(ast_builder, expr_group,
                                               loop_var)
                ctx.create_variable(
                    target, matrix.Vector(loop_indices,
                                          dt=primitive_types.i32))
                build_stmts(ast_builder, ctx, node.body)
                ast_builder.end_frontend_struct_for()
            else:
                _vars = []
                for name in targets:
                    var = expr.Expr(ast_builder.make_id_expr(""))
                    _vars.append(var)
                    ctx.create_variable(name, var)
                loop_var = node.iter.ptr
                expr_group = expr.make_expr_group(*_vars)
                impl.begin_frontend_struct_for(ast_builder, expr_group,
                                               loop_var)
                build_stmts(ast_builder, ctx, node.body)
                ast_builder.end_frontend_struct_for()
        return None

    @staticmethod
    def build_mesh_for(ast_builder, ctx, node):
        targets = ASTTransformer.get_for_loop_targets(node)
        if len(targets) != 1:
            raise TaichiSyntaxError(
                "Mesh for should have 1 loop target, found {len(targets)}")
        target = targets[0]

        with ctx.variable_scope_guard():
            var = expr.Expr(ast_builder.make_id_expr(""))
            ctx.mesh = node.iter.ptr.mesh
            assert isinstance(ctx.mesh, impl.MeshInstance)
            mesh_idx = mesh.MeshElementFieldProxy(ctx.mesh,
                                                  node.iter.ptr._type, var.ptr)
            ctx.create_variable(target, mesh_idx)
            ast_builder.begin_frontend_mesh_for(mesh_idx.ptr,
                                                ctx.mesh.mesh_ptr,
                                                node.iter.ptr._type)
            build_stmts(ast_builder, ctx, node.body)
            ctx.mesh = None
            ast_builder.end_frontend_mesh_for()
        return None

    @staticmethod
    def build_nested_mesh_for(ast_builder, ctx, node):
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
            loop_var = expr.Expr(ast_builder.make_id_expr(''))
            ctx.create_variable(loop_name, loop_var)
            begin = expr.Expr(0)
            end = node.iter.ptr.size
            ast_builder.begin_frontend_range_for(loop_var.ptr, begin.ptr,
                                                 end.ptr)
            entry_expr = ast_builder.get_relation_access(
                ctx.mesh.mesh_ptr, node.iter.ptr.from_index.ptr,
                node.iter.ptr.to_element_type, loop_var.ptr)
            entry_expr.type_check()
            mesh_idx = mesh.MeshElementFieldProxy(
                ctx.mesh, node.iter.ptr.to_element_type, entry_expr)
            ctx.create_variable(target, mesh_idx)
            build_stmts(ast_builder, ctx, node.body)
            ast_builder.end_frontend_range_for()

        return None

    @staticmethod
    def build_For(ast_builder, ctx, node):
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
                    ast_builder, ctx, node, double_decorator == 'grouped')
        with ctx.loop_scope_guard():
            if decorator == 'ndrange':
                if double_decorator != '':
                    raise TaichiSyntaxError(
                        "No decorator is allowed inside 'ti.ndrange")
                return ASTTransformer.build_ndrange_for(ast_builder, ctx, node)
            if decorator == 'grouped':
                if double_decorator == 'static':
                    raise TaichiSyntaxError(
                        "'ti.static' is not allowed inside 'ti.grouped'")
                elif double_decorator == 'ndrange':
                    return ASTTransformer.build_grouped_ndrange_for(
                        ast_builder, ctx, node)
                elif double_decorator == 'grouped':
                    raise TaichiSyntaxError("'ti.grouped' cannot be nested")
                else:
                    return ASTTransformer.build_struct_for(ast_builder,
                                                           ctx,
                                                           node,
                                                           is_grouped=True)
            elif isinstance(node.iter, ast.Call) and isinstance(
                    node.iter.func, ast.Name) and node.iter.func.id == 'range':
                return ASTTransformer.build_range_for(ast_builder, ctx, node)
            else:
                build_stmt(ast_builder, ctx, node.iter)
                if isinstance(node.iter.ptr, mesh.MeshElementField):
                    if not _ti_core.is_extension_supported(
                            impl.default_cfg().arch, _ti_core.Extension.mesh):
                        raise Exception(
                            'Backend ' + str(impl.default_cfg().arch) +
                            ' doesn\'t support MeshTaichi extension')
                    return ASTTransformer.build_mesh_for(
                        ast_builder, ctx, node)
                if isinstance(node.iter.ptr, mesh.MeshRelationAccessProxy):
                    return ASTTransformer.build_nested_mesh_for(
                        ast_builder, ctx, node)
                # Struct for
                return ASTTransformer.build_struct_for(ast_builder,
                                                       ctx,
                                                       node,
                                                       is_grouped=False)

    @staticmethod
    def build_While(ast_builder, ctx, node):
        if node.orelse:
            raise TaichiSyntaxError(
                "'else' clause for 'while' not supported in Taichi kernels")

        with ctx.loop_scope_guard():
            ast_builder.begin_frontend_while(expr.Expr(1).ptr)
            while_cond = build_stmt(ast_builder, ctx, node.test)
            impl.begin_frontend_if(ast_builder, while_cond)
            ast_builder.begin_frontend_if_true()
            ast_builder.pop_scope()
            ast_builder.begin_frontend_if_false()
            ast_builder.insert_break_stmt()
            ast_builder.pop_scope()
            build_stmts(ast_builder, ctx, node.body)
            ast_builder.pop_scope()
        return None

    @staticmethod
    def build_If(ast_builder, ctx, node):
        build_stmt(ast_builder, ctx, node.test)
        is_static_if = (ASTTransformer.get_decorator(ctx,
                                                     node.test) == "static")

        if is_static_if:
            if node.test.ptr:
                build_stmts(ast_builder, ctx, node.body)
            else:
                build_stmts(ast_builder, ctx, node.orelse)
            return node

        with ctx.non_static_scope_guard():
            impl.begin_frontend_if(ast_builder, node.test.ptr)
            ast_builder.begin_frontend_if_true()
            build_stmts(ast_builder, ctx, node.body)
            ast_builder.pop_scope()
            ast_builder.begin_frontend_if_false()
            build_stmts(ast_builder, ctx, node.orelse)
            ast_builder.pop_scope()
        return None

    @staticmethod
    def build_Expr(ast_builder, ctx, node):
        if not isinstance(
                node.value,
                ast.Call) or not impl.get_runtime().experimental_real_function:
            build_stmt(ast_builder, ctx, node.value)
            return None

        args = [build_stmt(ast_builder, ctx, node.value.func)] + [
            arg.ptr for arg in build_stmts(ast_builder, ctx, node.value.args)
        ]
        impl.insert_expr_stmt_if_ti_func(ast_builder, *args)

        return None

    @staticmethod
    def build_IfExp(ast_builder, ctx, node):
        build_stmt(ast_builder, ctx, node.test)
        build_stmt(ast_builder, ctx, node.body)
        build_stmt(ast_builder, ctx, node.orelse)

        if is_taichi_class(node.test.ptr) or is_taichi_class(
                node.body.ptr) or is_taichi_class(node.orelse.ptr):
            node.ptr = ti_ops.select(node.test.ptr, node.body.ptr,
                                     node.orelse.ptr)
            return node.ptr

        is_static_if = (ASTTransformer.get_decorator(ctx,
                                                     node.test) == "static")

        if is_static_if:
            if node.test.ptr:
                node.ptr = build_stmt(ast_builder, ctx, node.body)
            else:
                node.ptr = build_stmt(ast_builder, ctx, node.orelse)
            return node.ptr

        val = impl.expr_init(None)

        impl.begin_frontend_if(ast_builder, node.test.ptr)
        ast_builder.begin_frontend_if_true()
        val.assign(node.body.ptr)
        ast_builder.pop_scope()
        ast_builder.begin_frontend_if_false()
        val.assign(node.orelse.ptr)
        ast_builder.pop_scope()

        node.ptr = val
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
    def _handle_string_mod_args(ast_builder, ctx, node):
        msg = build_stmt(ast_builder, ctx, node.left)
        args = build_stmt(ast_builder, ctx, node.right)
        if not isinstance(args, collections.abc.Sequence):
            args = (args, )
        return msg, args

    @staticmethod
    def build_Assert(ast_builder, ctx, node):
        extra_args = []
        if node.msg is not None:
            if isinstance(node.msg, ast.Constant):
                msg = node.msg.value
            elif isinstance(node.msg, ast.Str):
                msg = node.msg.s
            elif ASTTransformer._is_string_mod_args(node.msg):
                msg, extra_args = ASTTransformer._handle_string_mod_args(
                    ast_builder, ctx, node.msg)
            else:
                raise ValueError(
                    f"assert info must be constant, not {ast.dump(node.msg)}")
        else:
            msg = astor.to_source(node.test)
        test = build_stmt(ast_builder, ctx, node.test)
        impl.ti_assert(test, msg.strip(), extra_args)
        return None

    @staticmethod
    def build_Break(ast_builder, ctx, node):
        if ctx.is_in_static_for():
            ctx.set_loop_status(LoopStatus.Break)
        else:
            ast_builder.insert_break_stmt()
        return None

    @staticmethod
    def build_Continue(ast_builder, ctx, node):
        if ctx.is_in_static_for():
            ctx.set_loop_status(LoopStatus.Continue)
        else:
            ast_builder.insert_continue_stmt()
        return None

    @staticmethod
    def build_Pass(ast_builder, ctx, node):
        return None

    @staticmethod
    def build_Raise(ast_builder, ctx, node):
        raise build_stmt(ast_builder, ctx, node.exc)


build_stmt = ASTTransformer()


def build_stmts(ast_builder, ctx, stmts):
    with ctx.variable_scope_guard():
        for stmt in stmts:
            if ctx.returned or ctx.loop_status() != LoopStatus.Normal:
                break
            else:
                build_stmt(ast_builder, ctx, stmt)
    return stmts
