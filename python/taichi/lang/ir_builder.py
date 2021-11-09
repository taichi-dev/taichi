import ast
import collections.abc
import warnings
from collections import ChainMap

import astor
from taichi.lang.ast.symbol_resolver import ASTResolver
from taichi.lang.ast_builder_utils import *
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.util import to_taichi_type

import taichi as ti


class IRBuilder(Builder):
    @staticmethod
    def build_Name(ctx, node):
        node.ptr = ctx.get_var_by_name(node.id)
        return node

    @staticmethod
    def build_Assign(ctx, node):
        node.value = build_stmt(ctx, node.value)
        node.targets = build_stmts(ctx, node.targets)

        is_static_assign = isinstance(
            node.value, ast.Call) and ASTResolver.resolve_to(
                node.value.func, ti.static, globals())

        # Keep all generated assign statements and compose single one at last.
        # The variable is introduced to support chained assignments.
        # Ref https://github.com/taichi-dev/taichi/issues/2659.
        for node_target in node.targets:
            IRBuilder.build_assign_unpack(ctx, node_target, node.value.ptr,
                                          is_static_assign)
        return node

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
            return IRBuilder.build_assign_basic(ctx, node_target, values,
                                                is_static_assign)
        targets = node_target.elts
        tmp_tuple = values if is_static_assign else ti.expr_init_list(
            values, len(targets))

        for i, target in enumerate(targets):
            IRBuilder.build_assign_basic(ctx, target, tmp_tuple[i],
                                         is_static_assign)

    @staticmethod
    def build_assign_basic(ctx, target, value, is_static_assign):
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
            var = ti.expr_init(value)
            ctx.create_variable(target.id, var)
        else:
            var = target.ptr
            var.assign(value)
        return var

    @staticmethod
    def build_NamedExpr(ctx, node):
        node.value = build_stmt(ctx, node.value)
        node.target = build_stmt(ctx, node.target)
        is_static_assign = isinstance(
            node.value, ast.Call) and ASTResolver.resolve_to(
                node.value.func, ti.static, globals())
        node.ptr = IRBuilder.build_assign_basic(ctx, node.target,
                                                node.value.ptr,
                                                is_static_assign)
        return node

    @staticmethod
    def is_tuple(node):
        if isinstance(node, ast.Tuple):
            return True
        if isinstance(node, ast.Index) and isinstance(node.value, ast.Tuple):
            return True
        return False

    @staticmethod
    def build_Subscript(ctx, node):
        node.value = build_stmt(ctx, node.value)
        node.slice = build_stmt(ctx, node.slice)
        if not IRBuilder.is_tuple(node.slice):
            node.slice.ptr = [node.slice.ptr]
        node.ptr = ti.subscript(node.value.ptr, *node.slice.ptr)
        return node

    @staticmethod
    def build_Tuple(ctx, node):
        node.elts = build_stmts(ctx, node.elts)
        node.ptr = tuple(elt.ptr for elt in node.elts)
        return node

    @staticmethod
    def build_List(ctx, node):
        node.elts = build_stmts(ctx, node.elts)
        node.ptr = [elt.ptr for elt in node.elts]
        return node

    @staticmethod
    def build_Dict(ctx, node):
        dic = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                dic.update(build_stmt(ctx, value).ptr)
            else:
                dic[build_stmt(ctx, key).ptr] = build_stmt(ctx, value).ptr
        node.ptr = dic
        return node

    @staticmethod
    def process_listcomp(ctx, node, result):
        result.append(build_stmt(ctx, node.elt).ptr)

    @staticmethod
    def process_dictcomp(ctx, node, result):
        key = build_stmt(ctx, node.key).ptr
        value = build_stmt(ctx, node.value).ptr
        result[key] = value

    @staticmethod
    def process_generators(ctx, node, now_comp, func, result):
        if now_comp >= len(node.generators):
            return func(ctx, node, result)
        target = node.generators[now_comp].target = build_stmt(
            ctx, node.generators[now_comp].target)
        iter = node.generators[now_comp].iter = build_stmt(
            ctx, node.generators[now_comp].iter)
        for value in iter.ptr:
            with ctx.variable_scope_guard():
                IRBuilder.build_assign_unpack(ctx, target, value, True)
                node.generators[now_comp].ifs = build_stmts(
                    ctx, node.generators[now_comp].ifs)
                IRBuilder.process_ifs(ctx, node, now_comp, 0, func, result)

    @staticmethod
    def process_ifs(ctx, node, now_comp, now_if, func, result):
        if now_if >= len(node.generators[now_comp].ifs):
            return IRBuilder.process_generators(ctx, node, now_comp + 1, func,
                                                result)
        cond = node.generators[now_comp].ifs[now_if].ptr
        if cond:
            IRBuilder.process_ifs(ctx, node, now_comp, now_if + 1, func,
                                  result)

    @staticmethod
    def build_comprehension(ctx, node):
        node.target = build_stmt(ctx, node.target)
        node.iter = build_stmt(ctx, node.iter)
        node.ifs = build_stmts(ctx, node.ifs)
        return node

    @staticmethod
    def build_ListComp(ctx, node):
        result = []
        IRBuilder.process_generators(ctx, node, 0, IRBuilder.process_listcomp,
                                     result)
        node.ptr = result
        return node

    @staticmethod
    def build_DictComp(ctx, node):
        result = {}
        IRBuilder.process_generators(ctx, node, 0, IRBuilder.process_dictcomp,
                                     result)
        node.ptr = result
        return node

    @staticmethod
    def build_Index(ctx, node):
        node.value = build_stmt(ctx, node.value)
        node.ptr = node.value.ptr
        return node

    @staticmethod
    def build_Constant(ctx, node):
        node.ptr = node.value
        return node

    @staticmethod
    def build_Num(ctx, node):
        node.ptr = node.n
        return node

    @staticmethod
    def build_Str(ctx, node):
        node.ptr = node.s
        return node

    @staticmethod
    def build_Bytes(ctx, node):
        node.ptr = node.s
        return node

    @staticmethod
    def build_NameConstant(ctx, node):
        node.ptr = node.value
        return node

    @staticmethod
    def build_keyword(ctx, node):
        node.value = build_stmt(ctx, node.value)
        if node.arg is None:
            node.ptr = node.value.ptr
        else:
            node.ptr = {node.arg: node.value.ptr}
        return node

    @staticmethod
    def build_Starred(ctx, node):
        node.value = build_stmt(ctx, node.value)
        node.ptr = node.value.ptr
        return node

    @staticmethod
    def build_JoinedStr(ctx, node):
        str_spec = ''
        args = []
        for sub_node in node.values:
            if isinstance(sub_node, ast.FormattedValue):
                str_spec += '{}'
                args.append(build_stmt(ctx, sub_node.value).ptr)
            elif isinstance(sub_node, ast.Constant):
                str_spec += sub_node.value
            else:
                raise TaichiSyntaxError("Invalid value for fstring.")

        args.insert(0, str_spec)
        node.ptr = ti.ti_format(*args)
        return node

    @staticmethod
    def build_Call(ctx, node):
        node.func = build_stmt(ctx, node.func)
        node.args = build_stmts(ctx, node.args)
        node.keywords = build_stmts(ctx, node.keywords)
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
                node.ptr = ti.ti_format(*args, **keywords)
            else:
                node.ptr = node.func.ptr(*args, **keywords)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == 'print':
                node.ptr = ti.ti_print(*args, **keywords)
            elif func_name == 'min':
                node.ptr = ti.ti_min(*args, **keywords)
            elif func_name == 'max':
                node.ptr = ti.ti_max(*args, **keywords)
            elif func_name == 'int':
                node.ptr = ti.ti_int(*args, **keywords)
            elif func_name == 'float':
                node.ptr = ti.ti_float(*args, **keywords)
            elif func_name == 'any':
                node.ptr = ti.ti_any(*args, **keywords)
            elif func_name == 'all':
                node.ptr = ti.ti_all(*args, **keywords)
            else:
                node.ptr = node.func.ptr(*args, **keywords)
        else:
            node.ptr = node.func.ptr(*args, **keywords)

        return node

    @staticmethod
    def build_FunctionDef(ctx, node):
        args = node.args
        assert args.vararg is None
        assert args.kwonlyargs == []
        assert args.kw_defaults == []
        assert args.kwarg is None

        arg_decls = []

        def transform_as_kernel():
            # Treat return type
            if node.returns is not None:
                ti.lang.kernel_arguments.decl_scalar_ret(ctx.func.return_type)

            for i, arg in enumerate(args.args):
                if isinstance(ctx.func.argument_annotations[i], ti.template):
                    continue
                elif isinstance(ctx.func.argument_annotations[i],
                                ti.linalg.sparse_matrix_builder):
                    ctx.create_variable(
                        arg.arg, ti.lang.kernel_arguments.decl_sparse_matrix())
                elif isinstance(ctx.func.argument_annotations[i], ti.any_arr):
                    ctx.create_variable(
                        arg.arg,
                        ti.lang.kernel_arguments.decl_any_arr_arg(
                            to_taichi_type(ctx.arg_features[i][0]),
                            ctx.arg_features[i][1], ctx.arg_features[i][2],
                            ctx.arg_features[i][3]))
                else:
                    ctx.create_variable(
                        arg.arg,
                        ti.lang.kernel_arguments.decl_scalar_arg(
                            ctx.func.argument_annotations[i]))
            # remove original args
            node.args.args = []

        if ctx.is_kernel:  # ti.kernel
            for decorator in node.decorator_list:
                if ASTResolver.resolve_to(decorator, ti.func, globals()):
                    raise TaichiSyntaxError(
                        "Function definition not allowed in 'ti.kernel'.")
            transform_as_kernel()

        else:  # ti.func
            for decorator in node.decorator_list:
                if ASTResolver.resolve_to(decorator, ti.func, globals()):
                    raise TaichiSyntaxError(
                        "Function definition not allowed in 'ti.func'.")
            # if impl.get_runtime().experimental_real_function:
            #     transform_as_kernel()
            if False:
                pass
            else:
                if len(args.args) != len(ctx.argument_data):
                    raise TaichiSyntaxError("Function argument of ")
                # Transform as force-inlined func
                for i, (arg,
                        data) in enumerate(zip(args.args, ctx.argument_data)):
                    # Remove annotations because they are not used.
                    args.args[i].annotation = None
                    # Template arguments are passed by reference.
                    if isinstance(ctx.func.argument_annotations[i],
                                  ti.template):
                        ctx.create_variable(ctx.func.argument_names[i], data)
                        continue
                    # Create a copy for non-template arguments,
                    # so that they are passed by value.
                    ctx.create_variable(arg.arg, ti.expr_init_func(data))

        with ctx.variable_scope_guard():
            build_stmts(ctx, node.body)

        return node

    @staticmethod
    def build_Return(ctx, node):
        node.value = build_stmt(ctx, node.value)
        if ctx.is_kernel:
            # TODO: check if it's at the end of a kernel, throw TaichiSyntaxError if not
            if node.value is not None:
                if ctx.func.return_type is None:
                    raise TaichiSyntaxError(
                        f'A {"kernel" if ctx.is_kernel else "function"} '
                        'with a return value must be annotated '
                        'with a return type, e.g. def func() -> ti.f32')
                ti.core.create_kernel_return(
                    ti.cast(ti.Expr(node.value.ptr), ctx.func.return_type).ptr)
                # For args[0], it is an ast.Attribute, because it loads the
                # attribute, |ptr|, of the expression |ret_expr|. Therefore we
                # only need to replace the object part, i.e. args[0].value
        else:
            ctx.return_data = node.value.ptr
        return node

    @staticmethod
    def build_Module(ctx, node):
        with ctx.variable_scope_guard():
            # Do NOT use |build_stmts| which inserts 'del' statements to the
            # end and deletes parameters passed into the module
            node.body = [build_stmt(ctx, stmt) for stmt in list(node.body)]
        return node

    @staticmethod
    def build_Attribute(ctx, node):
        node.value = build_stmt(ctx, node.value)
        node.ptr = getattr(node.value.ptr, node.attr)
        return node

    @staticmethod
    def build_BinOp(ctx, node):
        node.left = build_stmt(ctx, node.left)
        node.right = build_stmt(ctx, node.right)
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
        return node

    @staticmethod
    def build_AugAssign(ctx, node):
        node.target = build_stmt(ctx, node.target)
        node.value = build_stmt(ctx, node.value)
        node.ptr = node.target.ptr.augassign(node.value.ptr,
                                             type(node.op).__name__)
        return node

    @staticmethod
    def build_UnaryOp(ctx, node):
        node.operand = build_stmt(ctx, node.operand)
        op = {
            ast.UAdd: lambda l: l,
            ast.USub: lambda l: -l,
            ast.Not: lambda l: not l,
            ast.Invert: lambda l: ~l,
        }.get(type(node.op))
        node.ptr = op(node.operand.ptr)
        return node

    @staticmethod
    def build_BoolOp(ctx, node):
        node.values = build_stmts(ctx, node.values)
        op = {
            ast.And: lambda l, r: l and r,
            ast.Or: lambda l, r: l or r,
        }.get(type(node.op))
        result = op(node.values[0].ptr, node.values[1].ptr)
        for i in range(2, len(node.values)):
            result = op(result, node.values[i].ptr)
        node.ptr = result
        return node

    @staticmethod
    def build_Compare(ctx, node):
        node.left = build_stmt(ctx, node.left)
        node.comparators = build_stmts(ctx, node.comparators)
        op_dict = {
            ast.Eq: "Eq",
            ast.NotEq: "NotEq",
            ast.Lt: "Lt",
            ast.LtE: "LtE",
            ast.Gt: "Gt",
            ast.GtE: "GtE",
        }
        operands = [node.left.ptr
                    ] + [comparator.ptr for comparator in node.comparators]
        ops = []
        for node_op in node.ops:
            op = op_dict.get(type(node_op))
            if op is None:
                raise TaichiSyntaxError(
                    f'"{type(node_op).__name__}" is not supported in Taichi kernels.'
                )
            ops.append(op)
        node.ptr = ti.chain_compare(operands, ops)
        return node

    @staticmethod
    def get_decorator(ctx, node):
        if not isinstance(node, ast.Call):
            return ''
        for wanted, name in [
            (ti.static, 'static'),
            (ti.grouped, 'grouped'),
            (ti.ndrange, 'ndrange'),
        ]:
            if ASTResolver.resolve_to(node.func, wanted, ctx.globals):
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
        else:
            assert isinstance(node.target, ast.Tuple)
            return [name.id for name in node.target.elts]

    @staticmethod
    def build_static_for(ctx, node, is_grouped):
        ctx.set_static_loop()
        if is_grouped:
            assert len(node.iter.args[0].args) == 1
            ndrange_arg = build_stmt(ctx, node.iter.args[0].args[0]).ptr
            if not isinstance(ndrange_arg, ti.ndrange):
                raise TaichiSyntaxError(
                    "Only 'ti.ndrange' is allowed in 'ti.static(ti.grouped(...))'."
                )
            targets = IRBuilder.get_for_loop_targets(node)
            if len(targets) != 1:
                raise TaichiSyntaxError(
                    f"Group for should have 1 loop target, found {len(targets)}"
                )
            target = targets[0]
            for value in ndrange_arg:
                with ctx.variable_scope_guard():
                    ctx.create_variable(target, value)
                    node.body = build_stmts(ctx, node.body)
                    status = ctx.loop_status()
                    if status == LoopStatus.Break:
                        break
                    elif status == LoopStatus.Continue:
                        ctx.set_loop_status(LoopStatus.Normal)
        else:
            node.iter = build_stmt(ctx, node.iter)
            targets = IRBuilder.get_for_loop_targets(node)
            for target_values in node.iter.ptr:
                if not isinstance(target_values, collections.abc.Sequence):
                    target_values = [target_values]
                with ctx.variable_scope_guard():
                    for target, target_value in zip(targets, target_values):
                        ctx.create_variable(target, target_value)
                    node.body = build_stmts(ctx, node.body)
                    status = ctx.loop_status()
                    if status == LoopStatus.Break:
                        break
                    elif status == LoopStatus.Continue:
                        ctx.set_loop_status(LoopStatus.Normal)
        return node

    @staticmethod
    def build_range_for(ctx, node):
        with ctx.variable_scope_guard():
            loop_name = node.target.id
            ctx.check_loop_var(loop_name)
            loop_var = ti.Expr(ti.core.make_id_expr(''))
            ctx.create_variable(loop_name, loop_var)
            if len(node.iter.args) not in [1, 2]:
                raise TaichiSyntaxError(
                    f"Range should have 1 or 2 arguments, found {len(node.iter.args)}"
                )
            if len(node.iter.args) == 2:
                begin = ti.cast(
                    ti.Expr(build_stmt(ctx, node.iter.args[0]).ptr), ti.i32)
                end = ti.cast(ti.Expr(build_stmt(ctx, node.iter.args[1]).ptr),
                              ti.i32)
            else:
                begin = ti.cast(ti.Expr(0), ti.i32)
                end = ti.cast(ti.Expr(build_stmt(ctx, node.iter.args[0]).ptr),
                              ti.i32)
            ti.core.begin_frontend_range_for(loop_var.ptr, begin.ptr, end.ptr)
            node.body = build_stmts(ctx, node.body)
            ti.core.end_frontend_range_for()
        return node

    @staticmethod
    def build_ndrange_for(ctx, node):
        with ctx.variable_scope_guard():
            ndrange_var = ti.expr_init(build_stmt(ctx, node.iter).ptr)
            ndrange_begin = ti.cast(ti.Expr(0), ti.i32)
            ndrange_end = ti.cast(
                ti.Expr(ti.subscript(ndrange_var.acc_dimensions, 0)), ti.i32)
            ndrange_loop_var = ti.Expr(ti.core.make_id_expr(''))
            ti.core.begin_frontend_range_for(ndrange_loop_var.ptr,
                                             ndrange_begin.ptr,
                                             ndrange_end.ptr)
            I = ti.expr_init(ndrange_loop_var)
            targets = IRBuilder.get_for_loop_targets(node)
            for i in range(len(targets)):
                if i + 1 < len(targets):
                    target_tmp = ti.expr_init(
                        I // ndrange_var.acc_dimensions[i + 1])
                else:
                    target_tmp = ti.expr_init(I)
                ctx.create_variable(
                    targets[i],
                    ti.expr_init(
                        target_tmp +
                        ti.subscript(ti.subscript(ndrange_var.bounds, i), 0)))
                if i + 1 < len(targets):
                    I.assign(I -
                             target_tmp * ndrange_var.acc_dimensions[i + 1])
            node.body = build_stmts(ctx, node.body)
            ti.core.end_frontend_range_for()
        return node

    @staticmethod
    def build_grouped_ndrange_for(ctx, node):
        with ctx.variable_scope_guard():
            ndrange_var = ti.expr_init(build_stmt(ctx, node.iter.args[0]).ptr)
            ndrange_begin = ti.cast(ti.Expr(0), ti.i32)
            ndrange_end = ti.cast(
                ti.Expr(ti.subscript(ndrange_var.acc_dimensions, 0)), ti.i32)
            ndrange_loop_var = ti.Expr(ti.core.make_id_expr(''))
            ti.core.begin_frontend_range_for(ndrange_loop_var.ptr,
                                             ndrange_begin.ptr,
                                             ndrange_end.ptr)

            targets = IRBuilder.get_for_loop_targets(node)
            if len(targets) != 1:
                raise TaichiSyntaxError(
                    f"Group for should have 1 loop target, found {len(targets)}"
                )
            target = targets[0]
            target_var = ti.expr_init(
                ti.Vector([0] * len(ndrange_var.dimensions), dt=ti.i32))
            ctx.create_variable(target, target_var)
            I = ti.expr_init(ndrange_loop_var)
            for i in range(len(ndrange_var.dimensions)):
                if i + 1 < len(ndrange_var.dimensions):
                    target_tmp = I // ndrange_var.acc_dimensions[i + 1]
                else:
                    target_tmp = I
                ti.subscript(target_var,
                             i).assign(target_tmp + ndrange_var.bounds[i][0])
                if i + 1 < len(ndrange_var.dimensions):
                    I.assign(I -
                             target_tmp * ndrange_var.acc_dimensions[i + 1])
            node.body = build_stmts(ctx, node.body)
            ti.core.end_frontend_range_for()
        return node

    @staticmethod
    def build_struct_for(ctx, node, is_grouped):
        # for i, j in x
        # for I in ti.grouped(x)
        targets = IRBuilder.get_for_loop_targets(node)

        for target in targets:
            ctx.check_loop_var(target)

        with ctx.variable_scope_guard():
            if is_grouped:
                if len(targets) != 1:
                    raise TaichiSyntaxError(
                        f"Group for should have 1 loop target, found {len(targets)}"
                    )
                target = targets[0]
                loop_var = build_stmt(ctx, node.iter).ptr
                loop_indices = ti.lang.expr.make_var_list(
                    size=len(loop_var.shape))
                expr_group = ti.lang.expr.make_expr_group(loop_indices)
                ti.begin_frontend_struct_for(expr_group, loop_var)
                ctx.create_variable(target, ti.Vector(loop_indices, dt=ti.i32))
                node.body = build_stmts(ctx, node.body)
                ti.core.end_frontend_range_for()
            else:
                vars = []
                for name in targets:
                    var = ti.Expr(ti.core.make_id_expr(""))
                    vars.append(var)
                    ctx.create_variable(name, var)
                loop_var = build_stmt(ctx, node.iter).ptr
                expr_group = ti.lang.expr.make_expr_group(*vars)
                ti.begin_frontend_struct_for(expr_group, loop_var)
                node.body = build_stmts(ctx, node.body)
                ti.core.end_frontend_range_for()
        return node

    @staticmethod
    def build_For(ctx, node):
        if node.orelse:
            raise TaichiSyntaxError(
                "'else' clause for 'for' not supported in Taichi kernels")

        with ctx.control_scope_guard():

            decorator = IRBuilder.get_decorator(ctx, node.iter)
            double_decorator = ''
            if decorator != '' and len(node.iter.args) == 1:
                double_decorator = IRBuilder.get_decorator(
                    ctx, node.iter.args[0])
            ast.fix_missing_locations(node)

            if decorator == 'static':
                if double_decorator == 'static':
                    raise TaichiSyntaxError("'ti.static' cannot be nested")
                return IRBuilder.build_static_for(
                    ctx, node, double_decorator == 'grouped')
            elif decorator == 'ndrange':
                if double_decorator != '':
                    raise TaichiSyntaxError(
                        "No decorator is allowed inside 'ti.ndrange")
                return IRBuilder.build_ndrange_for(ctx, node)
            elif decorator == 'grouped':
                if double_decorator == 'static':
                    raise TaichiSyntaxError(
                        "'ti.static' is not allowed inside 'ti.grouped'")
                elif double_decorator == 'ndrange':
                    return IRBuilder.build_grouped_ndrange_for(ctx, node)
                elif double_decorator == 'grouped':
                    raise TaichiSyntaxError("'ti.grouped' cannot be nested")
                else:
                    return IRBuilder.build_struct_for(ctx,
                                                      node,
                                                      is_grouped=True)
            elif isinstance(node.iter, ast.Call) and isinstance(
                    node.iter.func, ast.Name) and node.iter.func.id == 'range':
                return IRBuilder.build_range_for(ctx, node)
            else:  # Struct for
                return IRBuilder.build_struct_for(ctx, node, is_grouped=False)

    @staticmethod
    def build_While(ctx, node):
        if node.orelse:
            raise TaichiSyntaxError(
                "'else' clause for 'while' not supported in Taichi kernels")

        with ctx.control_scope_guard():
            ti.core.begin_frontend_while(ti.Expr(1).ptr)
            while_cond = build_stmt(ctx, node.test).ptr
            ti.begin_frontend_if(while_cond)
            ti.core.begin_frontend_if_true()
            ti.core.pop_scope()
            ti.core.begin_frontend_if_false()
            ti.core.insert_break_stmt()
            ti.core.pop_scope()
            node.body = build_stmts(ctx, node.body)
            ti.core.pop_scope()
        return node

    @staticmethod
    def build_If(ctx, node):
        node.test = build_stmt(ctx, node.test)
        is_static_if = (IRBuilder.get_decorator(ctx, node.test) == "static")

        if is_static_if:
            if node.test.ptr:
                node.body = build_stmts(ctx, node.body)
            else:
                node.orelse = build_stmts(ctx, node.orelse)
            return node

        ti.begin_frontend_if(node.test.ptr)
        ti.core.begin_frontend_if_true()
        node.body = build_stmts(ctx, node.body)
        ti.core.pop_scope()
        ti.core.begin_frontend_if_false()
        node.orelse = build_stmts(ctx, node.orelse)
        ti.core.pop_scope()
        return node

    @staticmethod
    def build_Expr(ctx, node):
        node.value = build_stmt(ctx, node.value)
        return node

    @staticmethod
    def build_IfExp(ctx, node):
        node.test = build_stmt(ctx, node.test)
        is_static_if = (IRBuilder.get_decorator(ctx, node.test) == "static")

        if is_static_if:
            if node.test.ptr:
                node.body = build_stmt(ctx, node.body)
                node.ptr = node.body.ptr
            else:
                node.orelse = build_stmt(ctx, node.orelse)
                node.ptr = node.orelse.ptr
            return node

        val = ti.expr_init(None)

        ti.begin_frontend_if(node.test.ptr)
        ti.core.begin_frontend_if_true()
        node.body = build_stmt(ctx, node.body)
        val.assign(node.body.ptr)
        ti.core.pop_scope()
        ti.core.begin_frontend_if_false()
        node.orelse = build_stmt(ctx, node.orelse)
        val.assign(node.orelse.ptr)
        ti.core.pop_scope()

        node.ptr = val
        return node

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
        msg = build_stmt(ctx, node.left).ptr
        args = build_stmt(ctx, node.right).ptr
        if not isinstance(args, collections.abc.Sequence):
            args = (args, )
        return msg, args

    @staticmethod
    def build_Assert(ctx, node):
        extra_args = []
        if node.msg is not None:
            if isinstance(node.msg, ast.Constant):
                msg = node.msg.value
            elif isinstance(node.msg, ast.Str):
                msg = node.msg.s
            elif IRBuilder._is_string_mod_args(node.msg):
                msg, extra_args = IRBuilder._handle_string_mod_args(
                    ctx, node.msg)
            else:
                raise ValueError(
                    f"assert info must be constant, not {ast.dump(node.msg)}")
        else:
            msg = astor.to_source(node.test)
        test = build_stmt(ctx, node.test).ptr
        ti.ti_assert(test, msg.strip(), extra_args)
        return node

    @staticmethod
    def build_Break(ctx, node):
        if ctx.is_in_static():
            ctx.set_loop_status(LoopStatus.Break)
        else:
            ti.core.insert_break_stmt()
        return node

    @staticmethod
    def build_Continue(ctx, node):
        if ctx.is_in_static():
            ctx.set_loop_status(LoopStatus.Continue)
        else:
            ti.core.insert_continue_stmt()
        return node

    @staticmethod
    def build_Pass(ctx, node):
        return node


build_stmt = IRBuilder()


def build_stmts(ctx, stmts):
    result = []
    with ctx.variable_scope_guard(result):
        for stmt in list(stmts):
            if ctx.loop_status() == LoopStatus.Normal:
                result.append(build_stmt(ctx, stmt))
            else:
                result.append(stmt)
    return result
