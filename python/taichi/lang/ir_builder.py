import ast
import collections.abc
import warnings
from collections import ChainMap

from taichi.lang.ast.symbol_resolver import ASTResolver
from taichi.lang.ast_builder_utils import *
from taichi.lang.exception import TaichiSyntaxError

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
        if is_static_assign:
            return node

        # Keep all generated assign statements and compose single one at last.
        # The variable is introduced to support chained assignments.
        # Ref https://github.com/taichi-dev/taichi/issues/2659.
        for node_target in node.targets:
            if isinstance(node_target, ast.Tuple):
                IRBuilder.build_assign_unpack(ctx, node_target, node.value.ptr)
            else:
                IRBuilder.build_assign_basic(ctx, node_target, node.value.ptr)
        return node

    @staticmethod
    def build_assign_unpack(ctx, node_target, value):
        """Build the unpack assignments like this: (target1, target2) = (value1, value2).
        The function should be called only if the node target is a tuple.

        Args:
            ctx (ast_builder_utils.BuilderContext): The builder context.
            node_target (ast.Tuple): A list or tuple object. `node_target.elts` holds a
            list of nodes representing the elements.
            value: A node/list representing the values.
        """

        targets = node_target.elts
        tmp_tuple = ti.expr_init_list(value, len(targets))

        for i, target in enumerate(targets):
            IRBuilder.build_assign_basic(ctx, target, tmp_tuple[i])

    @staticmethod
    def build_assign_basic(ctx, target, value):
        """Build basic assginment like this: target = value.

         Args:
            ctx (ast_builder_utils.BuilderContext): The builder context.
            target (ast.Name): A variable name. `target.id` holds the name as
            a string.
            value: A node representing the value.
        """
        is_local = isinstance(target, ast.Name)
        if is_local and not ctx.is_var_declared(target.id):
            ctx.create_variable(target.id, ti.expr_init(value))
        else:
            var = target.ptr
            var.assign(value)

    @staticmethod
    def build_Subscript(ctx, node):
        node.value = build_stmt(ctx, node.value)
        node.slice = build_stmt(ctx, node.slice)
        if not isinstance(node.slice, ast.Tuple):
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
                node.returns = build_stmt(ctx, node.returns)
                ti.lang.kernel_arguments.decl_scalar_ret(node.returns.ptr)
                ctx.returns = node.returns.ptr

            for i, arg in enumerate(args.args):
                # Directly pass in template arguments,
                # such as class instances ("self"), fields, SNodes, etc.
                # if isinstance(ctx.func.argument_annotations[i], ti.template):
                #     continue
                # if isinstance(ctx.func.argument_annotations[i],
                #               ti.sparse_matrix_builder):
                #     arg_init = parse_stmt(
                #         'x = ti.lang.kernel_arguments.decl_sparse_matrix()')
                #     arg_init.targets[0].id = arg.arg
                #     ctx.create_variable(arg.arg)
                #     arg_decls.append(arg_init)
                # elif isinstance(ctx.func.argument_annotations[i], ti.any_arr):
                #     arg_init = parse_stmt(
                #         'x = ti.lang.kernel_arguments.decl_any_arr_arg(0, 0, 0, 0)'
                #     )
                #     arg_init.targets[0].id = arg.arg
                #     ctx.create_variable(arg.arg)
                #     array_dt = ctx.arg_features[i][0]
                #     array_dim = ctx.arg_features[i][1]
                #     array_element_shape = ctx.arg_features[i][2]
                #     array_layout = ctx.arg_features[i][3]
                #     array_dt = to_taichi_type(array_dt)
                #     dt_expr = 'ti.' + ti.core.data_type_name(array_dt)
                #     dt = parse_expr(dt_expr)
                #     arg_init.value.args[0] = dt
                #     arg_init.value.args[1] = parse_expr("{}".format(array_dim))
                #     arg_init.value.args[2] = parse_expr(
                #         "{}".format(array_element_shape))
                #     arg_init.value.args[3] = parse_expr(
                #         "ti.{}".format(array_layout))
                #     arg_decls.append(arg_init)
                if isinstance(ctx.func.argument_annotations[i], ti.template):
                    continue
                else:
                    arg.annotation = build_stmt(ctx, arg.annotation)
                    ctx.create_variable(
                        arg.arg,
                        ti.lang.kernel_arguments.decl_scalar_arg(
                            arg.annotation.ptr))
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
                if ctx.returns is None:
                    raise TaichiSyntaxError(
                        f'A {"kernel" if ctx.is_kernel else "function"} '
                        'with a return value must be annotated '
                        'with a return type, e.g. def func() -> ti.f32')
                ti.core.create_kernel_return(
                    ti.cast(ti.Expr(node.value.ptr), ctx.returns).ptr)
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
    def get_decorator(node):
        if not isinstance(node, ast.Call):
            return ''
        for wanted, name in [
            (ti.static, 'static'),
            (ti.grouped, 'grouped'),
            (ti.ndrange, 'ndrange'),
        ]:
            if ASTResolver.resolve_to(node.func, wanted, globals()):
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
        if is_grouped:
            pass
        else:
            node.iter = build_stmt(ctx, node.iter)
            targets = IRBuilder.get_for_loop_targets(node)
            for target_values in node.iter.ptr:
                if not isinstance(target_values, collections.abc.Sequence):
                    target_values = [target_values]
                with ctx.variable_scope_guard():
                    for target, target_value in zip(targets, target_values):
                        ctx.create_variable(target, target_value)
                    node.body = build_stmts_wo_scope(ctx, node.body)
        return node
#         # for i in ti.static(range(n))
#         # for i, j in ti.static(ti.ndrange(n))
#         # for I in ti.static(ti.grouped(ti.ndrange(n, m)))
#
#         ctx.current_control_scope().append('static')
#         node.body = build_stmts(ctx, node.body)
#         if is_grouped:
#             assert len(node.iter.args[0].args) == 1
#             template = '''
# if 1:
#     __ndrange_arg = 0
#     from taichi.lang.exception import TaichiSyntaxError
#     if not isinstance(__ndrange_arg, ti.ndrange):
#         raise TaichiSyntaxError("Only 'ti.ndrange' is allowed in 'ti.static(ti.grouped(...))'.")
#     pass
#     del a
#             '''
#             t = ast.parse(template).body[0]
#             t.body[0].value = node.iter.args[0].args[0]
#             t.body[3] = node
#             t.body[3].iter.args[0].args[0] = parse_expr('__ndrange_arg')
#         else:
#             t = parse_stmt('if 1: pass; del a')
#             t.body[0] = node
#         target = copy.deepcopy(node.target)
#         target.ctx = ast.Del()
#         if isinstance(target, ast.Tuple):
#             for tar in target.elts:
#                 tar.ctx = ast.Del()
#         t.body[-1].targets = [target]
#         return t

    @staticmethod
    def build_range_for(ctx, node):
        pass
#         # for i in range(n)
#         node.body = build_stmts(ctx, node.body)
#         loop_var = node.target.id
#         ctx.check_loop_var(loop_var)
#         template = '''
# if 1:
#     {} = ti.Expr(ti.core.make_id_expr(''))
#     ___begin = ti.Expr(0)
#     ___end = ti.Expr(0)
#     ___begin = ti.cast(___begin, ti.i32)
#     ___end = ti.cast(___end, ti.i32)
#     ti.core.begin_frontend_range_for({}.ptr, ___begin.ptr, ___end.ptr)
#     ti.core.end_frontend_range_for()
#         '''.format(loop_var, loop_var)
#         t = ast.parse(template).body[0]
#
#         assert len(node.iter.args) in [1, 2]
#         if len(node.iter.args) == 2:
#             bgn = build_expr(ctx, node.iter.args[0])
#             end = build_expr(ctx, node.iter.args[1])
#         else:
#             bgn = StmtBuilder.make_constant(value=0)
#             end = build_expr(ctx, node.iter.args[0])
#
#         t.body[1].value.args[0] = bgn
#         t.body[2].value.args[0] = end
#         t.body = t.body[:6] + node.body + t.body[6:]
#         t.body.append(parse_stmt('del {}'.format(loop_var)))
#         return ast.copy_location(t, node)

    @staticmethod
    def build_ndrange_for(ctx, node):
        pass
#         # for i, j in ti.ndrange(n)
#         template = f'''
# if ti.static(1):
#     __ndrange{id(node)} = 0
#     for __ndrange_I{id(node)} in range(0):
#         __I = __ndrange_I{id(node)}
#         '''
#         t = ast.parse(template).body[0]
#         t.body[0].value = node.iter
#         t_loop = t.body[1]
#         t_loop.iter.args[0] = parse_expr(
#             f'__ndrange{id(node)}.acc_dimensions[0]')
#         targets = StmtBuilder.get_for_loop_targets(node)
#         targets_tmp = ['__' + name for name in targets]
#         loop_body = t_loop.body
#         for i in range(len(targets)):
#             if i + 1 < len(targets):
#                 stmt = '{} = __I // __ndrange{}.acc_dimensions[{}]'.format(
#                     targets_tmp[i], id(node), i + 1)
#             else:
#                 stmt = '{} = __I'.format(targets_tmp[i])
#             loop_body.append(parse_stmt(stmt))
#             stmt = '{} = {} + __ndrange{}.bounds[{}][0]'.format(
#                 targets[i], targets_tmp[i], id(node), i)
#             loop_body.append(parse_stmt(stmt))
#             if i + 1 < len(targets):
#                 stmt = '__I = __I - {} * __ndrange{}.acc_dimensions[{}]'.format(
#                     targets_tmp[i], id(node), i + 1)
#                 loop_body.append(parse_stmt(stmt))
#         loop_body += node.body
#
#         node = ast.copy_location(t, node)
#         return build_stmt(ctx, node)  # further translate as a range for

    @staticmethod
    def build_grouped_ndrange_for(ctx, node):
        pass
#         # for I in ti.grouped(ti.ndrange(n, m))
#         node.body = build_stmts(ctx, node.body)
#         target = node.target.id
#         template = '''
# if ti.static(1):
#     __ndrange = 0
#     {} = ti.expr_init(ti.Vector([0] * len(__ndrange.dimensions), disable_local_tensor=True))
#     ___begin = ti.Expr(0)
#     ___end = __ndrange.acc_dimensions[0]
#     ___begin = ti.cast(___begin, ti.i32)
#     ___end = ti.cast(___end, ti.i32)
#     __ndrange_I = ti.Expr(ti.core.make_id_expr(''))
#     ti.core.begin_frontend_range_for(__ndrange_I.ptr, ___begin.ptr, ___end.ptr)
#     __I = __ndrange_I
#     for __grouped_I in range(len(__ndrange.dimensions)):
#         __grouped_I_tmp = 0
#         if __grouped_I + 1 < len(__ndrange.dimensions):
#             __grouped_I_tmp = __I // __ndrange.acc_dimensions[__grouped_I + 1]
#         else:
#             __grouped_I_tmp = __I
#         ti.subscript({}, __grouped_I).assign(__grouped_I_tmp + __ndrange.bounds[__grouped_I][0])
#         if __grouped_I + 1 < len(__ndrange.dimensions):
#             __I = __I - __grouped_I_tmp * __ndrange.acc_dimensions[__grouped_I + 1]
#     ti.core.end_frontend_range_for()
#         '''.format(target, target)
#         t = ast.parse(template).body[0]
#         node.iter.args[0].args = build_exprs(ctx, node.iter.args[0].args)
#         t.body[0].value = node.iter.args[0]
#         cut = len(t.body) - 1
#         t.body = t.body[:cut] + node.body + t.body[cut:]
#         return ast.copy_location(t, node)

    @staticmethod
    def build_struct_for(ctx, node, is_grouped):
        # for i, j in x
        # for I in ti.grouped(x)
        targets = IRBuilder.get_for_loop_targets(node)

        for loop_var in targets:
            ctx.check_loop_var(loop_var)

        if is_grouped:
            pass


#             template = '''
# if 1:
#     ___loop_var = 0
#     {} = ti.lang.expr.make_var_vector(size=len(___loop_var.shape))
#     ___expr_group = ti.lang.expr.make_expr_group({})
#     ti.begin_frontend_struct_for(___expr_group, ___loop_var)
#     ti.core.end_frontend_range_for()
#             '''.format(vars, vars)
#             t = ast.parse(template).body[0]
#             cut = 4
#             t.body[0].value = node.iter
#             t.body = t.body[:cut] + node.body + t.body[cut:]
        else:
            with ctx.variable_scope_guard():
                for name in targets:
                    ctx.create_variable(name,
                                        ti.Expr(ti.core.make_id_expr("")))
                vars = [ctx.get_var_by_name(name) for name in targets]
                node.iter = build_stmt(ctx, node.iter)
                ti.begin_frontend_struct_for(
                    ti.lang.expr.make_expr_group(*vars), node.iter.ptr)
                node.body = build_stmts_wo_scope(ctx, node.body)
                ti.core.end_frontend_range_for()
        return node

    @staticmethod
    def build_For(ctx, node):
        if node.orelse:
            raise TaichiSyntaxError(
                "'else' clause for 'for' not supported in Taichi kernels")

        with ctx.control_scope_guard():
            ctx.current_control_scope().append('for')

            decorator = IRBuilder.get_decorator(node.iter)
            double_decorator = ''
            if decorator != '' and len(node.iter.args) == 1:
                double_decorator = IRBuilder.get_decorator(node.iter.args[0])
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
    def build_If(ctx, node):
        node.test = build_stmt(ctx, node.test)
        is_static_if = (IRBuilder.get_decorator(node.test) == "static")

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

build_stmt = IRBuilder()


def build_stmts(ctx, stmts):
    result = []
    with ctx.variable_scope_guard(result):
        for stmt in list(stmts):
            result.append(build_stmt(ctx, stmt))
    return result


def build_stmts_wo_scope(ctx, stmts):
    result = []
    for stmt in list(stmts):
        result.append(build_stmt(ctx, stmt))
    return result
