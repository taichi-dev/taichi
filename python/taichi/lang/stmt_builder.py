import ast
import copy

from taichi.lang import impl
from taichi.lang.ast_builder_utils import *
from taichi.lang.ast_resolver import ASTResolver
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.expr_builder import build_expr, build_exprs
from taichi.lang.util import to_taichi_type

import taichi as ti


class StmtBuilder(Builder):
    @staticmethod
    def set_subscript_index(node, value):
        assert isinstance(node, ast.Subscript), type(node)
        if isinstance(node.slice, ast.Index):
            node.slice.value = value
        else:
            node.slice = value

    @staticmethod
    def make_single_statement(stmts):
        template = 'if 1: pass'
        t = ast.parse(template).body[0]
        t.body = stmts
        return t

    @staticmethod
    def make_constant(value):
        # Do not use ast.Constant which does not exist in python3.5
        node = parse_expr('0')
        node.value = value
        return node

    @staticmethod
    def build_AugAssign(ctx, node):
        node.target = build_expr(ctx, node.target)
        node.value = build_expr(ctx, node.value)
        template = 'x.augassign(0, 0)'
        t = ast.parse(template).body[0]
        t.value.func.value = node.target
        t.value.func.value.ctx = ast.Load()
        t.value.args[0] = node.value
        t.value.args[1] = ast.Str(s=type(node.op).__name__,
                                  ctx=ast.Load(),
                                  kind=None)
        return ast.copy_location(t, node)

    @staticmethod
    def _is_string_mod_args(msg):
        # 1. str % (a, b, c, ...)
        # 2. str % single_item
        # Note that |msg.right| may not be a tuple.
        return isinstance(msg, ast.BinOp) and isinstance(
            msg.left, ast.Str) and isinstance(msg.op, ast.Mod)

    @staticmethod
    def _handle_string_mod_args(ctx, msg):
        assert StmtBuilder._is_string_mod_args(msg)
        s = msg.left.s
        t = None
        if isinstance(msg.right, ast.Tuple):
            t = msg.right
        else:
            # assuming the format is `str % single_item`
            t = ast.Tuple(elts=[msg.right], ctx=ast.Load())
        t = build_expr(ctx, t)
        return s, t

    @staticmethod
    def build_Assert(ctx, node):
        extra_args = ast.List(elts=[], ctx=ast.Load())
        if node.msg is not None:
            if isinstance(node.msg, ast.Constant):
                msg = node.msg.value
            elif isinstance(node.msg, ast.Str):
                msg = node.msg.s
            elif StmtBuilder._is_string_mod_args(node.msg):
                msg = build_expr(ctx, node.msg)
                msg, extra_args = StmtBuilder._handle_string_mod_args(ctx, msg)
            else:
                raise ValueError(
                    f"assert info must be constant, not {ast.dump(node.msg)}")
        else:
            import astor
            msg = astor.to_source(node.test)
        node.test = build_expr(ctx, node.test)

        new_node = parse_stmt('ti.ti_assert(0, 0, [])')
        new_node.value.args[0] = node.test
        new_node.value.args[1] = parse_expr("'{}'".format(msg.strip()))
        new_node.value.args[2] = extra_args
        new_node = ast.copy_location(new_node, node)
        return new_node

    @staticmethod
    def build_Assign(ctx, node):
        assert (len(node.targets) == 1)
        node.value = build_expr(ctx, node.value)
        node.targets = build_exprs(ctx, node.targets)

        is_static_assign = isinstance(
            node.value, ast.Call) and ASTResolver.resolve_to(
                node.value.func, ti.static, globals())
        if is_static_assign:
            return node

        if isinstance(node.targets[0], ast.Tuple):
            targets = node.targets[0].elts

            # Create
            stmts = []

            holder = parse_stmt('__tmp_tuple = ti.expr_init_list(0, '
                                f'{len(targets)})')
            holder.value.args[0] = node.value

            stmts.append(holder)

            def tuple_indexed(i):
                indexing = parse_stmt('__tmp_tuple[0]')
                StmtBuilder.set_subscript_index(indexing.value,
                                                parse_expr("{}".format(i)))
                return indexing.value

            for i, target in enumerate(targets):
                is_local = isinstance(target, ast.Name)
                if is_local and ctx.is_creation(target.id):
                    var_name = target.id
                    target.ctx = ast.Store()
                    # Create, no AST resolution needed
                    init = ast.Attribute(value=ast.Name(id='ti',
                                                        ctx=ast.Load()),
                                         attr='expr_init',
                                         ctx=ast.Load())
                    rhs = ast.Call(
                        func=init,
                        args=[tuple_indexed(i)],
                        keywords=[],
                    )
                    ctx.create_variable(var_name)
                    stmts.append(
                        ast.Assign(targets=[target],
                                   value=rhs,
                                   type_comment=None))
                else:
                    # Assign
                    target.ctx = ast.Load()
                    func = ast.Attribute(value=target,
                                         attr='assign',
                                         ctx=ast.Load())
                    call = ast.Call(func=func,
                                    args=[tuple_indexed(i)],
                                    keywords=[])
                    stmts.append(ast.Expr(value=call))

            for stmt in stmts:
                ast.copy_location(stmt, node)
            stmts.append(parse_stmt('del __tmp_tuple'))
            return StmtBuilder.make_single_statement(stmts)
        else:
            is_local = isinstance(node.targets[0], ast.Name)
            if is_local and ctx.is_creation(node.targets[0].id):
                var_name = node.targets[0].id
                # Create, no AST resolution needed
                init = ast.Attribute(value=ast.Name(id='ti', ctx=ast.Load()),
                                     attr='expr_init',
                                     ctx=ast.Load())
                rhs = ast.Call(
                    func=init,
                    args=[node.value],
                    keywords=[],
                )
                ctx.create_variable(var_name)
                return ast.copy_location(
                    ast.Assign(targets=node.targets,
                               value=rhs,
                               type_comment=None), node)
            else:
                # Assign
                node.targets[0].ctx = ast.Load()
                func = ast.Attribute(value=node.targets[0],
                                     attr='assign',
                                     ctx=ast.Load())
                call = ast.Call(func=func, args=[node.value], keywords=[])
                return ast.copy_location(ast.Expr(value=call), node)

    @staticmethod
    def build_Try(ctx, node):
        raise TaichiSyntaxError(
            "Keyword 'try' not supported in Taichi kernels")

    @staticmethod
    def build_While(ctx, node):
        if node.orelse:
            raise TaichiSyntaxError(
                "'else' clause for 'while' not supported in Taichi kernels")

        with ctx.control_scope():
            ctx.current_control_scope().append('while')

            template = '''
if 1:
  ti.core.begin_frontend_while(ti.Expr(1).ptr)
  __while_cond = 0
  if __while_cond:
    pass
  else:
    break
  ti.core.pop_scope()
'''
            cond = node.test
            t = ast.parse(template).body[0]
            t.body[1].value = cond
            t.body = t.body[:3] + node.body + t.body[3:]

            t.body = build_stmts(ctx, t.body)
            return ast.copy_location(t, node)

    @staticmethod
    def build_If(ctx, node):
        node.test = build_expr(ctx, node.test)
        node.body = build_stmts(ctx, node.body)
        node.orelse = build_stmts(ctx, node.orelse)

        is_static_if = isinstance(node.test, ast.Call) and isinstance(
            node.test.func, ast.Attribute)
        if is_static_if:
            attr = node.test.func
            if attr.attr == 'static':
                is_static_if = True
            else:
                is_static_if = False

        if is_static_if:
            # Do nothing
            return node

        template = '''
if 1:
  __cond = 0
  ti.begin_frontend_if(__cond)
  ti.core.begin_frontend_if_true()
  ti.core.pop_scope()
  ti.core.begin_frontend_if_false()
  ti.core.pop_scope()
'''
        t = ast.parse(template).body[0]
        cond = node.test
        t.body[0].value = cond
        t.body = t.body[:5] + node.orelse + t.body[5:]
        t.body = t.body[:3] + node.body + t.body[3:]
        return ast.copy_location(t, node)

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
    def build_static_for(ctx, node, is_grouped):
        # for i in ti.static(range(n))
        # for i, j in ti.static(ti.ndrange(n))
        # for I in ti.static(ti.grouped(ti.ndrange(n, m)))

        ctx.current_control_scope().append('static')
        node.body = build_stmts(ctx, node.body)
        if is_grouped:
            assert len(node.iter.args[0].args) == 1
            template = '''
if 1:
    __ndrange_arg = 0
    from taichi.lang.exception import TaichiSyntaxError
    if not isinstance(__ndrange_arg, ti.ndrange):
        raise TaichiSyntaxError("Only 'ti.ndrange' is allowed in 'ti.static(ti.grouped(...))'.")
    pass
    del a
            '''
            t = ast.parse(template).body[0]
            t.body[0].value = node.iter.args[0].args[0]
            t.body[3] = node
            t.body[3].iter.args[0].args[0] = parse_expr('__ndrange_arg')
        else:
            t = parse_stmt('if 1: pass; del a')
            t.body[0] = node
        target = copy.deepcopy(node.target)
        target.ctx = ast.Del()
        if isinstance(target, ast.Tuple):
            for tar in target.elts:
                tar.ctx = ast.Del()
        t.body[-1].targets = [target]
        return t

    @staticmethod
    def build_range_for(ctx, node):
        # for i in range(n)
        node.body = build_stmts(ctx, node.body)
        loop_var = node.target.id
        ctx.check_loop_var(loop_var)
        template = '''
if 1:
    {} = ti.Expr(ti.core.make_id_expr(''))
    ___begin = ti.Expr(0)
    ___end = ti.Expr(0)
    ___begin = ti.cast(___begin, ti.i32)
    ___end = ti.cast(___end, ti.i32)
    ti.core.begin_frontend_range_for({}.ptr, ___begin.ptr, ___end.ptr)
    ti.core.end_frontend_range_for()
        '''.format(loop_var, loop_var)
        t = ast.parse(template).body[0]

        assert len(node.iter.args) in [1, 2]
        if len(node.iter.args) == 2:
            bgn = build_expr(ctx, node.iter.args[0])
            end = build_expr(ctx, node.iter.args[1])
        else:
            bgn = StmtBuilder.make_constant(value=0)
            end = build_expr(ctx, node.iter.args[0])

        t.body[1].value.args[0] = bgn
        t.body[2].value.args[0] = end
        t.body = t.body[:6] + node.body + t.body[6:]
        t.body.append(parse_stmt('del {}'.format(loop_var)))
        return ast.copy_location(t, node)

    @staticmethod
    def build_ndrange_for(ctx, node):
        # for i, j in ti.ndrange(n)
        template = f'''
if ti.static(1):
    __ndrange{id(node)} = 0
    for __ndrange_I{id(node)} in range(0):
        __I = __ndrange_I{id(node)}
        '''
        t = ast.parse(template).body[0]
        t.body[0].value = node.iter
        t_loop = t.body[1]
        t_loop.iter.args[0] = parse_expr(
            f'__ndrange{id(node)}.acc_dimensions[0]')
        targets = StmtBuilder.get_for_loop_targets(node)
        targets_tmp = ['__' + name for name in targets]
        loop_body = t_loop.body
        for i in range(len(targets)):
            if i + 1 < len(targets):
                stmt = '{} = __I // __ndrange{}.acc_dimensions[{}]'.format(
                    targets_tmp[i], id(node), i + 1)
            else:
                stmt = '{} = __I'.format(targets_tmp[i])
            loop_body.append(parse_stmt(stmt))
            stmt = '{} = {} + __ndrange{}.bounds[{}][0]'.format(
                targets[i], targets_tmp[i], id(node), i)
            loop_body.append(parse_stmt(stmt))
            if i + 1 < len(targets):
                stmt = '__I = __I - {} * __ndrange{}.acc_dimensions[{}]'.format(
                    targets_tmp[i], id(node), i + 1)
                loop_body.append(parse_stmt(stmt))
        loop_body += node.body

        node = ast.copy_location(t, node)
        return build_stmt(ctx, node)  # further translate as a range for

    @staticmethod
    def build_grouped_ndrange_for(ctx, node):
        # for I in ti.grouped(ti.ndrange(n, m))
        node.body = build_stmts(ctx, node.body)
        target = node.target.id
        template = '''
if ti.static(1):
    __ndrange = 0
    {} = ti.expr_init(ti.Vector([0] * len(__ndrange.dimensions), disable_local_tensor=True))
    ___begin = ti.Expr(0)
    ___end = __ndrange.acc_dimensions[0]
    ___begin = ti.cast(___begin, ti.i32)
    ___end = ti.cast(___end, ti.i32)
    __ndrange_I = ti.Expr(ti.core.make_id_expr(''))
    ti.core.begin_frontend_range_for(__ndrange_I.ptr, ___begin.ptr, ___end.ptr)
    __I = __ndrange_I
    for __grouped_I in range(len(__ndrange.dimensions)):
        __grouped_I_tmp = 0
        if __grouped_I + 1 < len(__ndrange.dimensions):
            __grouped_I_tmp = __I // __ndrange.acc_dimensions[__grouped_I + 1]
        else:
            __grouped_I_tmp = __I
        ti.subscript({}, __grouped_I).assign(__grouped_I_tmp + __ndrange.bounds[__grouped_I][0])
        if __grouped_I + 1 < len(__ndrange.dimensions):
            __I = __I - __grouped_I_tmp * __ndrange.acc_dimensions[__grouped_I + 1]
    ti.core.end_frontend_range_for()
        '''.format(target, target)
        t = ast.parse(template).body[0]
        node.iter.args[0].args = build_exprs(ctx, node.iter.args[0].args)
        t.body[0].value = node.iter.args[0]
        cut = len(t.body) - 1
        t.body = t.body[:cut] + node.body + t.body[cut:]
        return ast.copy_location(t, node)

    @staticmethod
    def build_struct_for(ctx, node, is_grouped):
        # for i, j in x
        # for I in ti.grouped(x)
        node.body = build_stmts(ctx, node.body)
        targets = StmtBuilder.get_for_loop_targets(node)

        for loop_var in targets:
            ctx.check_loop_var(loop_var)

        var_decl = ''.join(
            '    {} = ti.Expr(ti.core.make_id_expr(""))\n'.format(name)
            for name in targets)  # indent: 4 spaces
        vars = ', '.join(targets)
        if is_grouped:
            template = '''
if 1:
    ___loop_var = 0
    {} = ti.lang.expr.make_var_vector(size=len(___loop_var.shape))
    ___expr_group = ti.lang.expr.make_expr_group({})
    ti.begin_frontend_struct_for(___expr_group, ___loop_var)
    ti.core.end_frontend_range_for()
            '''.format(vars, vars)
            t = ast.parse(template).body[0]
            cut = 4
            t.body[0].value = node.iter
            t.body = t.body[:cut] + node.body + t.body[cut:]
        else:
            template = '''
if 1:
{}
    ___loop_var = 0
    ___expr_group = ti.lang.expr.make_expr_group({})
    ti.begin_frontend_struct_for(___expr_group, ___loop_var)
    ti.core.end_frontend_range_for()
            '''.format(var_decl, vars)
            t = ast.parse(template).body[0]
            cut = len(targets) + 3
            t.body[cut - 3].value = node.iter
            t.body = t.body[:cut] + node.body + t.body[cut:]
        for loop_var in reversed(targets):
            t.body.append(parse_stmt('del {}'.format(loop_var)))
        return ast.copy_location(t, node)

    @staticmethod
    def build_For(ctx, node):
        if node.orelse:
            raise TaichiSyntaxError(
                "'else' clause for 'for' not supported in Taichi kernels")

        with ctx.control_scope():
            ctx.current_control_scope().append('for')

            decorator = StmtBuilder.get_decorator(node.iter)
            double_decorator = ''
            if decorator != '' and len(node.iter.args) == 1:
                double_decorator = StmtBuilder.get_decorator(node.iter.args[0])
            ast.fix_missing_locations(node)

            if decorator == 'static':
                if double_decorator == 'static':
                    raise TaichiSyntaxError("'ti.static' cannot be nested")
                return StmtBuilder.build_static_for(
                    ctx, node, double_decorator == 'grouped')
            elif decorator == 'ndrange':
                if double_decorator != '':
                    raise TaichiSyntaxError(
                        "No decorator is allowed inside 'ti.ndrange")
                return StmtBuilder.build_ndrange_for(ctx, node)
            elif decorator == 'grouped':
                if double_decorator == 'static':
                    raise TaichiSyntaxError(
                        "'ti.static' is not allowed inside 'ti.grouped'")
                elif double_decorator == 'ndrange':
                    return StmtBuilder.build_grouped_ndrange_for(ctx, node)
                elif double_decorator == 'grouped':
                    raise TaichiSyntaxError("'ti.grouped' cannot be nested")
                else:
                    return StmtBuilder.build_struct_for(ctx,
                                                        node,
                                                        is_grouped=True)
            elif isinstance(node.iter, ast.Call) and isinstance(
                    node.iter.func, ast.Name) and node.iter.func.id == 'range':
                return StmtBuilder.build_range_for(ctx, node)
            else:  # Struct for
                return StmtBuilder.build_struct_for(ctx,
                                                    node,
                                                    is_grouped=False)

    @staticmethod
    def build_Break(ctx, node):
        if 'static' in ctx.current_control_scope():
            return node
        else:
            return parse_stmt('ti.core.insert_break_stmt()')

    @staticmethod
    def build_Continue(ctx, node):
        if 'static' in ctx.current_control_scope():
            return node
        else:
            return parse_stmt('ti.core.insert_continue_stmt()')

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
                ret_init = parse_stmt(
                    'ti.lang.kernel_arguments.decl_scalar_ret(0)')
                ret_init.value.args[0] = node.returns
                ctx.returns = node.returns
                arg_decls.append(ret_init)
                node.returns = None

            for i, arg in enumerate(args.args):
                # Directly pass in template arguments,
                # such as class instances ("self"), fields, SNodes, etc.
                if isinstance(ctx.func.argument_annotations[i], ti.template):
                    continue
                if isinstance(ctx.func.argument_annotations[i],
                              ti.sparse_matrix_builder):
                    arg_init = parse_stmt(
                        'x = ti.lang.kernel_arguments.decl_sparse_matrix()')
                    arg_init.targets[0].id = arg.arg
                    ctx.create_variable(arg.arg)
                    arg_decls.append(arg_init)
                elif isinstance(ctx.func.argument_annotations[i], ti.any_arr):
                    arg_init = parse_stmt(
                        'x = ti.lang.kernel_arguments.decl_any_arr_arg(0, 0, 0, 0)'
                    )
                    arg_init.targets[0].id = arg.arg
                    ctx.create_variable(arg.arg)
                    array_dt = ctx.arg_features[i][0]
                    array_dim = ctx.arg_features[i][1]
                    array_element_shape = ctx.arg_features[i][2]
                    array_layout = ctx.arg_features[i][3]
                    array_dt = to_taichi_type(array_dt)
                    dt_expr = 'ti.' + ti.core.data_type_name(array_dt)
                    dt = parse_expr(dt_expr)
                    arg_init.value.args[0] = dt
                    arg_init.value.args[1] = parse_expr("{}".format(array_dim))
                    arg_init.value.args[2] = parse_expr(
                        "{}".format(array_element_shape))
                    arg_init.value.args[3] = parse_expr(
                        "ti.{}".format(array_layout))
                    arg_decls.append(arg_init)
                else:
                    arg_init = parse_stmt(
                        'x = ti.lang.kernel_arguments.decl_scalar_arg(0)')
                    arg_init.targets[0].id = arg.arg
                    dt = arg.annotation
                    arg_init.value.args[0] = dt
                    arg_decls.append(arg_init)
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
            if impl.get_runtime().experimental_real_function:
                transform_as_kernel()
            else:
                # Transform as func (all parameters passed by value)
                arg_decls = []
                for i, arg in enumerate(args.args):
                    # Directly pass in template arguments,
                    # such as class instances ("self"), fields, SNodes, etc.
                    if isinstance(ctx.func.argument_annotations[i],
                                  ti.template):
                        continue
                    # Create a copy for non-template arguments,
                    # so that they are passed by value.
                    arg_init = parse_stmt('x = ti.expr_init_func(0)')
                    arg_init.targets[0].id = arg.arg
                    ctx.create_variable(arg.arg)
                    arg_init.value.args[0] = parse_expr(arg.arg +
                                                        '_by_value__')
                    args.args[i].arg += '_by_value__'
                    arg_decls.append(arg_init)

        with ctx.variable_scope():
            node.body = build_stmts(ctx, node.body)

        node.body = arg_decls + node.body
        node.body = [parse_stmt('import taichi as ti')] + node.body
        return node

    @staticmethod
    def build_Return(ctx, node):
        node.value = build_expr(ctx, node.value)
        if ctx.is_kernel or impl.get_runtime().experimental_real_function:
            # TODO: check if it's at the end of a kernel, throw TaichiSyntaxError if not
            if node.value is not None:
                if ctx.returns is None:
                    raise TaichiSyntaxError(
                        f'A {"kernel" if ctx.is_kernel else "function"} '
                        'with a return value must be annotated '
                        'with a return type, e.g. def func() -> ti.f32')
                ret_expr = parse_expr('ti.cast(ti.Expr(0), 0)')
                ret_expr.args[0].args[0] = node.value
                ret_expr.args[1] = ctx.returns
                ret_stmt = parse_stmt('ti.core.create_kernel_return(ret.ptr)')
                # For args[0], it is an ast.Attribute, because it loads the
                # attribute, |ptr|, of the expression |ret_expr|. Therefore we
                # only need to replace the object part, i.e. args[0].value
                ret_stmt.value.args[0].value = ret_expr
                return ret_stmt
        return node

    @staticmethod
    def build_Module(ctx, node):
        with ctx.variable_scope():
            # Do NOT use |build_stmts| which inserts 'del' statements to the
            # end and deletes parameters passed into the module
            node.body = [build_stmt(ctx, stmt) for stmt in list(node.body)]
        return node

    @staticmethod
    def build_Global(ctx, node):
        raise TaichiSyntaxError(
            "Keyword 'global' not supported in Taichi kernels")

    @staticmethod
    def build_Nonlocal(ctx, node):
        raise TaichiSyntaxError(
            "Keyword 'nonlocal' not supported in Taichi kernels")

    @staticmethod
    def build_Raise(ctx, node):
        node.exc = build_expr(ctx, node.exc)
        return node

    @staticmethod
    def build_Expr(ctx, node):
        if not isinstance(node.value, ast.Call):
            # A statement with a single expression.
            return node

        # A function call.
        node.value = build_expr(ctx, node.value)
        # Note that we can only return an ast.Expr instead of an ast.Call.

        if impl.get_runtime().experimental_real_function:
            # Generates code that inserts a FrontendExprStmt if the function
            # called is a Taichi function.
            # We cannot insert the FrontendExprStmt here because we do not
            # know if the function is a Taichi function now.
            node.value.args = [node.value.func] + node.value.args
            node.value.func = parse_expr('ti.insert_expr_stmt_if_ti_func')
        return node

    @staticmethod
    def build_Import(ctx, node):
        return node

    @staticmethod
    def build_ImportFrom(ctx, node):
        return node

    @staticmethod
    def build_Pass(ctx, node):
        return node


build_stmt = StmtBuilder()


def build_stmts(ctx, stmts):
    result = []
    with ctx.variable_scope(result):
        for stmt in list(stmts):
            result.append(build_stmt(ctx, stmt))
    return result
