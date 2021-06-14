import ast
import copy

from taichi.lang import impl
from taichi.lang.ast_resolver import ASTResolver
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.util import to_taichi_type

import taichi as ti


class ScopeGuard:
    def __init__(self, scopes, stmt_block=None):
        self.scopes = scopes
        self.stmt_block = stmt_block

    def __enter__(self):
        self.scopes.append([])

    def __exit__(self, exc_type, exc_val, exc_tb):
        local = self.scopes[-1]
        if self.stmt_block is not None:
            for var in reversed(local):
                stmt = ASTTransformerBase.parse_stmt('del var')
                stmt.targets[0].id = var
                self.stmt_block.append(stmt)
        self.scopes.pop()


# Total transform
# TODO: ASTTransformerBase -> ASTTransformer
# TODO: ASTTransformer -> ASTTransformerTotal
class ASTTransformer(object):
    def __init__(self, func=None, *args, **kwargs):
        self.pass_Preprocess = ASTTransformerPreprocess(func=func,
                                                        *args,
                                                        **kwargs)
        self.pass_Checks = ASTTransformerChecks(func=func)
        self.pass_transform_function_call = TransformFunctionCallAsStmt(
            func=func)

    @staticmethod
    def print_ast(tree, title=None):
        if not impl.get_runtime().print_preprocessed:
            return
        if title is not None:
            print(f'{title}:')
        import astor
        print(astor.to_source(tree.body[0], indent_with='    '))

    def visit(self, tree):
        self.print_ast(tree, 'Initial AST')
        self.pass_Preprocess.visit(tree)
        ast.fix_missing_locations(tree)
        self.print_ast(tree, 'Preprocessed')
        self.pass_Checks.visit(tree)
        self.print_ast(tree, 'Checked')
        self.pass_transform_function_call.visit(tree)
        ast.fix_missing_locations(tree)
        self.print_ast(tree, 'Final AST')


class ASTTransformerBase(ast.NodeTransformer):
    def __init__(self, func):
        super().__init__()
        self.func = func

    @staticmethod
    def parse_stmt(stmt):
        return ast.parse(stmt).body[0]

    @staticmethod
    def parse_expr(expr):
        return ast.parse(expr).body[0].value

    @staticmethod
    def func_call(name, *args):
        return ast.Call(func=ASTTransformerBase.parse_expr(name).value,
                        args=list(args),
                        keywords=[])

    @staticmethod
    def make_constant(value):
        # Do not use ast.Constant which does not exist in python3.5
        node = ASTTransformerBase.parse_expr('0')
        node.value = value
        return node

    @staticmethod
    def get_targets(node):
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


# First-pass transform
class ASTTransformerPreprocess(ASTTransformerBase):
    def __init__(self,
                 excluded_paremeters=(),
                 is_kernel=True,
                 func=None,
                 is_classfunc=False,
                 arg_features=None):
        super().__init__(func)
        self.local_scopes = []
        self.control_scopes = []
        self.excluded_parameters = excluded_paremeters
        self.is_kernel = is_kernel
        self.arg_features = arg_features
        self.returns = None

    # e.g.: FunctionDef, Module, Global
    def variable_scope(self, *args):
        return ScopeGuard(self.local_scopes, *args)

    # e.g.: For, While
    def control_scope(self):
        return ScopeGuard(self.control_scopes)

    def current_scope(self):
        return self.local_scopes[-1]

    def current_control_scope(self):
        return self.control_scopes[-1]

    def var_declared(self, name):
        for s in self.local_scopes:
            if name in s:
                return True
        return False

    def is_creation(self, name):
        return not self.var_declared(name)

    def create_variable(self, name):
        assert name not in self.current_scope(
        ), "Recreating variables is not allowed"
        self.current_scope().append(name)

    def generic_visit(self, node, body_names=[]):
        assert isinstance(body_names, list)
        for field, old_value in ast.iter_fields(node):
            if field in body_names:
                list_stmt = old_value
                with self.variable_scope(list_stmt):
                    for i, l in enumerate(list_stmt):
                        list_stmt[i] = self.visit(l)

            elif isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        template = 'x.augassign(0, 0)'
        t = ast.parse(template).body[0]
        t.value.func.value = node.target
        t.value.func.value.ctx = ast.Load()
        t.value.args[0] = node.value
        t.value.args[1] = ast.Str(s=type(node.op).__name__, ctx=ast.Load())
        return ast.copy_location(t, node)

    @staticmethod
    def make_single_statement(stmts):
        template = 'if 1: pass'
        t = ast.parse(template).body[0]
        t.body = stmts
        return t

    def visit_Assign(self, node):
        assert (len(node.targets) == 1)
        self.generic_visit(node)

        is_static_assign = isinstance(
            node.value, ast.Call) and ASTResolver.resolve_to(
                node.value.func, ti.static, globals())
        if is_static_assign:
            return node

        if isinstance(node.targets[0], ast.Tuple):
            targets = node.targets[0].elts

            # Create
            stmts = []

            holder = self.parse_stmt('__tmp_tuple = ti.expr_init_list(0, '
                                     f'{len(targets)})')
            holder.value.args[0] = node.value

            stmts.append(holder)

            def tuple_indexed(i):
                indexing = self.parse_stmt('__tmp_tuple[0]')
                self.set_subscript_index(indexing.value,
                                         self.parse_expr("{}".format(i)))
                return indexing.value

            for i, target in enumerate(targets):
                is_local = isinstance(target, ast.Name)
                if is_local and self.is_creation(target.id):
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
                    self.create_variable(var_name)
                    stmts.append(ast.Assign(targets=[target], value=rhs))
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
            stmts.append(self.parse_stmt('del __tmp_tuple'))
            return self.make_single_statement(stmts)
        else:
            is_local = isinstance(node.targets[0], ast.Name)
            if is_local and self.is_creation(node.targets[0].id):
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
                self.create_variable(var_name)
                return ast.copy_location(
                    ast.Assign(targets=node.targets, value=rhs), node)
            else:
                # Assign
                node.targets[0].ctx = ast.Load()
                func = ast.Attribute(value=node.targets[0],
                                     attr='assign',
                                     ctx=ast.Load())
                call = ast.Call(func=func, args=[node.value], keywords=[])
                return ast.copy_location(ast.Expr(value=call), node)

    def visit_Try(self, node):
        raise TaichiSyntaxError(
            "Keyword 'try' not supported in Taichi kernels")

    def visit_While(self, node):
        if node.orelse:
            raise TaichiSyntaxError(
                "'else' clause for 'while' not supported in Taichi kernels")

        with self.control_scope():
            self.current_control_scope().append('while')

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

            self.generic_visit(t, ['body'])
            return ast.copy_location(t, node)

    def visit_block(self, list_stmt):
        for i, l in enumerate(list_stmt):
            list_stmt[i] = self.visit(l)

    def visit_If(self, node):
        self.generic_visit(node, ['body', 'orelse'])

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

    def check_loop_var(self, loop_var):
        if self.var_declared(loop_var):
            raise TaichiSyntaxError(
                "Variable '{}' is already declared in the outer scope and cannot be used as loop variable"
                .format(loop_var))

    def visit_static_for(self, node, is_grouped):
        # for i in ti.static(range(n))
        # for i, j in ti.static(ti.ndrange(n))
        # for I in ti.static(ti.grouped(ti.ndrange(n, m)))

        self.current_control_scope().append('static')
        self.generic_visit(node, ['body'])
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
            t.body[3].iter.args[0].args[0] = self.parse_expr('__ndrange_arg')
        else:
            t = self.parse_stmt('if 1: pass; del a')
            t.body[0] = node
        target = copy.deepcopy(node.target)
        target.ctx = ast.Del()
        if isinstance(target, ast.Tuple):
            for tar in target.elts:
                tar.ctx = ast.Del()
        t.body[-1].targets = [target]
        return t

    def visit_range_for(self, node):
        # for i in range(n)
        self.generic_visit(node, ['body'])
        loop_var = node.target.id
        self.check_loop_var(loop_var)
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
            bgn = node.iter.args[0]
            end = node.iter.args[1]
        else:
            bgn = self.make_constant(value=0)
            end = node.iter.args[0]

        t.body[1].value.args[0] = bgn
        t.body[2].value.args[0] = end
        t.body = t.body[:6] + node.body + t.body[6:]
        t.body.append(self.parse_stmt('del {}'.format(loop_var)))
        return ast.copy_location(t, node)

    def visit_ndrange_for(self, node):
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
        t_loop.iter.args[0] = self.parse_expr(
            f'__ndrange{id(node)}.acc_dimensions[0]')
        targets = self.get_targets(node)
        targets_tmp = ['__' + name for name in targets]
        loop_body = t_loop.body
        for i in range(len(targets)):
            if i + 1 < len(targets):
                stmt = '{} = __I // __ndrange{}.acc_dimensions[{}]'.format(
                    targets_tmp[i], id(node), i + 1)
            else:
                stmt = '{} = __I'.format(targets_tmp[i])
            loop_body.append(self.parse_stmt(stmt))
            stmt = '{} = {} + __ndrange{}.bounds[{}][0]'.format(
                targets[i], targets_tmp[i], id(node), i)
            loop_body.append(self.parse_stmt(stmt))
            if i + 1 < len(targets):
                stmt = '__I = __I - {} * __ndrange{}.acc_dimensions[{}]'.format(
                    targets_tmp[i], id(node), i + 1)
                loop_body.append(self.parse_stmt(stmt))
        loop_body += node.body

        node = ast.copy_location(t, node)
        return self.visit(node)  # further translate as a range for

    def visit_grouped_ndrange_for(self, node):
        # for I in ti.grouped(ti.ndrange(n, m))
        self.generic_visit(node, ['body'])
        target = node.target.id
        template = '''
if ti.static(1):
    __ndrange = 0
    {} = ti.expr_init(ti.Vector([0] * len(__ndrange.dimensions)))
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
        t.body[0].value = node.iter.args[0]
        cut = len(t.body) - 1
        t.body = t.body[:cut] + node.body + t.body[cut:]
        return ast.copy_location(t, node)

    def visit_struct_for(self, node, is_grouped):
        # for i, j in x
        # for I in ti.grouped(x)
        self.generic_visit(node, ['body'])
        targets = self.get_targets(node)

        for loop_var in targets:
            self.check_loop_var(loop_var)

        var_decl = ''.join(
            '    {} = ti.Expr(ti.core.make_id_expr(""))\n'.format(name)
            for name in targets)  # indent: 4 spaces
        vars = ', '.join(targets)
        if is_grouped:
            template = '''
if 1:
    ___loop_var = 0
    {} = ti.lang.expr.make_var_vector(size=len(___loop_var.loop_range().shape))
    ___expr_group = ti.lang.expr.make_expr_group({})
    ti.begin_frontend_struct_for(___expr_group, ___loop_var.loop_range())
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
    ti.begin_frontend_struct_for(___expr_group, ___loop_var.loop_range())
    ti.core.end_frontend_range_for()
            '''.format(var_decl, vars)
            t = ast.parse(template).body[0]
            cut = len(targets) + 3
            t.body[cut - 3].value = node.iter
            t.body = t.body[:cut] + node.body + t.body[cut:]
        for loop_var in reversed(targets):
            t.body.append(self.parse_stmt('del {}'.format(loop_var)))
        return ast.copy_location(t, node)

    def visit_For(self, node):
        if node.orelse:
            raise TaichiSyntaxError(
                "'else' clause for 'for' not supported in Taichi kernels")

        with self.control_scope():
            self.current_control_scope().append('for')

            decorator = self.get_decorator(node.iter)
            double_decorator = ''
            if decorator != '' and len(node.iter.args) == 1:
                double_decorator = self.get_decorator(node.iter.args[0])
            ast.fix_missing_locations(node)

            if decorator == 'static':
                if double_decorator == 'static':
                    raise TaichiSyntaxError("'ti.static' cannot be nested")
                return self.visit_static_for(node,
                                             double_decorator == 'grouped')
            elif decorator == 'ndrange':
                if double_decorator != '':
                    raise TaichiSyntaxError(
                        "No decorator is allowed inside 'ti.ndrange")
                return self.visit_ndrange_for(node)
            elif decorator == 'grouped':
                if double_decorator == 'static':
                    raise TaichiSyntaxError(
                        "'ti.static' is not allowed inside 'ti.grouped'")
                elif double_decorator == 'ndrange':
                    return self.visit_grouped_ndrange_for(node)
                elif double_decorator == 'grouped':
                    raise TaichiSyntaxError("'ti.grouped' cannot be nested")
                else:
                    return self.visit_struct_for(node, is_grouped=True)
            elif isinstance(node.iter, ast.Call) and isinstance(
                    node.iter.func, ast.Name) and node.iter.func.id == 'range':
                return self.visit_range_for(node)
            else:  # Struct for
                return self.visit_struct_for(node, is_grouped=False)

    @staticmethod
    def get_subscript_index(node):
        assert isinstance(node, ast.Subscript), type(node)
        # ast.Index has been deprecated in Python 3.9,
        # use the index value directly instead :)
        if isinstance(node.slice, ast.Index):
            return node.slice.value
        return node.slice

    @staticmethod
    def set_subscript_index(node, value):
        assert isinstance(node, ast.Subscript), type(node)
        if isinstance(node.slice, ast.Index):
            node.slice.value = value
        else:
            node.slice = value

    def visit_Subscript(self, node):
        self.generic_visit(node)

        value = node.value
        indices = self.get_subscript_index(node)
        if isinstance(indices, ast.Tuple):
            indices = indices.elts
        else:
            indices = [indices]

        call = ast.Call(func=self.parse_expr('ti.subscript'),
                        args=[value] + indices,
                        keywords=[])
        return ast.copy_location(call, node)

    def visit_IfExp(self, node):
        self.generic_visit(node)

        call = ast.Call(func=self.parse_expr('ti.select'),
                        args=[node.test, node.body, node.orelse],
                        keywords=[])
        return ast.copy_location(call, node)

    def visit_Break(self, node):
        if 'static' in self.current_control_scope():
            return node
        else:
            return self.parse_stmt('ti.core.insert_break_stmt()')

    def visit_Continue(self, node):
        if 'static' in self.current_control_scope():
            return node
        else:
            return self.parse_stmt('ti.core.insert_continue_stmt()')

    def visit_Call(self, node):
        if not ASTResolver.resolve_to(node.func, ti.static, globals()):
            # Do not apply the generic visitor if the function called is ti.static
            self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == 'print':
                node.func = self.parse_expr('ti.ti_print')
            elif func_name == 'min':
                node.func = self.parse_expr('ti.ti_min')
            elif func_name == 'max':
                node.func = self.parse_expr('ti.ti_max')
            elif func_name == 'int':
                node.func = self.parse_expr('ti.ti_int')
            elif func_name == 'float':
                node.func = self.parse_expr('ti.ti_float')
            elif func_name == 'any':
                node.func = self.parse_expr('ti.ti_any')
            elif func_name == 'all':
                node.func = self.parse_expr('ti.ti_all')
            else:
                pass
        return node

    def visit_Module(self, node):
        with self.variable_scope():
            self.generic_visit(node)
        return node

    def visit_Global(self, node):
        with self.variable_scope():
            self.generic_visit(node)
        for name in node.names:
            self.create_variable(name)
        return node

    def visit_FunctionDef(self, node):
        args = node.args
        assert args.vararg is None
        assert args.kwonlyargs == []
        assert args.kw_defaults == []
        assert args.kwarg is None

        arg_decls = []

        def transform_as_kernel():
            # Treat return type
            if node.returns is not None:
                ret_init = self.parse_stmt(
                    'ti.lang.kernel_arguments.decl_scalar_ret(0)')
                ret_init.value.args[0] = node.returns
                self.returns = node.returns
                arg_decls.append(ret_init)
                node.returns = None

            for i, arg in enumerate(args.args):
                # Directly pass in template arguments,
                # such as class instances ("self"), fields, SNodes, etc.
                if isinstance(self.func.argument_annotations[i], ti.template):
                    continue
                if isinstance(self.func.argument_annotations[i], ti.ext_arr):
                    arg_init = self.parse_stmt(
                        'x = ti.lang.kernel_arguments.decl_ext_arr_arg(0, 0)')
                    arg_init.targets[0].id = arg.arg
                    self.create_variable(arg.arg)
                    array_dt = self.arg_features[i][0]
                    array_dim = self.arg_features[i][1]
                    array_dt = to_taichi_type(array_dt)
                    dt_expr = 'ti.' + ti.core.data_type_name(array_dt)
                    dt = self.parse_expr(dt_expr)
                    arg_init.value.args[0] = dt
                    arg_init.value.args[1] = self.parse_expr(
                        "{}".format(array_dim))
                    arg_decls.append(arg_init)
                else:
                    arg_init = self.parse_stmt(
                        'x = ti.lang.kernel_arguments.decl_scalar_arg(0)')
                    arg_init.targets[0].id = arg.arg
                    dt = arg.annotation
                    arg_init.value.args[0] = dt
                    arg_decls.append(arg_init)
            # remove original args
            node.args.args = []

        if self.is_kernel:  # ti.kernel
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
                    if isinstance(self.func.argument_annotations[i],
                                  ti.template):
                        continue
                    # Create a copy for non-template arguments,
                    # so that they are passed by value.
                    arg_init = self.parse_stmt('x = ti.expr_init_func(0)')
                    arg_init.targets[0].id = arg.arg
                    self.create_variable(arg.arg)
                    arg_init.value.args[0] = self.parse_expr(arg.arg +
                                                             '_by_value__')
                    args.args[i].arg += '_by_value__'
                    arg_decls.append(arg_init)

        with self.variable_scope():
            self.generic_visit(node)

        node.body = arg_decls + node.body
        node.body = [self.parse_stmt('import taichi as ti')] + node.body
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            # Python does not support overloading logical and & or
            new_node = self.parse_expr('ti.logical_not(0)')
            new_node.args[0] = node.operand
            node = new_node
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        comparators = [node.left] + node.comparators
        ops = []
        for i in range(len(node.ops)):
            if isinstance(node.ops[i], ast.Lt):
                op_str = 'Lt'
            elif isinstance(node.ops[i], ast.LtE):
                op_str = 'LtE'
            elif isinstance(node.ops[i], ast.Gt):
                op_str = 'Gt'
            elif isinstance(node.ops[i], ast.GtE):
                op_str = 'GtE'
            elif isinstance(node.ops[i], ast.Eq):
                op_str = 'Eq'
            elif isinstance(node.ops[i], ast.NotEq):
                op_str = 'NotEq'
            elif isinstance(node.ops[i], ast.In):
                raise TaichiSyntaxError(
                    '"in" is not supported in Taichi kernels.')
            elif isinstance(node.ops[i], ast.NotIn):
                raise TaichiSyntaxError(
                    '"not in" is not supported in Taichi kernels.')
            elif isinstance(node.ops[i], ast.Is):
                raise TaichiSyntaxError(
                    '"is" is not supported in Taichi kernels.')
            elif isinstance(node.ops[i], ast.IsNot):
                raise TaichiSyntaxError(
                    '"is not" is not supported in Taichi kernels.')
            else:
                raise Exception(f'Unknown operator {node.ops[i]}')
            ops += [ast.copy_location(ast.Str(s=op_str), node)]

        call = ast.Call(
            func=self.parse_expr('ti.chain_compare'),
            args=[
                ast.copy_location(ast.List(elts=comparators, ctx=ast.Load()),
                                  node),
                ast.copy_location(ast.List(elts=ops, ctx=ast.Load()), node)
            ],
            keywords=[])
        call = ast.copy_location(call, node)
        return call

    def visit_BoolOp(self, node):
        self.generic_visit(node)

        def make_node(a, b, token):
            new_node = self.parse_expr('ti.logical_{}(0, 0)'.format(token))
            new_node.args[0] = a
            new_node.args[1] = b
            return new_node

        token = ''
        if isinstance(node.op, ast.And):
            token = 'and'
        elif isinstance(node.op, ast.Or):
            token = 'or'
        else:
            print(node.op)
            print("BoolOp above not implemented")
            exit(0)

        new_node = node.values[0]
        for i in range(1, len(node.values)):
            new_node = make_node(new_node, node.values[i], token)

        return new_node

    def _is_string_mod_args(self, msg):
        # 1. str % (a, b, c, ...)
        # 2. str % single_item
        # Note that |msg.right| may not be a tuple.
        return isinstance(msg, ast.BinOp) and isinstance(
            msg.left, ast.Str) and isinstance(msg.op, ast.Mod)

    def _handle_string_mod_args(self, msg):
        assert self._is_string_mod_args(msg)
        s = msg.left.s
        t = None
        if isinstance(msg.right, ast.Tuple):
            t = msg.right
        else:
            # assuming the format is `str % single_item`
            t = ast.Tuple(elts=[msg.right], ctx=ast.Load())
        self.generic_visit(t)
        return s, t

    def visit_Assert(self, node):
        is_str_mod = False
        if node.msg is not None:
            if isinstance(node.msg, ast.Constant):
                msg = node.msg.value
            elif isinstance(node.msg, ast.Str):
                msg = node.msg.s
            elif self._is_string_mod_args(node.msg):
                # Delay the handling until we call generic_visit() on |node|.
                is_str_mod = True
            else:
                raise ValueError(
                    f"assert info must be constant, not {ast.dump(node.msg)}")
        else:
            import astor
            msg = astor.to_source(node.test)
        self.generic_visit(node)

        extra_args = ast.List(elts=[], ctx=ast.Load())
        if is_str_mod:
            msg, extra_args = self._handle_string_mod_args(node.msg)

        new_node = self.parse_stmt('ti.ti_assert(0, 0, [])')
        new_node.value.args[0] = node.test
        new_node.value.args[1] = self.parse_expr("'{}'".format(msg.strip()))
        new_node.value.args[2] = extra_args
        new_node = ast.copy_location(new_node, node)
        return new_node

    def visit_Return(self, node):
        self.generic_visit(node)
        if self.is_kernel or impl.get_runtime().experimental_real_function:
            # TODO: check if it's at the end of a kernel, throw TaichiSyntaxError if not
            if node.value is not None:
                if self.returns is None:
                    raise TaichiSyntaxError(
                        f'A {"kernel" if self.is_kernel else "function"} '
                        'with a return value must be annotated '
                        'with a return type, e.g. def func() -> ti.f32')
                ret_expr = self.parse_expr('ti.cast(ti.Expr(0), 0)')
                ret_expr.args[0].args[0] = node.value
                ret_expr.args[1] = self.returns
                ret_stmt = self.parse_stmt(
                    'ti.core.create_kernel_return(ret.ptr)')
                # For args[0], it is an ast.Attribute, because it loads the
                # attribute, |ptr|, of the expression |ret_expr|. Therefore we
                # only need to replace the object part, i.e. args[0].value
                ret_stmt.value.args[0].value = ret_expr
                return ret_stmt
        return node


# Second-pass transform
class ASTTransformerChecks(ASTTransformerBase):
    def __init__(self, func):
        super().__init__(func)
        self.has_return = False
        self.in_static_if = False

    def visit_If(self, node):
        node.test = self.visit(node.test)

        old_in_static_if = self.in_static_if
        self.in_static_if = self.get_decorator(node.test) == 'static'

        node.body = list(map(self.visit, node.body))
        if node.orelse is not None:
            node.orelse = list(map(self.visit, node.orelse))

        self.in_static_if = old_in_static_if

        return node

    def visit_Return(self, node):
        if self.in_static_if:  # we can have multiple return in static-if branches
            return node

        if not self.has_return:
            self.has_return = True
        else:
            raise TaichiSyntaxError(
                'Taichi functions/kernels cannot have multiple returns!'
                ' Consider using a local variable to walk around.')

        return node


# Transform a standalone Taichi function call expression into a statement.
class TransformFunctionCallAsStmt(ASTTransformerBase):
    def __init__(self, func):
        super().__init__(func)

    def visit_Call(self, node):
        node.args = [node.func] + node.args
        node.func = self.parse_expr('ti.maybe_transform_ti_func_call_to_stmt')
        return node
