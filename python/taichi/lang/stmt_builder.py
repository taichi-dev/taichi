import ast

from taichi.lang.ast_resolver import ASTResolver
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.ast_builder_utils import *
from taichi.lang.expr_builder import build_expr

import taichi as ti


class StmtBuilder(Builder):
    @staticmethod
    def make_single_statement(stmts):
        template = 'if 1: pass'
        t = ast.parse(template).body[0]
        t.body = stmts
        return t

    @staticmethod
    def build_AugAssign(ctx, node):
        node.target = build_expr(ctx, node.target)
        node.value = build_expr(ctx, node.value)
        template = 'x.augassign(0, 0)'
        t = ast.parse(template).body[0]
        t.value.func.value = node.target
        t.value.func.value.ctx = ast.Load()
        t.value.args[0] = node.value
        t.value.args[1] = ast.Str(s=type(node.op).__name__, ctx=ast.Load())
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
        assert _is_string_mod_args(msg)
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
        node.targets = [build_expr(ctx, t) for t in list(node.targets)]
        # node.type_comment?

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
                set_subscript_index(indexing.value,
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
                    ast.Assign(targets=node.targets, value=rhs), node)
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


build_stmt = StmtBuilder()
