import ast

from taichi.lang.ast_resolver import ASTResolver
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.ast_builder_utils import *
from taichi.lang.expr_builder import build_expr

import taichi as ti


class StmtBuilder(Builder):
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


build_stmt = StmtBuilder()
