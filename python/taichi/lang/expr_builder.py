import ast

from taichi.lang.ast_resolver import ASTResolver
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.ast_builder_utils import *

import taichi as ti


class ExprBuilder(Builder):
    @staticmethod
    def build_Subscript(ctx, node):
        value = build_expr(ctx, node.value)
        indices = get_subscript_index(node)
        if isinstance(indices, ast.Tuple):
            indices = indices.elts
        else:
            indices = [indices]

        call = ast.Call(func=parse_expr('ti.subscript'),
                        args=[value] + indices,
                        keywords=[])
        return ast.copy_location(call, node)

    @staticmethod
    def build_Compare(ctx, node):
        operands = build_exprs(ctx, [node.left] + list(node.comparators))
        operators = []
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
            operators += [
                ast.copy_location(ast.Str(s=op_str, kind=None), node)]

        call = ast.Call(
            func=parse_expr('ti.chain_compare'),
            args=[
                ast.copy_location(ast.List(elts=operands, ctx=ast.Load()),
                                  node),
                ast.copy_location(ast.List(elts=operators, ctx=ast.Load()),
                                  node)
            ],
            keywords=[])
        call = ast.copy_location(call, node)
        return call

    @staticmethod
    def build_Call(ctx, node):
        if ASTResolver.resolve_to(node.func, ti.static, globals()):
            # Do not modify the expression if the function called is ti.static
            return node
        node.func = build_expr(ctx, node.func)
        node.args = build_exprs(ctx, node.args)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == 'print':
                node.func = parse_expr('ti.ti_print')
            elif func_name == 'min':
                node.func = parse_expr('ti.ti_min')
            elif func_name == 'max':
                node.func = parse_expr('ti.ti_max')
            elif func_name == 'int':
                node.func = parse_expr('ti.ti_int')
            elif func_name == 'float':
                node.func = parse_expr('ti.ti_float')
            elif func_name == 'any':
                node.func = parse_expr('ti.ti_any')
            elif func_name == 'all':
                node.func = parse_expr('ti.ti_all')
            else:
                pass
        return node

    @staticmethod
    def build_IfExp(ctx, node):
        node.test = build_expr(ctx, node.test)
        node.body = build_expr(ctx, node.body)
        node.orelse = build_expr(ctx, node.orelse)

        call = ast.Call(func=parse_expr('ti.select'),
                        args=[node.test, node.body, node.orelse],
                        keywords=[])
        return ast.copy_location(call, node)

    @staticmethod
    def build_UnaryOp(ctx, node):
        node.operand = build_expr(ctx, node.operand)
        if isinstance(node.op, ast.Not):
            # Python does not support overloading logical and & or
            new_node = parse_expr('ti.logical_not(0)')
            new_node.args[0] = node.operand
            node = new_node
        return node

    @staticmethod
    def build_BoolOp(ctx, node):
        node.values = build_exprs(ctx, node.values)

        def make_node(a, b, token):
            new_node = parse_expr('ti.logical_{}(0, 0)'.format(token))
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

    @staticmethod
    def build_BinOp(ctx, node):
        node.left = build_expr(ctx, node.left)
        node.right = build_expr(ctx, node.right)
        return node

    @staticmethod
    def build_List(ctx, node):
        node.elts = build_exprs(ctx, node.elts)
        return node

    @staticmethod
    def build_Tuple(ctx, node):
        node.elts = build_exprs(ctx, node.elts)
        return node

    @staticmethod
    def build_Attribute(ctx, node):
        node.value = build_expr(ctx, node.value)
        return node

    @staticmethod
    def build_Name(ctx, node):
        return node

    @staticmethod
    def build_Constant(ctx, node):
        return node


build_expr = ExprBuilder()


def build_exprs(ctx, exprs):
    return [build_expr(ctx, expr) for expr in list(exprs)]
