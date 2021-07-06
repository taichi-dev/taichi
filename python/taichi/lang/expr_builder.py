import ast
import copy

from taichi.lang import impl
from taichi.lang.ast_resolver import ASTResolver
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.ast_builder_utils import *
from taichi.lang.util import to_taichi_type

import taichi as ti


class ExprBuilder(Builder):
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

    @staticmethod
    def build_Subscript(ctx, node):
        value = build_expr(ctx, node.value)
        indices = ExprBuilder.get_subscript_index(node)
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
        operands = [build_expr(ctx, e) for e in
                    [node.left] + list(node.comparators)]
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
            operators += [ast.copy_location(ast.Str(s=op_str), node)]

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
        node.args = [build_expr(ctx, py_arg) for py_arg in node.args]
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
    def build_Name(ctx, node):
        return node

    @staticmethod
    def build_Constant(ctx, node):
        return node


build_expr = ExprBuilder()
