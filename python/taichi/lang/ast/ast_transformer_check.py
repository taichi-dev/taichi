import ast

from taichi.lang import impl
from taichi.lang.ast.symbol_resolver import ASTResolver
from taichi.lang.exception import TaichiSyntaxError

import taichi as ti


# Performs checks at the Python AST level. Does not modify the AST.
class ASTTransformerChecks(ast.NodeTransformer):
    def __init__(self, func, global_vars):
        super().__init__()
        self.func = func
        self.has_return = False
        self.in_static_if = False
        self.globals = global_vars

    @staticmethod
    def get_decorator(global_vars, node):
        if not isinstance(node, ast.Call):
            return ''
        for wanted, name in [
            (ti.static, 'static'),
            (ti.grouped, 'grouped'),
            (ti.ndrange, 'ndrange'),
        ]:
            if ASTResolver.resolve_to(node.func, wanted, global_vars):
                return name
        return ''

    def visit_If(self, node):
        node.test = self.visit(node.test)

        old_in_static_if = self.in_static_if
        self.in_static_if = impl.get_runtime(
        ).experimental_real_function or self.get_decorator(
            self.globals, node.test) == 'static'

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
