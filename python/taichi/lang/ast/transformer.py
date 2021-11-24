import ast

import astor
from taichi.lang import impl
from taichi.lang.ast.symbol_resolver import ASTResolver
from taichi.lang.ast_builder_utils import ASTBuilderContext
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.ast_builder import ASTBuilder

import taichi as ti


def _print_ast(tree, title=None):
    if not impl.get_runtime().print_preprocessed:
        return
    if title is not None:
        ti.info(f'{title}:')
    print(astor.to_source(tree.body[0], indent_with='    '), flush=True)


def visit_tree(tree, ctx: ASTBuilderContext):
    _print_ast(tree, 'Initial AST')
    tree = ASTBuilder()(ctx, tree)
    ast.fix_missing_locations(tree)
    _print_ast(tree, 'Preprocessed')
    ASTTransformerChecks(func=ctx.func, global_vars=ctx.global_vars).visit(tree)
    return ctx.return_data


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
        self.in_static_if = self.get_decorator(self.globals,
                                               node.test) == 'static'

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
