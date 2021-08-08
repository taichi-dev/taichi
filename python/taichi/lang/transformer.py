import ast

from taichi.lang import impl
from taichi.lang.ast_resolver import ASTResolver
from taichi.lang.exception import TaichiSyntaxError

import taichi as ti


# Total transform
class ASTTransformerTotal(object):
    def __init__(self,
                 func=None,
                 excluded_parameters=(),
                 is_kernel=True,
                 arg_features=None):
        self.func = func
        self.excluded_parameters = excluded_parameters
        self.is_kernel = is_kernel
        self.arg_features = arg_features
        self.pass_Checks = ASTTransformerChecks(func=func)

    @staticmethod
    def print_ast(tree, title=None):
        if not impl.get_runtime().print_preprocessed:
            return
        if title is not None:
            print(f'{title}:')
        import astor
        print(astor.to_source(tree.body[0], indent_with='    '))

    def visit(self, tree):
        from taichi.lang.ast_builder_utils import BuilderContext
        from taichi.lang.stmt_builder import build_stmt
        self.print_ast(tree, 'Initial AST')
        ctx = BuilderContext(func=self.func,
                             excluded_parameters=self.excluded_parameters,
                             is_kernel=self.is_kernel,
                             arg_features=self.arg_features)
        # Convert Python AST to Python code that generates Taichi C++ AST.
        tree = build_stmt(ctx, tree)
        ast.fix_missing_locations(tree)
        self.print_ast(tree, 'Preprocessed')
        self.pass_Checks.visit(tree)  # does not modify the AST


class ASTTransformerBase(ast.NodeTransformer):
    def __init__(self, func):
        super().__init__()
        self.func = func

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


# Performs checks at the Python AST level. Does not modify the AST.
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
