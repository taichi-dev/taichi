import ast

import astor
from taichi.lang import impl
from taichi.lang.ast.symbol_resolver import ASTResolver
from taichi.lang.ast_builder_utils import IRBuilderContext
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.ir_builder import IRBuilder

import taichi as ti


# Total transform
class ASTTransformerTotal:
    def __init__(self,
                 func=None,
                 excluded_parameters=(),
                 is_kernel=True,
                 arg_features=None,
                 global_vars=None):
        self.func = func
        self.excluded_parameters = excluded_parameters
        self.is_kernel = is_kernel
        self.arg_features = arg_features
        self.pass_checks = ASTTransformerChecks(func=func,
                                                global_vars=global_vars)
        self.global_vars = global_vars

    @staticmethod
    def print_ast(tree, title=None):
        if not impl.get_runtime().print_preprocessed:
            return
        if title is not None:
            ti.info(f'{title}:')
        print(astor.to_source(tree.body[0], indent_with='    '), flush=True)

    def visit(self, tree, *arguments):
        self.print_ast(tree, 'Initial AST')
        ctx = IRBuilderContext(func=self.func,
                               excluded_parameters=self.excluded_parameters,
                               is_kernel=self.is_kernel,
                               arg_features=self.arg_features,
                               global_vars=self.global_vars,
                               argument_data=arguments)
        # Convert Python AST to Python code that generates Taichi C++ AST.

        tree = IRBuilder()(ctx, tree)
        ast.fix_missing_locations(tree)
        self.print_ast(tree, 'Preprocessed')
        self.pass_checks.visit(tree)  # does not modify the AST
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
