import ast

import astor
from taichi.lang import impl
from taichi.lang.ast.symbol_resolver import ASTResolver, ModuleResolver
from taichi.lang.ast_builder_utils import BuilderContext, IRBuilderContext
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.ir_builder import IRBuilder
from taichi.lang.stmt_builder import build_stmt

import taichi as ti


# Total transform
class ASTTransformerTotal(object):
    def __init__(self,
                 func=None,
                 excluded_parameters=(),
                 is_kernel=True,
                 arg_features=None,
                 globals=None):
        self.func = func
        self.excluded_parameters = excluded_parameters
        self.is_kernel = is_kernel
        self.arg_features = arg_features
        self.pass_checks = ASTTransformerChecks(func=func)
        self.rename_module = ASTTransformerUnifyModule(func=func)
        self.globals = globals

    @staticmethod
    def print_ast(tree, title=None):
        if not impl.get_runtime().print_preprocessed:
            return
        if title is not None:
            ti.info(f'{title}:')
        print(astor.to_source(tree.body[0], indent_with='    '), flush=True)

    def visit(self, tree, *arguments):
        if impl.get_runtime().experimental_ast_refactor:
            self.print_ast(tree, 'Initial AST')
            self.rename_module.visit(tree)
            self.print_ast(tree, 'AST with module renamed')
            ctx = IRBuilderContext(
                func=self.func,
                excluded_parameters=self.excluded_parameters,
                is_kernel=self.is_kernel,
                arg_features=self.arg_features,
                globals=self.globals,
                argument_data=arguments)
            # Convert Python AST to Python code that generates Taichi C++ AST.

            tree = IRBuilder()(ctx, tree)
            ast.fix_missing_locations(tree)
            self.print_ast(tree, 'Preprocessed')
            self.pass_checks.visit(tree)  # does not modify the AST
            return ctx.return_data
        self.print_ast(tree, 'Initial AST')
        self.rename_module.visit(tree)
        self.print_ast(tree, 'AST with module renamed')
        ctx = BuilderContext(func=self.func,
                             excluded_parameters=self.excluded_parameters,
                             is_kernel=self.is_kernel,
                             arg_features=self.arg_features)
        # Convert Python AST to Python code that generates Taichi C++ AST.
        tree = build_stmt(ctx, tree)
        ast.fix_missing_locations(tree)
        self.print_ast(tree, 'Preprocessed')
        self.pass_checks.visit(tree)  # does not modify the AST


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


class ASTTransformerUnifyModule(ast.NodeTransformer):
    """Rename the module alias to `ti`.
    module func calls like `<module alias>.<func-name>` will be renamed to
    `ti.<func-name>`
    """
    def __init__(self, func):
        super().__init__()
        self.func = func
        # Get the module alias from the global symbols table of the given func.
        self.custom_module_name = ModuleResolver.get_module_name(func.func, ti)
        self.default_module_name = "ti"

    def visit_Name(self, node):
        # Get the id of the ast.Name, rename if it equals to self.module_name.
        if node.id == self.custom_module_name:
            node.id = self.default_module_name
        return node


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
