import ast

from taichi.lang.exception import TaichiSyntaxError


class Builder(object):
    def __call__(self, ctx, node):
        method = getattr(self, 'build_' + node.__class__.__name__, None)
        if method is None:
            try:
                import astpretty
                error_msg = f'Unsupported node {node}:\n{astpretty.pformat(node)}'
            except:
                error_msg = f'Unsupported node {node}'
            raise TaichiSyntaxError(error_msg)
        return method(ctx, node)


def parse_stmt(stmt):
    return ast.parse(stmt).body[0]


def parse_expr(expr):
    return ast.parse(expr).body[0].value


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
                stmt = parse_stmt('del var')
                stmt.targets[0].id = var
                self.stmt_block.append(stmt)
        self.scopes.pop()


class BuilderContext:
    def __init__(self,
                 excluded_parameters=(),
                 is_kernel=True,
                 func=None,
                 arg_features=None):
        self.func = func
        self.local_scopes = []
        self.control_scopes = []
        self.excluded_parameters = excluded_parameters
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

    def check_loop_var(self, loop_var):
        if self.var_declared(loop_var):
            raise TaichiSyntaxError(
                "Variable '{}' is already declared in the outer scope and cannot be used as loop variable"
                .format(loop_var))
