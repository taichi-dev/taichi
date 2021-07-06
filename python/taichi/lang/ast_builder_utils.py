import ast


class Builder(object):
    def __call__(self, ctx, node):
        method = getattr(self, 'build_' + node.__class__.__name__, None)
        if method is None:
            raise Exception(f'Unsupported node {node}')
        return method(ctx, node)


def parse_stmt(stmt):
    return ast.parse(stmt).body[0]


def parse_expr(expr):
    return ast.parse(expr).body[0].value


def get_subscript_index(node):
    assert isinstance(node, ast.Subscript), type(node)
    # ast.Index has been deprecated in Python 3.9,
    # use the index value directly instead :)
    if isinstance(node.slice, ast.Index):
        return node.slice.value
    return node.slice


def set_subscript_index(node, value):
    assert isinstance(node, ast.Subscript), type(node)
    if isinstance(node.slice, ast.Index):
        node.slice.value = value
    else:
        node.slice = value


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
    def __init__(self):
        self.local_scopes = []
        self.control_scopes = []

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
