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
