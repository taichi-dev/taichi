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
