import ast
from collections import namedtuple

from taichi.lang.ast_resolver import ASTResolver


def test_ast_resolver_basic():
    # import within the function to avoid polluting the global scope
    import taichi as ti
    node = ast.parse('ti.kernel', mode='eval').body
    assert ASTResolver.resolve_to(node, ti.kernel, locals())


def test_ast_resolver_direct_import():
    from taichi import kernel
    node = ast.parse('kernel', mode='eval').body
    assert ASTResolver.resolve_to(node, kernel, locals())


def test_ast_resolver_alias():
    import taichi
    node = ast.parse('taichi.kernel', mode='eval').body
    assert ASTResolver.resolve_to(node, taichi.kernel, locals())

    import taichi as tc
    node = ast.parse('tc.kernel', mode='eval').body
    assert ASTResolver.resolve_to(node, tc.kernel, locals())


def test_ast_resolver_chain():
    import taichi as ti
    node = ast.parse('ti.lang.ops.atomic_add', mode='eval').body
    assert ASTResolver.resolve_to(node, ti.atomic_add, locals())


def test_ast_resolver_wrong_ti():
    import taichi
    fake_ti = namedtuple('FakeTi', ['kernel'])
    ti = fake_ti(kernel='fake')
    node = ast.parse('ti.kernel', mode='eval').body
    assert not ASTResolver.resolve_to(node, taichi.kernel, locals())
