"""Provides helpers to resolve AST nodes."""
import ast


class ASTResolver:
    """Provides helper methods to resolve AST nodes."""

    @staticmethod
    def resolve_to(node, wanted, scope):
        """Check if symbol ``node`` resolves to ``wanted`` object.

        This is only intended to check if a given AST node resolves to a symbol
        under some namespaces, e.g. the ``a.b.c.foo`` pattern, but not meant for
        more complicated expressions like ``(a + b).foo``.

        Args:
            node (Union[ast.Attribute, ast.Name]): an AST node to be resolved.
            wanted (Any): The expected python object.
            scope (Dict[str, Any]): Maps from symbol names to objects, for
                example, globals()

        Returns:
            bool: The checked result.
        """
        if isinstance(node, ast.Name):
            return scope.get(node.id) is wanted

        if not isinstance(node, ast.Attribute):
            return False

        v = node.value
        chain = [node.attr]
        while isinstance(v, ast.Attribute):
            chain.append(v.attr)
            v = v.value
        if not isinstance(v, ast.Name):
            # Example cases that fall under this branch:
            #
            # x[i].attr: ast.Subscript
            # (a + b).attr: ast.BinOp
            # ...
            return False
        chain.append(v.id)

        for attr in reversed(chain):
            try:
                if isinstance(scope, dict):
                    scope = scope[attr]
                else:
                    scope = getattr(scope, attr)
            except (KeyError, AttributeError):
                return False
        # The name ``scope`` here could be a bit confusing
        return scope is wanted
