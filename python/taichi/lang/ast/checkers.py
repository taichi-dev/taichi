import ast

from taichi.lang._wrap_inspect import getsourcefile, getsourcelines
from taichi.lang.exception import TaichiSyntaxError


class KernelSimplicityASTChecker(ast.NodeVisitor):
    class ScopeGuard:
        def __init__(self, checker):
            self.c = checker
            self._allows_for_loop = True
            self._allows_more_stmt = True

        @property
        def allows_for_loop(self):
            return self._allows_for_loop

        @property
        def allows_more_stmt(self):
            return self._allows_more_stmt

        def mark_no_more_for_loop(self):
            self._allows_for_loop = False

        def mark_no_more_stmt(self):
            self._allows_for_loop = False
            self._allows_more_stmt = False

        def __enter__(self):
            self.c._scope_guards.append(self)

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.c._scope_guards.pop()

    def __init__(self, func):
        super().__init__()
        self._func_file = getsourcefile(func)
        self._func_lineno = getsourcelines(func)[1]
        self._func_name = func.__name__
        self._scope_guards = []

    def new_scope(self):
        return KernelSimplicityASTChecker.ScopeGuard(self)

    @property
    def current_scope(self):
        return self._scope_guards[-1]

    @property
    def top_level(self):
        return len(self._scope_guards) == 0

    def get_error_location(self, node):
        # -1 because ast's lineno is 1-based.
        lineno = self._func_lineno + node.lineno - 1
        return f"file={self._func_file} kernel={self._func_name} line={lineno}"

    @staticmethod
    def should_check(node):
        if not isinstance(node, ast.stmt):
            return False
        # TODO(#536): Frontend pass should help make sure |func| is a valid AST for
        # Taichi.
        ignored = [ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]
        return not any(map(lambda t: isinstance(node, t), ignored))

    def generic_visit(self, node):
        if not self.should_check(node):
            super().generic_visit(node)
            return

        if not (self.top_level or self.current_scope.allows_more_stmt):
            raise TaichiSyntaxError(f"No more statements allowed, at {self.get_error_location(node)}")
        old_top_level = self.top_level
        if old_top_level:
            self._scope_guards.append(self.new_scope())
        # Marking here before the visit has the effect of disallow for-loops in
        # nested blocks. E.g. if |node| is a IfStmt, then the checker would disallow
        # for-loops inside it.
        self.current_scope.mark_no_more_for_loop()
        super().generic_visit(node)
        if old_top_level:
            self._scope_guards.pop()

    @staticmethod
    def visit_for(node):
        # TODO: since autodiff is enhanced, AST checker rules should be relaxed. This part should be updated.
        # original code is #def visit_For(self, node) without #@staticmethod  before fix pylint R0201
        return
        # is_static = (isinstance(node.iter, ast.Call)
        #              and isinstance(node.iter.func, ast.Attribute)
        #              and isinstance(node.iter.func.value, ast.Name)
        #              and node.iter.func.value.id == 'ti'
        #              and node.iter.func.attr == 'static')
        # if not (self.top_level or self.current_scope.allows_for_loop
        #         or is_static):
        #     raise TaichiSyntaxError(
        #         f'No more for loops allowed, at {self.get_error_location(node)}'
        #     )
        # with self.new_scope():
        #     super().generic_visit(node)
        #
        # if not (self.top_level or is_static):
        #     self.current_scope.mark_no_more_stmt()
