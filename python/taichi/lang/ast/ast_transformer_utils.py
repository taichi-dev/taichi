import ast
from enum import Enum
from sys import version_info
from textwrap import TextWrapper

import astor
from taichi._logging import info
from taichi.lang import impl
from taichi.lang.exception import TaichiCompilationError, TaichiSyntaxError


class Builder:
    def __call__(self, ctx, node):
        method = getattr(self, 'build_' + node.__class__.__name__, None)
        try:
            if method is None:
                try:
                    import astpretty  # pylint: disable=C0415
                    error_msg = f'Unsupported node {node}:\n{astpretty.pformat(node)}'
                except:
                    error_msg = f'Unsupported node {node}'
                raise TaichiSyntaxError(error_msg)
            return method(ctx, node)
        except Exception as e:
            if ctx.raised or not isinstance(node, (ast.stmt, ast.expr)):
                raise e
            msg = str(e)
            if not isinstance(e, TaichiCompilationError):
                msg = f"{e.__class__.__name__}: " + msg
            msg = ctx.get_pos_info(node) + msg
            ctx.raised = True
            raise TaichiCompilationError(msg)


class VariableScopeGuard:
    def __init__(self, scopes):
        self.scopes = scopes

    def __enter__(self):
        self.scopes.append({})

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes.pop()


class LoopStatus(Enum):
    Normal = 0
    Break = 1
    Continue = 2


class ControlScopeAttribute:
    def __init__(self):
        self.is_static = False
        self.status = LoopStatus.Normal


class ControlScopeGuard:
    def __init__(self, scopes):
        self.scopes = scopes

    def __enter__(self):
        self.scopes.append(ControlScopeAttribute())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes.pop()


class ASTTransformerContext:
    def __init__(self,
                 excluded_parameters=(),
                 is_kernel=True,
                 func=None,
                 arg_features=None,
                 global_vars=None,
                 argument_data=None,
                 file=None,
                 src=None,
                 start_lineno=None):
        self.func = func
        self.local_scopes = []
        self.control_scopes = []
        self.excluded_parameters = excluded_parameters
        self.is_kernel = is_kernel
        self.arg_features = arg_features
        self.returns = None
        self.global_vars = global_vars
        self.argument_data = argument_data
        self.return_data = None
        self.file = file
        self.src = src
        self.indent = 0
        for c in self.src[0]:
            if c == ' ':
                self.indent += 1
            else:
                break
        self.lineno_offset = start_lineno - 1
        self.raised = False

    # e.g.: FunctionDef, Module, Global
    def variable_scope_guard(self):
        return VariableScopeGuard(self.local_scopes)

    # e.g.: For, While
    def control_scope_guard(self):
        return ControlScopeGuard(self.control_scopes)

    def current_scope(self):
        return self.local_scopes[-1]

    def current_control_scope(self):
        return self.control_scopes[-1]

    def loop_status(self):
        if len(self.control_scopes):
            return self.control_scopes[-1].status
        return LoopStatus.Normal

    def set_loop_status(self, status):
        self.control_scopes[-1].status = status

    def set_static_loop(self):
        self.control_scopes[-1].is_static = True

    def is_in_static(self):
        if len(self.control_scopes):
            return self.control_scopes[-1].is_static
        return False

    def is_var_declared(self, name):
        for s in self.local_scopes:
            if name in s:
                return True
        return False

    def create_variable(self, name, var):
        if name in self.current_scope():
            raise TaichiSyntaxError("Recreating variables is not allowed")
        self.current_scope()[name] = var

    def check_loop_var(self, loop_var):
        if self.is_var_declared(loop_var):
            raise TaichiSyntaxError(
                f"Variable '{loop_var}' is already declared in the outer scope and cannot be used as loop variable"
            )

    def get_var_by_name(self, name):
        for s in reversed(self.local_scopes):
            if name in s:
                return s[name]
        return self.global_vars.get(name)

    def get_pos_info(self, node):
        msg = f'On line {node.lineno + self.lineno_offset} of file "{self.file}":\n'
        if version_info < (3, 8):
            msg += self.src[node.lineno - 1] + "\n"
            return msg
        col_offset = self.indent + node.col_offset
        end_col_offset = self.indent + node.end_col_offset

        wrapper = TextWrapper(width=80)

        def gen_line(code, hint):
            hint += ' ' * (len(code) - len(hint))
            code = wrapper.wrap(code)
            hint = wrapper.wrap(hint)
            if not len(code):
                return "\n\n"
            return "".join([c + '\n' + h + '\n' for c, h in zip(code, hint)])

        if node.lineno == node.end_lineno:
            hint = ' ' * col_offset + '^' * (end_col_offset - col_offset)
            msg += gen_line(self.src[node.lineno - 1], hint)
        else:
            for i in range(node.lineno - 1, node.end_lineno):
                last = len(self.src[i])
                while last > 0 and (self.src[i][last - 1].isspace() or
                                    not self.src[i][last - 1].isprintable()):
                    last -= 1
                first = 0
                while first < len(self.src[i]) and (
                        self.src[i][first].isspace()
                        or not self.src[i][first].isprintable()):
                    first += 1
                if i == node.lineno - 1:
                    hint = ' ' * col_offset + '^' * (last - col_offset)
                elif i == node.end_lineno - 1:
                    hint = ' ' * first + '^' * (end_col_offset - first)
                elif first < last:
                    hint = ' ' * first + '^' * (last - first)
                else:
                    hint = ''
                msg += gen_line(self.src[i], hint)
        return msg