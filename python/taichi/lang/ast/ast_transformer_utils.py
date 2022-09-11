import ast
import builtins
import traceback
from enum import Enum
from sys import version_info
from textwrap import TextWrapper
from typing import List

from taichi.lang import impl
from taichi.lang.exception import (TaichiCompilationError, TaichiNameError,
                                   TaichiSyntaxError,
                                   handle_exception_from_cpp)


class Builder:
    def __call__(self, ctx, node):
        method = getattr(self, 'build_' + node.__class__.__name__, None)
        try:
            if method is None:
                error_msg = f'Unsupported node "{node.__class__.__name__}"'
                raise TaichiSyntaxError(error_msg)
            info = ctx.get_pos_info(node) if isinstance(
                node, (ast.stmt, ast.expr)) else ""
            with impl.get_runtime().src_info_guard(info):
                return method(ctx, node)
        except Exception as e:
            if ctx.raised or not isinstance(node, (ast.stmt, ast.expr)):
                raise e.with_traceback(None)
            ctx.raised = True
            e = handle_exception_from_cpp(e)
            if not isinstance(e, TaichiCompilationError):
                msg = ctx.get_pos_info(node) + traceback.format_exc()
                raise TaichiCompilationError(msg) from None
            msg = ctx.get_pos_info(node) + str(e)
            raise type(e)(msg) from None


class VariableScopeGuard:
    def __init__(self, scopes):
        self.scopes = scopes

    def __enter__(self):
        self.scopes.append({})

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes.pop()


class StaticScopeStatus:
    def __init__(self):
        self.is_in_static_scope = False


class StaticScopeGuard:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        self.prev = self.status.is_in_static_scope
        self.status.is_in_static_scope = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.status.is_in_static_scope = self.prev


class NonStaticControlFlowStatus:
    def __init__(self):
        self.is_in_non_static_control_flow = False


class NonStaticControlFlowGuard:
    def __init__(self, status: NonStaticControlFlowStatus):
        self.status = status

    def __enter__(self):
        self.prev = self.status.is_in_non_static_control_flow
        self.status.is_in_non_static_control_flow = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.status.is_in_non_static_control_flow = self.prev


class LoopStatus(Enum):
    Normal = 0
    Break = 1
    Continue = 2


class LoopScopeAttribute:
    def __init__(self, is_static):
        self.is_static = is_static
        self.status = LoopStatus.Normal
        self.nearest_non_static_if = None


class LoopScopeGuard:
    def __init__(self, scopes, non_static_guard=None):
        self.scopes = scopes
        self.non_static_guard = non_static_guard

    def __enter__(self):
        self.scopes.append(LoopScopeAttribute(self.non_static_guard is None))
        if self.non_static_guard:
            self.non_static_guard.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes.pop()
        if self.non_static_guard:
            self.non_static_guard.__exit__(exc_type, exc_val, exc_tb)


class NonStaticIfGuard:
    def __init__(self, if_node: ast.If, loop_attribute: LoopScopeAttribute,
                 non_static_status: NonStaticControlFlowStatus):
        self.loop_attribute = loop_attribute
        self.if_node = if_node
        self.non_static_guard = NonStaticControlFlowGuard(non_static_status)

    def __enter__(self):
        if self.loop_attribute:
            self.old_non_static_if = self.loop_attribute.nearest_non_static_if
            self.loop_attribute.nearest_non_static_if = self.if_node
        self.non_static_guard.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.loop_attribute:
            self.loop_attribute.nearest_non_static_if = self.old_non_static_if
        self.non_static_guard.__exit__(exc_type, exc_val, exc_tb)


class ReturnStatus(Enum):
    NoReturn = 0
    ReturnedVoid = 1
    ReturnedValue = 2


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
                 start_lineno=None,
                 ast_builder=None,
                 is_real_function=False):
        self.func = func
        self.local_scopes = []
        self.loop_scopes: List[LoopScopeAttribute] = []
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
        self.non_static_control_flow_status = NonStaticControlFlowStatus()
        self.static_scope_status = StaticScopeStatus()
        self.returned = ReturnStatus.NoReturn
        self.ast_builder = ast_builder
        self.visited_funcdef = False
        self.is_real_function = is_real_function

    # e.g.: FunctionDef, Module, Global
    def variable_scope_guard(self):
        return VariableScopeGuard(self.local_scopes)

    # e.g.: For, While
    def loop_scope_guard(self, is_static=False):
        if is_static:
            return LoopScopeGuard(self.loop_scopes)
        return LoopScopeGuard(self.loop_scopes,
                              self.non_static_control_flow_guard())

    def non_static_if_guard(self, if_node: ast.If):
        return NonStaticIfGuard(
            if_node,
            self.current_loop_scope() if self.loop_scopes else None,
            self.non_static_control_flow_status)

    def non_static_control_flow_guard(self):
        return NonStaticControlFlowGuard(self.non_static_control_flow_status)

    def static_scope_guard(self):
        return StaticScopeGuard(self.static_scope_status)

    def current_scope(self):
        return self.local_scopes[-1]

    def current_loop_scope(self):
        return self.loop_scopes[-1]

    def loop_status(self):
        if self.loop_scopes:
            return self.loop_scopes[-1].status
        return LoopStatus.Normal

    def set_loop_status(self, status):
        self.loop_scopes[-1].status = status

    def is_in_static_for(self):
        if self.loop_scopes:
            return self.loop_scopes[-1].is_static
        return False

    def is_in_non_static_control_flow(self):
        return self.non_static_control_flow_status.is_in_non_static_control_flow

    def is_in_static_scope(self):
        return self.static_scope_status.is_in_static_scope

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
        if name in self.global_vars:
            return self.global_vars[name]
        try:
            return getattr(builtins, name)
        except AttributeError:
            raise TaichiNameError(f'Name "{name}" is not defined')

    def get_pos_info(self, node):
        msg = f'On line {node.lineno + self.lineno_offset} of file "{self.file}", in {self.func.func.__name__}:\n'
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
            node_type = node.__class__.__name__

            if node_type in ["For", "While", "FunctionDef", "If"]:
                end_lineno = max(node.body[0].lineno - 1, node.lineno)
            else:
                end_lineno = node.end_lineno

            for i in range(node.lineno - 1, end_lineno):
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
