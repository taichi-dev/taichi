from taichi._lib import core


class TaichiCompilationError(Exception):
    """Base class for all compilation exceptions.
    """
    pass


class TaichiSyntaxError(TaichiCompilationError, SyntaxError):
    """Thrown when a syntax error is found during compilation.
    """
    pass


class TaichiNameError(TaichiCompilationError, NameError):
    """Thrown when an undefine name is found during compilation.
    """
    pass


class TaichiTypeError(TaichiCompilationError, TypeError):
    """Thrown when a type mismatch is found during compilation.
    """
    pass


class TaichiRuntimeError(RuntimeError):
    """Thrown when the compiled program cannot be executed due to unspecified reasons.
    """
    pass


class TaichiRuntimeTypeError(TaichiRuntimeError, TypeError):
    def __init__(self, pos, needed, provided):
        message = f'Argument {pos} (type={provided}) cannot be converted into required type {needed}'
        super().__init__(message)


def handle_exception_from_cpp(exc):
    if isinstance(exc, core.TaichiTypeError):
        return TaichiTypeError(str(exc))
    if isinstance(exc, core.TaichiSyntaxError):
        return TaichiSyntaxError(str(exc))
    return exc


__all__ = [
    'TaichiSyntaxError', 'TaichiTypeError', 'TaichiCompilationError',
    'TaichiNameError', 'TaichiRuntimeError', 'TaichiRuntimeTypeError'
]
