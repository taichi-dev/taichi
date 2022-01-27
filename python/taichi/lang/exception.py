from taichi._lib import core


class TaichiCompilationError(Exception):
    pass


class TaichiSyntaxError(TaichiCompilationError, SyntaxError):
    pass


class TaichiNameError(TaichiCompilationError, NameError):
    pass


class TaichiTypeError(TaichiCompilationError, TypeError):
    pass


class TaichiRuntimeError(RuntimeError):
    pass


class TaichiRuntimeTypeError(TaichiRuntimeError, TypeError):
    def __init__(self, pos, needed, provided):
        message = f'Argument {pos} (type={provided}) cannot be converted into required type {needed}'
        super().__init__(message)


def handle_exception_from_cpp(exc):
    if isinstance(exc, core.TaichiTypeError):
        return TaichiTypeError(str(exc))
    return exc


__all__ = [
    'TaichiSyntaxError', 'TaichiTypeError', 'TaichiCompilationError',
    'TaichiNameError', 'TaichiRuntimeError', 'TaichiRuntimeTypeError'
]
