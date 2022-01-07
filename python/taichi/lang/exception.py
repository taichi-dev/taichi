from taichi._lib import core


class TaichiCompilationError(Exception):
    pass


class TaichiSyntaxError(TaichiCompilationError, SyntaxError):
    pass


class TaichiNameError(TaichiCompilationError, NameError):
    pass


class TaichiTypeError(TaichiCompilationError, TypeError):
    pass


class InvalidOperationError(Exception):
    pass


def handle_exception_from_cpp(exc):
    if isinstance(exc, core.TaichiTypeError):
        return TaichiTypeError(str(exc))
    return exc
