from taichi._lib import core


class TaichiCompilationError(Exception):
    """Base class for all compilation exceptions."""

    pass


class TaichiSyntaxError(TaichiCompilationError, SyntaxError):
    """Thrown when a syntax error is found during compilation."""

    pass


class TaichiNameError(TaichiCompilationError, NameError):
    """Thrown when an undefine name is found during compilation."""

    pass


class TaichiIndexError(TaichiCompilationError, IndexError):
    """Thrown when an index error is found during compilation."""

    pass


class TaichiTypeError(TaichiCompilationError, TypeError):
    """Thrown when a type mismatch is found during compilation."""

    pass


class TaichiRuntimeError(RuntimeError):
    """Thrown when the compiled program cannot be executed due to unspecified reasons."""

    pass


class TaichiAssertionError(TaichiRuntimeError, AssertionError):
    """Thrown when assertion fails at runtime."""

    pass


class TaichiRuntimeTypeError(TaichiRuntimeError, TypeError):
    @staticmethod
    def get(pos, needed, provided):
        return TaichiRuntimeTypeError(
            f"Argument {pos} (type={provided}) cannot be converted into required type {needed}"
        )

    @staticmethod
    def get_ret(needed, provided):
        return TaichiRuntimeTypeError(f"Return (type={provided}) cannot be converted into required type {needed}")


def handle_exception_from_cpp(exc):
    if isinstance(exc, core.TaichiTypeError):
        return TaichiTypeError(str(exc))
    if isinstance(exc, core.TaichiSyntaxError):
        return TaichiSyntaxError(str(exc))
    if isinstance(exc, core.TaichiIndexError):
        return TaichiIndexError(str(exc))
    if isinstance(exc, core.TaichiAssertionError):
        return TaichiAssertionError(str(exc))
    return exc


__all__ = [
    "TaichiSyntaxError",
    "TaichiTypeError",
    "TaichiCompilationError",
    "TaichiNameError",
    "TaichiRuntimeError",
    "TaichiRuntimeTypeError",
    "TaichiAssertionError",
]
