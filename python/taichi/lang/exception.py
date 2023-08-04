from taichi._lib import core, ccore


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


_TIE_ERROR_TO_PYTHON_EXCEPTION = {
    ccore.TIE_ERROR_INVALID_ARGUMENT: RuntimeError,
    ccore.TIE_ERROR_INVALID_RETURN_ARG: RuntimeError,
    ccore.TIE_ERROR_INVALID_HANDLE: RuntimeError,
    ccore.TIE_ERROR_INVALID_INDEX: RuntimeError,
    ccore.TIE_ERROR_TAICHI_TYPE_ERROR: TaichiTypeError,
    ccore.TIE_ERROR_TAICHI_SYNTAX_ERROR: TaichiSyntaxError,
    ccore.TIE_ERROR_TAICHI_INDEX_ERROR: TaichiIndexError,
    ccore.TIE_ERROR_TAICHI_RUNTIME_ERROR: TaichiRuntimeError,
    ccore.TIE_ERROR_TAICHI_ASSERTION_ERROR: TaichiAssertionError,
    ccore.TIE_ERROR_CALLBACK_FAILED: lambda ex: ex,
    ccore.TIE_ERROR_OUT_OF_MEMORY: RuntimeError,
    ccore.TIE_ERROR_UNKNOWN_CXX_EXCEPTION: RuntimeError,
}


def handle_exception_from_cpp(exc):
    if isinstance(exc, core.TaichiTypeError):
        return TaichiTypeError(str(exc))
    if isinstance(exc, core.TaichiSyntaxError):
        return TaichiSyntaxError(str(exc))
    if isinstance(exc, core.TaichiIndexError):
        return TaichiIndexError(str(exc))
    if isinstance(exc, core.TaichiAssertionError):
        return TaichiAssertionError(str(exc))
    if isinstance(exc, ccore.TieAPIError):
        err, ex = exc.err, exc.ex
        assert err in _TIE_ERROR_TO_PYTHON_EXCEPTION
        return _TIE_ERROR_TO_PYTHON_EXCEPTION[err](ex)
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
