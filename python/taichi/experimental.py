from taichi.lang.kernel_impl import real_func as _real_func
import warnings


def real_func(func):
    warnings.warn(
        "ti.experimental.real_func is deprecated because it is no longer experimental. " "Use ti.real_func instead.",
        DeprecationWarning,
    )
    return _real_func(func)


__all__ = ["real_func"]
