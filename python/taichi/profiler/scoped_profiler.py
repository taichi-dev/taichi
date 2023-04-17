from taichi._lib import core as _ti_core


def print_scoped_profiler_info():
    """Print time elapsed on the host tasks in a hierarchical format.

    This profiler is automatically on.

    Call function imports from C++ : _ti_core.print_profile_info()

    Example::

            >>> import taichi as ti
            >>> ti.init(arch=ti.cpu)
            >>> var = ti.field(ti.f32, shape=1)
            >>> @ti.kernel
            >>> def compute():
            >>>     var[0] = 1.0
            >>>     print("Setting var[0] =", var[0])
            >>> compute()
            >>> ti.profiler.print_scoped_profiler_info()
    """
    _ti_core.print_profile_info()


def clear_scoped_profiler_info():
    """Clear profiler's records about time elapsed on the host tasks.

    Call function imports from C++ : _ti_core.clear_profile_info()
    """
    _ti_core.clear_profile_info()


__all__ = ["print_scoped_profiler_info", "clear_scoped_profiler_info"]
