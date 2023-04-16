from taichi.lang.impl import get_runtime


def print_memory_profiler_info():
    """Memory profiling tool for LLVM backends with full sparse support.

    This profiler is automatically on.
    """
    get_runtime().materialize()
    get_runtime().prog.print_memory_profiler_info()


__all__ = ["print_memory_profiler_info"]
