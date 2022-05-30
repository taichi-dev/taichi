from taichi.lang import impl


def sync():
    """Blocks the calling thread until all the previously
    launched Taichi kernels have completed.
    """
    impl.get_runtime().sync()


def async_flush():
    impl.get_runtime().prog.async_flush()


__all__ = ['sync']
