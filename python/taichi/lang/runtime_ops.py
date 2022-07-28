from taichi.lang import impl


def sync():
    """Blocks the calling thread until all the previously
    launched Taichi kernels have completed.
    """
    impl.get_runtime().sync()


__all__ = ['sync']
