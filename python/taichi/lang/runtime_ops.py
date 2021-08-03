from taichi.lang import impl


def sync():
    impl.get_runtime().sync()


def async_flush():
    impl.get_runtime().prog.async_flush()
