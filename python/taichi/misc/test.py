def approx(expected, **kwargs):
    import taichi as ti
    from pytest import approx

    class boolean_integer:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return bool(self.value) == bool(other)

        def __ne__(self, other):
            return bool(self.value) != bool(other)

    if isinstance(expected, bool):
        return boolean_integer(expected)

    if ti.cfg.arch == ti.opengl:
        kwargs['rel'] = max(kwargs.get('rel', 1e-6), 1e-3)

    return approx(expected, **kwargs)


def allclose(x, y, **kwargs):
    return x == approx(y, **kwargs)
