import taichi as ti


class boolean_integer:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if self.value:
            return bool(other)
        else:
            return not bool(other)


def approx(expected, **kwargs):
    if isinstance(expected, bool):
        return boolean_integer(expected)

    from pytest import approx

    if ti.cfg.arch == ti.opengl:
        kwargs['rel'] = max(kwargs.get('rel', 1e-6), 1e-3)

    return approx(expected, **kwargs)
