import taichi as ti
import functools
import pytest


## Helper functions
def approx(expected, **kwargs):
    '''Tweaked pytest.approx for OpenGL low percisions'''
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

    return pytest.approx(expected, **kwargs)


def allclose(x, y, **kwargs):
    '''Same as: x == approx(y, **kwargs)'''
    return x == approx(y, **kwargs)


def make_temp_file(*args, **kwargs):
    '''Create a temporary file name'''
    import os
    from tempfile import mkstemp
    fd, name = mkstemp(*args, **kwargs)
    os.close(fd)
    return name


## Pytest options
@pytest.fixture(params=ti.supported_archs(), ids=ti.core.arch_name)
def archs(request):
    marker = request.node.get_closest_marker('taichi')

    if arch not in marker.get('archs', {}):
        raise pytest.skip(f'Arch={arch} not included in test')

    ti.init(arch=arch)
    yield


def test(*archs, extensions=[], excludes=[], **options):
    if not len(archs):
        archs = ti.supported_archs()

    def decorator(foo):
        @functools.wraps(foo)
        @pytest.mark.usefixtures('archs')
        @pytest.mark.taichi(archs=archs)
        def wrapped(*args, **kwargs):
            ti.init(arch=wrapped._ti_adhoc.arch, **options)
            foo(*args, **kwargs)

        wrapped._ti_adhoc = lambda x: x
        wrapped._ti_adhoc.archs = archs
        wrapped._ti_adhoc.excludes = excludes
        wrapped._ti_adhoc.extensions = extensions
        wrapped._ti_adhoc.options = options
        return wrapped
    return decorator
