from taichi.core import ti_core as _ti_core

import taichi as ti


# Helper functions
def get_rel_eps():
    arch = ti.cfg.arch
    if arch == ti.opengl:
        return 1e-3
    elif arch == ti.metal:
        # Debatable, different hardware could yield different precisions
        # On AMD Radeon Pro 5500M, 1e-6 works fine...
        # https://github.com/taichi-dev/taichi/pull/1779
        return 1e-4
    return 1e-6


def approx(expected, **kwargs):
    '''Tweaked pytest.approx for OpenGL low precisions'''
    import pytest

    class boolean_integer:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return bool(self.value) == bool(other)

        def __ne__(self, other):
            return bool(self.value) != bool(other)

    if isinstance(expected, bool):
        return boolean_integer(expected)

    kwargs['rel'] = max(kwargs.get('rel', 1e-6), get_rel_eps())

    return pytest.approx(expected, **kwargs)


def allclose(x, y, **kwargs):
    '''Same as: x == approx(y, **kwargs)'''
    return x == approx(y, **kwargs)


def make_temp_file(*args, **kwargs):
    '''Create a temporary file'''
    import os
    from tempfile import mkstemp
    fd, name = mkstemp(*args, **kwargs)
    os.close(fd)
    return name


# Pytest options
def _get_taichi_archs_fixture():
    import pytest

    @pytest.fixture(params=ti.supported_archs(), ids=_ti_core.arch_name)
    def taichi_archs(request):
        marker = request.node.get_closest_marker('taichi')
        req_arch = request.param

        def ti_init(arch=None, exclude=None, require=None, **options):
            if arch is None:
                arch = []
            if exclude is None:
                exclude = []
            if require is None:
                require = []
            if not isinstance(arch, (list, tuple)):
                arch = [arch]
            if not isinstance(exclude, (list, tuple)):
                exclude = [exclude]
            if not isinstance(require, (list, tuple)):
                require = [require]
            if len(arch) == 0:
                arch = ti.supported_archs()

            if (req_arch not in arch) or (req_arch in exclude):
                raise pytest.skip(f'Arch={req_arch} not included in this test')

            if not all(
                    _ti_core.is_extension_supported(req_arch, e)
                    for e in require):
                raise pytest.skip(
                    f'Arch={req_arch} some extension(s) not satisfied')

            ti.init(arch=req_arch, **options)

        ti_init(*marker.args, **marker.kwargs)
        yield

    return taichi_archs


def test(*args, **kwargs):
    '''
.. function:: ti.test(arch=[], exclude=[], require=[], **options)

    :parameter arch: backends to include
    :parameter exclude: backends to exclude
    :parameter require: extensions required
    :parameter options: other options to be passed into ``ti.init``
    '''
    def decorator(foo):
        import functools

        import pytest

        @pytest.mark.usefixtures('taichi_archs')
        @pytest.mark.taichi(*args, **kwargs)
        @functools.wraps(foo)
        def wrapped(*args, **kwargs):
            return foo(*args, **kwargs)

        return wrapped

    return decorator


__all__ = [
    'get_rel_eps',
    'approx',
    'allclose',
    'make_temp_file',
    'test',
]
