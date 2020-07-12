import taichi as ti
import functools

_has_pytest = False
try:
    import pytest
    _has_pytest = True
except:
    pass


if _has_pytest:

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
    def taichi_archs(request):
        marker = request.node.get_closest_marker('taichi')
        req_arch = request.param

        def ti_init(*archs, extensions=[], excludes=[], **options):
            archs = archs or ti.supported_archs()

            if (req_arch not in archs) or (req_arch in excludes):
                raise pytest.skip(f'Arch={req_arch} not included in test')

            if not all(
                    ti.core.is_extension_supported(req_arch, e)
                    for e in extensions):
                raise pytest.skip(f'Arch={req_arch} some extension not satisfied')

            ti.init(arch=req_arch, **options)

        ti_init(*marker.args, **marker.kwargs)
        yield


    def test(*args, **kwargs):
        def decorator(foo):
            @pytest.mark.usefixtures('taichi_archs')
            @pytest.mark.taichi(*args, **kwargs)
            @functools.wraps(foo)
            def wrapped(*_, **__):
                return foo(*_, **__)

            return wrapped

        return decorator
