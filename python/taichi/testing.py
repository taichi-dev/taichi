import copy
import itertools

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


class TestValue:
    def __init__(self, value, required_extensions):
        self.value = value
        self.required_extensions = required_extensions

    def get_value(self):
        return self.value

    def get_required_extensions(self):
        return self.required_extensions


_test_features = {
    # "packed": [
    #     TestValue(True, []),
    #     TestValue(False, [])
    # ],
    "dynamic_index":
    [TestValue(True, [ti.extension.dynamic_index]),
     TestValue(False, [])]
}


def test(arch=None, exclude=None, require=None, **options):
    '''
.. function:: ti.test(arch=[], exclude=[], require=[], **options)

    :parameter arch: backends to include
    :parameter exclude: backends to exclude
    :parameter require: extensions required
    :parameter options: other options to be passed into ``ti.init``
    '''

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

    def decorator(foo):
        import functools

        @functools.wraps(foo)
        def wrapped(*args, **kwargs):
            params = [ti.supported_archs()]
            params.extend(
                [_test_features[feature] for feature in _test_features])
            param_combinations = list(itertools.product(*params))

            for request_param in param_combinations:
                req_arch = request_param[0]

                if (req_arch not in arch) or (req_arch in exclude):
                    continue

                if not all(
                        _ti_core.is_extension_supported(req_arch, e)
                        for e in require):
                    continue

                skip = False
                current_options = copy.deepcopy(options)
                for i, feature in enumerate(_test_features):
                    feature_value = request_param[i + 1].get_value()
                    feature_require = request_param[
                        i + 1].get_required_extensions()
                    if current_options.get(
                            feature, feature_value) != feature_value or any(
                                not _ti_core.is_extension_supported(
                                    req_arch, e) for e in feature_require):
                        skip = True
                    else:
                        # Fill in the missing feature
                        current_options[feature] = request_param[
                            i + 1].get_value()
                if skip:
                    continue

                ti.init(arch=req_arch, **current_options)
                foo(*args, **kwargs)

        return wrapped

    return decorator


__all__ = [
    'get_rel_eps',
    'approx',
    'allclose',
    'make_temp_file',
    'test',
]
