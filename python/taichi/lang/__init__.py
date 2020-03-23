from .impl import *
from .matrix import Matrix
from .transformer import TaichiSyntaxError
from .ndrange import ndrange, GroupedNDRange
from copy import deepcopy as _deepcopy
import functools
import os

core = taichi_lang_core
runtime = get_runtime()

i = indices(0)
j = indices(1)
k = indices(2)
l = indices(3)
ij = indices(0, 1)
ijk = indices(0, 1, 2)
ijkl = indices(0, 1, 2, 3)
Vector = Matrix

outer_product = Matrix.outer_product
cross = Matrix.cross
dot = Matrix.dot
normalized = Matrix.normalized

cfg = default_cfg()
current_cfg = current_cfg()
x86_64 = core.x64
x64 = core.x64
arm64 = core.arm64
cuda = core.cuda
metal = core.metal
opengl = core.opengl
profiler_print = lambda: core.get_current_program().profiler_print()
profiler_clear = lambda: core.get_current_program().profiler_clear()
profiler_start = lambda n: core.get_current_program().profiler_start(n)
profiler_stop = lambda: core.get_current_program().profiler_stop()


class _Extension(object):
    def __init__(self):
        self.sparse = core.sparse
        self.data64 = core.data64
        self.adstack = core.adstack


extension = _Extension()
is_supported = core.is_supported


def reset():
    from .impl import reset as impl_reset
    impl_reset()
    global runtime
    runtime = get_runtime()


def init(default_fp=None,
         default_ip=None,
         print_preprocessed=None,
         debug=None,
         **kwargs):
    # Make a deepcopy in case these args reference to items from ti.cfg, which are
    # actually references. If no copy is made and the args are indeed references,
    # ti.reset() could override the args to their default values.
    default_fp = _deepcopy(default_fp)
    default_ip = _deepcopy(default_ip)
    kwargs = _deepcopy(kwargs)
    import taichi as ti
    ti.reset()

    if default_fp is None:  # won't override
        dfl_fp = os.environ.get("TI_DEFAULT_FP")
        if dfl_fp == 32:
            default_fp = core.DataType.f32
        elif dfl_fp == 64:
            default_fp = core.DataType.f64
        elif dfl_fp is not None:
            raise ValueError(
                f'Unrecognized TI_DEFAULT_FP: {dfl_fp}, should be 32 or 64')
    if default_ip is None:
        dfl_ip = os.environ.get("TI_DEFAULT_IP")
        if dfl_ip == 32:
            default_ip = core.DataType.i32
        elif dfl_ip == 64:
            default_ip = core.DataType.i64
        elif dfl_ip is not None:
            raise ValueError(
                f'Unrecognized TI_DEFAULT_IP: {dfl_ip}, should be 32 or 64')

    if print_preprocessed is None:  # won't override
        print_preprocessed = os.environ.get("TI_PRINT_PREPROCESSED")
        if print_preprocessed is not None:
            print_preprocessed = bool(int(print_preprocessed))

    if default_fp is not None:
        ti.get_runtime().set_default_fp(default_fp)
    if default_ip is not None:
        ti.get_runtime().set_default_ip(default_ip)
    if print_preprocessed is not None:
        ti.get_runtime().print_preprocessed = print_preprocessed

    if debug is None:
        debug = bool(int(os.environ.get('TI_DEBUG', '0')))
    if debug:
        ti.set_logging_level(ti.TRACE)
    ti.cfg.debug = debug

    unified_memory = os.environ.get('TI_USE_UNIFIED_MEMORY', '')
    if unified_memory != '':
        use_unified_memory = bool(int(unified_memory))
        ti.cfg.use_unified_memory = use_unified_memory
        if not use_unified_memory:
            ti.trace(
                'Unified memory disabled (env TI_USE_UNIFIED_MEMORY=0). This is experimental.'
            )

    for k, v in kwargs.items():
        setattr(ti.cfg, k, v)

    def boolean_config(key, name=None):
        if name is None:
            name = 'TI_' + key.upper()
        value = os.environ.get(name)
        if value is not None:
            setattr(ti.cfg, key, len(value) and bool(int(value)))

    # does override
    boolean_config("print_ir")
    boolean_config("verbose")
    boolean_config("fast_math")
    arch = os.environ.get("TI_ARCH")
    if arch is not None:
        ti.cfg.arch = ti.core.arch_from_name(arch)

    log_level = os.environ.get("TI_LOG_LEVEL")
    if log_level is not None:
        ti.set_logging_level(log_level.lower())

    ti.get_runtime().create_program()


def cache_shared(v):
    taichi_lang_core.cache(0, v.ptr)


def cache_l1(v):
    taichi_lang_core.cache(1, v.ptr)


parallelize = core.parallelize
serialize = lambda: parallelize(1)
vectorize = core.vectorize
block_dim = core.block_dim
cache = core.cache


def inversed(x):
    return x.inversed()


transposed = Matrix.transposed


def polar_decompose(A, dt=None):
    if dt is None:
        dt = get_runtime().default_fp
    from .linalg import polar_decompose
    return polar_decompose(A, dt)


def svd(A, dt=None):
    if dt is None:
        dt = get_runtime().default_fp
    from .linalg import svd
    return svd(A, dt)


determinant = Matrix.determinant
tr = Matrix.trace


def Tape(loss, clear_gradients=True):
    get_runtime().materialize()
    assert loss.snode().ptr.has_grad(), "gradient for loss not allocated"
    if clear_gradients:
        clear_all_gradients()
    loss[None] = 0
    loss.grad[None] = 1
    return runtime.get_tape(loss)


def clear_all_gradients():
    get_runtime().materialize()

    import taichi as ti

    def visit(node):
        places = []
        for i in range(node.ptr.get_num_ch()):
            ch = node.ptr.get_ch(i)
            if not ch.is_place():
                visit(SNode(ch))
            else:
                if not ch.is_primal():
                    places.append(ch.get_expr())

        places = tuple(places)
        if places:
            from .meta import clear_gradients
            clear_gradients(places)

    visit(ti.root)


schedules = [parallelize, vectorize, block_dim, cache]
lang_core = core


def static_print(*args, __p=print, **kwargs):
    __p(*args, **kwargs)


def benchmark(func, repeat=100, args=()):
    import taichi as ti
    import time
    for i in range(repeat // 3):
        func(*args)  # compile the kernel first
    ti.sync()
    t = time.time()
    for n in range(repeat):
        func(*args)
    ti.get_runtime().sync()
    elapsed = time.time() - t
    return elapsed / repeat


wanted_archs = None


def set_wanted_archs(archs):
    global wanted_archs
    wanted_archs = archs


def supported_archs():
    import taichi as ti
    archs = [ti.core.host_arch()]
    if ti.core.with_cuda():
        archs.append(cuda)
    if ti.core.with_metal():
        archs.append(metal)
    if ti.core.with_opengl():
        archs.append(opengl)
    if wanted_archs is not None:
        archs, old_archs = [], archs
        for arch in old_archs:
            if ti.core.arch_name(arch) in wanted_archs:
                archs.append(arch)
    return archs


class _ArchCheckers(object):
    def __init__(self):
        self._checkers = []

    def register(self, c):
        self._checkers.append(c)

    def __call__(self, arch):
        assert isinstance(arch, core.Arch)
        return all([c(arch) for c in self._checkers])


_tests_arch_checkers_argname = '_tests_arch_checkers'


def _get_or_make_arch_checkers(kwargs):
    k = _tests_arch_checkers_argname
    if k not in kwargs:
        kwargs[k] = _ArchCheckers()
    return kwargs[k]


# test with all archs
def all_archs_with(**kwargs):
    kwargs = _deepcopy(kwargs)

    def decorator(test):
        # @pytest.mark.parametrize decorator only knows about regular function args,
        # without *args or **kwargs. By decorating with @functools.wraps, the
        # signature of |test| is preserved, so that @ti.all_archs can be used after
        # the parametrization decorator.
        #
        # Full discussion: https://github.com/pytest-dev/pytest/issues/6810
        @functools.wraps(test)
        def wrapped(*test_args, **test_kwargs):
            import taichi as ti
            can_run_on = test_kwargs.pop(_tests_arch_checkers_argname,
                                         _ArchCheckers())
            # Filter away archs that don't support 64-bit data.
            fp = kwargs.get('default_fp', ti.f32)
            ip = kwargs.get('default_ip', ti.i32)
            if fp == ti.f64 or ip == ti.i64:
                can_run_on.register(
                    lambda arch: is_supported(arch, extension.data64))

            for arch in ti.supported_archs():
                if can_run_on(arch):
                    print('Running test on arch={}'.format(arch))
                    ti.init(arch=arch, **kwargs)
                    test(*test_args, **test_kwargs)
                else:
                    print('Skipped test on arch={}'.format(arch))

        return wrapped

    return decorator


# test with all archs
def all_archs(test):
    return all_archs_with()(test)


# Exclude the given archs when running the tests
#
# Example usage:
#
# @ti.archs_excluding(ti.cuda, ti.metal)
# def test_xx():
#   ...
#
# @ti.archs_excluding(ti.cuda, default_fp=ti.f64)
# def test_yy():
#   ...
def archs_excluding(*excluded_archs, **kwargs):
    # |kwargs| will be passed to all_archs_with(**kwargs)
    assert all([isinstance(a, core.Arch) for a in excluded_archs])
    excluded_archs = set(excluded_archs)

    def decorator(test):
        @functools.wraps(test)
        def wrapped(*test_args, **test_kwargs):
            def checker(arch):
                return arch not in excluded_archs

            _get_or_make_arch_checkers(test_kwargs).register(checker)
            return all_archs_with(**kwargs)(test)(*test_args, **test_kwargs)

        return wrapped

    return decorator


# Specifies the extension features the archs are required to support in order
# to run the test.
#
# Example usage:
#
# ti.require(ti.extension.data64)
# ti.all_archs_with(default_fp=ti.f64)
# def test_xx():
#   ...
def require(*exts):
    # Because this decorator injects an arch checker, its usage must be followed
    # with all_archs_with(), either directly or indirectly.
    assert all([isinstance(e, core.Extension) for e in exts])

    def decorator(test):
        @functools.wraps(test)
        def wrapped(*test_args, **test_kwargs):
            def checker(arch):
                return all([is_supported(arch, e) for e in exts])

            _get_or_make_arch_checkers(test_kwargs).register(checker)
            test(*test_args, **test_kwargs)

        return wrapped

    return decorator


def archs_support_sparse(test, **kwargs):
    wrapped = all_archs_with(**kwargs)(test)
    return require(extension.sparse)(wrapped)


def torch_test(func):
    import taichi as ti
    if ti.has_pytorch():
        return ti.all_archs(func)
    else:
        return lambda: None


# test with host arch only
def host_arch_only(func):
    import taichi as ti

    def test(*args, **kwargs):
        archs = [ti.core.host_arch()]
        for arch in archs:
            ti.init(arch=arch)
            func(*args, **kwargs)

    return test


def must_throw(ex):
    def decorator(func):
        def func__(*args, **kwargs):
            finishes = False
            try:
                host_arch_only(func)(*args, **kwargs)
                finishes = True
            except ex:
                # throws. test passed
                pass
            except Exception as err_actual:
                assert False, 'Exception {} instead of {} thrown'.format(
                    str(type(err_actual)), str(ex))
            if finishes:
                assert False, 'Test successfully finished instead of throwing {}'.format(
                    str(ex))

        return func__

    return decorator


def complex_kernel(func):
    def decorated(*args, **kwargs):
        get_runtime().inside_complex_kernel = True
        if get_runtime().target_tape:
            get_runtime().target_tape.insert(decorated, args)
        try:
            func(*args, **kwargs)
        finally:
            get_runtime().inside_complex_kernel = False

    decorated.grad = None
    return decorated


def complex_kernel_grad(primal):
    def decorator(func):
        def decorated(*args, **kwargs):
            func(*args, **kwargs)

        primal.grad = decorated
        return decorated

    return decorator


def sync():
    get_runtime().sync()


__all__ = [s for s in dir() if not s.startswith('_')]
