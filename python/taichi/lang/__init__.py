from .impl import *
from .util import deprecated
from .matrix import Matrix, Vector
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

outer_product = deprecated('ti.outer_product(a, b)',
                           'a.outer_product(b)')(Matrix.outer_product)
cross = deprecated('ti.cross(a, b)', 'a.cross(b)')(Matrix.cross)
dot = deprecated('ti.dot(a, b)', 'a.dot(b)')(Matrix.dot)
normalized = deprecated('ti.normalized(a)',
                        'a.normalized()')(Matrix.normalized)

cfg = default_cfg()
current_cfg = current_cfg()
x86_64 = core.x64
x64 = core.x64
arm64 = core.arm64
cuda = core.cuda
metal = core.metal
opengl = core.opengl
gpu = [cuda, metal, opengl]
cpu = core.host_arch()
kernel_profiler_print = lambda: core.get_current_program(
).kernel_profiler_print()
kernel_profiler_clear = lambda: core.get_current_program(
).kernel_profiler_clear()


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


def init(arch=None,
         default_fp=None,
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

    def bool_int(x):
        return bool(int(x))

    def environ_config(key, cast=bool_int):
        name = 'TI_' + key.upper()
        value = os.environ.get(name, '')
        if len(value):
            setattr(ti.cfg, key, cast(value))

        # TI_ASYNC=   : not work
        # TI_ASYNC=0  : False
        # TI_ASYNC=1  : True

    # does override
    environ_config("print_ir")
    environ_config("verbose")
    environ_config("fast_math")
    environ_config("async")
    environ_config("print_benchmark_stat")
    environ_config("device_memory_fraction", float)
    environ_config("device_memory_GB", float)

    # Q: Why not environ_config("gdb_trigger")?
    # A: We don't have ti.cfg.gdb_trigger yet.
    # Discussion: https://github.com/taichi-dev/taichi/pull/879
    gdb_trigger = os.environ.get('TI_GDB_TRIGGER', '')
    if len(gdb_trigger):
        ti.set_gdb_trigger(bool(int(gdb_trigger)))

    advanced_optimization = os.environ.get('TI_ADVANCED_OPTIMIZATION', '')
    if len(advanced_optimization):
        ti.core.toggle_advanced_optimization(bool(int(advanced_optimization)))

    # Q: Why not environ_config("arch", ti.core.arch_from_name)?
    # A: We need adaptive_arch_select for all.
    env_arch = os.environ.get("TI_ARCH")
    if env_arch is not None:
        print(f'Following TI_ARCH setting up for arch={env_arch}')
        arch = ti.core.arch_from_name(env_arch)

    ti.cfg.arch = adaptive_arch_select(arch)

    log_level = os.environ.get("TI_LOG_LEVEL")
    if log_level is not None:
        ti.set_logging_level(log_level.lower())

    ti.get_runtime().create_program()


def cache_shared(v):
    taichi_lang_core.cache(0, v.ptr)


def cache_read_only(v):
    taichi_lang_core.cache(1, v.ptr)


parallelize = core.parallelize
serialize = lambda: parallelize(1)
vectorize = core.vectorize
block_dim = core.block_dim
cache = core.cache

inversed = deprecated('ti.inversed(a)', 'a.inverse()')(Matrix.inversed)
transposed = deprecated('ti.transposed(a)', 'a.transpose()')(Matrix.transposed)


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


determinant = deprecated('ti.determinant(a)',
                         'a.determinant()')(Matrix.determinant)
tr = deprecated('ti.tr(a)', 'a.trace()')(Matrix.trace)


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


def benchmark(func, repeat=300, args=()):
    import taichi as ti
    import time
    # The reason why we run 4 times is to warm up instruction/data caches.
    # Discussion: https://github.com/taichi-dev/taichi/pull/1002#discussion_r426312136
    for i in range(4):
        func(*args)  # compile the kernel first
    ti.sync()
    t = time.time()
    for n in range(repeat):
        func(*args)
    ti.get_runtime().sync()
    elapsed = time.time() - t
    avg = elapsed / repeat * 1000  # miliseconds
    ti.stat_write(avg)


def stat_write(avg):
    name = os.environ.get('TI_CURRENT_BENCHMARK')
    if name is None:
        return
    import taichi as ti
    arch_name = ti.core.arch_name(ti.cfg.arch)
    output_dir = os.environ.get('TI_BENCHMARK_OUTPUT_DIR', '.')
    filename = f'{output_dir}/{name}__arch_{arch_name}.dat'
    with open(filename, 'w') as f:
        f.write(f'time_avg: {avg:.4f}')


def supported_archs():
    import taichi as ti
    archs = [ti.core.host_arch()]
    if ti.core.with_cuda():
        archs.append(cuda)
    if ti.core.with_metal():
        archs.append(metal)
    if ti.core.with_opengl():
        archs.append(opengl)
    wanted_archs = os.environ.get('TI_WANTED_ARCHS', '')
    want_exclude = wanted_archs.startswith('^')
    if want_exclude:
        wanted_archs = wanted_archs[1:]
    wanted_archs = wanted_archs.split(',')
    # Note, ''.split(',') gives you [''], which is not an empty array.
    wanted_archs = list(filter(lambda x: x != '', wanted_archs))
    if len(wanted_archs):
        archs, old_archs = [], archs
        for arch in old_archs:
            if want_exclude == (ti.core.arch_name(arch) not in wanted_archs):
                archs.append(arch)
    return archs


def adaptive_arch_select(arch):
    if arch is None:
        return cpu
    supported = supported_archs()
    if isinstance(arch, list):
        for a in arch:
            if a in supported:
                return a
    elif arch in supported:
        return arch
    print(f'Arch={arch} not supported, falling back to CPU')
    return cpu


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
# @ti.require(ti.extension.data64)
# @ti.all_archs_with(default_fp=ti.f64)
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
        # OpenGL somehow crashes torch test without a reason, unforturnately
        return ti.archs_excluding(ti.opengl)(func)
    else:
        return lambda: None


# test with host arch only
def host_arch_only(func):
    import taichi as ti

    @functools.wraps(func)
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
