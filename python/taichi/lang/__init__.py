from .impl import *
from .error import enable_excepthook
from .util import deprecated
from .matrix import Matrix, Vector
from .transformer import TaichiSyntaxError
from .ndrange import ndrange, GroupedNDRange
from copy import deepcopy as _deepcopy
import functools
import os

core = taichi_lang_core


def record_action_hint(s):
    core.record_action_hint(s)


def begin_recording(fn):
    core.begin_recording(fn)


def stop_recording():
    core.stop_recording()


runtime = get_runtime()

i = indices(0)
j = indices(1)
k = indices(2)
l = indices(3)
ij = indices(0, 1)
ji = indices(1, 0)
jk = indices(1, 2)
kj = indices(2, 1)
ik = indices(0, 2)
ki = indices(2, 0)
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
cc = core.cc
gpu = [cuda, metal, opengl]
cpu = core.host_arch()
kernel_profiler_print = lambda: core.get_current_program(
).kernel_profiler_print()
kernel_profiler_clear = lambda: core.get_current_program(
).kernel_profiler_clear()


class _Extension(object):
    def __init__(self):
        try:
            self.sparse = core.sparse
            self.data64 = core.data64
            self.adstack = core.adstack
            self.bls = core.bls
        except:
            # In case of adding an extension crashes the format server
            core.warn("Extension list loading failed.")


extension = _Extension()
is_extension_supported = core.is_extension_supported


def reset():
    from .impl import reset as impl_reset
    impl_reset()
    global runtime
    runtime = get_runtime()


class _EnvironmentConfigurator:
    def __init__(self, kwargs, cfg):
        self.cfg = cfg
        self.kwargs = kwargs
        self.keys = []

    def add(self, key, cast=None):
        cast = cast or self.bool_int

        self.keys.append(key)

        # TI_ASYNC=   : no effect
        # TI_ASYNC=0  : False
        # TI_ASYNC=1  : True
        name = 'TI_' + key.upper()
        value = os.environ.get(name, '')
        if len(value):
            self[key] = cast(value)
        elif key in self.kwargs:
            self[key] = self.kwargs[key]
            del self.kwargs[key]  # pop out

    def __getitem__(self, key):
        return getattr(self.cfg, key)

    def __setitem__(self, key, value):
        setattr(self.cfg, key, value)

    @staticmethod
    def bool_int(x):
        return bool(int(x))


class _SpecialConfig:
    # like CompileConfig in C++, this is the configuations that belong to other submodules
    def __init__(self):
        self.print_preprocessed = False
        self.log_level = 'info'
        self.gdb_trigger = False
        self.excepthook = False


def init(arch=None,
         default_fp=None,
         default_ip=None,
         _test_mode=False,
         **kwargs):
    import taichi as ti

    # Make a deepcopy in case these args reference to items from ti.cfg, which are
    # actually references. If no copy is made and the args are indeed references,
    # ti.reset() could override the args to their default values.
    default_fp = _deepcopy(default_fp)
    default_ip = _deepcopy(default_ip)
    kwargs = _deepcopy(kwargs)
    ti.reset()

    spec_cfg = _SpecialConfig()
    env_comp = _EnvironmentConfigurator(kwargs, ti.cfg)
    env_spec = _EnvironmentConfigurator(kwargs, spec_cfg)

    # configure default_fp/ip:
    # TODO: move these stuff to _SpecialConfig too:
    if default_fp is None:
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

    if default_fp is not None:
        ti.get_runtime().set_default_fp(default_fp)
    if default_ip is not None:
        ti.get_runtime().set_default_ip(default_ip)

    # submodule configurations (spec_cfg):
    env_spec.add('print_preprocessed')
    env_spec.add('log_level', str)
    env_spec.add('gdb_trigger')
    env_spec.add('excepthook')

    # compiler configuations (ti.cfg):
    # TODO(yuanming-hu): Maybe CUDA specific configs like device_memory_* should be moved
    # to somewhere like ti.cuda_cfg so that user don't get confused?
    for key in dir(ti.cfg):
        if key in ['default_fp', 'default_ip']:
            continue
        cast = type(getattr(ti.cfg, key))
        if cast is bool:
            cast = None
        env_comp.add(key, cast)

    unexpected_keys = kwargs.keys()
    if len(unexpected_keys):
        raise KeyError(
            f'Unrecognized keyword argument(s) for ti.init: {", ".join(unexpected_keys)}'
        )

    # dispatch configurations that are not in ti.cfg:
    if not _test_mode:
        ti.set_gdb_trigger(spec_cfg.gdb_trigger)
        ti.get_runtime().print_preprocessed = spec_cfg.print_preprocessed
        ti.set_logging_level(spec_cfg.log_level.lower())
        if spec_cfg.excepthook:
            # TODO(#1405): add a way to restore old excepthook
            ti.enable_excepthook()

    # select arch (backend):
    env_arch = os.environ.get('TI_ARCH')
    if env_arch is not None:
        ti.info(f'Following TI_ARCH setting up for arch={env_arch}')
        arch = ti.core.arch_from_name(env_arch)
    ti.cfg.arch = adaptive_arch_select(arch)
    print(f'[Taichi] Starting on arch={ti.core.arch_name(ti.cfg.arch)}')

    if _test_mode:
        return spec_cfg

    # create a new program:
    ti.get_runtime().create_program()


def cache_shared(*args):
    for v in args:
        taichi_lang_core.cache(0, v.ptr)


def cache_read_only(v):
    taichi_lang_core.cache(1, v.ptr)


def assume_in_range(val, base, low, high):
    return taichi_lang_core.expr_assume_in_range(
        Expr(val).ptr,
        Expr(base).ptr, low, high)


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


def is_arch_supported(arch):
    if arch == cuda:
        return core.with_cuda()
    elif arch == metal:
        return core.with_metal()
    elif arch == opengl:
        return core.with_opengl()
    elif arch == cc:
        return core.with_cc()
    elif arch == cpu:
        return True
    else:
        return False


def supported_archs():
    archs = [cpu, cuda, metal, opengl, cc]

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
            if want_exclude == (core.arch_name(arch) not in wanted_archs):
                archs.append(arch)

    archs, old_archs = [], archs
    for arch in old_archs:
        if is_arch_supported(arch):
            archs.append(arch)

    return archs


def adaptive_arch_select(arch):
    if arch is None:
        return cpu
    import taichi as ti
    supported = supported_archs()
    if isinstance(arch, list):
        for a in arch:
            if is_arch_supported(a):
                return a
    elif arch in supported:
        return arch
    ti.warn(f'Arch={arch} is not supported, falling back to CPU')
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
                can_run_on.register(lambda arch: is_extension_supported(
                    arch, extension.data64))

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
                return all([is_extension_supported(arch, e) for e in exts])

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
