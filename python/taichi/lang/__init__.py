import functools
import os
from copy import deepcopy as _deepcopy

from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl
from taichi.lang.impl import *
from taichi.lang.kernel_arguments import ext_arr, template
from taichi.lang.kernel_impl import (KernelArgError, KernelDefError,
                                     data_oriented, func, kernel, pyfunc)
from taichi.lang.matrix import Matrix, Vector
from taichi.lang.ndrange import GroupedNDRange, ndrange
from taichi.lang.ops import *
from taichi.lang.quant_impl import quant
from taichi.lang.runtime_ops import async_flush, sync
from taichi.lang.transformer import TaichiSyntaxError
from taichi.lang.type_factory_impl import type_factory
from taichi.lang.util import (has_pytorch, is_taichi_class, python_scope,
                              taichi_scope, to_numpy_type, to_pytorch_type,
                              to_taichi_type)
from taichi.snode.fields_builder import FieldsBuilder

import taichi as ti

# TODO(#2223): Remove
core = _ti_core

runtime = impl.get_runtime()

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
x86_64 = _ti_core.x64
x64 = _ti_core.x64
arm64 = _ti_core.arm64
cuda = _ti_core.cuda
metal = _ti_core.metal
opengl = _ti_core.opengl
cc = _ti_core.cc
wasm = _ti_core.wasm
gpu = [cuda, metal, opengl]
cpu = _ti_core.host_arch()
kernel_profiler_print = lambda: impl.get_runtime().prog.kernel_profiler_print()
kernel_profiler_clear = lambda: impl.get_runtime().prog.kernel_profiler_clear()
kernel_profiler_total_time = lambda: impl.get_runtime(
).prog.kernel_profiler_total_time()
timeline_clear = lambda: impl.get_runtime().prog.timeline_clear()
timeline_save = lambda fn: impl.get_runtime().prog.timeline_save(fn)

# Legacy API
type_factory_ = _ti_core.get_type_factory_instance()


def memory_profiler_print():
    impl.get_runtime().materialize()
    impl.get_runtime().prog.print_memory_profiler_info()


extension = _ti_core.Extension
is_extension_supported = _ti_core.is_extension_supported


def reset():
    impl.reset()
    global runtime
    runtime = impl.get_runtime()


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
            if key in self.kwargs:
                _ti_core.warn(
                    f'ti.init argument "{key}" overridden by environment variable {name}={value}'
                )
                del self.kwargs[key]  # mark as recognized
        elif key in self.kwargs:
            self[key] = self.kwargs[key]
            del self.kwargs[key]  # mark as recognized

    def __getitem__(self, key):
        return getattr(self.cfg, key)

    def __setitem__(self, key, value):
        setattr(self.cfg, key, value)

    @staticmethod
    def bool_int(x):
        return bool(int(x))


class _SpecialConfig:
    # like CompileConfig in C++, this is the configurations that belong to other submodules
    def __init__(self):
        self.print_preprocessed = False
        self.log_level = 'info'
        self.gdb_trigger = False
        self.excepthook = False
        self.experimental_real_function = False


def init(arch=None,
         default_fp=None,
         default_ip=None,
         _test_mode=False,
         **kwargs):

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
    env_default_fp = os.environ.get("TI_DEFAULT_FP")
    if env_default_fp:
        if default_fp is not None:
            _ti_core.warn(
                f'ti.init argument "default_fp" overridden by environment variable TI_DEFAULT_FP={env_default_fp}'
            )
        if env_default_fp == '32':
            default_fp = ti.f32
        elif env_default_fp == '64':
            default_fp = ti.f64
        elif env_default_fp is not None:
            raise ValueError(
                f'Invalid TI_DEFAULT_FP={env_default_fp}, should be 32 or 64')

    env_default_ip = os.environ.get("TI_DEFAULT_IP")
    if env_default_ip:
        if default_ip is not None:
            _ti_core.warn(
                f'ti.init argument "default_ip" overridden by environment variable TI_DEFAULT_IP={env_default_ip}'
            )
        if env_default_ip == '32':
            default_ip = ti.i32
        elif env_default_ip == '64':
            default_ip = ti.i64
        elif env_default_ip is not None:
            raise ValueError(
                f'Invalid TI_DEFAULT_IP={env_default_ip}, should be 32 or 64')

    if default_fp is not None:
        impl.get_runtime().set_default_fp(default_fp)
    if default_ip is not None:
        impl.get_runtime().set_default_ip(default_ip)

    # submodule configurations (spec_cfg):
    env_spec.add('print_preprocessed')
    env_spec.add('log_level', str)
    env_spec.add('gdb_trigger')
    env_spec.add('excepthook')
    env_spec.add('experimental_real_function')

    # compiler configurations (ti.cfg):
    for key in dir(ti.cfg):
        if key in ['arch', 'default_fp', 'default_ip']:
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
        impl.get_runtime().print_preprocessed = spec_cfg.print_preprocessed
        impl.get_runtime().experimental_real_function = \
            spec_cfg.experimental_real_function
        ti.set_logging_level(spec_cfg.log_level.lower())
        if spec_cfg.excepthook:
            # TODO(#1405): add a way to restore old excepthook
            ti.enable_excepthook()

    # select arch (backend):
    env_arch = os.environ.get('TI_ARCH')
    if env_arch is not None:
        ti.info(f'Following TI_ARCH setting up for arch={env_arch}')
        arch = _ti_core.arch_from_name(env_arch)
    ti.cfg.arch = adaptive_arch_select(arch)
    print(f'[Taichi] Starting on arch={_ti_core.arch_name(ti.cfg.arch)}')

    if _test_mode:
        return spec_cfg

    # create a new program:
    impl.get_runtime().create_program()

    ti.trace('Materializing runtime...')
    impl.get_runtime().prog.materialize_runtime()

    FieldsBuilder.clear_finalized_fbs()
    impl._root_fb = FieldsBuilder()


def no_activate(*args):
    for v in args:
        _ti_core.no_activate(v.snode.ptr)


def block_local(*args):
    for a in args:
        for v in a.get_field_members():
            _ti_core.insert_snode_access_flag(
                _ti_core.SNodeAccessFlag.block_local, v.ptr)


@deprecated('ti.cache_shared', 'ti.block_local')
def cache_shared(*args):
    block_local(*args)


def cache_read_only(*args):
    for a in args:
        for v in a.get_field_members():
            _ti_core.insert_snode_access_flag(
                _ti_core.SNodeAccessFlag.read_only, v.ptr)


def assume_in_range(val, base, low, high):
    return _ti_core.expr_assume_in_range(
        Expr(val).ptr,
        Expr(base).ptr, low, high)


def loop_unique(val, covers=None):
    if covers is None:
        covers = []
    if not isinstance(covers, (list, tuple)):
        covers = [covers]
    covers = [x.snode.ptr if isinstance(x, Expr) else x.ptr for x in covers]
    return _ti_core.expr_loop_unique(Expr(val).ptr, covers)


parallelize = _ti_core.parallelize
serialize = lambda: parallelize(1)
vectorize = _ti_core.vectorize
bit_vectorize = _ti_core.bit_vectorize
block_dim = _ti_core.block_dim

inversed = deprecated('ti.inversed(a)', 'a.inverse()')(Matrix.inversed)
transposed = deprecated('ti.transposed(a)', 'a.transpose()')(Matrix.transposed)


def polar_decompose(A, dt=None):
    if dt is None:
        dt = impl.get_runtime().default_fp
    from .linalg import polar_decompose
    return polar_decompose(A, dt)


def svd(A, dt=None):
    if dt is None:
        dt = impl.get_runtime().default_fp
    from .linalg import svd
    return svd(A, dt)


def eig(A, dt=None):
    """Compute the eigenvalues and right eigenvectors of a real matrix.

    Parameters
    ----------
    A: ti.Matrix(n, n)
        2D Matrix for which the eigenvalues and right eigenvectors will be computed.
    dt: Optional[DataType]
        The datatype for the eigenvalues and right eigenvectors

    Returns
    -------
    eigenvalues: ti.Matrix(n, 2)
        The eigenvalues in complex form. Each row stores one eigenvalue. The first number
        of the eigenvalue represents the real part and the second number represents the
        imaginary part.
    eigenvectors: ti.Matrix(n*2, n)
        The eigenvectors in complex form. Each column stores one eigenvector. Each eigenvector
        consists of n entries, each of which is represented by two numbers for its real part
        and imaginary part.
    """
    if dt is None:
        dt = impl.get_runtime().default_fp
    from taichi.lang import linalg
    if A.n == 2:
        return linalg.eig2x2(A, dt)
    raise Exception("Eigen solver only supports 2D matrices.")


def sym_eig(A, dt=None):
    """Compute the eigenvalues and right eigenvectors of a real symmetric matrix.

    Parameters
    ----------
    A: ti.Matrix(n, n)
        Symmetric Matrix for which the eigenvalues and right eigenvectors will be computed.
    dt: Optional[DataType]
        The datatype for the eigenvalues and right eigenvectors

    Returns
    -------
    eigenvalues: ti.Vector(n)
        The eigenvalues. Each entry store one eigen value.
    eigenvectors: ti.Matrix(n, n)
        The eigenvectors. Each column stores one eigenvector.
    """
    assert all(A == A.transpose()), "A needs to be symmetric"
    if dt is None:
        dt = impl.get_runtime().default_fp
    from taichi.lang import linalg
    if A.n == 2:
        return linalg.sym_eig2x2(A, dt)
    raise Exception("Symmetric eigen solver only supports 2D matrices.")


def randn(dt=None):
    if dt is None:
        dt = impl.get_runtime().default_fp
    from .random import randn
    return randn(dt)


determinant = deprecated('ti.determinant(a)',
                         'a.determinant()')(Matrix.determinant)
tr = deprecated('ti.tr(a)', 'a.trace()')(Matrix.trace)


def Tape(loss, clear_gradients=True):
    impl.get_runtime().materialize()
    if len(loss.shape) != 0:
        raise RuntimeError(
            'The loss of `Tape` must be a 0-D field, i.e. scalar')
    if not loss.snode.ptr.has_grad():
        raise RuntimeError(
            'Gradients of loss are not allocated, please use ti.field(..., needs_grad=True)'
            ' for all fields that are required by autodiff.')
    if clear_gradients:
        clear_all_gradients()

    from taichi.lang.meta import clear_loss
    clear_loss(loss)

    return runtime.get_tape(loss)


def clear_all_gradients():
    impl.get_runtime().materialize()

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
            from taichi.lang.meta import clear_gradients
            clear_gradients(places)

    for root_fb in FieldsBuilder.finalized_fbs():
        visit(root_fb)


def benchmark(func, repeat=300, args=()):
    import time

    def run_benchmark():
        compile_time = time.time()
        func(*args)  # compile the kernel first
        ti.sync()
        compile_time = time.time() - compile_time
        ti.stat_write('compilation_time', compile_time)
        codegen_stat = _ti_core.stat()
        for line in codegen_stat.split('\n'):
            try:
                a, b = line.strip().split(':')
            except:
                continue
            a = a.strip()
            b = int(float(b))
            if a == 'codegen_kernel_statements':
                ti.stat_write('compiled_inst', b)
            if a == 'codegen_offloaded_tasks':
                ti.stat_write('compiled_tasks', b)
            elif a == 'launched_tasks':
                ti.stat_write('launched_tasks', b)

        # Use 3 initial iterations to warm up
        # instruction/data caches. Discussion:
        # https://github.com/taichi-dev/taichi/pull/1002#discussion_r426312136
        for i in range(3):
            func(*args)
            ti.sync()
        ti.kernel_profiler_clear()
        t = time.time()
        for n in range(repeat):
            func(*args)
            ti.sync()
        elapsed = time.time() - t
        avg = elapsed / repeat
        ti.stat_write('wall_clk_t', avg)
        device_time = ti.kernel_profiler_total_time()
        avg_device_time = device_time / repeat
        ti.stat_write('exec_t', avg_device_time)

    run_benchmark()


def benchmark_plot(fn=None,
                   cases=None,
                   columns=None,
                   column_titles=None,
                   archs=None,
                   title=None,
                   bars='sync_vs_async',
                   bar_width=0.4,
                   bar_distance=0,
                   left_margin=0,
                   size=(12, 8)):
    import matplotlib.pyplot as plt
    import yaml
    if fn is None:
        fn = os.path.join(_ti_core.get_repo_dir(), 'benchmarks', 'output',
                          'benchmark.yml')

    with open(fn, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    if bars != 'sync_vs_async':  # need baseline
        baseline_dir = os.path.join(_ti_core.get_repo_dir(), 'benchmarks',
                                    'baseline')
        baseline_file = f'{baseline_dir}/benchmark.yml'
        with open(baseline_file, 'r') as f:
            baseline_data = yaml.load(f, Loader=yaml.SafeLoader)
    if cases is None:
        cases = list(data.keys())

    assert len(cases) >= 1
    if len(cases) == 1:
        cases = [cases[0], cases[0]]
        ti.warning(
            'Function benchmark_plot does not support plotting with only one case for now. Duplicating the item to move on.'
        )

    if columns is None:
        columns = list(data[cases[0]].keys())
    if column_titles is None:
        column_titles = columns
    normalize_to_lowest = lambda x: True
    figure, subfigures = plt.subplots(len(cases), len(columns))
    if title is None:
        title = 'Taichi Performance Benchmarks (Higher means more)'
    figure.suptitle(title, fontweight="bold")
    for col_id in range(len(columns)):
        subfigures[0][col_id].set_title(column_titles[col_id])
    for case_id in range(len(cases)):
        case = cases[case_id]
        subfigures[case_id][0].annotate(
            case,
            xy=(0, 0.5),
            xytext=(-subfigures[case_id][0].yaxis.labelpad - 5, 0),
            xycoords=subfigures[case_id][0].yaxis.label,
            textcoords='offset points',
            size='large',
            ha='right',
            va='center')
        for col_id in range(len(columns)):
            col = columns[col_id]
            if archs is None:
                current_archs = data[case][col].keys()
            else:
                current_archs = [
                    x for x in archs if x in data[case][col].keys()
                ]
            if bars == 'sync_vs_async':
                y_left = [
                    data[case][col][arch]['sync'] for arch in current_archs
                ]
                label_left = 'sync'
                y_right = [
                    data[case][col][arch]['async'] for arch in current_archs
                ]
                label_right = 'async'
            elif bars == 'sync_regression':
                y_left = [
                    baseline_data[case][col][arch]['sync']
                    for arch in current_archs
                ]
                label_left = 'before'
                y_right = [
                    data[case][col][arch]['sync'] for arch in current_archs
                ]
                label_right = 'after'
            elif bars == 'async_regression':
                y_left = [
                    baseline_data[case][col][arch]['async']
                    for arch in current_archs
                ]
                label_left = 'before'
                y_right = [
                    data[case][col][arch]['async'] for arch in current_archs
                ]
                label_right = 'after'
            else:
                raise RuntimeError('Unknown bars type')
            if normalize_to_lowest(col):
                for i in range(len(current_archs)):
                    maximum = max(y_left[i], y_right[i])
                    y_left[i] = y_left[i] / maximum if y_left[i] != 0 else 1
                    y_right[i] = y_right[i] / maximum if y_right[i] != 0 else 1
            ax = subfigures[case_id][col_id]
            bar_left = ax.bar(x=[
                i - bar_width / 2 - bar_distance / 2
                for i in range(len(current_archs))
            ],
                              height=y_left,
                              width=bar_width,
                              label=label_left,
                              color=(0.47, 0.69, 0.89, 1.0))
            bar_right = ax.bar(x=[
                i + bar_width / 2 + bar_distance / 2
                for i in range(len(current_archs))
            ],
                               height=y_right,
                               width=bar_width,
                               label=label_right,
                               color=(0.68, 0.26, 0.31, 1.0))
            ax.set_xticks(range(len(current_archs)))
            ax.set_xticklabels(current_archs)
            figure.legend((bar_left, bar_right), (label_left, label_right),
                          loc='lower center')
    figure.subplots_adjust(left=left_margin)

    fig = plt.gcf()
    fig.set_size_inches(size)

    plt.show()


def stat_write(key, value):
    import yaml
    case_name = os.environ.get('TI_CURRENT_BENCHMARK')
    if case_name is None:
        return
    if case_name.startswith('benchmark_'):
        case_name = case_name[10:]
    arch_name = _ti_core.arch_name(ti.cfg.arch)
    async_mode = 'async' if ti.cfg.async_mode else 'sync'
    output_dir = os.environ.get('TI_BENCHMARK_OUTPUT_DIR', '.')
    filename = f'{output_dir}/benchmark.yml'
    try:
        with open(filename, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    except FileNotFoundError:
        data = {}
    data.setdefault(case_name, {})
    data[case_name].setdefault(key, {})
    data[case_name][key].setdefault(arch_name, {})
    data[case_name][key][arch_name][async_mode] = value
    with open(filename, 'w') as f:
        yaml.dump(data, f, Dumper=yaml.SafeDumper)


def is_arch_supported(arch):
    arch_table = {
        cuda: _ti_core.with_cuda,
        metal: _ti_core.with_metal,
        opengl: _ti_core.with_opengl,
        cc: _ti_core.with_cc,
        wasm: lambda: True,
        cpu: lambda: True,
    }
    with_arch = arch_table.get(arch, lambda: False)
    try:
        return with_arch()
    except Exception as e:
        arch = _ti_core.arch_name(arch)
        _ti_core.warn(
            f"{e.__class__.__name__}: '{e}' occurred when detecting "
            f"{arch}, consider add `export TI_WITH_{arch.upper()}=0` "
            f" to environment variables to depress this warning message.")
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
            if want_exclude == (_ti_core.arch_name(arch) not in wanted_archs):
                archs.append(arch)

    archs, old_archs = [], archs
    for arch in old_archs:
        if is_arch_supported(arch):
            archs.append(arch)

    return archs


def adaptive_arch_select(arch):
    if arch is None:
        return cpu
    if not isinstance(arch, (list, tuple)):
        arch = [arch]
    for a in arch:
        if is_arch_supported(a):
            return a
    ti.warn(f'Arch={arch} is not supported, falling back to CPU')
    return cpu


class _ArchCheckers(object):
    def __init__(self):
        self._checkers = []

    def register(self, c):
        self._checkers.append(c)

    def __call__(self, arch):
        assert isinstance(arch, _ti_core.Arch)
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
    assert all([isinstance(a, _ti_core.Arch) for a in excluded_archs])
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
    assert all([isinstance(e, _ti_core.Extension) for e in exts])

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
    if ti.has_pytorch():
        # OpenGL somehow crashes torch test without a reason, unforturnately
        return ti.archs_excluding(ti.opengl)(func)
    else:
        return lambda: None


# test with host arch only
def host_arch_only(func):
    @functools.wraps(func)
    def test(*args, **kwargs):
        archs = [_ti_core.host_arch()]
        for arch in archs:
            ti.init(arch=arch)
            func(*args, **kwargs)

    return test


def archs_with(archs, **init_kwags):
    """
    Run the test on the given archs with the given init args.

    Args:
      archs: a list of Taichi archs
      init_kwargs: kwargs passed to ti.init()
    """
    def decorator(test):
        @functools.wraps(test)
        def wrapped(*test_args, **test_kwargs):
            for arch in archs:
                ti.init(arch=arch, **init_kwags)
                test(*test_args, **test_kwargs)

        return wrapped

    return decorator


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
        impl.get_runtime().inside_complex_kernel = True
        if impl.get_runtime().target_tape:
            impl.get_runtime().target_tape.insert(decorated, args)
        try:
            func(*args, **kwargs)
        finally:
            impl.get_runtime().inside_complex_kernel = False

    decorated.grad = None
    return decorated


def complex_kernel_grad(primal):
    def decorator(func):
        def decorated(*args, **kwargs):
            func(*args, **kwargs)

        primal.grad = decorated
        return decorated

    return decorator


__all__ = [s for s in dir() if not s.startswith('_')]
