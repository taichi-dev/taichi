import atexit
import datetime
import functools
import json
import os
import platform
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from copy import deepcopy as _deepcopy
from urllib import request

from taichi._lib import core as _ti_core
from taichi._lib.utils import locale_encode
from taichi.lang import impl
from taichi.lang._ndarray import ScalarNdarray
from taichi.lang._ndrange import GroupedNDRange, ndrange
from taichi.lang.any_array import AnyArray, AnyArrayAccess
from taichi.lang.enums import Layout
from taichi.lang.exception import (InvalidOperationError,
                                   TaichiCompilationError, TaichiNameError,
                                   TaichiSyntaxError, TaichiTypeError)
from taichi.lang.expr import Expr, make_expr_group
from taichi.lang.field import Field, ScalarField
from taichi.lang.impl import (axes, begin_frontend_if,
                              begin_frontend_struct_for, call_internal,
                              current_cfg, deactivate_all_snodes, expr_init,
                              expr_init_func, expr_init_list, field,
                              get_runtime, grouped,
                              insert_expr_stmt_if_ti_func, ndarray, one, root,
                              static, static_assert, static_print, stop_grad,
                              subscript, ti_assert, ti_float, ti_format,
                              ti_int, ti_print, zero)
from taichi.lang.kernel_arguments import SparseMatrixProxy
from taichi.lang.kernel_impl import (KernelArgError, KernelDefError,
                                     data_oriented, func, kernel, pyfunc)
from taichi.lang.matrix import Matrix, MatrixField, Vector
from taichi.lang.mesh import Mesh, MeshElementFieldProxy, TetMesh, TriMesh
from taichi.lang.ops import *  # pylint: disable=W0622
from taichi.lang.quant_impl import quant
from taichi.lang.runtime_ops import async_flush, sync
from taichi.lang.snode import (SNode, activate, append, deactivate, get_addr,
                               is_active, length, rescale_index)
from taichi.lang.sort import parallel_sort
from taichi.lang.source_builder import SourceBuilder
from taichi.lang.struct import Struct, StructField
from taichi.lang.tape import TapeImpl
from taichi.lang.type_factory_impl import type_factory
from taichi.lang.util import (cook_dtype, has_clangpp, has_pytorch,
                              is_taichi_class, python_scope, taichi_scope,
                              to_numpy_type, to_pytorch_type, to_taichi_type)
from taichi.profiler import KernelProfiler, get_default_kernel_profiler
from taichi.profiler.kernelmetrics import (CuptiMetric, default_cupti_metrics,
                                           get_predefined_cupti_metrics)
from taichi.snode.fields_builder import FieldsBuilder
from taichi.tools.util import set_gdb_trigger, warning
from taichi.types.annotations import any_arr, ext_arr, template
from taichi.types.primitive_types import (f16, f32, f64, i32, i64,
                                          integer_types, u32, u64)

from taichi import _logging

runtime = impl.get_runtime()

i = axes(0)
j = axes(1)
k = axes(2)
l = axes(3)
ij = axes(0, 1)
ik = axes(0, 2)
il = axes(0, 3)
jk = axes(1, 2)
jl = axes(1, 3)
kl = axes(2, 3)
ijk = axes(0, 1, 2)
ijl = axes(0, 1, 3)
ikl = axes(0, 2, 3)
jkl = axes(1, 2, 3)
ijkl = axes(0, 1, 2, 3)

cfg = impl.default_cfg()
x86_64 = _ti_core.x64
"""The x64 CPU backend.
"""
x64 = _ti_core.x64
"""The X64 CPU backend.
"""
arm64 = _ti_core.arm64
"""The ARM CPU backend.
"""
cuda = _ti_core.cuda
"""The CUDA backend.
"""
metal = _ti_core.metal
"""The Apple Metal backend.
"""
opengl = _ti_core.opengl
"""The OpenGL backend. OpenGL 4.3 required.
"""
# Skip annotating this one because it is barely maintained.
cc = _ti_core.cc
wasm = _ti_core.wasm
"""The WebAssembly backend.
"""
vulkan = _ti_core.vulkan
"""The Vulkan backend.
"""
dx11 = _ti_core.dx11
"""The DX11 backend.
"""
gpu = [cuda, metal, opengl, vulkan, dx11]
"""A list of GPU backends supported on the current system.

When this is used, Taichi automatically picks the matching GPU backend. If no
GPU is detected, Taichi falls back to the CPU backend.
"""
cpu = _ti_core.host_arch()
"""A list of CPU backends supported on the current system.

When this is used, Taichi automatically picks the matching CPU backend.
"""
timeline_clear = lambda: impl.get_runtime().prog.timeline_clear()  # pylint: disable=unnecessary-lambda
timeline_save = lambda fn: impl.get_runtime().prog.timeline_save(fn)  # pylint: disable=unnecessary-lambda

# Legacy API
type_factory_ = _ti_core.get_type_factory_instance()


def print_kernel_profile_info(mode='count'):
    """Print the profiling results of Taichi kernels.

    To enable this profiler, set ``kernel_profiler=True`` in ``ti.init()``.
    ``'count'`` mode: print the statistics (min,max,avg time) of launched kernels,
    ``'trace'`` mode: print the records of launched kernels with specific profiling metrics (time, memory load/store and core utilization etc.),
    and defaults to ``'count'``.

    Args:
        mode (str): the way to print profiling results.

    Example::

        >>> import taichi as ti

        >>> ti.init(ti.cpu, kernel_profiler=True)
        >>> var = ti.field(ti.f32, shape=1)

        >>> @ti.kernel
        >>> def compute():
        >>>     var[0] = 1.0

        >>> compute()
        >>> ti.print_kernel_profile_info()
        >>> # equivalent calls :
        >>> # ti.print_kernel_profile_info('count')

        >>> ti.print_kernel_profile_info('trace')

    Note:
        Currently the result of `KernelProfiler` could be incorrect on OpenGL
        backend due to its lack of support for `ti.sync()`.

        For advanced mode of `KernelProfiler`, please visit https://docs.taichi.graphics/docs/lang/articles/misc/profiler#advanced-mode.
    """
    get_default_kernel_profiler().print_info(mode)


def query_kernel_profile_info(name):
    """Query kernel elapsed time(min,avg,max) on devices using the kernel name.

    To enable this profiler, set `kernel_profiler=True` in `ti.init`.

    Args:
        name (str): kernel name.

    Returns:
        KernelProfilerQueryResult (class): with member variables(counter, min, max, avg)

    Example::

        >>> import taichi as ti

        >>> ti.init(ti.cpu, kernel_profiler=True)
        >>> n = 1024*1024
        >>> var = ti.field(ti.f32, shape=n)

        >>> @ti.kernel
        >>> def fill():
        >>>     for i in range(n):
        >>>         var[i] = 0.1

        >>> fill()
        >>> ti.clear_kernel_profile_info() #[1]
        >>> for i in range(100):
        >>>     fill()
        >>> query_result = ti.query_kernel_profile_info(fill.__name__) #[2]
        >>> print("kernel excuted times =",query_result.counter)
        >>> print("kernel elapsed time(min_in_ms) =",query_result.min)
        >>> print("kernel elapsed time(max_in_ms) =",query_result.max)
        >>> print("kernel elapsed time(avg_in_ms) =",query_result.avg)

    Note:
        [1] To get the correct result, query_kernel_profile_info() must be used in conjunction with
        clear_kernel_profile_info().

        [2] Currently the result of `KernelProfiler` could be incorrect on OpenGL
        backend due to its lack of support for `ti.sync()`.
    """
    return get_default_kernel_profiler().query_info(name)


def clear_kernel_profile_info():
    """Clear all KernelProfiler records."""
    get_default_kernel_profiler().clear_info()


def kernel_profiler_total_time():
    """Get elapsed time of all kernels recorded in KernelProfiler.

    Returns:
        time (float): total time in second.
    """
    return get_default_kernel_profiler().get_total_time()


def set_kernel_profiler_toolkit(toolkit_name='default'):
    """Set the toolkit used by KernelProfiler.

    Currently, we only support toolkits: ``'default'`` and ``'cupti'``.

    Args:
        toolkit_name (str): string of toolkit name.

    Returns:
        status (bool): whether the setting is successful or not.

    Example::

        >>> import taichi as ti

        >>> ti.init(arch=ti.cuda, kernel_profiler=True)
        >>> x = ti.field(ti.f32, shape=1024*1024)

        >>> @ti.kernel
        >>> def fill():
        >>>     for i in x:
        >>>         x[i] = i

        >>> ti.set_kernel_profiler_toolkit('cupti')
        >>> for i in range(100):
        >>>     fill()
        >>> ti.print_kernel_profile_info()

        >>> ti.set_kernel_profiler_toolkit('default')
        >>> for i in range(100):
        >>>     fill()
        >>> ti.print_kernel_profile_info()
    """
    return get_default_kernel_profiler().set_toolkit(toolkit_name)


def set_kernel_profile_metrics(metric_list=default_cupti_metrics):
    """Set metrics that will be collected by the CUPTI toolkit.

    Args:
        metric_list (list): a list of :class:`~taichi.lang.CuptiMetric()` instances, default value: :data:`~taichi.lang.default_cupti_metrics`.

    Example::

        >>> import taichi as ti

        >>> ti.init(kernel_profiler=True, arch=ti.cuda)
        >>> ti.set_kernel_profiler_toolkit('cupti')
        >>> num_elements = 128*1024*1024

        >>> x = ti.field(ti.f32, shape=num_elements)
        >>> y = ti.field(ti.f32, shape=())
        >>> y[None] = 0

        >>> @ti.kernel
        >>> def reduction():
        >>>     for i in x:
        >>>         y[None] += x[i]

        >>> # In the case of not pramater, Taichi will print its pre-defined metrics list
        >>> ti.get_predefined_cupti_metrics()
        >>> # get Taichi pre-defined metrics
        >>> profiling_metrics = ti.get_predefined_cupti_metrics('shared_access')

        >>> global_op_atom = ti.CuptiMetric(
        >>>     name='l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum',
        >>>     header=' global.atom ',
        >>>     format='    {:8.0f} ')
        >>> # add user defined metrics
        >>> profiling_metrics += [global_op_atom]

        >>> # metrics setting will be retained until the next configuration
        >>> ti.set_kernel_profile_metrics(profiling_metrics)
        >>> for i in range(16):
        >>>     reduction()
        >>> ti.print_kernel_profile_info('trace')

    Note:
        Metrics setting will be retained until the next configuration.
    """
    get_default_kernel_profiler().set_metrics(metric_list)


@contextmanager
def collect_kernel_profile_metrics(metric_list=default_cupti_metrics):
    """Set temporary metrics that will be collected by the CUPTI toolkit within this context.

    Args:
        metric_list (list): a list of :class:`~taichi.lang.CuptiMetric()` instances, default value: :data:`~taichi.lang.default_cupti_metrics`.

    Example::

        >>> import taichi as ti

        >>> ti.init(kernel_profiler=True, arch=ti.cuda)
        >>> ti.set_kernel_profiler_toolkit('cupti')
        >>> num_elements = 128*1024*1024

        >>> x = ti.field(ti.f32, shape=num_elements)
        >>> y = ti.field(ti.f32, shape=())
        >>> y[None] = 0

        >>> @ti.kernel
        >>> def reduction():
        >>>     for i in x:
        >>>         y[None] += x[i]

        >>> # In the case of not pramater, Taichi will print its pre-defined metrics list
        >>> ti.get_predefined_cupti_metrics()
        >>> # get Taichi pre-defined metrics
        >>> profiling_metrics = ti.get_predefined_cupti_metrics('device_utilization')

        >>> global_op_atom = ti.CuptiMetric(
        >>>     name='l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum',
        >>>     header=' global.atom ',
        >>>     format='    {:8.0f} ')
        >>> # add user defined metrics
        >>> profiling_metrics += [global_op_atom]

        >>> # metrics setting is temporary, and will be clear when exit from this context.
        >>> with ti.collect_kernel_profile_metrics(profiling_metrics):
        >>>     for i in range(16):
        >>>         reduction()
        >>>     ti.print_kernel_profile_info('trace')

    Note:
        The configuration of the ``metric_list`` will be clear when exit from this context.
    """
    get_default_kernel_profiler().set_metrics(metric_list)
    yield get_default_kernel_profiler()
    get_default_kernel_profiler().set_metrics()


def print_memory_profile_info():
    """Memory profiling tool for LLVM backends with full sparse support.

    This profiler is automatically on.
    """
    impl.get_runtime().materialize()
    impl.get_runtime().prog.print_memory_profiler_info()


extension = _ti_core.Extension


def is_extension_supported(arch, ext):
    """Checks whether an extension is supported on an arch.

    Args:
        arch (taichi_core.Arch): Specified arch.
        ext (taichi_core.Extension): Specified extension.

    Returns:
        bool: Whether `ext` is supported on `arch`.
    """
    return _ti_core.is_extension_supported(arch, ext)


def reset():
    """Resets Taichi to its initial state.

    This would destroy all the fields and kernels.
    """
    _ti_core.reset_snode_access_flag()
    impl.reset()
    global runtime
    runtime = impl.get_runtime()


class _EnvironmentConfigurator:
    def __init__(self, kwargs, _cfg):
        self.cfg = _cfg
        self.kwargs = kwargs
        self.keys = []

    def add(self, key, _cast=None):
        _cast = _cast or self.bool_int

        self.keys.append(key)

        # TI_ASYNC=   : no effect
        # TI_ASYNC=0  : False
        # TI_ASYNC=1  : True
        name = 'TI_' + key.upper()
        value = os.environ.get(name, '')
        if len(value):
            self[key] = _cast(value)
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
        self.log_level = 'info'
        self.gdb_trigger = False
        self.experimental_real_function = False
        self.short_circuit_operators = False
        self.ndarray_use_torch = False


def prepare_sandbox():
    '''
    Returns a temporary directory, which will be automatically deleted on exit.
    It may contain the taichi_core shared object or some misc. files.
    '''
    tmp_dir = tempfile.mkdtemp(prefix='taichi-')
    atexit.register(shutil.rmtree, tmp_dir)
    print(f'[Taichi] preparing sandbox at {tmp_dir}')
    os.mkdir(os.path.join(tmp_dir, 'runtime/'))
    return tmp_dir


def check_version():
    # Check Taichi version for the user.
    major = _ti_core.get_version_major()
    minor = _ti_core.get_version_minor()
    patch = _ti_core.get_version_patch()
    version = f'{major}.{minor}.{patch}'
    payload = {'version': version, 'platform': '', 'python': ''}

    system = platform.system()
    if system == 'Linux':
        payload['platform'] = 'manylinux1_x86_64'
    elif system == 'Windows':
        payload['platform'] = 'win_amd64'
    elif system == 'Darwin':
        if platform.release() < '19.0.0':
            payload['platform'] = 'macosx_10_14_x86_64'
        elif platform.machine() == 'x86_64':
            payload['platform'] = 'macosx_10_15_x86_64'
        else:
            payload['platform'] = 'macosx_11_0_arm64'

    python_version = platform.python_version()
    if python_version.startswith('3.6.'):
        payload['python'] = 'cp36'
    elif python_version.startswith('3.7.'):
        payload['python'] = 'cp37'
    elif python_version.startswith('3.8.'):
        payload['python'] = 'cp38'
    elif python_version.startswith('3.9.'):
        payload['python'] = 'cp39'

    # We do not want request exceptions break users' usage of Taichi.
    try:
        payload = json.dumps(payload)
        payload = payload.encode()
        req = request.Request('https://metadata.taichi.graphics/check_version',
                              method='POST')
        req.add_header('Content-Type', 'application/json')
        with request.urlopen(req, data=payload, timeout=5) as response:
            response = json.loads(response.read().decode('utf-8'))
            return response
    except:
        return None


def try_check_version():
    try:
        os.makedirs(_ti_core.get_repo_dir(), exist_ok=True)
        timestamp_path = os.path.join(_ti_core.get_repo_dir(), 'timestamp')
        cur_date = datetime.date.today()
        if os.path.exists(timestamp_path):
            last_time = ''
            with open(timestamp_path, 'r') as f:
                last_time = f.readlines()[0].rstrip()
            if cur_date.strftime('%Y-%m-%d') > last_time:
                response = check_version()
                if response is None:
                    return
                with open(timestamp_path, 'w') as f:
                    f.write((cur_date +
                             datetime.timedelta(days=7)).strftime('%Y-%m-%d'))
                    f.write('\n')
                    if response['status'] == 1:
                        f.write(response['latest_version'])
                    else:
                        f.write('0.0.0')
        else:
            response = check_version()
            if response is None:
                return
            with open(timestamp_path, 'w') as f:
                f.write((cur_date +
                         datetime.timedelta(days=7)).strftime('%Y-%m-%d'))
                f.write('\n')
                if response['status'] == 1:
                    f.write(response['latest_version'])
                else:
                    f.write('0.0.0')
    # Wildcard exception to catch potential file writing errors.
    except:
        pass


def init(arch=None,
         default_fp=None,
         default_ip=None,
         _test_mode=False,
         enable_fallback=True,
         **kwargs):
    """Initializes the Taichi runtime.

    This should always be the entry point of your Taichi program. Most
    importantly, it sets the backend used throughout the program.

    Args:
        arch: Backend to use. This is usually :const:`~taichi.lang.cpu` or :const:`~taichi.lang.gpu`.
        default_fp (Optional[type]): Default floating-point type.
        default_ip (Optional[type]): Default integral type.
        **kwargs: Taichi provides highly customizable compilation through
            ``kwargs``, which allows for fine grained control of Taichi compiler
            behavior. Below we list some of the most frequently used ones. For a
            complete list, please check out
            https://github.com/taichi-dev/taichi/blob/master/taichi/program/compile_config.h.

            * ``cpu_max_num_threads`` (int): Sets the number of threads used by the CPU thread pool.
            * ``debug`` (bool): Enables the debug mode, under which Taichi does a few more things like boundary checks.
            * ``print_ir`` (bool): Prints the CHI IR of the Taichi kernels.
            * ``packed`` (bool): Enables the packed memory layout. See https://docs.taichi.graphics/lang/articles/advanced/layout.
    """
    # Check version for users every 7 days if not disabled by users.
    skip = os.environ.get("TI_SKIP_VERSION_CHECK")
    if skip != 'ON':
        # We don't join this thread because we do not wish to block users.
        check_version_thread = threading.Thread(target=try_check_version,
                                                daemon=True)
        check_version_thread.start()

    # Make a deepcopy in case these args reference to items from ti.cfg, which are
    # actually references. If no copy is made and the args are indeed references,
    # ti.reset() could override the args to their default values.
    default_fp = _deepcopy(default_fp)
    default_ip = _deepcopy(default_ip)
    kwargs = _deepcopy(kwargs)
    reset()

    spec_cfg = _SpecialConfig()
    env_comp = _EnvironmentConfigurator(kwargs, cfg)
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
            default_fp = f32
        elif env_default_fp == '64':
            default_fp = f64
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
            default_ip = i32
        elif env_default_ip == '64':
            default_ip = i64
        elif env_default_ip is not None:
            raise ValueError(
                f'Invalid TI_DEFAULT_IP={env_default_ip}, should be 32 or 64')

    if default_fp is not None:
        impl.get_runtime().set_default_fp(default_fp)
    if default_ip is not None:
        impl.get_runtime().set_default_ip(default_ip)

    # submodule configurations (spec_cfg):
    env_spec.add('log_level', str)
    env_spec.add('gdb_trigger')
    env_spec.add('experimental_real_function')
    env_spec.add('short_circuit_operators')
    env_spec.add('ndarray_use_torch')

    # compiler configurations (ti.cfg):
    for key in dir(cfg):
        if key in ['arch', 'default_fp', 'default_ip']:
            continue
        _cast = type(getattr(cfg, key))
        if _cast is bool:
            _cast = None
        env_comp.add(key, _cast)

    unexpected_keys = kwargs.keys()

    if len(unexpected_keys):
        raise KeyError(
            f'Unrecognized keyword argument(s) for ti.init: {", ".join(unexpected_keys)}'
        )

    # dispatch configurations that are not in ti.cfg:
    if not _test_mode:
        set_gdb_trigger(spec_cfg.gdb_trigger)
        impl.get_runtime().experimental_real_function = \
            spec_cfg.experimental_real_function
        impl.get_runtime().short_circuit_operators = \
            spec_cfg.short_circuit_operators
        impl.get_runtime().ndarray_use_torch = \
            spec_cfg.ndarray_use_torch
        _logging.set_logging_level(spec_cfg.log_level.lower())

    # select arch (backend):
    env_arch = os.environ.get('TI_ARCH')
    if env_arch is not None:
        _logging.info(f'Following TI_ARCH setting up for arch={env_arch}')
        arch = _ti_core.arch_from_name(env_arch)
    cfg.arch = adaptive_arch_select(arch, enable_fallback, cfg.use_gles)
    if cfg.arch == cc:
        _ti_core.set_tmp_dir(locale_encode(prepare_sandbox()))
    print(f'[Taichi] Starting on arch={_ti_core.arch_name(cfg.arch)}')

    # Torch based ndarray on opengl backend allocates memory on host instead of opengl backend.
    # So it won't work.
    if cfg.arch == opengl and spec_cfg.ndarray_use_torch:
        _logging.warn(
            'Opengl backend doesn\'t support torch based ndarray. Setting ndarray_use_torch to False.'
        )
        impl.get_runtime().ndarray_use_torch = False

    if _test_mode:
        return spec_cfg

    get_default_kernel_profiler().set_kernel_profiler_mode(cfg.kernel_profiler)

    # create a new program:
    impl.get_runtime().create_program()

    _logging.trace('Materializing runtime...')
    impl.get_runtime().prog.materialize_runtime()

    impl._root_fb = FieldsBuilder()

    if not os.environ.get("TI_DISABLE_SIGNAL_HANDLERS", False):
        impl.get_runtime()._register_signal_handlers()

    return None


def no_activate(*args):
    for v in args:
        _ti_core.no_activate(v.snode.ptr)


def block_local(*args):
    """Hints Taichi to cache the fields and to enable the BLS optimization.

    Please visit https://docs.taichi.graphics/lang/articles/advanced/performance
    for how BLS is used.

    Args:
        *args (List[Field]): A list of sparse Taichi fields.
    """
    if impl.current_cfg().opt_level == 0:
        _logging.warn("""opt_level = 1 is enforced to enable bls analysis.""")
        impl.current_cfg().opt_level = 1
    for a in args:
        for v in a.get_field_members():
            _ti_core.insert_snode_access_flag(
                _ti_core.SNodeAccessFlag.block_local, v.ptr)


def mesh_local(*args):
    for a in args:
        for v in a.get_field_members():
            _ti_core.insert_snode_access_flag(
                _ti_core.SNodeAccessFlag.mesh_local, v.ptr)


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
global_thread_idx = _ti_core.insert_thread_idx_expr
mesh_patch_idx = _ti_core.insert_patch_idx_expr


def Tape(loss, clear_gradients=True):
    """Return a context manager of :class:`~taichi.lang.tape.TapeImpl`. The
    context manager would catching all of the callings of functions that
    decorated by :func:`~taichi.lang.kernel_impl.kernel` or
    :func:`~taichi.ad.grad_replaced` under `with` statement, and calculate
    all the partial gradients of a given loss variable by calling all of the
    gradient function of the callings caught in reverse order while `with`
    statement ended.

    See also :func:`~taichi.lang.kernel_impl.kernel` and
    :func:`~taichi.ad.grad_replaced` for gradient functions.

    Args:
        loss(:class:`~taichi.lang.expr.Expr`): The loss field, which shape should be ().
        clear_gradients(Bool): Before `with` body start, clear all gradients or not.

    Returns:
        :class:`~taichi.lang.tape.TapeImpl`: The context manager.

    Example::

        >>> @ti.kernel
        >>> def sum(a: ti.float32):
        >>>     for I in ti.grouped(x):
        >>>         y[None] += x[I] ** a
        >>>
        >>> with ti.Tape(loss = y):
        >>>     sum(2)"""
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

    from taichi._kernels import clear_loss  # pylint: disable=C0415
    clear_loss(loss)

    return runtime.get_tape(loss)


def clear_all_gradients():
    """Set all fields' gradients to 0."""
    impl.get_runtime().materialize()

    def visit(node):
        places = []
        for _i in range(node.ptr.get_num_ch()):
            ch = node.ptr.get_ch(_i)
            if not ch.is_place():
                visit(SNode(ch))
            else:
                if not ch.is_primal():
                    places.append(ch.get_expr())

        places = tuple(places)
        if places:
            from taichi._kernels import \
                clear_gradients  # pylint: disable=C0415
            clear_gradients(places)

    for root_fb in FieldsBuilder.finalized_roots():
        visit(root_fb)


def benchmark(_func, repeat=300, args=()):
    def run_benchmark():
        compile_time = time.time()
        _func(*args)  # compile the kernel first
        sync()
        compile_time = time.time() - compile_time
        stat_write('compilation_time', compile_time)
        codegen_stat = _ti_core.stat()
        for line in codegen_stat.split('\n'):
            try:
                a, b = line.strip().split(':')
            except:
                continue
            a = a.strip()
            b = int(float(b))
            if a == 'codegen_kernel_statements':
                stat_write('compiled_inst', b)
            if a == 'codegen_offloaded_tasks':
                stat_write('compiled_tasks', b)
            elif a == 'launched_tasks':
                stat_write('launched_tasks', b)

        # Use 3 initial iterations to warm up
        # instruction/data caches. Discussion:
        # https://github.com/taichi-dev/taichi/pull/1002#discussion_r426312136
        for _ in range(3):
            _func(*args)
            sync()
        clear_kernel_profile_info()
        t = time.time()
        for _ in range(repeat):
            _func(*args)
            sync()
        elapsed = time.time() - t
        avg = elapsed / repeat
        stat_write('wall_clk_t', avg)
        device_time = kernel_profiler_total_time()
        avg_device_time = device_time / repeat
        stat_write('exec_t', avg_device_time)

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
    import matplotlib.pyplot as plt  # pylint: disable=C0415
    import yaml  # pylint: disable=C0415
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
        warning(
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
    for case_id, case in enumerate(cases):
        subfigures[case_id][0].annotate(
            case,
            xy=(0, 0.5),
            xytext=(-subfigures[case_id][0].yaxis.labelpad - 5, 0),
            xycoords=subfigures[case_id][0].yaxis.label,
            textcoords='offset points',
            size='large',
            ha='right',
            va='center')
        for col_id, col in enumerate(columns):
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
                for _i in range(len(current_archs)):
                    maximum = max(y_left[_i], y_right[_i])
                    y_left[_i] = y_left[_i] / maximum if y_left[_i] != 0 else 1
                    y_right[
                        _i] = y_right[_i] / maximum if y_right[_i] != 0 else 1
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
    import yaml  # pylint: disable=C0415
    case_name = os.environ.get('TI_CURRENT_BENCHMARK')
    if case_name is None:
        return
    if case_name.startswith('benchmark_'):
        case_name = case_name[10:]
    arch_name = _ti_core.arch_name(cfg.arch)
    async_mode = 'async' if cfg.async_mode else 'sync'
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


def is_arch_supported(arch, use_gles=False):
    """Checks whether an arch is supported on the machine.

    Args:
        arch (taichi_core.Arch): Specified arch.
        use_gles (bool): If True, check is GLES is available otherwise
          check if GLSL is available. Only effective when `arch` is `ti.opengl`.
          Default is `False`.

    Returns:
        bool: Whether `arch` is supported on the machine.
    """

    arch_table = {
        cuda: _ti_core.with_cuda,
        metal: _ti_core.with_metal,
        opengl: functools.partial(_ti_core.with_opengl, use_gles),
        cc: _ti_core.with_cc,
        vulkan: _ti_core.with_vulkan,
        dx11: _ti_core.with_dx11,
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
            f"{arch}, consider adding `TI_ENABLE_{arch.upper()}=0` "
            f" to environment variables to suppress this warning message.")
        return False


def adaptive_arch_select(arch, enable_fallback, use_gles):
    if arch is None:
        return cpu
    if not isinstance(arch, (list, tuple)):
        arch = [arch]
    for a in arch:
        if is_arch_supported(a, use_gles):
            return a
    if not enable_fallback:
        raise RuntimeError(f'Arch={arch} is not supported')
    _logging.warn(f'Arch={arch} is not supported, falling back to CPU')
    return cpu


def get_host_arch_list():
    return [_ti_core.host_arch()]


__all__ = [s for s in dir() if not s.startswith('_')]
