import atexit
import functools
import os
import shutil
import tempfile
import warnings
from copy import deepcopy as _deepcopy

from taichi._lib import core as _ti_core
from taichi._lib.utils import locale_encode
from taichi.lang import impl
from taichi.lang.expr import Expr
from taichi.lang.impl import axes, get_runtime
from taichi.lang.snode import SNode
from taichi.profiler.kernel_profiler import get_default_kernel_profiler
from taichi.types.primitive_types import f32, f64, i32, i64

from taichi import _logging, _snode, _version_check

warnings.filterwarnings("once", category=DeprecationWarning, module="taichi")

# ----------------------
i = axes(0)
"""Axis 0. For multi-dimensional arrays it's the direction downward the rows.
For a 1d array it's the direction along this array.
"""
# ----------------------

j = axes(1)
"""Axis 1. For multi-dimensional arrays it's the direction across the columns.
"""
# ----------------------

k = axes(2)
"""Axis 2. For arrays of dimension `d` >= 3, view each cell as an array of
lower dimension d-2, it's the first axis of this cell.
"""
# ----------------------

l = axes(3)
"""Axis 3. For arrays of dimension `d` >= 4, view each cell as an array of
lower dimension d-2, it's the second axis of this cell.
"""
# ----------------------

ij = axes(0, 1)
"""Axes (0, 1).
"""
# ----------------------

ik = axes(0, 2)
"""Axes (0, 2).
"""
# ----------------------

il = axes(0, 3)
"""Axes (0, 3).
"""
# ----------------------

jk = axes(1, 2)
"""Axes (1, 2).
"""
# ----------------------

jl = axes(1, 3)
"""Axes (1, 3).
"""
# ----------------------

kl = axes(2, 3)
"""Axes (2, 3).
"""
# ----------------------

ijk = axes(0, 1, 2)
"""Axes (0, 1, 2).
"""
# ----------------------

ijl = axes(0, 1, 3)
"""Axes (0, 1, 3).
"""
# ----------------------

ikl = axes(0, 2, 3)
"""Axes (0, 2, 3).
"""
# ----------------------

jkl = axes(1, 2, 3)
"""Axes (1, 2, 3).
"""
# ----------------------

ijkl = axes(0, 1, 2, 3)
"""Axes (0, 1, 2, 3).
"""
# ----------------------

# ----------------------

x86_64 = _ti_core.x64
"""The x64 CPU backend.
"""
# ----------------------

x64 = _ti_core.x64
"""The X64 CPU backend.
"""
# ----------------------

arm64 = _ti_core.arm64
"""The ARM CPU backend.
"""
# ----------------------

cuda = _ti_core.cuda
"""The CUDA backend.
"""
# ----------------------

metal = _ti_core.metal
"""The Apple Metal backend.
"""
# ----------------------

opengl = _ti_core.opengl
"""The OpenGL backend. OpenGL 4.3 required.
"""
# ----------------------

# Skip annotating this one because it is barely maintained.
cc = _ti_core.cc

# ----------------------

wasm = _ti_core.wasm
"""The WebAssembly backend.
"""
# ----------------------

vulkan = _ti_core.vulkan
"""The Vulkan backend.
"""
# ----------------------

dx11 = _ti_core.dx11
"""The DX11 backend.
"""
# ----------------------

gpu = [cuda, metal, opengl, vulkan, dx11]
"""A list of GPU backends supported on the current system.
Currently contains 'cuda', 'metal', 'opengl', 'vulkan', 'dx11'.

When this is used, Taichi automatically picks the matching GPU backend. If no
GPU is detected, Taichi falls back to the CPU backend.
"""
# ----------------------

cpu = _ti_core.host_arch()
"""A list of CPU backends supported on the current system.
Currently contains 'x64', 'x86_64', 'arm64', 'cc', 'wasm'.

When this is used, Taichi automatically picks the matching CPU backend.
"""
# ----------------------

timeline_clear = lambda: impl.get_runtime().prog.timeline_clear()  # pylint: disable=unnecessary-lambda
timeline_save = lambda fn: impl.get_runtime().prog.timeline_save(fn)  # pylint: disable=unnecessary-lambda

# Legacy API
type_factory_ = _ti_core.get_type_factory_instance()

extension = _ti_core.Extension
"""An instance of Taichi extension.

The list of currently available extensions is ['sparse', 'async_mode', 'quant', \
    'mesh', 'quant_basic', 'data64', 'adstack', 'bls', 'assertion', \
        'extfunc', 'packed', 'dynamic_index'].
"""


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
    This will destroy all the allocated fields and kernels, and restore
    the runtime to its default configuration.

    Example::

        >>> a = ti.field(ti.i32, shape=())
        >>> a[None] = 1
        >>> print("before reset: ", a)
        before rest: 1
        >>>
        >>> ti.reset()
        >>> print("after reset: ", a)
        # will raise error because a is unavailable after reset.
    """
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
        self.short_circuit_operators = False


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


def check_require_version(require_version):
    '''
    Check if installed version meets the requirements.
    Allow to specify <major>.<minor>.<patch>.<hash>.
    <patch>.<hash> is optional. If not match, raise an exception.
    '''
    # Extract version number part (i.e. toss any revision / hash parts).
    version_number_str = require_version
    for c_idx, c in enumerate(require_version):
        if not (c.isdigit() or c == "."):
            version_number_str = require_version[:c_idx]
            break
    # Get required version.
    try:
        version_number_tuple = tuple(
            [int(n) for n in version_number_str.split(".")])
        major = version_number_tuple[0]
        minor = version_number_tuple[1]
        patch = 0
        if len(version_number_tuple) > 2:
            patch = version_number_tuple[2]
    except:
        raise Exception("The require_version should be formatted following PEP 440, " \
            "and inlucdes major, minor, and patch number, " \
            "e.g., major.minor.patch.") from None
    # Get installed version
    versions = [
        int(_ti_core.get_version_major()),
        int(_ti_core.get_version_minor()),
        int(_ti_core.get_version_patch()),
    ]
    # Match installed version and required version.
    match = major == versions[0] and (
        minor < versions[1] or minor == versions[1] and patch <= versions[2])

    if not match:
        raise Exception(
            f"Taichi version mismatch. Required version >= {major}.{minor}.{patch}, installed version = {_ti_core.get_version_string()}."
        )


def init(arch=None,
         default_fp=None,
         default_ip=None,
         _test_mode=False,
         enable_fallback=True,
         require_version=None,
         **kwargs):
    """Initializes the Taichi runtime.

    This should always be the entry point of your Taichi program. Most
    importantly, it sets the backend used throughout the program.

    Args:
        arch: Backend to use. This is usually :const:`~taichi.lang.cpu` or :const:`~taichi.lang.gpu`.
        default_fp (Optional[type]): Default floating-point type.
        default_ip (Optional[type]): Default integral type.
        require_version (Optional[string]): A version string.
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
    _version_check.start_version_check_thread()

    cfg = impl.default_cfg()
    # Check if installed version meets the requirements.
    if require_version is not None:
        check_require_version(require_version)

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
    env_spec.add('short_circuit_operators')

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
        _ti_core.set_core_trigger_gdb_when_crash(spec_cfg.gdb_trigger)
        impl.get_runtime().short_circuit_operators = \
            spec_cfg.short_circuit_operators
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

    # user selected visible device
    visible_device = os.environ.get("TI_VISIBLE_DEVICE")
    if visible_device and (cfg.arch == vulkan or _ti_core.GGUI_AVAILABLE):
        _ti_core.set_vulkan_visible_device(visible_device)

    if _test_mode:
        return spec_cfg

    get_default_kernel_profiler().set_kernel_profiler_mode(cfg.kernel_profiler)

    # create a new program:
    impl.get_runtime().create_program()

    _logging.trace('Materializing runtime...')
    impl.get_runtime().prog.materialize_runtime()

    impl._root_fb = _snode.FieldsBuilder()

    if not os.environ.get("TI_DISABLE_SIGNAL_HANDLERS", False):
        impl.get_runtime()._register_signal_handlers()

    return None


def no_activate(*args):
    for v in args:
        get_runtime().prog.no_activate(v._snode.ptr)


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
        for v in a._get_field_members():
            get_runtime().prog.current_ast_builder().insert_snode_access_flag(
                _ti_core.SNodeAccessFlag.block_local, v.ptr)


def mesh_local(*args):
    for a in args:
        for v in a._get_field_members():
            get_runtime().prog.current_ast_builder().insert_snode_access_flag(
                _ti_core.SNodeAccessFlag.mesh_local, v.ptr)


def cache_read_only(*args):
    for a in args:
        for v in a._get_field_members():
            get_runtime().prog.current_ast_builder().insert_snode_access_flag(
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


def _parallelize(v):
    get_runtime().prog.current_ast_builder().parallelize(v)
    if v == 1:
        get_runtime().prog.current_ast_builder().strictly_serialize()


def _serialize():
    _parallelize(1)


def _block_dim(dim):
    """Set the number of threads in a block to `dim`.
    """
    get_runtime().prog.current_ast_builder().block_dim(dim)


def loop_config(block_dim=None, serialize=None, parallelize=None):
    if block_dim is not None:
        _block_dim(block_dim)

    if serialize:
        _parallelize(1)
    elif parallelize is not None:
        _parallelize(parallelize)


def global_thread_idx():
    return impl.get_runtime().prog.current_ast_builder(
    ).insert_thread_idx_expr()


def mesh_patch_idx():
    return impl.get_runtime().prog.current_ast_builder().insert_patch_idx_expr(
    )


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
        >>>     sum(2)
    """
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

    return impl.get_runtime().get_tape(loss)


def clear_all_gradients():
    """Set the gradients of all fields to zero.
    """
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

    for root_fb in _snode.FieldsBuilder._finalized_roots():
        visit(root_fb)


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


__all__ = [
    'i', 'ij', 'ijk', 'ijkl', 'ijl', 'ik', 'ikl', 'il', 'j', 'jk', 'jkl', 'jl',
    'k', 'kl', 'l', 'x86_64', 'x64', 'dx11', 'wasm', 'arm64', 'cc', 'cpu',
    'cuda', 'gpu', 'metal', 'opengl', 'vulkan', 'extension', 'loop_config',
    'global_thread_idx', 'Tape', 'assume_in_range', 'block_local',
    'cache_read_only', 'clear_all_gradients', 'init', 'mesh_local',
    'no_activate', 'reset', 'mesh_patch_idx'
]
