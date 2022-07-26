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

gpu = [cuda, metal, vulkan, opengl, dx11]
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

extension = _ti_core.Extension
"""An instance of Taichi extension.

The list of currently available extensions is ['sparse', 'quant', \
    'mesh', 'quant_basic', 'data64', 'adstack', 'bls', 'assertion', \
        'extfunc', 'packed', 'dynamic_index'].
"""


def is_extension_supported(arch, ext):
    """Checks whether an extension is supported on an arch.

    Args:
        arch (taichi_python.Arch): Specified arch.
        ext (taichi_python.Extension): Specified extension.

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

        # TI_OFFLINE_CACHE=   : no effect
        # TI_OFFLINE_CACHE=0  : False
        # TI_OFFLINE_CACHE=1  : True
        name = 'TI_' + key.upper()
        value = os.environ.get(name, '')
        if key in self.kwargs:
            self[key] = self.kwargs[key]
            if value:
                _ti_core.warn(
                    f'Environment variable {name}={value} overridden by ti.init argument "{key}"'
                )
            del self.kwargs[key]  # mark as recognized
        elif value:
            self[key] = _cast(value)

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
        self.short_circuit_operators = True


def prepare_sandbox():
    '''
    Returns a temporary directory, which will be automatically deleted on exit.
    It may contain the taichi_python shared object or some misc. files.
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
            * ``packed`` (bool): Enables the packed memory layout. See https://docs.taichi-lang.org/docs/layout.
    """
    # Check version for users every 7 days if not disabled by users.
    _version_check.start_version_check_thread()

    # FIXME(https://github.com/taichi-dev/taichi/issues/4811): save the current working directory since it may be
    # changed by the Vulkan backend initialization on OS X.
    current_dir = os.getcwd()

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
                f'Environment variable TI_DEFAULT_FP={env_default_fp} overridden by ti.init argument "default_fp"'
            )
        elif env_default_fp == '32':
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
                f'Environment variable TI_DEFAULT_IP={env_default_ip} overridden by ti.init argument "default_ip"'
            )
        elif env_default_ip == '32':
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

    if cfg.debug:
        impl.get_runtime()._register_signal_handlers()

    # Recover the current working directory (https://github.com/taichi-dev/taichi/issues/4811)
    os.chdir(current_dir)
    return None


def no_activate(*args):
    """Deactivates a SNode pointer.
    """
    for v in args:
        get_runtime().prog.no_activate(v._snode.ptr)


def block_local(*args):
    """Hints Taichi to cache the fields and to enable the BLS optimization.

    Please visit https://docs.taichi-lang.org/docs/performance
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
    """Hints the compiler to cache the mesh attributes
    and to enable the mesh BLS optimization,
    only available for backends supporting `ti.extension.mesh` and to use with mesh-for loop.

    Related to https://github.com/taichi-dev/taichi/issues/3608

    Args:
        *args (List[Attribute]): A list of mesh attributes or fields accessed as attributes.

    Examples::

        # instantiate model
        mesh_builder = ti.Mesh.tri()
        mesh_builder.verts.place({
            'x' : ti.f32,
            'y' : ti.f32
        })
        model = mesh_builder.build(meta)

        @ti.kernel
        def foo():
            # hint the compiler to cache mesh vertex attribute `x` and `y`.
            ti.mesh_local(model.verts.x, model.verts.y)
            for v0 in model.verts: # mesh-for loop
                for v1 in v0.verts:
                    v0.x += v1.y
    """
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
    """Hints the compiler that a value is between a specified range,
    for the compiler to perform scatchpad optimization, and return the
    value untouched.

    The assumed range is `[base + low, base + high)`.

    Args:

        val (Number): The input value.
        base (Number): The base point for the range interval.
        low (Number): The lower offset relative to `base` (included).
        high (Number): The higher offset relative to `base` (excluded).

    Returns:
        Return the input `value` untouched.

    Example::

        >>> # hint the compiler that x is in range [8, 12).
        >>> x = ti.assume_in_range(x, 10, -2, 2)
        >>> x
        10
    """
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
    """Sets the number of threads to use on CPU.
    """
    get_runtime().prog.current_ast_builder().parallelize(v)
    if v == 1:
        get_runtime().prog.current_ast_builder().strictly_serialize()


def _serialize():
    """Sets the number of threads to 1.
    """
    _parallelize(1)


def _block_dim(dim):
    """Set the number of threads in a block to `dim`.
    """
    get_runtime().prog.current_ast_builder().block_dim(dim)


def _block_dim_adaptive(block_dim_adaptive):
    """Enable/Disable backends set block_dim adaptively.
    """
    if get_runtime().prog.config.arch != cpu:
        _logging.warn('Adaptive block_dim is supported on CPU backend only')
    else:
        get_runtime().prog.config.cpu_block_dim_adaptive = block_dim_adaptive


def _bit_vectorize():
    """Enable bit vectorization of struct fors on quant_arrays.
    """
    get_runtime().prog.current_ast_builder().bit_vectorize()


def loop_config(*,
                block_dim=None,
                serialize=False,
                parallelize=None,
                block_dim_adaptive=True,
                bit_vectorize=False):
    """Sets directives for the next loop

    Args:
        block_dim (int): The number of threads in a block on GPU
        serialize (bool): Whether to let the for loop execute serially, `serialize=True` equals to `parallelize=1`
        parallelize (int): The number of threads to use on CPU
        block_dim_adaptive (bool): Whether to allow backends set block_dim adaptively, enabled by default
        bit_vectorize (bool): Whether to enable bit vectorization of struct fors on quant_arrays.

    Examples::

        @ti.kernel
        def break_in_serial_for() -> ti.i32:
            a = 0
            ti.loop_config(serialize=True)
            for i in range(100):  # This loop runs serially
                a += i
                if i == 10:
                    break
            return a

        break_in_serial_for()  # returns 55

        n = 128
        val = ti.field(ti.i32, shape=n)
        @ti.kernel
        def fill():
            ti.loop_config(parallelize=8, block_dim=16)
            # If the kernel is run on the CPU backend, 8 threads will be used to run it
            # If the kernel is run on the CUDA backend, each block will have 16 threads.
            for i in range(n):
                val[i] = i

        u1 = ti.types.quant.int(bits=1, signed=False)
        x = ti.field(dtype=u1)
        y = ti.field(dtype=u1)
        cell = ti.root.dense(ti.ij, (128, 4))
        cell.quant_array(ti.j, 32).place(x)
        cell.quant_array(ti.j, 32).place(y)
        @ti.kernel
        def copy():
            ti.loop_config(bit_vectorize=True)
            # 32 bits, instead of 1 bit, will be copied at a time
            for i, j in x:
                y[i, j] = x[i, j]
    """
    if block_dim is not None:
        _block_dim(block_dim)

    if serialize:
        _parallelize(1)
    elif parallelize is not None:
        _parallelize(parallelize)

    if not block_dim_adaptive:
        _block_dim_adaptive(block_dim_adaptive)

    if bit_vectorize:
        _bit_vectorize()


def global_thread_idx():
    """Returns the global thread id of this running thread,
    only available for cpu and cuda backends.

    For cpu backends this is equal to the cpu thread id,
    For cuda backends this is equal to `block_id * block_dim + thread_id`.

    Example::

        >>> f = ti.field(ti.f32, shape=(16, 16))
        >>> @ti.kernel
        >>> def test():
        >>>     for i in ti.grouped(f):
        >>>         print(ti.global_thread_idx())
        >>>
        test()
    """
    return impl.get_runtime().prog.current_ast_builder(
    ).insert_thread_idx_expr()


def mesh_patch_idx():
    """Returns the internal mesh patch id of this running thread,
    only available for backends supporting `ti.extension.mesh` and to use within mesh-for loop.

    Related to https://github.com/taichi-dev/taichi/issues/3608
    """
    return impl.get_runtime().prog.current_ast_builder().insert_patch_idx_expr(
    )


def is_arch_supported(arch, use_gles=False):
    """Checks whether an arch is supported on the machine.

    Args:
        arch (taichi_python.Arch): Specified arch.
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


def get_compute_stream_device_time_elapsed_us() -> float:
    return impl.get_runtime().prog.get_compute_stream_device_time_elapsed_us()


__all__ = [
    'i', 'ij', 'ijk', 'ijkl', 'ijl', 'ik', 'ikl', 'il', 'j', 'jk', 'jkl', 'jl',
    'k', 'kl', 'l', 'x86_64', 'x64', 'dx11', 'wasm', 'arm64', 'cc', 'cpu',
    'cuda', 'gpu', 'metal', 'opengl', 'vulkan', 'extension', 'loop_config',
    'global_thread_idx', 'assume_in_range', 'block_local', 'cache_read_only',
    'init', 'mesh_local', 'no_activate', 'reset', 'mesh_patch_idx',
    'get_compute_stream_device_time_elapsed_us'
]
