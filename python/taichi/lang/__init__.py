import inspect
import platform

from taichi._lib import core as _ti_core
from taichi._lib.utils import locale_encode
from taichi.lang import impl
from taichi.lang._ndrange import ndrange
from taichi.lang.enums import Layout
from taichi.lang.exception import (TaichiCompilationError, TaichiNameError,
                                   TaichiSyntaxError, TaichiTypeError)
from taichi.lang.impl import (axes, deactivate_all_snodes, field, grouped,
                              ndarray, one, root, static, static_assert,
                              static_print, stop_grad, zero)
from taichi.lang.kernel_arguments import SparseMatrixProxy
from taichi.lang.kernel_impl import (KernelArgError, KernelDefError,
                                     data_oriented, func, kernel, pyfunc)
from taichi.lang.matrix import Matrix, MatrixField, Vector
from taichi.lang.mesh import Mesh, MeshElementFieldProxy, TetMesh, TriMesh
from taichi.lang.misc import *  # pylint: disable=W0622
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
from taichi.profiler import KernelProfiler, get_default_kernel_profiler
from taichi.profiler.kernelmetrics import (CuptiMetric, default_cupti_metrics,
                                           get_predefined_cupti_metrics)
from taichi.tools.util import set_gdb_trigger, warning
from taichi.types.annotations import any_arr, ext_arr, template
from taichi.types.primitive_types import f16, f32, f64, i32, i64, u32, u64

from taichi import _logging, _snode

__all__ = [
    s for s in dir()
    if not s.startswith('_') and not inspect.ismodule(globals()[s])
    or s in ['tape', 'sort']
]
