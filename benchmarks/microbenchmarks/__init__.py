from .atomic_ops import AtomicOpsPlan
from .fill import FillPlan
from .math_opts import MathOpsPlan
from .memcpy import MemcpyPlan
from .saxpy import SaxpyPlan
from .stencil2d import Stencil2DPlan

benchmark_plan_list = [
    AtomicOpsPlan, FillPlan, MathOpsPlan, MemcpyPlan, SaxpyPlan, Stencil2DPlan
]
