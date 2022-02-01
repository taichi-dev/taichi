from .atomic_ops import AtomicOpsPlan
from .fill import FillPlan
from .math_opts_throughput import MathOpsThroughputPlan
from .stencil2d import Stencil2DPlan

benchmark_plan_list = [
    FillPlan, AtomicOpsPlan, MathOpsThroughputPlan, Stencil2DPlan
]
