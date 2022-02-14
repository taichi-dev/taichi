from .atomic_ops import AtomicOpsPlan
from .fill import FillPlan
from .math_opts_throughput import MathOpsThroughputPlan
from .memcpy import MemcpyPlan

benchmark_plan_list = [
    FillPlan, AtomicOpsPlan, MathOpsThroughputPlan, MemcpyPlan
]
