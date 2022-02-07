from .atomic_ops import AtomicOpsPlan
from .fill import FillPlan
from .math_opts_throughput import MathOpsThroughputPlan

benchmark_plan_list = [FillPlan, AtomicOpsPlan, MathOpsThroughputPlan]
