from .atomic_ops import AtomicOpsPlan
from .fill import FillPlan
from .math_opts_throughput import MathOpsThroughputPlan
from .matrixops import MatrixOperationPlan

benchmark_plan_list = [FillPlan, AtomicOpsPlan, MathOpsThroughputPlan, MatrixOperationPlan]
