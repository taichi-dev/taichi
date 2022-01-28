from microbenchmarks._items import BenchmarkItem, DataType
from microbenchmarks._metric import MetricType
from microbenchmarks._plan import BenchmarkPlan
from microbenchmarks._utils import dtype_size, scaled_repeat_times

import taichi as ti


def matrix_operations_default(arch, repeat, matrix_op, block_mn, element_num,
                              dtype, get_metric):
    m, n = block_mn
    global_matrixA = ti.Matrix.field(m, n, dtype, shape=element_num)
    global_matrixB = ti.Matrix.field(m, n, dtype, shape=element_num)
    global_matrixC = ti.Matrix.field(m, n, dtype, shape=element_num)

    @ti.kernel
    def fill_matrixA():
        for e in global_matrixA:
            for i, j in ti.static(range(m, n)):
                global_matrixA[e][i, j] = ti.random(dtype)

    @ti.kernel
    def fill_matrixB():
        for e in global_matrixB:
            for i, j in ti.static(range(m, n)):
                global_matrixB[e][i, j] = ti.random(dtype)

    @ti.kernel
    def op_throughput():
        for e in range(element_num):
            #prelogue
            A = global_matrixA[e]
            B = global_matrixB[e]
            C = ti.Matrix.zero(dtype, m, n)
            C = matrix_op(C, A, B)  #C += A@B
            #loop
            for i in range(2048):
                for j in ti.static(range(4)):  #16*4*4=256
                    A = matrix_op(A, C, B)  #A += C@B
                    C = matrix_op(C, A, B)  #C += A@B
                    B = matrix_op(B, A, C)  #B += A@C
                    C = matrix_op(C, A, B)  #C += A@B
            #epilogue
            global_matrixC[e] = C

    fill_matrixA()
    fill_matrixB()
    return get_metric(repeat, op_throughput)


@ti.func
def matrix_mul(C, A, B):
    C = A @ B + C
    return C


@ti.func
def matrix_add(C, A, B):
    C = A + B
    return C


class MatrixOps(BenchmarkItem):
    name = 'matrix_op'

    def __init__(self):
        self._items = {
            'mat_mul': matrix_mul,
            'mat_add': matrix_add,
        }


class BlockMN(BenchmarkItem):
    name = 'block_mn'

    def __init__(self):
        self._items = {
            'block_mn11': (1, 1),
            'block_mn22': (2, 2),
            'block_mn33': (3, 3),
            'block_mn44': (4, 4),
        }


class ElementNum(BenchmarkItem):
    name = 'element_num'

    def __init__(self):
        self._items = {
            'element16384': 16384,
            #enough threads for filling CUDA cores
        }


class MatrixOperationPlan(BenchmarkPlan):
    def __init__(self, arch: str):
        super().__init__('matrix_ops', arch, basic_repeat_times=10)
        dtype = DataType()
        dtype.remove(['i64', 'f64'])
        self.create_plan(MatrixOps(), BlockMN(), ElementNum(), dtype,
                         MetricType())
        self.add_func(['element16384'], matrix_operations_default)
        self.print_plan()
