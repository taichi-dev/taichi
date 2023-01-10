from taichi._lib import core as _ti_core
from taichi.lang.exception import TaichiRuntimeError
from taichi.types import f32, f64


class CG:
    def __init__(self, A, b, x0=None, max_iter=50, atol=1e-6):
        self.dtype = A.dtype
        if self.dtype == f32:
            self.cg_solver = _ti_core.make_float_cg_solver(
                A.matrix, max_iter, atol, True)
        elif self.dtype == f64:
            self.cg_solver = _ti_core.make_double_cg_solver(
                A.matrix, max_iter, atol, True)
        else:
            raise TaichiRuntimeError(f'Unsupported CG dtype: {self.dtype}')
        self.cg_solver.set_b(b)
        self.cg_solver.set_x(x0)

    def solve(self):
        self.cg_solver.solve()
        return self.cg_solver.get_x(), self.cg_solver.is_success()
