from taichi._lib import core as _ti_core
from taichi.lang.exception import TaichiRuntimeError
from taichi.types import f32, f64
from taichi.lang.impl import get_runtime


class CG:
    def __init__(self, A, b, x0=None, max_iter=50, atol=1e-6):
        self.dtype = A.dtype
        self.ti_arch = get_runtime().prog.config().arch
        if self.ti_arch == _ti_core.Arch.cuda:
            self.cg_solver = _ti_core.make_cucg_solver(A.matrix, max_iter, atol)
        else:
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


class CG_Cuda:
    def __init__(self, A, b, x0=None, max_iter=50, atol=1e-6):
        self.dtype = A.dtype
        self.A = A
        self.b = b
        self.x0 = x0 
        self.max_iter = max_iter
        self.atol = atol

    def solve(self):
        r0 = self.b - self.A @ self.x0
        if r0.norm() < self.atol:
            return self.x0, True
        p = r0
        r = r0
        x = self.x0
        k = 0
        while k < self.max_iter:
            Ap = self.A @ p
            alpha = r.norm_sqr() / (p @ Ap)
            x_next = x + alpha * p
            r_next = r - alpha * Ap
            if r_next.norm() < self.atol:
                return x_next, True
            beta = r_next.norm_sqr() / r.norm_sqr()
            p = r_next + beta * p
            r = r_next
            x = x_next
            k += 1
        return x, True