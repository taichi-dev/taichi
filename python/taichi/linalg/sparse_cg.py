import numpy as np
from taichi._lib import core as _ti_core
from taichi.lang._ndarray import Ndarray, ScalarNdarray
from taichi.lang.exception import TaichiRuntimeError
from taichi.lang.impl import get_runtime
from taichi.types import f32, f64


class SparseCG:
    """Conjugate-gradient solver built for SparseMatrix.

    Use conjugate-gradient method to solve the linear system Ax = b, where A is SparseMatrix.

    Args:
        A (SparseMatrix): The coefficient matrix A of the linear system.
        b (numpy ndarray, taichi Ndarray): The right-hand side of the linear system.
        x0 (numpy ndarray, taichi Ndarray): The initial guess for the solution.
        max_iter (int): Maximum number of iterations.
        atol: Tolerance(absolute) for convergence.
    """

    def __init__(self, A, b, x0=None, max_iter=50, atol=1e-6):
        self.dtype = A.dtype
        self.ti_arch = get_runtime().prog.config().arch
        self.matrix = A
        self.b = b
        if self.ti_arch == _ti_core.Arch.cuda:
            self.cg_solver = _ti_core.make_cucg_solver(A.matrix, max_iter, atol, True)
        elif self.ti_arch == _ti_core.Arch.x64 or self.ti_arch == _ti_core.Arch.arm64:
            if self.dtype == f32:
                self.cg_solver = _ti_core.make_float_cg_solver(A.matrix, max_iter, atol, True)
            elif self.dtype == f64:
                self.cg_solver = _ti_core.make_double_cg_solver(A.matrix, max_iter, atol, True)
            else:
                raise TaichiRuntimeError(f"Unsupported CG dtype: {self.dtype}")
            if isinstance(b, Ndarray):
                self.cg_solver.set_b_ndarray(get_runtime().prog, b.arr)
            elif isinstance(b, np.ndarray):
                self.cg_solver.set_b(b)
            if isinstance(x0, Ndarray):
                self.cg_solver.set_x_ndarray(get_runtime().prog, x0.arr)
            elif isinstance(x0, np.ndarray):
                self.cg_solver.set_x(x0)
        else:
            raise TaichiRuntimeError(f"Unsupported CG arch: {self.ti_arch}")

    def solve(self):
        if self.ti_arch == _ti_core.Arch.cuda:
            if isinstance(self.b, Ndarray):
                x = ScalarNdarray(self.b.dtype, [self.matrix.m])
                self.cg_solver.solve(get_runtime().prog, x.arr, self.b.arr)
                return x, True
            raise TaichiRuntimeError(f"Unsupported CG RHS type: {type(self.b)}")
        else:
            self.cg_solver.solve()
            return self.cg_solver.get_x(), self.cg_solver.is_success()
