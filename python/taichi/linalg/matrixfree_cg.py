from math import sqrt

from taichi.lang.exception import TaichiRuntimeError, TaichiTypeError

import taichi as ti


@ti.data_oriented
class LinearOperator:
    def __init__(self, matvec_kernel):
        self._matvec = matvec_kernel

    def matvec(self, x, Ax):
        if x.shape != Ax.shape:
            raise TaichiRuntimeError(f"Dimension mismatch x.shape{x.shape} != Ax.shape{Ax.shape}.")
        self._matvec(x, Ax)


def MatrixFreeCG(A, b, x, tol=1e-6, maxiter=5000, quiet=True):
    """Matrix-free conjugate-gradient solver.

    Use conjugate-gradient method to solve the linear system Ax = b, where A is implicitly
    represented as a LinearOperator.

    Args:
        A (LinearOperator): The coefficient matrix A of the linear system.
        b (Field): The right-hand side of the linear system.
        x (Field): The initial guess for the solution.
        maxiter (int): Maximum number of iterations.
        atol: Tolerance(absolute) for convergence.
        quiet (bool): Switch to turn on/off iteration log.
    """

    if b.dtype != x.dtype:
        raise TaichiTypeError(f"Dtype mismatch b.dtype({b.dtype}) != x.dtype({x.dtype}).")
    if str(b.dtype) == "f32":
        solver_dtype = ti.f32
    elif str(b.dtype) == "f64":
        solver_dtype = ti.f64
    else:
        raise TaichiTypeError(f"Not supported dtype: {b.dtype}")
    if b.shape != x.shape:
        raise TaichiRuntimeError(f"Dimension mismatch b.shape{b.shape} != x.shape{x.shape}.")

    size = b.shape
    vector_fields_builder = ti.FieldsBuilder()
    p = ti.field(dtype=solver_dtype)
    r = ti.field(dtype=solver_dtype)
    Ap = ti.field(dtype=solver_dtype)
    vector_fields_builder.dense(ti.ij, size).place(p, r, Ap)
    vector_fields_snode_tree = vector_fields_builder.finalize()

    scalar_builder = ti.FieldsBuilder()
    alpha = ti.field(dtype=solver_dtype)
    beta = ti.field(dtype=solver_dtype)
    scalar_builder.place(alpha, beta)
    scalar_snode_tree = scalar_builder.finalize()

    @ti.kernel
    def init():
        for I in ti.grouped(x):
            r[I] = b[I]
            p[I] = 0.0
            Ap[I] = 0.0

    @ti.kernel
    def reduce(p: ti.template(), q: ti.template()) -> solver_dtype:
        result = 0.0
        for I in ti.grouped(p):
            result += p[I] * q[I]
        return result

    @ti.kernel
    def update_x():
        for I in ti.grouped(x):
            x[I] += alpha[None] * p[I]

    @ti.kernel
    def update_r():
        for I in ti.grouped(r):
            r[I] -= alpha[None] * Ap[I]

    @ti.kernel
    def update_p():
        for I in ti.grouped(p):
            p[I] = r[I] + beta[None] * p[I]

    def solve():
        init()
        initial_rTr = reduce(r, r)
        if not quiet:
            print(f">>> Initial residual = {initial_rTr:e}")
        old_rTr = initial_rTr
        update_p()
        # -- Main loop --
        for i in range(maxiter):
            A._matvec(p, Ap)  # compute Ap = A x p
            pAp = reduce(p, Ap)
            alpha[None] = old_rTr / pAp
            update_x()
            update_r()
            new_rTr = reduce(r, r)
            if sqrt(new_rTr) < tol:
                if not quiet:
                    print(">>> Conjugate Gradient method converged.")
                    print(f">>> #iterations {i}")
                break
            beta[None] = new_rTr / old_rTr
            update_p()
            old_rTr = new_rTr
            if not quiet:
                print(f">>> Iter = {i+1:4}, Residual = {sqrt(new_rTr):e}")

    solve()
    vector_fields_snode_tree.destroy()
    scalar_snode_tree.destroy()
