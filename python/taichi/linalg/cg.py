from taichi._lib import core as _ti_core


class CG:
    def __init__(self, A, b, x0=None, max_iter=50, atol=1e-6):
        self.cg_solver = _ti_core.make_cg_solver(A.matrix, max_iter, atol,
                                                 True)
        self.cg_solver.set_b(b)
        self.cg_solver.set_x(x0)

    def solve(self):
        self.cg_solver.solve()
        return self.cg_solver.get_x()
