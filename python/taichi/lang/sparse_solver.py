from taichi.lang.sparse_matrix import SparseMatrix


class SparseSolver:
    def __init__(self, solver_type="LLT"):
        solver_type_list = ["LLT", "LDLT", "LU"]
        if solver_type in solver_type_list:
            from taichi.core.util import ti_core as _ti_core
            from taichi.lang.impl import get_runtime
            taichi_arch = get_runtime().prog.config.arch
            assert taichi_arch == _ti_core.Arch.x64 or taichi_arch == _ti_core.Arch.arm64, "SparseSolver only supports CPU for now."
            self.solver = _ti_core.get_sparse_solver(solver_type)
        else:
            assert False, f"The solver type {solver_type} is not support for now. Only {solver_type_list} are supported."

    @staticmethod
    def type_assert(sparse_matrix):
        assert False, f"The parameter type: {type(sparse_matrix)} is not supported in linear solvers for now."

    def compute(self, sparse_matrix):
        if isinstance(sparse_matrix, SparseMatrix):
            self.solver.compute(sparse_matrix.matrix)
        else:
            self.type_assert(sparse_matrix)

    def analyze_pattern(self, sparse_matrix):
        if isinstance(sparse_matrix, SparseMatrix):
            self.solver.analyze_pattern(sparse_matrix.matrix)
        else:
            self.type_assert(sparse_matrix)

    def factorize(self, sparse_matrix):
        if isinstance(sparse_matrix, SparseMatrix):
            self.solver.factorize(sparse_matrix.matrix)
        else:
            self.type_assert(sparse_matrix)

    def solve(self, b):
        import numpy as np
        from taichi.lang import Field
        if isinstance(b, Field):
            return self.solver.solve(b.to_numpy())
        elif isinstance(b, np.ndarray):
            return self.solver.solve(b)
        else:
            assert False, f"The parameter type: {type(b)} is not supported in linear solvers for now."

    def info(self):
        return self.solver.info()
