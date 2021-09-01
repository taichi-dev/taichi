import numpy as np
from taichi.lang.sparse_matrix import SparseMatrix


class SparseSolver:
    def __init__(self, solver_type="LLT"):
        solver_map = {"LLT": 0, "LDLT": 1, "LU": 2}
        if solver_type in solver_map:
            from taichi.core.util import ti_core as _ti_core
            if solver_type == "LLT":
                self.solver = _ti_core.SparseLLTSolver()
            elif solver_type == "LDLT":
                self.solver = _ti_core.SparseLDLTSolver()
            elif solver_type == "LU":
                self.solver = _ti_core.SparseLUSolver()
        else:
            assert False, f"The solver type is not support for now."

    @staticmethod
    def type_assert(sparse_matrix):
        assert False, f"The parameter type: {type(sparse_matrix)} is not support in linear solver for now."

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
        from taichi.lang import Field
        if isinstance(b, Field):
            return self.solver.solve(b.to_numpy())
        else:
            assert False, f"The parameter type: {type(b)} is not support in linear solver for now."
