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

    def compute(self, sparse_matrix):
        if isinstance(sparse_matrix, SparseMatrix):
            self.solver.compute(sparse_matrix.matrix)
        else:
            assert False, f"The parameter type: {type(sparse_matrix)} is not support in linear solver for now."

    def analyzePattern(self, sparse_matrix):
        if isinstance(sparse_matrix, SparseMatrix):
            self.solver.analyzePattern(sparse_matrix.matrix)
        else:
            assert False, f"The parameter type: {type(sparse_matrix)} is not support in linear solver for now."

    def factorize(self, sparse_matrix):
        if isinstance(sparse_matrix, SparseMatrix):
            self.solver.factorize(sparse_matrix.matrix)
        else:
            assert False, f"The parameter type: {type(sparse_matrix)} is not support in linear solver for now."

    def solve(self, b):
        if isinstance(b, np.ndarray):
            return self.solver.solve(b)
        else:
            assert False, f"The parameter type: {type(b)} is not support in linear solver for now."
