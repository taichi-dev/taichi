import numpy as np
import taichi.lang
from taichi.core.util import ti_core as _ti_core
from taichi.linalg import SparseMatrix
from taichi.type.primitive_types import f32


class SparseSolver:
    """Sparse linear system solver

    Use this class to solve linear systems represented by sparse matrices.

    Args:
        solver_type (str): The factorization type.
        ordering (str): The method for matrices re-ordering.
    """
    def __init__(self, dtype=f32, solver_type="LLT", ordering="AMD"):
        solver_type_list = ["LLT", "LDLT", "LU"]
        solver_ordering = ['AMD', 'COLAMD']
        if solver_type in solver_type_list and ordering in solver_ordering:
            taichi_arch = taichi.lang.impl.get_runtime().prog.config.arch
            assert taichi_arch == _ti_core.Arch.x64 or taichi_arch == _ti_core.Arch.arm64, "SparseSolver only supports CPU for now."
            self.solver = _ti_core.make_sparse_solver(solver_type, ordering)
        else:
            assert False, f"The solver type {solver_type} with {ordering} is not supported for now. Only {solver_type_list} with {solver_ordering} are supported."

    @staticmethod
    def type_assert(sparse_matrix):
        assert False, f"The parameter type: {type(sparse_matrix)} is not supported in linear solvers for now."

    def compute(self, sparse_matrix):
        """This method is equivalent to calling both `analyze_pattern` and then `factorize`.

        Args:
            sparse_matrix (SparseMatrix): The sparse matrix to be computed.
        """
        if isinstance(sparse_matrix, SparseMatrix):
            self.solver.compute(sparse_matrix.matrix)
        else:
            self.type_assert(sparse_matrix)

    def analyze_pattern(self, sparse_matrix):
        """Reorder the nonzero elements of the matrix, such that the factorization step creates less fill-in.

        Args:
            sparse_matrix (SparseMatrix): The sparse matrix to be analyzed.
        """
        if isinstance(sparse_matrix, SparseMatrix):
            self.solver.analyze_pattern(sparse_matrix.matrix)
        else:
            self.type_assert(sparse_matrix)

    def factorize(self, sparse_matrix):
        """Do the factorization step

        Args:
            sparse_matrix (SparseMatrix): The sparse matrix to be factorized.
        """
        if isinstance(sparse_matrix, SparseMatrix):
            self.solver.factorize(sparse_matrix.matrix)
        else:
            self.type_assert(sparse_matrix)

    def solve(self, b):
        """Computes the solution of the linear systems.
        Args:
            b (numpy.array or Field): The right-hand side of the linear systems.

        Returns:
            numpy.array: The solution of linear systems.
        """
        if isinstance(b, taichi.lang.Field):
            return self.solver.solve(b.to_numpy())
        elif isinstance(b, np.ndarray):
            return self.solver.solve(b)
        else:
            assert False, f"The parameter type: {type(b)} is not supported in linear solvers for now."

    def info(self):
        """Check if the linear systems are solved successfully.

        Returns:
            bool: True if the solving process succeeded, False otherwise.
        """
        return self.solver.info()
