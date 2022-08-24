import numpy as np
import taichi.lang
from taichi._lib import core as _ti_core
from taichi.lang.exception import TaichiRuntimeError
from taichi.lang.field import Field
from taichi.lang.impl import get_runtime
from taichi.lang.matrix import Ndarray
from taichi.linalg.sparse_matrix import SparseMatrix
from taichi.types.primitive_types import f32


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
            assert taichi_arch == _ti_core.Arch.x64 or taichi_arch == _ti_core.Arch.arm64 or taichi_arch == _ti_core.Arch.cuda, "SparseSolver only supports CPU and CUDA for now."
            if taichi_arch == _ti_core.Arch.cuda:
                self.solver = _ti_core.make_cusparse_solver(
                    dtype, solver_type, ordering)
            else:
                self.solver = _ti_core.make_sparse_solver(
                    dtype, solver_type, ordering)
        else:
            raise TaichiRuntimeError(
                f"The solver type {solver_type} with {ordering} is not supported for now. Only {solver_type_list} with {solver_ordering} are supported."
            )

    @staticmethod
    def _type_assert(sparse_matrix):
        raise TaichiRuntimeError(
            f"The parameter type: {type(sparse_matrix)} is not supported in linear solvers for now."
        )

    def compute(self, sparse_matrix):
        """This method is equivalent to calling both `analyze_pattern` and then `factorize`.

        Args:
            sparse_matrix (SparseMatrix): The sparse matrix to be computed.
        """
        if isinstance(sparse_matrix, SparseMatrix):
            self.solver.compute(sparse_matrix.matrix)
        else:
            self._type_assert(sparse_matrix)

    def analyze_pattern(self, sparse_matrix):
        """Reorder the nonzero elements of the matrix, such that the factorization step creates less fill-in.

        Args:
            sparse_matrix (SparseMatrix): The sparse matrix to be analyzed.
        """
        if isinstance(sparse_matrix, SparseMatrix):
            self.solver.analyze_pattern(sparse_matrix.matrix)
        else:
            self._type_assert(sparse_matrix)

    def factorize(self, sparse_matrix):
        """Do the factorization step

        Args:
            sparse_matrix (SparseMatrix): The sparse matrix to be factorized.
        """
        if isinstance(sparse_matrix, SparseMatrix):
            self.solver.factorize(sparse_matrix.matrix)
        else:
            self._type_assert(sparse_matrix)

    def solve(self, b):  # pylint: disable=R1710
        """Computes the solution of the linear systems.
        Args:
            b (numpy.array or Field): The right-hand side of the linear systems.

        Returns:
            numpy.array: The solution of linear systems.
        """
        if isinstance(b, Field):
            return self.solver.solve(b.to_numpy())
        if isinstance(b, np.ndarray):
            return self.solver.solve(b)
        raise TaichiRuntimeError(
            f"The parameter type: {type(b)} is not supported in linear solvers for now."
        )

    def solve_cu(self, sparse_matrix, b, x):
        if isinstance(sparse_matrix, SparseMatrix) and isinstance(
                b, Ndarray) and isinstance(x, Ndarray):
            self.solver.solve_cu(get_runtime().prog, sparse_matrix.matrix,
                                 b.arr, x.arr)
        else:
            raise TaichiRuntimeError(
                f"The parameter type: {type(sparse_matrix)}, {type(b)} and {type(x)} is not supported in linear solvers for now."
            )

    def info(self):
        """Check if the linear systems are solved successfully.

        Returns:
            bool: True if the solving process succeeded, False otherwise.
        """
        return self.solver.info()
