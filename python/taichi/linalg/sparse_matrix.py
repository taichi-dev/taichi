import numpy as np
from taichi._lib import core as _ti_core
from taichi.lang.field import Field
from taichi.types.primitive_types import f32


class SparseMatrix:
    """Taichi's Sparse Matrix class

    A sparse matrix allows the programmer to solve a large linear system.

    Args:
        n (int): the first dimension of a sparse matrix.
        m (int): the second dimension of a sparse matrix.
        sm (SparseMatrix): another sparse matrix that will be built from.
    """
    def __init__(self, n=None, m=None, sm=None, dtype=f32):
        if sm is None:
            self.n = n
            self.m = m if m else n
            self.matrix = _ti_core.create_sparse_matrix(n, m)
        else:
            self.n = sm.num_rows()
            self.m = sm.num_cols()
            self.matrix = sm

    def __add__(self, other):
        """Addition operation for sparse matrix.

        Returns:
            The result sparse matrix of the addition.
        """
        assert self.n == other.n and self.m == other.m, f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
        sm = self.matrix + other.matrix
        return SparseMatrix(sm=sm)

    def __sub__(self, other):
        """Subtraction operation for sparse matrix.

        Returns:
             The result sparse matrix of the subtraction.
        """
        assert self.n == other.n and self.m == other.m, f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
        sm = self.matrix - other.matrix
        return SparseMatrix(sm=sm)

    def __mul__(self, other):
        """Sparse matrix's multiplication against real numbers or the hadamard product against another matrix

        Args:
            other (float or SparseMatrix): the other operand of multiplication.
        Returns:
            The result of multiplication.
        """
        if isinstance(other, float):
            sm = self.matrix * other
            return SparseMatrix(sm=sm)
        if isinstance(other, SparseMatrix):
            assert self.n == other.n and self.m == other.m, f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
            sm = self.matrix * other.matrix
            return SparseMatrix(sm=sm)

        return None

    def __rmul__(self, other):
        """Right scalar multiplication for sparse matrix.

        Args:
            other (float): the other operand of scalar multiplication.
        Returns:
            The result of multiplication.
        """
        if isinstance(other, float):
            sm = other * self.matrix
            return SparseMatrix(sm=sm)

        return None

    def transpose(self):
        """Sparse Matrix transpose.

        Returns:
            The transposed sparse mastrix.
        """
        sm = self.matrix.transpose()
        return SparseMatrix(sm=sm)

    def __matmul__(self, other):
        """Matrix multiplication.

        Args:
            other (SparseMatrix, Field, or numpy.array): the other sparse matrix of the multiplication.
        Returns:
            The result of matrix multiplication.
        """
        if isinstance(other, SparseMatrix):
            assert self.m == other.n, f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
            sm = self.matrix.matmul(other.matrix)
            return SparseMatrix(sm=sm)
        if isinstance(other, Field):
            assert self.m == other.shape[
                0], f"Dimension mismatch between sparse matrix ({self.n}, {self.m}) and vector ({other.shape})"
            return self.matrix.mat_vec_mul(other.to_numpy())
        if isinstance(other, np.ndarray):
            assert self.m == other.shape[
                0], f"Dimension mismatch between sparse matrix ({self.n}, {self.m}) and vector ({other.shape})"
            return self.matrix.mat_vec_mul(other)
        assert False, f"Sparse matrix-matrix/vector multiplication does not support {type(other)} for now. Supported types are SparseMatrix, ti.field, and numpy.ndarray."

    def __getitem__(self, indices):
        return self.matrix.get_element(indices[0], indices[1])

    def __setitem__(self, indices, value):
        self.matrix.set_element(indices[0], indices[1], value)

    def __str__(self):
        """Python scope matrix print support."""
        return self.matrix.to_string()

    def __repr__(self):
        return self.matrix.to_string()

    def shape(self):
        """The shape of the sparse matrix."""
        return (self.n, self.m)


class SparseMatrixBuilder:
    """A python wrap around sparse matrix builder.

    Use this builder to fill the sparse matrix.

    Args:
        num_rows (int): the first dimension of a sparse matrix.
        num_cols (int): the second dimension of a sparse matrix.
        max_num_triplets (int): the maximum number of triplets.
    """
    def __init__(self,
                 num_rows=None,
                 num_cols=None,
                 max_num_triplets=0,
                 dtype=f32):
        self.num_rows = num_rows
        self.num_cols = num_cols if num_cols else num_rows
        if num_rows is not None:
            self.ptr = _ti_core.create_sparse_matrix_builder(
                num_rows, num_cols, max_num_triplets)

    def get_addr(self):
        """Get the address of the sparse matrix"""
        return self.ptr.get_addr()

    def print_triplets(self):
        """Print the triplets stored in the builder"""
        self.ptr.print_triplets()

    def build(self, dtype=f32, _format='CSR'):
        """Create a sparse matrix using the triplets"""
        sm = self.ptr.build()
        return SparseMatrix(sm=sm)


sparse_matrix_builder = SparseMatrixBuilder
# Alias for :class:`SparseMatrixBuilder`
