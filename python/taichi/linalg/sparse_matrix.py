from functools import reduce

import numpy as np
from taichi._lib import core as _ti_core
from taichi.lang._ndarray import Ndarray, ScalarNdarray
from taichi.lang.exception import TaichiRuntimeError
from taichi.lang.field import Field
from taichi.lang.impl import get_runtime
from taichi.types import f32


class SparseMatrix:
    """Taichi's Sparse Matrix class

    A sparse matrix allows the programmer to solve a large linear system.

    Args:
        n (int): the first dimension of a sparse matrix.
        m (int): the second dimension of a sparse matrix.
        sm (SparseMatrix): another sparse matrix that will be built from.
    """

    def __init__(self, n=None, m=None, sm=None, dtype=f32, storage_format="col_major"):
        self.dtype = dtype
        if sm is None:
            self.n = n
            self.m = m if m else n
            self.matrix = get_runtime().prog.create_sparse_matrix(n, m, dtype, storage_format)
        else:
            self.n = sm.num_rows()
            self.m = sm.num_cols()
            self.matrix = sm

    def __iadd__(self, other):
        """Addition operation for sparse matrix.

        Returns:
            The result sparse matrix of the addition.
        """
        assert (
            self.n == other.n and self.m == other.m
        ), f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
        self.matrix += other.matrix
        return self

    def __add__(self, other):
        """Addition operation for sparse matrix.

        Returns:
            The result sparse matrix of the addition.
        """
        assert (
            self.n == other.n and self.m == other.m
        ), f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
        sm = self.matrix + other.matrix
        return SparseMatrix(sm=sm)

    def __isub__(self, other):
        """Subtraction operation for sparse matrix.

        Returns:
             The result sparse matrix of the subtraction.
        """
        assert (
            self.n == other.n and self.m == other.m
        ), f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
        self.matrix -= other.matrix
        return self

    def __sub__(self, other):
        """Subtraction operation for sparse matrix.

        Returns:
             The result sparse matrix of the subtraction.
        """
        assert (
            self.n == other.n and self.m == other.m
        ), f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
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
            sm = other * self.matrix
            return SparseMatrix(sm=sm)
        if isinstance(other, SparseMatrix):
            assert (
                self.n == other.n and self.m == other.m
            ), f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
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
            sm = self.matrix * other
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
            assert (
                self.m == other.n
            ), f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
            sm = self.matrix.matmul(other.matrix)
            return SparseMatrix(sm=sm)
        if isinstance(other, Field):
            assert (
                self.m == other.shape[0]
            ), f"Dimension mismatch between sparse matrix ({self.n}, {self.m}) and vector ({other.shape})"
            return self.matrix.mat_vec_mul(other.to_numpy())
        if isinstance(other, np.ndarray):
            assert (
                self.m == other.shape[0]
            ), f"Dimension mismatch between sparse matrix ({self.n}, {self.m}) and vector ({other.shape})"
            return self.matrix.mat_vec_mul(other)
        if isinstance(other, Ndarray):
            if self.m != other.shape[0]:
                raise TaichiRuntimeError(
                    f"Dimension mismatch between sparse matrix ({self.n}, {self.m}) and vector ({other.shape})"
                )
            res = ScalarNdarray(dtype=other.dtype, arr_shape=(self.n,))
            self.matrix.spmv(get_runtime().prog, other.arr, res.arr)
            return res
        raise TaichiRuntimeError(
            f"Sparse matrix-matrix/vector multiplication does not support {type(other)} for now. Supported types are SparseMatrix, ti.field, and numpy ndarray."
        )

    def __getitem__(self, indices):
        return self.matrix.get_element(indices[0], indices[1])

    def __setitem__(self, indices, value):
        self.matrix.set_element(indices[0], indices[1], value)

    def __str__(self):
        """Python scope matrix print support."""
        return self.matrix.to_string()

    def __repr__(self):
        return self.matrix.to_string()

    @property
    def shape(self):
        """The shape of the sparse matrix."""
        return (self.n, self.m)

    def build_from_ndarray(self, ndarray):
        """Build the sparse matrix from a ndarray.

        Args:
            ndarray (Union[ti.ndarray, ti.Vector.ndarray, ti.Matrix.ndarray]): the ndarray to build the sparse matrix from.

        Raises:
            TaichiRuntimeError: If the input is not a ndarray or the length is not divisible by 3.

        Example::
            >>> N = 5
            >>> triplets = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=10, layout=ti.Layout.AOS)
            >>> @ti.kernel
            >>> def fill(triplets: ti.types.ndarray()):
            >>>     for i in range(N):
            >>>        triplets[i] = ti.Vector([i, (i + 1) % N, i+1], dt=ti.f32)
            >>> fill(triplets)
            >>> A = ti.linalg.SparseMatrix(n=N, m=N, dtype=ti.f32)
            >>> A.build_from_ndarray(triplets)
            >>> print(A)
            [0, 1, 0, 0, 0]
            [0, 0, 2, 0, 0]
            [0, 0, 0, 3, 0]
            [0, 0, 0, 0, 4]
            [5, 0, 0, 0, 0]
        """
        if isinstance(ndarray, Ndarray):
            num_scalars = reduce(lambda x, y: x * y, ndarray.shape + ndarray.element_shape)
            if num_scalars % 3 != 0:
                raise TaichiRuntimeError("The number of ndarray elements must have a length that is divisible by 3.")
            get_runtime().prog.make_sparse_matrix_from_ndarray(self.matrix, ndarray.arr)
        else:
            raise TaichiRuntimeError(
                "Sparse matrix only supports building from [ti.ndarray, ti.Vector.ndarray, ti.Matrix.ndarray]"
            )

    def mmwrite(self, filename):
        """Writes the sparse matrix to Matrix Market file-like target.

        Args:
            filename (str): the file name to write the sparse matrix to.
        """
        self.matrix.mmwrite(filename)


class SparseMatrixBuilder:
    """A python wrap around sparse matrix builder.

    Use this builder to fill the sparse matrix.

    Args:
        num_rows (int): the first dimension of a sparse matrix.
        num_cols (int): the second dimension of a sparse matrix.
        max_num_triplets (int): the maximum number of triplets.
        dtype (ti.dtype): the data type of the sparse matrix.
        storage_format (str): the storage format of the sparse matrix.
    """

    def __init__(
        self,
        num_rows=None,
        num_cols=None,
        max_num_triplets=0,
        dtype=f32,
        storage_format="col_major",
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols if num_cols else num_rows
        self.dtype = dtype
        if num_rows is not None:
            taichi_arch = get_runtime().prog.config().arch
            if taichi_arch in [
                _ti_core.Arch.x64,
                _ti_core.Arch.arm64,
                _ti_core.Arch.cuda,
            ]:
                self.ptr = _ti_core.SparseMatrixBuilder(
                    num_rows,
                    num_cols,
                    max_num_triplets,
                    dtype,
                    storage_format,
                )
                self.ptr.create_ndarray(get_runtime().prog)
            else:
                raise TaichiRuntimeError("SparseMatrix only supports CPU and CUDA for now.")

    def _get_addr(self):
        """Get the address of the sparse matrix"""
        return self.ptr.get_addr()

    def _get_ndarray_addr(self):
        """Get the address of the ndarray"""
        return self.ptr.get_ndarray_data_ptr()

    def print_triplets(self):
        """Print the triplets stored in the builder"""
        taichi_arch = get_runtime().prog.config().arch
        if taichi_arch in [_ti_core.Arch.x64, _ti_core.Arch.arm64]:
            self.ptr.print_triplets_eigen()
        elif taichi_arch == _ti_core.Arch.cuda:
            self.ptr.print_triplets_cuda()

    def build(self, dtype=f32, _format="CSR"):
        """Create a sparse matrix using the triplets"""
        taichi_arch = get_runtime().prog.config().arch
        if taichi_arch in [_ti_core.Arch.x64, _ti_core.Arch.arm64]:
            sm = self.ptr.build()
            return SparseMatrix(sm=sm, dtype=self.dtype)
        if taichi_arch == _ti_core.Arch.cuda:
            if self.dtype != f32:
                raise TaichiRuntimeError("CUDA sparse matrix only supports f32.")
            sm = self.ptr.build_cuda()
            return SparseMatrix(sm=sm, dtype=self.dtype)
        raise TaichiRuntimeError("Sparse matrix only supports CPU and CUDA backends.")

    def __del__(self):
        if get_runtime() is not None and get_runtime().prog is not None:
            self.ptr.delete_ndarray(get_runtime().prog)


__all__ = ["SparseMatrix", "SparseMatrixBuilder"]
