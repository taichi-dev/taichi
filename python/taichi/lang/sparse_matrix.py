class SparseMatrix:
    def __init__(self, n=None, m=None, sm=None):
        if sm is None:
            self.n = n
            self.m = m if m else n
            from taichi.core.util import ti_core as _ti_core
            self.matrix = _ti_core.create_sparse_matrix(n, m)
        else:
            self.n = sm.num_rows()
            self.m = sm.num_cols()
            self.matrix = sm

    def __add__(self, other):
        assert self.n == other.n and self.m == other.m, f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
        sm = self.matrix + other.matrix
        return SparseMatrix(sm=sm)

    def __sub__(self, other):
        assert self.n == other.n and self.m == other.m, f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
        sm = self.matrix - other.matrix
        return SparseMatrix(sm=sm)

    def __mul__(self, other):
        if isinstance(other, float):
            sm = self.matrix * other
            return SparseMatrix(sm=sm)
        elif isinstance(other, SparseMatrix):
            assert self.n == other.n and self.m == other.m, f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
            sm = self.matrix * other.matrix
            return SparseMatrix(sm=sm)

    def __rmul__(self, other):
        if isinstance(other, float):
            sm = other * self.matrix
            return SparseMatrix(sm=sm)


    def transpose(self):
        sm = self.matrix.transpose()
        return SparseMatrix(sm=sm)

    def __matmul__(self, other):
        sm = self.matrix.matmul(other.matrix)
        return SparseMatrix(sm=sm)

    def __getitem__(self, item):
        return self.matrix.get_coeff(item[0], item[1])

    def __str__(self):
        return self.matrix.to_string()

    def __repr__(self):
        return self.matrix.to_string()


class SparseMatrixBuilder:
    def __init__(self, n=None, m=None, max_num_triplets=0):
        self.n = n
        self.m = m if m else n
        if n is not None and m is not None:
            from taichi.core.util import ti_core as _ti_core
            self.ptr = _ti_core.create_sparse_matrix_builder(
                n, m, max_num_triplets)

    def get_addr(self):
        return self.ptr.get_addr()

    def print_triplets(self):
        self.ptr.print_triplets()

    def build(self):
        sm = self.ptr.build()
        return SparseMatrix(sm=sm)



