class SparseMatrix:
    def __init__(self, n=None, m=None, sm=None):
        if sm is None:
            self.n = n
            self.m = m if m else n
            from taichi.lang.impl import get_runtime
            self.matrix = get_runtime().create_sparse_matrix(n, m)
        else:
            self.n = sm.num_rows()
            self.m = sm.num_cols()
            self.matrix = sm

    def print(self):
        self.matrix.print()

    def solve(self, b):
        self.matrix.solve(b)

    def __add__(self, other):
        assert self.n == other.n and self.m == other.m, f"Dimension mismatch between sparse matrices ({self.n}, {self.m}), ({other.n}, {other.m})"
        sm = self.matrix + other.matrix
        return SparseMatrix(sm=sm)

    def __sub__(self, other):
        assert self.n == other.n and self.m == other.m, f"Dimension mismatch between sparse matrices ({self.n}, {self.m}), ({other.n}, {other.m})"
        sm = self.matrix - other.matrix
        return SparseMatrix(sm=sm)

    def __mul__(self, other):
        if isinstance(other, float):
            sm = other * self.matrix
            return SparseMatrix(sm=sm)
        elif isinstance(other, SparseMatrix):
            assert self.n == other.n and self.m == other.m, f"Dimension mismatch between sparse matrices ({self.n}, {self.m}), ({other.n}, {other.m})"
            sm = self.matrix * other.matrix
            return SparseMatrix(sm=sm)

    def transpose(self):
        sm = self.matrix.transpose()
        return SparseMatrix(sm=sm)

    def __matmul__(self, other):
        sm = self.matrix.matmult(other.matrix)
        return SparseMatrix(sm=sm)

    def get_ele(self, i, j):
        return self.matrix.get_coeff(i, j)


class SparseMatrixBuilder:
    def __init__(self, n, m=None, max_num_triplets=0):
        self.n = n
        self.m = m if m else n
        from taichi.lang.impl import get_runtime
        self.ptr = get_runtime().create_sparse_matrix_builder(
            n, m, max_num_triplets)
        print(f"Creating a sparse matrix of size ({n}, {m})...")

    def get_addr(self):
        return self.ptr.get_addr()

    def print_triplets(self):
        self.ptr.print_triplets()

    def print(self):
        self.ptr.print()

    def build(self, other):
        self.ptr.build(other.matrix)


class SparseMatrixEntry:
    def __init__(self, ptr, i, j):
        self.ptr = ptr
        self.i = i
        self.j = j

    def augassign(self, value, op):
        assert op == 'Add' or op == 'Sub', f"Only operation '+=' and '-=' is support right now"
        from taichi.lang.impl import call_internal
        if op == 'Add':
            call_internal("insert_triplet", self.ptr, self.i, self.j, value)
        elif op == 'Sub':
            call_internal("insert_triplet", self.ptr, self.i, self.j, -value)


class SparseMatrixProxy:
    is_taichi_class = True

    def __init__(self, ptr):
        self.ptr = ptr

    def subscript(self, i, j):
        return SparseMatrixEntry(self.ptr, i, j)
