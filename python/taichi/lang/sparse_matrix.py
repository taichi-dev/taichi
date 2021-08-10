class SparseMatrix:
    def __init__(self, n, m=None, max_num_triplets=0):
        self.n = n
        self.m = m if m else n
        from taichi.lang.impl import get_runtime
        self.ptr = get_runtime().create_sparse_matrix(n, m, max_num_triplets)
        print(f"Creating a sparse matrix of size ({n}, {m})...")

    def get_addr(self):
        return self.ptr.get_addr()

    def print_triplets(self):
        self.ptr.print_triplets()

    def print(self):
        self.ptr.print()

    def build(self):
        self.ptr.build()

class SparseMatrixEntry:
    def __init__(self, ptr, i, j):
        self.ptr = ptr
        self.i = i
        self.j = j

    def __iadd__(self, other):
        from taichi.lang.impl import call_internal
        call_internal("insert_triplet", self.ptr, self.i, self.j, other)

class SparseMatrixProxy:
    def __init__(self, ptr):
        self.ptr = ptr

    def insert(self, i, j, val):
        from taichi.lang.impl import call_internal
        call_internal("insert_triplet", self.ptr, i, j, val)

