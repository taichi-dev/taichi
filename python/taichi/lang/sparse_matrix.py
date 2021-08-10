
class SparseMatrix:
    def __init__(self, n, m=None):
        self.n = n
        self.m = m if m else n
        self.ptr = 123431232141234123
        print(f"Creating a sparse matrix of size ({n}, {m})...")

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

