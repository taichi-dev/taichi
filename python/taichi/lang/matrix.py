from . import expr
from . import impl
import copy
import numbers
import numpy as np
from .util import to_numpy_type, to_pytorch_type
from .common_ops import TaichiOperations


def broadcast_if_scalar(func):
    def broadcasted(self, other, *args, **kwargs):
        if isinstance(other, expr.Expr) or isinstance(other, numbers.Number):
            other = self.broadcast(expr.Expr(other))
        return func(self, other, *args, **kwargs)

    return broadcasted


class Matrix(TaichiOperations):
    is_taichi_class = True

    def __init__(self,
                 n=1,
                 m=1,
                 dt=None,
                 empty=False,
                 shape=None,
                 layout=None,
                 needs_grad=False,
                 keep_raw=False,
                 rows=None,
                 cols=None):
        self.grad = None
        if rows is not None or cols is not None:
            if rows is not None and cols is not None:
                raise Exception("cannot specify both rows and columns")
            if cols is not None:
                rows = cols
            self.n = len(rows)
            if isinstance(rows[0], Matrix):
                for row in rows:
                    assert row.m == 1, "inputs must be vectors, ie. size n by 1"
                    assert row.n == rows[
                        0].n, "input vectors must be the same shape"
                self.m = rows[0].n
                # l-value copy:
                self.entries = [row(i) for row in rows for i in range(row.n)]
            elif isinstance(rows[0], list):
                for row in rows:
                    assert len(row) == len(
                        rows[0]), "input lists must be the same shape"
                self.m = len(rows[0])
                # l-value copy:
                self.entries = [x for row in rows for x in row]
            else:
                raise Exception(
                    "rows/cols must be list of lists or lists of vectors")
            if cols is not None:
                t = self.transposed(self)
                self.n = t.n
                self.m = t.m
                self.entries = t.entries
        elif isinstance(n, list):
            if n == []:
                mat = []
            elif not isinstance(n[0], list):
                if impl.get_runtime().inside_kernel:
                    # wrap potential constants with Expr
                    if keep_raw:
                        mat = [list([x]) for x in n]
                    else:
                        mat = [list([expr.Expr(x)]) for x in n]
                else:
                    mat = [[x] for x in n]
            elif isinstance(n[0], Matrix):
                raise Exception(
                    'cols/rows required when using list of vectors')
            else:
                mat = n
            self.n = len(mat)
            if len(mat) > 0:
                self.m = len(mat[0])
            else:
                self.m = 1
            self.entries = [x for row in mat for x in row]
        else:
            self.entries = []
            self.n = n
            self.m = m
            self.dt = dt
            if empty:
                self.entries = [None] * n * m
            else:
                if dt is None:
                    for i in range(n * m):
                        self.entries.append(impl.expr_init(None))
                else:
                    assert not impl.inside_kernel()
                    for i in range(n * m):
                        self.entries.append(impl.var(dt))
                    self.grad = self.make_grad()

        if layout is not None:
            assert shape is not None, 'layout is useless without shape'
        if shape is not None:
            if isinstance(shape, numbers.Number):
                shape = (shape, )
            import taichi as ti
            if layout is None:
                layout = ti.AOS

            dim = len(shape)
            if layout.soa:
                for i, e in enumerate(self.entries):
                    ti.root.dense(ti.index_nd(dim), shape).place(e)
                    if needs_grad:
                        ti.root.dense(ti.index_nd(dim), shape).place(e.grad)
            else:
                var_list = []
                for i, e in enumerate(self.entries):
                    var_list.append(e)
                if needs_grad:
                    for i, e in enumerate(self.entries):
                        var_list.append(e.grad)
                ti.root.dense(ti.index_nd(dim), shape).place(*tuple(var_list))

    def is_global(self):
        results = [False for _ in self.entries]
        for i, e in enumerate(self.entries):
            if isinstance(e, expr.Expr):
                if e.ptr.is_global_var():
                    results[i] = True
            assert results[i] == results[
                0], "Matrices with mixed global/local entries are not allowed"
        return results[0]

    def assign(self, other):
        if isinstance(other, expr.Expr):
            raise Exception('Cannot assign scalar expr to Matrix/Vector.')
        if not isinstance(other, Matrix):
            other = Matrix(other)
        assert other.n == self.n and other.m == self.m
        for i in range(self.n * self.m):
            self.entries[i].assign(other.entries[i])

    def element_wise_binary(self, foo, other):
        ret = Matrix(self.n, self.m)
        if isinstance(other, Matrix):
            assert self.m == other.m and self.n == other.n
            for i in range(self.n * self.m):
                ret.entries[i] = foo(self.entries[i], other.entries[i])
        else:  # assumed to be scalar
            other = expr.Expr(other)
            for i in range(self.n * self.m):
                ret.entries[i] = foo(self.entries[i], other)
        return ret

    def element_wise_unary(self, foo):
        ret = Matrix(self.n, self.m)
        for i in range(self.n * self.m):
            ret.entries[i] = foo(self.entries[i])
        return ret

    def __matmul__(self, other):
        assert self.m == other.n
        ret = Matrix(self.n, other.m)
        for i in range(self.n):
            for j in range(other.m):
                ret(i, j).assign(self(i, 0) * other(0, j))
                for k in range(1, other.n):
                    ret(i, j).assign(ret(i, j) + self(i, k) * other(k, j))
        return ret

    def broadcast(self, scalar):
        ret = Matrix(self.n, self.m, empty=True)
        for i in range(self.n * self.m):
            ret.entries[i] = scalar
        return ret

    def linearize_entry_id(self, *args):
        assert 1 <= len(args) <= 2
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        if len(args) == 1:
            args = args + (0, )
        assert 0 <= args[0] < self.n
        assert 0 <= args[1] < self.m
        return args[0] * self.m + args[1]

    def __call__(self, *args, **kwargs):
        assert kwargs == {}
        return self.entries[self.linearize_entry_id(*args)]

    def get_tensor_members(self):
        return self.entries

    def get_entry(self, *args, **kwargs):
        assert kwargs == {}
        return self.entries[self.linearize_entry_id(*args)]

    def set_entry(self, i, j, e):
        self.entries[self.linearize_entry_id(i, j)] = e

    def place(self, snode):
        for e in self.entries:
            snode.place(e)

    def subscript(self, *indices):
        if self.is_global():
            ret = Matrix(self.n, self.m, empty=True)
            for i, e in enumerate(self.entries):
                ret.entries[i] = impl.subscript(e, *indices)
            return ret
        else:
            assert len(indices) in [1, 2]
            i = indices[0]
            if len(indices) >= 2:
                j = indices[1]
            else:
                j = 0
            return self(i, j)

    class Proxy:
        def __init__(self, mat, index):
            self.mat = mat
            self.index = index

        def __getitem__(self, item):
            if not isinstance(item, list):
                item = [item]
            return self.mat(*item)[self.index]

        def __setitem__(self, key, value):
            if not isinstance(key, list):
                key = [key]
            self.mat(*key)[self.index] = value

    # host access
    def __getitem__(self, index):
        return Matrix.Proxy(self, index)
        ret = [[] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.m):
                ret[i].append(self(i, j)[index])
        return ret

    # host access
    def __setitem__(self, index, item):
        if not isinstance(item[0], list):
            item = [[i] for i in item]
        for i in range(self.n):
            for j in range(self.m):
                self(i, j)[index] = item[i][j]

    def copy(self):
        ret = Matrix(self.n, self.m)
        ret.entries = copy.copy(self.entries)
        return ret

    def variable(self):
        ret = self.copy()
        ret.entries = [impl.expr_init(e) for e in ret.entries]
        return ret

    def cast(self, dt):
        ret = self.copy()
        if type(dt) is type and issubclass(dt, numbers.Number):
            import taichi as ti
            if dt is float:
                dt = ti.get_runtime().default_fp
            elif dt is int:
                dt = ti.get_runtime().default_ip
            else:
                assert False
        for i in range(len(self.entries)):
            ret.entries[i] = impl.cast(ret.entries[i], dt)
        return ret

    def abs(self):
        ret = self.copy()
        for i in range(len(self.entries)):
            ret.entries[i] = impl.abs(ret.entries[i])
        return ret

    def trace(self):
        assert self.n == self.m
        sum = expr.Expr(self(0, 0))
        for i in range(1, self.n):
            sum = sum + self(i, i)
        return sum

    def inverse(self):
        assert self.n == self.m, 'Only square matrices are invertible'
        if self.n == 1:
            return Matrix([1 / self(0, 0)])
        elif self.n == 2:
            inv_det = impl.expr_init(1.0 / self.determinant(self))
            # Discussion: https://github.com/taichi-dev/taichi/pull/943#issuecomment-626344323
            return inv_det * Matrix([[self(1, 1), -self(0, 1)],
                                     [-self(1, 0), self(0, 0)]]).variable()
        elif self.n == 3:
            n = 3
            import taichi as ti
            inv_determinant = ti.expr_init(1.0 / ti.determinant(self))
            entries = [[0] * n for _ in range(n)]

            def E(x, y):
                return self(x % n, y % n)

            for i in range(n):
                for j in range(n):
                    entries[j][i] = ti.expr_init(
                        inv_determinant * (E(i + 1, j + 1) * E(i + 2, j + 2) -
                                           E(i + 2, j + 1) * E(i + 1, j + 2)))
            return Matrix(entries)
        elif self.n == 4:
            n = 4
            import taichi as ti
            inv_determinant = ti.expr_init(1.0 / ti.determinant(self))
            entries = [[0] * n for _ in range(n)]

            def E(x, y):
                return self(x % n, y % n)

            for i in range(n):
                for j in range(n):
                    entries[j][i] = ti.expr_init(
                        inv_determinant * (-1)**(i + j) *
                        ((E(i + 1, j + 1) *
                          (E(i + 2, j + 2) * E(i + 3, j + 3) -
                           E(i + 3, j + 2) * E(i + 2, j + 3)) -
                          E(i + 2, j + 1) *
                          (E(i + 1, j + 2) * E(i + 3, j + 3) -
                           E(i + 3, j + 2) * E(i + 1, j + 3)) +
                          E(i + 3, j + 1) *
                          (E(i + 1, j + 2) * E(i + 2, j + 3) -
                           E(i + 2, j + 2) * E(i + 1, j + 3)))))
            return Matrix(entries)
        else:
            raise Exception(
                "Inversions of matrices with sizes >= 5 are not supported")

    def inversed(self):
        return self.inverse()

    @staticmethod
    def normalized(a, eps=0):
        assert a.m == 1
        invlen = 1.0 / (Matrix.norm(a) + eps)
        return invlen * a

    @staticmethod
    def floor(a):
        b = Matrix(a.n, a.m)
        for i in range(len(a.entries)):
            b.entries[i] = impl.floor(a.entries[i])
        return b

    @staticmethod
    def outer_product(a, b):
        assert a.m == 1
        assert b.m == 1
        c = Matrix(a.n, b.n)
        for i in range(a.n):
            for j in range(b.n):
                c(i, j).assign(a(i) * b(j))
        return c

    @staticmethod
    def transposed(a):
        ret = Matrix(a.m, a.n, empty=True)
        for i in range(a.n):
            for j in range(a.m):
                ret.set_entry(j, i, a(i, j))
        return ret

    def T(self):
        return self.transposed(self)

    @staticmethod
    def determinant(a):
        if a.n == 2 and a.m == 2:
            return a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)
        elif a.n == 3 and a.m == 3:
            return a(0, 0) * (a(1, 1) * a(2, 2) - a(2, 1) * a(1, 2)) - a(
                1, 0) * (a(0, 1) * a(2, 2) - a(2, 1) * a(0, 2)) + a(
                    2, 0) * (a(0, 1) * a(1, 2) - a(1, 1) * a(0, 2))
        elif a.n == 4 and a.m == 4:
            import taichi as ti
            n = 4

            def E(x, y):
                return a(x % n, y % n)

            det = ti.expr_init(0.0)
            for i in range(4):
                det = det + (-1.0)**i * (
                    a(i, 0) *
                    (E(i + 1, 1) *
                     (E(i + 2, 2) * E(i + 3, 3) - E(i + 3, 2) * E(i + 2, 3)) -
                     E(i + 2, 1) *
                     (E(i + 1, 2) * E(i + 3, 3) - E(i + 3, 2) * E(i + 1, 3)) +
                     E(i + 3, 1) *
                     (E(i + 1, 2) * E(i + 2, 3) - E(i + 2, 2) * E(i + 1, 3))))
            return det
        else:
            raise Exception(
                "Determinants of matrices with sizes >= 5 are not supported")

    @staticmethod
    def cross(a, b):
        assert a.n == 3 and a.m == 1
        assert b.n == 3 and b.m == 1
        return Matrix([
            a(1) * b(2) - a(2) * b(1),
            a(2) * b(0) - a(0) * b(2),
            a(0) * b(1) - a(1) * b(0),
        ])

    @staticmethod
    def diag(dim, val):
        ret = Matrix(dim, dim)
        for i in range(dim):
            for j in range(dim):
                ret.set_entry(i, j, 0)
        for i in range(dim):
            ret.set_entry(i, i, val)
        return ret

    def loop_range(self):
        return self.entries[0]

    # TODO
    @broadcast_if_scalar
    def augassign(self, other, op):
        if not isinstance(other, Matrix):
            other = Matrix(other)
        assert self.n == other.n and self.m == other.m
        for i in range(len(self.entries)):
            self.entries[i].augassign(other.entries[i], op)

    def atomic_add(self, other):
        assert self.n == other.n and self.m == other.m
        for i in range(len(self.entries)):
            self.entries[i].atomic_add(other.entries[i])

    def make_grad(self):
        ret = Matrix(self.n, self.m, empty=True)
        for i in range(len(ret.entries)):
            ret.entries[i] = self.entries[i].grad
        return ret

    def sum(self):
        ret = self.entries[0]
        for i in range(1, len(self.entries)):
            ret = ret + self.entries[i]
        return ret

    def norm(self, l=2, eps=0):
        assert l == 2
        return impl.sqrt(self.norm_sqr() + eps)

    # TODO: avoid using sqr here since people might consider "sqr" as "square root". New name TBD.
    def norm_sqr(self):
        return (self**2).sum()

    def max(self):
        ret = self.entries[0]
        for i in range(1, len(self.entries)):
            ret = impl.max(ret, self.entries[i])
        return ret

    def min(self):
        ret = self.entries[0]
        for i in range(1, len(self.entries)):
            ret = impl.min(ret, self.entries[i])
        return ret

    def dot(self, other):
        assert self.m == 1 and other.m == 1
        return (self.transposed(self) @ other).subscript(0, 0)

    def fill(self, val):
        if isinstance(val, numbers.Number):
            val = tuple(
                [tuple([val for _ in range(self.m)]) for _ in range(self.n)])
        elif isinstance(val[0], numbers.Number):
            assert self.m == 1
            val = tuple([(v, ) for v in val])
        if isinstance(val, Matrix):
            val_tuple = []
            for i in range(val.n):
                row = []
                for j in range(val.m):
                    row.append(val.get_entry(i, j))
                row = tuple(row)
                val_tuple.append(row)
            val = tuple(val_tuple)
        assert len(val) == self.n
        assert len(val[0]) == self.m
        from .meta import fill_matrix
        fill_matrix(self, val)

    def to_numpy(self, as_vector=False):
        if as_vector:
            assert self.m == 1, "This matrix is not a vector"
            dim_ext = (self.n, )
        else:
            dim_ext = (self.n, self.m)
        ret = np.empty(self.loop_range().shape() + dim_ext,
                       dtype=to_numpy_type(
                           self.loop_range().snode().data_type()))
        from .meta import matrix_to_ext_arr
        matrix_to_ext_arr(self, ret, as_vector)
        import taichi as ti
        ti.sync()
        return ret

    def to_torch(self, as_vector=False, device=None):
        import torch
        if as_vector:
            assert self.m == 1, "This matrix is not a vector"
            dim_ext = (self.n, )
        else:
            dim_ext = (self.n, self.m)
        ret = torch.empty(self.loop_range().shape() + dim_ext,
                          dtype=to_pytorch_type(
                              self.loop_range().snode().data_type()),
                          device=device)
        from .meta import matrix_to_ext_arr
        matrix_to_ext_arr(self, ret, as_vector)
        import taichi as ti
        ti.sync()
        return ret

    def from_numpy(self, ndarray):
        if len(ndarray.shape) == self.loop_range().dim() + 1:
            as_vector = True
            assert self.m == 1, "This matrix is not a vector"
        else:
            as_vector = False
            assert len(ndarray.shape) == self.loop_range().dim() + 2
        from .meta import ext_arr_to_matrix
        ext_arr_to_matrix(ndarray, self, as_vector)
        import taichi as ti
        ti.sync()

    def from_torch(self, torch_tensor):
        return self.from_numpy(torch_tensor.contiguous())

    @staticmethod
    def zero(dt, n, m=1):
        import taichi as ti
        return ti.Matrix([[ti.cast(0, dt) for _ in range(m)]
                          for _ in range(n)])

    @staticmethod
    def one(dt, n):
        import taichi as ti
        return ti.Matrix([[ti.cast(1, dt) for _ in range(n)]
                          for _ in range(n)])

    @staticmethod
    def unit(n, i, dt=None):
        import taichi as ti
        if dt is None:
            dt = ti.get_runtime().default_ip
        assert 0 <= i < n
        return ti.Matrix([ti.cast(int(j == i), dt) for j in range(n)])

    @staticmethod
    def identity(dt, n):
        import taichi as ti
        return ti.Matrix([[ti.cast(int(i == j), dt) for j in range(n)]
                          for i in range(n)])

    @staticmethod
    def rotation2d(alpha):
        import taichi as ti
        return ti.Matrix([[ti.cos(alpha), -ti.sin(alpha)],
                          [ti.sin(alpha), ti.cos(alpha)]])
