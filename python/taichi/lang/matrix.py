import copy
import numbers
from collections.abc import Iterable

import numpy as np
from taichi.lang import expr, impl
from taichi.lang import kernel_impl as kern_mod
from taichi.lang import ops as ops_mod
from taichi.lang.common_ops import TaichiOperations
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.util import (in_python_scope, is_taichi_class, python_scope,
                              taichi_scope, to_numpy_type, to_pytorch_type)
from taichi.misc.util import deprecated, warning

import taichi as ti


class Matrix(TaichiOperations):
    is_taichi_class = True

    # TODO(archibate): move the last two line to **kwargs,
    # since they're not commonly used as positional args.
    def __init__(self,
                 n=1,
                 m=1,
                 dt=None,
                 shape=None,
                 offset=None,
                 empty=False,
                 layout=None,
                 needs_grad=False,
                 keep_raw=False,
                 rows=None,
                 cols=None):
        self.grad = None

        # construct from rows or cols (deprecated)
        if rows is not None or cols is not None:
            warning(
                f"ti.Matrix(rows=[...]) or ti.Matrix(cols=[...]) is deprecated, use ti.Matrix.rows([...]) or ti.Matrix.cols([...]) instead.",
                DeprecationWarning,
                stacklevel=2)
            if rows is not None and cols is not None:
                raise Exception("cannot specify both rows and columns")
            self.dt = dt
            mat = Matrix.cols(cols) if cols is not None else Matrix.rows(rows)
            self.n = mat.n
            self.m = mat.m
            self.entries = mat.entries
            return

        elif empty == True:
            warning(
                f"ti.Matrix(n, m, empty=True) is deprecated, use ti.Matrix.empty(n, m) instead",
                DeprecationWarning,
                stacklevel=2)
            self.dt = dt
            self.entries = [[None] * m for _ in range(n)]
            return

        elif isinstance(n, (list, tuple, np.ndarray)):
            if len(n) == 0:
                mat = []
            elif isinstance(n[0], Matrix):
                raise Exception(
                    'cols/rows required when using list of vectors')
            elif not isinstance(n[0], Iterable):
                if impl.inside_kernel():
                    # wrap potential constants with Expr
                    if keep_raw:
                        mat = [list([x]) for x in n]
                    else:
                        mat = [list([expr.Expr(x)]) for x in n]
                else:
                    mat = [[x] for x in n]
            else:
                mat = [list(r) for r in n]
            self.n = len(mat)
            if len(mat) > 0:
                self.m = len(mat[0])
            else:
                self.m = 1
            self.entries = [x for row in mat for x in row]

        else:
            if dt is None:
                # create a local matrix with specific (n, m)
                self.entries = [impl.expr_init(None) for i in range(n * m)]
                self.n = n
                self.m = m
            else:
                # construct global matrix (deprecated)
                warning(
                    "Declaring global matrices using `ti.Matrix(n, m, dt, shape)` is deprecated, "
                    "use `ti.Matrix.field(n, m, dtype, shape)` instead",
                    DeprecationWarning,
                    stacklevel=2)
                mat = Matrix.field(n=n,
                                   m=m,
                                   dtype=dt,
                                   shape=shape,
                                   offset=offset,
                                   needs_grad=needs_grad,
                                   layout=layout)
                self.n = mat.n
                self.m = mat.m
                self.entries = mat.entries
                self.grad = mat.grad

        if self.n * self.m > 32:
            warning(
                f'Taichi matrices/vectors with {self.n}x{self.m} > 32 entries are not suggested.'
                ' Matrices/vectors will be automatically unrolled at compile-time for performance.'
                ' So the compilation time could be extremely long if the matrix size is too big.'
                ' You may use a field to store a large matrix like this, e.g.:\n'
                f'    x = ti.field(ti.f32, ({self.n}, {self.m})).\n'
                ' See https://taichi.readthedocs.io/en/stable/tensor_matrix.html#matrix-size'
                ' for more details.',
                UserWarning,
                stacklevel=2)

    def is_global(self):
        results = [False for _ in self.entries]
        for i, e in enumerate(self.entries):
            if isinstance(e, expr.Expr):
                if e.is_global():
                    results[i] = True
            assert results[i] == results[0], \
                "Matrices with mixed global/local entries are not allowed"
        return results[0]

    def element_wise_binary(self, foo, other):
        _taichi_skip_traceback = 1
        ret = self.empty_copy()
        if isinstance(other, (list, tuple)):
            other = Matrix(other)
        if isinstance(other, Matrix):
            assert self.m == other.m and self.n == other.n, f"Dimension mismatch between shapes ({self.n}, {self.m}), ({other.n}, {other.m})"
            for i in range(self.n * self.m):
                ret.entries[i] = foo(self.entries[i], other.entries[i])
        else:  # assumed to be scalar
            for i in range(self.n * self.m):
                ret.entries[i] = foo(self.entries[i], other)
        return ret

    def broadcast_copy(self, other):
        if isinstance(other, (list, tuple)):
            other = Matrix(other)
        if not isinstance(other, Matrix):
            ret = self.empty_copy()
            ret.entries = [other for _ in ret.entries]
            other = ret
        assert self.m == other.m and self.n == other.n, f"Dimension mismatch between shapes ({self.n}, {self.m}), ({other.n}, {other.m})"
        return other

    def element_wise_ternary(self, foo, other, extra):
        ret = self.empty_copy()
        other = self.broadcast_copy(other)
        extra = self.broadcast_copy(extra)
        for i in range(self.n * self.m):
            ret.entries[i] = foo(self.entries[i], other.entries[i],
                                 extra.entries[i])
        return ret

    def element_wise_writeback_binary(self, foo, other):
        ret = self.empty_copy()
        if isinstance(other, (list, tuple)):
            other = Matrix(other)
        if is_taichi_class(other):
            other = other.variable()
        if foo.__name__ == 'assign' and not isinstance(other, Matrix):
            raise TaichiSyntaxError(
                'cannot assign scalar expr to '
                f'taichi class {type(self)}, maybe you want to use `a.fill(b)` instead?'
            )
        if isinstance(other, Matrix):
            assert self.m == other.m and self.n == other.n, f"Dimension mismatch between shapes ({self.n}, {self.m}), ({other.n}, {other.m})"
            for i in range(self.n * self.m):
                ret.entries[i] = foo(self.entries[i], other.entries[i])
        else:  # assumed to be scalar
            for i in range(self.n * self.m):
                ret.entries[i] = foo(self.entries[i], other)
        return ret

    def element_wise_unary(self, foo):
        _taichi_skip_traceback = 1
        ret = self.empty_copy()
        for i in range(self.n * self.m):
            ret.entries[i] = foo(self.entries[i])
        return ret

    def __matmul__(self, other):
        _taichi_skip_traceback = 1
        assert isinstance(other, Matrix), "rhs of `@` is not a matrix / vector"
        assert self.m == other.n, f"Dimension mismatch between shapes ({self.n}, {self.m}), ({other.n}, {other.m})"
        del _taichi_skip_traceback
        ret = Matrix.new(self.n, other.m)
        for i in range(self.n):
            for j in range(other.m):
                acc = self(i, 0) * other(0, j)
                for k in range(1, other.n):
                    acc = acc + self(i, k) * other(k, j)
                ret.set_entry(i, j, acc)
        return ret

    def linearize_entry_id(self, *args):
        assert 1 <= len(args) <= 2
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        if len(args) == 1:
            args = args + (0, )
        _taichi_skip_traceback = 1
        # TODO(#1004): See if it's possible to support indexing at runtime
        for i, a in enumerate(args):
            if not isinstance(a, int):
                raise TaichiSyntaxError(
                    f'The {i}-th index of a Matrix/Vector must be a compile-time constant '
                    f'integer, got {type(a)}.\n'
                    'This is because matrix operations will be **unrolled** at compile-time '
                    'for performance reason.\n'
                    'If you want to *iterate through matrix elements*, use a static range:\n'
                    '  for i in ti.static(range(3)):\n'
                    '    print(i, "-th component is", vec[i])\n'
                    'See https://taichi.readthedocs.io/en/stable/meta.html#when-to-use-for-loops-with-ti-static for more details.'
                )
        assert 0 <= args[0] < self.n, \
                f"The 0-th matrix index is out of range: 0 <= {args[0]} < {self.n}"
        assert 0 <= args[1] < self.m, \
                f"The 1-th matrix index is out of range: 0 <= {args[1]} < {self.m}"
        return args[0] * self.m + args[1]

    def __call__(self, *args, **kwargs):
        _taichi_skip_traceback = 1
        assert kwargs == {}
        return self.entries[self.linearize_entry_id(*args)]

    def get_field_members(self):
        return self.entries

    @deprecated('x.get_tensor_members()', 'x.get_field_members()')
    def get_tensor_members(self):
        return self.get_field_members()

    def get_entry(self, *args, **kwargs):
        assert kwargs == {}
        return self.entries[self.linearize_entry_id(*args)]

    def set_entry(self, i, j, e):
        idx = self.linearize_entry_id(i, j)
        if impl.inside_kernel():
            self.entries[idx].assign(e)
        else:
            self.entries[idx] = e

    def place(self, snode):
        for e in self.entries:
            snode.place(e)

    @taichi_scope
    def subscript(self, *indices):
        _taichi_skip_traceback = 1
        if self.is_global():
            ret = self.empty_copy()
            for i, e in enumerate(self.entries):
                ret.entries[i] = impl.subscript(e, *indices)
            return ret
        else:
            assert len(indices) in [1, 2]
            i = indices[0]
            j = 0 if len(indices) == 1 else indices[1]
            return self(i, j)

    @property
    def x(self):
        _taichi_skip_traceback = 1
        if impl.inside_kernel():
            return self.subscript(0)
        else:
            return self[0]

    @property
    def y(self):
        _taichi_skip_traceback = 1
        if impl.inside_kernel():
            return self.subscript(1)
        else:
            return self[1]

    @property
    def z(self):
        _taichi_skip_traceback = 1
        if impl.inside_kernel():
            return self.subscript(2)
        else:
            return self[2]

    @property
    def w(self):
        _taichi_skip_traceback = 1
        if impl.inside_kernel():
            return self.subscript(3)
        else:
            return self[3]

    # since Taichi-scope use v.x.assign() instead
    @x.setter
    @python_scope
    def x(self, value):
        _taichi_skip_traceback = 1
        self[0] = value

    @y.setter
    @python_scope
    def y(self, value):
        _taichi_skip_traceback = 1
        self[1] = value

    @z.setter
    @python_scope
    def z(self, value):
        _taichi_skip_traceback = 1
        self[2] = value

    @w.setter
    @python_scope
    def w(self, value):
        _taichi_skip_traceback = 1
        self[3] = value

    class Proxy:
        def __init__(self, mat, index):
            """Proxy when a tensor of Matrices is accessed by host."""
            self.mat = mat
            self.index = index

        @python_scope
        def __getitem__(self, item):
            if not isinstance(item, (list, tuple)):
                item = [item]
            return self.mat(*item)[self.index]

        @python_scope
        def __setitem__(self, key, value):
            if not isinstance(key, (list, tuple)):
                key = [key]
            self.mat(*key)[self.index] = value

        @property
        def x(self):
            return self[0]

        @property
        def y(self):
            return self[1]

        @property
        def z(self):
            return self[2]

        @property
        def w(self):
            return self[3]

        @x.setter
        def x(self, value):
            self[0] = value

        @y.setter
        def y(self, value):
            self[1] = value

        @z.setter
        def z(self, value):
            self[2] = value

        @w.setter
        def w(self, value):
            self[3] = value

        @property
        def value(self):
            ret = self.mat.empty_copy()
            for i in range(self.mat.n):
                for j in range(self.mat.m):
                    ret.entries[i * self.mat.m + j] = self.mat(i,
                                                               j)[self.index]
            return ret

    # host access & python scope operation
    @python_scope
    def __getitem__(self, indices):
        if self.is_global():
            return Matrix.Proxy(self, indices)

        if not isinstance(indices, (list, tuple)):
            indices = [indices]
        assert len(indices) in [1, 2]
        i = indices[0]
        j = 0 if len(indices) == 1 else indices[1]
        return self(i, j)

    @python_scope
    def __setitem__(self, indices, item):
        if self.is_global():
            if not isinstance(item, (list, tuple)):
                item = list(item)
            if not isinstance(item[0], (list, tuple)):
                item = [[i] for i in item]
            for i in range(self.n):
                for j in range(self.m):
                    self(i, j)[indices] = item[i][j]
            return

        if not isinstance(indices, (list, tuple)):
            indices = [indices]
        assert len(indices) in [1, 2]
        i = indices[0]
        j = 0 if len(indices) == 1 else indices[1]
        self.set_entry(i, j, item)

    def __len__(self):
        return self.n

    def __iter__(self):
        if self.m == 1:
            return (self(i) for i in range(self.n))
        else:
            return ([self(i, j) for j in range(self.m)] for i in range(self.n))

    def empty_copy(self):
        return Matrix.empty(self.n, self.m)

    def copy(self):
        ret = self.empty_copy()
        ret.entries = copy.copy(self.entries)
        return ret

    @taichi_scope
    def variable(self):
        ret = self.copy()
        ret.entries = [impl.expr_init(e) for e in ret.entries]
        return ret

    @taichi_scope
    def cast(self, dtype):
        _taichi_skip_traceback = 1
        ret = self.copy()
        for i in range(len(self.entries)):
            ret.entries[i] = ops_mod.cast(ret.entries[i], dtype)
        return ret

    def trace(self):
        assert self.n == self.m
        sum = self(0, 0)
        for i in range(1, self.n):
            sum = sum + self(i, i)
        return sum

    @taichi_scope
    def inverse(self):
        assert self.n == self.m, 'Only square matrices are invertible'
        if self.n == 1:
            return Matrix([1 / self(0, 0)])
        elif self.n == 2:
            inv_det = impl.expr_init(1.0 / self.determinant())
            # Discussion: https://github.com/taichi-dev/taichi/pull/943#issuecomment-626344323
            return inv_det * Matrix([[self(1, 1), -self(0, 1)],
                                     [-self(1, 0), self(0, 0)]]).variable()
        elif self.n == 3:
            n = 3
            inv_determinant = impl.expr_init(1.0 / self.determinant())
            entries = [[0] * n for _ in range(n)]

            def E(x, y):
                return self(x % n, y % n)

            for i in range(n):
                for j in range(n):
                    entries[j][i] = impl.expr_init(
                        inv_determinant * (E(i + 1, j + 1) * E(i + 2, j + 2) -
                                           E(i + 2, j + 1) * E(i + 1, j + 2)))
            return Matrix(entries)
        elif self.n == 4:
            n = 4
            inv_determinant = impl.expr_init(1.0 / self.determinant())
            entries = [[0] * n for _ in range(n)]

            def E(x, y):
                return self(x % n, y % n)

            for i in range(n):
                for j in range(n):
                    entries[j][i] = impl.expr_init(
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

    inversed = deprecated('a.inversed()', 'a.inverse()')(inverse)

    @kern_mod.pyfunc
    def normalized(self, eps=0):
        impl.static(
            impl.static_assert(self.m == 1,
                               "normalized() only works on vector"))
        invlen = 1 / (self.norm() + eps)
        return invlen * self

    @staticmethod
    @deprecated('ti.Matrix.transposed(a)', 'a.transpose()')
    def transposed(a):
        return a.transpose()

    @deprecated('a.T()', 'a.transpose()')
    def T(self):
        return self.transpose()

    @kern_mod.pyfunc
    def transpose(self):
        ret = Matrix([[self[i, j] for i in range(self.n)]
                      for j in range(self.m)])
        return ret

    @taichi_scope
    def determinant(a):
        if a.n == 2 and a.m == 2:
            return a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)
        elif a.n == 3 and a.m == 3:
            return a(0, 0) * (a(1, 1) * a(2, 2) - a(2, 1) * a(1, 2)) - a(
                1, 0) * (a(0, 1) * a(2, 2) - a(2, 1) * a(0, 2)) + a(
                    2, 0) * (a(0, 1) * a(1, 2) - a(1, 1) * a(0, 2))
        elif a.n == 4 and a.m == 4:
            n = 4

            def E(x, y):
                return a(x % n, y % n)

            det = impl.expr_init(0.0)
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
    def diag(dim, val):
        ret = Matrix(dim, dim)
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    ret.set_entry(i, j, val)
                else:
                    ret.set_entry(i, j, 0 * val)
                    # TODO: need a more systematic way to create a "0" with the right type
        return ret

    def loop_range(self):
        return self.entries[0]

    @property
    def shape(self):
        # Took `self.entries[0]` as a representation of this tensor-of-matrices.
        # https://github.com/taichi-dev/taichi/issues/1069#issuecomment-635712140
        return self.loop_range().shape

    @deprecated('x.dim()', 'len(x.shape)')
    def dim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self.loop_range().dtype

    @deprecated('x.data_type()', 'x.dtype')
    def data_type(self):
        return self.dtype

    @property
    def snode(self):
        return self.loop_range().snode

    def make_grad(self):
        ret = self.empty_copy()
        for i in range(len(ret.entries)):
            ret.entries[i] = self.entries[i].grad
        return ret

    def sum(self):
        ret = self.entries[0]
        for i in range(1, len(self.entries)):
            ret = ret + self.entries[i]
        return ret

    @kern_mod.pyfunc
    def norm(self, eps=0):
        return ops_mod.sqrt(self.norm_sqr() + eps)

    @kern_mod.pyfunc
    def norm_inv(self, eps=0):
        return ops_mod.rsqrt(self.norm_sqr() + eps)

    @kern_mod.pyfunc
    def norm_sqr(self):
        return (self**2).sum()

    @kern_mod.pyfunc
    def max(self):
        return ops_mod.ti_max(*self.entries)

    @kern_mod.pyfunc
    def min(self):
        return ops_mod.ti_min(*self.entries)

    def any(self):
        ret = ti.cmp_ne(self.entries[0], 0)
        for i in range(1, len(self.entries)):
            ret = ret + ti.cmp_ne(self.entries[i], 0)
        return -ti.cmp_lt(ret, 0)

    def all(self):
        ret = ti.cmp_ne(self.entries[0], 0)
        for i in range(1, len(self.entries)):
            ret = ret + ti.cmp_ne(self.entries[i], 0)
        return -ti.cmp_eq(ret, -len(self.entries))

    def fill(self, val):
        if impl.inside_kernel():

            def assign_renamed(x, y):
                return ti.assign(x, y)

            return self.element_wise_writeback_binary(assign_renamed, val)

        if isinstance(val, numbers.Number):
            val = tuple(
                [tuple([val for _ in range(self.m)]) for _ in range(self.n)])
        elif isinstance(val,
                        (list, tuple)) and isinstance(val[0], numbers.Number):
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
        from taichi.lang.meta import fill_matrix
        fill_matrix(self, val)

    @python_scope
    def to_numpy(self, keep_dims=False, as_vector=None, dtype=None):
        # Discussion: https://github.com/taichi-dev/taichi/pull/1046#issuecomment-633548858
        if as_vector is not None:
            warning(
                'v.to_numpy(as_vector=True) is deprecated, '
                'please use v.to_numpy() directly instead',
                DeprecationWarning,
                stacklevel=3)
        as_vector = self.m == 1 and not keep_dims
        shape_ext = (self.n, ) if as_vector else (self.n, self.m)

        if not self.is_global():
            return np.array(self.entries).reshape(shape_ext)

        if dtype is None:
            dtype = to_numpy_type(self.dtype)
        ret = np.zeros(self.shape + shape_ext, dtype=dtype)
        from taichi.lang.meta import matrix_to_ext_arr
        matrix_to_ext_arr(self, ret, as_vector)
        return ret

    @python_scope
    def to_torch(self, device=None, keep_dims=False):
        import torch
        as_vector = self.m == 1 and not keep_dims
        shape_ext = (self.n, ) if as_vector else (self.n, self.m)
        ret = torch.empty(self.shape + shape_ext,
                          dtype=to_pytorch_type(self.dtype),
                          device=device)
        from taichi.lang.meta import matrix_to_ext_arr
        matrix_to_ext_arr(self, ret, as_vector)
        ti.sync()
        return ret

    @python_scope
    def from_numpy(self, ndarray):
        if len(ndarray.shape) == len(self.loop_range().shape) + 1:
            as_vector = True
            assert self.m == 1, "This matrix is not a vector"
        else:
            as_vector = False
            assert len(ndarray.shape) == len(self.loop_range().shape) + 2
        dim_ext = 1 if as_vector else 2
        assert len(ndarray.shape) == len(self.loop_range().shape) + dim_ext
        from taichi.lang.meta import ext_arr_to_matrix
        ext_arr_to_matrix(ndarray, self, as_vector)
        ti.sync()

    @python_scope
    def from_torch(self, torch_tensor):
        return self.from_numpy(torch_tensor.contiguous())

    @python_scope
    def copy_from(self, other):
        assert isinstance(other, Matrix)
        from taichi.lang.meta import tensor_to_tensor
        assert len(self.shape) == len(other.shape)
        tensor_to_tensor(self, other)

    @taichi_scope
    def __ti_repr__(self):
        yield '['
        for i in range(self.n):
            if i:
                yield ', '
            if self.m != 1:
                yield '['
            for j in range(self.m):
                if j:
                    yield ', '
                yield self(i, j)
            if self.m != 1:
                yield ']'
        yield ']'

    def __str__(self):
        """Python scope matrix print support."""
        if impl.inside_kernel():
            '''
            It seems that when pybind11 got an type mismatch, it will try
            to invoke `repr` to show the object... e.g.:

            TypeError: make_const_expr_f32(): incompatible function arguments. The following argument types are supported:
                1. (arg0: float) -> taichi_core.Expr

            Invoked with: <Taichi 2x1 Matrix>

            So we have to make it happy with a dummy string...
            '''
            return f'<{self.n}x{self.m} ti.Matrix>'
        else:
            return str(self.to_numpy())

    def __repr__(self):
        if self.is_global():
            # make interactive shell happy, prevent materialization
            return f'<{self.n}x{self.m} ti.Matrix.field>'
        else:
            return str(self.to_numpy())

    @staticmethod
    @taichi_scope
    def zero(dt, n, m=1):
        return Matrix([[ti.cast(0, dt) for _ in range(m)] for _ in range(n)])

    @staticmethod
    @taichi_scope
    def one(dt, n, m=1):
        return Matrix([[ti.cast(1, dt) for _ in range(m)] for _ in range(n)])

    @staticmethod
    @taichi_scope
    def unit(n, i, dt=None):
        if dt is None:
            dt = int
        assert 0 <= i < n
        return Matrix([ti.cast(int(j == i), dt) for j in range(n)])

    @staticmethod
    @taichi_scope
    def identity(dt, n):
        return Matrix([[ti.cast(int(i == j), dt) for j in range(n)]
                       for i in range(n)])

    @staticmethod
    def rotation2d(alpha):
        return Matrix([[ti.cos(alpha), -ti.sin(alpha)],
                       [ti.sin(alpha), ti.cos(alpha)]])

    @classmethod
    @python_scope
    def field(cls,
              n,
              m,
              dtype,
              shape=None,
              offset=None,
              needs_grad=False,
              layout=None):  # TODO(archibate): deprecate layout
        '''ti.Matrix.field'''
        self = cls.empty(n, m)
        self.entries = []
        self.n = n
        self.m = m
        self.dt = dtype

        if isinstance(dtype, (list, tuple, np.ndarray)):
            # set different dtype for each element in Matrix
            # see #2135
            if m == 1:
                assert len(np.shape(dtype)) == 1 and len(
                    dtype
                ) == n, f'Please set correct dtype list for Vector. The shape of dtype list should be ({n}, ) instead of {np.shape(dtype)}'
                for i in range(n):
                    self.entries.append(impl.field(dtype[i]))
            else:
                assert len(np.shape(dtype)) == 2 and len(dtype) == n and len(
                    dtype[0]
                ) == m, f'Please set correct dtype list for Matrix. The shape of dtype list should be ({n}, {m}) instead of {np.shape(dtype)}'
                for i in range(n):
                    for j in range(m):
                        self.entries.append(impl.field(dtype[i][j]))
        else:
            for _ in range(n * m):
                self.entries.append(impl.field(dtype))
        self.grad = self.make_grad()

        if layout is not None:
            assert shape is not None, 'layout is useless without shape'
        if shape is None:
            assert offset is None, "shape cannot be None when offset is being set"

        if shape is not None:
            if isinstance(shape, numbers.Number):
                shape = (shape, )
            if isinstance(offset, numbers.Number):
                offset = (offset, )

            if offset is not None:
                assert len(shape) == len(
                    offset
                ), f'The dimensionality of shape and offset must be the same  ({len(shape)} != {len(offset)})'

            if layout is None:
                layout = ti.AOS

            dim = len(shape)
            if layout.soa:
                for i, e in enumerate(self.entries):
                    ti.root.dense(impl.index_nd(dim),
                                  shape).place(e, offset=offset)
                    if needs_grad:
                        ti.root.dense(impl.index_nd(dim),
                                      shape).place(e.grad, offset=offset)
            else:
                var_list = []
                for i, e in enumerate(self.entries):
                    var_list.append(e)
                if needs_grad:
                    for i, e in enumerate(self.entries):
                        var_list.append(e.grad)
                ti.root.dense(impl.index_nd(dim),
                              shape).place(*tuple(var_list), offset=offset)
        return self

    @classmethod
    @python_scope
    @deprecated('ti.Matrix.var', 'ti.Matrix.field')
    def var(cls, n, m, dt, *args, **kwargs):
        '''ti.Matrix.var'''
        _taichi_skip_traceback = 1
        return cls.field(n, m, dt, *args, **kwargs)

    @classmethod
    def _Vector_field(cls, n, dtype, *args, **kwargs):
        '''ti.Vector.field'''
        _taichi_skip_traceback = 1
        return cls.field(n, 1, dtype, *args, **kwargs)

    @classmethod
    @deprecated('ti.Vector.var', 'ti.Vector.field')
    def _Vector_var(cls, n, dt, *args, **kwargs):
        '''ti.Vector.var'''
        _taichi_skip_traceback = 1
        return cls._Vector_field(n, dt, *args, **kwargs)

    @staticmethod
    def rows(rows):
        mat = Matrix()
        mat.n = len(rows)
        if isinstance(rows[0], Matrix):
            for row in rows:
                assert row.m == 1, "Inputs must be vectors, i.e. m == 1"
                assert row.n == rows[
                    0].n, "Input vectors must share the same shape"
            mat.m = rows[0].n
            # l-value copy:
            mat.entries = [row(i) for row in rows for i in range(row.n)]
        elif isinstance(rows[0], list):
            for row in rows:
                assert len(row) == len(
                    rows[0]), "Input lists share the same shape"
            mat.m = len(rows[0])
            # l-value copy:
            mat.entries = [x for row in rows for x in row]
        else:
            raise Exception(
                "Cols/rows must be a list of lists, or a list of vectors")
        return mat

    @staticmethod
    def cols(cols):
        return Matrix.rows(cols).transpose()

    @classmethod
    def empty(cls, n, m):
        return cls([[None] * m for _ in range(n)])

    @classmethod
    def new(cls, n, m):
        if impl.inside_kernel():
            return cls(n, m)
        else:
            return cls.empty(n, m)

    def __hash__(self):
        # TODO: refactor KernelTemplateMapper
        # If not, we get `unhashable type: Matrix` when
        # using matrices as template arguments.
        return id(self)

    @kern_mod.pyfunc
    def dot(self, other):
        impl.static(
            impl.static_assert(self.m == 1, "lhs for dot is not a vector"))
        impl.static(
            impl.static_assert(other.m == 1, "rhs for dot is not a vector"))
        return (self * other).sum()

    @kern_mod.pyfunc
    def _cross3d(self, other):
        ret = Matrix([
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ])
        return ret

    @kern_mod.pyfunc
    def _cross2d(self, other):
        ret = self[0] * other[1] - self[1] * other[0]
        return ret

    def cross(self, other):
        if self.n == 3 and self.m == 1 and other.n == 3 and other.m == 1:
            return self._cross3d(other)

        elif self.n == 2 and self.m == 1 and other.n == 2 and other.m == 1:
            return self._cross2d(other)

        else:
            raise ValueError(
                "Cross product is only supported between pairs of 2D/3D vectors"
            )

    @kern_mod.pyfunc
    def outer_product(self, other):
        impl.static(
            impl.static_assert(self.m == 1,
                               "lhs for outer_product is not a vector"))
        impl.static(
            impl.static_assert(other.m == 1,
                               "rhs for outer_product is not a vector"))
        ret = Matrix([[self[i] * other[j] for j in range(other.n)]
                      for i in range(self.n)])
        return ret


# TODO: deprecate ad-hoc use ti.Matrix() as global (#1500:2.2/2)
def Vector(n, dt=None, shape=None, offset=None, **kwargs):
    return Matrix(n, 1, dt=dt, shape=shape, offset=offset, **kwargs)


Vector.var = Matrix._Vector_var
Vector.field = Matrix._Vector_field
Vector.zero = Matrix.zero
Vector.one = Matrix.one
Vector.dot = Matrix.dot
Vector.cross = Matrix.cross
Vector.outer_product = Matrix.outer_product
Vector.unit = Matrix.unit
Vector.normalized = Matrix.normalized
