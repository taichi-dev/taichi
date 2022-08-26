import numbers
import warnings
from collections.abc import Iterable

import numpy as np
from taichi._lib import core as ti_python_core
from taichi.lang import expr, impl
from taichi.lang import ops as ops_mod
from taichi.lang import runtime_ops
from taichi.lang._ndarray import Ndarray, NdarrayHostAccess
from taichi.lang.common_ops import TaichiOperations
from taichi.lang.enums import Layout
from taichi.lang.exception import (TaichiCompilationError, TaichiSyntaxError,
                                   TaichiTypeError)
from taichi.lang.field import Field, ScalarField, SNodeHostAccess
from taichi.lang.swizzle_generator import SwizzleGenerator
from taichi.lang.util import (cook_dtype, in_python_scope, python_scope,
                              taichi_scope, to_numpy_type, to_paddle_type,
                              to_pytorch_type, warning)
from taichi.types import primitive_types
from taichi.types.compound_types import CompoundType, TensorType


def _gen_swizzles(cls):
    swizzle_gen = SwizzleGenerator()
    # https://www.khronos.org/opengl/wiki/Data_Type_(GLSL)#Swizzling
    KEYGROUP_SET = ['xyzw', 'rgba', 'stpq']

    def make_valid_attribs_checker(key_group):
        def check(instance, pattern):
            valid_attribs = set(key_group[:instance.n])
            pattern_set = set(pattern)
            diff = pattern_set - valid_attribs
            if len(diff):
                valid_attribs = tuple(sorted(valid_attribs))
                pattern = tuple(pattern)
                raise TaichiSyntaxError(
                    f'vec{instance.n} only has '
                    f'attributes={valid_attribs}, got={pattern}')

        return check

    for key_group in KEYGROUP_SET:
        for index, attr in enumerate(key_group):

            def gen_property(attr, attr_idx, key_group):
                checker = make_valid_attribs_checker(key_group)

                def prop_getter(instance):
                    checker(instance, attr)
                    return instance._impl._get_entry_and_read([attr_idx])

                @python_scope
                def prop_setter(instance, value):
                    checker(instance, attr)
                    instance[attr_idx] = value

                return property(prop_getter, prop_setter)

            prop = gen_property(attr, index, key_group)
            setattr(cls, attr, prop)

    for key_group in KEYGROUP_SET:
        sw_patterns = swizzle_gen.generate(key_group, required_length=4)
        # len=1 accessors are handled specially above
        sw_patterns = filter(lambda p: len(p) > 1, sw_patterns)
        for pat in sw_patterns:
            # Create a function for value capturing
            def gen_property(pattern, key_group):
                checker = make_valid_attribs_checker(key_group)
                prop_key = ''.join(pattern)

                def prop_getter(instance):
                    checker(instance, pattern)
                    res = []
                    for ch in pattern:
                        res.append(
                            instance._impl._get_entry(key_group.index(ch)))
                    return Vector(res, is_ref=True)

                def prop_setter(instance, value):
                    if len(pattern) != len(value):
                        raise TaichiCompilationError(
                            f'value len does not match the swizzle pattern={prop_key}'
                        )
                    checker(instance, pattern)
                    for ch, val in zip(pattern, value):
                        if in_python_scope():
                            instance[key_group.index(ch)] = val
                        else:
                            instance(key_group.index(ch))._assign(val)

                prop = property(prop_getter, prop_setter)
                return prop_key, prop

            prop_key, prop = gen_property(pat, key_group)
            setattr(cls, prop_key, prop)
    return cls


def make_matrix(arr, dt=None):
    assert len(arr) > 0, "Cannot create empty matrix"
    is_matrix = isinstance(arr[0], Iterable)
    if dt is None:
        dt = _make_entries_initializer(is_matrix).infer_dt(arr)
    if not is_matrix:
        return impl.Expr(
            impl.make_matrix_expr([len(arr)], dt,
                                  [expr.Expr(elt).ptr for elt in arr]))
    return impl.Expr(
        impl.make_matrix_expr(
            [len(arr), len(arr[0])], dt,
            [expr.Expr(elt).ptr for row in arr for elt in row]))


class _MatrixBaseImpl:
    def __init__(self, m, n, entries):
        self.m = m
        self.n = n
        self.entries = entries

    def _get_entry(self, *indices):
        return self.entries[self._linearize_entry_id(*indices)]

    def _get_entry_and_read(self, indices):
        # Can be invoked in both Python and Taichi scope. `indices` must be
        # compile-time constants (e.g. Python values)
        ret = self._get_entry(*indices)

        if isinstance(ret, SNodeHostAccess):
            ret = ret.accessor.getter(*ret.key)
        elif isinstance(ret, NdarrayHostAccess):
            ret = ret.getter()
        return ret

    def _linearize_entry_id(self, *args):
        assert 1 <= len(args) <= 2
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        if len(args) == 1:
            args = args + (0, )
        # TODO(#1004): See if it's possible to support indexing at runtime
        for i, a in enumerate(args):
            if not isinstance(a, (int, np.integer)):
                raise TaichiSyntaxError(
                    f'The {i}-th index of a Matrix/Vector must be a compile-time constant '
                    f'integer, got {type(a)}.\n'
                    'This is because matrix operations will be **unrolled** at compile-time '
                    'for performance reason.\n'
                    'If you want to *iterate through matrix elements*, use a static range:\n'
                    '  for i in ti.static(range(3)):\n'
                    '    print(i, "-th component is", vec[i])\n'
                    'See https://docs.taichi-lang.org/docs/meta#when-to-use-tistatic-with-for-loops for more details.'
                    'Or turn on ti.init(..., dynamic_index=True) to support indexing with variables!'
                )
        assert 0 <= args[0] < self.n, \
            f"The 0-th matrix index is out of range: 0 <= {args[0]} < {self.n}"
        assert 0 <= args[1] < self.m, \
            f"The 1-th matrix index is out of range: 0 <= {args[1]} < {self.m}"
        return args[0] * self.m + args[1]


class _PyScopeMatrixImpl(_MatrixBaseImpl):
    @python_scope
    def __getitem__(self, indices):
        """Access to the element at the given indices in a matrix.

        Args:
            indices (Sequence[Expr]): the indices of the element.

        Returns:
            The value of the element at a specific position of a matrix.

        """
        return self.subscript_scope_ignored(indices)

    def subscript_scope_ignored(self, indices):
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
        assert len(indices) in [1, 2]
        i = indices[0]
        j = 0 if len(indices) == 1 else indices[1]
        if isinstance(i, slice) or isinstance(j, slice):
            return self._get_slice(i, j)
        return self._get_entry_and_read([i, j])

    @python_scope
    def __setitem__(self, indices, item):
        """Set the element value at the given indices in a matrix.

        Args:
            indices (Sequence[Expr]): the indices of a element.

        """
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
        assert len(indices) in [1, 2]
        i = indices[0]
        j = 0 if len(indices) == 1 else indices[1]
        idx = self._linearize_entry_id(i, j)
        if isinstance(self.entries[idx], SNodeHostAccess):
            self.entries[idx].accessor.setter(item, *self.entries[idx].key)
        elif isinstance(self.entries[idx], NdarrayHostAccess):
            self.entries[idx].setter(item)
        else:
            self.entries[idx] = item

    def _get_slice(self, a, b):
        if not isinstance(a, slice):
            a = [a]
        else:
            a = range(a.start or 0, a.stop or self.n, a.step or 1)
        if not isinstance(b, slice):
            b = [b]
        else:
            b = range(b.start or 0, b.stop or self.m, b.step or 1)
        return Matrix([[self._get_entry(i, j) for j in b] for i in a])

    def _set_entries(self, value):
        if not isinstance(value, (list, tuple)):
            value = list(value)
        if not isinstance(value[0], (list, tuple)):
            value = [[i] for i in value]
        for i in range(self.n):
            for j in range(self.m):
                self[i, j] = value[i][j]


class _TiScopeMatrixImpl(_MatrixBaseImpl):
    def __init__(self, m, n, entries, local_tensor_proxy,
                 dynamic_index_stride):
        super().__init__(m, n, entries)
        self.any_array_access = None
        self.local_tensor_proxy = local_tensor_proxy
        self.dynamic_index_stride = dynamic_index_stride

    @taichi_scope
    def _subscript(self, is_global_mat, *indices, get_ref=False):
        assert len(indices) in [1, 2]
        i = indices[0]
        j = 0 if len(indices) == 1 else indices[1]
        has_slice = False
        if isinstance(i, slice):
            i = self._calc_slice(i, 0)
            has_slice = True
        if isinstance(j, slice):
            j = self._calc_slice(j, 1)
            has_slice = True

        if has_slice:
            if not isinstance(i, list):
                i = [i]
            if not isinstance(j, list):
                j = [j]
            if len(indices) == 1:
                return Vector([self._subscript(is_global_mat, a) for a in i],
                              is_ref=get_ref)
            return Matrix([[self._subscript(is_global_mat, a, b) for b in j]
                           for a in i],
                          is_ref=get_ref,
                          ndim=1)

        if self.any_array_access:
            return self.any_array_access.subscript(i, j)
        if self.local_tensor_proxy is not None:
            if len(indices) == 1:
                return impl.make_index_expr(self.local_tensor_proxy, (i, ))
            return impl.make_index_expr(self.local_tensor_proxy, (i, j))
        if impl.current_cfg(
        ).dynamic_index and is_global_mat and self.dynamic_index_stride:
            return impl.make_stride_expr(self.entries[0].ptr, (i, j),
                                         (self.n, self.m),
                                         self.dynamic_index_stride)
        return self._get_entry(i, j)

    def _calc_slice(self, index, dim):
        start, stop, step = index.start or 0, index.stop or (
            self.n if dim == 0 else self.m), index.step or 1

        def helper(x):
            #  TODO(mzmzm): support variable in slice
            if isinstance(x, expr.Expr):
                raise TaichiCompilationError(
                    "Taichi does not support variables in slice now, please use constant instead of it."
                )
            return x

        start, stop, step = helper(start), helper(stop), helper(step)
        return [_ for _ in range(start, stop, step)]


class _MatrixEntriesInitializer:
    def pyscope_or_ref(self, arr):
        raise NotImplementedError('Override')

    def no_dynamic_index(self, arr, dt):
        raise NotImplementedError('Override')

    def with_dynamic_index(self, arr, dt):
        raise NotImplementedError('Override')

    def _get_entry_to_infer(self, arr):
        raise NotImplementedError('Override')

    def infer_dt(self, arr):
        entry = self._get_entry_to_infer(arr)
        if isinstance(entry, (int, np.integer)):
            return impl.get_runtime().default_ip
        if isinstance(entry, float):
            return impl.get_runtime().default_fp
        if isinstance(entry, expr.Expr):
            dt = entry.ptr.get_ret_type()
            if dt == ti_python_core.DataType_unknown:
                raise TypeError(
                    'Element type of the matrix cannot be inferred. Please set dt instead for now.'
                )
            return dt
        raise Exception(
            'dt required when using dynamic_index for local tensor')


def _make_entries_initializer(is_matrix: bool) -> _MatrixEntriesInitializer:
    class _VecImpl(_MatrixEntriesInitializer):
        def pyscope_or_ref(self, arr):
            return [[x] for x in arr]

        def no_dynamic_index(self, arr, dt):
            return [[impl.expr_init(ops_mod.cast(x, dt) if dt else x)]
                    for x in arr]

        def with_dynamic_index(self, arr, dt):
            local_tensor_proxy = impl.expr_init_local_tensor(
                [len(arr)], dt,
                expr.make_expr_group([expr.Expr(x) for x in arr]))
            mat = []
            for i in range(len(arr)):
                mat.append(
                    list([
                        impl.make_index_expr(
                            local_tensor_proxy,
                            (expr.Expr(i, dtype=primitive_types.i32), ))
                    ]))
            return local_tensor_proxy, mat

        def _get_entry_to_infer(self, arr):
            return arr[0]

    class _MatImpl(_MatrixEntriesInitializer):
        def pyscope_or_ref(self, arr):
            return [list(row) for row in arr]

        def no_dynamic_index(self, arr, dt):
            return [[
                impl.expr_init(ops_mod.cast(x, dt) if dt else x) for x in row
            ] for row in arr]

        def with_dynamic_index(self, arr, dt):
            local_tensor_proxy = impl.expr_init_local_tensor(
                [len(arr), len(arr[0])], dt,
                expr.make_expr_group(
                    [expr.Expr(x) for row in arr for x in row]))

            mat = []
            for i in range(len(arr)):
                mat.append([])
                for j in range(len(arr[0])):
                    mat[i].append(
                        impl.make_index_expr(
                            local_tensor_proxy,
                            (expr.Expr(i, dtype=primitive_types.i32),
                             expr.Expr(j, dtype=primitive_types.i32))))
            return local_tensor_proxy, mat

        def _get_entry_to_infer(self, arr):
            return arr[0][0]

    return _MatImpl() if is_matrix else _VecImpl()


@_gen_swizzles
class Matrix(TaichiOperations):
    """The matrix class.

    A matrix is a 2-D rectangular array with scalar entries, it's row-majored, and is
    aligned continuously. We recommend only use matrix with no more than 32 elements for
    efficiency considerations.

    Note: in taichi a matrix is strictly two-dimensional and only stores scalars.

    Args:
        arr (Union[list, tuple, np.ndarray]): the initial values of a matrix.
        dt (:mod:`~taichi.types.primitive_types`): the element data type.
        suppress_warning (bool): whether raise warning or not when the matrix contains more \
            than 32 elements.
        ndim (int optional): the number of dimensions of the matrix; forced reshape if given.

    Example::

        use a 2d list to initialize a matrix

        >>> @ti.kernel
        >>> def test():
        >>>     n = 5
        >>>     M = ti.Matrix([[0] * n for _ in range(n)], ti.i32)
        >>>     print(M)  # a 5x5 matrix with integer elements

        get the number of rows and columns via the `n`, `m` property:

        >>> M = ti.Matrix([[0, 1], [2, 3], [4, 5]], ti.i32)
        >>> M.n  # number of rows
        3
        >>> M.m  # number of cols
        >>> 2

        you can even initialize a matrix with an empty list:

        >>> M = ti.Matrix([[], []], ti.i32)
        >>> M.n
        2
        >>> M.m
        0
    """
    _is_taichi_class = True
    __array_priority__ = 1000

    def __init__(self,
                 arr,
                 dt=None,
                 suppress_warning=False,
                 is_ref=False,
                 ndim=None):
        local_tensor_proxy = None

        if not isinstance(arr, (list, tuple, np.ndarray)):
            raise TaichiTypeError(
                "An Matrix/Vector can only be initialized with an array-like object"
            )
        if len(arr) == 0:
            mat = []
            self.ndim = 0
        elif isinstance(arr[0], Matrix):
            raise Exception('cols/rows required when using list of vectors')
        else:
            is_matrix = isinstance(arr[0], Iterable)
            initializer = _make_entries_initializer(is_matrix)
            self.ndim = 2 if is_matrix else 1

            if in_python_scope() or is_ref:
                mat = initializer.pyscope_or_ref(arr)
            elif not impl.current_cfg().dynamic_index:
                mat = initializer.no_dynamic_index(arr, dt)
            else:
                if not ti_python_core.is_extension_supported(
                        impl.current_cfg().arch,
                        ti_python_core.Extension.dynamic_index):
                    raise Exception(
                        f"Backend {impl.current_cfg().arch} doesn't support dynamic index"
                    )
                if dt is None:
                    dt = initializer.infer_dt(arr)
                local_tensor_proxy, mat = initializer.with_dynamic_index(
                    arr, dt)
        self.n, self.m = len(mat), 1
        if len(mat) > 0:
            self.m = len(mat[0])
        entries = [x for row in mat for x in row]

        if ndim is not None:
            # override ndim after reading data from mat
            assert ndim in (0, 1, 2)
            self.ndim = ndim

        if self.n * self.m > 32 and not suppress_warning:
            warning(
                f'Taichi matrices/vectors with {self.n}x{self.m} > 32 entries are not suggested.'
                ' Matrices/vectors will be automatically unrolled at compile-time for performance.'
                ' So the compilation time could be extremely long if the matrix size is too big.'
                ' You may use a field to store a large matrix like this, e.g.:\n'
                f'    x = ti.field(ti.f32, ({self.n}, {self.m})).\n'
                ' See https://docs.taichi-lang.org/docs/field#matrix-size'
                ' for more details.',
                UserWarning,
                stacklevel=2)
        m, n = self.m, self.n
        if in_python_scope():
            self._impl = _PyScopeMatrixImpl(m, n, entries)
        else:
            self._impl = _TiScopeMatrixImpl(m, n, entries, local_tensor_proxy,
                                            None)

    def _element_wise_binary(self, foo, other):
        other = self._broadcast_copy(other)
        return Matrix([[foo(self(i, j), other(i, j)) for j in range(self.m)]
                       for i in range(self.n)],
                      ndim=self.ndim)

    def _broadcast_copy(self, other):
        if isinstance(other, (list, tuple)):
            other = Matrix(other)
        if not isinstance(other, Matrix):
            other = Matrix([[other for _ in range(self.m)]
                            for _ in range(self.n)],
                           ndim=self.ndim)
        assert self.m == other.m and self.n == other.n, f"Dimension mismatch between shapes ({self.n}, {self.m}), ({other.n}, {other.m})"
        return other

    def _element_wise_ternary(self, foo, other, extra):
        other = self._broadcast_copy(other)
        extra = self._broadcast_copy(extra)
        return Matrix([[
            foo(self(i, j), other(i, j), extra(i, j)) for j in range(self.m)
        ] for i in range(self.n)],
                      ndim=self.ndim)

    def _element_wise_writeback_binary(self, foo, other):
        if foo.__name__ == 'assign' and not isinstance(other,
                                                       (list, tuple, Matrix)):
            raise TaichiSyntaxError(
                'cannot assign scalar expr to '
                f'taichi class {type(self)}, maybe you want to use `a.fill(b)` instead?'
            )
        other = self._broadcast_copy(other)
        entries = [[foo(self(i, j), other(i, j)) for j in range(self.m)]
                   for i in range(self.n)]
        return self if foo.__name__ == 'assign' else Matrix(entries,
                                                            ndim=self.ndim)

    def _element_wise_unary(self, foo):
        return Matrix([[foo(self(i, j)) for j in range(self.m)]
                       for i in range(self.n)],
                      ndim=self.ndim)

    def __matmul__(self, other):
        """Matrix-matrix or matrix-vector multiply.

        Args:
            other (Union[Matrix, Vector]): a matrix or a vector.

        Returns:
            The matrix-matrix product or matrix-vector product.

        """
        assert isinstance(other, Matrix), "rhs of `@` is not a matrix / vector"
        assert self.m == other.n, f"Dimension mismatch between shapes ({self.n}, {self.m}), ({other.n}, {other.m})"
        entries = []
        for i in range(self.n):
            entries.append([])
            for j in range(other.m):
                acc = self(i, 0) * other(0, j)
                for k in range(1, other.n):
                    acc = acc + self(i, k) * other(k, j)
                entries[i].append(acc)
        return Matrix(entries)

    # host access & python scope operation
    def __len__(self):
        """Get the length of each row of a matrix"""
        # TODO: When this is a vector, should return its dimension?
        return self.n

    def __iter__(self):
        if self.m == 1:
            return (self(i) for i in range(self.n))
        return ([self(i, j) for j in range(self.m)] for i in range(self.n))

    @python_scope
    def __getitem__(self, indices):
        """Access to the element at the given indices in a matrix.

        Args:
            indices (Sequence[Expr]): the indices of the element.

        Returns:
            The value of the element at a specific position of a matrix.

        """
        if not isinstance(indices, Iterable):
            indices = [indices]
        assert len(
            indices
        ) == self.ndim, f"Expected {self.ndim} indices, got {len(indices)}"
        return self._impl[indices]

    @python_scope
    def __setitem__(self, indices, item):
        """Set the element value at the given indices in a matrix.

        Args:
            indices (Sequence[Expr]): the indices of a element.

        """
        if not isinstance(indices, Iterable):
            indices = [indices]
        assert len(
            indices
        ) == self.ndim, f"Expected {self.ndim} indices, got {len(indices)}"
        self._impl[indices] = item

    def __call__(self, *args, **kwargs):
        # TODO: It's quite hard to search for __call__, consider replacing this
        # with a method of actual names?
        assert kwargs == {}
        return self._impl._get_entry_and_read(args)

    @python_scope
    def _set_entries(self, value):
        self._impl._set_entries(value)

    @property
    def entries(self):
        return self._impl.entries

    @property
    def any_array_access(self):
        return self._impl.any_array_access

    @any_array_access.setter
    def any_array_access(self, value):
        self._impl.any_array_access = value

    @property
    def local_tensor_proxy(self):
        return self._impl.local_tensor_proxy

    @property
    def dynamic_index_stride(self):
        return self._impl.dynamic_index_stride

    @taichi_scope
    def _subscript(self, *indices, get_ref=False):
        assert len(
            indices
        ) == self.ndim, f"Expected {self.ndim} indices, got {len(indices)}"
        if isinstance(self._impl, _PyScopeMatrixImpl):
            # This can happen in these cases:
            # 1. A Python scope matrix is passed into a Taichi kernel as ti.template()
            # 2. Taichi kernel directlly uses a matrix (global variable) created in the Python scope.
            return self._impl.subscript_scope_ignored(indices)
        is_global_mat = isinstance(self, _MatrixFieldElement)
        return self._impl._subscript(is_global_mat, *indices, get_ref=get_ref)

    def to_list(self):
        """Return this matrix as a 1D `list`.

        This is similar to `numpy.ndarray`'s `flatten` and `ravel` methods,
        the difference is that this function always returns a new list.
        """
        return [[self(i, j) for j in range(self.m)] for i in range(self.n)]

    @taichi_scope
    def cast(self, dtype):
        """Cast the matrix elements to a specified data type.

        Args:
            dtype (:mod:`~taichi.types.primitive_types`): data type of the
                returned matrix.

        Returns:
            :class:`taichi.Matrix`: A new matrix with the specified data dtype.

        Example::

            >>> A = ti.Matrix([0, 1, 2], ti.i32)
            >>> B = A.cast(ti.f32)
            >>> B
            [0.0, 1.0, 2.0]
        """
        return Matrix(
            [[ops_mod.cast(self(i, j), dtype) for j in range(self.m)]
             for i in range(self.n)],
            ndim=self.ndim)

    def trace(self):
        """The sum of a matrix diagonal elements.

        To call this method the matrix must be square-like.

        Returns:
            The sum of a matrix diagonal elements.

        Example::

            >>> m = ti.Matrix([[1, 2], [3, 4]])
            >>> m.trace()
            5
        """
        assert self.n == self.m
        _sum = self(0, 0)
        for i in range(1, self.n):
            _sum = _sum + self(i, i)
        return _sum

    @taichi_scope
    def inverse(self):
        """Returns the inverse of this matrix.

        Note:
            The matrix dimension should be less than or equal to 4.

        Returns:
            :class:`~taichi.Matrix`: The inverse of a matrix.

        Raises:
            Exception: Inversions of matrices with sizes >= 5 are not supported.
        """
        assert self.n == self.m, 'Only square matrices are invertible'
        if self.n == 1:
            return Matrix([1 / self(0, 0)])
        if self.n == 2:
            inv_determinant = impl.expr_init(1.0 / self.determinant())
            return inv_determinant * Matrix([[self(
                1, 1), -self(0, 1)], [-self(1, 0), self(0, 0)]])
        if self.n == 3:
            n = 3
            inv_determinant = impl.expr_init(1.0 / self.determinant())
            entries = [[0] * n for _ in range(n)]

            def E(x, y):
                return self(x % n, y % n)

            for i in range(n):
                for j in range(n):
                    entries[j][i] = inv_determinant * (
                        E(i + 1, j + 1) * E(i + 2, j + 2) -
                        E(i + 2, j + 1) * E(i + 1, j + 2))
            return Matrix(entries)
        if self.n == 4:
            n = 4
            inv_determinant = impl.expr_init(1.0 / self.determinant())
            entries = [[0] * n for _ in range(n)]

            def E(x, y):
                return self(x % n, y % n)

            for i in range(n):
                for j in range(n):
                    entries[j][i] = inv_determinant * (-1)**(i + j) * ((
                        E(i + 1, j + 1) *
                        (E(i + 2, j + 2) * E(i + 3, j + 3) -
                         E(i + 3, j + 2) * E(i + 2, j + 3)) - E(i + 2, j + 1) *
                        (E(i + 1, j + 2) * E(i + 3, j + 3) -
                         E(i + 3, j + 2) * E(i + 1, j + 3)) + E(i + 3, j + 1) *
                        (E(i + 1, j + 2) * E(i + 2, j + 3) -
                         E(i + 2, j + 2) * E(i + 1, j + 3))))
            return Matrix(entries)
        raise Exception(
            "Inversions of matrices with sizes >= 5 are not supported")

    def normalized(self, eps=0):
        """Normalize a vector, i.e. matrices with the second dimension being
        equal to one.

        The normalization of a vector `v` is a vector of length 1
        and has the same direction with `v`. It's equal to `v/|v|`.

        Args:
            eps (float): a safe-guard value for sqrt, usually 0.

        Example::

            >>> a = ti.Vector([3, 4], ti.f32)
            >>> a.normalized()
            [0.6, 0.8]
        """
        impl.static(
            impl.static_assert(self.m == 1,
                               "normalized() only works on vector"))
        invlen = 1 / (self.norm() + eps)
        return invlen * self

    def transpose(self):
        """Returns the transpose of a matrix.

        Returns:
            :class:`~taichi.Matrix`: The transpose of this matrix.

        Example::

            >>> A = ti.Matrix([[0, 1], [2, 3]])
            >>> A.transpose()
            [[0, 2], [1, 3]]
        """
        from taichi._funcs import _matrix_transpose  # pylint: disable=C0415
        return _matrix_transpose(self)

    @taichi_scope
    def determinant(a):
        """Returns the determinant of this matrix.

        Note:
            The matrix dimension should be less than or equal to 4.

        Returns:
            dtype: The determinant of this matrix.

        Raises:
            Exception: Determinants of matrices with sizes >= 5 are not supported.
        """
        if a.n == 1 and a.m == 1:
            return a(0, 0)
        if a.n == 2 and a.m == 2:
            return a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)
        if a.n == 3 and a.m == 3:
            return a(0, 0) * (a(1, 1) * a(2, 2) - a(2, 1) * a(1, 2)) - a(
                1, 0) * (a(0, 1) * a(2, 2) - a(2, 1) * a(0, 2)) + a(
                    2, 0) * (a(0, 1) * a(1, 2) - a(1, 1) * a(0, 2))
        if a.n == 4 and a.m == 4:
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
        raise Exception(
            "Determinants of matrices with sizes >= 5 are not supported")

    @staticmethod
    def diag(dim, val):
        """Returns a diagonal square matrix with the diagonals filled
        with `val`.

        Args:
            dim (int): the dimension of the wanted square matrix.
            val (TypeVar): value for the diagonal elements.

        Returns:
            :class:`~taichi.Matrix`: The wanted diagonal matrix.

        Example::

            >>> m = ti.Matrix.diag(3, 1)
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        """
        # TODO: need a more systematic way to create a "0" with the right type
        return Matrix([[val if i == j else 0 * val for j in range(dim)]
                       for i in range(dim)])

    def sum(self):
        """Return the sum of all elements.

        Example::

            >>> m = ti.Matrix([[1, 2], [3, 4]])
            >>> m.sum()
            10
        """
        ret = self.entries[0]
        for i in range(1, len(self.entries)):
            ret = ret + self.entries[i]
        return ret

    def norm(self, eps=0):
        """Returns the square root of the sum of the absolute squares
        of its elements.

        Args:
            eps (Number): a safe-guard value for sqrt, usually 0.

        Example::

            >>> a = ti.Vector([3, 4])
            >>> a.norm()
            5

        Returns:
            The square root of the sum of the absolute squares of its elements.
        """
        return ops_mod.sqrt(self.norm_sqr() + eps)

    def norm_inv(self, eps=0):
        """The inverse of the matrix :func:`~taichi.lang.matrix.Matrix.norm`.

        Args:
            eps (float): a safe-guard value for sqrt, usually 0.

        Returns:
            The inverse of the matrix/vector `norm`.
        """
        return ops_mod.rsqrt(self.norm_sqr() + eps)

    def norm_sqr(self):
        """Returns the sum of the absolute squares of its elements."""
        return (self * self).sum()

    def max(self):
        """Returns the maximum element value."""
        return ops_mod.max(*self.entries)

    def min(self):
        """Returns the minimum element value."""
        return ops_mod.min(*self.entries)

    def any(self):
        """Test whether any element not equal zero.

        Returns:
            bool: `True` if any element is not equal zero, `False` otherwise.

        Example::

            >>> v = ti.Vector([0, 0, 1])
            >>> v.any()
            True
        """
        ret = False
        for entry in self.entries:
            ret = ret | ops_mod.cmp_ne(entry, 0)
        return ret & True

    def all(self):
        """Test whether all element not equal zero.

        Returns:
            bool: `True` if all elements are not equal zero, `False` otherwise.

        Example::

            >>> v = ti.Vector([0, 0, 1])
            >>> v.all()
            False
        """
        ret = True
        for entry in self.entries:
            ret = ret & ops_mod.cmp_ne(entry, 0)
        return ret

    @taichi_scope
    def fill(self, val):
        """Fills the matrix with a specified value, must be called
        in Taichi scope.

        Args:
            val (Union[int, float]): Value to fill.

        Example::

            >>> A = ti.Matrix([0, 1, 2, 3])
            >>> A.fill(-1)
            >>> A
            [-1, -1, -1, -1]
        """
        def assign_renamed(x, y):
            return ops_mod.assign(x, y)

        return self._element_wise_writeback_binary(assign_renamed, val)

    @python_scope
    def to_numpy(self, keep_dims=False):
        """Converts this matrix to a numpy array.

        Args:
            keep_dims (bool, optional): Whether to keep the dimension
                after conversion. If set to `False`, the resulting numpy array
                will discard the axis of length one.

        Returns:
            numpy.ndarray: The result numpy array.

        Example::

            >>> A = ti.Matrix([[0], [1], [2], [3]])
            >>> A.to_numpy(keep_dims=False)
            >>> A
            array([0, 1, 2, 3])
        """
        as_vector = self.m == 1 and not keep_dims
        shape_ext = (self.n, ) if as_vector else (self.n, self.m)
        return np.array(self.to_list()).reshape(shape_ext)

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
                1. (arg0: float) -> taichi_python.Expr

            Invoked with: <Taichi 2x1 Matrix>

            So we have to make it happy with a dummy string...
            '''
            return f'<{self.n}x{self.m} ti.Matrix>'
        return str(self.to_numpy())

    def __repr__(self):
        return str(self.to_numpy())

    @staticmethod
    @taichi_scope
    def zero(dt, n, m=None):
        """Constructs a Matrix filled with zeros.

        Args:
            dt (DataType): The desired data type.
            n (int): The first dimension (row) of the matrix.
            m (int, optional): The second dimension (column) of the matrix.

        Returns:
            :class:`~taichi.lang.matrix.Matrix`: A :class:`~taichi.lang.matrix.Matrix` instance filled with zeros.

        """
        if m is None:
            return Vector([ops_mod.cast(0, dt) for _ in range(n)])
        return Matrix([[ops_mod.cast(0, dt) for _ in range(m)]
                       for _ in range(n)])

    @staticmethod
    @taichi_scope
    def one(dt, n, m=None):
        """Constructs a Matrix filled with ones.

        Args:
            dt (DataType): The desired data type.
            n (int): The first dimension (row) of the matrix.
            m (int, optional): The second dimension (column) of the matrix.

        Returns:
            :class:`~taichi.lang.matrix.Matrix`: A :class:`~taichi.lang.matrix.Matrix` instance filled with ones.

        """
        if m is None:
            return Vector([ops_mod.cast(1, dt) for _ in range(n)])
        return Matrix([[ops_mod.cast(1, dt) for _ in range(m)]
                       for _ in range(n)])

    @staticmethod
    @taichi_scope
    def unit(n, i, dt=None):
        """Constructs a n-D vector with the `i`-th entry being equal to one and
        the remaining entries are all zeros.

        Args:
            n (int): The length of the vector.
            i (int): The index of the entry that will be filled with one.
            dt (:mod:`~taichi.types.primitive_types`, optional): The desired data type.

        Returns:
            :class:`~taichi.Matrix`: The returned vector.

        Example::

            >>> A = ti.Matrix.unit(3, 1)
            >>> A
            [0, 1, 0]
        """
        if dt is None:
            dt = int
        assert 0 <= i < n
        return Vector([ops_mod.cast(int(j == i), dt) for j in range(n)])

    @staticmethod
    @taichi_scope
    def identity(dt, n):
        """Constructs an identity Matrix with shape (n, n).

        Args:
            dt (DataType): The desired data type.
            n (int): The number of rows/columns.

        Returns:
            :class:`~taichi.Matrix`: An `n x n` identity matrix.
        """
        return Matrix([[ops_mod.cast(int(i == j), dt) for j in range(n)]
                       for i in range(n)])

    @staticmethod
    def rotation2d(alpha):
        """Returns the matrix representation of the 2D
        anti-clockwise rotation of angle `alpha`. The angle `alpha`
        is in radians.

        Example::

            >>> import math
            >>> ti.Matrix.rotation2d(math.pi/4)
            [[ 0.70710678 -0.70710678]
             [ 0.70710678  0.70710678]]
        """
        warnings.warn(
            "`ti.Matrix.rotation2d(alpha)` is renamed to `ti.Math.rotation2d(ang)`",
            DeprecationWarning)
        return Matrix([[ops_mod.cos(alpha), -ops_mod.sin(alpha)],
                       [ops_mod.sin(alpha),
                        ops_mod.cos(alpha)]])

    @classmethod
    @python_scope
    def field(cls,
              n,
              m,
              dtype,
              shape=None,
              order=None,
              name="",
              offset=None,
              needs_grad=False,
              needs_dual=False,
              layout=Layout.AOS,
              ndim=None):
        """Construct a data container to hold all elements of the Matrix.

        Args:
            n (int): The desired number of rows of the Matrix.
            m (int): The desired number of columns of the Matrix.
            dtype (DataType, optional): The desired data type of the Matrix.
            shape (Union[int, tuple of int], optional): The desired shape of the Matrix.
            order (str, optional): order of the shape laid out in memory.
            name (string, optional): The custom name of the field.
            offset (Union[int, tuple of int], optional): The coordinate offset
                of all elements in a field.
            needs_grad (bool, optional): Whether the Matrix need grad field (reverse mode autodiff).
            needs_dual (bool, optional): Whether the Matrix need dual field (forward mode autodiff).
            layout (Layout, optional): The field layout, either Array Of
                Structure (AOS) or Structure Of Array (SOA).

        Returns:
            :class:`~taichi.Matrix`: A matrix.
        """
        entries = []
        element_dim = ndim if ndim is not None else 2
        if isinstance(dtype, (list, tuple, np.ndarray)):
            # set different dtype for each element in Matrix
            # see #2135
            if m == 1:
                assert len(np.shape(dtype)) == 1 and len(
                    dtype
                ) == n, f'Please set correct dtype list for Vector. The shape of dtype list should be ({n}, ) instead of {np.shape(dtype)}'
                for i in range(n):
                    entries.append(
                        impl.create_field_member(dtype[i],
                                                 name=name,
                                                 needs_grad=needs_grad,
                                                 needs_dual=needs_dual))
            else:
                assert len(np.shape(dtype)) == 2 and len(dtype) == n and len(
                    dtype[0]
                ) == m, f'Please set correct dtype list for Matrix. The shape of dtype list should be ({n}, {m}) instead of {np.shape(dtype)}'
                for i in range(n):
                    for j in range(m):
                        entries.append(
                            impl.create_field_member(dtype[i][j],
                                                     name=name,
                                                     needs_grad=needs_grad,
                                                     needs_dual=needs_dual))
        else:
            for _ in range(n * m):
                entries.append(
                    impl.create_field_member(dtype,
                                             name=name,
                                             needs_grad=needs_grad,
                                             needs_dual=needs_dual))
        entries, entries_grad, entries_dual = zip(*entries)

        entries, entries_grad, entries_dual = MatrixField(
            entries, n, m, element_dim), MatrixField(entries_grad, n, m,
                                                     element_dim), MatrixField(
                                                         entries_grad, n, m,
                                                         element_dim)

        entries._set_grad(entries_grad)
        entries._set_dual(entries_dual)

        impl.get_runtime().matrix_fields.append(entries)

        if shape is None:
            if offset is not None:
                raise TaichiSyntaxError(
                    'shape cannot be None when offset is set')
            if order is not None:
                raise TaichiSyntaxError(
                    'shape cannot be None when order is set')
        else:
            if isinstance(shape, numbers.Number):
                shape = (shape, )
            if isinstance(offset, numbers.Number):
                offset = (offset, )
            dim = len(shape)
            if offset is not None and dim != len(offset):
                raise TaichiSyntaxError(
                    f'The dimensionality of shape and offset must be the same ({dim} != {len(offset)})'
                )
            axis_seq = []
            shape_seq = []
            if order is not None:
                if dim != len(order):
                    raise TaichiSyntaxError(
                        f'The dimensionality of shape and order must be the same ({dim} != {len(order)})'
                    )
                if dim != len(set(order)):
                    raise TaichiSyntaxError(
                        'The axes in order must be different')
                for ch in order:
                    axis = ord(ch) - ord('i')
                    if axis < 0 or axis >= dim:
                        raise TaichiSyntaxError(f'Invalid axis {ch}')
                    axis_seq.append(axis)
                    shape_seq.append(shape[axis])
            else:
                axis_seq = list(range(dim))
                shape_seq = list(shape)
            same_level = order is None
            if layout == Layout.SOA:
                for e in entries._get_field_members():
                    impl._create_snode(axis_seq, shape_seq,
                                       same_level).place(ScalarField(e),
                                                         offset=offset)
                if needs_grad:
                    for e in entries_grad._get_field_members():
                        impl._create_snode(axis_seq, shape_seq,
                                           same_level).place(ScalarField(e),
                                                             offset=offset)
                if needs_dual:
                    for e in entries_dual._get_field_members():
                        impl._create_snode(axis_seq, shape_seq,
                                           same_level).place(ScalarField(e),
                                                             offset=offset)
            else:
                impl._create_snode(axis_seq, shape_seq,
                                   same_level).place(entries, offset=offset)
                if needs_grad:
                    impl._create_snode(axis_seq, shape_seq,
                                       same_level).place(entries_grad,
                                                         offset=offset)
                if needs_dual:
                    impl._create_snode(axis_seq, shape_seq,
                                       same_level).place(entries_dual,
                                                         offset=offset)
        return entries

    @classmethod
    @python_scope
    def ndarray(cls, n, m, dtype, shape, layout=Layout.AOS):
        """Defines a Taichi ndarray with matrix elements.
        This function must be called in Python scope, and after `ti.init` is called.

        Args:
            n (int): Number of rows of the matrix.
            m (int): Number of columns of the matrix.
            dtype (DataType): Data type of each value.
            shape (Union[int, tuple[int]]): Shape of the ndarray.
            layout (Layout, optional): Memory layout, AOS by default.

        Example::

            The code below shows how a Taichi ndarray with matrix elements \
            can be declared and defined::

                >>> x = ti.Matrix.ndarray(4, 5, ti.f32, shape=(16, 8))
        """
        if isinstance(shape, numbers.Number):
            shape = (shape, )
        return MatrixNdarray(n, m, dtype, shape, layout)

    @staticmethod
    def rows(rows):
        """Constructs a matrix by concatenating a list of
        vectors/lists row by row. Must be called in Taichi scope.

        Args:
            rows (List): A list of Vector (1-D Matrix) or a list of list.

        Returns:
            :class:`~taichi.Matrix`: A matrix.

        Example::

            >>> @ti.kernel
            >>> def test():
            >>>     v1 = ti.Vector([1, 2, 3])
            >>>     v2 = ti.Vector([4, 5, 6])
            >>>     m = ti.Matrix.rows([v1, v2])
            >>>     print(m)
            >>>
            >>> test()
            [[1, 2, 3], [4, 5, 6]]
        """
        if isinstance(rows[0], Matrix):
            for row in rows:
                assert row.m == 1, "Inputs must be vectors, i.e. m == 1"
                assert row.n == rows[
                    0].n, "Input vectors must share the same shape"
            # l-value copy:
            return Matrix([[row(i) for i in range(row.n)] for row in rows])
        if isinstance(rows[0], list):
            for row in rows:
                assert len(row) == len(
                    rows[0]), "Input lists share the same shape"
            # l-value copy:
            return Matrix([[x for x in row] for row in rows])
        raise Exception(
            "Cols/rows must be a list of lists, or a list of vectors")

    @staticmethod
    def cols(cols):
        """Constructs a Matrix instance by concatenating Vectors/lists column by column.

        Args:
            cols (List): A list of Vector (1-D Matrix) or a list of list.

        Returns:
            :class:`~taichi.Matrix`: A matrix.

        Example::

            >>> @ti.kernel
            >>> def test():
            >>>     v1 = ti.Vector([1, 2, 3])
            >>>     v2 = ti.Vector([4, 5, 6])
            >>>     m = ti.Matrix.cols([v1, v2])
            >>>     print(m)
            >>>
            >>> test()
            [[1, 4], [2, 5], [3, 6]]
        """
        return Matrix.rows(cols).transpose()

    def __hash__(self):
        # TODO: refactor KernelTemplateMapper
        # If not, we get `unhashable type: Matrix` when
        # using matrices as template arguments.
        return id(self)

    def dot(self, other):
        """Performs the dot product of two vectors.

        To call this method, both multiplicatives must be vectors.

        Args:
            other (:class:`~taichi.Matrix`): The input Vector.

        Returns:
            DataType: The dot product result (scalar) of the two Vectors.

        Example::

            >>> v1 = ti.Vector([1, 2, 3])
            >>> v2 = ti.Vector([3, 4, 5])
            >>> v1.dot(v2)
            26
        """
        impl.static(
            impl.static_assert(self.m == 1, "lhs for dot is not a vector"))
        impl.static(
            impl.static_assert(other.m == 1, "rhs for dot is not a vector"))
        return (self * other).sum()

    def _cross3d(self, other):
        from taichi._funcs import _matrix_cross3d  # pylint: disable=C0415
        return _matrix_cross3d(self, other)

    def _cross2d(self, other):
        from taichi._funcs import _matrix_cross2d  # pylint: disable=C0415
        return _matrix_cross2d(self, other)

    def cross(self, other):
        """Performs the cross product with the input vector (1-D Matrix).

        Both two vectors must have the same dimension <= 3.

        For two 2d vectors (x1, y1) and (x2, y2), the return value is the
        scalar `x1*y2 - x2*y1`.

        For two 3d vectors `v` and `w`, the return value is the 3d vector
        `v x w`.

        Args:
            other (:class:`~taichi.Matrix`): The input Vector.

        Returns:
            :class:`~taichi.Matrix`: The cross product of the two Vectors.
        """
        if self.n == 3 and self.m == 1 and other.n == 3 and other.m == 1:
            return self._cross3d(other)

        if self.n == 2 and self.m == 1 and other.n == 2 and other.m == 1:
            return self._cross2d(other)

        raise ValueError(
            "Cross product is only supported between pairs of 2D/3D vectors")

    def outer_product(self, other):
        """Performs the outer product with the input Vector (1-D Matrix).

        The outer_product of two vectors `v = (x1, x2, ..., xn)`,
        `w = (y1, y2, ..., yn)` is a `n` times `n` square matrix, and its `(i, j)`
        entry is equal to `xi*yj`.

        Args:
            other (:class:`~taichi.Matrix`): The input Vector.

        Returns:
            :class:`~taichi.Matrix`: The outer product of the two Vectors.
        """
        from taichi._funcs import \
            _matrix_outer_product  # pylint: disable=C0415
        return _matrix_outer_product(self, other)


class Vector(Matrix):
    def __init__(self, arr, dt=None, **kwargs):
        """Constructs a vector from given array.

        A vector is an instance of a 2-D matrix with the second dimension being equal to 1.

        Args:
            arr (Union[list, tuple, np.ndarray]): The initial values of the Vector.
            dt (:mod:`~taichi.types.primitive_types`): data type of the vector.

        Returns:
            :class:`~taichi.Matrix`: A vector instance.
        Example::
            >>> u = ti.Vector([1, 2])
            >>> print(u.m, u.n)  # verify a vector is a matrix of shape (n, 1)
            2 1
            >>> v = ti.Vector([3, 4])
            >>> u + v
            [4 6]
        """
        super().__init__(arr, dt=dt, **kwargs)

    @classmethod
    def field(cls, n, dtype, *args, **kwargs):
        """ti.Vector.field"""
        ndim = kwargs.get("ndim", 1)
        assert ndim == 1
        kwargs["ndim"] = 1
        return super().field(n, 1, dtype, *args, **kwargs)

    @classmethod
    @python_scope
    def ndarray(cls, n, dtype, shape, layout=Layout.AOS):
        """Defines a Taichi ndarray with vector elements.

        Args:
            n (int): Size of the vector.
            dtype (DataType): Data type of each value.
            shape (Union[int, tuple[int]]): Shape of the ndarray.
            layout (Layout, optional): Memory layout, AOS by default.

        Example:
            The code below shows how a Taichi ndarray with vector elements can be declared and defined::

                >>> x = ti.Vector.ndarray(3, ti.f32, shape=(16, 8))
        """
        if isinstance(shape, numbers.Number):
            shape = (shape, )
        return VectorNdarray(n, dtype, shape, layout)


class _IntermediateMatrix(Matrix):
    """Intermediate matrix class for compiler internal use only.

    Args:
        n (int): Number of rows of the matrix.
        m (int): Number of columns of the matrix.
        entries (List[Expr]): All entries of the matrix.
    """
    def __init__(self, n, m, entries, ndim=None):
        assert isinstance(entries, list)
        assert n * m == len(entries), "Number of entries doesn't match n * m"
        self.n = n
        self.m = m
        if ndim is not None:
            self.ndim = ndim
        else:
            if len(entries) == 0:
                self.ndim = 0
            else:
                self.ndim = 2 if isinstance(entries[0], Iterable) else 1
        self._impl = _TiScopeMatrixImpl(m,
                                        n,
                                        entries,
                                        local_tensor_proxy=None,
                                        dynamic_index_stride=None)


class _MatrixFieldElement(_IntermediateMatrix):
    """Matrix field element class for compiler internal use only.

    Args:
        field (MatrixField): The matrix field.
        indices (taichi_python.ExprGroup): Indices of the element.
    """
    def __init__(self, field, indices):
        super().__init__(
            field.n,
            field.m, [
                expr.Expr(
                    ti_python_core.subscript(
                        e.ptr, indices,
                        impl.get_runtime().get_current_src_info()))
                for e in field._get_field_members()
            ],
            ndim=getattr(field, "ndim", 2))
        self._impl.dynamic_index_stride = field.dynamic_index_stride


class MatrixField(Field):
    """Taichi matrix field with SNode implementation.

    Args:
        vars (List[Expr]): Field members.
        n (Int): Number of rows.
        m (Int): Number of columns.
        ndim (Int): Number of dimensions; forced reshape if given.
    """
    def __init__(self, _vars, n, m, ndim=2):
        assert len(_vars) == n * m
        assert ndim in (0, 1, 2)
        super().__init__(_vars)
        self.n = n
        self.m = m
        self.ndim = ndim
        self.dynamic_index_stride = None

    def get_scalar_field(self, *indices):
        """Creates a ScalarField using a specific field member.

        Args:
            indices (Tuple[Int]): Specified indices of the field member.

        Returns:
            ScalarField: The result ScalarField.
        """
        assert len(indices) in [1, 2]
        i = indices[0]
        j = 0 if len(indices) == 1 else indices[1]
        return ScalarField(self.vars[i * self.m + j])

    def _calc_dynamic_index_stride(self):
        # Algorithm: https://github.com/taichi-dev/taichi/issues/3810
        paths = [ScalarField(var).snode._path_from_root() for var in self.vars]
        num_members = len(paths)
        if num_members == 1:
            self.dynamic_index_stride = 0
            return
        length = len(paths[0])
        if any(
                len(path) != length or ti_python_core.is_quant(path[length -
                                                                    1]._dtype)
                for path in paths):
            return
        for i in range(length):
            if any(path[i] != paths[0][i] for path in paths):
                depth_below_lca = i
                break
        for i in range(depth_below_lca, length - 1):
            if any(path[i].ptr.type != ti_python_core.SNodeType.dense
                   or path[i]._cell_size_bytes != paths[0][i]._cell_size_bytes
                   or path[i + 1]._offset_bytes_in_parent_cell != paths[0][
                       i + 1]._offset_bytes_in_parent_cell for path in paths):
                return
        stride = paths[1][depth_below_lca]._offset_bytes_in_parent_cell - \
            paths[0][depth_below_lca]._offset_bytes_in_parent_cell
        for i in range(2, num_members):
            if stride != paths[i][depth_below_lca]._offset_bytes_in_parent_cell \
                    - paths[i - 1][depth_below_lca]._offset_bytes_in_parent_cell:
                return
        self.dynamic_index_stride = stride

    def fill(self, val):
        """Fills this matrix field with specified values.

        Args:
            val (Union[Number, List, Tuple, Matrix]): Values to fill,
                should have consistent dimension consistent with `self`.
        """
        if isinstance(val, numbers.Number):
            val = tuple(
                [tuple([val for _ in range(self.m)]) for _ in range(self.n)])
        elif isinstance(val,
                        (list, tuple)) and isinstance(val[0], numbers.Number):
            assert self.m == 1
            val = tuple([(v, ) for v in val])
        elif isinstance(val, Matrix):
            val_tuple = []
            for i in range(val.n):
                row = []
                for j in range(val.m):
                    row.append(val(i, j))
                row = tuple(row)
                val_tuple.append(row)
            val = tuple(val_tuple)
        assert len(val) == self.n
        assert len(val[0]) == self.m

        if in_python_scope():
            from taichi._kernels import fill_matrix  # pylint: disable=C0415
            fill_matrix(self, val)
        else:
            from taichi._funcs import \
                field_fill_taichi_scope  # pylint: disable=C0415
            field_fill_taichi_scope(self, val)

    @python_scope
    def to_numpy(self, keep_dims=False, dtype=None):
        """Converts the field instance to a NumPy array.

        Args:
            keep_dims (bool, optional): Whether to keep the dimension after conversion.
                When keep_dims=True, on an n-D matrix field, the numpy array always has n+2 dims, even for 1x1, 1xn, nx1 matrix fields.
                When keep_dims=False, the resulting numpy array should skip the matrix dims with size 1.
                For example, a 4x1 or 1x4 matrix field with 5x6x7 elements results in an array of shape 5x6x7x4.
            dtype (DataType, optional): The desired data type of returned numpy array.

        Returns:
            numpy.ndarray: The result NumPy array.
        """
        if dtype is None:
            dtype = to_numpy_type(self.dtype)
        as_vector = self.m == 1 and not keep_dims
        shape_ext = (self.n, ) if as_vector else (self.n, self.m)
        arr = np.zeros(self.shape + shape_ext, dtype=dtype)
        from taichi._kernels import matrix_to_ext_arr  # pylint: disable=C0415
        matrix_to_ext_arr(self, arr, as_vector)
        runtime_ops.sync()
        return arr

    def to_torch(self, device=None, keep_dims=False):
        """Converts the field instance to a PyTorch tensor.

        Args:
            device (torch.device, optional): The desired device of returned tensor.
            keep_dims (bool, optional): Whether to keep the dimension after conversion.
                See :meth:`~taichi.lang.field.MatrixField.to_numpy` for more detailed explanation.

        Returns:
            torch.tensor: The result torch tensor.
        """
        import torch  # pylint: disable=C0415
        as_vector = self.m == 1 and not keep_dims
        shape_ext = (self.n, ) if as_vector else (self.n, self.m)
        # pylint: disable=E1101
        arr = torch.empty(self.shape + shape_ext,
                          dtype=to_pytorch_type(self.dtype),
                          device=device)
        from taichi._kernels import matrix_to_ext_arr  # pylint: disable=C0415
        matrix_to_ext_arr(self, arr, as_vector)
        runtime_ops.sync()
        return arr

    def to_paddle(self, place=None, keep_dims=False):
        """Converts the field instance to a Paddle tensor.

        Args:
            place (paddle.CPUPlace()/CUDAPlace(n), optional): The desired place of returned tensor.
            keep_dims (bool, optional): Whether to keep the dimension after conversion.
                See :meth:`~taichi.lang.field.MatrixField.to_numpy` for more detailed explanation.

        Returns:
            paddle.Tensor: The result paddle tensor.
        """
        import paddle  # pylint: disable=C0415
        as_vector = self.m == 1 and not keep_dims and self.ndim == 1
        shape_ext = (self.n, ) if as_vector else (self.n, self.m)
        # pylint: disable=E1101
        # paddle.empty() doesn't support argument `place``
        arr = paddle.to_tensor(paddle.empty(self.shape + shape_ext,
                                            to_paddle_type(self.dtype)),
                               place=place)
        from taichi._kernels import matrix_to_ext_arr  # pylint: disable=C0415
        matrix_to_ext_arr(self, arr, as_vector)
        runtime_ops.sync()
        return arr

    @python_scope
    def from_numpy(self, arr):
        """Copies an `numpy.ndarray` into this field.

        Example::

            >>> m = ti.Matrix.field(2, 2, ti.f32, shape=(3, 3))
            >>> arr = numpp.ones((3, 3, 2, 2))
            >>> m.from_numpy(arr)
        """
        if len(arr.shape) == len(self.shape) + 1:
            as_vector = True
            assert self.m == 1, "This is not a vector field"
        else:
            as_vector = False
            assert len(arr.shape) == len(self.shape) + 2
        dim_ext = 1 if as_vector else 2
        assert len(arr.shape) == len(self.shape) + dim_ext
        from taichi._kernels import ext_arr_to_matrix  # pylint: disable=C0415
        ext_arr_to_matrix(arr, self, as_vector)
        runtime_ops.sync()

    @python_scope
    def __setitem__(self, key, value):
        self._initialize_host_accessors()
        self[key]._set_entries(value)

    @python_scope
    def __getitem__(self, key):
        self._initialize_host_accessors()
        key = self._pad_key(key)
        _host_access = self._host_access(key)
        return Matrix([[_host_access[i * self.m + j] for j in range(self.m)]
                       for i in range(self.n)],
                      ndim=self.ndim)

    def __repr__(self):
        # make interactive shell happy, prevent materialization
        return f'<{self.n}x{self.m} ti.Matrix.field>'


class MatrixType(CompoundType):
    def __init__(self, n, m, ndim, dtype):
        self.n = n
        self.m = m
        self.ndim = ndim
        self.dtype = cook_dtype(dtype)

    def __call__(self, *args):
        """Return a matrix matching the shape and dtype.

        This function will try to convert the input to a `n x m` matrix, with n, m being
        the number of rows/cols of this matrix type.

        Example::

            >>> mat4x3 = MatrixType(4, 3, float)
            >>> mat2x6 = MatrixType(2, 6, float)

            Create from n x m scalars, of a 1d list of n x m scalars:

                >>> m = mat4x3([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
                >>> m = mat4x3(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

            Create from n vectors/lists, with each one of dimension m:

                >>> m = mat4x3([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])

            Create from a single scalar

                >>> m = mat4x3(1)

            Create from another 2d list/matrix, as long as they have the same number of entries

                >>> m = mat4x3([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
                >>> m = mat4x3(m)
                >>> k = mat2x6(m)

        """
        if len(args) == 0:
            raise TaichiSyntaxError(
                "Custom type instances need to be created with an initial value."
            )
        if len(args) == 1:
            # initialize by a single scalar, e.g. matnxm(1)
            if isinstance(args[0], (numbers.Number, expr.Expr)):
                return self.filled_with_scalar(args[0])
            args = args[0]
        # collect all input entries to a 1d list and then reshape
        # this is mostly for glsl style like vec4(v.xyz, 1.)
        entries = []
        for x in args:
            if isinstance(x, (list, tuple)):
                entries += x
            elif isinstance(x, np.ndarray):
                entries += list(x.ravel())
            elif isinstance(x, Matrix):
                entries += x.entries
            else:
                entries.append(x)

        if len(entries) != self.m * self.n:
            raise TaichiSyntaxError(
                f"Incompatible arguments for the custom vector/matrix type: ({self.n}, {self.m}), ({len(entries)})"
            )
        entries = [[entries[k * self.m + i] for i in range(self.m)]
                   for k in range(self.n)]

        #  type cast
        return self.cast(Matrix(entries, dt=self.dtype, ndim=self.ndim))

    def cast(self, mat):
        if in_python_scope():
            return Matrix([[
                int(mat(i, j)) if self.dtype in primitive_types.integer_types
                else float(mat(i, j)) for j in range(self.m)
            ] for i in range(self.n)],
                          ndim=self.ndim)
        return mat.cast(self.dtype)

    def filled_with_scalar(self, value):
        return self.cast(
            Matrix([[value for _ in range(self.m)] for _ in range(self.n)],
                   ndim=self.ndim))

    def field(self, **kwargs):
        assert kwargs.get("ndim", self.ndim) == self.ndim
        kwargs.update({"ndim": self.ndim})
        return Matrix.field(self.n, self.m, dtype=self.dtype, **kwargs)


class VectorType(MatrixType):
    def __init__(self, n, dtype):
        super().__init__(n, 1, 1, dtype)

    def __call__(self, *args):
        """Return a vector matching the shape and dtype.

        This function will try to convert the input to a `n`-component vector.

        Example::

            >>> vec3 = VectorType(3, float)

            Create from n scalars:

                >>> v = vec3(1, 2, 3)

            Create from a list/tuple of n scalars:

                >>> v = vec3([1, 2, 3])

            Create from a single scalar

                >>> v = vec3(1)

        """
        if len(args) == 0:
            raise TaichiSyntaxError(
                "Custom type instances need to be created with an initial value."
            )
        if len(args) == 1:
            # initialize by a single scalar, e.g. matnxm(1)
            if isinstance(args[0], (numbers.Number, expr.Expr)):
                return self.filled_with_scalar(args[0])
            args = args[0]
        # collect all input entries to a 1d list and then reshape
        # this is mostly for glsl style like vec4(v.xyz, 1.)
        entries = []
        for x in args:
            if isinstance(x, (list, tuple)):
                entries += x
            elif isinstance(x, np.ndarray):
                entries += list(x.ravel())
            elif isinstance(x, Matrix):
                entries += x.entries
            else:
                entries.append(x)

        if len(entries) != self.n:
            raise TaichiSyntaxError(
                f"Incompatible arguments for the custom vector type: ({self.n}), ({len(entries)})"
            )

        #  type cast
        return self.cast(Vector(entries, dt=self.dtype))

    def cast(self, vec):
        if in_python_scope():
            return Vector([
                int(vec(i)) if self.dtype in primitive_types.integer_types else
                float(vec(i)) for i in range(self.n)
            ])
        return vec.cast(self.dtype)

    def filled_with_scalar(self, value):
        return self.cast(Vector([value for _ in range(self.n)]))

    def field(self, **kwargs):
        return Vector.field(self.n, dtype=self.dtype, **kwargs)


class MatrixNdarray(Ndarray):
    """Taichi ndarray with matrix elements.

    Args:
        n (int): Number of rows of the matrix.
        m (int): Number of columns of the matrix.
        dtype (DataType): Data type of each value.
        shape (Union[int, tuple[int]]): Shape of the ndarray.
        layout (Layout): Memory layout.

    Example::

        >>> arr = ti.MatrixNdarray(2, 2, ti.f32, shape=(3, 3), layout=Layout.SOA)
    """
    def __init__(self, n, m, dtype, shape, layout):
        self.n = n
        self.m = m
        super().__init__()
        # TODO(zhanlue): remove self.dtype and migrate its usages to element_type
        self.dtype = cook_dtype(dtype)

        self.layout = layout
        self.shape = tuple(shape)
        self.element_type = TensorType((self.n, self.m), dtype)
        # TODO: we should pass in element_type, shape, layout instead.
        self.arr = impl.get_runtime().prog.create_ndarray(
            cook_dtype(self.element_type.ptr), shape, layout)

    @property
    def element_shape(self):
        """Returns the shape of each element (a 2D matrix) in this ndarray.

        Example::

            >>> arr = ti.MatrixNdarray(2, 2, ti.f32, shape=(3, 3), layout=Layout.SOA)
            >>> arr.element_shape
            (2, 2)
        """
        return tuple(self.arr.element_shape())

    @python_scope
    def __setitem__(self, key, value):
        if not isinstance(value, (list, tuple)):
            value = list(value)
        if not isinstance(value[0], (list, tuple)):
            value = [[i] for i in value]
        for i in range(self.n):
            for j in range(self.m):
                self[key][i, j] = value[i][j]

    @python_scope
    def __getitem__(self, key):
        key = () if key is None else (
            key, ) if isinstance(key, numbers.Number) else tuple(key)
        return Matrix(
            [[NdarrayHostAccess(self, key, (i, j)) for j in range(self.m)]
             for i in range(self.n)])

    @python_scope
    def to_numpy(self):
        """Converts this ndarray to a `numpy.ndarray`.

        Example::

            >>> arr = ti.MatrixNdarray(2, 2, ti.f32, shape=(2, 1), layout=Layout.SOA)
            >>> arr.to_numpy()
            [[[[0. 0.]
               [0. 0.]]]

             [[[0. 0.]
               [0. 0.]]]]
        """
        return self._ndarray_matrix_to_numpy(self.layout, as_vector=0)

    @python_scope
    def from_numpy(self, arr):
        """Copies the data of a `numpy.ndarray` into this array.

        Example::

            >>> m = ti.MatrixNdarray(2, 2, ti.f32, shape=(2, 1), layout=0)
            >>> arr = np.ones((2, 1, 2, 2))
            >>> m.from_numpy(arr)
        """
        self._ndarray_matrix_from_numpy(arr, self.layout, as_vector=0)

    @python_scope
    def __deepcopy__(self, memo=None):
        ret_arr = MatrixNdarray(self.n, self.m, self.dtype, self.shape,
                                self.layout)
        ret_arr.copy_from(self)
        return ret_arr

    @python_scope
    def _fill_by_kernel(self, val):
        from taichi._kernels import \
            fill_ndarray_matrix  # pylint: disable=C0415
        fill_ndarray_matrix(self, val)

    @python_scope
    def __repr__(self):
        return f'<{self.n}x{self.m} {self.layout} ti.Matrix.ndarray>'


class VectorNdarray(Ndarray):
    """Taichi ndarray with vector elements.

    Args:
        n (int): Size of the vector.
        dtype (DataType): Data type of each value.
        shape (Tuple[int]): Shape of the ndarray.
        layout (Layout): Memory layout.

    Example::

        >>> a = ti.VectorNdarray(3, ti.f32, (3, 3), layout=Layout.SOA)
    """
    def __init__(self, n, dtype, shape, layout):
        self.n = n
        super().__init__()
        # TODO(zhanlue): remove self.dtype and migrate its usages to element_type
        self.dtype = cook_dtype(dtype)

        self.layout = layout
        self.shape = tuple(shape)
        self.element_type = TensorType((n, ), self.dtype)
        self.arr = impl.get_runtime().prog.create_ndarray(
            cook_dtype(self.element_type.ptr), shape, layout)

    @property
    def element_shape(self):
        """Gets the dimension of the vector of this ndarray.

        Example::

            >>> a = ti.VectorNdarray(3, ti.f32, (3, 3), layout=Layout.SOA)
            >>> a.element_shape
            (3,)
        """
        return tuple(self.arr.element_shape())

    @python_scope
    def __setitem__(self, key, value):
        if not isinstance(value, (list, tuple)):
            value = list(value)
        for i in range(self.n):
            self[key][i] = value[i]

    @python_scope
    def __getitem__(self, key):
        key = () if key is None else (
            key, ) if isinstance(key, numbers.Number) else tuple(key)
        return Vector(
            [NdarrayHostAccess(self, key, (i, )) for i in range(self.n)])

    @python_scope
    def to_numpy(self):
        """Converts this vector ndarray to a `numpy.ndarray`.

        Example::

            >>> a = ti.VectorNdarray(3, ti.f32, (2, 2), layout=Layout.SOA)
            >>> a.to_numpy()
            array([[[0., 0., 0.],
                    [0., 0., 0.]],

                   [[0., 0., 0.],
                    [0., 0., 0.]]], dtype=float32)
        """
        return self._ndarray_matrix_to_numpy(self.layout, as_vector=1)

    @python_scope
    def from_numpy(self, arr):
        """Copies the data from a `numpy.ndarray` into this ndarray.

        The shape and data type of `arr` must match this ndarray.

        Example::

            >>> import numpy as np
            >>> a = ti.VectorNdarray(3, ti.f32, (2, 2), 0)
            >>> b = np.ones((2, 2, 3), dtype=np.float32)
            >>> a.from_numpy(b)
        """
        self._ndarray_matrix_from_numpy(arr, self.layout, as_vector=1)

    @python_scope
    def __deepcopy__(self, memo=None):
        ret_arr = VectorNdarray(self.n, self.dtype, self.shape, self.layout)
        ret_arr.copy_from(self)
        return ret_arr

    @python_scope
    def _fill_by_kernel(self, val):
        from taichi._kernels import \
            fill_ndarray_matrix  # pylint: disable=C0415
        fill_ndarray_matrix(self, val)

    @python_scope
    def __repr__(self):
        return f'<{self.n} {self.layout} ti.Vector.ndarray>'


__all__ = ["Matrix", "Vector", "MatrixField", "MatrixNdarray", "VectorNdarray"]
