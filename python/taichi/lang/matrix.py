import functools
import numbers
from collections.abc import Iterable
from itertools import product

import numpy as np
from taichi._lib import core as ti_python_core
from taichi.lang import expr, impl
from taichi.lang import ops as ops_mod
from taichi.lang import runtime_ops
from taichi.lang._ndarray import Ndarray, NdarrayHostAccess
from taichi.lang.common_ops import TaichiOperations
from taichi.lang.enums import Layout
from taichi.lang.exception import (
    TaichiRuntimeError,
    TaichiRuntimeTypeError,
    TaichiSyntaxError,
    TaichiTypeError,
)
from taichi.lang.field import Field, ScalarField, SNodeHostAccess
from taichi.lang.util import (
    cook_dtype,
    get_traceback,
    in_python_scope,
    python_scope,
    taichi_scope,
    to_numpy_type,
    to_paddle_type,
    to_pytorch_type,
    warning,
)
from taichi.types import primitive_types
from taichi.types.compound_types import CompoundType
from taichi.types.utils import is_signed
from taichi._lib.utils import ti_python_core as _ti_python_core

_type_factory = _ti_python_core.get_type_factory_instance()


def _generate_swizzle_patterns(key_group: str, required_length=4):
    """Generate vector swizzle patterns from a given set of characters.

    Example:

        For `key_group=xyzw` and `required_length=4`, this function will return a
        list consists of all possible strings (no repeats) in characters
        `x`, `y`, `z`, `w` and of length<=4:
        [`x`, `y`, `z`, `w`, `xx`, `xy`, `yx`, ..., `xxxx`, `xxxy`, `xyzw`, ...]
        The length of the list will be 4 + 4x4 + 4x4x4 + 4x4x4x4 = 340.
    """
    result = []
    for k in range(1, required_length + 1):
        result.extend(product(key_group, repeat=k))
    result = ["".join(pat) for pat in result]
    return result


def _gen_swizzles(cls):
    # https://www.khronos.org/opengl/wiki/Data_Type_(GLSL)#Swizzling
    KEYGROUP_SET = ["xyzw", "rgba", "stpq"]
    cls._swizzle_to_keygroup = {}
    cls._keygroup_to_checker = {}

    def make_valid_attribs_checker(key_group):
        def check(instance, pattern):
            valid_attribs = set(key_group[: instance.n])
            pattern_set = set(pattern)
            diff = pattern_set - valid_attribs
            if len(diff):
                valid_attribs = tuple(sorted(valid_attribs))
                pattern = tuple(pattern)
                raise TaichiSyntaxError(f"vec{instance.n} only has " f"attributes={valid_attribs}, got={pattern}")

        return check

    for key_group in KEYGROUP_SET:
        cls._keygroup_to_checker[key_group] = make_valid_attribs_checker(key_group)
        for index, attr in enumerate(key_group):

            def gen_property(attr, attr_idx, key_group):
                checker = cls._keygroup_to_checker[key_group]

                def prop_getter(instance):
                    checker(instance, attr)
                    return instance[attr_idx]

                @python_scope
                def prop_setter(instance, value):
                    checker(instance, attr)
                    instance[attr_idx] = value

                return property(prop_getter, prop_setter)

            prop = gen_property(attr, index, key_group)
            setattr(cls, attr, prop)
            cls._swizzle_to_keygroup[attr] = key_group

    for key_group in KEYGROUP_SET:
        sw_patterns = _generate_swizzle_patterns(key_group, required_length=4)
        # len=1 accessors are handled specially above
        sw_patterns = filter(lambda p: len(p) > 1, sw_patterns)
        for prop_key in sw_patterns:
            # Create a function for value capturing
            def gen_property(pattern, key_group):
                checker = cls._keygroup_to_checker[key_group]

                def prop_getter(instance):
                    checker(instance, pattern)
                    res = []
                    for ch in pattern:
                        res.append(instance[key_group.index(ch)])
                    return Vector(res)

                @python_scope
                def prop_setter(instance, value):
                    if len(pattern) != len(value):
                        raise TaichiRuntimeError(f"value len does not match the swizzle pattern={pattern}")
                    checker(instance, pattern)
                    for ch, val in zip(pattern, value):
                        instance[key_group.index(ch)] = val

                prop = property(prop_getter, prop_setter)
                return prop

            prop = gen_property(prop_key, key_group)
            setattr(cls, prop_key, prop)
            cls._swizzle_to_keygroup[prop_key] = key_group
    return cls


def _infer_entry_dt(entry):
    if isinstance(entry, (int, np.integer)):
        return impl.get_runtime().default_ip
    if isinstance(entry, (float, np.floating)):
        return impl.get_runtime().default_fp
    if isinstance(entry, expr.Expr):
        dt = entry.ptr.get_rvalue_type()
        if dt == ti_python_core.DataType_unknown:
            raise TaichiTypeError("Element type of the matrix cannot be inferred. Please set dt instead for now.")
        return dt
    raise TaichiTypeError("Element type of the matrix is invalid.")


def _infer_array_dt(arr):
    assert len(arr) > 0
    return functools.reduce(ti_python_core.promoted_type, map(_infer_entry_dt, arr))


def make_matrix_with_shape(arr, shape, dt):
    return expr.Expr(
        impl.get_runtime()
        .compiling_callable.ast_builder()
        .make_matrix_expr(
            shape,
            dt,
            [expr.Expr(elt).ptr for elt in arr],
            ti_python_core.DebugInfo(impl.get_runtime().get_current_src_info()),
        )
    )


def make_matrix(arr, dt=None):
    if len(arr) == 0:
        # the only usage of an empty vector is to serve as field indices
        shape = [0]
        dt = primitive_types.i32
    else:
        if isinstance(arr[0], Iterable):  # matrix
            shape = [len(arr), len(arr[0])]
            arr = [elt for row in arr for elt in row]
        else:  # vector
            shape = [len(arr)]
        if dt is None:
            dt = _infer_array_dt(arr)
        else:
            dt = cook_dtype(dt)
    return expr.Expr(
        impl.get_runtime()
        .compiling_callable.ast_builder()
        .make_matrix_expr(
            shape,
            dt,
            [expr.Expr(elt).ptr for elt in arr],
            ti_python_core.DebugInfo(impl.get_runtime().get_current_src_info()),
        )
    )


def _read_host_access(x):
    if isinstance(x, SNodeHostAccess):
        return x.accessor.getter(*x.key)
    assert isinstance(x, NdarrayHostAccess)
    return x.getter()


def _write_host_access(x, value):
    if isinstance(x, SNodeHostAccess):
        x.accessor.setter(value, *x.key)
    else:
        assert isinstance(x, NdarrayHostAccess)
        x.setter(value)


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
    _is_matrix_class = True
    __array_priority__ = 1000

    def __init__(self, arr, dt=None):
        if not isinstance(arr, (list, tuple, np.ndarray)):
            raise TaichiTypeError("An Matrix/Vector can only be initialized with an array-like object")
        if len(arr) == 0:
            self.ndim = 0
            self.n, self.m = 0, 0
            self.entries = np.array([])
            self.is_host_access = False
        elif isinstance(arr[0], Matrix):
            raise Exception("cols/rows required when using list of vectors")
        elif isinstance(arr[0], Iterable):  # matrix
            self.ndim = 2
            self.n, self.m = len(arr), len(arr[0])
            if isinstance(arr[0][0], (SNodeHostAccess, NdarrayHostAccess)):
                self.entries = arr
                self.is_host_access = True
            else:
                self.entries = np.array(arr, None if dt is None else to_numpy_type(dt))
                self.is_host_access = False
        else:  # vector
            self.ndim = 1
            self.n, self.m = len(arr), 1
            if isinstance(arr[0], (SNodeHostAccess, NdarrayHostAccess)):
                self.entries = arr
                self.is_host_access = True
            else:
                self.entries = np.array(arr, None if dt is None else to_numpy_type(dt))
                self.is_host_access = False

        if self.n * self.m > 32:
            warning(
                f"Taichi matrices/vectors with {self.n}x{self.m} > 32 entries are not suggested."
                " Matrices/vectors will be automatically unrolled at compile-time for performance."
                " So the compilation time could be extremely long if the matrix size is too big."
                " You may use a field to store a large matrix like this, e.g.:\n"
                f"    x = ti.field(ti.f32, ({self.n}, {self.m})).\n"
                " See https://docs.taichi-lang.org/docs/field#matrix-size"
                " for more details.",
                UserWarning,
                stacklevel=2,
            )

    def get_shape(self):
        if self.ndim == 1:
            return (self.n,)
        if self.ndim == 2:
            return (self.n, self.m)
        return None

    def __matmul__(self, other):
        """Matrix-matrix or matrix-vector multiply.

        Args:
            other (Union[Matrix, Vector]): a matrix or a vector.

        Returns:
            The matrix-matrix product or matrix-vector product.

        """
        from taichi.lang import matrix_ops  # pylint: disable=C0415

        return matrix_ops.matmul(self, other)

    # host access & python scope operation
    def __len__(self):
        """Get the length of each row of a matrix"""
        # TODO: When this is a vector, should return its dimension?
        return self.n

    def __iter__(self):
        if self.ndim == 1:
            return (self[i] for i in range(self.n))
        return ([self[i, j] for j in range(self.m)] for i in range(self.n))

    def __getitem__(self, indices):
        """Access to the element at the given indices in a matrix.

        Args:
            indices (Sequence[Expr]): the indices of the element.

        Returns:
            The value of the element at a specific position of a matrix.

        """
        entry = self._get_entry(indices)
        if self.is_host_access:
            return _read_host_access(entry)
        return entry

    @python_scope
    def __setitem__(self, indices, item):
        """Set the element value at the given indices in a matrix.

        Args:
            indices (Sequence[Expr]): the indices of a element.

        """
        if self.is_host_access:
            entry = self._get_entry(indices)
            _write_host_access(entry, item)
        else:
            if not isinstance(indices, (list, tuple)):
                indices = [indices]
            assert len(indices) in [1, 2]
            assert len(indices) == self.ndim, f"Expected {self.ndim} indices, got {len(indices)}"
            if self.ndim == 1:
                self.entries[indices[0]] = item
            else:
                self.entries[indices[0]][indices[1]] = item

    def _get_entry(self, indices):
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
        assert len(indices) in [1, 2]
        assert len(indices) == self.ndim, f"Expected {self.ndim} indices, got {len(indices)}"
        if self.ndim == 1:
            return self.entries[indices[0]]
        return self.entries[indices[0]][indices[1]]

    def _get_slice(self, a, b):
        if isinstance(a, slice):
            a = range(a.start or 0, a.stop or self.n, a.step or 1)
        if isinstance(b, slice):
            b = range(b.start or 0, b.stop or self.m, b.step or 1)
        if isinstance(a, range) and isinstance(b, range):
            return Matrix([[self._get_entry(i, j) for j in b] for i in a])
        if isinstance(a, range):  # b is not range
            return Vector([self._get_entry(i, b) for i in a])
        # a is not range while b is range
        return Vector([self._get_entry(a, j) for j in b])

    @python_scope
    def _set_entries(self, value):
        if isinstance(value, Matrix):
            value = value.to_list()
        if self.is_host_access:
            if self.ndim == 1:
                for i in range(self.n):
                    _write_host_access(self.entries[i], value[i])
            else:
                for i in range(self.n):
                    for j in range(self.m):
                        _write_host_access(self.entries[i][j], value[i][j])
        else:
            if self.ndim == 1:
                for i in range(self.n):
                    self.entries[i] = value[i]
            else:
                for i in range(self.n):
                    for j in range(self.m):
                        self.entries[i][j] = value[i][j]

    @property
    def _members(self):
        return self.entries

    def to_list(self):
        """Return this matrix as a 1D `list`.

        This is similar to `numpy.ndarray`'s `flatten` and `ravel` methods,
        the difference is that this function always returns a new list.
        """
        if self.is_host_access:
            if self.ndim == 1:
                return [_read_host_access(self.entries[i]) for i in range(self.n)]
            assert self.ndim == 2
            return [[_read_host_access(self.entries[i][j]) for j in range(self.m)] for i in range(self.n)]
        return self.entries.tolist()

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
        if self.ndim == 1:
            return Vector([ops_mod.cast(self[i], dtype) for i in range(self.n)])
        return Matrix([[ops_mod.cast(self[i, j], dtype) for j in range(self.m)] for i in range(self.n)])

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
        # pylint: disable-msg=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.trace(self)

    def inverse(self):
        """Returns the inverse of this matrix.

        Note:
            The matrix dimension should be less than or equal to 4.

        Returns:
            :class:`~taichi.Matrix`: The inverse of a matrix.

        Raises:
            Exception: Inversions of matrices with sizes >= 5 are not supported.
        """
        from taichi.lang import matrix_ops  # pylint: disable=C0415

        return matrix_ops.inverse(self)

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
        # pylint: disable-msg=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.normalized(self, eps)

    def transpose(self):
        """Returns the transpose of a matrix.

        Returns:
            :class:`~taichi.Matrix`: The transpose of this matrix.

        Example::

            >>> A = ti.Matrix([[0, 1], [2, 3]])
            >>> A.transpose()
            [[0, 2], [1, 3]]
        """
        # pylint: disable=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.transpose(self)

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
        # pylint: disable=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.determinant(a)

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
        # pylint: disable=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.diag(dim, val)

    def sum(self):
        """Return the sum of all elements.

        Example::

            >>> m = ti.Matrix([[1, 2], [3, 4]])
            >>> m.sum()
            10
        """
        # pylint: disable=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.sum(self)

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
        # pylint: disable=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.norm(self, eps=eps)

    def norm_inv(self, eps=0):
        """The inverse of the matrix :func:`~taichi.lang.matrix.Matrix.norm`.

        Args:
            eps (float): a safe-guard value for sqrt, usually 0.

        Returns:
            The inverse of the matrix/vector `norm`.
        """
        # pylint: disable=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.norm_inv(self, eps=eps)

    def norm_sqr(self):
        """Returns the sum of the absolute squares of its elements."""
        # pylint: disable=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.norm_sqr(self)

    def max(self):
        """Returns the maximum element value."""
        # pylint: disable=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.max(self)

    def min(self):
        """Returns the minimum element value."""
        # pylint: disable=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.min(self)

    def any(self):
        """Test whether any element not equal zero.

        Returns:
            bool: `True` if any element is not equal zero, `False` otherwise.

        Example::

            >>> v = ti.Vector([0, 0, 1])
            >>> v.any()
            True
        """
        # pylint: disable=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.any(self)

    def all(self):
        """Test whether all element not equal zero.

        Returns:
            bool: `True` if all elements are not equal zero, `False` otherwise.

        Example::

            >>> v = ti.Vector([0, 0, 1])
            >>> v.all()
            False
        """
        # pylint: disable=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.all(self)

    def fill(self, val):
        """Fills the matrix with a specified value.

        Args:
            val (Union[int, float]): Value to fill.

        Example::

            >>> A = ti.Matrix([0, 1, 2, 3])
            >>> A.fill(-1)
            >>> A
            [-1, -1, -1, -1]
        """
        # pylint: disable=C0415
        from taichi.lang import matrix_ops

        return matrix_ops.fill(self, val)

    def to_numpy(self):
        """Converts this matrix to a numpy array.

        Returns:
            numpy.ndarray: The result numpy array.

        Example::

            >>> A = ti.Matrix([[0], [1], [2], [3]])
            >>> A.to_numpy()
            >>> A
            array([[0], [1], [2], [3]])
        """
        if self.is_host_access:
            return np.array(self.to_list())
        return self.entries

    @taichi_scope
    def __ti_repr__(self):
        yield "["
        for i in range(self.n):
            if i:
                yield ", "
            if self.m != 1:
                yield "["
            for j in range(self.m):
                if j:
                    yield ", "
                yield self(i, j)
            if self.m != 1:
                yield "]"
        yield "]"

    def __str__(self):
        """Python scope matrix print support."""
        if impl.inside_kernel():
            """
            It seems that when pybind11 got an type mismatch, it will try
            to invoke `repr` to show the object... e.g.:

            TypeError: make_const_expr_f32(): incompatible function arguments. The following argument types are supported:
                1. (arg0: float) -> taichi_python.Expr

            Invoked with: <Taichi 2x1 Matrix>

            So we have to make it happy with a dummy string...
            """
            return f"<{self.n}x{self.m} ti.Matrix>"
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
        from taichi.lang import matrix_ops  # pylint: disable=C0415

        if m is None:
            return matrix_ops._filled_vector(n, dt, 0)
        return matrix_ops._filled_matrix(n, m, dt, 0)

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
        from taichi.lang import matrix_ops  # pylint: disable=C0415

        if m is None:
            return matrix_ops._filled_vector(n, dt, 1)
        return matrix_ops._filled_matrix(n, m, dt, 1)

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
        from taichi.lang import matrix_ops  # pylint: disable=C0415

        if dt is None:
            dt = int
        assert 0 <= i < n
        return matrix_ops._unit_vector(n, i, dt)

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
        from taichi.lang import matrix_ops  # pylint: disable=C0415

        return matrix_ops._identity_matrix(n, dt)

    @classmethod
    @python_scope
    def field(
        cls,
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
        ndim=None,
    ):
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
                assert (
                    len(np.shape(dtype)) == 1 and len(dtype) == n
                ), f"Please set correct dtype list for Vector. The shape of dtype list should be ({n}, ) instead of {np.shape(dtype)}"
                for i in range(n):
                    entries.append(
                        impl.create_field_member(
                            dtype[i],
                            name=name,
                            needs_grad=needs_grad,
                            needs_dual=needs_dual,
                        )
                    )
            else:
                assert (
                    len(np.shape(dtype)) == 2 and len(dtype) == n and len(dtype[0]) == m
                ), f"Please set correct dtype list for Matrix. The shape of dtype list should be ({n}, {m}) instead of {np.shape(dtype)}"
                for i in range(n):
                    for j in range(m):
                        entries.append(
                            impl.create_field_member(
                                dtype[i][j],
                                name=name,
                                needs_grad=needs_grad,
                                needs_dual=needs_dual,
                            )
                        )
        else:
            for _ in range(n * m):
                entries.append(impl.create_field_member(dtype, name=name, needs_grad=needs_grad, needs_dual=needs_dual))
        entries, entries_grad, entries_dual = zip(*entries)

        entries = MatrixField(entries, n, m, element_dim)
        if all(entries_grad):
            entries_grad = MatrixField(entries_grad, n, m, element_dim)
            entries._set_grad(entries_grad)
        if all(entries_dual):
            entries_dual = MatrixField(entries_dual, n, m, element_dim)
            entries._set_dual(entries_dual)

        impl.get_runtime().matrix_fields.append(entries)

        if shape is None:
            if offset is not None:
                raise TaichiSyntaxError("shape cannot be None when offset is set")
            if order is not None:
                raise TaichiSyntaxError("shape cannot be None when order is set")
        else:
            if isinstance(shape, numbers.Number):
                shape = (shape,)
            if isinstance(offset, numbers.Number):
                offset = (offset,)
            dim = len(shape)
            if offset is not None and dim != len(offset):
                raise TaichiSyntaxError(
                    f"The dimensionality of shape and offset must be the same ({dim} != {len(offset)})"
                )
            axis_seq = []
            shape_seq = []
            if order is not None:
                if dim != len(order):
                    raise TaichiSyntaxError(
                        f"The dimensionality of shape and order must be the same ({dim} != {len(order)})"
                    )
                if dim != len(set(order)):
                    raise TaichiSyntaxError("The axes in order must be different")
                for ch in order:
                    axis = ord(ch) - ord("i")
                    if axis < 0 or axis >= dim:
                        raise TaichiSyntaxError(f"Invalid axis {ch}")
                    axis_seq.append(axis)
                    shape_seq.append(shape[axis])
            else:
                axis_seq = list(range(dim))
                shape_seq = list(shape)
            same_level = order is None
            if layout == Layout.SOA:
                for e in entries._get_field_members():
                    impl._create_snode(axis_seq, shape_seq, same_level).place(ScalarField(e), offset=offset)
                if needs_grad:
                    for e in entries_grad._get_field_members():
                        impl._create_snode(axis_seq, shape_seq, same_level).place(ScalarField(e), offset=offset)
                if needs_dual:
                    for e in entries_dual._get_field_members():
                        impl._create_snode(axis_seq, shape_seq, same_level).place(ScalarField(e), offset=offset)
            else:
                impl._create_snode(axis_seq, shape_seq, same_level).place(entries, offset=offset)
                if needs_grad:
                    impl._create_snode(axis_seq, shape_seq, same_level).place(entries_grad, offset=offset)
                if needs_dual:
                    impl._create_snode(axis_seq, shape_seq, same_level).place(entries_dual, offset=offset)
        return entries

    @classmethod
    @python_scope
    def ndarray(cls, n, m, dtype, shape):
        """Defines a Taichi ndarray with matrix elements.
        This function must be called in Python scope, and after `ti.init` is called.

        Args:
            n (int): Number of rows of the matrix.
            m (int): Number of columns of the matrix.
            dtype (DataType): Data type of each value.
            shape (Union[int, tuple[int]]): Shape of the ndarray.

        Example::

            The code below shows how a Taichi ndarray with matrix elements \
            can be declared and defined::

                >>> x = ti.Matrix.ndarray(4, 5, ti.f32, shape=(16, 8))
        """
        if isinstance(shape, numbers.Number):
            shape = (shape,)
        return MatrixNdarray(n, m, dtype, shape)

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
        from taichi.lang import matrix_ops  # pylint: disable=C0415

        return matrix_ops.rows(rows)

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
        from taichi.lang import matrix_ops  # pylint: disable=C0415

        return matrix_ops.cols(cols)

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
        from taichi.lang import matrix_ops  # pylint: disable=C0415

        return matrix_ops.dot(self, other)

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
        from taichi.lang import matrix_ops  # pylint: disable=C0415

        return matrix_ops.cross(self, other)

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
        from taichi.lang import matrix_ops  # pylint: disable=C0415

        return matrix_ops.outer_product(self, other)


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

    def get_shape(self):
        return (self.n,)

    @classmethod
    def field(cls, n, dtype, *args, **kwargs):
        """ti.Vector.field"""
        ndim = kwargs.get("ndim", 1)
        assert ndim == 1
        kwargs["ndim"] = 1
        return super().field(n, 1, dtype, *args, **kwargs)

    @classmethod
    @python_scope
    def ndarray(cls, n, dtype, shape):
        """Defines a Taichi ndarray with vector elements.

        Args:
            n (int): Size of the vector.
            dtype (DataType): Data type of each value.
            shape (Union[int, tuple[int]]): Shape of the ndarray.

        Example:
            The code below shows how a Taichi ndarray with vector elements can be declared and defined::

                >>> x = ti.Vector.ndarray(3, ti.f32, shape=(16, 8))
        """
        if isinstance(shape, numbers.Number):
            shape = (shape,)
        return VectorNdarray(n, dtype, shape)


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
        self.ptr = ti_python_core.expr_matrix_field([var.ptr for var in self.vars], [n, m][:ndim])

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

    def _get_dynamic_index_stride(self):
        if self.ptr.get_dynamic_indexable():
            return self.ptr.get_dynamic_index_stride()
        return None

    def _calc_dynamic_index_stride(self):
        # Algorithm: https://github.com/taichi-dev/taichi/issues/3810
        paths = [ScalarField(var).snode._path_from_root() for var in self.vars]
        num_members = len(paths)
        if num_members == 1:
            self.ptr.set_dynamic_index_stride(0)
            return
        length = len(paths[0])
        if any(len(path) != length or ti_python_core.is_quant(path[length - 1]._dtype) for path in paths):
            return
        for i in range(length):
            if any(path[i] != paths[0][i] for path in paths):
                depth_below_lca = i
                break
        for i in range(depth_below_lca, length - 1):
            if any(
                path[i].ptr.type != ti_python_core.SNodeType.dense
                or path[i]._cell_size_bytes != paths[0][i]._cell_size_bytes
                or path[i + 1]._offset_bytes_in_parent_cell != paths[0][i + 1]._offset_bytes_in_parent_cell
                for path in paths
            ):
                return
        stride = (
            paths[1][depth_below_lca]._offset_bytes_in_parent_cell
            - paths[0][depth_below_lca]._offset_bytes_in_parent_cell
        )
        for i in range(2, num_members):
            if (
                stride
                != paths[i][depth_below_lca]._offset_bytes_in_parent_cell
                - paths[i - 1][depth_below_lca]._offset_bytes_in_parent_cell
            ):
                return
        self.ptr.set_dynamic_index_stride(stride)

    def fill(self, val):
        """Fills this matrix field with specified values.

        Args:
            val (Union[Number, Expr, List, Tuple, Matrix]): Values to fill,
                should have consistent dimension consistent with `self`.
        """
        if isinstance(val, numbers.Number) or (isinstance(val, expr.Expr) and not val.is_tensor()):
            if self.ndim == 2:
                val = tuple(tuple(val for _ in range(self.m)) for _ in range(self.n))
            else:
                assert self.ndim == 1
                val = tuple(val for _ in range(self.n))
        elif isinstance(val, expr.Expr) and val.is_tensor():
            assert val.n == self.n
            if self.ndim != 1:
                assert val.m == self.m
        else:
            if isinstance(val, Matrix):
                val = val.to_list()
            assert isinstance(val, (list, tuple))
            val = tuple(tuple(x) if isinstance(x, list) else x for x in val)
            assert len(val) == self.n
            if self.ndim != 1:
                assert len(val[0]) == self.m
        if in_python_scope():
            from taichi._kernels import field_fill_python_scope  # pylint: disable=C0415

            field_fill_python_scope(self, val)
        else:
            from taichi._funcs import field_fill_taichi_scope  # pylint: disable=C0415

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
        shape_ext = (self.n,) if as_vector else (self.n, self.m)
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
        shape_ext = (self.n,) if as_vector else (self.n, self.m)
        # pylint: disable=E1101
        arr = torch.empty(self.shape + shape_ext, dtype=to_pytorch_type(self.dtype), device=device)
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
        shape_ext = (self.n,) if as_vector else (self.n, self.m)
        # pylint: disable=E1101
        # paddle.empty() doesn't support argument `place``
        arr = paddle.to_tensor(
            paddle.empty(self.shape + shape_ext, to_paddle_type(self.dtype)),
            place=place,
        )
        from taichi._kernels import matrix_to_ext_arr  # pylint: disable=C0415

        matrix_to_ext_arr(self, arr, as_vector)
        runtime_ops.sync()
        return arr

    @python_scope
    def _from_external_arr(self, arr):
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
    def from_numpy(self, arr):
        """Copies an `numpy.ndarray` into this field.

        Example::

            >>> m = ti.Matrix.field(2, 2, ti.f32, shape=(3, 3))
            >>> arr = numpy.ones((3, 3, 2, 2))
            >>> m.from_numpy(arr)
        """

        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        self._from_external_arr(arr)

    @python_scope
    def __setitem__(self, key, value):
        self._initialize_host_accessors()
        self[key]._set_entries(value)

    @python_scope
    def __getitem__(self, key):
        self._initialize_host_accessors()
        key = self._pad_key(key)
        _host_access = self._host_access(key)
        if self.ndim == 1:
            return Vector([_host_access[i] for i in range(self.n)])
        return Matrix([[_host_access[i * self.m + j] for j in range(self.m)] for i in range(self.n)])

    def __repr__(self):
        # make interactive shell happy, prevent materialization
        return f"<{self.n}x{self.m} ti.Matrix.field>"


class MatrixType(CompoundType):
    def __init__(self, n, m, ndim, dtype):
        self.n = n
        self.m = m
        self.ndim = ndim
        # FIXME(haidong): dtypes should not be left empty for ndarray.
        #                 Remove the None dtype when we are ready to break legacy code.
        if dtype is not None:
            self.dtype = cook_dtype(dtype)
            shape = (n, m) if ndim == 2 else (n,)
            self.tensor_type = _type_factory.get_tensor_type(shape, self.dtype)
        else:
            self.dtype = None
            self.tensor_type = None

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
            raise TaichiSyntaxError("Custom type instances need to be created with an initial value.")
        if len(args) == 1:
            # Init from a real Matrix
            if isinstance(args[0], expr.Expr) and args[0].ptr.is_tensor():
                arg = args[0]
                shape = arg.ptr.get_rvalue_type().shape()
                assert self.ndim == len(shape)
                assert self.n == shape[0]
                if self.ndim > 1:
                    assert self.m == shape[1]
                return expr.Expr(arg.ptr)

            # initialize by a single scalar, e.g. matnxm(1)
            if isinstance(args[0], (numbers.Number, expr.Expr)):
                entries = [args[0] for _ in range(self.m) for _ in range(self.n)]
                return self._instantiate(entries)
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
                entries += x.to_list()
            else:
                entries.append(x)

        return self._instantiate(entries)

    def from_taichi_object(self, func_ret, ret_index=()):
        return self(
            [
                expr.Expr(
                    ti_python_core.make_get_element_expr(
                        func_ret.ptr,
                        ret_index + (i,),
                        _ti_python_core.DebugInfo(impl.get_runtime().get_current_src_info()),
                    )
                )
                for i in range(self.m * self.n)
            ]
        )

    def from_kernel_struct_ret(self, launch_ctx, ret_index=()):
        if self.dtype in primitive_types.integer_types:
            if is_signed(cook_dtype(self.dtype)):
                get_ret_func = launch_ctx.get_struct_ret_int
            else:
                get_ret_func = launch_ctx.get_struct_ret_uint
        elif self.dtype in primitive_types.real_types:
            get_ret_func = launch_ctx.get_struct_ret_float
        else:
            raise TaichiRuntimeTypeError(f"Invalid return type on index={ret_index}")
        return self([get_ret_func(ret_index + (i,)) for i in range(self.m * self.n)])

    def set_kernel_struct_args(self, mat, launch_ctx, ret_index=()):
        if self.dtype in primitive_types.integer_types:
            if is_signed(cook_dtype(self.dtype)):
                set_arg_func = launch_ctx.set_struct_arg_int
            else:
                set_arg_func = launch_ctx.set_struct_arg_uint
        elif self.dtype in primitive_types.real_types:
            set_arg_func = launch_ctx.set_struct_arg_float
        else:
            raise TaichiRuntimeTypeError(f"Invalid return type on index={ret_index}")
        if self.ndim == 1:
            for i in range(self.n):
                set_arg_func(ret_index + (i,), mat[i])
        else:
            for i in range(self.n):
                for j in range(self.m):
                    set_arg_func(ret_index + (i * self.m + j,), mat[i, j])

    def set_argpack_struct_args(self, mat, argpack, ret_index=()):
        if self.dtype in primitive_types.integer_types:
            if is_signed(cook_dtype(self.dtype)):
                set_arg_func = argpack.set_arg_int
            else:
                set_arg_func = argpack.set_arg_uint
        elif self.dtype in primitive_types.real_types:
            set_arg_func = argpack.set_arg_float
        else:
            raise TaichiRuntimeTypeError(f"Invalid return type on index={ret_index}")
        if self.ndim == 1:
            for i in range(self.n):
                set_arg_func(ret_index + (i,), mat[i])
        else:
            for i in range(self.n):
                for j in range(self.m):
                    set_arg_func(ret_index + (i * self.m + j,), mat[i, j])

    def _instantiate_in_python_scope(self, entries):
        entries = [[entries[k * self.m + i] for i in range(self.m)] for k in range(self.n)]
        return Matrix(
            [
                [
                    int(entries[i][j]) if self.dtype in primitive_types.integer_types else float(entries[i][j])
                    for j in range(self.m)
                ]
                for i in range(self.n)
            ],
            dt=self.dtype,
        )

    def _instantiate(self, entries):
        if in_python_scope():
            return self._instantiate_in_python_scope(entries)

        return make_matrix_with_shape(entries, [self.n, self.m], self.dtype)

    def field(self, **kwargs):
        assert kwargs.get("ndim", self.ndim) == self.ndim
        kwargs.update({"ndim": self.ndim})
        return Matrix.field(self.n, self.m, dtype=self.dtype, **kwargs)

    def ndarray(self, **kwargs):
        assert kwargs.get("ndim", self.ndim) == self.ndim
        kwargs.update({"ndim": self.ndim})
        return Matrix.ndarray(self.n, self.m, dtype=self.dtype, **kwargs)

    def get_shape(self):
        if self.ndim == 1:
            return (self.n,)
        return (self.n, self.m)

    def to_string(self):
        dtype_str = self.dtype.to_string() if self.dtype is not None else ""
        return f"MatrixType[{self.n},{self.m}, {dtype_str}]"

    def check_matched(self, other):
        if self.ndim != len(other.shape()):
            return False
        if self.dtype is not None and self.dtype != other.element_type():
            return False
        shape = self.get_shape()
        for i in range(self.ndim):
            if shape[i] is not None and shape[i] != other.shape()[i]:
                return False
        return True


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
            raise TaichiSyntaxError("Custom type instances need to be created with an initial value.")
        if len(args) == 1:
            # Init from a real Matrix
            if isinstance(args[0], expr.Expr) and args[0].ptr.is_tensor():
                arg = args[0]
                shape = arg.ptr.get_rvalue_type().shape()
                assert len(shape) == 1
                assert self.n == shape[0]
                return expr.Expr(arg.ptr)

            # initialize by a single scalar, e.g. matnxm(1)
            if isinstance(args[0], (numbers.Number, expr.Expr)):
                entries = [args[0] for _ in range(self.n)]
                return self._instantiate(entries)
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
                entries += x.to_list()
            else:
                entries.append(x)

        #  type cast
        return self._instantiate(entries)

    def _instantiate_in_python_scope(self, entries):
        return Vector(
            [
                int(entries[i]) if self.dtype in primitive_types.integer_types else float(entries[i])
                for i in range(self.n)
            ],
            dt=self.dtype,
        )

    def _instantiate(self, entries):
        if in_python_scope():
            return self._instantiate_in_python_scope(entries)

        return make_matrix_with_shape(entries, [self.n], self.dtype)

    def field(self, **kwargs):
        return Vector.field(self.n, dtype=self.dtype, **kwargs)

    def ndarray(self, **kwargs):
        return Vector.ndarray(self.n, dtype=self.dtype, **kwargs)

    def to_string(self):
        dtype_str = self.dtype.to_string() if self.dtype is not None else ""
        return f"VectorType[{self.n}, {dtype_str}]"


class MatrixNdarray(Ndarray):
    """Taichi ndarray with matrix elements.

    Args:
        n (int): Number of rows of the matrix.
        m (int): Number of columns of the matrix.
        dtype (DataType): Data type of each value.
        shape (Union[int, tuple[int]]): Shape of the ndarray.

    Example::

        >>> arr = ti.MatrixNdarray(2, 2, ti.f32, shape=(3, 3))
    """

    def __init__(self, n, m, dtype, shape):
        self.n = n
        self.m = m
        super().__init__()
        # TODO(zhanlue): remove self.dtype and migrate its usages to element_type
        self.dtype = cook_dtype(dtype)

        self.layout = Layout.AOS
        self.shape = tuple(shape)
        self.element_type = _type_factory.get_tensor_type((self.n, self.m), self.dtype)
        # TODO: we should pass in element_type, shape, layout instead.
        self.arr = impl.get_runtime().prog.create_ndarray(
            cook_dtype(self.element_type),
            shape,
            Layout.AOS,
            zero_fill=True,
            dbg_info=ti_python_core.DebugInfo(get_traceback()),
        )

    @property
    def element_shape(self):
        """Returns the shape of each element (a 2D matrix) in this ndarray.

        Example::

            >>> arr = ti.MatrixNdarray(2, 2, ti.f32, shape=(3, 3))
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
        key = () if key is None else (key,) if isinstance(key, numbers.Number) else tuple(key)
        return Matrix([[NdarrayHostAccess(self, key, (i, j)) for j in range(self.m)] for i in range(self.n)])

    @python_scope
    def to_numpy(self):
        """Converts this ndarray to a `numpy.ndarray`.

        Example::

            >>> arr = ti.MatrixNdarray(2, 2, ti.f32, shape=(2, 1))
            >>> arr.to_numpy()
            [[[[0. 0.]
               [0. 0.]]]

             [[[0. 0.]
               [0. 0.]]]]
        """
        return self._ndarray_matrix_to_numpy(as_vector=0)

    @python_scope
    def from_numpy(self, arr):
        """Copies the data of a `numpy.ndarray` into this array.

        Example::

            >>> m = ti.MatrixNdarray(2, 2, ti.f32, shape=(2, 1), layout=0)
            >>> arr = np.ones((2, 1, 2, 2))
            >>> m.from_numpy(arr)
        """
        self._ndarray_matrix_from_numpy(arr, as_vector=0)

    @python_scope
    def __deepcopy__(self, memo=None):
        ret_arr = MatrixNdarray(self.n, self.m, self.dtype, self.shape)
        ret_arr.copy_from(self)
        return ret_arr

    @python_scope
    def _fill_by_kernel(self, val):
        from taichi._kernels import fill_ndarray_matrix  # pylint: disable=C0415

        shape = self.element_type.shape()
        n = shape[0]
        m = 1
        if len(shape) > 1:
            m = shape[1]

        prim_dtype = self.element_type.element_type()
        matrix_type = MatrixType(n, m, len(shape), prim_dtype)
        if isinstance(val, Matrix):
            value = val
        else:
            value = matrix_type(val)
        fill_ndarray_matrix(self, value)

    @python_scope
    def __repr__(self):
        return f"<{self.n}x{self.m} {Layout.AOS} ti.Matrix.ndarray>"


class VectorNdarray(Ndarray):
    """Taichi ndarray with vector elements.

    Args:
        n (int): Size of the vector.
        dtype (DataType): Data type of each value.
        shape (Tuple[int]): Shape of the ndarray.

    Example::

        >>> a = ti.VectorNdarray(3, ti.f32, (3, 3))
    """

    def __init__(self, n, dtype, shape):
        self.n = n
        super().__init__()
        # TODO(zhanlue): remove self.dtype and migrate its usages to element_type
        self.dtype = cook_dtype(dtype)

        self.layout = Layout.AOS
        self.shape = tuple(shape)
        self.element_type = _type_factory.get_tensor_type((n,), self.dtype)
        self.arr = impl.get_runtime().prog.create_ndarray(
            cook_dtype(self.element_type),
            shape,
            Layout.AOS,
            zero_fill=True,
            dbg_info=ti_python_core.DebugInfo(get_traceback()),
        )

    @property
    def element_shape(self):
        """Gets the dimension of the vector of this ndarray.

        Example::

            >>> a = ti.VectorNdarray(3, ti.f32, (3, 3))
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
        key = () if key is None else (key,) if isinstance(key, numbers.Number) else tuple(key)
        return Vector([NdarrayHostAccess(self, key, (i,)) for i in range(self.n)])

    @python_scope
    def to_numpy(self):
        """Converts this vector ndarray to a `numpy.ndarray`.

        Example::

            >>> a = ti.VectorNdarray(3, ti.f32, (2, 2))
            >>> a.to_numpy()
            array([[[0., 0., 0.],
                    [0., 0., 0.]],

                   [[0., 0., 0.],
                    [0., 0., 0.]]], dtype=float32)
        """
        return self._ndarray_matrix_to_numpy(as_vector=1)

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
        self._ndarray_matrix_from_numpy(arr, as_vector=1)

    @python_scope
    def __deepcopy__(self, memo=None):
        ret_arr = VectorNdarray(self.n, self.dtype, self.shape)
        ret_arr.copy_from(self)
        return ret_arr

    @python_scope
    def _fill_by_kernel(self, val):
        from taichi._kernels import fill_ndarray_matrix  # pylint: disable=C0415

        shape = self.element_type.shape()
        prim_dtype = self.element_type.element_type()
        vector_type = VectorType(shape[0], prim_dtype)
        if isinstance(val, Vector):
            value = val
        else:
            value = vector_type(val)
        fill_ndarray_matrix(self, value)

    @python_scope
    def __repr__(self):
        return f"<{self.n} {Layout.AOS} ti.Vector.ndarray>"


__all__ = ["Matrix", "Vector", "MatrixField", "MatrixNdarray", "VectorNdarray"]
