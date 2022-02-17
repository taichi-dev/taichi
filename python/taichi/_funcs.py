import math

from taichi.lang import impl, matrix, ops
from taichi.lang.impl import expr_init, get_runtime, static
from taichi.lang.kernel_impl import func, pyfunc
from taichi.lang.matrix import Matrix, Vector
from taichi.types import f32, f64


@func
def _randn(dt):
    """
    Generate a random float sampled from univariate standard normal
    (Gaussian) distribution of mean 0 and variance 1, using the
    Box-Muller transformation.
    """
    assert dt == f32 or dt == f64
    u1 = ops.cast(1.0, dt) - ops.random(dt)
    u2 = ops.random(dt)
    r = ops.sqrt(-2 * ops.log(u1))
    c = ops.cos(math.tau * u2)
    return r * c


def randn(dt=None):
    """Generate a random float sampled from univariate standard normal
    (Gaussian) distribution of mean 0 and variance 1, using the
    Box-Muller transformation. Must be called in Taichi scope.

    Args:
        dt (DataType): Data type of the required random number. Default to `None`.
            If set to `None` `dt` will be determined dynamically in runtime.

    Returns:
        The generated random float.

    Example::

        >>> @ti.kernel
        >>> def main():
        >>>     print(ti.randn())
        >>>
        >>> main()
        -0.463608
    """
    if dt is None:
        dt = impl.get_runtime().default_fp
    return _randn(dt)


@pyfunc
def _matrix_transpose(mat):
    """Permute the first two axes of the matrix.

    Args:
        mat (:class:`~taichi.lang.matrix.Matrix`): Input matrix.

    Returns:
        Transpose of the input matrix.
    """
    return matrix.Matrix([[mat[i, j] for i in range(mat.n)]
                          for j in range(mat.m)])


@pyfunc
def _matrix_cross3d(self, other):
    return matrix.Matrix([
        self[1] * other[2] - self[2] * other[1],
        self[2] * other[0] - self[0] * other[2],
        self[0] * other[1] - self[1] * other[0],
    ])


@pyfunc
def _matrix_cross2d(self, other):
    return self[0] * other[1] - self[1] * other[0]


@pyfunc
def _matrix_outer_product(self, other):
    """Perform the outer product with the input Vector (1-D Matrix).

    Args:
        other (:class:`~taichi.lang.matrix.Matrix`): The input Vector (1-D Matrix) to perform the outer product.

    Returns:
        :class:`~taichi.lang.matrix.Matrix`: The outer product result (Matrix) of the two Vectors.

    """
    impl.static(
        impl.static_assert(self.m == 1,
                           "lhs for outer_product is not a vector"))
    impl.static(
        impl.static_assert(other.m == 1,
                           "rhs for outer_product is not a vector"))
    return matrix.Matrix([[self[i] * other[j] for j in range(other.n)]
                          for i in range(self.n)])


@func
def polar_decompose2d(A, dt):
    """Perform polar decomposition (A=UP) for 2x2 matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Polar_decomposition.

    Args:
        A (ti.Matrix(2, 2)): input 2x2 matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        Decomposed 2x2 matrices `U` and `P`.
    """
    x, y = A(0, 0) + A(1, 1), A(1, 0) - A(0, 1)
    scale = (1.0 / ops.sqrt(x * x + y * y))
    c = x * scale
    s = y * scale
    r = Matrix([[c, -s], [s, c]], dt=dt)
    return r, r.transpose() @ A


@func
def polar_decompose3d(A, dt):
    """Perform polar decomposition (A=UP) for 3x3 matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Polar_decomposition.

    Args:
        A (ti.Matrix(3, 3)): input 3x3 matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        Decomposed 3x3 matrices `U` and `P`.
    """
    U, sig, V = svd(A, dt)
    return U @ V.transpose(), V @ sig @ V.transpose()


# https://www.seas.upenn.edu/~cffjiang/research/svd/svd.pdf
@func
def svd2d(A, dt):
    """Perform singular value decomposition (A=USV^T) for 2x2 matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Singular_value_decomposition.

    Args:
        A (ti.Matrix(2, 2)): input 2x2 matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        Decomposed 2x2 matrices `U`, 'S' and `V`.
    """
    R, S = polar_decompose2d(A, dt)
    c, s = ops.cast(0.0, dt), ops.cast(0.0, dt)
    s1, s2 = ops.cast(0.0, dt), ops.cast(0.0, dt)
    if abs(S[0, 1]) < 1e-5:
        c, s = 1, 0
        s1, s2 = S[0, 0], S[1, 1]
    else:
        tao = ops.cast(0.5, dt) * (S[0, 0] - S[1, 1])
        w = ops.sqrt(tao**2 + S[0, 1]**2)
        t = ops.cast(0.0, dt)
        if tao > 0:
            t = S[0, 1] / (tao + w)
        else:
            t = S[0, 1] / (tao - w)
        c = 1 / ops.sqrt(t**2 + 1)
        s = -t * c
        s1 = c**2 * S[0, 0] - 2 * c * s * S[0, 1] + s**2 * S[1, 1]
        s2 = s**2 * S[0, 0] + 2 * c * s * S[0, 1] + c**2 * S[1, 1]
    V = Matrix.zero(dt, 2, 2)
    if s1 < s2:
        tmp = s1
        s1 = s2
        s2 = tmp
        V = Matrix([[-s, c], [-c, -s]], dt=dt)
    else:
        V = Matrix([[c, s], [-s, c]], dt=dt)
    U = R @ V
    return U, Matrix([[s1, ops.cast(0, dt)], [ops.cast(0, dt), s2]], dt=dt), V


def svd3d(A, dt, iters=None):
    """Perform singular value decomposition (A=USV^T) for 3x3 matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Singular_value_decomposition.

    Args:
        A (ti.Matrix(3, 3)): input 3x3 matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.
        iters (int): iteration number to control algorithm precision.

    Returns:
        Decomposed 3x3 matrices `U`, 'S' and `V`.
    """
    assert A.n == 3 and A.m == 3
    inputs = tuple([e.ptr for e in A.entries])
    assert dt in [f32, f64]
    if iters is None:
        if dt == f32:
            iters = 5
        else:
            iters = 8
    if dt == f32:
        rets = get_runtime().prog.current_ast_builder().sifakis_svd_f32(
            *inputs, iters)
    else:
        rets = get_runtime().prog.current_ast_builder().sifakis_svd_f64(
            *inputs, iters)
    assert len(rets) == 21
    U_entries = rets[:9]
    V_entries = rets[9:18]
    sig_entries = rets[18:]
    U = expr_init(Matrix.zero(dt, 3, 3))
    V = expr_init(Matrix.zero(dt, 3, 3))
    sigma = expr_init(Matrix.zero(dt, 3, 3))
    for i in range(3):
        for j in range(3):
            U(i, j)._assign(U_entries[i * 3 + j])
            V(i, j)._assign(V_entries[i * 3 + j])
        sigma(i, i)._assign(sig_entries[i])
    return U, sigma, V


@func
def eig2x2(A, dt):
    """Compute the eigenvalues and right eigenvectors (Av=lambda v) of a 2x2 real matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix.

    Args:
        A (ti.Matrix(2, 2)): input 2x2 matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        eigenvalues (ti.Matrix(2, 2)): The eigenvalues in complex form. Each row stores one eigenvalue. The first number of the eigenvalue represents the real part and the second number represents the imaginary part.
        eigenvectors: (ti.Matrix(4, 2)): The eigenvectors in complex form. Each column stores one eigenvector. Each eigenvector consists of 2 entries, each of which is represented by two numbers for its real part and imaginary part.
    """
    tr = A.trace()
    det = A.determinant()
    gap = tr**2 - 4 * det
    lambda1 = Vector.zero(dt, 2)
    lambda2 = Vector.zero(dt, 2)
    v1 = Vector.zero(dt, 4)
    v2 = Vector.zero(dt, 4)
    if gap > 0:
        lambda1 = Vector([tr + ops.sqrt(gap), 0.0], dt=dt) * 0.5
        lambda2 = Vector([tr - ops.sqrt(gap), 0.0], dt=dt) * 0.5
        A1 = A - lambda1[0] * Matrix.identity(dt, 2)
        A2 = A - lambda2[0] * Matrix.identity(dt, 2)
        if all(A1 == Matrix.zero(dt, 2, 2)) and all(
                A1 == Matrix.zero(dt, 2, 2)):
            v1 = Vector([0.0, 0.0, 1.0, 0.0]).cast(dt)
            v2 = Vector([1.0, 0.0, 0.0, 0.0]).cast(dt)
        else:
            v1 = Vector([A2[0, 0], 0.0, A2[1, 0], 0.0], dt=dt).normalized()
            v2 = Vector([A1[0, 0], 0.0, A1[1, 0], 0.0], dt=dt).normalized()
    else:
        lambda1 = Vector([tr, ops.sqrt(-gap)], dt=dt) * 0.5
        lambda2 = Vector([tr, -ops.sqrt(-gap)], dt=dt) * 0.5
        A1r = A - lambda1[0] * Matrix.identity(dt, 2)
        A1i = -lambda1[1] * Matrix.identity(dt, 2)
        A2r = A - lambda2[0] * Matrix.identity(dt, 2)
        A2i = -lambda2[1] * Matrix.identity(dt, 2)
        v1 = Vector([A2r[0, 0], A2i[0, 0], A2r[1, 0], A2i[1, 0]],
                    dt=dt).normalized()
        v2 = Vector([A1r[0, 0], A1i[0, 0], A1r[1, 0], A1i[1, 0]],
                    dt=dt).normalized()
    eigenvalues = Matrix.rows([lambda1, lambda2])
    eigenvectors = Matrix.cols([v1, v2])

    return eigenvalues, eigenvectors


@func
def sym_eig2x2(A, dt):
    """Compute the eigenvalues and right eigenvectors (Av=lambda v) of a 2x2 real symmetric matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix.

    Args:
        A (ti.Matrix(2, 2)): input 2x2 symmetric matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        eigenvalues (ti.Vector(2)): The eigenvalues. Each entry store one eigen value.
        eigenvectors (ti.Matrix(2, 2)): The eigenvectors. Each column stores one eigenvector.
    """
    tr = A.trace()
    det = A.determinant()
    gap = tr**2 - 4 * det
    lambda1 = (tr + ops.sqrt(gap)) * 0.5
    lambda2 = (tr - ops.sqrt(gap)) * 0.5
    eigenvalues = Vector([lambda1, lambda2], dt=dt)

    A1 = A - lambda1 * Matrix.identity(dt, 2)
    A2 = A - lambda2 * Matrix.identity(dt, 2)
    v1 = Vector.zero(dt, 2)
    v2 = Vector.zero(dt, 2)
    if all(A1 == Matrix.zero(dt, 2, 2)) and all(A1 == Matrix.zero(dt, 2, 2)):
        v1 = Vector([0.0, 1.0]).cast(dt)
        v2 = Vector([1.0, 0.0]).cast(dt)
    else:
        v1 = Vector([A2[0, 0], A2[1, 0]], dt=dt).normalized()
        v2 = Vector([A1[0, 0], A1[1, 0]], dt=dt).normalized()
    eigenvectors = Matrix.cols([v1, v2])
    return eigenvalues, eigenvectors


@func
def _svd(A, dt):
    """Perform singular value decomposition (A=USV^T) for arbitrary size matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Singular_value_decomposition.
    2D implementation refers to :func:`taichi.svd2d`.
    3D implementation refers to :func:`taichi.svd3d`.

    Args:
        A (ti.Matrix(n, n)): input nxn matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        Decomposed nxn matrices `U`, 'S' and `V`.
    """
    if static(A.n == 2):  # pylint: disable=R1705
        ret = svd2d(A, dt)
        return ret
    else:
        return svd3d(A, dt)


@func
def _polar_decompose(A, dt):
    """Perform polar decomposition (A=UP) for arbitrary size matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Polar_decomposition.
    2D implementation refers to :func:`taichi.polar_decompose2d`.
    3D implementation refers to :func:`taichi.polar_decompose3d`.

    Args:
        A (ti.Matrix(n, n)): input nxn matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        Decomposed nxn matrices `U` and `P`.
    """
    if static(A.n == 2):  # pylint: disable=R1705
        ret = polar_decompose2d(A, dt)
        return ret
    else:
        return polar_decompose3d(A, dt)


def polar_decompose(A, dt=None):
    """Perform polar decomposition (A=UP) for arbitrary size matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Polar_decomposition.
    This is only a wrapper for :func:`taichi.polar_decompose`.

    Args:
        A (ti.Matrix(n, n)): input nxn matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        Decomposed nxn matrices `U` and `P`.
    """
    if dt is None:
        dt = impl.get_runtime().default_fp
    if A.n != 2 and A.n != 3:
        raise Exception(
            "Polar decomposition only supports 2D and 3D matrices.")
    return _polar_decompose(A, dt)


def svd(A, dt=None):
    """Perform singular value decomposition (A=USV^T) for arbitrary size matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Singular_value_decomposition.
    This is only a wrappers for :func:`taichi.svd`.

    Args:
        A (ti.Matrix(n, n)): input nxn matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        Decomposed nxn matrices `U`, 'S' and `V`.
    """
    if dt is None:
        dt = impl.get_runtime().default_fp
    if A.n != 2 and A.n != 3:
        raise Exception("SVD only supports 2D and 3D matrices.")
    return _svd(A, dt)


def eig(A, dt=None):
    """Compute the eigenvalues and right eigenvectors of a real matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix.
    2D implementation refers to :func:`taichi.eig2x2`.

    Args:
        A (ti.Matrix(n, n)): 2D Matrix for which the eigenvalues and right eigenvectors will be computed.
        dt (DataType): The datatype for the eigenvalues and right eigenvectors.

    Returns:
        eigenvalues (ti.Matrix(n, 2)): The eigenvalues in complex form. Each row stores one eigenvalue. The first number of the eigenvalue represents the real part and the second number represents the imaginary part.
        eigenvectors (ti.Matrix(n*2, n)): The eigenvectors in complex form. Each column stores one eigenvector. Each eigenvector consists of n entries, each of which is represented by two numbers for its real part and imaginary part.
    """
    if dt is None:
        dt = impl.get_runtime().default_fp
    if A.n == 2:
        return eig2x2(A, dt)
    raise Exception("Eigen solver only supports 2D matrices.")


def sym_eig(A, dt=None):
    """Compute the eigenvalues and right eigenvectors of a real symmetric matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix.
    2D implementation refers to :func:`taichi.sym_eig2x2`.

    Args:
        A (ti.Matrix(n, n)): Symmetric Matrix for which the eigenvalues and right eigenvectors will be computed.
        dt (DataType): The datatype for the eigenvalues and right eigenvectors.

    Returns:
        eigenvalues (ti.Vector(n)): The eigenvalues. Each entry store one eigen value.
        eigenvectors (ti.Matrix(n, n)): The eigenvectors. Each column stores one eigenvector.
    """
    assert all(A == A.transpose()), "A needs to be symmetric"
    if dt is None:
        dt = impl.get_runtime().default_fp
    if A.n == 2:
        return sym_eig2x2(A, dt)
    raise Exception("Symmetric eigen solver only supports 2D matrices.")


__all__ = ['randn', 'polar_decompose', 'eig', 'sym_eig', 'svd']
