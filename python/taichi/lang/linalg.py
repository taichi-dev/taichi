from taichi.core.util import ti_core as _ti_core
from taichi.lang.impl import expr_init

import taichi as ti


@ti.func
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
    scale = (1.0 / ti.sqrt(x * x + y * y))
    c = x * scale
    s = y * scale
    r = ti.Matrix([[c, -s], [s, c]], dt=dt)
    return r, r.transpose() @ A


@ti.func
def polar_decompose3d(A, dt):
    """Perform polar decomposition (A=UP) for 3x3 matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Polar_decomposition.

    Args:
        A (ti.Matrix(3, 3)): input 3x3 matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        Decomposed 3x3 matrices `U` and `P`.
    """
    U, sig, V = ti.svd(A, dt)
    return U @ V.transpose(), V @ sig @ V.transpose()


# https://www.seas.upenn.edu/~cffjiang/research/svd/svd.pdf
@ti.func
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
    c, s = ti.cast(0.0, dt), ti.cast(0.0, dt)
    s1, s2 = ti.cast(0.0, dt), ti.cast(0.0, dt)
    if abs(S[0, 1]) < 1e-5:
        c, s = 1, 0
        s1, s2 = S[0, 0], S[1, 1]
    else:
        tao = ti.cast(0.5, dt) * (S[0, 0] - S[1, 1])
        w = ti.sqrt(tao**2 + S[0, 1]**2)
        t = ti.cast(0.0, dt)
        if tao > 0:
            t = S[0, 1] / (tao + w)
        else:
            t = S[0, 1] / (tao - w)
        c = 1 / ti.sqrt(t**2 + 1)
        s = -t * c
        s1 = c**2 * S[0, 0] - 2 * c * s * S[0, 1] + s**2 * S[1, 1]
        s2 = s**2 * S[0, 0] + 2 * c * s * S[0, 1] + c**2 * S[1, 1]
    V = ti.Matrix.zero(dt, 2, 2)
    if s1 < s2:
        tmp = s1
        s1 = s2
        s2 = tmp
        V = ti.Matrix([[-s, c], [-c, -s]], dt=dt)
    else:
        V = ti.Matrix([[c, s], [-s, c]], dt=dt)
    U = R @ V
    return U, ti.Matrix([[s1, ti.cast(0, dt)], [ti.cast(0, dt), s2]], dt=dt), V


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
    assert dt in [ti.f32, ti.f64]
    if iters is None:
        if dt == ti.f32:
            iters = 5
        else:
            iters = 8
    if dt == ti.f32:
        rets = _ti_core.sifakis_svd_f32(*inputs, iters)
    else:
        rets = _ti_core.sifakis_svd_f64(*inputs, iters)
    assert len(rets) == 21
    U_entries = rets[:9]
    V_entries = rets[9:18]
    sig_entries = rets[18:]
    U = expr_init(ti.Matrix.zero(dt, 3, 3))
    V = expr_init(ti.Matrix.zero(dt, 3, 3))
    sigma = expr_init(ti.Matrix.zero(dt, 3, 3))
    for i in range(3):
        for j in range(3):
            U(i, j).assign(U_entries[i * 3 + j])
            V(i, j).assign(V_entries[i * 3 + j])
        sigma(i, i).assign(sig_entries[i])
    return U, sigma, V


@ti.func
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
    lambda1 = ti.Vector.zero(dt, 2)
    lambda2 = ti.Vector.zero(dt, 2)
    v1 = ti.Vector.zero(dt, 4)
    v2 = ti.Vector.zero(dt, 4)
    if gap > 0:
        lambda1 = ti.Vector([tr + ti.sqrt(gap), 0.0], dt=dt) * 0.5
        lambda2 = ti.Vector([tr - ti.sqrt(gap), 0.0], dt=dt) * 0.5
        A1 = A - lambda1[0] * ti.Matrix.identity(dt, 2)
        A2 = A - lambda2[0] * ti.Matrix.identity(dt, 2)
        if all(A1 == ti.Matrix.zero(dt, 2, 2)) and all(
                A1 == ti.Matrix.zero(dt, 2, 2)):
            v1 = ti.Vector([0.0, 0.0, 1.0, 0.0]).cast(dt)
            v2 = ti.Vector([1.0, 0.0, 0.0, 0.0]).cast(dt)
        else:
            v1 = ti.Vector([A2[0, 0], 0.0, A2[1, 0], 0.0], dt=dt).normalized()
            v2 = ti.Vector([A1[0, 0], 0.0, A1[1, 0], 0.0], dt=dt).normalized()
    else:
        lambda1 = ti.Vector([tr, ti.sqrt(-gap)], dt=dt) * 0.5
        lambda2 = ti.Vector([tr, -ti.sqrt(-gap)], dt=dt) * 0.5
        A1r = A - lambda1[0] * ti.Matrix.identity(dt, 2)
        A1i = -lambda1[1] * ti.Matrix.identity(dt, 2)
        A2r = A - lambda2[0] * ti.Matrix.identity(dt, 2)
        A2i = -lambda2[1] * ti.Matrix.identity(dt, 2)
        v1 = ti.Vector([A2r[0, 0], A2i[0, 0], A2r[1, 0], A2i[1, 0]],
                       dt=dt).normalized()
        v2 = ti.Vector([A1r[0, 0], A1i[0, 0], A1r[1, 0], A1i[1, 0]],
                       dt=dt).normalized()
    eigenvalues = ti.Matrix.rows([lambda1, lambda2])
    eigenvectors = ti.Matrix.cols([v1, v2])

    return eigenvalues, eigenvectors


@ti.func
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
    lambda1 = (tr + ti.sqrt(gap)) * 0.5
    lambda2 = (tr - ti.sqrt(gap)) * 0.5
    eigenvalues = ti.Vector([lambda1, lambda2], dt=dt)

    A1 = A - lambda1 * ti.Matrix.identity(dt, 2)
    A2 = A - lambda2 * ti.Matrix.identity(dt, 2)
    v1 = ti.Vector.zero(dt, 2)
    v2 = ti.Vector.zero(dt, 2)
    if all(A1 == ti.Matrix.zero(dt, 2, 2)) and all(
            A1 == ti.Matrix.zero(dt, 2, 2)):
        v1 = ti.Vector([0.0, 1.0]).cast(dt)
        v2 = ti.Vector([1.0, 0.0]).cast(dt)
    else:
        v1 = ti.Vector([A2[0, 0], A2[1, 0]], dt=dt).normalized()
        v2 = ti.Vector([A1[0, 0], A1[1, 0]], dt=dt).normalized()
    eigenvectors = ti.Matrix.cols([v1, v2])
    return eigenvalues, eigenvectors


@ti.func
def svd(A, dt):
    """Perform singular value decomposition (A=USV^T) for arbitrary size matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Singular_value_decomposition.
    2D implementation refers to :func:`taichi.lang.linalg.svd2d`.
    3D implementation refers to :func:`taichi.lang.linalg.svd3d`.

    Args:
        A (ti.Matrix(n, n)): input nxn matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        Decomposed nxn matrices `U`, 'S' and `V`.
    """
    if ti.static(A.n == 2):
        ret = svd2d(A, dt)
        return ret
    elif ti.static(A.n == 3):
        return svd3d(A, dt)
    else:
        raise Exception("SVD only supports 2D and 3D matrices.")


@ti.func
def polar_decompose(A, dt):
    """Perform polar decomposition (A=UP) for arbitrary size matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Polar_decomposition.
    2D implementation refers to :func:`taichi.lang.linalg.polar_decompose2d`.
    3D implementation refers to :func:`taichi.lang.linalg.polar_decompose3d`.

    Args:
        A (ti.Matrix(n, n)): input nxn matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        Decomposed nxn matrices `U` and `P`.
    """
    if ti.static(A.n == 2):
        ret = polar_decompose2d(A, dt)
        return ret
    elif ti.static(A.n == 3):
        return polar_decompose3d(A, dt)
    else:
        raise Exception(
            "Polar decomposition only supports 2D and 3D matrices.")
