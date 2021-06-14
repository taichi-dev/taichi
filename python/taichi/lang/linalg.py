from taichi.core.util import ti_core as _ti_core
from taichi.lang.impl import expr_init

import taichi as ti


@ti.func
def polar_decompose2d(a, dt):
    x, y = a(0, 0) + a(1, 1), a(1, 0) - a(0, 1)
    scale = (1.0 / ti.sqrt(x * x + y * y))
    c = x * scale
    s = y * scale
    r = ti.Matrix([[c, -s], [s, c]])
    return r, r.transpose() @ a


@ti.func
def polar_decompose3d(A, dt):
    U, sig, V = ti.svd(A, dt)
    return U @ V.transpose(), V @ sig @ V.transpose()


# https://www.seas.upenn.edu/~cffjiang/research/svd/svd.pdf
@ti.func
def svd2d(A, dt):
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
        V = [[-s, c], [-c, -s]]
    else:
        V = [[c, s], [-s, c]]
    U = R @ V
    return U, ti.Matrix([[s1, ti.cast(0, dt)], [ti.cast(0, dt), s2]]), V


def svd3d(A, dt, iters=None):
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
    tr = A.trace()
    det = A.determinant()
    gap = tr**2 - 4 * det
    lambda1 = ti.Vector.zero(dt, 2)
    lambda2 = ti.Vector.zero(dt, 2)
    v1 = ti.Vector.zero(dt, 4)
    v2 = ti.Vector.zero(dt, 4)
    if gap > 0:
        lambda1 = ti.Vector([tr + ti.sqrt(gap), ti.cast(0.0, dt)]) * 0.5
        lambda2 = ti.Vector([tr - ti.sqrt(gap), ti.cast(0.0, dt)]) * 0.5
        A1 = A - lambda1[0] * ti.Matrix.identity(dt, 2)
        A2 = A - lambda2[0] * ti.Matrix.identity(dt, 2)
        if all(A1 == ti.Matrix.zero(dt, 2, 2)) and all(
                A1 == ti.Matrix.zero(dt, 2, 2)):
            v1 = ti.Vector([0.0, 0.0, 1.0, 0.0]).cast(dt)
            v2 = ti.Vector([1.0, 0.0, 0.0, 0.0]).cast(dt)
        else:
            v1 = ti.Vector([A2[0, 0], 0.0, A2[1, 0],
                            0.0]).cast(dt).normalized()
            v2 = ti.Vector([A1[0, 0], 0.0, A1[1, 0],
                            0.0]).cast(dt).normalized()
    else:
        lambda1 = ti.Vector([tr, ti.sqrt(-gap)]) * 0.5
        lambda2 = ti.Vector([tr, -ti.sqrt(-gap)]) * 0.5
        A1r = A - lambda1[0] * ti.Matrix.identity(dt, 2)
        A1i = -lambda1[1] * ti.Matrix.identity(dt, 2)
        A2r = A - lambda2[0] * ti.Matrix.identity(dt, 2)
        A2i = -lambda2[1] * ti.Matrix.identity(dt, 2)
        v1 = ti.Vector([A2r[0, 0], A2i[0, 0], A2r[1, 0],
                        A2i[1, 0]]).cast(dt).normalized()
        v2 = ti.Vector([A1r[0, 0], A1i[0, 0], A1r[1, 0],
                        A1i[1, 0]]).cast(dt).normalized()
    eigenvalues = ti.Matrix.rows([lambda1, lambda2])
    eigenvectors = ti.Matrix.cols([v1, v2])

    return eigenvalues, eigenvectors


@ti.func
def sym_eig2x2(A, dt):
    tr = A.trace()
    det = A.determinant()
    gap = tr**2 - 4 * det
    lambda1 = (tr + ti.sqrt(gap)) * 0.5
    lambda2 = (tr - ti.sqrt(gap)) * 0.5
    eigenvalues = ti.Vector([lambda1, lambda2]).cast(dt)

    A1 = A - lambda1 * ti.Matrix.identity(dt, 2)
    A2 = A - lambda2 * ti.Matrix.identity(dt, 2)
    v1 = ti.Vector.zero(dt, 2)
    v2 = ti.Vector.zero(dt, 2)
    if all(A1 == ti.Matrix.zero(dt, 2, 2)) and all(
            A1 == ti.Matrix.zero(dt, 2, 2)):
        v1 = ti.Vector([0.0, 1.0]).cast(dt)
        v2 = ti.Vector([1.0, 0.0]).cast(dt)
    else:
        v1 = ti.Vector([A2[0, 0], A2[1, 0]]).cast(dt).normalized()
        v2 = ti.Vector([A1[0, 0], A1[1, 0]]).cast(dt).normalized()
    eigenvectors = ti.Matrix.cols([v1, v2])
    return eigenvalues, eigenvectors


@ti.func
def svd(A, dt):
    if ti.static(A.n == 2):
        ret = svd2d(A, dt)
        return ret
    elif ti.static(A.n == 3):
        return svd3d(A, dt)
    else:
        raise Exception("SVD only supports 2D and 3D matrices.")


@ti.func
def polar_decompose(A, dt):
    if ti.static(A.n == 2):
        ret = polar_decompose2d(A, dt)
        return ret
    elif ti.static(A.n == 3):
        return polar_decompose3d(A, dt)
    else:
        raise Exception(
            "Polar decomposition only supports 2D and 3D matrices.")
