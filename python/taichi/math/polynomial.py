"""
Math polynomial module.
"""
from taichi.lang.kernel_impl import func
from taichi.types import template
from taichi.lang.ops import (cos, sin, exp, sqrt)
from taichi.lang.impl import static


@func
def lagval(x_field: template(), c_field: template()):
    """
    Evaluate a Laguerre series in-place on a 1D Taichi array.

    For coefficients ``c_field`` of length ``n + 1``, this computes

    .. math::
        p(x) = \\sum_{k=0}^{n} c_k L_k(x)

    for every element in ``x_field`` and writes the result back into
    ``x_field``.

    This function is intended to be called inside a Taichi kernel,
    much like other math functions in Taichi's math module.
    The evaluation uses Clenshaw recursion, and closely 
    follows the numpy lagval implementation.

    Parameters
    ----------
    x_field : template
        1D mutable container (for example ``ti.ndarray`` or templated field)
        containing the input ``x`` values. Updated in-place.
    c_field : template
        1D coefficient container where ``c_field[k]`` is the coefficient
        of :math:`L_k(x)`.

    Returns
    -------
    template
        The same container as ``x_field`` after in-place update.
    """

    c_len = c_field.shape[0]

    if c_len == 1:
        for j in range(x_field.shape[0]):
            x_field[j] = c_field[0]
    elif c_len == 2:
        for j in range(x_field.shape[0]):
            x_field[j] = c_field[0] + c_field[1]*(1 - x_field[j])
    else:
        for j in range(x_field.shape[0]):
            nd = c_len
            c0 = c_field[c_len-2]
            c1 = c_field[c_len-1]
            for i in range(3, c_len + 1):
                tmp = c0
                nd = nd - 1
                c0 = c_field[c_len-i] - (c1*(nd - 1))/nd
                c1 = tmp + (c1*((2*nd - 1) - x_field[j]))/nd
            x_field[j] = c0 + c1*(1 - x_field[j])

    return x_field

@func
def hermval(x_field: template(), c_field: template()):
    """
    Evaluate an Hermite series in-place on a 1D Taichi array.

    For coefficients ``c_field`` of length ``n + 1``, this computes

    .. math::
        p(x) = \\sum_{k=0}^{n} c_k H_k(x)

    for every element in ``x_field`` and writes the result back into
    ``x_field``.

    This function is intended to be called inside a Taichi kernel,
    much like other math functions in Taichi's math module.
    The evaluation uses Clenshaw recursion, and closely 
    follows the numpy hermval implementation.

    Parameters
    ----------
    x_field : template
        1D mutable container (for example ``ti.ndarray`` or templated field)
        containing the input ``x`` values. Updated in-place.
    c_field : template
        1D coefficient container where ``c_field[k]`` is the coefficient
        of :math:`H_k(x)`.

    Returns
    -------
    template
        The same container as ``x_field`` after in-place update.

    """
    c_len = c_field.shape[0]
    if c_len == 1:
        for j in range(x_field.shape[0]):
            x_field[j] = c_field[0]
    elif c_len == 2:
        for j in range(x_field.shape[0]):
            x_field[j] = c_field[0] + c_field[1]*x_field[j]*2
    else:
        for j in range(x_field.shape[0]):
            nd = c_len
            c0 = c_field[c_len-2]
            c1 = c_field[c_len-1]
            for i in range(3, c_len + 1):
                tmp = c0
                nd = nd - 1
                c0 = c_field[c_len-i] - c1*(2*(nd - 1))
                c1 = tmp + c1*x_field[j]*2
            x_field[j] = c0 + c1*x_field[j]*2
    return x_field

@func
def chebval(x_field: template(), c_field: template()):
    """
    Evaluate a Chebyshev series in-place on a 1D Taichi array.

    For coefficients ``c_field`` of length ``n + 1``, this computes

    .. math:: p(x) = \\sum_{k=0}^{n} c_k T_k(x)

    for every element in ``x_field`` and writes the result back into
    ``x_field``.

    This function is intended to be called inside a Taichi kernel,
    much like other math functions in Taichi's math module.
    The evaluation uses Clenshaw recursion, and closely 
    follows the numpy chebval implementation.

    Parameters
    ----------
    x_field : template
        1D mutable container (for example ``ti.ndarray`` or templated field)
        containing the input ``x`` values. Updated in-place.
    c_field : template
        1D coefficient container where ``c_field[k]`` is the coefficient
        of :math:`T_k(x)`.

    Returns
    -------
    template
        The same container as ``x_field`` after in-place update.

    """
    c_len = c_field.shape[0]

    if c_len == 1:
        for j in range(x_field.shape[0]):
            x_field[j] = c_field[0]
    elif c_len == 2:
        for j in range(x_field.shape[0]):
            x_field[j] = c_field[0] + c_field[1]*x_field[j]
    else:
        for j in range(x_field.shape[0]):
            c0 = c_field[c_len-2]
            c1 = c_field[c_len-1]
            for i in range(3, c_len + 1):
                tmp = c0
                c0 = c_field[c_len-i] - c1
                c1 = tmp + c1*2*x_field[j]
            x_field[j] = c0 + c1*x_field[j]
    return x_field

@func
def legval(x_field: template(), c_field: template()):
    """
    Evaluate a Legendre series in-place on a 1D Taichi array.

    For coefficients ``c_field`` of length ``n + 1``, this computes

    .. math:: p(x) = \\sum_{k=0}^{n} c_k P_k(x)

    for every element in ``x_field`` and writes the result back into
    ``x_field``.

    This function is intended to be called inside a Taichi kernel,
    much like other math functions in Taichi's math module.
    The evaluation uses Clenshaw recursion, and closely 
    follows the numpy legval implementation.

    Parameters
    ----------
    x_field : template
        1D mutable container (for example ``ti.ndarray`` or templated field)
        containing the input ``x`` values. Updated in-place.
    c_field : template
        1D coefficient container where ``c_field[k]`` is the coefficient
        of :math:`P_k(x)`.

    Returns
    -------
    template
        The same container as ``x_field`` after in-place update.

    """
    c_len = c_field.shape[0]

    if c_len == 1:
        for j in range(x_field.shape[0]):
            x_field[j] = c_field[0]
    elif c_len == 2:
        for j in range(x_field.shape[0]):
            x_field[j] = c_field[0] + c_field[1]*x_field[j]
    else:
        for j in range(x_field.shape[0]):
            nd = c_len
            c0 = c_field[c_len-2]
            c1 = c_field[c_len-1]
            for i in range(3, c_len + 1):
                tmp = c0
                nd = nd - 1
                c0 = c_field[c_len-i] - (c1*(nd - 1))/nd
                c1 = tmp + (c1*x_field[j]*(2*nd - 1))/nd
            x_field[j] = c0 + c1*x_field[j]

    return x_field

@func
def polyval(x_field: template(), c_field: template()):
    """
    Evaluate a power series in-place on a 1D Taichi array.

    For coefficients ``c_field`` of length ``n + 1``, this computes

    .. math:: p(x) = \\sum_{k=0}^{n} c_k x^k

    for every element in ``x_field`` and writes the result back into
    ``x_field``.

    This function is intended to be called inside a Taichi kernel,
    much like other math functions in Taichi's math module.
    The evaluation uses Horner's method, and closely 
    follows the numpy polyval implementation.

    Parameters
    ----------
    x_field : template
        1D mutable container (for example ``ti.ndarray`` or templated field)
        containing the input ``x`` values. Updated in-place.
    c_field : template
        1D coefficient container where ``c_field[k]`` is the coefficient
        of :math:`x^k`.

    Returns
    -------
    template
        The same container as ``x_field`` after in-place update.

    """

    c_len = c_field.shape[0]

    for j in range(x_field.shape[0]):
        c0 = c_field[c_len-1]
        for i in range(2, c_len + 1):
            c0 = c_field[c_len-i] + c0*x_field[j]
        x_field[j] = c0
    
    return x_field

@func
def fourval(x_field: template(), c_field: template()):
    """
    Evaluate a Fourier series in-place on a 1D Taichi array.

    Coefficients are interpreted in the order:
    ``[c0, cos(1*x), sin(1*x), cos(2*x), sin(2*x), ...]``.
    For every element in ``x_field``, this computes the Fourier series value
    and writes it back into ``x_field``.

    Parameters
    ----------
    x_field : template
        1D mutable container (for example ``ti.ndarray`` or templated field)
        containing the input ``x`` values. Updated in-place.
    c_field : template
        1D coefficient container in interleaved cosine/sine order.

    Returns
    -------
    template
        The same container as ``x_field`` after in-place update.
    """
    
    c_len = c_field.shape[0]

    if c_len == 1:
        for j in range(x_field.shape[0]):
            x_field[j] = 0.5 * c_field[0]
    else:
        k_max = c_len // 2
        for j in range(x_field.shape[0]):
            cx = cos(x_field[j])
            sx = sin(x_field[j])

            bc1 = 0.0
            bc2 = 0.0
            bs1 = 0.0
            bs2 = 0.0

            for k in range(k_max):
                ia = 2 * (k_max-k)

                ak = c_field[ia-1] if (ia - 1) < c_len else 0.0
                bk = c_field[ia] if ia < c_len else 0.0

                bc0 = ak + 2.0 * cx * bc1 - bc2
                bs0 = bk + 2.0 * cx * bs1 - bs2

                bc2 = bc1
                bc1 = bc0
                bs2 = bs1
                bs1 = bs0

            x_field[j] = 0.5 * c_field[0] + (bc1 * cx - bc2) + bs1 * sx

    return x_field


@func
def lagmatrix(x_field: template(), use_orth_weight: template(), matrix_field: template()):
    """
    Build a Laguerre pseudo-Vandermonde matrix in-place.

    For each input sample ``x_field[i]`` and degree ``j``, this writes

    .. math::
        \\text{matrix\\_field}[i, j] = w(x_i) L_j(x_i), \\quad 0 \\le j \\le deg,

    where ``deg = matrix_field.shape[1] - 1``,
    ``w(x) = exp(-x/2)`` when ``use_orth_weight`` is ``True``, and
    ``w(x) = 1`` otherwise.

    Parameters
    ----------
    x_field : template
        1D input container of ``x`` values.
    use_orth_weight : bool
        If ``True``, apply the Laguerre orthogonality weight ``exp(-x/2)``.
    matrix_field : template
        2D output container written in-place. Row count should match
        ``x_field.shape[0]``. This implementation assumes at least 2 columns.
        Reason: column 0 is the trivial base term (``w(x)`` or ``1``).

    Returns
    -------
    template
        The same container as ``matrix_field`` after in-place update.
    """
    
    for i in range(matrix_field.shape[0]):
        if static(use_orth_weight):
            matrix_field[i, 0] = exp(-0.5 * x_field[i])
            matrix_field[i, 1] = matrix_field[i, 0] * (1.0 - x_field[i])
        else:
           matrix_field[i, 0] = 1.0
           matrix_field[i, 1] = 1.0 - x_field[i]

        for j in range(2, matrix_field.shape[1]):
            matrix_field[i, j] = ((2 * j - 1 - x_field[i]) *  matrix_field[i, j-1] - (j - 1) * matrix_field[i, j-2]) / j

    return matrix_field

@func
def hermmatrix(x_field: template(), use_orth_weight: template(), matrix_field: template()):
    """
    Build a Hermite pseudo-Vandermonde matrix in-place.

    For each input sample ``x_field[i]`` and degree ``j``, this writes

    .. math::
        \\text{matrix\\_field}[i, j] = w(x_i) H_j(x_i), \\quad 0 \\le j \\le deg,

    where ``deg = matrix_field.shape[1] - 1``,
    ``w(x) = exp(-x^2/2)`` when ``use_orth_weight`` is ``True``, and
    ``w(x) = 1`` otherwise.

    Parameters
    ----------
    x_field : template
        1D input container of ``x`` values.
    use_orth_weight : bool
        If ``True``, apply the Hermite root-weight ``exp(-x^2/2)``.
    matrix_field : template
        2D output container written in-place. Row count should match
        ``x_field.shape[0]``. This implementation assumes at least 2 columns.

    Returns
    -------
    template
        The same container as ``matrix_field`` after in-place update.
    """
    for i in range(matrix_field.shape[0]):
        matrix_field[i, 0] = exp(-0.5 * x_field[i] * x_field[i]) if static(use_orth_weight) else 1.0
        matrix_field[i, 1] = matrix_field[i, 0] * (2.0 * x_field[i])

        for j in range(2, matrix_field.shape[1]):
            matrix_field[i, j] = matrix_field[i, j-1] * (2.0 * x_field[i]) - matrix_field[i, j-2] * (2 * (j - 1))

    return matrix_field

@func
def chebmatrix(x_field: template(), use_orth_weight: template(), matrix_field: template()):
    """
    Build a Chebyshev pseudo-Vandermonde matrix in-place.

    For each input sample ``x_field[i]`` and degree ``j``, this writes

    .. math::
        \\text{matrix\\_field}[i, j] = w(x_i) T_j(x_i), \\quad 0 \\le j \\le deg,

    where ``deg = matrix_field.shape[1] - 1``,
    ``w(x) = (1 - x^2)^(-1/4)`` when ``use_orth_weight`` is ``True``, and
    ``w(x) = 1`` otherwise.

    Parameters
    ----------
    x_field : template
        1D input container of ``x`` values.
    use_orth_weight : template
        If ``True``, apply the Chebyshev orthogonality weight
        ``(1 - x^2)^(-1/4)``.
    matrix_field : template
        2D output container written in-place. Row count should match
        ``x_field.shape[0]``. This implementation assumes at least 2 columns.

    Returns
    -------
    template
        The same container as ``matrix_field`` after in-place update.
    """
    for i in range(matrix_field.shape[0]):
        if static(use_orth_weight):
            matrix_field[i, 0] = 1.0 / sqrt(sqrt(1.0 - x_field[i] * x_field[i]))
            matrix_field[i, 1] = matrix_field[i, 0] * x_field[i]
        else:
            matrix_field[i, 0] = 1.0
            matrix_field[i, 1] = x_field[i]

        for j in range(2, matrix_field.shape[1]):
            matrix_field[i, j] = matrix_field[i, j - 1] * (2.0 * x_field[i]) - matrix_field[i, j - 2]

    return matrix_field

@func
def legmatrix(x_field: template(), matrix_field: template()):
    """
    Build a Legendre pseudo-Vandermonde matrix in-place.

    For each input sample ``x_field[i]`` and degree ``j``, this writes

    .. math::
        \\text{matrix\\_field}[i, j] = P_j(x_i), \\quad 0 \\le j \\le deg,

    where ``deg = matrix_field.shape[1] - 1``.

    Parameters
    ----------
    x_field : template
        1D input container of ``x`` values.
    matrix_field : template
        2D output container written in-place. Row count should match
        ``x_field.shape[0]``. This implementation assumes at least 2 columns.

    Returns
    -------
    template
        The same container as ``matrix_field`` after in-place update.
    """
    for i in range(matrix_field.shape[0]):
        matrix_field[i, 0] = 1.0
        matrix_field[i, 1] = x_field[i]

        for j in range(2, matrix_field.shape[1]):
            matrix_field[i, j] = (
                matrix_field[i, j - 1] * x_field[i] * (2 * j - 1)
                - matrix_field[i, j - 2] * (j - 1)
            ) / j

    return matrix_field

@func
def polymatrix(x_field: template(), matrix_field: template()):
    """
    Build a power-basis Vandermonde matrix in-place.

    For each input sample ``x_field[i]`` and degree ``j``, this writes

    .. math::
        \\text{matrix\\_field}[i, j] = x_i^j, \\quad 0 \\le j \\le deg,

    where ``deg = matrix_field.shape[1] - 1``.

    Parameters
    ----------
    x_field : template
        1D input container of ``x`` values.
    matrix_field : template
        2D output container written in-place. Row count should match
        ``x_field.shape[0]``. This implementation assumes at least 2 columns.

    Returns
    -------
    template
        The same container as ``matrix_field`` after in-place update.
    """
    for i in range(matrix_field.shape[0]):
        matrix_field[i, 0] = 1.0
        matrix_field[i, 1] = x_field[i]

        for j in range(2, matrix_field.shape[1]):
            matrix_field[i, j] = matrix_field[i, j - 1] * x_field[i]

    return matrix_field

@func
def fourmatrix(x_field: template(), matrix_field: template()):
    """
    Build a trigonometric design matrix in-place (Fourier basis).

    Column order matches ``fourval`` coefficients:
    ``[c0, cos(1*x), sin(1*x), cos(2*x), sin(2*x), ...]``,
    with the constant term represented as ``0.5 * c0``.

    Parameters
    ----------
    x_field : template
        1D input container of ``x`` values.
    matrix_field : template
        2D output container written in-place. Row count should match
        ``x_field.shape[0]``.

    Returns
    -------
    template
        The same container as ``matrix_field`` after in-place update.
    """
    for i in range(matrix_field.shape[0]):
        matrix_field[i, 0] = 0.5

        if matrix_field.shape[1] > 1:
            matrix_field[i, 1] = cos(x_field[i])

        if matrix_field.shape[1] > 2:
            matrix_field[i, 2] = sin(x_field[i])

        for k in range(2, matrix_field.shape[1] // 2 + 1):
            matrix_field[i, 2 * k - 1] = (
                matrix_field[i, 2 * k - 3] * matrix_field[i, 1]
                - matrix_field[i, 2 * k - 2] * matrix_field[i, 2]
            )
            if 2 * k < matrix_field.shape[1]:
                matrix_field[i, 2 * k] = (
                    matrix_field[i, 2 * k - 2] * matrix_field[i, 1]
                    + matrix_field[i, 2 * k - 3] * matrix_field[i, 2]
                )

    return matrix_field





__all__ = ["lagval", "hermval", "chebval", "legval", "polyval", "fourval", "lagmatrix", "hermmatrix", "chebmatrix", "legmatrix", "polymatrix", "fourmatrix"]
