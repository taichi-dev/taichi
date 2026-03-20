
import numpy as np

# Imports/Code to evaluate basis functions *series* for a defined set of coefficients
from numpy.polynomial.laguerre import lagval
from numpy.polynomial.hermite import hermval
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.legendre import legval

def fourval(x, coeffs):
    """
    Evaluate the Fourier series of a given set of coefficients.

    Parameters:
    x: array-like - Input values (stock prices or other variables)
    coeffs: array-like - Coefficients of the Fourier series

    Returns:
    array - Fourier basis functions evaluated at x
    """
    coeffs = np.asarray(coeffs)
    F = np.zeros_like(x, dtype=np.result_type(x, coeffs))
    
    if coeffs.shape[0] > 0: # We have more than 0 coeffs
        F = F + 0.5 * coeffs[0] # First term is constant
    for i in range(1, coeffs.shape[0]):
        k = (i + 1) // 2
        if i % 2 == 1:
            F = F + coeffs[i] * np.cos(k * x)
        else:
            F = F + coeffs[i] * np.sin(k * x)
    return F





# Code to evaluate (weighted) basis function *matrices*
def lagmatrix(x, x_length, num_basis_functions, use_orth_weight):
    """
    Compute a set of Laguerre polynomials with custom weight.

    Parameters:
    x: array-like - Input values (stock prices or other variables)
    x_length: int - Length of the input array
    num_basis_functions: int - Degree of the Laguerre polynomial to compute
    use_orth_weight: bool - Whether to use the orthogonality weight

    Returns:
    array - Laguerre basis functions evaluated at x
    """
    weight = 1 if not use_orth_weight else np.exp(-x / 2)
    L = np.zeros((x_length, num_basis_functions))  # Create an array for the basis functions
    for i in range(num_basis_functions):
        L[:, i] = weight * lagval(x, [0] * i + [1])  # Evaluate Laguerre polynomial i. If I remove np.exp(-x/2) * then I get the same Laguerre polynomials as QuantLib.
    return L

def hermmatrix(x, x_length, num_basis_functions, use_orth_weight):
    """
    Compute a set of Hermite polynomials with custom weight.

    Parameters:
    x: array-like - Input values (stock prices or other variables)
    x_length: int - Length of the input array
    num_basis_functions: int - Degree of the Laguerre polynomial to compute
    use_orth_weight: bool - Whether to use the orthogonality weight

    Returns:
    array - Laguerre basis functions evaluated at x
    """
    weight = 1 if not use_orth_weight else np.exp(-x**2 / 2)
    L = np.zeros((x_length, num_basis_functions))  # Create an array for the basis functions
    for i in range(num_basis_functions):
        L[:, i] =  weight * hermval(x, [0] * i + [1])  # Evaluate Laguerre polynomial i. If I remove np.exp(-x/2) * then I get the same Laguerre polynomials as QuantLib.
    return L

def chebmatrix(x, x_length, num_basis_functions, use_orth_weight):
    """
    Compute a set of Chebyshev polynomials with custom weight.

    Parameters:
    x: array-like - Input values (stock prices or other variables)
    x_length: int - Length of the input array
    num_basis_functions: int - Degree of the Laguerre polynomial to compute
    use_orth_weight: bool - Whether to use the orthogonality weight

    Returns:
    array - Laguerre basis functions evaluated at x
    """
    weight = 1 if not use_orth_weight else (1-x**2)**(-1/4)
    L = np.zeros((x_length, num_basis_functions))  # Create an array for the basis functions
    for i in range(num_basis_functions):
        L[:, i] =  weight * chebval(x, [0] * i + [1])  # Evaluate Laguerre polynomial i. If I remove np.exp(-x/2) * then I get the same Laguerre polynomials as QuantLib.
    return L

def legmatrix(x, x_length, num_basis_functions, _):
    """
    Compute a set of Legendre polynomials.

    Parameters:
    x: array-like - Input values (stock prices or other variables)
    num_basis_functions: int - Degree of the Laguerre polynomial to compute

    Returns:
    array - Laguerre basis functions evaluated at x
    """
    L = np.zeros((x_length, num_basis_functions))  # Create an array for the basis functions
    for i in range(num_basis_functions):
        L[:, i] =  legval(x, [0] * i + [1])  
    return L

def polymatrix(x, x_length, num_basis_functions, _):
    """
    Compute a set of monomials.

    Parameters:
    x: array-like - Input values (stock prices or other variables)
    num_basis_functions: int - Degree of the Laguerre polynomial to compute

    Returns:
    array - Laguerre basis functions evaluated at x
    """
    L = np.zeros((x_length, num_basis_functions))  # Create an array for the basis functions
    for i in range(num_basis_functions):
        L[:, i] =  x**i  
    return L

def fourmatrix(x, x_length, num_basis_functions, _):
    """
    Construct the first `num_basis_functions` Fourier basis functions sampled at x:

      [constant,
       cos(1⋅x), sin(1⋅x),
       cos(2⋅x), sin(2⋅x),
       …]

    Parameters
    ----------
    x : array-like, shape (N,)
        Array of input locations at which to evaluate basis functions (does not need to be in a particular range).
    x_length : int
        Number of points in x (for shape of output array).
    num_basis_functions : int
        Number of basis functions to include (length of the returned basis).

    Returns
    -------
    F : ndarray, shape (N, num_basis_functions)
        Matrix of basis functions evaluated at `x`. Columns are ordered:
            - F[:, 0] = constant term (1/2),
            - F[:, 1] = cos(1⋅x),
            - F[:, 2] = sin(1⋅x),
            - F[:, 3] = cos(2⋅x),
            - F[:, 4] = sin(2⋅x),
            ...
        This is the design matrix for projecting onto the truncated Fourier basis.
    """
    F = np.zeros((x_length, num_basis_functions))

    # constant term (first basis function)
    if num_basis_functions > 0:
        F[:, 0] = 1/2

    # Subsequent basis: cos/sin pairs
    for i in range(1, num_basis_functions):
        F[:, i] = fourval(x,  [0] * i + [1])

    return F

