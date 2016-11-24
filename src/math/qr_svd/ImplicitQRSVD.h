/**
Copyright (c) 2016 Theodore Gast, Chuyuan Fu, Chenfanfu Jiang, Joseph Teran

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

If the code is used in an article, the following paper shall be cited:
@techreport{qrsvd:2016,
  title={Implicit-shifted Symmetric QR Singular Value Decomposition of 3x3 Matrices},
  author={Gast, Theodore and Fu, Chuyuan and Jiang, Chenfanfu and Teran, Joseph},
  year={2016},
  institution={University of California Los Angeles}
}

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

################################################################################
This file implements 2D and 3D polar decompositions and SVDs.

T may be float or double.

2D Polar:
    Eigen::Matrix<T, 2, 2> A,R,S;
    A<<1,2,3,4;
    JIXIE::polarDecomposition(A, R, S);
    // R will be the closest rotation to A
    // S will be symmetric

2D SVD:
    Eigen::Matrix<T, 2, 2> A;
    A<<1,2,3,4;
    Eigen::Matrix<T, 2, 1> S;
    Eigen::Matrix<T, 2, 2> U;
    Eigen::Matrix<T, 2, 2> V;
    JIXIE::singularValueDecomposition(A,U,S,V);
    // A = U S V'
    // U and V will be rotations
    // S will be singular values sorted by decreasing magnitude. Only the last one may be negative.

3D Polar:
    Eigen::Matrix<T, 3, 3> A,R,S;
    A<<1,2,3,4,5,6;
    JIXIE::polarDecomposition(A, R, S);
    // R will be the closest rotation to A
    // S will be symmetric

3D SVD:
    Eigen::Matrix<T, 3, 3> A;
    A<<1,2,3,4,5,6;
    Eigen::Matrix<T, 3, 1> S;
    Eigen::Matrix<T, 3, 3> U;
    Eigen::Matrix<T, 3, 3> V;
    JIXIE::singularValueDecomposition(A,U,S,V);
    // A = U S V'
    // U and V will be rotations
    // S will be singular values sorted by decreasing magnitude. Only the last one may be negative.

################################################################################
*/

/**
SVD based on implicit QR with Wilkinson Shift
*/
#ifndef JIXIE_IMPLICIT_QR_SVD_H
#define JIXIE_IMPLICIT_QR_SVD_H

#include "Tools.h"

namespace JIXIE {

/**
    Class for givens rotation.
    Row rotation G*A corresponds to something like
    c -s  0
    ( s  c  0 ) A
    0  0  1
    Column rotation A G' corresponds to something like
    c -s  0
    A ( s  c  0 )
    0  0  1

    c and s are always computed so that
    ( c -s ) ( a )  =  ( * )
    s  c     b       ( 0 )

    Assume rowi<rowk.
    */
template <class T>
class GivensRotation {
public:
    int rowi;
    int rowk;
    T c;
    T s;

    inline GivensRotation(int rowi_in, int rowk_in)
        : rowi(rowi_in)
        , rowk(rowk_in)
        , c(1)
        , s(0)
    {
    }

    inline GivensRotation(T a, T b, int rowi_in, int rowk_in)
        : rowi(rowi_in)
        , rowk(rowk_in)
    {
        compute(a, b);
    }

    ~GivensRotation() {}

    inline void transposeInPlace()
    {
        s = -s;
    }

    /**
        Compute c and s from a and b so that
        ( c -s ) ( a )  =  ( * )
        s  c     b       ( 0 )
        */
    inline void compute(const T a, const T b)
    {
        using std::sqrt;

        T d = a * a + b * b;
        c = 1;
        s = 0;
        if (d != 0) {
            // T t = 1 / sqrt(d);
            T t = JIXIE::MATH_TOOLS::rsqrt(d);
            c = a * t;
            s = -b * t;
        }
    }

    /**
        This function computes c and s so that
        ( c -s ) ( a )  =  ( 0 )
        s  c     b       ( * )
        */
    inline void computeUnconventional(const T a, const T b)
    {
        using std::sqrt;

        T d = a * a + b * b;
        c = 0;
        s = 1;
        if (d != 0) {
            // T t = 1 / sqrt(d);
            T t = JIXIE::MATH_TOOLS::rsqrt(d);
            s = a * t;
            c = b * t;
        }
    }
    /**
      Fill the R with the entries of this rotation
        */
    template <class MatrixType>
    inline void fill(const MatrixType& R) const
    {
        MatrixType& A = const_cast<MatrixType&>(R);
        A = MatrixType::Identity();
        A(rowi, rowi) = c;
        A(rowk, rowi) = -s;
        A(rowi, rowk) = s;
        A(rowk, rowk) = c;
    }

    /**
        This function does something like
        c -s  0
        ( s  c  0 ) A -> A
        0  0  1
        It only affects row i and row k of A.
        */
    template <class MatrixType>
    inline void rowRotation(MatrixType& A) const
    {
        for (int j = 0; j < MatrixType::ColsAtCompileTime; j++) {
            T tau1 = A(rowi, j);
            T tau2 = A(rowk, j);
            A(rowi, j) = c * tau1 - s * tau2;
            A(rowk, j) = s * tau1 + c * tau2;
        }
    }

    /**
        This function does something like
        c  s  0
        A ( -s  c  0 )  -> A
        0  0  1
        It only affects column i and column k of A.
        */
    template <class MatrixType>
    inline void columnRotation(MatrixType& A) const
    {
        for (int j = 0; j < MatrixType::RowsAtCompileTime; j++) {
            T tau1 = A(j, rowi);
            T tau2 = A(j, rowk);
            A(j, rowi) = c * tau1 - s * tau2;
            A(j, rowk) = s * tau1 + c * tau2;
        }
    }

    /**
      Multiply givens must be for same row and column
      **/
    inline void operator*=(const GivensRotation<T>& A)
    {
        T new_c = c * A.c - s * A.s;
        T new_s = s * A.c + c * A.s;
        c = new_c;
        s = new_s;
    }

    /**
      Multiply givens must be for same row and column
      **/
    inline GivensRotation<T> operator*(const GivensRotation<T>& A) const
    {
        GivensRotation<T> r(*this);
        r *= A;
        return r;
    }
};

/**
    \brief zero chasing the 3X3 matrix to bidiagonal form
    original form of H:   x x 0
    x x x
    0 0 x
    after zero chase:
    x x 0
    0 x x
    0 0 x
    */
template <class T>
inline void zeroChase(Eigen::Matrix<T, 3, 3>& H, Eigen::Matrix<T, 3, 3>& U, Eigen::Matrix<T, 3, 3>& V)
{

    /**
        Reduce H to of form
        x x +
        0 x x
        0 0 x
        */
    GivensRotation<T> r1(H(0, 0), H(1, 0), 0, 1);
    /**
        Reduce H to of form
        x x 0
        0 x x
        0 + x
        Can calculate r2 without multiplying by r1 since both entries are in first two
        rows thus no need to divide by sqrt(a^2+b^2)
        */
    GivensRotation<T> r2(1, 2);
    if (H(1, 0) != 0)
        r2.compute(H(0, 0) * H(0, 1) + H(1, 0) * H(1, 1), H(0, 0) * H(0, 2) + H(1, 0) * H(1, 2));
    else
        r2.compute(H(0, 1), H(0, 2));

    r1.rowRotation(H);

    /* GivensRotation<T> r2(H(0, 1), H(0, 2), 1, 2); */
    r2.columnRotation(H);
    r2.columnRotation(V);

    /**
        Reduce H to of form
        x x 0
        0 x x
        0 0 x
        */
    GivensRotation<T> r3(H(1, 1), H(2, 1), 1, 2);
    r3.rowRotation(H);

    // Save this till end for better cache coherency
    // r1.rowRotation(u_transpose);
    // r3.rowRotation(u_transpose);
    r1.columnRotation(U);
    r3.columnRotation(U);
}

/**
     \brief make a 3X3 matrix to upper bidiagonal form
     original form of H:   x x x
                           x x x
                           x x x
     after zero chase:
                           x x 0
                           0 x x
                           0 0 x
  */
template <class T>
inline void makeUpperBidiag(Eigen::Matrix<T, 3, 3>& H, Eigen::Matrix<T, 3, 3>& U, Eigen::Matrix<T, 3, 3>& V)
{
    U = Eigen::Matrix<T, 3, 3>::Identity();
    V = Eigen::Matrix<T, 3, 3>::Identity();

    /**
      Reduce H to of form
                          x x x
                          x x x
                          0 x x
    */

    GivensRotation<T> r(H(1, 0), H(2, 0), 1, 2);
    r.rowRotation(H);
    // r.rowRotation(u_transpose);
    r.columnRotation(U);
    // zeroChase(H, u_transpose, V);
    zeroChase(H, U, V);
}

/**
     \brief make a 3X3 matrix to lambda shape
     original form of H:   x x x
     *                     x x x
     *                     x x x
     after :
     *                     x 0 0
     *                     x x 0
     *                     x 0 x
  */
template <class T>
inline void makeLambdaShape(Eigen::Matrix<T, 3, 3>& H, Eigen::Matrix<T, 3, 3>& U, Eigen::Matrix<T, 3, 3>& V)
{
    U = Eigen::Matrix<T, 3, 3>::Identity();
    V = Eigen::Matrix<T, 3, 3>::Identity();

    /**
      Reduce H to of form
      *                    x x 0
      *                    x x x
      *                    x x x
      */

    GivensRotation<T> r1(H(0, 1), H(0, 2), 1, 2);
    r1.columnRotation(H);
    r1.columnRotation(V);

    /**
      Reduce H to of form
      *                    x x 0
      *                    x x 0
      *                    x x x
      */

    r1.computeUnconventional(H(1, 2), H(2, 2));
    r1.rowRotation(H);
    r1.columnRotation(U);

    /**
      Reduce H to of form
      *                    x x 0
      *                    x x 0
      *                    x 0 x
      */

    GivensRotation<T> r2(H(2, 0), H(2, 1), 0, 1);
    r2.columnRotation(H);
    r2.columnRotation(V);

    /**
      Reduce H to of form
      *                    x 0 0
      *                    x x 0
      *                    x 0 x
      */
    r2.computeUnconventional(H(0, 1), H(1, 1));
    r2.rowRotation(H);
    r2.columnRotation(U);
}

/**
   \brief 2x2 polar decomposition.
   \param[in] A matrix.
   \param[out] R Robustly a rotation matrix in givens form
   \param[out] S_Sym Symmetric. Whole matrix is stored

   Whole matrix S is stored since its faster to calculate due to simd vectorization
   Polar guarantees negative sign is on the small magnitude singular value.
   S is guaranteed to be the closest one to identity.
   R is guaranteed to be the closest rotation to A.
*/
template <class TA, class T, class TS>
inline std::enable_if_t<isSize<TA>(2, 2) && isSize<TS>(2, 2)>
polarDecomposition(const Eigen::MatrixBase<TA>& A,
    GivensRotation<T>& R,
    const Eigen::MatrixBase<TS>& S_Sym)
{
    Eigen::Matrix<T, 2, 1> x(A(0, 0) + A(1, 1), A(1, 0) - A(0, 1));
    T denominator = x.norm();
    R.c = (T)1;
    R.s = (T)0;
    if (denominator != 0) {
        /*
          No need to use a tolerance here because x(0) and x(1) always have
          smaller magnitude then denominator, therefore overflow never happens.
        */
        R.c = x(0) / denominator;
        R.s = -x(1) / denominator;
    }
    Eigen::MatrixBase<TS>& S = const_cast<Eigen::MatrixBase<TS>&>(S_Sym);
    S = A;
    R.rowRotation(S);
}

/**
   \brief 2x2 polar decomposition.
   \param[in] A matrix.
   \param[out] R Robustly a rotation matrix.
   \param[out] S_Sym Symmetric. Whole matrix is stored

   Whole matrix S is stored since its faster to calculate due to simd vectorization
   Polar guarantees negative sign is on the small magnitude singular value.
   S is guaranteed to be the closest one to identity.
   R is guaranteed to be the closest rotation to A.
*/
template <class TA, class TR, class TS>
inline std::enable_if_t<isSize<TA>(2, 2) && isSize<TR>(2, 2) && isSize<TS>(2, 2)>
polarDecomposition(const Eigen::MatrixBase<TA>& A,
    const Eigen::MatrixBase<TR>& R,
    const Eigen::MatrixBase<TS>& S_Sym)
{
    using T = ScalarType<TA>;
    GivensRotation<T> r(0, 1);
    polarDecomposition(A, r, S_Sym);
    r.fill(R);
}

/**
   \brief 2x2 SVD (singular value decomposition) A=USV'
   \param[in] A Input matrix.
   \param[out] U Robustly a rotation matrix in Givens form
   \param[out] Sigma Vector of singular values sorted with decreasing magnitude. The second one can be negative.
   \param[out] V Robustly a rotation matrix in Givens form
*/
template <class TA, class T, class Ts>
inline std::enable_if_t<isSize<TA>(2, 2) && isSize<Ts>(2, 1)>
singularValueDecomposition(
    const Eigen::MatrixBase<TA>& A,
    GivensRotation<T>& U,
    const Eigen::MatrixBase<Ts>& Sigma,
    GivensRotation<T>& V,
    const ScalarType<TA> tol = 64 * std::numeric_limits<ScalarType<TA> >::epsilon())
{
    using std::sqrt;
    Eigen::MatrixBase<Ts>& sigma = const_cast<Eigen::MatrixBase<Ts>&>(Sigma);

    Eigen::Matrix<T, 2, 2> S_Sym;
    polarDecomposition(A, U, S_Sym);
    T cosine, sine;
    T x = S_Sym(0, 0);
    T y = S_Sym(0, 1);
    T z = S_Sym(1, 1);
    if (y == 0) {
        // S is already diagonal
        cosine = 1;
        sine = 0;
        sigma(0) = x;
        sigma(1) = z;
    }
    else {
        T tau = 0.5 * (x - z);
        T w = sqrt(tau * tau + y * y);
        // w > y > 0
        T t;
        if (tau > 0) {
            // tau + w > w > y > 0 ==> division is safe
            t = y / (tau + w);
        }
        else {
            // tau - w < -w < -y < 0 ==> division is safe
            t = y / (tau - w);
        }
        cosine = T(1) / sqrt(t * t + T(1));
        sine = -t * cosine;
        /*
          V = [cosine -sine; sine cosine]
          Sigma = V'SV. Only compute the diagonals for efficiency.
          Also utilize symmetry of S and don't form V yet.
        */
        T c2 = cosine * cosine;
        T csy = 2 * cosine * sine * y;
        T s2 = sine * sine;
        sigma(0) = c2 * x - csy + s2 * z;
        sigma(1) = s2 * x + csy + c2 * z;
    }

    // Sorting
    // Polar already guarantees negative sign is on the small magnitude singular value.
    if (sigma(0) < sigma(1)) {
        std::swap(sigma(0), sigma(1));
        V.c = -sine;
        V.s = cosine;
    }
    else {
        V.c = cosine;
        V.s = sine;
    }
    U *= V;
}
/**
   \brief 2x2 SVD (singular value decomposition) A=USV'
   \param[in] A Input matrix.
   \param[out] U Robustly a rotation matrix.
   \param[out] Sigma Vector of singular values sorted with decreasing magnitude. The second one can be negative.
   \param[out] V Robustly a rotation matrix.
*/
template <class TA, class TU, class Ts, class TV>
inline std::enable_if_t<isSize<TA>(2, 2) && isSize<TU>(2, 2) && isSize<TV>(2, 2) && isSize<Ts>(2, 1)>
singularValueDecomposition(
    const Eigen::MatrixBase<TA>& A,
    const Eigen::MatrixBase<TU>& U,
    const Eigen::MatrixBase<Ts>& Sigma,
    const Eigen::MatrixBase<TV>& V,
    const ScalarType<TA> tol = 64 * std::numeric_limits<ScalarType<TA> >::epsilon())
{
    using T = ScalarType<TA>;
    GivensRotation<T> gv(0, 1);
    GivensRotation<T> gu(0, 1);
    singularValueDecomposition(A, gu, Sigma, gv);

    gu.fill(U);
    gv.fill(V);
}

/**
  \brief compute wilkinsonShift of the block
  a1     b1
  b1     a2
  based on the wilkinsonShift formula
  mu = c + d - sign (d) \ sqrt (d*d + b*b), where d = (a-c)/2

  */
template <class T>
T wilkinsonShift(const T a1, const T b1, const T a2)
{
    using std::sqrt;
    using std::fabs;
    using std::copysign;

    T d = (T)0.5 * (a1 - a2);
    T bs = b1 * b1;

    T mu = a2 - copysign(bs / (fabs(d) + sqrt(d * d + bs)), d);
    // T mu = a2 - bs / ( d + sign_d*sqrt (d*d + bs));
    return mu;
}

/**
  \brief Helper function of 3X3 SVD for processing 2X2 SVD
  */
template <int t, class T>
inline void process(Eigen::Matrix<T, 3, 3>& B, Eigen::Matrix<T, 3, 3>& U, Eigen::Matrix<T, 3, 1>& sigma, Eigen::Matrix<T, 3, 3>& V)
{
    int other = (t == 1) ? 0 : 2;
    GivensRotation<T> u(0, 1);
    GivensRotation<T> v(0, 1);
    sigma(other) = B(other, other);
    singularValueDecomposition(B.template block<2, 2>(t, t), u, sigma.template block<2, 1>(t, 0), v);
    u.rowi += t;
    u.rowk += t;
    v.rowi += t;
    v.rowk += t;
    u.columnRotation(U);
    v.columnRotation(V);
}

/**
  \brief Helper function of 3X3 SVD for flipping signs due to flipping signs of sigma
  */
template <class T>
inline void flipSign(int i, Eigen::Matrix<T, 3, 3>& U, Eigen::Matrix<T, 3, 1>& sigma)
{
    sigma(i) = -sigma(i);
    U.col(i) = -U.col(i);
}

/**
  \brief Helper function of 3X3 SVD for sorting singular values
  */
template <int t, class T>
std::enable_if_t<t == 0> sort(Eigen::Matrix<T, 3, 3>& U, Eigen::Matrix<T, 3, 1>& sigma, Eigen::Matrix<T, 3, 3>& V)
{
    using std::fabs;

    // Case: sigma(0) > |sigma(1)| >= |sigma(2)|
    if (fabs(sigma(1)) >= fabs(sigma(2))) {
        if (sigma(1) < 0) {
            flipSign(1, U, sigma);
            flipSign(2, U, sigma);
        }
        return;
    }

    //fix sign of sigma for both cases
    if (sigma(2) < 0) {
        flipSign(1, U, sigma);
        flipSign(2, U, sigma);
    }

    //swap sigma(1) and sigma(2) for both cases
    std::swap(sigma(1), sigma(2));
    U.col(1).swap(U.col(2));
    V.col(1).swap(V.col(2));

    // Case: |sigma(2)| >= sigma(0) > |simga(1)|
    if (sigma(1) > sigma(0)) {
        std::swap(sigma(0), sigma(1));
        U.col(0).swap(U.col(1));
        V.col(0).swap(V.col(1));
    }

    // Case: sigma(0) >= |sigma(2)| > |simga(1)|
    else {
        U.col(2) = -U.col(2);
        V.col(2) = -V.col(2);
    }
}

/**
  \brief Helper function of 3X3 SVD for sorting singular values
  */
template <int t, class T>
std::enable_if_t<t == 1> sort(Eigen::Matrix<T, 3, 3>& U, Eigen::Matrix<T, 3, 1>& sigma, Eigen::Matrix<T, 3, 3>& V)
{
    using std::fabs;

    // Case: |sigma(0)| >= sigma(1) > |sigma(2)|
    if (fabs(sigma(0)) >= sigma(1)) {
        if (sigma(0) < 0) {
            flipSign(0, U, sigma);
            flipSign(2, U, sigma);
        }
        return;
    }

    //swap sigma(0) and sigma(1) for both cases
    std::swap(sigma(0), sigma(1));
    U.col(0).swap(U.col(1));
    V.col(0).swap(V.col(1));

    // Case: sigma(1) > |sigma(2)| >= |sigma(0)|
    if (fabs(sigma(1)) < fabs(sigma(2))) {
        std::swap(sigma(1), sigma(2));
        U.col(1).swap(U.col(2));
        V.col(1).swap(V.col(2));
    }

    // Case: sigma(1) >= |sigma(0)| > |sigma(2)|
    else {
        U.col(1) = -U.col(1);
        V.col(1) = -V.col(1);
    }

    // fix sign for both cases
    if (sigma(1) < 0) {
        flipSign(1, U, sigma);
        flipSign(2, U, sigma);
    }
}

/**
  \brief 3X3 SVD (singular value decomposition) A=USV'
  \param[in] A Input matrix.
  \param[out] U is a rotation matrix.
  \param[out] sigma Diagonal matrix, sorted with decreasing magnitude. The third one can be negative.
  \param[out] V is a rotation matrix.
  */
template <class T>
inline int singularValueDecomposition(const Eigen::Matrix<T, 3, 3>& A,
    Eigen::Matrix<T, 3, 3>& U,
    Eigen::Matrix<T, 3, 1>& sigma,
    Eigen::Matrix<T, 3, 3>& V,
    T tol = 128 * std::numeric_limits<T>::epsilon())
{
    using std::fabs;
    using std::sqrt;
    using std::max;
    Eigen::Matrix<T, 3, 3> B = A;
    U = Eigen::Matrix<T, 3, 3>::Identity();
    V = Eigen::Matrix<T, 3, 3>::Identity();

    makeUpperBidiag(B, U, V);

    int count = 0;
    T mu = (T)0;
    GivensRotation<T> r(0, 1);

    T alpha_1 = B(0, 0);
    T beta_1 = B(0, 1);
    T alpha_2 = B(1, 1);
    T alpha_3 = B(2, 2);
    T beta_2 = B(1, 2);
    T gamma_1 = alpha_1 * beta_1;
    T gamma_2 = alpha_2 * beta_2;
    tol *= max((T)0.5 * sqrt(alpha_1 * alpha_1 + alpha_2 * alpha_2 + alpha_3 * alpha_3 + beta_1 * beta_1 + beta_2 * beta_2), (T)1);

    /**
      Do implicit shift QR until A^T A is block diagonal
      */

    while (fabs(beta_2) > tol && fabs(beta_1) > tol
        && fabs(alpha_1) > tol && fabs(alpha_2) > tol
        && fabs(alpha_3) > tol) {
        mu = wilkinsonShift(alpha_2 * alpha_2 + beta_1 * beta_1, gamma_2, alpha_3 * alpha_3 + beta_2 * beta_2);

        r.compute(alpha_1 * alpha_1 - mu, gamma_1);
        r.columnRotation(B);

        r.columnRotation(V);
        zeroChase(B, U, V);

        alpha_1 = B(0, 0);
        beta_1 = B(0, 1);
        alpha_2 = B(1, 1);
        alpha_3 = B(2, 2);
        beta_2 = B(1, 2);
        gamma_1 = alpha_1 * beta_1;
        gamma_2 = alpha_2 * beta_2;
        count++;
    }
    /**
      Handle the cases of one of the alphas and betas being 0
      Sorted by ease of handling and then frequency
      of occurrence

      If B is of form
      x x 0
      0 x 0
      0 0 x
      */
    if (fabs(beta_2) <= tol) {
        process<0>(B, U, sigma, V);
        sort<0>(U, sigma, V);
    }
    /**
      If B is of form
      x 0 0
      0 x x
      0 0 x
      */
    else if (fabs(beta_1) <= tol) {
        process<1>(B, U, sigma, V);
        sort<1>(U, sigma, V);
    }
    /**
      If B is of form
      x x 0
      0 0 x
      0 0 x
      */
    else if (fabs(alpha_2) <= tol) {
        /**
        Reduce B to
        x x 0
        0 0 0
        0 0 x
        */
        GivensRotation<T> r1(1, 2);
        r1.computeUnconventional(B(1, 2), B(2, 2));
        r1.rowRotation(B);
        r1.columnRotation(U);

        process<0>(B, U, sigma, V);
        sort<0>(U, sigma, V);
    }
    /**
      If B is of form
      x x 0
      0 x x
      0 0 0
      */
    else if (fabs(alpha_3) <= tol) {
        /**
        Reduce B to
        x x +
        0 x 0
        0 0 0
        */
        GivensRotation<T> r1(1, 2);
        r1.compute(B(1, 1), B(1, 2));
        r1.columnRotation(B);
        r1.columnRotation(V);
        /**
        Reduce B to
        x x 0
        + x 0
        0 0 0
        */
        GivensRotation<T> r2(0, 2);
        r2.compute(B(0, 0), B(0, 2));
        r2.columnRotation(B);
        r2.columnRotation(V);

        process<0>(B, U, sigma, V);
        sort<0>(U, sigma, V);
    }
    /**
      If B is of form
      0 x 0
      0 x x
      0 0 x
      */
    else if (fabs(alpha_1) <= tol) {
        /**
        Reduce B to
        0 0 +
        0 x x
        0 0 x
        */
        GivensRotation<T> r1(0, 1);
        r1.computeUnconventional(B(0, 1), B(1, 1));
        r1.rowRotation(B);
        r1.columnRotation(U);

        /**
        Reduce B to
        0 0 0
        0 x x
        0 + x
        */
        GivensRotation<T> r2(0, 2);
        r2.computeUnconventional(B(0, 2), B(2, 2));
        r2.rowRotation(B);
        r2.columnRotation(U);

        process<1>(B, U, sigma, V);
        sort<1>(U, sigma, V);
    }

    return count;
}

/**
       \brief 3X3 polar decomposition.
       \param[in] A matrix.
       \param[out] R Robustly a rotation matrix.
       \param[out] S_Sym Symmetric. Whole matrix is stored

       Whole matrix S is stored
       Polar guarantees negative sign is on the small magnitude singular value.
       S is guaranteed to be the closest one to identity.
       R is guaranteed to be the closest rotation to A.
    */
template <class T>
inline void polarDecomposition(const Eigen::Matrix<T, 3, 3>& A,
    Eigen::Matrix<T, 3, 3>& R,
    Eigen::Matrix<T, 3, 3>& S_Sym)
{
    Eigen::Matrix<T, 3, 3> U;
    Eigen::Matrix<T, 3, 1> sigma;
    Eigen::Matrix<T, 3, 3> V;

    singularValueDecomposition(A, U, sigma, V);
    R.noalias() = U * V.transpose();
    S_Sym.noalias() = V * Eigen::DiagonalMatrix<T, 3, 3>(sigma) * V.transpose();
}
}
#endif
