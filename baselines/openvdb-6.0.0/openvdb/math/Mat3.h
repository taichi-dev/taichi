///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2018 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#ifndef OPENVDB_MATH_MAT3_H_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_MAT3_H_HAS_BEEN_INCLUDED

#include <openvdb/Exceptions.h>
#include "Vec3.h"
#include "Mat.h"
#include <algorithm> // for std::copy()
#include <cassert>
#include <cmath>
#include <iomanip>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

template<typename T> class Vec3;
template<typename T> class Mat4;
template<typename T> class Quat;

/// @class Mat3 Mat3.h
/// @brief 3x3 matrix class.
template<typename T>
class Mat3: public Mat<3, T>
{
public:
    /// Data type held by the matrix.
    using value_type = T;
    using ValueType = T;
    using MyBase = Mat<3, T>;
    /// Trivial constructor, the matrix is NOT initialized
    Mat3() {}

    /// Constructor given the quaternion rotation, e.g.    Mat3f m(q);
    /// The quaternion is normalized and used to construct the matrix
    Mat3(const Quat<T> &q)
    { setToRotation(q); }


    /// Constructor given array of elements, the ordering is in row major form:
    /** @verbatim
        a b c
        d e f
        g h i
        @endverbatim */
    template<typename Source>
    Mat3(Source a, Source b, Source c,
         Source d, Source e, Source f,
         Source g, Source h, Source i)
    {
        MyBase::mm[0] = static_cast<ValueType>(a);
        MyBase::mm[1] = static_cast<ValueType>(b);
        MyBase::mm[2] = static_cast<ValueType>(c);
        MyBase::mm[3] = static_cast<ValueType>(d);
        MyBase::mm[4] = static_cast<ValueType>(e);
        MyBase::mm[5] = static_cast<ValueType>(f);
        MyBase::mm[6] = static_cast<ValueType>(g);
        MyBase::mm[7] = static_cast<ValueType>(h);
        MyBase::mm[8] = static_cast<ValueType>(i);
    } // constructor1Test

    /// Construct matrix from rows or columns vectors (defaults to rows
    /// for historical reasons)
    template<typename Source>
    Mat3(const Vec3<Source> &v1, const Vec3<Source> &v2, const Vec3<Source> &v3, bool rows = true)
    {
        if (rows) {
            this->setRows(v1, v2, v3);
        } else {
            this->setColumns(v1, v2, v3);
        }
    }

    /// Constructor given array of elements, the ordering is in row major form:\n
    /// a[0] a[1] a[2]\n
    /// a[3] a[4] a[5]\n
    /// a[6] a[7] a[8]\n
    template<typename Source>
    Mat3(Source *a)
    {
        MyBase::mm[0] = a[0];
        MyBase::mm[1] = a[1];
        MyBase::mm[2] = a[2];
        MyBase::mm[3] = a[3];
        MyBase::mm[4] = a[4];
        MyBase::mm[5] = a[5];
        MyBase::mm[6] = a[6];
        MyBase::mm[7] = a[7];
        MyBase::mm[8] = a[8];
    } // constructor1Test

    /// Copy constructor
    Mat3(const Mat<3, T> &m)
    {
        for (int i=0; i<3; ++i) {
            for (int j=0; j<3; ++j) {
                MyBase::mm[i*3 + j] = m[i][j];
            }
        }
    }

    /// Conversion constructor
    template<typename Source>
    explicit Mat3(const Mat3<Source> &m)
    {
        for (int i=0; i<3; ++i) {
            for (int j=0; j<3; ++j) {
                MyBase::mm[i*3 + j] = static_cast<T>(m[i][j]);
            }
        }
    }

    /// Conversion from Mat4 (copies top left)
    explicit Mat3(const Mat4<T> &m)
    {
        for (int i=0; i<3; ++i) {
            for (int j=0; j<3; ++j) {
                MyBase::mm[i*3 + j] = m[i][j];
            }
        }
    }

    /// Predefined constant for identity matrix
    static const Mat3<T>& identity() {
        static const Mat3<T> sIdentity = Mat3<T>(
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        );
        return sIdentity;
    }

    /// Predefined constant for zero matrix
    static const Mat3<T>& zero() {
        static const Mat3<T> sZero = Mat3<T>(
            0, 0, 0,
            0, 0, 0,
            0, 0, 0
        );
        return sZero;
    }

    /// Set ith row to vector v
    void setRow(int i, const Vec3<T> &v)
    {
        // assert(i>=0 && i<3);
        int i3 = i * 3;

        MyBase::mm[i3+0] = v[0];
        MyBase::mm[i3+1] = v[1];
        MyBase::mm[i3+2] = v[2];
    } // rowColumnTest

    /// Get ith row, e.g.    Vec3d v = m.row(1);
    Vec3<T> row(int i) const
    {
        // assert(i>=0 && i<3);
        return Vec3<T>((*this)(i,0), (*this)(i,1), (*this)(i,2));
    } // rowColumnTest

    /// Set jth column to vector v
    void setCol(int j, const Vec3<T>& v)
    {
        // assert(j>=0 && j<3);
        MyBase::mm[0+j] = v[0];
        MyBase::mm[3+j] = v[1];
        MyBase::mm[6+j] = v[2];
    } // rowColumnTest

    /// Get jth column, e.g.    Vec3d v = m.col(0);
    Vec3<T> col(int j) const
    {
        // assert(j>=0 && j<3);
        return Vec3<T>((*this)(0,j), (*this)(1,j), (*this)(2,j));
    } // rowColumnTest

    // NB: The following two methods were changed to
    // work around a gccWS5 compiler issue related to strict
    // aliasing (see FX-475).

    //@{
    /// Array style reference to ith row
    /// e.g.    m[1][2] = 4;
    T* operator[](int i) { return &(MyBase::mm[i*3]); }
    const T* operator[](int i) const { return &(MyBase::mm[i*3]); }
    //@}

    T* asPointer() {return MyBase::mm;}
    const T* asPointer() const {return MyBase::mm;}

    /// Alternative indexed reference to the elements
    /// Note that the indices are row first and column second.
    /// e.g.    m(0,0) = 1;
    T& operator()(int i, int j)
    {
        // assert(i>=0 && i<3);
        // assert(j>=0 && j<3);
        return MyBase::mm[3*i+j];
    } // trivial

    /// Alternative indexed constant reference to the elements,
    /// Note that the indices are row first and column second.
    /// e.g.    float f = m(1,0);
    T operator()(int i, int j) const
    {
        // assert(i>=0 && i<3);
        // assert(j>=0 && j<3);
        return MyBase::mm[3*i+j];
    } // trivial

    /// Set the rows of this matrix to the vectors v1, v2, v3
    void setRows(const Vec3<T> &v1, const Vec3<T> &v2, const Vec3<T> &v3)
    {
        MyBase::mm[0] = v1[0];
        MyBase::mm[1] = v1[1];
        MyBase::mm[2] = v1[2];
        MyBase::mm[3] = v2[0];
        MyBase::mm[4] = v2[1];
        MyBase::mm[5] = v2[2];
        MyBase::mm[6] = v3[0];
        MyBase::mm[7] = v3[1];
        MyBase::mm[8] = v3[2];
    } // setRows

    /// Set the columns of this matrix to the vectors v1, v2, v3
    void setColumns(const Vec3<T> &v1, const Vec3<T> &v2, const Vec3<T> &v3)
    {
        MyBase::mm[0] = v1[0];
        MyBase::mm[1] = v2[0];
        MyBase::mm[2] = v3[0];
        MyBase::mm[3] = v1[1];
        MyBase::mm[4] = v2[1];
        MyBase::mm[5] = v3[1];
        MyBase::mm[6] = v1[2];
        MyBase::mm[7] = v2[2];
        MyBase::mm[8] = v3[2];
    } // setColumns

    /// Set diagonal and symmetric triangular components
    void setSymmetric(const Vec3<T> &vdiag, const Vec3<T> &vtri)
    {
        MyBase::mm[0] = vdiag[0];
        MyBase::mm[1] = vtri[0];
        MyBase::mm[2] = vtri[1];
        MyBase::mm[3] = vtri[0];
        MyBase::mm[4] = vdiag[1];
        MyBase::mm[5] = vtri[2];
        MyBase::mm[6] = vtri[1];
        MyBase::mm[7] = vtri[2];
        MyBase::mm[8] = vdiag[2];
    } // setSymmetricTest

    /// Return a matrix with the prescribed diagonal and symmetric triangular components.
    static Mat3 symmetric(const Vec3<T> &vdiag, const Vec3<T> &vtri)
    {
        return Mat3(
                    vdiag[0], vtri[0], vtri[1],
                    vtri[0], vdiag[1], vtri[2],
                    vtri[1], vtri[2], vdiag[2]
                    );
    }

    /// Set the matrix as cross product of the given vector
    void setSkew(const Vec3<T> &v)
    {*this = skew(v);}

    /// @brief Set this matrix to the rotation matrix specified by the quaternion
    /// @details The quaternion is normalized and used to construct the matrix.
    /// Note that the matrix is transposed to match post-multiplication semantics.
    void setToRotation(const Quat<T> &q)
    {*this = rotation<Mat3<T> >(q);}

    /// @brief Set this matrix to the rotation specified by @a axis and @a angle
    /// @details The axis must be unit vector
    void setToRotation(const Vec3<T> &axis, T angle)
    {*this = rotation<Mat3<T> >(axis, angle);}

    /// Set this matrix to zero
    void setZero()
    {
        MyBase::mm[0] = 0;
        MyBase::mm[1] = 0;
        MyBase::mm[2] = 0;
        MyBase::mm[3] = 0;
        MyBase::mm[4] = 0;
        MyBase::mm[5] = 0;
        MyBase::mm[6] = 0;
        MyBase::mm[7] = 0;
        MyBase::mm[8] = 0;
    } // trivial

    /// Set this matrix to identity
    void setIdentity()
    {
        MyBase::mm[0] = 1;
        MyBase::mm[1] = 0;
        MyBase::mm[2] = 0;
        MyBase::mm[3] = 0;
        MyBase::mm[4] = 1;
        MyBase::mm[5] = 0;
        MyBase::mm[6] = 0;
        MyBase::mm[7] = 0;
        MyBase::mm[8] = 1;
    } // trivial

    /// Assignment operator
    template<typename Source>
    const Mat3& operator=(const Mat3<Source> &m)
    {
        const Source *src = m.asPointer();

        // don't suppress type conversion warnings
        std::copy(src, (src + this->numElements()), MyBase::mm);
        return *this;
    } // opEqualToTest

    /// Return @c true if this matrix is equivalent to @a m within a tolerance of @a eps.
    bool eq(const Mat3 &m, T eps=1.0e-8) const
    {
        return (isApproxEqual(MyBase::mm[0],m.mm[0],eps) &&
                isApproxEqual(MyBase::mm[1],m.mm[1],eps) &&
                isApproxEqual(MyBase::mm[2],m.mm[2],eps) &&
                isApproxEqual(MyBase::mm[3],m.mm[3],eps) &&
                isApproxEqual(MyBase::mm[4],m.mm[4],eps) &&
                isApproxEqual(MyBase::mm[5],m.mm[5],eps) &&
                isApproxEqual(MyBase::mm[6],m.mm[6],eps) &&
                isApproxEqual(MyBase::mm[7],m.mm[7],eps) &&
                isApproxEqual(MyBase::mm[8],m.mm[8],eps));
    } // trivial

    /// Negation operator, for e.g.   m1 = -m2;
    Mat3<T> operator-() const
    {
        return Mat3<T>(
                       -MyBase::mm[0], -MyBase::mm[1], -MyBase::mm[2],
                       -MyBase::mm[3], -MyBase::mm[4], -MyBase::mm[5],
                       -MyBase::mm[6], -MyBase::mm[7], -MyBase::mm[8]
                       );
    } // trivial

    /// Multiplication operator, e.g.   M = scalar * M;
    // friend Mat3 operator*(T scalar, const Mat3& m) {
    //     return m*scalar;
    // }

    /// Multiply each element of this matrix by @a scalar.
    template <typename S>
    const Mat3<T>& operator*=(S scalar)
    {
        MyBase::mm[0] *= scalar;
        MyBase::mm[1] *= scalar;
        MyBase::mm[2] *= scalar;
        MyBase::mm[3] *= scalar;
        MyBase::mm[4] *= scalar;
        MyBase::mm[5] *= scalar;
        MyBase::mm[6] *= scalar;
        MyBase::mm[7] *= scalar;
        MyBase::mm[8] *= scalar;
        return *this;
    }

    /// Add each element of the given matrix to the corresponding element of this matrix.
    template <typename S>
    const Mat3<T> &operator+=(const Mat3<S> &m1)
    {
        const S *s = m1.asPointer();

        MyBase::mm[0] += s[0];
        MyBase::mm[1] += s[1];
        MyBase::mm[2] += s[2];
        MyBase::mm[3] += s[3];
        MyBase::mm[4] += s[4];
        MyBase::mm[5] += s[5];
        MyBase::mm[6] += s[6];
        MyBase::mm[7] += s[7];
        MyBase::mm[8] += s[8];
        return *this;
    }

    /// Subtract each element of the given matrix from the corresponding element of this matrix.
    template <typename S>
    const Mat3<T> &operator-=(const Mat3<S> &m1)
    {
        const S *s = m1.asPointer();

        MyBase::mm[0] -= s[0];
        MyBase::mm[1] -= s[1];
        MyBase::mm[2] -= s[2];
        MyBase::mm[3] -= s[3];
        MyBase::mm[4] -= s[4];
        MyBase::mm[5] -= s[5];
        MyBase::mm[6] -= s[6];
        MyBase::mm[7] -= s[7];
        MyBase::mm[8] -= s[8];
        return *this;
    }

    /// Multiply this matrix by the given matrix.
    template <typename S>
    const Mat3<T> &operator*=(const Mat3<S> &m1)
    {
        Mat3<T> m0(*this);
        const T* s0 = m0.asPointer();
        const S* s1 = m1.asPointer();

        MyBase::mm[0] = static_cast<T>(s0[0] * s1[0] +
                                       s0[1] * s1[3] +
                                       s0[2] * s1[6]);
        MyBase::mm[1] = static_cast<T>(s0[0] * s1[1] +
                                       s0[1] * s1[4] +
                                       s0[2] * s1[7]);
        MyBase::mm[2] = static_cast<T>(s0[0] * s1[2] +
                                       s0[1] * s1[5] +
                                       s0[2] * s1[8]);

        MyBase::mm[3] = static_cast<T>(s0[3] * s1[0] +
                                       s0[4] * s1[3] +
                                       s0[5] * s1[6]);
        MyBase::mm[4] = static_cast<T>(s0[3] * s1[1] +
                                       s0[4] * s1[4] +
                                       s0[5] * s1[7]);
        MyBase::mm[5] = static_cast<T>(s0[3] * s1[2] +
                                       s0[4] * s1[5] +
                                       s0[5] * s1[8]);

        MyBase::mm[6] = static_cast<T>(s0[6] * s1[0] +
                                       s0[7] * s1[3] +
                                       s0[8] * s1[6]);
        MyBase::mm[7] = static_cast<T>(s0[6] * s1[1] +
                                       s0[7] * s1[4] +
                                       s0[8] * s1[7]);
        MyBase::mm[8] = static_cast<T>(s0[6] * s1[2] +
                                       s0[7] * s1[5] +
                                       s0[8] * s1[8]);

        return *this;
    }

    /// @brief Return the cofactor matrix of this matrix.
    Mat3 cofactor() const
    {
        return Mat3<T>(
          MyBase::mm[4] * MyBase::mm[8] - MyBase::mm[5] * MyBase::mm[7],
          MyBase::mm[5] * MyBase::mm[6] - MyBase::mm[3] * MyBase::mm[8],
          MyBase::mm[3] * MyBase::mm[7] - MyBase::mm[4] * MyBase::mm[6],
          MyBase::mm[2] * MyBase::mm[7] - MyBase::mm[1] * MyBase::mm[8],
          MyBase::mm[0] * MyBase::mm[8] - MyBase::mm[2] * MyBase::mm[6],
          MyBase::mm[1] * MyBase::mm[6] - MyBase::mm[0] * MyBase::mm[7],
          MyBase::mm[1] * MyBase::mm[5] - MyBase::mm[2] * MyBase::mm[4],
          MyBase::mm[2] * MyBase::mm[3] - MyBase::mm[0] * MyBase::mm[5],
          MyBase::mm[0] * MyBase::mm[4] - MyBase::mm[1] * MyBase::mm[3]);
    }

    /// Return the adjoint of this matrix, i.e., the transpose of its cofactor.
    Mat3 adjoint() const
    {
        return Mat3<T>(
          MyBase::mm[4] * MyBase::mm[8] - MyBase::mm[5] * MyBase::mm[7],
          MyBase::mm[2] * MyBase::mm[7] - MyBase::mm[1] * MyBase::mm[8],
          MyBase::mm[1] * MyBase::mm[5] - MyBase::mm[2] * MyBase::mm[4],
          MyBase::mm[5] * MyBase::mm[6] - MyBase::mm[3] * MyBase::mm[8],
          MyBase::mm[0] * MyBase::mm[8] - MyBase::mm[2] * MyBase::mm[6],
          MyBase::mm[2] * MyBase::mm[3] - MyBase::mm[0] * MyBase::mm[5],
          MyBase::mm[3] * MyBase::mm[7] - MyBase::mm[4] * MyBase::mm[6],
          MyBase::mm[1] * MyBase::mm[6] - MyBase::mm[0] * MyBase::mm[7],
          MyBase::mm[0] * MyBase::mm[4] - MyBase::mm[1] * MyBase::mm[3]);

    } // adjointTest

    /// returns transpose of this
    Mat3 transpose() const
    {
        return Mat3<T>(
          MyBase::mm[0], MyBase::mm[3], MyBase::mm[6],
          MyBase::mm[1], MyBase::mm[4], MyBase::mm[7],
          MyBase::mm[2], MyBase::mm[5], MyBase::mm[8]);

    } // transposeTest

    /// returns inverse of this
    /// @throws ArithmeticError if singular
    Mat3 inverse(T tolerance = 0) const
    {
        Mat3<T> inv(this->adjoint());

        const T det = inv.mm[0]*MyBase::mm[0] + inv.mm[1]*MyBase::mm[3] + inv.mm[2]*MyBase::mm[6];

        // If the determinant is 0, m was singular and the result will be invalid.
        if (isApproxEqual(det,T(0.0),tolerance)) {
            OPENVDB_THROW(ArithmeticError, "Inversion of singular 3x3 matrix");
        }
        return inv * (T(1)/det);
    } // invertTest

    /// Determinant of matrix
    T det() const
    {
        const T co00 = MyBase::mm[4]*MyBase::mm[8] - MyBase::mm[5]*MyBase::mm[7];
        const T co10 = MyBase::mm[5]*MyBase::mm[6] - MyBase::mm[3]*MyBase::mm[8];
        const T co20 = MyBase::mm[3]*MyBase::mm[7] - MyBase::mm[4]*MyBase::mm[6];
        return MyBase::mm[0]*co00  + MyBase::mm[1]*co10 + MyBase::mm[2]*co20;
    } // determinantTest

    /// Trace of matrix
    T trace() const
    {
        return MyBase::mm[0]+MyBase::mm[4]+MyBase::mm[8];
    }

    /// This function snaps a specific axis to a specific direction,
    /// preserving scaling. It does this using minimum energy, thus
    /// posing a unique solution if basis & direction arent parralel.
    /// Direction need not be unit.
    Mat3 snapBasis(Axis axis, const Vec3<T> &direction)
    {
        return snapMatBasis(*this, axis, direction);
    }

    /// Return the transformed vector by this matrix.
    /// This function is equivalent to post-multiplying the matrix.
    template<typename T0>
    Vec3<T0> transform(const Vec3<T0> &v) const
    {
        return static_cast< Vec3<T0> >(v * *this);
    } // xformVectorTest

    /// Return the transformed vector by transpose of this matrix.
    /// This function is equivalent to pre-multiplying the matrix.
    template<typename T0>
    Vec3<T0> pretransform(const Vec3<T0> &v) const
    {
        return static_cast< Vec3<T0> >(*this * v);
    } // xformTVectorTest


    /// @brief Treat @a diag as a diagonal matrix and return the product
    /// of this matrix with @a diag (from the right).
    Mat3 timesDiagonal(const Vec3<T>& diag) const
    {
        Mat3 ret(*this);

        ret.mm[0] *= diag(0);
        ret.mm[1] *= diag(1);
        ret.mm[2] *= diag(2);
        ret.mm[3] *= diag(0);
        ret.mm[4] *= diag(1);
        ret.mm[5] *= diag(2);
        ret.mm[6] *= diag(0);
        ret.mm[7] *= diag(1);
        ret.mm[8] *= diag(2);
        return ret;
    }
}; // class Mat3


/// @relates Mat3
/// @brief Equality operator, does exact floating point comparisons
template <typename T0, typename T1>
bool operator==(const Mat3<T0> &m0, const Mat3<T1> &m1)
{
    const T0 *t0 = m0.asPointer();
    const T1 *t1 = m1.asPointer();

    for (int i=0; i<9; ++i) {
        if (!isExactlyEqual(t0[i], t1[i])) return false;
    }
    return true;
}

/// @relates Mat3
/// @brief Inequality operator, does exact floating point comparisons
template <typename T0, typename T1>
bool operator!=(const Mat3<T0> &m0, const Mat3<T1> &m1) { return !(m0 == m1); }

/// @relates Mat3
/// @brief Multiply each element of the given matrix by @a scalar and return the result.
template <typename S, typename T>
Mat3<typename promote<S, T>::type> operator*(S scalar, const Mat3<T> &m)
{ return m*scalar; }

/// @relates Mat3
/// @brief Multiply each element of the given matrix by @a scalar and return the result.
template <typename S, typename T>
Mat3<typename promote<S, T>::type> operator*(const Mat3<T> &m, S scalar)
{
    Mat3<typename promote<S, T>::type> result(m);
    result *= scalar;
    return result;
}

/// @relates Mat3
/// @brief Add corresponding elements of @a m0 and @a m1 and return the result.
template <typename T0, typename T1>
Mat3<typename promote<T0, T1>::type> operator+(const Mat3<T0> &m0, const Mat3<T1> &m1)
{
    Mat3<typename promote<T0, T1>::type> result(m0);
    result += m1;
    return result;
}

/// @relates Mat3
/// @brief Subtract corresponding elements of @a m0 and @a m1 and return the result.
template <typename T0, typename T1>
Mat3<typename promote<T0, T1>::type> operator-(const Mat3<T0> &m0, const Mat3<T1> &m1)
{
    Mat3<typename promote<T0, T1>::type> result(m0);
    result -= m1;
    return result;
}


/// @brief Multiply @a m0 by @a m1 and return the resulting matrix.
template <typename T0, typename T1>
Mat3<typename promote<T0, T1>::type>operator*(const Mat3<T0> &m0, const Mat3<T1> &m1)
{
    Mat3<typename promote<T0, T1>::type> result(m0);
    result *= m1;
    return result;
}

/// @relates Mat3
/// @brief Multiply @a _m by @a _v and return the resulting vector.
template<typename T, typename MT>
inline Vec3<typename promote<T, MT>::type>
operator*(const Mat3<MT> &_m, const Vec3<T> &_v)
{
    MT const *m = _m.asPointer();
    return Vec3<typename promote<T, MT>::type>(
        _v[0]*m[0] + _v[1]*m[1] + _v[2]*m[2],
        _v[0]*m[3] + _v[1]*m[4] + _v[2]*m[5],
        _v[0]*m[6] + _v[1]*m[7] + _v[2]*m[8]);
}

/// @relates Mat3
/// @brief Multiply @a _v by @a _m and return the resulting vector.
template<typename T, typename MT>
inline Vec3<typename promote<T, MT>::type>
operator*(const Vec3<T> &_v, const Mat3<MT> &_m)
{
    MT const *m = _m.asPointer();
    return Vec3<typename promote<T, MT>::type>(
        _v[0]*m[0] + _v[1]*m[3] + _v[2]*m[6],
        _v[0]*m[1] + _v[1]*m[4] + _v[2]*m[7],
        _v[0]*m[2] + _v[1]*m[5] + _v[2]*m[8]);
}

/// @relates Mat3
/// @brief Multiply @a _v by @a _m and replace @a _v with the resulting vector.
template<typename T, typename MT>
inline Vec3<T> &operator *= (Vec3<T> &_v, const Mat3<MT> &_m)
{
    Vec3<T> mult = _v * _m;
    _v = mult;
    return _v;
}

/// Returns outer product of v1, v2, i.e. v1 v2^T if v1 and v2 are
/// column vectors, e.g.   M = Mat3f::outerproduct(v1,v2);
template <typename T>
Mat3<T> outerProduct(const Vec3<T>& v1, const Vec3<T>& v2)
{
    return Mat3<T>(v1[0]*v2[0], v1[0]*v2[1], v1[0]*v2[2],
                   v1[1]*v2[0], v1[1]*v2[1], v1[1]*v2[2],
                   v1[2]*v2[0], v1[2]*v2[1], v1[2]*v2[2]);
}// outerProduct


/// Interpolate the rotation between m1 and m2 using Mat::powSolve.
/// Unlike slerp, translation is not treated independently.
/// This results in smoother animation results.
template<typename T, typename T0>
Mat3<T> powLerp(const Mat3<T0> &m1, const Mat3<T0> &m2, T t)
{
    Mat3<T> x = m1.inverse() * m2;
    powSolve(x, x, t);
    Mat3<T> m = m1 * x;
    return m;
}


namespace mat3_internal {

template<typename T>
inline void
pivot(int i, int j, Mat3<T>& S, Vec3<T>& D, Mat3<T>& Q)
{
    const int& n = Mat3<T>::size;  // should be 3
    T temp;
    /// scratch variables used in pivoting
    double cotan_of_2_theta;
    double tan_of_theta;
    double cosin_of_theta;
    double sin_of_theta;
    double z;

    double Sij = S(i,j);

    double Sjj_minus_Sii = D[j] - D[i];

    if (fabs(Sjj_minus_Sii) * (10*math::Tolerance<T>::value()) > fabs(Sij)) {
        tan_of_theta = Sij / Sjj_minus_Sii;
    } else {
        /// pivot on Sij
        cotan_of_2_theta = 0.5*Sjj_minus_Sii / Sij ;

        if (cotan_of_2_theta < 0.) {
            tan_of_theta =
                -1./(sqrt(1. + cotan_of_2_theta*cotan_of_2_theta) - cotan_of_2_theta);
        } else {
            tan_of_theta =
                1./(sqrt(1. + cotan_of_2_theta*cotan_of_2_theta) + cotan_of_2_theta);
        }
    }

    cosin_of_theta = 1./sqrt( 1. + tan_of_theta * tan_of_theta);
    sin_of_theta = cosin_of_theta * tan_of_theta;
    z = tan_of_theta * Sij;
    S(i,j) = 0;
    D[i] -= z;
    D[j] += z;
    for (int k = 0; k < i; ++k) {
        temp = S(k,i);
        S(k,i) = cosin_of_theta * temp - sin_of_theta * S(k,j);
        S(k,j)= sin_of_theta * temp + cosin_of_theta * S(k,j);
    }
    for (int k = i+1; k < j; ++k) {
        temp = S(i,k);
        S(i,k) = cosin_of_theta * temp - sin_of_theta * S(k,j);
        S(k,j) = sin_of_theta * temp + cosin_of_theta * S(k,j);
    }
    for (int k = j+1; k < n; ++k) {
        temp = S(i,k);
        S(i,k) = cosin_of_theta * temp - sin_of_theta * S(j,k);
        S(j,k) = sin_of_theta * temp + cosin_of_theta * S(j,k);
    }
    for (int k = 0; k < n; ++k)
        {
            temp = Q(k,i);
            Q(k,i) = cosin_of_theta * temp - sin_of_theta*Q(k,j);
            Q(k,j) = sin_of_theta * temp + cosin_of_theta*Q(k,j);
        }
}

} // namespace mat3_internal


/// @brief Use Jacobi iterations to decompose a symmetric 3x3 matrix
/// (diagonalize and compute eigenvectors)
/// @details This is based on the "Efficient numerical diagonalization of Hermitian 3x3 matrices"
/// Joachim Kopp.  arXiv.org preprint: physics/0610206
/// with the addition of largest pivot
template<typename T>
inline bool
diagonalizeSymmetricMatrix(const Mat3<T>& input, Mat3<T>& Q, Vec3<T>& D,
    unsigned int MAX_ITERATIONS=250)
{
    /// use Givens rotation matrix to eliminate off-diagonal entries.
    /// initialize the rotation matrix as idenity
    Q  = Mat3<T>::identity();
    int n = Mat3<T>::size;  // should be 3

    /// temp matrix.  Assumed to be symmetric
    Mat3<T> S(input);

    for (int i = 0; i < n; ++i) {
        D[i] = S(i,i);
    }

    unsigned int iterations(0);
    /// Just iterate over all the non-diagonal enteries
    /// using the largest as a pivot.
    do {
        /// check for absolute convergence
        /// are symmetric off diagonals all zero
        double er = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j) {
                er += fabs(S(i,j));
            }
        }
        if (std::abs(er) < math::Tolerance<T>::value()) {
            return true;
        }
        iterations++;

        T max_element = 0;
        int ip = 0;
        int jp = 0;
        /// loop over all the off-diagonals above the diagonal
        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j){

                if ( fabs(D[i]) * (10*math::Tolerance<T>::value()) > fabs(S(i,j))) {
                    /// value too small to pivot on
                    S(i,j) = 0;
                }
                if (fabs(S(i,j)) > max_element) {
                    max_element = fabs(S(i,j));
                    ip = i;
                    jp = j;
                }
            }
        }
        mat3_internal::pivot(ip, jp, S, D, Q);
    } while (iterations < MAX_ITERATIONS);

    return false;
}


using Mat3s = Mat3<float>;
using Mat3d = Mat3<double>;
using Mat3f = Mat3d;

} // namespace math


template<> inline math::Mat3s zeroVal<math::Mat3s>() { return math::Mat3s::zero(); }
template<> inline math::Mat3d zeroVal<math::Mat3d>() { return math::Mat3d::zero(); }

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_MAT3_H_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
