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

#ifndef OPENVDB_MATH_MAT4_H_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_MAT4_H_HAS_BEEN_INCLUDED

#include <openvdb/Exceptions.h>
#include <openvdb/Platform.h>
#include "Math.h"
#include "Mat3.h"
#include "Vec3.h"
#include "Vec4.h"
#include <algorithm> // for std::copy(), std::swap()
#include <cassert>
#include <iomanip>
#include <cmath>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

template<typename T> class Vec4;


/// @class Mat4 Mat4.h
/// @brief 4x4 -matrix class.
template<typename T>
class Mat4: public Mat<4, T>
{
public:
    /// Data type held by the matrix.
    using value_type = T;
    using ValueType = T;
    using MyBase = Mat<4, T>;

    /// Trivial constructor, the matrix is NOT initialized
    Mat4() {}

    /// Constructor given array of elements, the ordering is in row major form:
    /** @verbatim
        a[ 0] a[1]  a[ 2] a[ 3]
        a[ 4] a[5]  a[ 6] a[ 7]
        a[ 8] a[9]  a[10] a[11]
        a[12] a[13] a[14] a[15]
        @endverbatim */
    template<typename Source>
    Mat4(Source *a)
    {
        for (int i = 0; i < 16; i++) {
            MyBase::mm[i] = a[i];
        }
    }

    /// Constructor given array of elements, the ordering is in row major form:
    /** @verbatim
        a b c d
        e f g h
        i j k l
        m n o p
        @endverbatim */
    template<typename Source>
    Mat4(Source a, Source b, Source c, Source d,
         Source e, Source f, Source g, Source h,
         Source i, Source j, Source k, Source l,
         Source m, Source n, Source o, Source p)
    {
        MyBase::mm[ 0] = T(a);
        MyBase::mm[ 1] = T(b);
        MyBase::mm[ 2] = T(c);
        MyBase::mm[ 3] = T(d);

        MyBase::mm[ 4] = T(e);
        MyBase::mm[ 5] = T(f);
        MyBase::mm[ 6] = T(g);
        MyBase::mm[ 7] = T(h);

        MyBase::mm[ 8] = T(i);
        MyBase::mm[ 9] = T(j);
        MyBase::mm[10] = T(k);
        MyBase::mm[11] = T(l);

        MyBase::mm[12] = T(m);
        MyBase::mm[13] = T(n);
        MyBase::mm[14] = T(o);
        MyBase::mm[15] = T(p);
    }

    /// Construct matrix from rows or columns vectors (defaults to rows
    /// for historical reasons)
    template<typename Source>
    Mat4(const Vec4<Source> &v1, const Vec4<Source> &v2,
         const Vec4<Source> &v3, const Vec4<Source> &v4, bool rows = true)
    {
        if (rows) {
            this->setRows(v1, v2, v3, v4);
        } else {
            this->setColumns(v1, v2, v3, v4);
        }
    }

    /// Copy constructor
    Mat4(const Mat<4, T> &m)
    {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                MyBase::mm[i*4 + j] = m[i][j];
            }
        }
    }

    /// Conversion constructor
    template<typename Source>
    explicit Mat4(const Mat4<Source> &m)
    {
        const Source *src = m.asPointer();

        for (int i=0; i<16; ++i) {
            MyBase::mm[i] = static_cast<T>(src[i]);
        }
    }

    /// Predefined constant for identity matrix
    static const Mat4<T>& identity() {
        static const Mat4<T> sIdentity = Mat4<T>(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        );
        return sIdentity;
    }

    /// Predefined constant for zero matrix
    static const Mat4<T>& zero() {
        static const Mat4<T> sZero = Mat4<T>(
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        );
        return sZero;
    }

    /// Set ith row to vector v
    void setRow(int i, const Vec4<T> &v)
    {
        // assert(i>=0 && i<4);
        int i4 = i * 4;
        MyBase::mm[i4+0] = v[0];
        MyBase::mm[i4+1] = v[1];
        MyBase::mm[i4+2] = v[2];
        MyBase::mm[i4+3] = v[3];
    }

    /// Get ith row, e.g.    Vec4f v = m.row(1);
    Vec4<T> row(int i) const
    {
        // assert(i>=0 && i<3);
        return Vec4<T>((*this)(i,0), (*this)(i,1), (*this)(i,2), (*this)(i,3));
    }

    /// Set jth column to vector v
    void setCol(int j, const Vec4<T>& v)
    {
        // assert(j>=0 && j<4);
        MyBase::mm[ 0+j] = v[0];
        MyBase::mm[ 4+j] = v[1];
        MyBase::mm[ 8+j] = v[2];
        MyBase::mm[12+j] = v[3];
    }

    /// Get jth column, e.g.    Vec4f v = m.col(0);
    Vec4<T> col(int j) const
    {
        // assert(j>=0 && j<4);
        return Vec4<T>((*this)(0,j), (*this)(1,j), (*this)(2,j), (*this)(3,j));
    }

    //@{
    /// Array style reference to ith row
    /// e.g.    m[1][3] = 4;
    T* operator[](int i) { return &(MyBase::mm[i<<2]); }
    const T* operator[](int i) const { return &(MyBase::mm[i<<2]); }
    //@}

    /// Direct access to the internal data
    T* asPointer() {return MyBase::mm;}
    const T* asPointer() const {return MyBase::mm;}

    /// Alternative indexed reference to the elements
    /// Note that the indices are row first and column second.
    /// e.g.    m(0,0) = 1;
    T& operator()(int i, int j)
    {
        // assert(i>=0 && i<4);
        // assert(j>=0 && j<4);
        return MyBase::mm[4*i+j];
    }

    /// Alternative indexed constant reference to the elements,
    /// Note that the indices are row first and column second.
    /// e.g.    float f = m(1,0);
    T operator()(int i, int j) const
    {
        // assert(i>=0 && i<4);
        // assert(j>=0 && j<4);
        return MyBase::mm[4*i+j];
    }

    /// Set the rows of this matrix to the vectors v1, v2, v3, v4
    void setRows(const Vec4<T> &v1, const Vec4<T> &v2,
                 const Vec4<T> &v3, const Vec4<T> &v4)
    {
        MyBase::mm[ 0] = v1[0];
        MyBase::mm[ 1] = v1[1];
        MyBase::mm[ 2] = v1[2];
        MyBase::mm[ 3] = v1[3];

        MyBase::mm[ 4] = v2[0];
        MyBase::mm[ 5] = v2[1];
        MyBase::mm[ 6] = v2[2];
        MyBase::mm[ 7] = v2[3];

        MyBase::mm[ 8] = v3[0];
        MyBase::mm[ 9] = v3[1];
        MyBase::mm[10] = v3[2];
        MyBase::mm[11] = v3[3];

        MyBase::mm[12] = v4[0];
        MyBase::mm[13] = v4[1];
        MyBase::mm[14] = v4[2];
        MyBase::mm[15] = v4[3];
    }

    /// Set the columns of this matrix to the vectors v1, v2, v3, v4
    void setColumns(const Vec4<T> &v1, const Vec4<T> &v2,
                    const Vec4<T> &v3, const Vec4<T> &v4)
    {
        MyBase::mm[ 0] = v1[0];
        MyBase::mm[ 1] = v2[0];
        MyBase::mm[ 2] = v3[0];
        MyBase::mm[ 3] = v4[0];

        MyBase::mm[ 4] = v1[1];
        MyBase::mm[ 5] = v2[1];
        MyBase::mm[ 6] = v3[1];
        MyBase::mm[ 7] = v4[1];

        MyBase::mm[ 8] = v1[2];
        MyBase::mm[ 9] = v2[2];
        MyBase::mm[10] = v3[2];
        MyBase::mm[11] = v4[2];

        MyBase::mm[12] = v1[3];
        MyBase::mm[13] = v2[3];
        MyBase::mm[14] = v3[3];
        MyBase::mm[15] = v4[3];
    }

    // Set this matrix to zero
    void setZero()
    {
        MyBase::mm[ 0] = 0;
        MyBase::mm[ 1] = 0;
        MyBase::mm[ 2] = 0;
        MyBase::mm[ 3] = 0;
        MyBase::mm[ 4] = 0;
        MyBase::mm[ 5] = 0;
        MyBase::mm[ 6] = 0;
        MyBase::mm[ 7] = 0;
        MyBase::mm[ 8] = 0;
        MyBase::mm[ 9] = 0;
        MyBase::mm[10] = 0;
        MyBase::mm[11] = 0;
        MyBase::mm[12] = 0;
        MyBase::mm[13] = 0;
        MyBase::mm[14] = 0;
        MyBase::mm[15] = 0;
    }

    /// Set this matrix to identity
    void setIdentity()
    {
        MyBase::mm[ 0] = 1;
        MyBase::mm[ 1] = 0;
        MyBase::mm[ 2] = 0;
        MyBase::mm[ 3] = 0;

        MyBase::mm[ 4] = 0;
        MyBase::mm[ 5] = 1;
        MyBase::mm[ 6] = 0;
        MyBase::mm[ 7] = 0;

        MyBase::mm[ 8] = 0;
        MyBase::mm[ 9] = 0;
        MyBase::mm[10] = 1;
        MyBase::mm[11] = 0;

        MyBase::mm[12] = 0;
        MyBase::mm[13] = 0;
        MyBase::mm[14] = 0;
        MyBase::mm[15] = 1;
    }


    /// Set upper left to a Mat3
    void setMat3(const Mat3<T> &m)
    {
        for (int i = 0; i < 3; i++)
            for (int j=0; j < 3; j++)
                MyBase::mm[i*4+j] = m[i][j];
    }

    Mat3<T> getMat3() const
    {
        Mat3<T> m;

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                m[i][j] = MyBase::mm[i*4+j];

        return m;
    }

    /// Return the translation component
    Vec3<T> getTranslation() const
    {
        return Vec3<T>(MyBase::mm[12], MyBase::mm[13], MyBase::mm[14]);
    }

    void setTranslation(const Vec3<T> &t)
    {
        MyBase::mm[12] = t[0];
        MyBase::mm[13] = t[1];
        MyBase::mm[14] = t[2];
    }

    /// Assignment operator
    template<typename Source>
    const Mat4& operator=(const Mat4<Source> &m)
    {
        const Source *src = m.asPointer();

        // don't suppress warnings when assigning from different numerical types
        std::copy(src, (src + this->numElements()), MyBase::mm);
        return *this;
    }

    /// Return @c true if this matrix is equivalent to @a m within a tolerance of @a eps.
    bool eq(const Mat4 &m, T eps=1.0e-8) const
    {
        for (int i = 0; i < 16; i++) {
            if (!isApproxEqual(MyBase::mm[i], m.mm[i], eps))
                return false;
        }
        return true;
    }

    /// Negation operator, for e.g.   m1 = -m2;
    Mat4<T> operator-() const
    {
        return Mat4<T>(
                       -MyBase::mm[ 0], -MyBase::mm[ 1], -MyBase::mm[ 2], -MyBase::mm[ 3],
                       -MyBase::mm[ 4], -MyBase::mm[ 5], -MyBase::mm[ 6], -MyBase::mm[ 7],
                       -MyBase::mm[ 8], -MyBase::mm[ 9], -MyBase::mm[10], -MyBase::mm[11],
                       -MyBase::mm[12], -MyBase::mm[13], -MyBase::mm[14], -MyBase::mm[15]
                       );
    } // trivial

    /// Multiply each element of this matrix by @a scalar.
    template <typename S>
    const Mat4<T>& operator*=(S scalar)
    {
        MyBase::mm[ 0] *= scalar;
        MyBase::mm[ 1] *= scalar;
        MyBase::mm[ 2] *= scalar;
        MyBase::mm[ 3] *= scalar;

        MyBase::mm[ 4] *= scalar;
        MyBase::mm[ 5] *= scalar;
        MyBase::mm[ 6] *= scalar;
        MyBase::mm[ 7] *= scalar;

        MyBase::mm[ 8] *= scalar;
        MyBase::mm[ 9] *= scalar;
        MyBase::mm[10] *= scalar;
        MyBase::mm[11] *= scalar;

        MyBase::mm[12] *= scalar;
        MyBase::mm[13] *= scalar;
        MyBase::mm[14] *= scalar;
        MyBase::mm[15] *= scalar;
        return *this;
    }

    /// Add each element of the given matrix to the corresponding element of this matrix.
    template <typename S>
    const Mat4<T> &operator+=(const Mat4<S> &m1)
    {
        const S* s = m1.asPointer();

        MyBase::mm[ 0] += s[ 0];
        MyBase::mm[ 1] += s[ 1];
        MyBase::mm[ 2] += s[ 2];
        MyBase::mm[ 3] += s[ 3];

        MyBase::mm[ 4] += s[ 4];
        MyBase::mm[ 5] += s[ 5];
        MyBase::mm[ 6] += s[ 6];
        MyBase::mm[ 7] += s[ 7];

        MyBase::mm[ 8] += s[ 8];
        MyBase::mm[ 9] += s[ 9];
        MyBase::mm[10] += s[10];
        MyBase::mm[11] += s[11];

        MyBase::mm[12] += s[12];
        MyBase::mm[13] += s[13];
        MyBase::mm[14] += s[14];
        MyBase::mm[15] += s[15];

        return *this;
    }

    /// Subtract each element of the given matrix from the corresponding element of this matrix.
    template <typename S>
    const Mat4<T> &operator-=(const Mat4<S> &m1)
    {
        const S* s = m1.asPointer();

        MyBase::mm[ 0] -= s[ 0];
        MyBase::mm[ 1] -= s[ 1];
        MyBase::mm[ 2] -= s[ 2];
        MyBase::mm[ 3] -= s[ 3];

        MyBase::mm[ 4] -= s[ 4];
        MyBase::mm[ 5] -= s[ 5];
        MyBase::mm[ 6] -= s[ 6];
        MyBase::mm[ 7] -= s[ 7];

        MyBase::mm[ 8] -= s[ 8];
        MyBase::mm[ 9] -= s[ 9];
        MyBase::mm[10] -= s[10];
        MyBase::mm[11] -= s[11];

        MyBase::mm[12] -= s[12];
        MyBase::mm[13] -= s[13];
        MyBase::mm[14] -= s[14];
        MyBase::mm[15] -= s[15];

        return *this;
    }

    /// Multiply this matrix by the given matrix.
    template <typename S>
    const Mat4<T> &operator*=(const Mat4<S> &m1)
    {
        Mat4<T> m0(*this);

        const T* s0 = m0.asPointer();
        const S* s1 = m1.asPointer();

        for (int i = 0; i < 4; i++) {
            int i4 = 4 * i;
            MyBase::mm[i4+0] = static_cast<T>(s0[i4+0] * s1[ 0] +
                                              s0[i4+1] * s1[ 4] +
                                              s0[i4+2] * s1[ 8] +
                                              s0[i4+3] * s1[12]);

            MyBase::mm[i4+1] = static_cast<T>(s0[i4+0] * s1[ 1] +
                                              s0[i4+1] * s1[ 5] +
                                              s0[i4+2] * s1[ 9] +
                                              s0[i4+3] * s1[13]);

            MyBase::mm[i4+2] = static_cast<T>(s0[i4+0] * s1[ 2] +
                                              s0[i4+1] * s1[ 6] +
                                              s0[i4+2] * s1[10] +
                                              s0[i4+3] * s1[14]);

            MyBase::mm[i4+3] = static_cast<T>(s0[i4+0] * s1[ 3] +
                                              s0[i4+1] * s1[ 7] +
                                              s0[i4+2] * s1[11] +
                                              s0[i4+3] * s1[15]);
        }
        return *this;
    }

    /// @return transpose of this
    Mat4 transpose() const
    {
        return Mat4<T>(
                       MyBase::mm[ 0], MyBase::mm[ 4], MyBase::mm[ 8], MyBase::mm[12],
                       MyBase::mm[ 1], MyBase::mm[ 5], MyBase::mm[ 9], MyBase::mm[13],
                       MyBase::mm[ 2], MyBase::mm[ 6], MyBase::mm[10], MyBase::mm[14],
                       MyBase::mm[ 3], MyBase::mm[ 7], MyBase::mm[11], MyBase::mm[15]
                       );
    }


    /// @return inverse of this
    /// @throw ArithmeticError if singular
    Mat4 inverse(T tolerance = 0) const
    {
        //
        // inv [ A  | b ]  =  [ E  | f ]    A: 3x3, b: 3x1, c': 1x3 d: 1x1
        //     [ c' | d ]     [ g' | h ]
        //
        // If A is invertible use
        //
        //   E  = A^-1 + p*h*r
        //   p  = A^-1 * b
        //   f  = -p * h
        //   g' = -h * c'
        //   h  = 1 / (d - c'*p)
        //   r' = c'*A^-1
        //
        // Otherwise use gauss-jordan elimination
        //

        //
        // We create this alias to ourself so we can easily use own subscript
        // operator.
        const Mat4<T>& m(*this);

        T m0011 = m[0][0] * m[1][1];
        T m0012 = m[0][0] * m[1][2];
        T m0110 = m[0][1] * m[1][0];
        T m0210 = m[0][2] * m[1][0];
        T m0120 = m[0][1] * m[2][0];
        T m0220 = m[0][2] * m[2][0];

        T detA = m0011 * m[2][2] - m0012 * m[2][1] - m0110 * m[2][2]
               + m0210 * m[2][1] + m0120 * m[1][2] - m0220 * m[1][1];

        bool hasPerspective =
                (!isExactlyEqual(m[0][3], T(0.0)) ||
                 !isExactlyEqual(m[1][3], T(0.0)) ||
                 !isExactlyEqual(m[2][3], T(0.0)) ||
                 !isExactlyEqual(m[3][3], T(1.0)));

        T det;
        if (hasPerspective) {
            det = m[0][3] * det3(m, 1,2,3, 0,2,1)
                + m[1][3] * det3(m, 2,0,3, 0,2,1)
                + m[2][3] * det3(m, 3,0,1, 0,2,1)
                + m[3][3] * detA;
        } else {
            det = detA * m[3][3];
        }

        Mat4<T> inv;
        bool invertible;

        if (isApproxEqual(det,T(0.0),tolerance)) {
            invertible = false;

        } else if (isApproxEqual(detA,T(0.0),T(1e-8))) {
            // det is too small to rely on inversion by subblocks
            invertible = m.invert(inv, tolerance);

        } else {
            invertible = true;
            detA = 1.0 / detA;

            //
            // Calculate A^-1
            //
            inv[0][0] = detA * ( m[1][1] * m[2][2] - m[1][2] * m[2][1]);
            inv[0][1] = detA * (-m[0][1] * m[2][2] + m[0][2] * m[2][1]);
            inv[0][2] = detA * ( m[0][1] * m[1][2] - m[0][2] * m[1][1]);

            inv[1][0] = detA * (-m[1][0] * m[2][2] + m[1][2] * m[2][0]);
            inv[1][1] = detA * ( m[0][0] * m[2][2] - m0220);
            inv[1][2] = detA * ( m0210   - m0012);

            inv[2][0] = detA * ( m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            inv[2][1] = detA * ( m0120 - m[0][0] * m[2][1]);
            inv[2][2] = detA * ( m0011 - m0110);

            if (hasPerspective) {
                //
                // Calculate r, p, and h
                //
                Vec3<T> r;
                r[0] = m[3][0] * inv[0][0] + m[3][1] * inv[1][0]
                     + m[3][2] * inv[2][0];
                r[1] = m[3][0] * inv[0][1] + m[3][1] * inv[1][1]
                     + m[3][2] * inv[2][1];
                r[2] = m[3][0] * inv[0][2] + m[3][1] * inv[1][2]
                     + m[3][2] * inv[2][2];

                Vec3<T> p;
                p[0] = inv[0][0] * m[0][3] + inv[0][1] * m[1][3]
                     + inv[0][2] * m[2][3];
                p[1] = inv[1][0] * m[0][3] + inv[1][1] * m[1][3]
                     + inv[1][2] * m[2][3];
                p[2] = inv[2][0] * m[0][3] + inv[2][1] * m[1][3]
                     + inv[2][2] * m[2][3];

                T h = m[3][3] - p.dot(Vec3<T>(m[3][0],m[3][1],m[3][2]));
                if (isApproxEqual(h,T(0.0),tolerance)) {
                    invertible = false;

                } else {
                    h = 1.0 / h;

                    //
                    // Calculate h, g, and f
                    //
                    inv[3][3] = h;
                    inv[3][0] = -h * r[0];
                    inv[3][1] = -h * r[1];
                    inv[3][2] = -h * r[2];

                    inv[0][3] = -h * p[0];
                    inv[1][3] = -h * p[1];
                    inv[2][3] = -h * p[2];

                    //
                    // Calculate E
                    //
                    p *= h;
                    inv[0][0] += p[0] * r[0];
                    inv[0][1] += p[0] * r[1];
                    inv[0][2] += p[0] * r[2];
                    inv[1][0] += p[1] * r[0];
                    inv[1][1] += p[1] * r[1];
                    inv[1][2] += p[1] * r[2];
                    inv[2][0] += p[2] * r[0];
                    inv[2][1] += p[2] * r[1];
                    inv[2][2] += p[2] * r[2];
                }
            } else {
                // Equations are much simpler in the non-perspective case
                inv[3][0] = - (m[3][0] * inv[0][0] + m[3][1] * inv[1][0]
                                + m[3][2] * inv[2][0]);
                inv[3][1] = - (m[3][0] * inv[0][1] + m[3][1] * inv[1][1]
                                + m[3][2] * inv[2][1]);
                inv[3][2] = - (m[3][0] * inv[0][2] + m[3][1] * inv[1][2]
                                + m[3][2] * inv[2][2]);
                inv[0][3] = 0.0;
                inv[1][3] = 0.0;
                inv[2][3] = 0.0;
                inv[3][3] = 1.0;
            }
        }

        if (!invertible) OPENVDB_THROW(ArithmeticError, "Inversion of singular 4x4 matrix");
        return inv;
    }


    /// Determinant of matrix
    T det() const
    {
        const T *ap;
        Mat3<T> submat;
        T       det;
        T       *sp;
        int     i, j, k, sign;

        det = 0;
        sign = 1;
        for (i = 0; i < 4; i++) {
            ap = &MyBase::mm[ 0];
            sp = submat.asPointer();
            for (j = 0; j < 4; j++) {
                for (k = 0; k < 4; k++) {
                    if ((k != i) && (j != 0)) {
                        *sp++ = *ap;
                    }
                    ap++;
                }
            }

            det += sign * MyBase::mm[i] * submat.det();
            sign = -sign;
        }

        return det;
    }

    /// Sets the matrix to a matrix that translates by v
    static Mat4 translation(const Vec3d& v)
    {
        return Mat4(
            T(1),     T(0),    T(0),     T(0),
            T(0),     T(1),    T(0),     T(0),
            T(0),     T(0),    T(1),     T(0),
            T(v.x()), T(v.y()),T(v.z()), T(1));
    }

    /// Sets the matrix to a matrix that translates by v
    template <typename T0>
    void setToTranslation(const Vec3<T0>& v)
    {
        MyBase::mm[ 0] = 1;
        MyBase::mm[ 1] = 0;
        MyBase::mm[ 2] = 0;
        MyBase::mm[ 3] = 0;

        MyBase::mm[ 4] = 0;
        MyBase::mm[ 5] = 1;
        MyBase::mm[ 6] = 0;
        MyBase::mm[ 7] = 0;

        MyBase::mm[ 8] = 0;
        MyBase::mm[ 9] = 0;
        MyBase::mm[10] = 1;
        MyBase::mm[11] = 0;

        MyBase::mm[12] = v.x();
        MyBase::mm[13] = v.y();
        MyBase::mm[14] = v.z();
        MyBase::mm[15] = 1;
    }

    /// Left multiples by the specified translation, i.e.  Trans * (*this)
    template <typename T0>
    void preTranslate(const Vec3<T0>& tr)
    {
        Vec3<T> tmp(tr.x(), tr.y(), tr.z());
        Mat4<T> Tr = Mat4<T>::translation(tmp);

        *this =  Tr * (*this);

    }

    /// Right multiplies by the specified translation matrix, i.e. (*this) * Trans
    template <typename T0>
    void postTranslate(const Vec3<T0>& tr)
    {
        Vec3<T> tmp(tr.x(), tr.y(), tr.z());
        Mat4<T> Tr = Mat4<T>::translation(tmp);

        *this = (*this) * Tr;

    }


    /// Sets the matrix to a matrix that scales by v
    template <typename T0>
    void setToScale(const Vec3<T0>& v)
    {
        this->setIdentity();
        MyBase::mm[ 0] = v.x();
        MyBase::mm[ 5] = v.y();
        MyBase::mm[10] = v.z();
    }

    // Left multiples by the specified scale matrix, i.e. Sc * (*this)
    template <typename T0>
    void preScale(const Vec3<T0>& v)
    {
        MyBase::mm[ 0] *= v.x();
        MyBase::mm[ 1] *= v.x();
        MyBase::mm[ 2] *= v.x();
        MyBase::mm[ 3] *= v.x();

        MyBase::mm[ 4] *= v.y();
        MyBase::mm[ 5] *= v.y();
        MyBase::mm[ 6] *= v.y();
        MyBase::mm[ 7] *= v.y();

        MyBase::mm[ 8] *= v.z();
        MyBase::mm[ 9] *= v.z();
        MyBase::mm[10] *= v.z();
        MyBase::mm[11] *= v.z();
    }



    // Right multiples by the specified scale matrix, i.e. (*this) * Sc
    template <typename T0>
    void postScale(const Vec3<T0>& v)
    {

        MyBase::mm[ 0] *= v.x();
        MyBase::mm[ 1] *= v.y();
        MyBase::mm[ 2] *= v.z();

        MyBase::mm[ 4] *= v.x();
        MyBase::mm[ 5] *= v.y();
        MyBase::mm[ 6] *= v.z();

        MyBase::mm[ 8] *= v.x();
        MyBase::mm[ 9] *= v.y();
        MyBase::mm[10] *= v.z();

        MyBase::mm[12] *= v.x();
        MyBase::mm[13] *= v.y();
        MyBase::mm[14] *= v.z();

    }


    /// @brief Sets the matrix to a rotation about the given axis.
    /// @param axis The axis (one of X, Y, Z) to rotate about.
    /// @param angle The rotation angle, in radians.
    void setToRotation(Axis axis, T angle) {*this = rotation<Mat4<T> >(axis, angle);}

    /// @brief Sets the matrix to a rotation about an arbitrary axis
    /// @param axis The axis of rotation (cannot be zero-length)
    /// @param angle The rotation angle, in radians.
    void setToRotation(const Vec3<T>& axis, T angle) {*this = rotation<Mat4<T> >(axis, angle);}

    /// @brief Sets the matrix to a rotation that maps v1 onto v2 about the cross
    /// product of v1 and v2.
    void setToRotation(const Vec3<T>& v1, const Vec3<T>& v2) {*this = rotation<Mat4<T> >(v1, v2);}


    /// @brief Left multiplies by a rotation clock-wiseabout the given axis into this matrix.
    /// @param axis The axis (one of X, Y, Z) of rotation.
    /// @param angle The clock-wise rotation angle, in radians.
    void preRotate(Axis axis, T angle)
    {
        T c = static_cast<T>(cos(angle));
        T s = -static_cast<T>(sin(angle)); // the "-" makes it clockwise

        switch (axis) {
        case X_AXIS:
            {
                T a4, a5, a6, a7;

                a4 = c * MyBase::mm[ 4] - s * MyBase::mm[ 8];
                a5 = c * MyBase::mm[ 5] - s * MyBase::mm[ 9];
                a6 = c * MyBase::mm[ 6] - s * MyBase::mm[10];
                a7 = c * MyBase::mm[ 7] - s * MyBase::mm[11];


                MyBase::mm[ 8] = s * MyBase::mm[ 4] + c * MyBase::mm[ 8];
                MyBase::mm[ 9] = s * MyBase::mm[ 5] + c * MyBase::mm[ 9];
                MyBase::mm[10] = s * MyBase::mm[ 6] + c * MyBase::mm[10];
                MyBase::mm[11] = s * MyBase::mm[ 7] + c * MyBase::mm[11];

                MyBase::mm[ 4] = a4;
                MyBase::mm[ 5] = a5;
                MyBase::mm[ 6] = a6;
                MyBase::mm[ 7] = a7;
            }
            break;

        case Y_AXIS:
            {
                T a0, a1, a2, a3;

                a0 = c * MyBase::mm[ 0] + s * MyBase::mm[ 8];
                a1 = c * MyBase::mm[ 1] + s * MyBase::mm[ 9];
                a2 = c * MyBase::mm[ 2] + s * MyBase::mm[10];
                a3 = c * MyBase::mm[ 3] + s * MyBase::mm[11];

                MyBase::mm[ 8] = -s * MyBase::mm[ 0] + c * MyBase::mm[ 8];
                MyBase::mm[ 9] = -s * MyBase::mm[ 1] + c * MyBase::mm[ 9];
                MyBase::mm[10] = -s * MyBase::mm[ 2] + c * MyBase::mm[10];
                MyBase::mm[11] = -s * MyBase::mm[ 3] + c * MyBase::mm[11];


                MyBase::mm[ 0] = a0;
                MyBase::mm[ 1] = a1;
                MyBase::mm[ 2] = a2;
                MyBase::mm[ 3] = a3;
            }
            break;

        case Z_AXIS:
            {
                T a0, a1, a2, a3;

                a0 = c * MyBase::mm[ 0] - s * MyBase::mm[ 4];
                a1 = c * MyBase::mm[ 1] - s * MyBase::mm[ 5];
                a2 = c * MyBase::mm[ 2] - s * MyBase::mm[ 6];
                a3 = c * MyBase::mm[ 3] - s * MyBase::mm[ 7];

                MyBase::mm[ 4] = s * MyBase::mm[ 0] + c * MyBase::mm[ 4];
                MyBase::mm[ 5] = s * MyBase::mm[ 1] + c * MyBase::mm[ 5];
                MyBase::mm[ 6] = s * MyBase::mm[ 2] + c * MyBase::mm[ 6];
                MyBase::mm[ 7] = s * MyBase::mm[ 3] + c * MyBase::mm[ 7];

                MyBase::mm[ 0] = a0;
                MyBase::mm[ 1] = a1;
                MyBase::mm[ 2] = a2;
                MyBase::mm[ 3] = a3;
            }
            break;

        default:
            assert(axis==X_AXIS || axis==Y_AXIS || axis==Z_AXIS);
        }
    }


    /// @brief Right multiplies by a rotation clock-wiseabout the given axis into this matrix.
    /// @param axis The axis (one of X, Y, Z) of rotation.
    /// @param angle The clock-wise rotation angle, in radians.
    void postRotate(Axis axis, T angle)
    {
        T c = static_cast<T>(cos(angle));
        T s = -static_cast<T>(sin(angle)); // the "-" makes it clockwise



        switch (axis) {
        case X_AXIS:
            {
                T a2, a6, a10, a14;

                a2  = c * MyBase::mm[ 2] - s * MyBase::mm[ 1];
                a6  = c * MyBase::mm[ 6] - s * MyBase::mm[ 5];
                a10 = c * MyBase::mm[10] - s * MyBase::mm[ 9];
                a14 = c * MyBase::mm[14] - s * MyBase::mm[13];


                MyBase::mm[ 1] = c * MyBase::mm[ 1] + s * MyBase::mm[ 2];
                MyBase::mm[ 5] = c * MyBase::mm[ 5] + s * MyBase::mm[ 6];
                MyBase::mm[ 9] = c * MyBase::mm[ 9] + s * MyBase::mm[10];
                MyBase::mm[13] = c * MyBase::mm[13] + s * MyBase::mm[14];

                MyBase::mm[ 2] = a2;
                MyBase::mm[ 6] = a6;
                MyBase::mm[10] = a10;
                MyBase::mm[14] = a14;
            }
            break;

        case Y_AXIS:
            {
                T a2, a6, a10, a14;

                a2  = c * MyBase::mm[ 2] + s * MyBase::mm[ 0];
                a6  = c * MyBase::mm[ 6] + s * MyBase::mm[ 4];
                a10 = c * MyBase::mm[10] + s * MyBase::mm[ 8];
                a14 = c * MyBase::mm[14] + s * MyBase::mm[12];

                MyBase::mm[ 0] = c * MyBase::mm[ 0] - s * MyBase::mm[ 2];
                MyBase::mm[ 4] = c * MyBase::mm[ 4] - s * MyBase::mm[ 6];
                MyBase::mm[ 8] = c * MyBase::mm[ 8] - s * MyBase::mm[10];
                MyBase::mm[12] = c * MyBase::mm[12] - s * MyBase::mm[14];

                MyBase::mm[ 2] = a2;
                MyBase::mm[ 6] = a6;
                MyBase::mm[10] = a10;
                MyBase::mm[14] = a14;
            }
            break;

        case Z_AXIS:
            {
                T a1, a5, a9, a13;

                a1  = c * MyBase::mm[ 1] - s * MyBase::mm[ 0];
                a5  = c * MyBase::mm[ 5] - s * MyBase::mm[ 4];
                a9  = c * MyBase::mm[ 9] - s * MyBase::mm[ 8];
                a13 = c * MyBase::mm[13] - s * MyBase::mm[12];

                MyBase::mm[ 0] = c * MyBase::mm[ 0] + s * MyBase::mm[ 1];
                MyBase::mm[ 4] = c * MyBase::mm[ 4] + s * MyBase::mm[ 5];
                MyBase::mm[ 8] = c * MyBase::mm[ 8] + s * MyBase::mm[ 9];
                MyBase::mm[12] = c * MyBase::mm[12] + s * MyBase::mm[13];

                MyBase::mm[ 1] = a1;
                MyBase::mm[ 5] = a5;
                MyBase::mm[ 9] = a9;
                MyBase::mm[13] = a13;

            }
            break;

        default:
            assert(axis==X_AXIS || axis==Y_AXIS || axis==Z_AXIS);
        }
    }

    /// @brief Sets the matrix to a shear along axis0 by a fraction of axis1.
    /// @param axis0 The fixed axis of the shear.
    /// @param axis1 The shear axis.
    /// @param shearby The shear factor.
    void setToShear(Axis axis0, Axis axis1, T shearby)
    {
        *this = shear<Mat4<T> >(axis0, axis1, shearby);
    }


    /// @brief Left multiplies a shearing transformation into the matrix.
    /// @see setToShear
    void preShear(Axis axis0, Axis axis1, T shear)
    {
        int index0 = static_cast<int>(axis0);
        int index1 = static_cast<int>(axis1);

        // to row "index1" add a multiple of the index0 row
        MyBase::mm[index1 * 4 + 0] += shear * MyBase::mm[index0 * 4 + 0];
        MyBase::mm[index1 * 4 + 1] += shear * MyBase::mm[index0 * 4 + 1];
        MyBase::mm[index1 * 4 + 2] += shear * MyBase::mm[index0 * 4 + 2];
        MyBase::mm[index1 * 4 + 3] += shear * MyBase::mm[index0 * 4 + 3];
    }


    /// @brief Right multiplies a shearing transformation into the matrix.
    /// @see setToShear
    void postShear(Axis axis0, Axis axis1, T shear)
    {
        int index0 = static_cast<int>(axis0);
        int index1 = static_cast<int>(axis1);

        // to collumn "index0" add a multiple of the index1 row
        MyBase::mm[index0 +  0] += shear * MyBase::mm[index1 +  0];
        MyBase::mm[index0 +  4] += shear * MyBase::mm[index1 +  4];
        MyBase::mm[index0 +  8] += shear * MyBase::mm[index1 +  8];
        MyBase::mm[index0 + 12] += shear * MyBase::mm[index1 + 12];

    }

    /// Transform a Vec4 by post-multiplication.
    template<typename T0>
    Vec4<T0> transform(const Vec4<T0> &v) const
    {
        return static_cast< Vec4<T0> >(v * *this);
    }

    /// Transform a Vec3 by post-multiplication, without homogenous division.
    template<typename T0>
    Vec3<T0> transform(const Vec3<T0> &v) const
    {
        return static_cast< Vec3<T0> >(v * *this);
    }

    /// Transform a Vec4 by pre-multiplication.
    template<typename T0>
    Vec4<T0> pretransform(const Vec4<T0> &v) const
    {
        return static_cast< Vec4<T0> >(*this * v);
    }

    /// Transform a Vec3 by pre-multiplication, without homogenous division.
    template<typename T0>
    Vec3<T0> pretransform(const Vec3<T0> &v) const
    {
        return static_cast< Vec3<T0> >(*this * v);
    }

    /// Transform a Vec3 by post-multiplication, doing homogenous divison.
    template<typename T0>
    Vec3<T0> transformH(const Vec3<T0> &p) const
    {
        T0  w;

        // w = p * (*this).col(3);
        w = static_cast<T0>(p[0] * MyBase::mm[ 3] + p[1] * MyBase::mm[ 7]
            + p[2] * MyBase::mm[11] + MyBase::mm[15]);

        if ( !isExactlyEqual(w , 0.0) ) {
            return Vec3<T0>(static_cast<T0>((p[0] * MyBase::mm[ 0] + p[1] * MyBase::mm[ 4] +
                                             p[2] * MyBase::mm[ 8] + MyBase::mm[12]) / w),
                            static_cast<T0>((p[0] * MyBase::mm[ 1] + p[1] * MyBase::mm[ 5] +
                                             p[2] * MyBase::mm[ 9] + MyBase::mm[13]) / w),
                            static_cast<T0>((p[0] * MyBase::mm[ 2] + p[1] * MyBase::mm[ 6] +
                                             p[2] * MyBase::mm[10] + MyBase::mm[14]) / w));
        }

        return Vec3<T0>(0, 0, 0);
    }

    /// Transform a Vec3 by pre-multiplication, doing homogenous division.
    template<typename T0>
    Vec3<T0> pretransformH(const Vec3<T0> &p) const
    {
        T0  w;

        // w = p * (*this).col(3);
        w = p[0] * MyBase::mm[12] + p[1] * MyBase::mm[13] + p[2] * MyBase::mm[14] + MyBase::mm[15];

        if ( !isExactlyEqual(w , 0.0) ) {
            return Vec3<T0>(static_cast<T0>((p[0] * MyBase::mm[ 0] + p[1] * MyBase::mm[ 1] +
                                             p[2] * MyBase::mm[ 2] + MyBase::mm[ 3]) / w),
                            static_cast<T0>((p[0] * MyBase::mm[ 4] + p[1] * MyBase::mm[ 5] +
                                             p[2] * MyBase::mm[ 6] + MyBase::mm[ 7]) / w),
                            static_cast<T0>((p[0] * MyBase::mm[ 8]  + p[1] * MyBase::mm[ 9] +
                                             p[2] * MyBase::mm[10] + MyBase::mm[11]) / w));
        }

        return Vec3<T0>(0, 0, 0);
    }

    /// Transform a Vec3 by post-multiplication, without translation.
    template<typename T0>
    Vec3<T0> transform3x3(const Vec3<T0> &v) const
    {
        return Vec3<T0>(
            static_cast<T0>(v[0] * MyBase::mm[ 0] + v[1] * MyBase::mm[ 4] + v[2] * MyBase::mm[ 8]),
            static_cast<T0>(v[0] * MyBase::mm[ 1] + v[1] * MyBase::mm[ 5] + v[2] * MyBase::mm[ 9]),
            static_cast<T0>(v[0] * MyBase::mm[ 2] + v[1] * MyBase::mm[ 6] + v[2] * MyBase::mm[10]));
    }


private:
    bool invert(Mat4<T> &inverse, T tolerance) const;

    T det2(const Mat4<T> &a, int i0, int i1, int j0, int j1) const {
        int i0row = i0 * 4;
        int i1row = i1 * 4;
        return a.mm[i0row+j0]*a.mm[i1row+j1] - a.mm[i0row+j1]*a.mm[i1row+j0];
    }

    T det3(const Mat4<T> &a, int i0, int i1, int i2,
           int j0, int j1, int j2) const {
        int i0row = i0 * 4;
        return a.mm[i0row+j0]*det2(a, i1,i2, j1,j2) +
            a.mm[i0row+j1]*det2(a, i1,i2, j2,j0) +
            a.mm[i0row+j2]*det2(a, i1,i2, j0,j1);
    }
}; // class Mat4


/// @relates Mat4
/// @brief Equality operator, does exact floating point comparisons
template <typename T0, typename T1>
bool operator==(const Mat4<T0> &m0, const Mat4<T1> &m1)
{
    const T0 *t0 = m0.asPointer();
    const T1 *t1 = m1.asPointer();

    for (int i=0; i<16; ++i) if (!isExactlyEqual(t0[i], t1[i])) return false;
    return true;
}

/// @relates Mat4
/// @brief Inequality operator, does exact floating point comparisons
template <typename T0, typename T1>
bool operator!=(const Mat4<T0> &m0, const Mat4<T1> &m1) { return !(m0 == m1); }

/// @relates Mat4
/// @brief Multiply each element of the given matrix by @a scalar and return the result.
template <typename S, typename T>
Mat4<typename promote<S, T>::type> operator*(S scalar, const Mat4<T> &m)
{
    return m*scalar;
}

/// @relates Mat4
/// @brief Multiply each element of the given matrix by @a scalar and return the result.
template <typename S, typename T>
Mat4<typename promote<S, T>::type> operator*(const Mat4<T> &m, S scalar)
{
    Mat4<typename promote<S, T>::type> result(m);
    result *= scalar;
    return result;
}

/// @relates Mat4
/// @brief Multiply @a _m by @a _v and return the resulting vector.
template<typename T, typename MT>
inline Vec4<typename promote<T, MT>::type>
operator*(const Mat4<MT> &_m,
          const Vec4<T> &_v)
{
    MT const *m = _m.asPointer();
    return Vec4<typename promote<T, MT>::type>(
        _v[0]*m[0]  + _v[1]*m[1]  + _v[2]*m[2]  + _v[3]*m[3],
        _v[0]*m[4]  + _v[1]*m[5]  + _v[2]*m[6]  + _v[3]*m[7],
        _v[0]*m[8]  + _v[1]*m[9]  + _v[2]*m[10] + _v[3]*m[11],
        _v[0]*m[12] + _v[1]*m[13] + _v[2]*m[14] + _v[3]*m[15]);
}

/// @relates Mat4
/// @brief Multiply @a _v by @a _m and return the resulting vector.
template<typename T, typename MT>
inline Vec4<typename promote<T, MT>::type>
operator*(const Vec4<T> &_v,
          const Mat4<MT> &_m)
{
    MT const *m = _m.asPointer();
    return Vec4<typename promote<T, MT>::type>(
        _v[0]*m[0] + _v[1]*m[4] + _v[2]*m[8]  + _v[3]*m[12],
        _v[0]*m[1] + _v[1]*m[5] + _v[2]*m[9]  + _v[3]*m[13],
        _v[0]*m[2] + _v[1]*m[6] + _v[2]*m[10] + _v[3]*m[14],
        _v[0]*m[3] + _v[1]*m[7] + _v[2]*m[11] + _v[3]*m[15]);
}

/// @relates Mat4
/// @brief Multiply @a _m by @a _v and return the resulting vector.
template<typename T, typename MT>
inline Vec3<typename promote<T, MT>::type>
operator*(const Mat4<MT> &_m, const Vec3<T> &_v)
{
    MT const *m = _m.asPointer();
    return Vec3<typename promote<T, MT>::type>(
        _v[0]*m[0] + _v[1]*m[1] + _v[2]*m[2]  + m[3],
        _v[0]*m[4] + _v[1]*m[5] + _v[2]*m[6]  + m[7],
        _v[0]*m[8] + _v[1]*m[9] + _v[2]*m[10] + m[11]);
}

/// @relates Mat4
/// @brief Multiply @a _v by @a _m and return the resulting vector.
template<typename T, typename MT>
inline Vec3<typename promote<T, MT>::type>
operator*(const Vec3<T> &_v, const Mat4<MT> &_m)
{
    MT const *m = _m.asPointer();
    return Vec3<typename promote<T, MT>::type>(
        _v[0]*m[0] + _v[1]*m[4] + _v[2]*m[8]  + m[12],
        _v[0]*m[1] + _v[1]*m[5] + _v[2]*m[9]  + m[13],
        _v[0]*m[2] + _v[1]*m[6] + _v[2]*m[10] + m[14]);
}

/// @relates Mat4
/// @brief Add corresponding elements of @a m0 and @a m1 and return the result.
template <typename T0, typename T1>
Mat4<typename promote<T0, T1>::type>
operator+(const Mat4<T0> &m0, const Mat4<T1> &m1)
{
    Mat4<typename promote<T0, T1>::type> result(m0);
    result += m1;
    return result;
}

/// @relates Mat4
/// @brief Subtract corresponding elements of @a m0 and @a m1 and return the result.
template <typename T0, typename T1>
Mat4<typename promote<T0, T1>::type>
operator-(const Mat4<T0> &m0, const Mat4<T1> &m1)
{
    Mat4<typename promote<T0, T1>::type> result(m0);
    result -= m1;
    return result;
}

/// @relates Mat4
/// @brief Multiply @a m0 by @a m1 and return the resulting matrix.
template <typename T0, typename T1>
Mat4<typename promote<T0, T1>::type>
operator*(const Mat4<T0> &m0, const Mat4<T1> &m1)
{
    Mat4<typename promote<T0, T1>::type> result(m0);
    result *= m1;
    return result;
}


/// Transform a Vec3 by pre-multiplication, without translation.
/// Presumes this matrix is inverse of coordinate transform
/// Synonymous to "pretransform3x3"
template<typename T0, typename T1>
Vec3<T1> transformNormal(const Mat4<T0> &m, const Vec3<T1> &n)
{
    return Vec3<T1>(
        static_cast<T1>(m[0][0]*n[0] + m[0][1]*n[1] + m[0][2]*n[2]),
        static_cast<T1>(m[1][0]*n[0] + m[1][1]*n[1] + m[1][2]*n[2]),
        static_cast<T1>(m[2][0]*n[0] + m[2][1]*n[1] + m[2][2]*n[2]));
}


/// Invert via gauss-jordan elimination. Modified from dreamworks internal mx library
template<typename T>
bool Mat4<T>::invert(Mat4<T> &inverse, T tolerance) const
{
    Mat4<T> temp(*this);
    inverse.setIdentity();

    // Forward elimination step
    double det = 1.0;
    for (int i = 0; i < 4; ++i) {
        int row = i;
        double max = fabs(temp[i][i]);

        for (int k = i+1; k < 4; ++k) {
            if (fabs(temp[k][i]) > max) {
                row = k;
                max = fabs(temp[k][i]);
            }
        }

        if (isExactlyEqual(max, 0.0)) return false;

        // must move pivot to row i
        if (row != i) {
            det = -det;
            for (int k = 0; k < 4; ++k) {
                std::swap(temp[row][k], temp[i][k]);
                std::swap(inverse[row][k], inverse[i][k]);
            }
        }

        double pivot = temp[i][i];
        det *= pivot;

        // scale row i
        for (int k = 0; k < 4; ++k) {
            temp[i][k] /= pivot;
            inverse[i][k] /= pivot;
        }

        // eliminate in rows below i
        for (int j = i+1; j < 4; ++j) {
            double t = temp[j][i];
            if (!isExactlyEqual(t, 0.0)) {
                // subtract scaled row i from row j
                for (int k = 0; k < 4; ++k) {
                    temp[j][k] -= temp[i][k] * t;
                    inverse[j][k] -= inverse[i][k] * t;
                }
            }
        }
    }

    // Backward elimination step
    for (int i = 3; i > 0; --i) {
        for (int j = 0; j < i; ++j) {
            double t = temp[j][i];

            if (!isExactlyEqual(t, 0.0)) {
                for (int k = 0; k < 4; ++k) {
                    inverse[j][k] -= inverse[i][k]*t;
                }
            }
        }
    }
    return det*det >= tolerance*tolerance;
}

template <typename T>
inline bool isAffine(const Mat4<T>& m) {
    return (m.col(3) == Vec4<T>(0, 0, 0, 1));
}

template <typename T>
inline bool hasTranslation(const Mat4<T>& m) {
    return (m.row(3) != Vec4<T>(0, 0, 0, 1));
}


using Mat4s = Mat4<float>;
using Mat4d = Mat4<double>;
using Mat4f = Mat4d;

} // namespace math


template<> inline math::Mat4s zeroVal<math::Mat4s>() { return math::Mat4s::zero(); }
template<> inline math::Mat4d zeroVal<math::Mat4d>() { return math::Mat4d::zero(); }

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_UTIL_MAT4_H_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
