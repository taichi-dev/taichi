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
//
/// @file Mat.h
/// @author Joshua Schpok

#ifndef OPENVDB_MATH_MAT_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_MAT_HAS_BEEN_INCLUDED

#include "Math.h"
#include <openvdb/Exceptions.h>
#include <algorithm> // for std::max()
#include <cmath>
#include <iostream>
#include <string>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

/// @class Mat "Mat.h"
/// A base class for square matrices.
template<unsigned SIZE, typename T>
class Mat
{
public:
    using value_type = T;
    using ValueType = T;
    enum SIZE_ { size = SIZE };

    // Number of cols, rows, elements
    static unsigned numRows() { return SIZE; }
    static unsigned numColumns() { return SIZE; }
    static unsigned numElements() { return SIZE*SIZE; }

    /// Default ctor.  Does nothing.  Required because declaring a copy (or
    /// other) constructor means the default constructor gets left out.
    Mat() { }

    /// Copy constructor.  Used when the class signature matches exactly.
    Mat(Mat const &src) {
        for (unsigned i(0); i < numElements(); ++i) {
            mm[i] = src.mm[i];
        }
    }

    Mat& operator=(Mat const& src) {
        if (&src != this) {
            for (unsigned i = 0; i < numElements(); ++i) {
                mm[i] = src.mm[i];
            }
        }
        return *this;
    }

    /// @return string representation of matrix
    /// Since output is multiline, optional indentation argument prefixes
    /// each newline with that much white space. It does not indent
    /// the first line, since you might be calling this inline:
    ///
    /// cout << "matrix: " << mat.str(7)
    ///
    /// matrix: [[1 2]
    ///          [3 4]]
    std::string
    str(unsigned indentation = 0) const {

        std::string ret;
        std::string indent;

        // We add +1 since we're indenting one for the first '['
        indent.append(indentation+1, ' ');

        ret.append("[");

        // For each row,
        for (unsigned i(0); i < SIZE; i++) {

            ret.append("[");

            // For each column
            for (unsigned j(0); j < SIZE; j++) {

                // Put a comma after everything except the last
                if (j) ret.append(", ");
                ret.append(std::to_string(mm[(i*SIZE)+j]));
            }

            ret.append("]");

            // At the end of every row (except the last)...
            if (i < SIZE - 1) {
                // ...suffix the row bracket with a comma, newline, and advance indentation.
                ret.append(",\n");
                ret.append(indent);
            }
        }

        ret.append("]");

        return ret;
    }

    /// Write a Mat to an output stream
    friend std::ostream& operator<<(
        std::ostream& ostr,
        const Mat<SIZE, T>& m)
    {
        ostr << m.str();
        return ostr;
    }

    void write(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&mm), sizeof(T)*SIZE*SIZE);
    }

    void read(std::istream& is) {
        is.read(reinterpret_cast<char*>(&mm), sizeof(T)*SIZE*SIZE);
    }

    /// Return the maximum of the absolute of all elements in this matrix
    T absMax() const {
        T x = static_cast<T>(std::fabs(mm[0]));
        for (unsigned i = 1; i < numElements(); ++i) {
            x = std::max(x, static_cast<T>(std::fabs(mm[i])));
        }
        return x;
    }

    /// True if a Nan is present in this matrix
    bool isNan() const {
        for (unsigned i = 0; i < numElements(); ++i) {
            if (std::isnan(mm[i])) return true;
        }
        return false;
    }

    /// True if an Inf is present in this matrix
    bool isInfinite() const {
        for (unsigned i = 0; i < numElements(); ++i) {
            if (std::isinf(mm[i])) return true;
        }
        return false;
    }

    /// True if no Nan or Inf values are present
    bool isFinite() const {
        for (unsigned i = 0; i < numElements(); ++i) {
            if (!std::isfinite(mm[i])) return false;
        }
        return true;
    }

    /// True if all elements are exactly zero
    bool isZero() const {
        for (unsigned i = 0; i < numElements(); ++i) {
            if (!math::isZero(mm[i])) return false;
        }
        return true;
    }

protected:
    T mm[SIZE*SIZE];
};


template<typename T> class Quat;
template<typename T> class Vec3;

/// @brief Return the rotation matrix specified by the given quaternion.
/// @details The quaternion is normalized and used to construct the matrix.
/// Note that the matrix is transposed to match post-multiplication semantics.
template<class MatType>
MatType
rotation(const Quat<typename MatType::value_type> &q,
    typename MatType::value_type eps = static_cast<typename MatType::value_type>(1.0e-8))
{
    using T = typename MatType::value_type;

    T qdot(q.dot(q));
    T s(0);

    if (!isApproxEqual(qdot, T(0.0),eps)) {
        s = T(2.0 / qdot);
    }

    T x  = s*q.x();
    T y  = s*q.y();
    T z  = s*q.z();
    T wx = x*q.w();
    T wy = y*q.w();
    T wz = z*q.w();
    T xx = x*q.x();
    T xy = y*q.x();
    T xz = z*q.x();
    T yy = y*q.y();
    T yz = z*q.y();
    T zz = z*q.z();

    MatType r;
    r[0][0]=T(1) - (yy+zz); r[0][1]=xy + wz;        r[0][2]=xz - wy;
    r[1][0]=xy - wz;        r[1][1]=T(1) - (xx+zz); r[1][2]=yz + wx;
    r[2][0]=xz + wy;        r[2][1]=yz - wx;        r[2][2]=T(1) - (xx+yy);

    if(MatType::numColumns() == 4) padMat4(r);
    return r;
}



/// @brief Return a matrix for rotation by @a angle radians about the given @a axis.
/// @param axis   The axis (one of X, Y, Z) to rotate about.
/// @param angle  The rotation angle, in radians.
template<class MatType>
MatType
rotation(Axis axis, typename MatType::value_type angle)
{
    using T = typename MatType::value_type;
    T c = static_cast<T>(cos(angle));
    T s = static_cast<T>(sin(angle));

    MatType result;
    result.setIdentity();

    switch (axis) {
    case X_AXIS:
        result[1][1]  = c;
        result[1][2]  = s;
        result[2][1]  = -s;
        result[2][2] = c;
        return result;
    case Y_AXIS:
        result[0][0]  = c;
        result[0][2]  = -s;
        result[2][0]  = s;
        result[2][2] = c;
        return result;
    case Z_AXIS:
        result[0][0] = c;
        result[0][1] = s;
        result[1][0] = -s;
        result[1][1] = c;
        return result;
    default:
        throw ValueError("Unrecognized rotation axis");
    }
}


/// @brief Return a matrix for rotation by @a angle radians about the given @a axis.
/// @note The axis must be a unit vector.
template<class MatType>
MatType
rotation(const Vec3<typename MatType::value_type> &_axis, typename MatType::value_type angle)
{
    using T = typename MatType::value_type;
    T txy, txz, tyz, sx, sy, sz;

    Vec3<T> axis(_axis.unit());

    // compute trig properties of angle:
    T c(cos(double(angle)));
    T s(sin(double(angle)));
    T t(1 - c);

    MatType result;
    // handle diagonal elements
    result[0][0] = axis[0]*axis[0] * t + c;
    result[1][1] = axis[1]*axis[1] * t + c;
    result[2][2] = axis[2]*axis[2] * t + c;

    txy = axis[0]*axis[1] * t;
    sz = axis[2] * s;

    txz = axis[0]*axis[2] * t;
    sy = axis[1] * s;

    tyz = axis[1]*axis[2] * t;
    sx = axis[0] * s;

    // right handed space
    // Contribution from rotation about 'z'
    result[0][1] = txy + sz;
    result[1][0] = txy - sz;
    // Contribution from rotation about 'y'
    result[0][2] = txz - sy;
    result[2][0] = txz + sy;
    // Contribution from rotation about 'x'
    result[1][2] = tyz + sx;
    result[2][1] = tyz - sx;

    if(MatType::numColumns() == 4) padMat4(result);
    return MatType(result);
}


/// @brief Return the Euler angles composing the given rotation matrix.
/// @details Optional axes arguments describe in what order elementary rotations
/// are applied. Note that in our convention, XYZ means Rz * Ry * Rx.
/// Because we are using rows rather than columns to represent the
/// local axes of a coordinate frame, the interpretation from a local
/// reference point of view is to first rotate about the x axis, then
/// about the newly rotated y axis, and finally by the new local z axis.
/// From a fixed reference point of view, the interpretation is to
/// rotate about the stationary world z, y, and x axes respectively.
///
/// Irrespective of the Euler angle convention, in the case of distinct
/// axes, eulerAngles() returns the x, y, and z angles in the corresponding
/// x, y, z components of the returned Vec3. For the XZX convention, the
/// left X value is returned in Vec3.x, and the right X value in Vec3.y.
/// For the ZXZ convention the left Z value is returned in Vec3.z and
/// the right Z value in Vec3.y
///
/// Examples of reconstructing r from its Euler angle decomposition
///
/// v = eulerAngles(r, ZYX_ROTATION);
/// rx.setToRotation(Vec3d(1,0,0), v[0]);
/// ry.setToRotation(Vec3d(0,1,0), v[1]);
/// rz.setToRotation(Vec3d(0,0,1), v[2]);
/// r = rx * ry * rz;
///
/// v = eulerAngles(r, ZXZ_ROTATION);
/// rz1.setToRotation(Vec3d(0,0,1), v[2]);
/// rx.setToRotation (Vec3d(1,0,0), v[0]);
/// rz2.setToRotation(Vec3d(0,0,1), v[1]);
/// r = rz2 * rx * rz1;
///
/// v = eulerAngles(r, XZX_ROTATION);
/// rx1.setToRotation (Vec3d(1,0,0), v[0]);
/// rx2.setToRotation (Vec3d(1,0,0), v[1]);
/// rz.setToRotation  (Vec3d(0,0,1), v[2]);
/// r = rx2 * rz * rx1;
///
template<class MatType>
Vec3<typename MatType::value_type>
eulerAngles(
    const MatType& mat,
    RotationOrder rotationOrder,
    typename MatType::value_type eps = static_cast<typename MatType::value_type>(1.0e-8))
{
    using ValueType = typename MatType::value_type;
    using V = Vec3<ValueType>;
    ValueType phi, theta, psi;

    switch(rotationOrder)
    {
    case XYZ_ROTATION:
        if (isApproxEqual(mat[2][0], ValueType(1.0), eps)) {
            theta = ValueType(M_PI_2);
            phi = ValueType(0.5 * atan2(mat[1][2], mat[1][1]));
            psi = phi;
        } else if (isApproxEqual(mat[2][0], ValueType(-1.0), eps)) {
            theta = ValueType(-M_PI_2);
            phi = ValueType(0.5 * atan2(mat[1][2], mat[1][1]));
            psi = -phi;
        } else {
            psi = ValueType(atan2(-mat[1][0],mat[0][0]));
            phi = ValueType(atan2(-mat[2][1],mat[2][2]));
            theta = ValueType(atan2(mat[2][0],
                sqrt( mat[2][1]*mat[2][1] +
                    mat[2][2]*mat[2][2])));
        }
        return V(phi, theta, psi);
    case ZXY_ROTATION:
        if (isApproxEqual(mat[1][2], ValueType(1.0), eps)) {
            theta = ValueType(M_PI_2);
            phi = ValueType(0.5 * atan2(mat[0][1], mat[0][0]));
            psi = phi;
        } else if (isApproxEqual(mat[1][2], ValueType(-1.0), eps)) {
            theta = ValueType(-M_PI/2);
            phi = ValueType(0.5 * atan2(mat[0][1],mat[2][1]));
            psi = -phi;
        } else {
            psi = ValueType(atan2(-mat[0][2], mat[2][2]));
            phi = ValueType(atan2(-mat[1][0], mat[1][1]));
            theta = ValueType(atan2(mat[1][2],
                        sqrt(mat[0][2] * mat[0][2] +
                                mat[2][2] * mat[2][2])));
        }
        return V(theta, psi, phi);

    case YZX_ROTATION:
        if (isApproxEqual(mat[0][1], ValueType(1.0), eps)) {
            theta = ValueType(M_PI_2);
            phi = ValueType(0.5 * atan2(mat[2][0], mat[2][2]));
            psi = phi;
        } else if (isApproxEqual(mat[0][1], ValueType(-1.0), eps)) {
            theta = ValueType(-M_PI/2);
            phi = ValueType(0.5 * atan2(mat[2][0], mat[1][0]));
            psi = -phi;
        } else {
            psi = ValueType(atan2(-mat[2][1], mat[1][1]));
            phi = ValueType(atan2(-mat[0][2], mat[0][0]));
            theta = ValueType(atan2(mat[0][1],
                sqrt(mat[0][0] * mat[0][0] +
                        mat[0][2] * mat[0][2])));
        }
        return V(psi, phi, theta);

    case XZX_ROTATION:

        if (isApproxEqual(mat[0][0], ValueType(1.0), eps)) {
            theta = ValueType(0.0);
            phi = ValueType(0.5 * atan2(mat[1][2], mat[1][1]));
            psi = phi;
        } else if (isApproxEqual(mat[0][0], ValueType(-1.0), eps)) {
            theta = ValueType(M_PI);
            psi = ValueType(0.5 * atan2(mat[2][1], -mat[1][1]));
            phi = - psi;
        } else {
            psi = ValueType(atan2(mat[2][0], -mat[1][0]));
            phi = ValueType(atan2(mat[0][2], mat[0][1]));
            theta = ValueType(atan2(sqrt(mat[0][1] * mat[0][1] +
                                mat[0][2] * mat[0][2]),
                            mat[0][0]));
        }
        return V(phi, psi, theta);

    case ZXZ_ROTATION:

        if (isApproxEqual(mat[2][2], ValueType(1.0), eps)) {
            theta = ValueType(0.0);
            phi = ValueType(0.5 * atan2(mat[0][1], mat[0][0]));
            psi = phi;
        } else if (isApproxEqual(mat[2][2], ValueType(-1.0), eps)) {
            theta = ValueType(M_PI);
            phi = ValueType(0.5 * atan2(mat[0][1], mat[0][0]));
            psi = -phi;
        } else {
            psi = ValueType(atan2(mat[0][2], mat[1][2]));
            phi = ValueType(atan2(mat[2][0], -mat[2][1]));
            theta = ValueType(atan2(sqrt(mat[0][2] * mat[0][2] +
                                mat[1][2] * mat[1][2]),
                            mat[2][2]));
        }
        return V(theta, psi, phi);

    case YXZ_ROTATION:

        if (isApproxEqual(mat[2][1], ValueType(1.0), eps)) {
            theta = ValueType(-M_PI_2);
            phi = ValueType(0.5 * atan2(-mat[1][0], mat[0][0]));
            psi = phi;
        } else if (isApproxEqual(mat[2][1], ValueType(-1.0), eps)) {
            theta = ValueType(M_PI_2);
            phi = ValueType(0.5 * atan2(mat[1][0], mat[0][0]));
            psi = -phi;
        } else {
            psi = ValueType(atan2(mat[0][1], mat[1][1]));
            phi = ValueType(atan2(mat[2][0], mat[2][2]));
            theta = ValueType(atan2(-mat[2][1],
                sqrt(mat[0][1] * mat[0][1] +
                        mat[1][1] * mat[1][1])));
        }
        return V(theta, phi, psi);

    case ZYX_ROTATION:

        if (isApproxEqual(mat[0][2], ValueType(1.0), eps)) {
            theta = ValueType(-M_PI_2);
            phi = ValueType(0.5 * atan2(-mat[1][0], mat[1][1]));
            psi = phi;
        } else if (isApproxEqual(mat[0][2], ValueType(-1.0), eps)) {
            theta = ValueType(M_PI_2);
            phi = ValueType(0.5 * atan2(mat[2][1], mat[2][0]));
            psi = -phi;
        } else {
            psi = ValueType(atan2(mat[1][2], mat[2][2]));
            phi = ValueType(atan2(mat[0][1], mat[0][0]));
            theta = ValueType(atan2(-mat[0][2],
                sqrt(mat[0][1] * mat[0][1] +
                        mat[0][0] * mat[0][0])));
        }
        return V(psi, theta, phi);

    case XZY_ROTATION:

        if (isApproxEqual(mat[1][0], ValueType(-1.0), eps)) {
            theta = ValueType(M_PI_2);
            psi = ValueType(0.5 * atan2(mat[2][1], mat[2][2]));
            phi = -psi;
        } else if (isApproxEqual(mat[1][0], ValueType(1.0), eps)) {
            theta = ValueType(-M_PI_2);
            psi = ValueType(0.5 * atan2(- mat[2][1], mat[2][2]));
            phi = psi;
        } else {
            psi = ValueType(atan2(mat[2][0], mat[0][0]));
            phi = ValueType(atan2(mat[1][2], mat[1][1]));
            theta = ValueType(atan2(- mat[1][0],
                            sqrt(mat[1][1] * mat[1][1] +
                                    mat[1][2] * mat[1][2])));
        }
        return V(phi, psi, theta);
    }

    OPENVDB_THROW(NotImplementedError, "Euler extraction sequence not implemented");
}


/// @brief Return a rotation matrix that maps @a v1 onto @a v2
/// about the cross product of @a v1 and @a v2.
/// <a name="rotation_v1_v2"></a>
template<typename MatType, typename ValueType1, typename ValueType2>
inline MatType
rotation(
    const Vec3<ValueType1>& _v1,
    const Vec3<ValueType2>& _v2,
    typename MatType::value_type eps = static_cast<typename MatType::value_type>(1.0e-8))
{
    using T = typename MatType::value_type;

    Vec3<T> v1(_v1);
    Vec3<T> v2(_v2);

    // Check if v1 and v2 are unit length
    if (!isApproxEqual(T(1), v1.dot(v1), eps)) {
        v1.normalize();
    }
    if (!isApproxEqual(T(1), v2.dot(v2), eps)) {
        v2.normalize();
    }

    Vec3<T> cross;
    cross.cross(v1, v2);

    if (isApproxEqual(cross[0], zeroVal<T>(), eps) &&
        isApproxEqual(cross[1], zeroVal<T>(), eps) &&
        isApproxEqual(cross[2], zeroVal<T>(), eps)) {


        // Given two unit vectors v1 and v2 that are nearly parallel, build a
        // rotation matrix that maps v1 onto v2. First find which principal axis
        // p is closest to perpendicular to v1. Find a reflection that exchanges
        // v1 and p, and find a reflection that exchanges p2 and v2. The desired
        // rotation matrix is the composition of these two reflections. See the
        // paper "Efficiently Building a Matrix to Rotate One Vector to
        // Another" by Tomas Moller and John Hughes in Journal of Graphics
        // Tools Vol 4, No 4 for details.

        Vec3<T> u, v, p(0.0, 0.0, 0.0);

        double x = Abs(v1[0]);
        double y = Abs(v1[1]);
        double z = Abs(v1[2]);

        if (x < y) {
            if (z < x) {
                p[2] = 1;
            } else {
                p[0] = 1;
            }
        } else {
            if (z < y) {
                p[2] = 1;
            } else {
                p[1] = 1;
            }
        }
        u = p - v1;
        v = p - v2;

        double udot = u.dot(u);
        double vdot = v.dot(v);

        double a = -2 / udot;
        double b = -2 / vdot;
        double c = 4 * u.dot(v) / (udot * vdot);

        MatType result;
        result.setIdentity();

        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 3; i++)
                result[i][j] = static_cast<T>(
                    a * u[i] * u[j] + b * v[i] * v[j] + c * v[j] * u[i]);
        }
        result[0][0] += 1.0;
        result[1][1] += 1.0;
        result[2][2] += 1.0;

        if(MatType::numColumns() == 4) padMat4(result);
        return result;

    } else {
        double c = v1.dot(v2);
        double a = (1.0 - c) / cross.dot(cross);

        double a0 = a * cross[0];
        double a1 = a * cross[1];
        double a2 = a * cross[2];

        double a01 = a0 * cross[1];
        double a02 = a0 * cross[2];
        double a12 = a1 * cross[2];

        MatType r;

        r[0][0] = static_cast<T>(c + a0 * cross[0]);
        r[0][1] = static_cast<T>(a01 + cross[2]);
        r[0][2] = static_cast<T>(a02 - cross[1]);
        r[1][0] = static_cast<T>(a01 - cross[2]);
        r[1][1] = static_cast<T>(c + a1 * cross[1]);
        r[1][2] = static_cast<T>(a12 + cross[0]);
        r[2][0] = static_cast<T>(a02 + cross[1]);
        r[2][1] = static_cast<T>(a12 - cross[0]);
        r[2][2] = static_cast<T>(c + a2 * cross[2]);

        if(MatType::numColumns() == 4) padMat4(r);
        return r;

    }
}


/// Return a matrix that scales by @a s.
template<class MatType>
MatType
scale(const Vec3<typename MatType::value_type>& s)
{
    // Gets identity, then sets top 3 diagonal
    // Inefficient by 3 sets.

    MatType result;
    result.setIdentity();
    result[0][0] = s[0];
    result[1][1] = s[1];
    result[2][2] = s[2];

    return result;
}


/// Return a Vec3 representing the lengths of the passed matrix's upper 3&times;3's rows.
template<class MatType>
Vec3<typename MatType::value_type>
getScale(const MatType &mat)
{
    using V = Vec3<typename MatType::value_type>;
    return V(
        V(mat[0][0], mat[0][1], mat[0][2]).length(),
        V(mat[1][0], mat[1][1], mat[1][2]).length(),
        V(mat[2][0], mat[2][1], mat[2][2]).length());
}


/// @brief Return a copy of the given matrix with its upper 3&times;3 rows normalized.
/// @details This can be geometrically interpreted as a matrix with no scaling
/// along its major axes.
template<class MatType>
MatType
unit(const MatType &mat, typename MatType::value_type eps = 1.0e-8)
{
    Vec3<typename MatType::value_type> dud;
    return unit(mat, eps, dud);
}


/// @brief Return a copy of the given matrix with its upper 3&times;3 rows normalized,
/// and return the length of each of these rows in @a scaling.
/// @details This can be geometrically interpretted as a matrix with no scaling
/// along its major axes, and the scaling in the input vector
template<class MatType>
MatType
unit(
    const MatType &in,
    typename MatType::value_type eps,
    Vec3<typename MatType::value_type>& scaling)
{
    using T = typename MatType::value_type;
    MatType result(in);

    for (int i(0); i < 3; i++) {
        try {
            const Vec3<T> u(
                Vec3<T>(in[i][0], in[i][1], in[i][2]).unit(eps, scaling[i]));
            for (int j=0; j<3; j++) result[i][j] = u[j];
        } catch (ArithmeticError&) {
            for (int j=0; j<3; j++) result[i][j] = 0;
        }
    }
    return result;
}


/// @brief Set the matrix to a shear along @a axis0 by a fraction of @a axis1.
/// @param axis0 The fixed axis of the shear.
/// @param axis1 The shear axis.
/// @param shear The shear factor.
template <class MatType>
MatType
shear(Axis axis0, Axis axis1, typename MatType::value_type shear)
{
    int index0 = static_cast<int>(axis0);
    int index1 = static_cast<int>(axis1);

    MatType result;
    result.setIdentity();
    if (axis0 == axis1) {
        result[index1][index0] = shear + 1;
    } else {
        result[index1][index0] = shear;
    }

    return result;
}


/// Return a matrix as the cross product of the given vector.
template<class MatType>
MatType
skew(const Vec3<typename MatType::value_type> &skew)
{
    using T = typename MatType::value_type;

    MatType r;
    r[0][0] = T(0);      r[0][1] = skew.z();  r[0][2] = -skew.y();
    r[1][0] = -skew.z(); r[1][1] = T(0);      r[2][1] = skew.x();
    r[2][0] = skew.y();  r[2][1] = -skew.x(); r[2][2] = T(0);

    if(MatType::numColumns() == 4) padMat4(r);
    return r;
}


/// @brief Return an orientation matrix such that z points along @a direction,
/// and y is along the @a direction / @a vertical plane.
template<class MatType>
MatType
aim(const Vec3<typename MatType::value_type>& direction,
    const Vec3<typename MatType::value_type>& vertical)
{
    using T = typename MatType::value_type;
    Vec3<T> forward(direction.unit());
    Vec3<T> horizontal(vertical.unit().cross(forward).unit());
    Vec3<T> up(forward.cross(horizontal).unit());

    MatType r;

    r[0][0]=horizontal.x(); r[0][1]=horizontal.y(); r[0][2]=horizontal.z();
    r[1][0]=up.x();         r[1][1]=up.y();         r[1][2]=up.z();
    r[2][0]=forward.x();    r[2][1]=forward.y();    r[2][2]=forward.z();

    if(MatType::numColumns() == 4) padMat4(r);
    return r;
}

/// @brief    This function snaps a specific axis to a specific direction,
///           preserving scaling.
/// @details  It does this using minimum energy, thus posing a unique solution if
///           basis & direction aren't parallel.
/// @note     @a direction need not be unit.
template<class MatType>
inline MatType
snapMatBasis(const MatType& source, Axis axis, const Vec3<typename MatType::value_type>& direction)
{
    using T = typename MatType::value_type;

    Vec3<T> unitDir(direction.unit());
    Vec3<T> ourUnitAxis(source.row(axis).unit());

    // Are the two parallel?
    T parallel = unitDir.dot(ourUnitAxis);

    // Already snapped!
    if (isApproxEqual(parallel, T(1.0))) return source;

    if (isApproxEqual(parallel, T(-1.0))) {
        OPENVDB_THROW(ValueError, "Cannot snap to inverse axis");
    }

    // Find angle between our basis and the one specified
    T angleBetween(angle(unitDir, ourUnitAxis));
    // Caclulate axis to rotate along
    Vec3<T> rotationAxis = unitDir.cross(ourUnitAxis);

    MatType rotation;
    rotation.setToRotation(rotationAxis, angleBetween);

    return source * rotation;
}

/// @brief Write 0s along Mat4's last row and column, and a 1 on its diagonal.
/// @details Useful initialization when we're initializing just the 3&times;3 block.
template<class MatType>
static MatType&
padMat4(MatType& dest)
{
    dest[0][3] = dest[1][3] = dest[2][3] = 0;
    dest[3][2] = dest[3][1] = dest[3][0] = 0;
    dest[3][3] = 1;

    return dest;
}


/// @brief Solve for A=B*B, given A.
/// @details Denman-Beavers square root iteration
template<typename MatType>
inline void
sqrtSolve(const MatType& aA, MatType& aB, double aTol=0.01)
{
    unsigned int iterations = static_cast<unsigned int>(log(aTol)/log(0.5));

    MatType Y[2], Z[2];
    Y[0] = aA;
    Z[0] = MatType::identity();

    unsigned int current = 0;
    for (unsigned int iteration=0; iteration < iterations; iteration++) {
        unsigned int last = current;
        current = !current;

        MatType invY = Y[last].inverse();
        MatType invZ = Z[last].inverse();

        Y[current] = 0.5 * (Y[last] + invZ);
        Z[current] = 0.5 * (Z[last] + invY);
    }
    aB = Y[current];
}


template<typename MatType>
inline void
powSolve(const MatType& aA, MatType& aB, double aPower, double aTol=0.01)
{
    unsigned int iterations = static_cast<unsigned int>(log(aTol)/log(0.5));

    const bool inverted = (aPower < 0.0);
    if (inverted) { aPower = -aPower; }

    unsigned int whole = static_cast<unsigned int>(aPower);
    double fraction = aPower - whole;

    MatType R = MatType::identity();
    MatType partial = aA;

    double contribution = 1.0;
    for (unsigned int iteration = 0; iteration < iterations; iteration++) {
        sqrtSolve(partial, partial, aTol);
        contribution *= 0.5;
        if (fraction >= contribution) {
            R *= partial;
            fraction -= contribution;
        }
    }

    partial = aA;
    while (whole) {
        if (whole & 1) { R *= partial; }
        whole >>= 1;
        if (whole) { partial *= partial; }
    }

    if (inverted) { aB = R.inverse(); }
    else { aB = R; }
}


/// @brief Determine if a matrix is an identity matrix.
template<typename MatType>
inline bool
isIdentity(const MatType& m)
{
    return m.eq(MatType::identity());
}


/// @brief Determine if a matrix is invertible.
template<typename MatType>
inline bool
isInvertible(const MatType& m)
{
    using ValueType = typename MatType::ValueType;
    return !isApproxEqual(m.det(), ValueType(0));
}


/// @brief Determine if a matrix is symmetric.
/// @details This implicitly uses math::isApproxEqual() to determine equality.
template<typename MatType>
inline bool
isSymmetric(const MatType& m)
{
    return m.eq(m.transpose());
}


/// Determine if a matrix is unitary (i.e., rotation or reflection).
template<typename MatType>
inline bool
isUnitary(const MatType& m)
{
    using ValueType = typename MatType::ValueType;
    if (!isApproxEqual(std::abs(m.det()), ValueType(1.0))) return false;
    // check that the matrix transpose is the inverse
    MatType temp = m * m.transpose();
    return temp.eq(MatType::identity());
}


/// Determine if a matrix is diagonal.
template<typename MatType>
inline bool
isDiagonal(const MatType& mat)
{
    int n = MatType::size;
    typename MatType::ValueType temp(0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                temp += std::abs(mat(i,j));
            }
        }
    }
    return isApproxEqual(temp, typename MatType::ValueType(0.0));
}


/// Return the <i>L</i><sub>&infin;</sub> norm of an <i>N</i>&times;<i>N</i> matrix.
template<typename MatType>
typename MatType::ValueType
lInfinityNorm(const MatType& matrix)
{
    int n = MatType::size;
    typename MatType::ValueType norm = 0;

    for( int j = 0; j<n; ++j) {
        typename MatType::ValueType column_sum = 0;

        for (int i = 0; i<n; ++i) {
            column_sum += fabs(matrix(i,j));
        }
        norm = std::max(norm, column_sum);
    }

    return norm;
}


/// Return the <i>L</i><sub>1</sub> norm of an <i>N</i>&times;<i>N</i> matrix.
template<typename MatType>
typename MatType::ValueType
lOneNorm(const MatType& matrix)
{
    int n = MatType::size;
    typename MatType::ValueType norm = 0;

    for( int i = 0; i<n; ++i) {
        typename MatType::ValueType row_sum = 0;

        for (int j = 0; j<n; ++j) {
            row_sum += fabs(matrix(i,j));
        }
        norm = std::max(norm, row_sum);
    }

    return norm;
}


/// @brief Decompose an invertible 3&times;3 matrix into a unitary matrix
/// followed by a symmetric matrix (positive semi-definite Hermitian),
/// i.e., M = U * S.
/// @details If det(U) = 1 it is a rotation, otherwise det(U) = -1,
/// meaning there is some part reflection.
/// See "Computing the polar decomposition with applications"
/// Higham, N.J. - SIAM J. Sc. Stat Comput 7(4):1160-1174
template<typename MatType>
bool
polarDecomposition(const MatType& input, MatType& unitary,
    MatType& positive_hermitian, unsigned int MAX_ITERATIONS=100)
{
    unitary = input;
    MatType new_unitary(input);
    MatType unitary_inv;

    if (fabs(unitary.det()) < math::Tolerance<typename MatType::ValueType>::value()) return false;

    unsigned int iteration(0);

    typename MatType::ValueType linf_of_u;
    typename MatType::ValueType l1nm_of_u;
    typename MatType::ValueType linf_of_u_inv;
    typename MatType::ValueType l1nm_of_u_inv;
    typename MatType::ValueType l1_error = 100;
    double gamma;

    do {
        unitary_inv = unitary.inverse();
        linf_of_u = lInfinityNorm(unitary);
        l1nm_of_u = lOneNorm(unitary);

        linf_of_u_inv = lInfinityNorm(unitary_inv);
        l1nm_of_u_inv = lOneNorm(unitary_inv);

        gamma = sqrt( sqrt( (l1nm_of_u_inv * linf_of_u_inv ) / (l1nm_of_u * linf_of_u) ));

        new_unitary = 0.5*(gamma * unitary + (1./gamma) * unitary_inv.transpose() );

        l1_error = lInfinityNorm(unitary - new_unitary);
        unitary = new_unitary;

        /// this generally converges in less than ten iterations
        if (iteration > MAX_ITERATIONS) return false;
        iteration++;
    } while (l1_error > math::Tolerance<typename MatType::ValueType>::value());

    positive_hermitian = unitary.transpose() * input;
    return true;
}

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_MAT_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
