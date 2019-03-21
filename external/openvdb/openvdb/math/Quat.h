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

#ifndef OPENVDB_MATH_QUAT_H_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_QUAT_H_HAS_BEEN_INCLUDED

#include "Mat.h"
#include "Mat3.h"
#include "Math.h"
#include "Vec3.h"
#include <openvdb/Exceptions.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

template<typename T> class Quat;

/// Linear interpolation between the two quaternions
template <typename T>
Quat<T> slerp(const Quat<T> &q1, const Quat<T> &q2, T t, T tolerance=0.00001)
{
    T qdot, angle, sineAngle;

    qdot = q1.dot(q2);

    if (fabs(qdot) >= 1.0) {
        angle     = 0; // not necessary but suppresses compiler warning
        sineAngle = 0;
    } else {
        angle     = acos(qdot);
        sineAngle = sin(angle);
    }

    //
    // Denominator close to 0 corresponds to the case where the
    // two quaternions are close to the same rotation. In this
    // case linear interpolation is used but we normalize to
    // guarantee unit length
    //
    if (sineAngle <= tolerance) {
        T s = 1.0 - t;

        Quat<T> qtemp(s * q1[0] + t * q2[0], s * q1[1] + t * q2[1],
                      s * q1[2] + t * q2[2], s * q1[3] + t * q2[3]);
        //
        // Check the case where two close to antipodal quaternions were
        // blended resulting in a nearly zero result which can happen,
        // for example, if t is close to 0.5. In this case it is not safe
        // to project back onto the sphere.
        //
        double lengthSquared = qtemp.dot(qtemp);

        if (lengthSquared <= tolerance * tolerance) {
            qtemp = (t < 0.5) ? q1 : q2;
        } else {
            qtemp *= 1.0 / sqrt(lengthSquared);
        }
        return qtemp;
    } else {

        T sine  = 1.0 / sineAngle;
        T a = sin((1.0 - t) * angle) * sine;
        T b = sin(t * angle) * sine;
        return Quat<T>(a * q1[0] + b * q2[0], a * q1[1] + b * q2[1],
                       a * q1[2] + b * q2[2], a * q1[3] + b * q2[3]);
    }

}

template<typename T>
class Quat
{
public:
    /// Trivial constructor, the quaternion is NOT initialized
    Quat() {}

    /// Constructor with four arguments, e.g.   Quatf q(1,2,3,4);
    Quat(T x, T y, T z, T w)
    {
        mm[0] = x;
        mm[1] = y;
        mm[2] = z;
        mm[3] = w;

    }

    /// Constructor with array argument, e.g.   float a[4]; Quatf q(a);
    Quat(T *a)
    {
        mm[0] = a[0];
        mm[1] = a[1];
        mm[2] = a[2];
        mm[3] = a[3];

    }

    /// Constructor given rotation as axis and angle, the axis must be
    /// unit vector
    Quat(const Vec3<T> &axis, T angle)
    {
        // assert( REL_EQ(axis.length(), 1.) );

        T s = T(sin(angle*T(0.5)));

        mm[0] = axis.x() * s;
        mm[1] = axis.y() * s;
        mm[2] = axis.z() * s;

        mm[3] = T(cos(angle*T(0.5)));

    }

    /// Constructor given rotation as axis and angle
    Quat(math::Axis axis, T angle)
    {
        T s = T(sin(angle*T(0.5)));

        mm[0] = (axis==math::X_AXIS) * s;
        mm[1] = (axis==math::Y_AXIS) * s;
        mm[2] = (axis==math::Z_AXIS) * s;

        mm[3] = T(cos(angle*T(0.5)));
    }

    /// Constructor given a rotation matrix
    template<typename T1>
    Quat(const Mat3<T1> &rot) {

        // verify that the matrix is really a rotation
        if(!isUnitary(rot)) {  // unitary is reflection or rotation
             OPENVDB_THROW(ArithmeticError,
                "A non-rotation matrix can not be used to construct a quaternion");
        }
        if (!isApproxEqual(rot.det(), T1(1))) { // rule out reflection
             OPENVDB_THROW(ArithmeticError,
                "A reflection matrix can not be used to construct a quaternion");
        }

        T trace(rot.trace());
        if (trace > 0) {

            T q_w = 0.5 * std::sqrt(trace+1);
            T factor = 0.25 / q_w;

            mm[0] = factor * (rot(1,2) - rot(2,1));
            mm[1] = factor * (rot(2,0) - rot(0,2));
            mm[2] = factor * (rot(0,1) - rot(1,0));
            mm[3] = q_w;
        }  else if (rot(0,0) > rot(1,1) && rot(0,0) > rot(2,2)) {

            T q_x = 0.5 * sqrt(rot(0,0)- rot(1,1)-rot(2,2)+1);
            T factor = 0.25 / q_x;

            mm[0] = q_x;
            mm[1] = factor * (rot(0,1) + rot(1,0));
            mm[2] = factor * (rot(2,0) + rot(0,2));
            mm[3] = factor * (rot(1,2) - rot(2,1));
        } else if (rot(1,1) > rot(2,2)) {

            T q_y = 0.5 * sqrt(rot(1,1)-rot(0,0)-rot(2,2)+1);
            T factor = 0.25 / q_y;

            mm[0] =  factor * (rot(0,1) + rot(1,0));
            mm[1] = q_y;
            mm[2] = factor * (rot(1,2) + rot(2,1));
            mm[3] = factor * (rot(2,0) - rot(0,2));
        } else {

            T q_z = 0.5 * sqrt(rot(2,2)-rot(0,0)-rot(1,1)+1);
            T factor = 0.25 / q_z;

            mm[0] = factor * (rot(2,0) + rot(0,2));
            mm[1] = factor * (rot(1,2) + rot(2,1));
            mm[2] = q_z;
            mm[3] = factor * (rot(0,1) - rot(1,0));
        }
    }

    /// Copy constructor
    Quat(const Quat &q)
    {
        mm[0] = q.mm[0];
        mm[1] = q.mm[1];
        mm[2] = q.mm[2];
        mm[3] = q.mm[3];

    }

    /// Reference to the component, e.g.   q.x() = 4.5f;
    T& x() { return mm[0]; }
    T& y() { return mm[1]; }
    T& z() { return mm[2]; }
    T& w() { return mm[3]; }

    /// Get the component, e.g.   float f = q.w();
    T x() const { return mm[0]; }
    T y() const { return mm[1]; }
    T z() const { return mm[2]; }
    T w() const { return mm[3]; }

    // Number of elements
    static unsigned numElements() { return 4; }

    /// Array style reference to the components, e.g.   q[3] = 1.34f;
    T& operator[](int i) { return mm[i]; }

    /// Array style constant reference to the components, e.g.  float f = q[1];
    T operator[](int i) const { return mm[i]; }

    /// Cast to T*
    operator T*() { return mm; }
    operator const T*() const { return mm; }

    /// Alternative indexed reference to the elements
    T& operator()(int i) { return mm[i]; }

    /// Alternative indexed constant reference to the elements,
    T operator()(int i) const { return mm[i]; }

    /// Return angle of rotation
    T angle() const
    {
        T sqrLength = mm[0]*mm[0] + mm[1]*mm[1] + mm[2]*mm[2];

        if ( sqrLength > 1.0e-8 ) {

            return T(T(2.0) * acos(mm[3]));

        } else {

            return T(0.0);
        }
    }

    /// Return axis of rotation
    Vec3<T> axis() const
    {
        T sqrLength = mm[0]*mm[0] + mm[1]*mm[1] + mm[2]*mm[2];

        if ( sqrLength > 1.0e-8 ) {

            T invLength = T(T(1)/sqrt(sqrLength));

            return Vec3<T>( mm[0]*invLength, mm[1]*invLength, mm[2]*invLength );
        } else {

            return Vec3<T>(1,0,0);
        }
    }


    /// "this" quaternion gets initialized to [x, y, z, w]
    Quat& init(T x, T y, T z, T w)
    {
        mm[0] = x; mm[1] = y; mm[2] = z; mm[3] = w;
        return *this;
    }

    /// "this" quaternion gets initialized to identity, same as setIdentity()
    Quat& init() { return setIdentity(); }

    /// Set "this" quaternion to rotation specified by axis and angle,
    /// the axis must be unit vector
    Quat& setAxisAngle(const Vec3<T>& axis, T angle)
    {

        T s = T(sin(angle*T(0.5)));

        mm[0] = axis.x() * s;
        mm[1] = axis.y() * s;
        mm[2] = axis.z() * s;

        mm[3] = T(cos(angle*T(0.5)));

        return *this;
    } // axisAngleTest

    /// Set "this" vector to zero
    Quat& setZero()
    {
        mm[0] = mm[1] = mm[2] = mm[3] = 0;
        return *this;
    }

    /// Set "this" vector to identity
    Quat& setIdentity()
    {
        mm[0] = mm[1] = mm[2] = 0;
        mm[3] = 1;
        return *this;
    }

    /// Returns vector of x,y,z rotational components
    Vec3<T> eulerAngles(RotationOrder rotationOrder) const
    { return math::eulerAngles(Mat3<T>(*this), rotationOrder); }

    /// Assignment operator
    Quat& operator=(const Quat &q)
    {
        mm[0] = q.mm[0];
        mm[1] = q.mm[1];
        mm[2] = q.mm[2];
        mm[3] = q.mm[3];

        return *this;
    }

    /// Equality operator, does exact floating point comparisons
    bool operator==(const Quat &q) const
    {
        return (isExactlyEqual(mm[0],q.mm[0]) &&
                isExactlyEqual(mm[1],q.mm[1]) &&
                isExactlyEqual(mm[2],q.mm[2]) &&
                isExactlyEqual(mm[3],q.mm[3]) );
    }

    /// Test if "this" is equivalent to q with tolerance of eps value
    bool eq(const Quat &q, T eps=1.0e-7) const
    {
        return isApproxEqual(mm[0],q.mm[0],eps) && isApproxEqual(mm[1],q.mm[1],eps) &&
            isApproxEqual(mm[2],q.mm[2],eps) && isApproxEqual(mm[3],q.mm[3],eps) ;
    } // trivial

    /// Add quaternion q to "this" quaternion, e.g.   q += q1;
    Quat& operator+=(const Quat &q)
    {
        mm[0] += q.mm[0];
        mm[1] += q.mm[1];
        mm[2] += q.mm[2];
        mm[3] += q.mm[3];

        return *this;
    }

    /// Subtract quaternion q from "this" quaternion, e.g.   q -= q1;
    Quat& operator-=(const Quat &q)
    {
        mm[0] -= q.mm[0];
        mm[1] -= q.mm[1];
        mm[2] -= q.mm[2];
        mm[3] -= q.mm[3];

        return *this;
    }

    /// Scale "this" quaternion by scalar, e.g.   q *= scalar;
    Quat& operator*=(T scalar)
    {
        mm[0] *= scalar;
        mm[1] *= scalar;
        mm[2] *= scalar;
        mm[3] *= scalar;

        return *this;
    }

    /// Return (this+q), e.g.   q = q1 + q2;
    Quat operator+(const Quat &q) const
    {
        return Quat<T>(mm[0]+q.mm[0], mm[1]+q.mm[1], mm[2]+q.mm[2], mm[3]+q.mm[3]);
    }

    /// Return (this-q), e.g.   q = q1 - q2;
    Quat operator-(const Quat &q) const
    {
        return Quat<T>(mm[0]-q.mm[0], mm[1]-q.mm[1], mm[2]-q.mm[2], mm[3]-q.mm[3]);
    }

    /// Return (this*q), e.g.   q = q1 * q2;
    Quat operator*(const Quat &q) const
    {
        Quat<T> prod;

        prod.mm[0] = mm[3]*q.mm[0] + mm[0]*q.mm[3] + mm[1]*q.mm[2] - mm[2]*q.mm[1];
        prod.mm[1] = mm[3]*q.mm[1] + mm[1]*q.mm[3] + mm[2]*q.mm[0] - mm[0]*q.mm[2];
        prod.mm[2] = mm[3]*q.mm[2] + mm[2]*q.mm[3] + mm[0]*q.mm[1] - mm[1]*q.mm[0];
        prod.mm[3] = mm[3]*q.mm[3] - mm[0]*q.mm[0] - mm[1]*q.mm[1] - mm[2]*q.mm[2];

        return prod;

    }

    /// Assigns this to (this*q), e.g.   q *= q1;
    Quat operator*=(const Quat &q)
    {
        *this = *this * q;
        return *this;
    }

    /// Return (this*scalar), e.g.   q = q1 * scalar;
    Quat operator*(T scalar) const
    {
        return Quat<T>(mm[0]*scalar, mm[1]*scalar, mm[2]*scalar, mm[3]*scalar);
    }

    /// Return (this/scalar), e.g.   q = q1 / scalar;
    Quat operator/(T scalar) const
    {
        return Quat<T>(mm[0]/scalar, mm[1]/scalar, mm[2]/scalar, mm[3]/scalar);
    }

    /// Negation operator, e.g.   q = -q;
    Quat operator-() const
    { return Quat<T>(-mm[0], -mm[1], -mm[2], -mm[3]); }

    /// this = q1 + q2
    /// "this", q1 and q2 need not be distinct objects, e.g. q.add(q1,q);
    Quat& add(const Quat &q1, const Quat &q2)
    {
        mm[0] = q1.mm[0] + q2.mm[0];
        mm[1] = q1.mm[1] + q2.mm[1];
        mm[2] = q1.mm[2] + q2.mm[2];
        mm[3] = q1.mm[3] + q2.mm[3];

        return *this;
    }

    /// this = q1 - q2
    /// "this", q1 and q2 need not be distinct objects, e.g. q.sub(q1,q);
    Quat& sub(const Quat &q1, const Quat &q2)
    {
        mm[0] = q1.mm[0] - q2.mm[0];
        mm[1] = q1.mm[1] - q2.mm[1];
        mm[2] = q1.mm[2] - q2.mm[2];
        mm[3] = q1.mm[3] - q2.mm[3];

        return *this;
    }

    /// this = q1 * q2
    /// q1 and q2 must be distinct objects than "this", e.g.  q.mult(q1,q2);
    Quat& mult(const Quat &q1, const Quat &q2)
    {
        mm[0] = q1.mm[3]*q2.mm[0] + q1.mm[0]*q2.mm[3] +
                q1.mm[1]*q2.mm[2] - q1.mm[2]*q2.mm[1];
        mm[1] = q1.mm[3]*q2.mm[1] + q1.mm[1]*q2.mm[3] +
                q1.mm[2]*q2.mm[0] - q1.mm[0]*q2.mm[2];
        mm[2] = q1.mm[3]*q2.mm[2] + q1.mm[2]*q2.mm[3] +
                q1.mm[0]*q2.mm[1] - q1.mm[1]*q2.mm[0];
        mm[3] = q1.mm[3]*q2.mm[3] - q1.mm[0]*q2.mm[0] -
                q1.mm[1]*q2.mm[1] - q1.mm[2]*q2.mm[2];

        return *this;
    }

    /// this =  scalar*q, q need not be distinct object than "this",
    /// e.g. q.scale(1.5,q1);
    Quat& scale(T scale, const Quat &q)
    {
        mm[0] = scale * q.mm[0];
        mm[1] = scale * q.mm[1];
        mm[2] = scale * q.mm[2];
        mm[3] = scale * q.mm[3];

        return *this;
    }

    /// Dot product
    T dot(const Quat &q) const
    {
        return (mm[0]*q.mm[0] + mm[1]*q.mm[1] + mm[2]*q.mm[2] + mm[3]*q.mm[3]);
    }

    /// Return the quaternion rate corrsponding to the angular velocity omega
    /// and "this" current rotation
    Quat derivative(const Vec3<T>& omega) const
    {
        return Quat<T>( +w()*omega.x() -z()*omega.y() +y()*omega.z() ,
                        +z()*omega.x() +w()*omega.y() -x()*omega.z() ,
                        -y()*omega.x() +x()*omega.y() +w()*omega.z() ,
                        -x()*omega.x() -y()*omega.y() -z()*omega.z() );
    }

    /// this = normalized this
    bool normalize(T eps = T(1.0e-8))
    {
        T d = T(sqrt(mm[0]*mm[0] + mm[1]*mm[1] + mm[2]*mm[2] + mm[3]*mm[3]));
        if( isApproxEqual(d, T(0.0), eps) ) return false;
        *this *= ( T(1)/d );
        return true;
    }

    /// this = normalized this
    Quat unit() const
    {
        T d = sqrt(mm[0]*mm[0] + mm[1]*mm[1] + mm[2]*mm[2] + mm[3]*mm[3]);
        if( isExactlyEqual(d , T(0.0) ) )
            OPENVDB_THROW(ArithmeticError,
                "Normalizing degenerate quaternion");
        return *this / d;
    }

    /// returns inverse of this
    Quat inverse(T tolerance = T(0))
    {
        T d = mm[0]*mm[0] + mm[1]*mm[1] + mm[2]*mm[2] + mm[3]*mm[3];
        if( isApproxEqual(d, T(0.0), tolerance) )
            OPENVDB_THROW(ArithmeticError,
                "Cannot invert degenerate quaternion");
        Quat result = *this/-d;
        result.mm[3] = -result.mm[3];
        return result;
    }


    /// Return the conjugate of "this", same as invert without
    /// unit quaternion test
    Quat conjugate() const
    {
        return Quat<T>(-mm[0], -mm[1], -mm[2], mm[3]);
    }

    /// Return rotated vector by "this" quaternion
    Vec3<T> rotateVector(const Vec3<T> &v) const
    {
        Mat3<T> m(*this);
        return m.transform(v);
    }

    /// Predefined constants, e.g.   Quat q = Quat::identity();
    static Quat zero() { return Quat<T>(0,0,0,0); }
    static Quat identity() { return Quat<T>(0,0,0,1); }

     /// @return string representation of Classname
    std::string str() const
    {
        std::ostringstream buffer;

        buffer << "[";

        // For each column
        for (unsigned j(0); j < 4; j++) {
            if (j) buffer << ", ";
            buffer << mm[j];
        }

        buffer << "]";

        return buffer.str();
    }

    /// Output to the stream, e.g.   std::cout << q << std::endl;
    friend std::ostream& operator<<(std::ostream &stream, const Quat &q)
    {
        stream << q.str();
        return stream;
    }

    friend Quat slerp<>(const Quat &q1, const Quat &q2, T t, T tolerance);

    void write(std::ostream& os) const { os.write(static_cast<char*>(&mm), sizeof(T) * 4); }
    void read(std::istream& is) { is.read(static_cast<char*>(&mm), sizeof(T) * 4); }

protected:
    T mm[4];
};

/// Multiply each element of the given quaternion by @a scalar and return the result.
template <typename S, typename T>
Quat<T> operator*(S scalar, const Quat<T> &q) { return q*scalar; }


/// @brief Interpolate between m1 and m2.
/// Converts to quaternion  form and uses slerp
/// m1 and m2 must be rotation matrices!
template <typename T, typename T0>
Mat3<T> slerp(const Mat3<T0> &m1, const Mat3<T0> &m2, T t)
{
    using MatType = Mat3<T>;

    Quat<T> q1(m1);
    Quat<T> q2(m2);

    if (q1.dot(q2) < 0) q2 *= -1;

    Quat<T> qslerp = slerp<T>(q1, q2, static_cast<T>(t));
    MatType m = rotation<MatType>(qslerp);
    return m;
}



/// Interpolate between m1 and m4 by converting m1 ... m4  into
/// quaternions and treating them as control points of a Bezier
/// curve using slerp in place of lerp in the De Castlejeau evaluation
/// algorithm. Just like a cubic Bezier curve, this will interpolate
/// m1 at t = 0 and m4 at t = 1 but in general will not pass through
/// m2 and m3.  Unlike a standard Bezier curve this curve will not have
/// the convex hull property.
/// m1 ... m4 must be rotation matrices!
template <typename T, typename T0>
Mat3<T> bezLerp(const Mat3<T0> &m1, const Mat3<T0> &m2,
                const Mat3<T0> &m3, const Mat3<T0> &m4,
                T t)
{
    Mat3<T> m00, m01, m02, m10, m11;

    m00 = slerp(m1, m2, t);
    m01 = slerp(m2, m3, t);
    m02 = slerp(m3, m4, t);

    m10 = slerp(m00, m01, t);
    m11 = slerp(m01, m02, t);

    return slerp(m10, m11, t);
}

using Quats = Quat<float>;
using Quatd = Quat<double>;

} // namespace math


template<> inline math::Quats zeroVal<math::Quats >() { return math::Quats::zero(); }
template<> inline math::Quatd zeroVal<math::Quatd >() { return math::Quatd::zero(); }

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif //OPENVDB_MATH_QUAT_H_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
