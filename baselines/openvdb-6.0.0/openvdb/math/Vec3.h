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

#ifndef OPENVDB_MATH_VEC3_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_VEC3_HAS_BEEN_INCLUDED

#include <openvdb/Exceptions.h>
#include "Math.h"
#include "Tuple.h"
#include <algorithm>
#include <cmath>
#include <type_traits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

template<typename T> class Mat3;

template<typename T>
class Vec3: public Tuple<3, T>
{
public:
    using value_type = T;
    using ValueType = T;

    /// Trivial constructor, the vector is NOT initialized
    Vec3() {}

    /// @brief Construct a vector all of whose components have the given value.
    explicit Vec3(T val) { this->mm[0] = this->mm[1] = this->mm[2] = val; }

    /// Constructor with three arguments, e.g.   Vec3d v(1,2,3);
    Vec3(T x, T y, T z)
    {
        this->mm[0] = x;
        this->mm[1] = y;
        this->mm[2] = z;
    }

    /// Constructor with array argument, e.g.   double a[3]; Vec3d v(a);
    template <typename Source>
    Vec3(Source *a)
    {
        this->mm[0] = a[0];
        this->mm[1] = a[1];
        this->mm[2] = a[2];
    }

    /// @brief Construct a Vec3 from a 3-Tuple with a possibly different value type.
    /// @details Type conversion warnings are suppressed.
    template<typename Source>
    explicit Vec3(const Tuple<3, Source> &v)
    {
        this->mm[0] = static_cast<T>(v[0]);
        this->mm[1] = static_cast<T>(v[1]);
        this->mm[2] = static_cast<T>(v[2]);
    }

    /// @brief Construct a vector all of whose components have the given value,
    /// which may be of an arithmetic type different from this vector's value type.
    /// @details Type conversion warnings are suppressed.
    template<typename Other>
    explicit Vec3(Other val,
        typename std::enable_if<std::is_arithmetic<Other>::value, Conversion>::type = Conversion{})
    {
        this->mm[0] = this->mm[1] = this->mm[2] = static_cast<T>(val);
    }

    /// @brief Construct a Vec3 from another Vec3 with a possibly different value type.
    /// @details Type conversion warnings are suppressed.
    template<typename Other>
    Vec3(const Vec3<Other>& v)
    {
        this->mm[0] = static_cast<T>(v[0]);
        this->mm[1] = static_cast<T>(v[1]);
        this->mm[2] = static_cast<T>(v[2]);
    }

    /// Reference to the component, e.g.   v.x() = 4.5f;
    T& x() { return this->mm[0]; }
    T& y() { return this->mm[1]; }
    T& z() { return this->mm[2]; }

    /// Get the component, e.g.   float f = v.y();
    T x() const { return this->mm[0]; }
    T y() const { return this->mm[1]; }
    T z() const { return this->mm[2]; }

    T* asPointer() { return this->mm; }
    const T* asPointer() const { return this->mm; }

    /// Alternative indexed reference to the elements
    T& operator()(int i) { return this->mm[i]; }

    /// Alternative indexed constant reference to the elements,
    T operator()(int i) const { return this->mm[i]; }

    /// "this" vector gets initialized to [x, y, z],
    /// calling v.init(); has same effect as calling v = Vec3::zero();
    const Vec3<T>& init(T x=0, T y=0, T z=0)
    {
        this->mm[0] = x; this->mm[1] = y; this->mm[2] = z;
        return *this;
    }


    /// Set "this" vector to zero
    const Vec3<T>& setZero()
    {
        this->mm[0] = 0; this->mm[1] = 0; this->mm[2] = 0;
        return *this;
    }

    /// @brief Assignment operator
    /// @details Type conversion warnings are not suppressed.
    template<typename Source>
    const Vec3<T>& operator=(const Vec3<Source> &v)
    {
        // note: don't static_cast because that suppresses warnings
        this->mm[0] = v[0];
        this->mm[1] = v[1];
        this->mm[2] = v[2];

        return *this;
    }

    /// Test if "this" vector is equivalent to vector v with tolerance of eps
    bool eq(const Vec3<T> &v, T eps = static_cast<T>(1.0e-7)) const
    {
        return isRelOrApproxEqual(this->mm[0], v.mm[0], eps, eps) &&
               isRelOrApproxEqual(this->mm[1], v.mm[1], eps, eps) &&
               isRelOrApproxEqual(this->mm[2], v.mm[2], eps, eps);
    }


    /// Negation operator, for e.g.   v1 = -v2;
    Vec3<T> operator-() const { return Vec3<T>(-this->mm[0], -this->mm[1], -this->mm[2]); }

    /// this = v1 + v2
    /// "this", v1 and v2 need not be distinct objects, e.g. v.add(v1,v);
    template <typename T0, typename T1>
    const Vec3<T>& add(const Vec3<T0> &v1, const Vec3<T1> &v2)
    {
        this->mm[0] = v1[0] + v2[0];
        this->mm[1] = v1[1] + v2[1];
        this->mm[2] = v1[2] + v2[2];

        return *this;
    }

    /// this = v1 - v2
    /// "this", v1 and v2 need not be distinct objects, e.g. v.sub(v1,v);
    template <typename T0, typename T1>
    const Vec3<T>& sub(const Vec3<T0> &v1, const Vec3<T1> &v2)
    {
        this->mm[0] = v1[0] - v2[0];
        this->mm[1] = v1[1] - v2[1];
        this->mm[2] = v1[2] - v2[2];

        return *this;
    }

    /// this =  scalar*v, v need not be a distinct object from "this",
    /// e.g. v.scale(1.5,v1);
    template <typename T0, typename T1>
    const Vec3<T>& scale(T0 scale, const Vec3<T1> &v)
    {
        this->mm[0] = scale * v[0];
        this->mm[1] = scale * v[1];
        this->mm[2] = scale * v[2];

        return *this;
    }

    template <typename T0, typename T1>
    const Vec3<T> &div(T0 scale, const Vec3<T1> &v)
    {
        this->mm[0] = v[0] / scale;
        this->mm[1] = v[1] / scale;
        this->mm[2] = v[2] / scale;

        return *this;
    }

    /// Dot product
    T dot(const Vec3<T> &v) const
    {
        return
            this->mm[0]*v.mm[0] +
            this->mm[1]*v.mm[1] +
            this->mm[2]*v.mm[2];
    }

    /// Length of the vector
    T length() const
    {
        return static_cast<T>(sqrt(double(
            this->mm[0]*this->mm[0] +
            this->mm[1]*this->mm[1] +
            this->mm[2]*this->mm[2])));
    }


    /// Squared length of the vector, much faster than length() as it
    /// does not involve square root
    T lengthSqr() const
    {
        return
            this->mm[0]*this->mm[0] +
            this->mm[1]*this->mm[1] +
            this->mm[2]*this->mm[2];
    }

    /// Return the cross product of "this" vector and v;
    Vec3<T> cross(const Vec3<T> &v) const
    {
        return Vec3<T>(this->mm[1]*v.mm[2] - this->mm[2]*v.mm[1],
                    this->mm[2]*v.mm[0] - this->mm[0]*v.mm[2],
                    this->mm[0]*v.mm[1] - this->mm[1]*v.mm[0]);
    }


    /// this = v1 cross v2, v1 and v2 must be distinct objects than "this"
    const Vec3<T>& cross(const Vec3<T> &v1, const Vec3<T> &v2)
    {
        // assert(this!=&v1);
        // assert(this!=&v2);
        this->mm[0] = v1.mm[1]*v2.mm[2] - v1.mm[2]*v2.mm[1];
        this->mm[1] = v1.mm[2]*v2.mm[0] - v1.mm[0]*v2.mm[2];
        this->mm[2] = v1.mm[0]*v2.mm[1] - v1.mm[1]*v2.mm[0];
        return *this;
    }

    /// Multiply each element of this vector by @a scalar.
    template <typename S>
    const Vec3<T> &operator*=(S scalar)
    {
        this->mm[0] = static_cast<T>(this->mm[0] * scalar);
        this->mm[1] = static_cast<T>(this->mm[1] * scalar);
        this->mm[2] = static_cast<T>(this->mm[2] * scalar);
        return *this;
    }

    /// Multiply each element of this vector by the corresponding element of the given vector.
    template <typename S>
    const Vec3<T> &operator*=(const Vec3<S> &v1)
    {
        this->mm[0] *= v1[0];
        this->mm[1] *= v1[1];
        this->mm[2] *= v1[2];
        return *this;
    }

    /// Divide each element of this vector by @a scalar.
    template <typename S>
    const Vec3<T> &operator/=(S scalar)
    {
        this->mm[0] /= scalar;
        this->mm[1] /= scalar;
        this->mm[2] /= scalar;
        return *this;
    }

    /// Divide each element of this vector by the corresponding element of the given vector.
    template <typename S>
    const Vec3<T> &operator/=(const Vec3<S> &v1)
    {
        this->mm[0] /= v1[0];
        this->mm[1] /= v1[1];
        this->mm[2] /= v1[2];
        return *this;
    }

    /// Add @a scalar to each element of this vector.
    template <typename S>
    const Vec3<T> &operator+=(S scalar)
    {
        this->mm[0] = static_cast<T>(this->mm[0] + scalar);
        this->mm[1] = static_cast<T>(this->mm[1] + scalar);
        this->mm[2] = static_cast<T>(this->mm[2] + scalar);
        return *this;
    }

    /// Add each element of the given vector to the corresponding element of this vector.
    template <typename S>
    const Vec3<T> &operator+=(const Vec3<S> &v1)
    {
        this->mm[0] += v1[0];
        this->mm[1] += v1[1];
        this->mm[2] += v1[2];
        return *this;
    }

    /// Subtract @a scalar from each element of this vector.
    template <typename S>
    const Vec3<T> &operator-=(S scalar)
    {
        this->mm[0] -= scalar;
        this->mm[1] -= scalar;
        this->mm[2] -= scalar;
        return *this;
    }

    /// Subtract each element of the given vector from the corresponding element of this vector.
    template <typename S>
    const Vec3<T> &operator-=(const Vec3<S> &v1)
    {
        this->mm[0] -= v1[0];
        this->mm[1] -= v1[1];
        this->mm[2] -= v1[2];
        return *this;
    }

    /// Return a reference to itself after the exponent has been
    /// applied to all the vector components.
    inline const Vec3<T>& exp()
    {
        this->mm[0] = std::exp(this->mm[0]);
        this->mm[1] = std::exp(this->mm[1]);
        this->mm[2] = std::exp(this->mm[2]);
        return *this;
    }

    /// Return a reference to itself after log has been
    /// applied to all the vector components.
    inline const Vec3<T>& log()
    {
        this->mm[0] = std::log(this->mm[0]);
        this->mm[1] = std::log(this->mm[1]);
        this->mm[2] = std::log(this->mm[2]);
        return *this;
    }

    /// Return the sum of all the vector components.
    inline T sum() const
    {
        return this->mm[0] + this->mm[1] + this->mm[2];
    }

    /// Return the product of all the vector components.
    inline T product() const
    {
        return this->mm[0] * this->mm[1] * this->mm[2];
    }

    /// this = normalized this
    bool normalize(T eps = T(1.0e-7))
    {
        T d = length();
        if (isApproxEqual(d, T(0), eps)) {
            return false;
        }
        *this *= (T(1) / d);
        return true;
    }


    /// return normalized this, throws if null vector
    Vec3<T> unit(T eps=0) const
    {
        T d;
        return unit(eps, d);
    }

    /// return normalized this and length, throws if null vector
    Vec3<T> unit(T eps, T& len) const
    {
        len = length();
        if (isApproxEqual(len, T(0), eps)) {
            OPENVDB_THROW(ArithmeticError, "Normalizing null 3-vector");
        }
        return *this / len;
    }

    /// return normalized this, or (1, 0, 0) if this is null vector
    Vec3<T> unitSafe() const
    {
        T l2 = lengthSqr();
        return l2 ? *this / static_cast<T>(sqrt(l2)) : Vec3<T>(1, 0 ,0);
    }

    // Number of cols, rows, elements
    static unsigned numRows() { return 1; }
    static unsigned numColumns() { return 3; }
    static unsigned numElements() { return 3; }

    /// Returns the scalar component of v in the direction of onto, onto need
    /// not be unit. e.g   double c = Vec3d::component(v1,v2);
    T component(const Vec3<T> &onto, T eps = static_cast<T>(1.0e-7)) const
    {
        T l = onto.length();
        if (isApproxEqual(l, T(0), eps)) return 0;

        return dot(onto)*(T(1)/l);
    }

    /// Return the projection of v onto the vector, onto need not be unit
    /// e.g.   Vec3d a = vprojection(n);
    Vec3<T> projection(const Vec3<T> &onto, T eps = static_cast<T>(1.0e-7)) const
    {
        T l = onto.lengthSqr();
        if (isApproxEqual(l, T(0), eps)) return Vec3::zero();

        return onto*(dot(onto)*(T(1)/l));
    }

    /// Return an arbitrary unit vector perpendicular to v
    /// Vector this must be a unit vector
    /// e.g.   v = v.normalize(); Vec3d n = v.getArbPerpendicular();
    Vec3<T> getArbPerpendicular() const
    {
        Vec3<T> u;
        T l;

        if ( fabs(this->mm[0]) >= fabs(this->mm[1]) ) {
            // v.x or v.z is the largest magnitude component, swap them
            l = this->mm[0]*this->mm[0] + this->mm[2]*this->mm[2];
            l = static_cast<T>(T(1)/sqrt(double(l)));
            u.mm[0] = -this->mm[2]*l;
            u.mm[1] = T(0);
            u.mm[2] = +this->mm[0]*l;
        } else {
            // W.y or W.z is the largest magnitude component, swap them
            l = this->mm[1]*this->mm[1] + this->mm[2]*this->mm[2];
            l = static_cast<T>(T(1)/sqrt(double(l)));
            u.mm[0] = T(0);
            u.mm[1] = +this->mm[2]*l;
            u.mm[2] = -this->mm[1]*l;
        }

        return u;
    }

    /// Return a vector with the components of this in ascending order
    Vec3<T> sorted() const
    {
        Vec3<T> r(*this);
        if( r.mm[0] > r.mm[1] ) std::swap(r.mm[0], r.mm[1]);
        if( r.mm[1] > r.mm[2] ) std::swap(r.mm[1], r.mm[2]);
        if( r.mm[0] > r.mm[1] ) std::swap(r.mm[0], r.mm[1]);
        return r;
    }

    /// Return the vector (z, y, x)
    Vec3<T> reversed() const
    {
        return Vec3<T>(this->mm[2], this->mm[1], this->mm[0]);
    }

    /// Predefined constants, e.g.   Vec3d v = Vec3d::xNegAxis();
    static Vec3<T> zero() { return Vec3<T>(0, 0, 0); }
    static Vec3<T> ones() { return Vec3<T>(1, 1, 1); }
};


/// Equality operator, does exact floating point comparisons
template <typename T0, typename T1>
inline bool operator==(const Vec3<T0> &v0, const Vec3<T1> &v1)
{
    return isExactlyEqual(v0[0], v1[0]) && isExactlyEqual(v0[1], v1[1])
        && isExactlyEqual(v0[2], v1[2]);
}

/// Inequality operator, does exact floating point comparisons
template <typename T0, typename T1>
inline bool operator!=(const Vec3<T0> &v0, const Vec3<T1> &v1) { return !(v0==v1); }

/// Multiply each element of the given vector by @a scalar and return the result.
template <typename S, typename T>
inline Vec3<typename promote<S, T>::type> operator*(S scalar, const Vec3<T> &v) { return v*scalar; }

/// Multiply each element of the given vector by @a scalar and return the result.
template <typename S, typename T>
inline Vec3<typename promote<S, T>::type> operator*(const Vec3<T> &v, S scalar)
{
    Vec3<typename promote<S, T>::type> result(v);
    result *= scalar;
    return result;
}

/// Multiply corresponding elements of @a v0 and @a v1 and return the result.
template <typename T0, typename T1>
inline Vec3<typename promote<T0, T1>::type> operator*(const Vec3<T0> &v0, const Vec3<T1> &v1)
{
    Vec3<typename promote<T0, T1>::type> result(v0[0] * v1[0], v0[1] * v1[1], v0[2] * v1[2]);
    return result;
}


/// Divide @a scalar by each element of the given vector and return the result.
template <typename S, typename T>
inline Vec3<typename promote<S, T>::type> operator/(S scalar, const Vec3<T> &v)
{
    return Vec3<typename promote<S, T>::type>(scalar/v[0], scalar/v[1], scalar/v[2]);
}

/// Divide each element of the given vector by @a scalar and return the result.
template <typename S, typename T>
inline Vec3<typename promote<S, T>::type> operator/(const Vec3<T> &v, S scalar)
{
    Vec3<typename promote<S, T>::type> result(v);
    result /= scalar;
    return result;
}

/// Divide corresponding elements of @a v0 and @a v1 and return the result.
template <typename T0, typename T1>
inline Vec3<typename promote<T0, T1>::type> operator/(const Vec3<T0> &v0, const Vec3<T1> &v1)
{
    Vec3<typename promote<T0, T1>::type> result(v0[0] / v1[0], v0[1] / v1[1], v0[2] / v1[2]);
    return result;
}

/// Add corresponding elements of @a v0 and @a v1 and return the result.
template <typename T0, typename T1>
inline Vec3<typename promote<T0, T1>::type> operator+(const Vec3<T0> &v0, const Vec3<T1> &v1)
{
    Vec3<typename promote<T0, T1>::type> result(v0);
    result += v1;
    return result;
}

/// Add @a scalar to each element of the given vector and return the result.
template <typename S, typename T>
inline Vec3<typename promote<S, T>::type> operator+(const Vec3<T> &v, S scalar)
{
    Vec3<typename promote<S, T>::type> result(v);
    result += scalar;
    return result;
}

/// Subtract corresponding elements of @a v0 and @a v1 and return the result.
template <typename T0, typename T1>
inline Vec3<typename promote<T0, T1>::type> operator-(const Vec3<T0> &v0, const Vec3<T1> &v1)
{
    Vec3<typename promote<T0, T1>::type> result(v0);
    result -= v1;
    return result;
}

/// Subtract @a scalar from each element of the given vector and return the result.
template <typename S, typename T>
inline Vec3<typename promote<S, T>::type> operator-(const Vec3<T> &v, S scalar)
{
    Vec3<typename promote<S, T>::type> result(v);
    result -= scalar;
    return result;
}

/// Angle between two vectors, the result is between [0, pi],
/// e.g.   double a = Vec3d::angle(v1,v2);
template <typename T>
inline T angle(const Vec3<T> &v1, const Vec3<T> &v2)
{
    Vec3<T> c = v1.cross(v2);
    return static_cast<T>(atan2(c.length(), v1.dot(v2)));
}

template <typename T>
inline bool
isApproxEqual(const Vec3<T>& a, const Vec3<T>& b)
{
    return a.eq(b);
}
template <typename T>
inline bool
isApproxEqual(const Vec3<T>& a, const Vec3<T>& b, const Vec3<T>& eps)
{
    return isApproxEqual(a.x(), b.x(), eps.x()) &&
           isApproxEqual(a.y(), b.y(), eps.y()) &&
           isApproxEqual(a.z(), b.z(), eps.z());
}

template<typename T>
inline Vec3<T>
Abs(const Vec3<T>& v)
{
    return Vec3<T>(Abs(v[0]), Abs(v[1]), Abs(v[2]));
}

/// Orthonormalize vectors v1, v2 and v3 and store back the resulting
/// basis e.g.   Vec3d::orthonormalize(v1,v2,v3);
template <typename T>
inline void orthonormalize(Vec3<T> &v1, Vec3<T> &v2, Vec3<T> &v3)
{
    // If the input vectors are v0, v1, and v2, then the Gram-Schmidt
    // orthonormalization produces vectors u0, u1, and u2 as follows,
    //
    //   u0 = v0/|v0|
    //   u1 = (v1-(u0*v1)u0)/|v1-(u0*v1)u0|
    //   u2 = (v2-(u0*v2)u0-(u1*v2)u1)/|v2-(u0*v2)u0-(u1*v2)u1|
    //
    // where |A| indicates length of vector A and A*B indicates dot
    // product of vectors A and B.

    // compute u0
    v1.normalize();

    // compute u1
    T d0 = v1.dot(v2);
    v2 -= v1*d0;
    v2.normalize();

    // compute u2
    T d1 = v2.dot(v3);
    d0 = v1.dot(v3);
    v3 -= v1*d0 + v2*d1;
    v3.normalize();
}

/// @remark We are switching to a more explicit name because the semantics
/// are different from std::min/max. In that case, the function returns a
/// reference to one of the objects based on a comparator. Here, we must
/// fabricate a new object which might not match either of the inputs.

/// Return component-wise minimum of the two vectors.
template <typename T>
inline Vec3<T> minComponent(const Vec3<T> &v1, const Vec3<T> &v2)
{
    return Vec3<T>(
            std::min(v1.x(), v2.x()),
            std::min(v1.y(), v2.y()),
            std::min(v1.z(), v2.z()));
}

/// Return component-wise maximum of the two vectors.
template <typename T>
inline Vec3<T> maxComponent(const Vec3<T> &v1, const Vec3<T> &v2)
{
    return Vec3<T>(
            std::max(v1.x(), v2.x()),
            std::max(v1.y(), v2.y()),
            std::max(v1.z(), v2.z()));
}

/// @brief Return a vector with the exponent applied to each of
/// the components of the input vector.
template <typename T>
inline Vec3<T> Exp(Vec3<T> v) { return v.exp(); }

/// @brief Return a vector with log applied to each of
/// the components of the input vector.
template <typename T>
inline Vec3<T> Log(Vec3<T> v) { return v.log(); }

using Vec3i = Vec3<int32_t>;
using Vec3ui = Vec3<uint32_t>;
using Vec3s = Vec3<float>;
using Vec3d = Vec3<double>;

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_VEC3_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
