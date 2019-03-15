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

#ifndef OPENVDB_MATH_VEC2_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_VEC2_HAS_BEEN_INCLUDED

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

template<typename T> class Mat2;

template<typename T>
class Vec2: public Tuple<2, T>
{
public:
    using value_type = T;
    using ValueType = T;

    /// Trivial constructor, the vector is NOT initialized
    Vec2() {}

    /// @brief Construct a vector all of whose components have the given value.
    explicit Vec2(T val) { this->mm[0] = this->mm[1] = val; }

    /// Constructor with two arguments, e.g.   Vec2f v(1,2,3);
    Vec2(T x, T y)
    {
        this->mm[0] = x;
        this->mm[1] = y;
    }

    /// Constructor with array argument, e.g.   float a[2]; Vec2f v(a);
    template <typename Source>
    Vec2(Source *a)
    {
        this->mm[0] = a[0];
        this->mm[1] = a[1];
    } // trivial

    /// Conversion constructor
    template<typename Source>
    explicit Vec2(const Tuple<2, Source> &t)
    {
        this->mm[0] = static_cast<T>(t[0]);
        this->mm[1] = static_cast<T>(t[1]);
    }

    /// @brief Construct a vector all of whose components have the given value,
    /// which may be of an arithmetic type different from this vector's value type.
    /// @details Type conversion warnings are suppressed.
    template<typename Other>
    explicit Vec2(Other val,
        typename std::enable_if<std::is_arithmetic<Other>::value, Conversion>::type = Conversion{})
    {
        this->mm[0] = this->mm[1] = static_cast<T>(val);
    }

    /// Reference to the component, e.g.   v.x() = 4.5f;
    T& x() {return this->mm[0];}
    T& y() {return this->mm[1];}

    /// Get the component, e.g.   float f = v.y();
    T x() const {return this->mm[0];}
    T y() const {return this->mm[1];}

    /// Alternative indexed reference to the elements
    T& operator()(int i) {return this->mm[i];}

    /// Alternative indexed constant reference to the elements,
    T operator()(int i) const {return this->mm[i];}

    T* asPointer() {return this->mm;}
    const T* asPointer() const {return this->mm;}

    /// "this" vector gets initialized to [x, y, z],
    /// calling v.init(); has same effect as calling v = Vec2::zero();
    const Vec2<T>& init(T x=0, T y=0)
    {
        this->mm[0] = x; this->mm[1] = y;
        return *this;
    }

    /// Set "this" vector to zero
    const Vec2<T>& setZero()
    {
        this->mm[0] = 0; this->mm[1] = 0;
        return *this;
    }

    /// Assignment operator
    template<typename Source>
    const Vec2<T>& operator=(const Vec2<Source> &v)
    {
        // note: don't static_cast because that suppresses warnings
        this->mm[0] = v[0];
        this->mm[1] = v[1];

        return *this;
    }

    /// Equality operator, does exact floating point comparisons
    bool operator==(const Vec2<T> &v) const
    {
        return (isExactlyEqual(this->mm[0], v.mm[0]) && isExactlyEqual(this->mm[1], v.mm[1]));
    }

    /// Inequality operator, does exact floating point comparisons
    bool operator!=(const Vec2<T> &v) const { return !(*this==v); }

    /// Test if "this" vector is equivalent to vector v with tolerance of eps
    bool eq(const Vec2<T> &v, T eps = static_cast<T>(1.0e-7)) const
    {
        return isApproxEqual(this->mm[0], v.mm[0], eps) &&
               isApproxEqual(this->mm[1], v.mm[1], eps);
    } // trivial

    /// Negation operator, for e.g.   v1 = -v2;
    Vec2<T> operator-() const {return Vec2<T>(-this->mm[0], -this->mm[1]);}

    /// this = v1 + v2
    /// "this", v1 and v2 need not be distinct objects, e.g. v.add(v1,v);
    template <typename T0, typename T1>
    const Vec2<T>& add(const Vec2<T0> &v1, const Vec2<T1> &v2)
    {
        this->mm[0] = v1[0] + v2[0];
        this->mm[1] = v1[1] + v2[1];

        return *this;
    }

    /// this = v1 - v2
    /// "this", v1 and v2 need not be distinct objects, e.g. v.sub(v1,v);
    template <typename T0, typename T1>
    const Vec2<T>& sub(const Vec2<T0> &v1, const Vec2<T1> &v2)
    {
        this->mm[0] = v1[0] - v2[0];
        this->mm[1] = v1[1] - v2[1];

        return *this;
    }

    /// this =  scalar*v, v need not be a distinct object from "this",
    /// e.g. v.scale(1.5,v1);
    template <typename T0, typename T1>
    const Vec2<T>& scale(T0 scalar, const Vec2<T1> &v)
    {
        this->mm[0] = scalar * v[0];
        this->mm[1] = scalar * v[1];

        return *this;
    }

    template <typename T0, typename T1>
    const Vec2<T> &div(T0 scalar, const Vec2<T1> &v)
    {
        this->mm[0] = v[0] / scalar;
        this->mm[1] = v[1] / scalar;

        return *this;
    }

    /// Dot product
    T dot(const Vec2<T> &v) const { return this->mm[0]*v[0] + this->mm[1]*v[1]; } // trivial

    /// Length of the vector
    T length() const
    {
        return static_cast<T>(sqrt(double(this->mm[0]*this->mm[0] + this->mm[1]*this->mm[1])));
    }

    /// Squared length of the vector, much faster than length() as it
    /// does not involve square root
    T lengthSqr() const { return (this->mm[0]*this->mm[0] + this->mm[1]*this->mm[1]); }

    /// Return a reference to itsef after the exponent has been
    /// applied to all the vector components.
    inline const Vec2<T>& exp()
    {
        this->mm[0] = std::exp(this->mm[0]);
        this->mm[1] = std::exp(this->mm[1]);
        return *this;
    }

    /// Return a reference to itself after log has been
    /// applied to all the vector components.
    inline const Vec2<T>& log()
    {
        this->mm[0] = std::log(this->mm[0]);
        this->mm[1] = std::log(this->mm[1]);
        return *this;
    }

    /// Return the sum of all the vector components.
    inline T sum() const
    {
        return this->mm[0] + this->mm[1];
    }

    /// Return the product of all the vector components.
    inline T product() const
    {
        return this->mm[0] * this->mm[1];
    }

    /// this = normalized this
    bool normalize(T eps = static_cast<T>(1.0e-8))
    {
        T d = length();
        if (isApproxEqual(d, T(0), eps)) {
            return false;
        }
        *this *= (T(1) / d);
        return true;
    }

    /// return normalized this, throws if null vector
    Vec2<T> unit(T eps=0) const
    {
        T d;
        return unit(eps, d);
    }

    /// return normalized this and length, throws if null vector
    Vec2<T> unit(T eps, T& len) const
    {
        len = length();
        if (isApproxEqual(len, T(0), eps)) {
            OPENVDB_THROW(ArithmeticError, "Normalizing null 2-vector");
        }
        return *this / len;
    }

    /// return normalized this, or (1, 0) if this is null vector
    Vec2<T> unitSafe() const
    {
        T l2 = lengthSqr();
        return l2 ? *this/static_cast<T>(sqrt(l2)) : Vec2<T>(1,0);
    }

    /// Multiply each element of this vector by @a scalar.
    template <typename S>
    const Vec2<T> &operator*=(S scalar)
    {
        this->mm[0] *= scalar;
        this->mm[1] *= scalar;
        return *this;
    }

    /// Multiply each element of this vector by the corresponding element of the given vector.
    template <typename S>
    const Vec2<T> &operator*=(const Vec2<S> &v1)
    {
        this->mm[0] *= v1[0];
        this->mm[1] *= v1[1];
        return *this;
    }

    /// Divide each element of this vector by @a scalar.
    template <typename S>
    const Vec2<T> &operator/=(S scalar)
    {
        this->mm[0] /= scalar;
        this->mm[1] /= scalar;
        return *this;
    }

    /// Divide each element of this vector by the corresponding element of the given vector.
    template <typename S>
    const Vec2<T> &operator/=(const Vec2<S> &v1)
    {
        this->mm[0] /= v1[0];
        this->mm[1] /= v1[1];
        return *this;
    }

    /// Add @a scalar to each element of this vector.
    template <typename S>
    const Vec2<T> &operator+=(S scalar)
    {
        this->mm[0] += scalar;
        this->mm[1] += scalar;
        return *this;
    }

    /// Add each element of the given vector to the corresponding element of this vector.
    template <typename S>
    const Vec2<T> &operator+=(const Vec2<S> &v1)
    {
        this->mm[0] += v1[0];
        this->mm[1] += v1[1];
        return *this;
    }

    /// Subtract @a scalar from each element of this vector.
    template <typename S>
    const Vec2<T> &operator-=(S scalar)
    {
        this->mm[0] -= scalar;
        this->mm[1] -= scalar;
        return *this;
    }

    /// Subtract each element of the given vector from the corresponding element of this vector.
    template <typename S>
    const Vec2<T> &operator-=(const Vec2<S> &v1)
    {
        this->mm[0] -= v1[0];
        this->mm[1] -= v1[1];
        return *this;
    }

    // Number of cols, rows, elements
    static unsigned numRows() { return 1; }
    static unsigned numColumns() { return 2; }
    static unsigned numElements() { return 2; }

    /// Returns the scalar component of v in the direction of onto, onto need
    /// not be unit. e.g   float c = Vec2f::component(v1,v2);
    T component(const Vec2<T> &onto, T eps = static_cast<T>(1.0e-8)) const
    {
        T l = onto.length();
        if (isApproxEqual(l,  T(0), eps)) return 0;

        return dot(onto)*(T(1)/l);
    }

    /// Return the projection of v onto the vector, onto need not be unit
    /// e.g.   Vec2f v = Vec2f::projection(v,n);
    Vec2<T> projection(const Vec2<T> &onto, T eps = static_cast<T>(1.0e-8)) const
    {
        T l = onto.lengthSqr();
        if (isApproxEqual(l, T(0), eps)) return Vec2::zero();

        return onto*(dot(onto)*(T(1)/l));
    }

    /// Return an arbitrary unit vector perpendicular to v
    /// Vector v must be a unit vector
    /// e.g.   v.normalize(); Vec2f n = Vec2f::getArbPerpendicular(v);
    Vec2<T> getArbPerpendicular() const { return Vec2<T>(-this->mm[1], this->mm[0]); }

    /// Predefined constants, e.g.   Vec2f v = Vec2f::xNegAxis();
    static Vec2<T> zero() { return Vec2<T>(0, 0); }
    static Vec2<T> ones() { return Vec2<T>(1, 1); }
};


/// Multiply each element of the given vector by @a scalar and return the result.
template <typename S, typename T>
inline Vec2<typename promote<S, T>::type> operator*(S scalar, const Vec2<T> &v)
{
    return v * scalar;
}

/// Multiply each element of the given vector by @a scalar and return the result.
template <typename S, typename T>
inline Vec2<typename promote<S, T>::type> operator*(const Vec2<T> &v, S scalar)
{
    Vec2<typename promote<S, T>::type> result(v);
    result *= scalar;
    return result;
}

/// Multiply corresponding elements of @a v0 and @a v1 and return the result.
template <typename T0, typename T1>
inline Vec2<typename promote<T0, T1>::type> operator*(const Vec2<T0> &v0, const Vec2<T1> &v1)
{
    Vec2<typename promote<T0, T1>::type> result(v0[0] * v1[0], v0[1] * v1[1]);
    return result;
}

/// Divide @a scalar by each element of the given vector and return the result.
template <typename S, typename T>
inline Vec2<typename promote<S, T>::type> operator/(S scalar, const Vec2<T> &v)
{
    return Vec2<typename promote<S, T>::type>(scalar/v[0], scalar/v[1]);
}

/// Divide each element of the given vector by @a scalar and return the result.
template <typename S, typename T>
inline Vec2<typename promote<S, T>::type> operator/(const Vec2<T> &v, S scalar)
{
    Vec2<typename promote<S, T>::type> result(v);
    result /= scalar;
    return result;
}

/// Divide corresponding elements of @a v0 and @a v1 and return the result.
template <typename T0, typename T1>
inline Vec2<typename promote<T0, T1>::type> operator/(const Vec2<T0> &v0, const Vec2<T1> &v1)
{
    Vec2<typename promote<T0, T1>::type> result(v0[0] / v1[0], v0[1] / v1[1]);
    return result;
}

/// Add corresponding elements of @a v0 and @a v1 and return the result.
template <typename T0, typename T1>
inline Vec2<typename promote<T0, T1>::type> operator+(const Vec2<T0> &v0, const Vec2<T1> &v1)
{
    Vec2<typename promote<T0, T1>::type> result(v0);
    result += v1;
    return result;
}

/// Add @a scalar to each element of the given vector and return the result.
template <typename S, typename T>
inline Vec2<typename promote<S, T>::type> operator+(const Vec2<T> &v, S scalar)
{
    Vec2<typename promote<S, T>::type> result(v);
    result += scalar;
    return result;
}

/// Subtract corresponding elements of @a v0 and @a v1 and return the result.
template <typename T0, typename T1>
inline Vec2<typename promote<T0, T1>::type> operator-(const Vec2<T0> &v0, const Vec2<T1> &v1)
{
    Vec2<typename promote<T0, T1>::type> result(v0);
    result -= v1;
    return result;
}

/// Subtract @a scalar from each element of the given vector and return the result.
template <typename S, typename T>
inline Vec2<typename promote<S, T>::type> operator-(const Vec2<T> &v, S scalar)
{
    Vec2<typename promote<S, T>::type> result(v);
    result -= scalar;
    return result;
}

/// Angle between two vectors, the result is between [0, pi],
/// e.g.   float a = Vec2f::angle(v1,v2);
template <typename T>
inline T angle(const Vec2<T> &v1, const Vec2<T> &v2)
{
    T c = v1.dot(v2);
    return acos(c);
}

template <typename T>
inline bool
isApproxEqual(const Vec2<T>& a, const Vec2<T>& b)
{
    return a.eq(b);
}
template <typename T>
inline bool
isApproxEqual(const Vec2<T>& a, const Vec2<T>& b, const Vec2<T>& eps)
{
    return isApproxEqual(a.x(), b.x(), eps.x()) &&
           isApproxEqual(a.y(), b.y(), eps.y());
}

template<typename T>
inline Vec2<T>
Abs(const Vec2<T>& v)
{
    return Vec2<T>(Abs(v[0]), Abs(v[1]));
}

/// Orthonormalize vectors v1 and v2 and store back the resulting basis
/// e.g.   Vec2f::orthonormalize(v1,v2);
template <typename T>
inline void orthonormalize(Vec2<T> &v1, Vec2<T> &v2)
{
    // If the input vectors are v0, v1, and v2, then the Gram-Schmidt
    // orthonormalization produces vectors u0, u1, and u2 as follows,
    //
    //   u0 = v0/|v0|
    //   u1 = (v1-(u0*v1)u0)/|v1-(u0*v1)u0|
    //
    // where |A| indicates length of vector A and A*B indicates dot
    // product of vectors A and B.

    // compute u0
    v1.normalize();

    // compute u1
    T d0 = v1.dot(v2);
    v2 -= v1*d0;
    v2.normalize();
}


/// \remark We are switching to a more explicit name because the semantics
/// are different from std::min/max. In that case, the function returns a
/// reference to one of the objects based on a comparator. Here, we must
/// fabricate a new object which might not match either of the inputs.

/// Return component-wise minimum of the two vectors.
template <typename T>
inline Vec2<T> minComponent(const Vec2<T> &v1, const Vec2<T> &v2)
{
    return Vec2<T>(
            std::min(v1.x(), v2.x()),
            std::min(v1.y(), v2.y()));
}

/// Return component-wise maximum of the two vectors.
template <typename T>
inline Vec2<T> maxComponent(const Vec2<T> &v1, const Vec2<T> &v2)
{
    return Vec2<T>(
            std::max(v1.x(), v2.x()),
            std::max(v1.y(), v2.y()));
}

/// @brief Return a vector with the exponent applied to each of
/// the components of the input vector.
template <typename T>
inline Vec2<T> Exp(Vec2<T> v) { return v.exp(); }

/// @brief Return a vector with log applied to each of
/// the components of the input vector.
template <typename T>
inline Vec2<T> Log(Vec2<T> v) { return v.log(); }

using Vec2i = Vec2<int32_t>;
using Vec2ui = Vec2<uint32_t>;
using Vec2s = Vec2<float>;
using Vec2d = Vec2<double>;

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_VEC2_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
