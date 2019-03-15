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
/// @file Ray.h
///
/// @author Ken Museth
///
/// @brief A Ray class.

#ifndef OPENVDB_MATH_RAY_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_RAY_HAS_BEEN_INCLUDED

#include "Math.h"
#include "Vec3.h"
#include "Transform.h"
#include <algorithm> // for std::swap()
#include <iostream> // for std::ostream
#include <limits> // for std::numeric_limits<Type>::max()

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

template<typename RealT = double>
class Ray
{
public:
    static_assert(std::is_floating_point<RealT>::value,
        "math::Ray requires a floating-point value type");

    using RealType = RealT;
    using Vec3Type = Vec3<RealT>;
    using Vec3T = Vec3Type;

    struct TimeSpan {
        RealT t0, t1;
        /// @brief Default constructor
        TimeSpan() {}
        /// @brief Constructor
        TimeSpan(RealT _t0, RealT _t1) : t0(_t0), t1(_t1) {}
        /// @brief Set both times
        inline void set(RealT _t0, RealT _t1) { t0=_t0; t1=_t1; }
        /// @brief Get both times
        inline void get(RealT& _t0, RealT& _t1) const { _t0=t0; _t1=t1; }
        /// @brief Return @c true if t1 is larger than t0 by at least eps.
        inline bool valid(RealT eps=math::Delta<RealT>::value()) const { return (t1-t0)>eps; }
        /// @brief Return the midpoint of the ray.
        inline RealT mid() const { return 0.5*(t0 + t1); }
        /// @brief Multiplies both times
        inline void scale(RealT s) {assert(s>0); t0*=s; t1*=s; }
        /// @brief Return @c true if time is inclusive
        inline bool test(RealT t) const { return (t>=t0 && t<=t1); }
    };

    Ray(const Vec3Type& eye = Vec3Type(0,0,0),
        const Vec3Type& direction = Vec3Type(1,0,0),
        RealT t0 = math::Delta<RealT>::value(),
        RealT t1 = std::numeric_limits<RealT>::max())
        : mEye(eye), mDir(direction), mInvDir(1/mDir), mTimeSpan(t0, t1)
    {
    }

    inline void setEye(const Vec3Type& eye) { mEye = eye; }

    inline void setDir(const Vec3Type& dir)
    {
        mDir = dir;
        mInvDir = 1/mDir;
    }

    inline void setMinTime(RealT t0) { assert(t0>0); mTimeSpan.t0 = t0; }

    inline void setMaxTime(RealT t1) { assert(t1>0); mTimeSpan.t1 = t1; }

    inline void setTimes(
        RealT t0 = math::Delta<RealT>::value(),
        RealT t1 = std::numeric_limits<RealT>::max())
    {
        assert(t0>0 && t1>0);
        mTimeSpan.set(t0, t1);
    }

    inline void scaleTimes(RealT scale) { mTimeSpan.scale(scale); }

    inline void reset(
        const Vec3Type& eye,
        const Vec3Type& direction,
        RealT t0 = math::Delta<RealT>::value(),
        RealT t1 = std::numeric_limits<RealT>::max())
    {
        this->setEye(eye);
        this->setDir(direction);
        this->setTimes(t0, t1);
    }

    inline const Vec3T& eye() const {return mEye;}

    inline const Vec3T& dir() const {return mDir;}

    inline const Vec3T& invDir() const {return mInvDir;}

    inline RealT t0() const {return mTimeSpan.t0;}

    inline RealT t1() const {return mTimeSpan.t1;}

    /// @brief Return the position along the ray at the specified time.
    inline Vec3R operator()(RealT time) const { return mEye + mDir * time; }

    /// @brief Return the starting point of the ray.
    inline Vec3R start() const { return (*this)(mTimeSpan.t0); }

    /// @brief Return the endpoint of the ray.
    inline Vec3R end() const { return (*this)(mTimeSpan.t1); }

    /// @brief Return the midpoint of the ray.
    inline Vec3R mid() const { return (*this)(mTimeSpan.mid()); }

    /// @brief Return @c true if t1 is larger than t0 by at least eps.
    inline bool valid(RealT eps=math::Delta<float>::value()) const { return mTimeSpan.valid(eps); }

    /// @brief Return @c true if @a time is within t0 and t1, both inclusive.
    inline bool test(RealT time) const { return mTimeSpan.test(time); }

    /// @brief Return a new Ray that is transformed with the specified map.
    /// @param map  the map from which to construct the new Ray.
    /// @warning Assumes a linear map and a normalized direction.
    /// @details The requirement that the direction is normalized
    /// follows from the transformation of t0 and t1 - and that fact that
    /// we want applyMap and applyInverseMap to be inverse operations.
    template<typename MapType>
    inline Ray applyMap(const MapType& map) const
    {
        assert(map.isLinear());
        assert(math::isRelOrApproxEqual(mDir.length(), RealT(1),
            Tolerance<RealT>::value(), Delta<RealT>::value()));
        const Vec3T eye = map.applyMap(mEye);
        const Vec3T dir = map.applyJacobian(mDir);
        const RealT length = dir.length();
        return Ray(eye, dir/length, length*mTimeSpan.t0, length*mTimeSpan.t1);
    }

    /// @brief Return a new Ray that is transformed with the inverse of the specified map.
    /// @param map  the map from which to construct the new Ray by inverse mapping.
    /// @warning Assumes a linear map and a normalized direction.
    /// @details The requirement that the direction is normalized
    /// follows from the transformation of t0 and t1 - and that fact that
    /// we want applyMap and applyInverseMap to be inverse operations.
    template<typename MapType>
    inline Ray applyInverseMap(const MapType& map) const
    {
        assert(map.isLinear());
        assert(math::isRelOrApproxEqual(mDir.length(), RealT(1), Tolerance<RealT>::value(), Delta<RealT>::value()));
        const Vec3T eye = map.applyInverseMap(mEye);
        const Vec3T dir = map.applyInverseJacobian(mDir);
        const RealT length = dir.length();
        return Ray(eye, dir/length, length*mTimeSpan.t0, length*mTimeSpan.t1);
    }

    /// @brief Return a new ray in world space, assuming the existing
    /// ray is represented in the index space of the specified grid.
    template<typename GridType>
    inline Ray indexToWorld(const GridType& grid) const
    {
        return this->applyMap(*(grid.transform().baseMap()));
    }

    /// @brief Return a new ray in the index space of the specified
    /// grid, assuming the existing ray is represented in world space.
    template<typename GridType>
    inline Ray worldToIndex(const GridType& grid) const
    {
        return this->applyInverseMap(*(grid.transform().baseMap()));
    }

    /// @brief Return true if this ray intersects the specified sphere.
    /// @param center The center of the sphere in the same space as this ray.
    /// @param radius The radius of the sphere in the same units as this ray.
    /// @param t0     The first intersection point if an intersection exists.
    /// @param t1     The second intersection point if an intersection exists.
    /// @note If the return value is true, i.e. a hit, and t0 =
    /// this->t0() or t1 == this->t1() only one true intersection exist.
    inline bool intersects(const Vec3T& center, RealT radius, RealT& t0, RealT& t1) const
    {
        const Vec3T origin = mEye - center;
        const RealT A = mDir.lengthSqr();
        const RealT B = 2 * mDir.dot(origin);
        const RealT C = origin.lengthSqr() - radius * radius;
        const RealT D = B * B - 4 * A * C;

        if (D < 0) return false;

        const RealT Q = RealT(-0.5)*(B<0 ? (B + Sqrt(D)) : (B - Sqrt(D)));

        t0 = Q / A;
        t1 = C / Q;

        if (t0 > t1) std::swap(t0, t1);
        if (t0 < mTimeSpan.t0) t0 = mTimeSpan.t0;
        if (t1 > mTimeSpan.t1) t1 = mTimeSpan.t1;
        return t0 <= t1;
    }

    /// @brief Return true if this ray intersects the specified sphere.
    /// @param center The center of the sphere in the same space as this ray.
    /// @param radius The radius of the sphere in the same units as this ray.
    inline bool intersects(const Vec3T& center, RealT radius) const
    {
        RealT t0, t1;
        return this->intersects(center, radius, t0, t1)>0;
    }

    /// @brief Return true if this ray intersects the specified sphere.
    /// @note For intersection this ray is clipped to the two intersection points.
    /// @param center The center of the sphere in the same space as this ray.
    /// @param radius The radius of the sphere in the same units as this ray.
    inline bool clip(const Vec3T& center, RealT radius)
    {
        RealT t0, t1;
        const bool hit = this->intersects(center, radius, t0, t1);
        if (hit) mTimeSpan.set(t0, t1);
        return hit;
    }

    /// @brief Return true if the Ray intersects the specified
    /// axisaligned bounding box.
    /// @param bbox Axis-aligned bounding box in the same space as the Ray.
    /// @param t0   If an intersection is detected this is assigned
    ///             the time for the first intersection point.
    /// @param t1   If an intersection is detected this is assigned
    ///             the time for the second intersection point.
    template<typename BBoxT>
    inline bool intersects(const BBoxT& bbox, RealT& t0, RealT& t1) const
    {
        mTimeSpan.get(t0, t1);
        for (int i = 0; i < 3; ++i) {
            RealT a = (bbox.min()[i] - mEye[i]) * mInvDir[i];
            RealT b = (bbox.max()[i] - mEye[i]) * mInvDir[i];
            if (a > b) std::swap(a, b);
            if (a > t0) t0 = a;
            if (b < t1) t1 = b;
            if (t0 > t1) return false;
        }
        return true;
    }

    /// @brief Return true if this ray intersects the specified bounding box.
    /// @param bbox Axis-aligned bounding box in the same space as this ray.
    template<typename BBoxT>
    inline bool intersects(const BBoxT& bbox) const
    {
        RealT t0, t1;
        return this->intersects(bbox, t0, t1);
    }

    /// @brief Return true if this ray intersects the specified bounding box.
    /// @note For intersection this ray is clipped to the two intersection points.
    /// @param bbox Axis-aligned bounding box in the same space as this ray.
    template<typename BBoxT>
    inline bool clip(const BBoxT& bbox)
    {
        RealT t0, t1;
        const bool hit = this->intersects(bbox, t0, t1);
        if (hit) mTimeSpan.set(t0, t1);
        return hit;
    }

    /// @brief Return true if the Ray intersects the plane specified
    /// by a normal and distance from the origin.
    /// @param normal   Normal of the plane.
    /// @param distance Distance of the plane to the origin.
    /// @param t        Time of intersection, if one exists.
    inline bool intersects(const Vec3T& normal, RealT distance, RealT& t) const
      {
          const RealT cosAngle = mDir.dot(normal);
          if (math::isApproxZero(cosAngle)) return false;//parallel
          t = (distance - mEye.dot(normal))/cosAngle;
          return this->test(t);
      }

    /// @brief Return true if the Ray intersects the plane specified
    /// by a normal and point.
    /// @param normal   Normal of the plane.
    /// @param point    Point in the plane.
    /// @param t        Time of intersection, if one exists.
    inline bool intersects(const Vec3T& normal, const Vec3T& point, RealT& t) const
      {
          return this->intersects(normal, point.dot(normal), t);
      }

private:
    Vec3T mEye, mDir, mInvDir;
    TimeSpan mTimeSpan;
}; // end of Ray class


/// @brief Output streaming of the Ray class.
/// @note Primarily intended for debugging.
template<typename RealT>
inline std::ostream& operator<<(std::ostream& os, const Ray<RealT>& r)
{
    os << "eye=" << r.eye() << " dir=" << r.dir() << " 1/dir="<<r.invDir()
       << " t0=" << r.t0()  << " t1="  << r.t1();
    return os;
}

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_RAY_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
