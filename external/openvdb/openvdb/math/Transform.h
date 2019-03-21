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

#ifndef OPENVDB_MATH_TRANSFORM_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_TRANSFORM_HAS_BEEN_INCLUDED

#include "Maps.h"
#include <openvdb/Types.h>
#include <iosfwd>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

// Forward declaration
class Transform;


// Utility methods

/// @brief Calculate an axis-aligned bounding box in index space from an
/// axis-aligned bounding box in world space.
/// @see Transform::worldToIndex(const BBoxd&) const
OPENVDB_API void
calculateBounds(const Transform& t, const Vec3d& minWS, const Vec3d& maxWS,
    Vec3d& minIS, Vec3d& maxIS);

/// @todo Calculate an axis-aligned bounding box in index space from a
/// bounding sphere in world space.
//void calculateBounds(const Transform& t, const Vec3d& center, const Real radius,
//    Vec3d& minIS, Vec3d& maxIS);


////////////////////////////////////////


/// @class Transform
class OPENVDB_API Transform
{
public:
    using Ptr = SharedPtr<Transform>;
    using ConstPtr = SharedPtr<const Transform>;

    Transform(): mMap(MapBase::Ptr(new ScaleMap())) {}
    Transform(const MapBase::Ptr&);
    Transform(const Transform&);
    ~Transform() {}

    Ptr copy() const { return Ptr(new Transform(mMap->copy())); }

    //@{
    /// @brief Create and return a shared pointer to a new transform.
    static Transform::Ptr createLinearTransform(double voxelSize = 1.0);
    static Transform::Ptr createLinearTransform(const Mat4R&);
    static Transform::Ptr createFrustumTransform(const BBoxd&, double taper,
        double depth, double voxelSize = 1.0);
    //@}

    /// Return @c true if the transformation map is exclusively linear/affine.
    bool isLinear() const { return mMap->isLinear(); }

    /// Return @c true if the transform is equivalent to an idenity.
    bool isIdentity() const ;
    /// Return the transformation map's type-name
    Name mapType() const { return mMap->type(); }


    //@{
    /// @brief Update the linear (affine) map by prepending or
    /// postfixing the appropriate operation.  In the case of
    /// a frustum, the pre-operations apply to the linear part
    /// of the transform and not the entire transform, while the
    /// post-operations are allways applied last.
    void preRotate(double radians, const Axis axis = X_AXIS);
    void preTranslate(const Vec3d&);
    void preScale(const Vec3d&);
    void preScale(double);
    void preShear(double shear, Axis axis0, Axis axis1);
    void preMult(const Mat4d&);
    void preMult(const Mat3d&);

    void postRotate(double radians, const Axis axis = X_AXIS);
    void postTranslate(const Vec3d&);
    void postScale(const Vec3d&);
    void postScale(double);
    void postShear(double shear, Axis axis0, Axis axis1);
    void postMult(const Mat4d&);
    void postMult(const Mat3d&);
    //@}

    /// Return the size of a voxel using the linear component of the map.
    Vec3d voxelSize() const { return mMap->voxelSize(); }
    /// @brief Return the size of a voxel at position (x, y, z).
    /// @note Maps that have a nonlinear component (e.g., perspective and frustum maps)
    /// have position-dependent voxel sizes.
    Vec3d voxelSize(const Vec3d& xyz) const { return mMap->voxelSize(xyz); }

    /// Return the voxel volume of the linear component of the map.
    double voxelVolume() const { return mMap->determinant(); }
    /// Return the voxel volume at position (x, y, z).
    double voxelVolume(const Vec3d& xyz) const { return mMap->determinant(xyz); }
    /// Return true if the voxels in world space are uniformly sized cubes
    bool hasUniformScale() const { return mMap->hasUniformScale(); }

    //@{
    /// @brief Apply this transformation to the given coordinates.
    Vec3d indexToWorld(const Vec3d& xyz) const { return mMap->applyMap(xyz); }
    Vec3d indexToWorld(const Coord& ijk) const { return mMap->applyMap(ijk.asVec3d()); }
    Vec3d worldToIndex(const Vec3d& xyz) const { return mMap->applyInverseMap(xyz); }
    Coord worldToIndexCellCentered(const Vec3d& xyz) const {return Coord::round(worldToIndex(xyz));}
    Coord worldToIndexNodeCentered(const Vec3d& xyz) const {return Coord::floor(worldToIndex(xyz));}
    //@}

    //@{
    /// @brief Apply this transformation to the given index-space bounding box.
    /// @return an axis-aligned world-space bounding box
    BBoxd indexToWorld(const CoordBBox&) const;
    BBoxd indexToWorld(const BBoxd&) const;
    //@}
    //@{
    /// @brief Apply the inverse of this transformation to the given world-space bounding box.
    /// @return an axis-aligned index-space bounding box
    BBoxd worldToIndex(const BBoxd&) const;
    CoordBBox worldToIndexCellCentered(const BBoxd&) const;
    CoordBBox worldToIndexNodeCentered(const BBoxd&) const;
    //@}

    //@{
    /// Return a base pointer to the transformation map.
    MapBase::ConstPtr baseMap() const { return mMap; }
    MapBase::Ptr baseMap() { return mMap; }
    //@}

    //@{
    /// @brief Return the result of downcasting the base map pointer to a
    /// @c MapType pointer, or return a null pointer if the types are incompatible.
    template<typename MapType> typename MapType::Ptr map();
    template<typename MapType> typename MapType::ConstPtr map() const;
    template<typename MapType> typename MapType::ConstPtr constMap() const;
    //@}

    /// Unserialize this transform from the given stream.
    void read(std::istream&);
    /// Serialize this transform to the given stream.
    void write(std::ostream&) const;

    /// @brief Print a description of this transform.
    /// @param os      a stream to which to write textual information
    /// @param indent  a string with which to prefix each line of text
    void print(std::ostream& os = std::cout, const std::string& indent = "") const;

    bool operator==(const Transform& other) const;
    inline bool operator!=(const Transform& other) const { return !(*this == other); }

private:
    MapBase::Ptr mMap;
}; // class Transform


OPENVDB_API std::ostream& operator<<(std::ostream&, const Transform&);


////////////////////////////////////////


template<typename MapType>
inline typename MapType::Ptr
Transform::map()
{
    if (mMap->type() == MapType::mapType()) {
        return StaticPtrCast<MapType>(mMap);
    }
    return typename MapType::Ptr();
}


template<typename MapType>
inline typename MapType::ConstPtr
Transform::map() const
{
    return ConstPtrCast<const MapType>(
        const_cast<Transform*>(this)->map<MapType>());
}


template<typename MapType>
inline typename MapType::ConstPtr
Transform::constMap() const
{
    return map<MapType>();
}


////////////////////////////////////////


/// Helper function used internally by processTypedMap()
template<typename ResolvedMapType, typename OpType>
inline void
doProcessTypedMap(Transform& transform, OpType& op)
{
    ResolvedMapType& resolvedMap = *transform.map<ResolvedMapType>();
#ifdef _MSC_VER
    op.operator()<ResolvedMapType>(resolvedMap);
#else
    op.template operator()<ResolvedMapType>(resolvedMap);
#endif
}

/// Helper function used internally by processTypedMap()
template<typename ResolvedMapType, typename OpType>
inline void
doProcessTypedMap(const Transform& transform, OpType& op)
{
    const ResolvedMapType& resolvedMap = *transform.map<ResolvedMapType>();
#ifdef _MSC_VER
    op.operator()<ResolvedMapType>(resolvedMap);
#else
    op.template operator()<ResolvedMapType>(resolvedMap);
#endif
}


/// @brief Utility function that, given a generic map pointer,
/// calls a functor on the fully-resoved map
///
/// Usage:
/// @code
/// struct Foo {
///     template<typename MapT>
///     void operator()(const MapT&  map) const { blah }
/// };
///
/// processTypedMap(myMap, Foo());
/// @endcode
///
/// @return @c false if the grid type is unknown or unhandled.
template<typename TransformType, typename OpType>
bool
processTypedMap(TransformType& transform, OpType& op)
{
    using namespace openvdb;

    const Name mapType = transform.mapType();
    if (mapType == UniformScaleMap::mapType()) {
        doProcessTypedMap<UniformScaleMap, OpType>(transform, op);

    } else if (mapType == UniformScaleTranslateMap::mapType()) {
        doProcessTypedMap<UniformScaleTranslateMap, OpType>(transform, op);

    } else if (mapType == ScaleMap::mapType()) {
        doProcessTypedMap<ScaleMap, OpType>(transform, op);

    } else if  (mapType == ScaleTranslateMap::mapType()) {
        doProcessTypedMap<ScaleTranslateMap, OpType>(transform, op);

    } else if (mapType == UnitaryMap::mapType()) {
        doProcessTypedMap<UnitaryMap, OpType>(transform, op);

    } else if (mapType == AffineMap::mapType()) {
        doProcessTypedMap<AffineMap, OpType>(transform, op);

    } else if (mapType == TranslationMap::mapType()) {
        doProcessTypedMap<TranslationMap, OpType>(transform, op);

    } else if (mapType == NonlinearFrustumMap::mapType()) {
        doProcessTypedMap<NonlinearFrustumMap, OpType>(transform, op);
    } else {
        return false;
    }
    return true;
}

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_TRANSFORM_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
