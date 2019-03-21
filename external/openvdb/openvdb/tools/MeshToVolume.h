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

/// @file   MeshToVolume.h
///
/// @brief  Convert polygonal meshes that consist of quads and/or triangles
///         into signed or unsigned distance field volumes.
///
/// @note   The signed distance field conversion requires a closed surface
///         but not necessarily a manifold surface. Supports surfaces with
///         self intersections and degenerate faces and is independent of
///         mesh surface normals / polygon orientation.
///
/// @author Mihai Alden

#ifndef OPENVDB_TOOLS_MESH_TO_VOLUME_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MESH_TO_VOLUME_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h> // for OPENVDB_HAS_CXX11
#include <openvdb/Types.h>
#include <openvdb/math/FiniteDifference.h> // for GodunovsNormSqrd
#include <openvdb/math/Proximity.h> // for closestPointOnTriangleToPoint
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/util/Util.h>

#include "ChangeBackground.h"
#include "Prune.h" // for pruneInactive and pruneLevelSet
#include "SignedFloodFill.h" // for signedFloodFillWithValues

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/partitioner.h>
#include <tbb/task_group.h>
#include <tbb/task_scheduler_init.h>

#include <boost/mpl/at.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/size.hpp>

#include <algorithm> // for std::sort()
#include <cmath> // for std::isfinite(), std::isnan()
#include <deque>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////


/// @brief Mesh to volume conversion flags
enum MeshToVolumeFlags {

    /// Switch from the default signed distance field conversion that classifies
    /// regions as either inside or outside the mesh boundary to a unsigned distance
    /// field conversion that only computes distance values. This conversion type
    /// does not require a closed watertight mesh.
    UNSIGNED_DISTANCE_FIELD = 0x1,

    /// Disable the cleanup step that removes voxels created by self intersecting
    /// portions of the mesh.
    DISABLE_INTERSECTING_VOXEL_REMOVAL = 0x2,

    /// Disable the distance renormalization step that smooths out bumps caused
    /// by self intersecting or overlapping portions of the mesh
    DISABLE_RENORMALIZATION = 0x4,

    /// Disable the cleanup step that removes active voxels that exceed the
    /// narrow band limits. (Only relevant for small limits)
    DISABLE_NARROW_BAND_TRIMMING = 0x8
};


/// @brief  Convert polygonal meshes that consist of quads and/or triangles into
///         signed or unsigned distance field volumes.
///
/// @note   Requires a closed surface but not necessarily a manifold surface.
///         Supports surfaces with self intersections and degenerate faces
///         and is independent of mesh surface normals.
///
/// @interface MeshDataAdapter
/// Expected interface for the MeshDataAdapter class
/// @code
/// struct MeshDataAdapter {
///   size_t polygonCount() const;        // Total number of polygons
///   size_t pointCount() const;          // Total number of points
///   size_t vertexCount(size_t n) const; // Vertex count for polygon n
///
///   // Return position pos in local grid index space for polygon n and vertex v
///   void getIndexSpacePoint(size_t n, size_t v, openvdb::Vec3d& pos) const;
/// };
/// @endcode
///
/// @param mesh               mesh data access class that conforms to the MeshDataAdapter
///                           interface
/// @param transform          world-to-index-space transform
/// @param exteriorBandWidth  exterior narrow band width in voxel units
/// @param interiorBandWidth  interior narrow band width in voxel units
///                           (set to std::numeric_limits<float>::max() to fill object
///                           interior with distance values)
/// @param flags              optional conversion flags defined in @c MeshToVolumeFlags
/// @param polygonIndexGrid   optional grid output that will contain the closest-polygon
///                           index for each voxel in the narrow band region
template <typename GridType, typename MeshDataAdapter>
inline typename GridType::Ptr
meshToVolume(
  const MeshDataAdapter& mesh,
  const math::Transform& transform,
  float exteriorBandWidth = 3.0f,
  float interiorBandWidth = 3.0f,
  int flags = 0,
  typename GridType::template ValueConverter<Int32>::Type * polygonIndexGrid = nullptr);


/// @brief  Convert polygonal meshes that consist of quads and/or triangles into
///         signed or unsigned distance field volumes.
///
/// @param interrupter        a callback to interrupt the conversion process that conforms
///                           to the util::NullInterrupter interface
/// @param mesh               mesh data access class that conforms to the MeshDataAdapter
///                           interface
/// @param transform          world-to-index-space transform
/// @param exteriorBandWidth  exterior narrow band width in voxel units
/// @param interiorBandWidth  interior narrow band width in voxel units (set this value to
///                           std::numeric_limits<float>::max() to fill interior regions
///                           with distance values)
/// @param flags              optional conversion flags defined in @c MeshToVolumeFlags
/// @param polygonIndexGrid   optional grid output that will contain the closest-polygon
///                           index for each voxel in the active narrow band region
template <typename GridType, typename MeshDataAdapter, typename Interrupter>
inline typename GridType::Ptr
meshToVolume(
    Interrupter& interrupter,
    const MeshDataAdapter& mesh,
    const math::Transform& transform,
    float exteriorBandWidth = 3.0f,
    float interiorBandWidth = 3.0f,
    int flags = 0,
    typename GridType::template ValueConverter<Int32>::Type * polygonIndexGrid = nullptr);


////////////////////////////////////////


/// @brief    Contiguous quad and triangle data adapter class
///
/// @details  PointType and PolygonType must provide element access
///           through the square brackets operator.
/// @details  Points are assumed to be in local grid index space.
/// @details  The PolygonType tuple can have either three or four components
///           this property must be specified in a static member variable
///           named @c size, similar to the math::Tuple class.
/// @details  A four component tuple can represent a quads or a triangle
///           if the fourth component set to @c util::INVALID_INDEX
template<typename PointType, typename PolygonType>
struct QuadAndTriangleDataAdapter {

    QuadAndTriangleDataAdapter(const std::vector<PointType>& points,
        const std::vector<PolygonType>& polygons)
        : mPointArray(points.empty() ? nullptr : &points[0])
        , mPointArraySize(points.size())
        , mPolygonArray(polygons.empty() ? nullptr : &polygons[0])
        , mPolygonArraySize(polygons.size())
    {
    }

    QuadAndTriangleDataAdapter(const PointType * pointArray, size_t pointArraySize,
        const PolygonType* polygonArray, size_t polygonArraySize)
        : mPointArray(pointArray)
        , mPointArraySize(pointArraySize)
        , mPolygonArray(polygonArray)
        , mPolygonArraySize(polygonArraySize)
    {
    }

    size_t polygonCount() const { return mPolygonArraySize; }
    size_t pointCount() const { return mPointArraySize; }

    /// @brief  Vertex count for polygon @a n
    size_t vertexCount(size_t n) const {
        return (PolygonType::size == 3 || mPolygonArray[n][3] == util::INVALID_IDX) ? 3 : 4;
    }

    /// @brief  Returns position @a pos in local grid index space
    ///         for polygon @a n and vertex @a v
    void getIndexSpacePoint(size_t n, size_t v, Vec3d& pos) const {
        const PointType& p = mPointArray[mPolygonArray[n][int(v)]];
        pos[0] = double(p[0]);
        pos[1] = double(p[1]);
        pos[2] = double(p[2]);
    }

private:
    PointType     const * const mPointArray;
    size_t                const mPointArraySize;
    PolygonType   const * const mPolygonArray;
    size_t                const mPolygonArraySize;
}; // struct QuadAndTriangleDataAdapter


////////////////////////////////////////


// Convenience functions for the mesh to volume converter that wrap stl containers.
//
// Note the meshToVolume() method declared above is more flexible and better suited
// for arbitrary data structures.


/// @brief Convert a triangle mesh to a level set volume.
///
/// @return a grid of type @c GridType containing a narrow-band level set
///         representation of the input mesh.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Requires a closed surface but not necessarily a manifold surface.
///         Supports surfaces with self intersections and degenerate faces
///         and is independent of mesh surface normals.
///
/// @param xform        transform for the output grid
/// @param points       list of world space point positions
/// @param triangles    triangle index list
/// @param halfWidth    half the width of the narrow band, in voxel units
template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH));

/// Adds support for a @a interrupter callback used to cancel the conversion.
template<typename GridType, typename Interrupter>
inline typename GridType::Ptr
meshToLevelSet(
    Interrupter& interrupter,
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH));


/// @brief Convert a quad mesh to a level set volume.
///
/// @return a grid of type @c GridType containing a narrow-band level set
///         representation of the input mesh.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Requires a closed surface but not necessarily a manifold surface.
///         Supports surfaces with self intersections and degenerate faces
///         and is independent of mesh surface normals.
///
/// @param xform        transform for the output grid
/// @param points       list of world space point positions
/// @param quads        quad index list
/// @param halfWidth    half the width of the narrow band, in voxel units
template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec4I>& quads,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH));

/// Adds support for a @a interrupter callback used to cancel the conversion.
template<typename GridType, typename Interrupter>
inline typename GridType::Ptr
meshToLevelSet(
    Interrupter& interrupter,
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec4I>& quads,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH));


/// @brief Convert a triangle and quad mesh to a level set volume.
///
/// @return a grid of type @c GridType containing a narrow-band level set
///         representation of the input mesh.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Requires a closed surface but not necessarily a manifold surface.
///         Supports surfaces with self intersections and degenerate faces
///         and is independent of mesh surface normals.
///
/// @param xform        transform for the output grid
/// @param points       list of world space point positions
/// @param triangles    triangle index list
/// @param quads        quad index list
/// @param halfWidth    half the width of the narrow band, in voxel units
template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH));

/// Adds support for a @a interrupter callback used to cancel the conversion.
template<typename GridType, typename Interrupter>
inline typename GridType::Ptr
meshToLevelSet(
    Interrupter& interrupter,
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH));


/// @brief Convert a triangle and quad mesh to a signed distance field
///        with an asymmetrical narrow band.
///
/// @return a grid of type @c GridType containing a narrow-band signed
///         distance field representation of the input mesh.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Requires a closed surface but not necessarily a manifold surface.
///         Supports surfaces with self intersections and degenerate faces
///         and is independent of mesh surface normals.
///
/// @param xform        transform for the output grid
/// @param points       list of world space point positions
/// @param triangles    triangle index list
/// @param quads        quad index list
/// @param exBandWidth  the exterior narrow-band width in voxel units
/// @param inBandWidth  the interior narrow-band width in voxel units
template<typename GridType>
inline typename GridType::Ptr
meshToSignedDistanceField(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float exBandWidth,
    float inBandWidth);

/// Adds support for a @a interrupter callback used to cancel the conversion.
template<typename GridType, typename Interrupter>
inline typename GridType::Ptr
meshToSignedDistanceField(
    Interrupter& interrupter,
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float exBandWidth,
    float inBandWidth);


/// @brief Convert a triangle and quad mesh to an unsigned distance field.
///
/// @return a grid of type @c GridType containing a narrow-band unsigned
///         distance field representation of the input mesh.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Does not requires a closed surface.
///
/// @param xform        transform for the output grid
/// @param points       list of world space point positions
/// @param triangles    triangle index list
/// @param quads        quad index list
/// @param bandWidth    the width of the narrow band, in voxel units
template<typename GridType>
inline typename GridType::Ptr
meshToUnsignedDistanceField(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float bandWidth);

/// Adds support for a @a interrupter callback used to cancel the conversion.
template<typename GridType, typename Interrupter>
inline typename GridType::Ptr
meshToUnsignedDistanceField(
    Interrupter& interrupter,
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float bandWidth);


////////////////////////////////////////


/// @brief  Return a grid of type @c GridType containing a narrow-band level set
///         representation of a box.
///
/// @param bbox       a bounding box in world units
/// @param xform      world-to-index-space transform
/// @param halfWidth  half the width of the narrow band, in voxel units
template<typename GridType, typename VecType>
inline typename GridType::Ptr
createLevelSetBox(const math::BBox<VecType>& bbox,
    const openvdb::math::Transform& xform,
    typename VecType::ValueType halfWidth = LEVEL_SET_HALF_WIDTH);


////////////////////////////////////////


/// @brief  Traces the exterior voxel boundary of closed objects in the input
///         volume @a tree. Exterior voxels are marked with a negative sign,
///         voxels with a value below @c 0.75 are left unchanged and act as
///         the boundary layer.
///
/// @note   Does not propagate sign information into tile regions.
template <typename FloatTreeT>
inline void
traceExteriorBoundaries(FloatTreeT& tree);


////////////////////////////////////////


/// @brief  Extracts and stores voxel edge intersection data from a mesh.
class MeshToVoxelEdgeData
{
public:

    //////////

    ///@brief Internal edge data type.
    struct EdgeData {
        EdgeData(float dist = 1.0)
            : mXDist(dist), mYDist(dist), mZDist(dist)
            , mXPrim(util::INVALID_IDX)
            , mYPrim(util::INVALID_IDX)
            , mZPrim(util::INVALID_IDX)
        {
        }

        //@{
        /// Required by several of the tree nodes
        /// @note These methods don't perform meaningful operations.
        bool operator< (const EdgeData&) const { return false; }
        bool operator> (const EdgeData&) const { return false; }
        template<class T> EdgeData operator+(const T&) const { return *this; }
        template<class T> EdgeData operator-(const T&) const { return *this; }
        EdgeData operator-() const { return *this; }
        //@}

        bool operator==(const EdgeData& rhs) const
        {
            return mXPrim == rhs.mXPrim && mYPrim == rhs.mYPrim && mZPrim == rhs.mZPrim;
        }

        float mXDist, mYDist, mZDist;
        Index32 mXPrim, mYPrim, mZPrim;
    };

    using TreeType = tree::Tree4<EdgeData, 5, 4, 3>::Type;
    using Accessor = tree::ValueAccessor<TreeType>;


    //////////


    MeshToVoxelEdgeData();


    /// @brief  Threaded method to extract voxel edge data, the closest
    ///         intersection point and corresponding primitive index,
    ///         from the given mesh.
    ///
    /// @param pointList    List of points in grid index space, preferably unique
    ///                     and shared by different polygons.
    /// @param polygonList  List of triangles and/or quads.
    void convert(const std::vector<Vec3s>& pointList, const std::vector<Vec4I>& polygonList);


    /// @brief  Returns intersection points with corresponding primitive
    ///         indices for the given @c ijk voxel.
    void getEdgeData(Accessor& acc, const Coord& ijk,
        std::vector<Vec3d>& points, std::vector<Index32>& primitives);

    /// @return An accessor of @c MeshToVoxelEdgeData::Accessor type that
    ///         provides random read access to the internal tree.
    Accessor getAccessor() { return Accessor(mTree); }

private:
    void operator=(const MeshToVoxelEdgeData&) {}
    TreeType mTree;
    class GenEdgeData;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


// Internal utility objects and implementation details

namespace mesh_to_volume_internal {

template<typename PointType>
struct TransformPoints {

    TransformPoints(const PointType* pointsIn, PointType* pointsOut,
        const math::Transform& xform)
        : mPointsIn(pointsIn), mPointsOut(pointsOut), mXform(&xform)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        Vec3d pos;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            const PointType& wsP = mPointsIn[n];
            pos[0] = double(wsP[0]);
            pos[1] = double(wsP[1]);
            pos[2] = double(wsP[2]);

            pos = mXform->worldToIndex(pos);

            PointType& isP = mPointsOut[n];
            isP[0] = typename PointType::value_type(pos[0]);
            isP[1] = typename PointType::value_type(pos[1]);
            isP[2] = typename PointType::value_type(pos[2]);
        }
    }

    PointType        const * const mPointsIn;
    PointType              * const mPointsOut;
    math::Transform  const * const mXform;
}; // TransformPoints


template<typename ValueType>
struct Tolerance
{
    static ValueType epsilon() { return ValueType(1e-7); }
    static ValueType minNarrowBandWidth() { return ValueType(1.0 + 1e-6); }
};


////////////////////////////////////////


template<typename TreeType>
class CombineLeafNodes
{
public:

    using Int32TreeType = typename TreeType::template ValueConverter<Int32>::Type;

    using LeafNodeType = typename TreeType::LeafNodeType;
    using Int32LeafNodeType = typename Int32TreeType::LeafNodeType;

    CombineLeafNodes(TreeType& lhsDistTree, Int32TreeType& lhsIdxTree,
        LeafNodeType ** rhsDistNodes, Int32LeafNodeType ** rhsIdxNodes)
        : mDistTree(&lhsDistTree)
        , mIdxTree(&lhsIdxTree)
        , mRhsDistNodes(rhsDistNodes)
        , mRhsIdxNodes(rhsIdxNodes)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        tree::ValueAccessor<TreeType> distAcc(*mDistTree);
        tree::ValueAccessor<Int32TreeType> idxAcc(*mIdxTree);

        using DistValueType = typename LeafNodeType::ValueType;
        using IndexValueType = typename Int32LeafNodeType::ValueType;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            const Coord& origin = mRhsDistNodes[n]->origin();

            LeafNodeType* lhsDistNode = distAcc.probeLeaf(origin);
            Int32LeafNodeType* lhsIdxNode = idxAcc.probeLeaf(origin);

            DistValueType* lhsDistData = lhsDistNode->buffer().data();
            IndexValueType* lhsIdxData = lhsIdxNode->buffer().data();

            const DistValueType* rhsDistData = mRhsDistNodes[n]->buffer().data();
            const IndexValueType* rhsIdxData = mRhsIdxNodes[n]->buffer().data();


            for (Index32 offset = 0; offset < LeafNodeType::SIZE; ++offset) {

                if (rhsIdxData[offset] != Int32(util::INVALID_IDX)) {

                    const DistValueType& lhsValue = lhsDistData[offset];
                    const DistValueType& rhsValue = rhsDistData[offset];

                    if (rhsValue < lhsValue) {
                        lhsDistNode->setValueOn(offset, rhsValue);
                        lhsIdxNode->setValueOn(offset, rhsIdxData[offset]);
                    } else if (math::isExactlyEqual(rhsValue, lhsValue)) {
                        lhsIdxNode->setValueOn(offset,
                            std::min(lhsIdxData[offset], rhsIdxData[offset]));
                    }
                }
            }

            delete mRhsDistNodes[n];
            delete mRhsIdxNodes[n];
        }
    }

private:

    TreeType * const mDistTree;
    Int32TreeType * const mIdxTree;

    LeafNodeType ** const mRhsDistNodes;
    Int32LeafNodeType ** const mRhsIdxNodes;
}; // class CombineLeafNodes


////////////////////////////////////////


template<typename TreeType>
struct StashOriginAndStoreOffset
{
    using LeafNodeType = typename TreeType::LeafNodeType;

    StashOriginAndStoreOffset(std::vector<LeafNodeType*>& nodes, Coord* coordinates)
        : mNodes(nodes.empty() ? nullptr : &nodes[0]), mCoordinates(coordinates)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            Coord& origin = const_cast<Coord&>(mNodes[n]->origin());
            mCoordinates[n] = origin;
            origin[0] = static_cast<int>(n);
        }
    }

    LeafNodeType ** const mNodes;
    Coord * const mCoordinates;
};


template<typename TreeType>
struct RestoreOrigin
{
    using LeafNodeType = typename TreeType::LeafNodeType;

    RestoreOrigin(std::vector<LeafNodeType*>& nodes, const Coord* coordinates)
        : mNodes(nodes.empty() ? nullptr : &nodes[0]), mCoordinates(coordinates)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            Coord& origin = const_cast<Coord&>(mNodes[n]->origin());
            origin[0] = mCoordinates[n][0];
        }
    }

    LeafNodeType         ** const mNodes;
    Coord           const * const mCoordinates;
};


template<typename TreeType>
class ComputeNodeConnectivity
{
public:
    using LeafNodeType = typename TreeType::LeafNodeType;

    ComputeNodeConnectivity(const TreeType& tree, const Coord* coordinates,
        size_t* offsets, size_t numNodes, const CoordBBox& bbox)
        : mTree(&tree)
        , mCoordinates(coordinates)
        , mOffsets(offsets)
        , mNumNodes(numNodes)
        , mBBox(bbox)
    {
    }

    ComputeNodeConnectivity(const ComputeNodeConnectivity&) = default;

    // Disallow assignment
    ComputeNodeConnectivity& operator=(const ComputeNodeConnectivity&) = delete;

    void operator()(const tbb::blocked_range<size_t>& range) const {

        size_t* offsetsNextX = mOffsets;
        size_t* offsetsPrevX = mOffsets + mNumNodes;
        size_t* offsetsNextY = mOffsets + mNumNodes * 2;
        size_t* offsetsPrevY = mOffsets + mNumNodes * 3;
        size_t* offsetsNextZ = mOffsets + mNumNodes * 4;
        size_t* offsetsPrevZ = mOffsets + mNumNodes * 5;

        tree::ValueAccessor<const TreeType> acc(*mTree);
        Coord ijk;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            const Coord& origin = mCoordinates[n];
            offsetsNextX[n] = findNeighbourNode(acc, origin, Coord(LeafNodeType::DIM, 0, 0));
            offsetsPrevX[n] = findNeighbourNode(acc, origin, Coord(-LeafNodeType::DIM, 0, 0));
            offsetsNextY[n] = findNeighbourNode(acc, origin, Coord(0, LeafNodeType::DIM, 0));
            offsetsPrevY[n] = findNeighbourNode(acc, origin, Coord(0, -LeafNodeType::DIM, 0));
            offsetsNextZ[n] = findNeighbourNode(acc, origin, Coord(0, 0, LeafNodeType::DIM));
            offsetsPrevZ[n] = findNeighbourNode(acc, origin, Coord(0, 0, -LeafNodeType::DIM));
        }
    }

    size_t findNeighbourNode(tree::ValueAccessor<const TreeType>& acc,
        const Coord& start, const Coord& step) const
    {
        Coord ijk = start + step;
        CoordBBox bbox(mBBox);

        while (bbox.isInside(ijk)) {
            const LeafNodeType* node = acc.probeConstLeaf(ijk);
            if (node) return static_cast<size_t>(node->origin()[0]);
            ijk += step;
        }

        return std::numeric_limits<size_t>::max();
    }


private:
    TreeType    const * const mTree;
    Coord       const * const mCoordinates;
    size_t            * const mOffsets;

    const size_t    mNumNodes;
    const CoordBBox mBBox;
}; // class ComputeNodeConnectivity


template<typename TreeType>
struct LeafNodeConnectivityTable
{
    enum { INVALID_OFFSET = std::numeric_limits<size_t>::max() };

    using LeafNodeType = typename TreeType::LeafNodeType;

    LeafNodeConnectivityTable(TreeType& tree)
    {
        mLeafNodes.reserve(tree.leafCount());
        tree.getNodes(mLeafNodes);

        if (mLeafNodes.empty()) return;

        CoordBBox bbox;
        tree.evalLeafBoundingBox(bbox);

        const tbb::blocked_range<size_t> range(0, mLeafNodes.size());

        // stash the leafnode origin coordinate and temporarily store the
        // linear offset in the origin.x variable.
        std::unique_ptr<Coord[]> coordinates{new Coord[mLeafNodes.size()]};
        tbb::parallel_for(range,
            StashOriginAndStoreOffset<TreeType>(mLeafNodes, coordinates.get()));

        // build the leafnode offset table
        mOffsets.reset(new size_t[mLeafNodes.size() * 6]);


        tbb::parallel_for(range, ComputeNodeConnectivity<TreeType>(
            tree, coordinates.get(), mOffsets.get(), mLeafNodes.size(), bbox));

        // restore the leafnode origin coordinate
        tbb::parallel_for(range, RestoreOrigin<TreeType>(mLeafNodes, coordinates.get()));
    }

    size_t size() const { return mLeafNodes.size(); }

    std::vector<LeafNodeType*>& nodes() { return mLeafNodes; }
    const std::vector<LeafNodeType*>& nodes() const { return mLeafNodes; }


    const size_t* offsetsNextX() const { return mOffsets.get(); }
    const size_t* offsetsPrevX() const { return mOffsets.get() + mLeafNodes.size(); }

    const size_t* offsetsNextY() const { return mOffsets.get() + mLeafNodes.size() * 2; }
    const size_t* offsetsPrevY() const { return mOffsets.get() + mLeafNodes.size() * 3; }

    const size_t* offsetsNextZ() const { return mOffsets.get() + mLeafNodes.size() * 4; }
    const size_t* offsetsPrevZ() const { return mOffsets.get() + mLeafNodes.size() * 5; }

private:
    std::vector<LeafNodeType*> mLeafNodes;
    std::unique_ptr<size_t[]> mOffsets;
}; // struct LeafNodeConnectivityTable


template<typename TreeType>
class SweepExteriorSign
{
public:

    enum Axis { X_AXIS = 0, Y_AXIS = 1, Z_AXIS = 2 };

    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using ConnectivityTable = LeafNodeConnectivityTable<TreeType>;

    SweepExteriorSign(Axis axis, const std::vector<size_t>& startNodeIndices,
        ConnectivityTable& connectivity)
        : mStartNodeIndices(startNodeIndices.empty() ? nullptr : &startNodeIndices[0])
        , mConnectivity(&connectivity)
        , mAxis(axis)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        std::vector<LeafNodeType*>& nodes = mConnectivity->nodes();

        // Z Axis
        size_t idxA = 0, idxB = 1;
        Index step = 1;

        const size_t* nextOffsets = mConnectivity->offsetsNextZ();
        const size_t* prevOffsets = mConnectivity->offsetsPrevZ();

        if (mAxis == Y_AXIS) {

            idxA = 0;
            idxB = 2;
            step = LeafNodeType::DIM;

            nextOffsets = mConnectivity->offsetsNextY();
            prevOffsets = mConnectivity->offsetsPrevY();

        } else if (mAxis == X_AXIS) {

            idxA = 1;
            idxB = 2;
            step = LeafNodeType::DIM * LeafNodeType::DIM;

            nextOffsets = mConnectivity->offsetsNextX();
            prevOffsets = mConnectivity->offsetsPrevX();
        }

        Coord ijk(0, 0, 0);

        int& a = ijk[idxA];
        int& b = ijk[idxB];

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            size_t startOffset = mStartNodeIndices[n];
            size_t lastOffset = startOffset;

            Index pos(0);

            for (a = 0; a < int(LeafNodeType::DIM); ++a) {
                for (b = 0; b < int(LeafNodeType::DIM); ++b) {

                    pos =  LeafNodeType::coordToOffset(ijk);
                    size_t offset = startOffset;

                    // sweep in +axis direction until a boundary voxel is hit.
                    while ( offset != ConnectivityTable::INVALID_OFFSET &&
                            traceVoxelLine(*nodes[offset], pos, step) ) {

                        lastOffset = offset;
                        offset = nextOffsets[offset];
                    }

                    // find last leafnode in +axis direction
                    offset = lastOffset;
                    while (offset != ConnectivityTable::INVALID_OFFSET) {
                        lastOffset = offset;
                        offset = nextOffsets[offset];
                    }

                    // sweep in -axis direction until a boundary voxel is hit.
                    offset = lastOffset;
                    pos += step * (LeafNodeType::DIM - 1);
                    while ( offset != ConnectivityTable::INVALID_OFFSET &&
                            traceVoxelLine(*nodes[offset], pos, -step)) {
                        offset = prevOffsets[offset];
                    }
                }
            }
        }
    }


    bool traceVoxelLine(LeafNodeType& node, Index pos, Index step) const {

        ValueType* data = node.buffer().data();

        bool isOutside = true;

        for (Index i = 0; i < LeafNodeType::DIM; ++i) {

            ValueType& dist = data[pos];

            if (dist < ValueType(0.0)) {
                isOutside = true;
            } else {
                // Boundary voxel check. (Voxel that intersects the surface)
                if (!(dist > ValueType(0.75))) isOutside = false;

                if (isOutside) dist = ValueType(-dist);
            }

            pos += step;
        }

        return isOutside;
    }


private:
    size_t              const * const mStartNodeIndices;
    ConnectivityTable         * const mConnectivity;

    const Axis mAxis;
}; // class SweepExteriorSign


template<typename LeafNodeType>
inline void
seedFill(LeafNodeType& node)
{
    using ValueType = typename LeafNodeType::ValueType;
    using Queue = std::deque<Index>;


    ValueType* data = node.buffer().data();

    // find seed points
    Queue seedPoints;
    for (Index pos = 0; pos < LeafNodeType::SIZE; ++pos) {
        if (data[pos] < 0.0) seedPoints.push_back(pos);
    }

    if (seedPoints.empty()) return;

    // clear sign information
    for (Queue::iterator it = seedPoints.begin(); it != seedPoints.end(); ++it) {
        ValueType& dist = data[*it];
        dist = -dist;
    }

    // flood fill

    Coord ijk(0, 0, 0);
    Index pos(0), nextPos(0);

    while (!seedPoints.empty()) {

        pos = seedPoints.back();
        seedPoints.pop_back();

        ValueType& dist = data[pos];

        if (!(dist < ValueType(0.0))) {

            dist = -dist; // flip sign

            ijk = LeafNodeType::offsetToLocalCoord(pos);

            if (ijk[0] != 0) { // i - 1, j, k
                nextPos = pos - LeafNodeType::DIM * LeafNodeType::DIM;
                if (data[nextPos] > ValueType(0.75)) seedPoints.push_back(nextPos);
            }

            if (ijk[0] != (LeafNodeType::DIM - 1)) { // i + 1, j, k
                nextPos = pos + LeafNodeType::DIM * LeafNodeType::DIM;
                if (data[nextPos] > ValueType(0.75)) seedPoints.push_back(nextPos);
            }

            if (ijk[1] != 0) { // i, j - 1, k
                nextPos = pos - LeafNodeType::DIM;
                if (data[nextPos] > ValueType(0.75)) seedPoints.push_back(nextPos);
            }

            if (ijk[1] != (LeafNodeType::DIM - 1)) { // i, j + 1, k
                nextPos = pos + LeafNodeType::DIM;
                if (data[nextPos] > ValueType(0.75)) seedPoints.push_back(nextPos);
            }

            if (ijk[2] != 0) { // i, j, k - 1
                nextPos = pos - 1;
                if (data[nextPos] > ValueType(0.75)) seedPoints.push_back(nextPos);
            }

            if (ijk[2] != (LeafNodeType::DIM - 1)) { // i, j, k + 1
                nextPos = pos + 1;
                if (data[nextPos] > ValueType(0.75)) seedPoints.push_back(nextPos);
            }
        }
    }
} // seedFill()


template<typename LeafNodeType>
inline bool
scanFill(LeafNodeType& node)
{
    bool updatedNode = false;

    using ValueType = typename LeafNodeType::ValueType;
    ValueType* data = node.buffer().data();

    Coord ijk(0, 0, 0);

    bool updatedSign = true;
    while (updatedSign) {

        updatedSign = false;

        for (Index pos = 0; pos < LeafNodeType::SIZE; ++pos) {

            ValueType& dist = data[pos];

            if (!(dist < ValueType(0.0)) && dist > ValueType(0.75)) {

                ijk = LeafNodeType::offsetToLocalCoord(pos);

                // i, j, k - 1
                if (ijk[2] != 0 && data[pos - 1] < ValueType(0.0)) {
                    updatedSign = true;
                    dist = ValueType(-dist);

                // i, j, k + 1
                } else if (ijk[2] != (LeafNodeType::DIM - 1) && data[pos + 1] < ValueType(0.0)) {
                    updatedSign = true;
                    dist = ValueType(-dist);

                // i, j - 1, k
                } else if (ijk[1] != 0 && data[pos - LeafNodeType::DIM] < ValueType(0.0)) {
                    updatedSign = true;
                    dist = ValueType(-dist);

                // i, j + 1, k
                } else if (ijk[1] != (LeafNodeType::DIM - 1)
                    && data[pos + LeafNodeType::DIM] < ValueType(0.0))
                {
                    updatedSign = true;
                    dist = ValueType(-dist);

                // i - 1, j, k
                } else if (ijk[0] != 0
                    && data[pos - LeafNodeType::DIM * LeafNodeType::DIM] < ValueType(0.0))
                {
                    updatedSign = true;
                    dist = ValueType(-dist);

                // i + 1, j, k
                } else if (ijk[0] != (LeafNodeType::DIM - 1)
                    && data[pos + LeafNodeType::DIM * LeafNodeType::DIM] < ValueType(0.0))
                {
                    updatedSign = true;
                    dist = ValueType(-dist);
                }
            }
        } // end value loop

        updatedNode |= updatedSign;
    } // end update loop

    return updatedNode;
} // scanFill()


template<typename TreeType>
class SeedFillExteriorSign
{
public:
    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;

    SeedFillExteriorSign(std::vector<LeafNodeType*>& nodes, bool* changedNodeMask)
        : mNodes(nodes.empty() ? nullptr : &nodes[0])
        , mChangedNodeMask(changedNodeMask)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            if (mChangedNodeMask[n]) {
                //seedFill(*mNodes[n]);
                mChangedNodeMask[n] = scanFill(*mNodes[n]);
            }
        }
    }

    LeafNodeType    ** const mNodes;
    bool             * const mChangedNodeMask;
};


template<typename ValueType>
struct FillArray
{
    FillArray(ValueType* array, const ValueType v) : mArray(array), mValue(v) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        const ValueType v = mValue;
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            mArray[n] = v;
        }
    }

    ValueType * const mArray;
    const ValueType mValue;
};


template<typename ValueType>
inline void
fillArray(ValueType* array, const ValueType val, const size_t length)
{
    const auto grainSize = std::max<size_t>(
        length / tbb::task_scheduler_init::default_num_threads(), 1024);
    const tbb::blocked_range<size_t> range(0, length, grainSize);
    tbb::parallel_for(range, FillArray<ValueType>(array, val), tbb::simple_partitioner());
}


template<typename TreeType>
class SyncVoxelMask
{
public:
    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;

    SyncVoxelMask(std::vector<LeafNodeType*>& nodes,
        const bool* changedNodeMask,  bool* changedVoxelMask)
        : mNodes(nodes.empty() ? nullptr : &nodes[0])
        , mChangedNodeMask(changedNodeMask)
        , mChangedVoxelMask(changedVoxelMask)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            if (mChangedNodeMask[n]) {
                bool* mask = &mChangedVoxelMask[n * LeafNodeType::SIZE];

                ValueType* data = mNodes[n]->buffer().data();

                for (Index pos = 0; pos < LeafNodeType::SIZE; ++pos) {
                    if (mask[pos]) {
                        data[pos] = ValueType(-data[pos]);
                        mask[pos] = false;
                    }
                }
            }
        }
    }

    LeafNodeType      ** const mNodes;
    bool         const * const mChangedNodeMask;
    bool          * const mChangedVoxelMask;
};


template<typename TreeType>
class SeedPoints
{
public:
    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using ConnectivityTable = LeafNodeConnectivityTable<TreeType>;

    SeedPoints(ConnectivityTable& connectivity,
        bool* changedNodeMask, bool* nodeMask, bool* changedVoxelMask)
        : mConnectivity(&connectivity)
        , mChangedNodeMask(changedNodeMask)
        , mNodeMask(nodeMask)
        , mChangedVoxelMask(changedVoxelMask)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            if (!mChangedNodeMask[n]) {

                bool changedValue = false;

                changedValue |= processZ(n, /*firstFace=*/true);
                changedValue |= processZ(n, /*firstFace=*/false);

                changedValue |= processY(n, /*firstFace=*/true);
                changedValue |= processY(n, /*firstFace=*/false);

                changedValue |= processX(n, /*firstFace=*/true);
                changedValue |= processX(n, /*firstFace=*/false);

                mNodeMask[n] = changedValue;
            }
        }
    }


    bool processZ(const size_t n, bool firstFace) const
    {
        const size_t offset =
            firstFace ? mConnectivity->offsetsPrevZ()[n] : mConnectivity->offsetsNextZ()[n];
        if (offset != ConnectivityTable::INVALID_OFFSET && mChangedNodeMask[offset]) {

            bool* mask = &mChangedVoxelMask[n * LeafNodeType::SIZE];

            const ValueType* lhsData = mConnectivity->nodes()[n]->buffer().data();
            const ValueType* rhsData = mConnectivity->nodes()[offset]->buffer().data();

            const Index lastOffset = LeafNodeType::DIM - 1;
            const Index lhsOffset =
                firstFace ? 0 : lastOffset, rhsOffset = firstFace ? lastOffset : 0;

            Index tmpPos(0), pos(0);
            bool changedValue = false;

            for (Index x = 0; x < LeafNodeType::DIM; ++x) {
                tmpPos = x << (2 * LeafNodeType::LOG2DIM);
                for (Index y = 0; y < LeafNodeType::DIM; ++y) {
                    pos = tmpPos + (y << LeafNodeType::LOG2DIM);

                    if (lhsData[pos + lhsOffset] > ValueType(0.75)) {
                        if (rhsData[pos + rhsOffset] < ValueType(0.0)) {
                            changedValue = true;
                            mask[pos + lhsOffset] = true;
                        }
                    }
                }
            }

            return changedValue;
        }

        return false;
    }

    bool processY(const size_t n, bool firstFace) const
    {
        const size_t offset =
            firstFace ? mConnectivity->offsetsPrevY()[n] : mConnectivity->offsetsNextY()[n];
        if (offset != ConnectivityTable::INVALID_OFFSET && mChangedNodeMask[offset]) {

            bool* mask = &mChangedVoxelMask[n * LeafNodeType::SIZE];

            const ValueType* lhsData = mConnectivity->nodes()[n]->buffer().data();
            const ValueType* rhsData = mConnectivity->nodes()[offset]->buffer().data();

            const Index lastOffset = LeafNodeType::DIM * (LeafNodeType::DIM - 1);
            const Index lhsOffset =
                firstFace ? 0 : lastOffset, rhsOffset = firstFace ? lastOffset : 0;

            Index tmpPos(0), pos(0);
            bool changedValue = false;

            for (Index x = 0; x < LeafNodeType::DIM; ++x) {
                tmpPos = x << (2 * LeafNodeType::LOG2DIM);
                for (Index z = 0; z < LeafNodeType::DIM; ++z) {
                    pos = tmpPos + z;

                    if (lhsData[pos + lhsOffset] > ValueType(0.75)) {
                        if (rhsData[pos + rhsOffset] < ValueType(0.0)) {
                            changedValue = true;
                            mask[pos + lhsOffset] = true;
                        }
                    }
                }
            }

            return changedValue;
        }

        return false;
    }

    bool processX(const size_t n, bool firstFace) const
    {
        const size_t offset =
            firstFace ? mConnectivity->offsetsPrevX()[n] : mConnectivity->offsetsNextX()[n];
        if (offset != ConnectivityTable::INVALID_OFFSET && mChangedNodeMask[offset]) {

            bool* mask = &mChangedVoxelMask[n * LeafNodeType::SIZE];

            const ValueType* lhsData = mConnectivity->nodes()[n]->buffer().data();
            const ValueType* rhsData = mConnectivity->nodes()[offset]->buffer().data();

            const Index lastOffset =  LeafNodeType::DIM * LeafNodeType::DIM * (LeafNodeType::DIM-1);
            const Index lhsOffset =
                firstFace ? 0 : lastOffset, rhsOffset = firstFace ? lastOffset : 0;

            Index tmpPos(0), pos(0);
            bool changedValue = false;

            for (Index y = 0; y < LeafNodeType::DIM; ++y) {
                tmpPos = y << LeafNodeType::LOG2DIM;
                for (Index z = 0; z < LeafNodeType::DIM; ++z) {
                    pos = tmpPos + z;

                    if (lhsData[pos + lhsOffset] > ValueType(0.75)) {
                        if (rhsData[pos + rhsOffset] < ValueType(0.0)) {
                            changedValue = true;
                            mask[pos + lhsOffset] = true;
                        }
                    }
                }
            }

            return changedValue;
        }

        return false;
    }

    ConnectivityTable   * const mConnectivity;
    bool                * const mChangedNodeMask;
    bool                * const mNodeMask;
    bool                * const mChangedVoxelMask;
};


////////////////////////////////////////

template<typename TreeType, typename MeshDataAdapter>
struct ComputeIntersectingVoxelSign
{
    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using Int32TreeType = typename TreeType::template ValueConverter<Int32>::Type;
    using Int32LeafNodeType = typename Int32TreeType::LeafNodeType;

    using PointArray = std::unique_ptr<Vec3d[]>;
    using MaskArray = std::unique_ptr<bool[]>;
    using LocalData = std::pair<PointArray, MaskArray>;
    using LocalDataTable = tbb::enumerable_thread_specific<LocalData>;

    ComputeIntersectingVoxelSign(
        std::vector<LeafNodeType*>& distNodes,
        const TreeType& distTree,
        const Int32TreeType& indexTree,
        const MeshDataAdapter& mesh)
        : mDistNodes(distNodes.empty() ? nullptr : &distNodes[0])
        , mDistTree(&distTree)
        , mIndexTree(&indexTree)
        , mMesh(&mesh)
        , mLocalDataTable(new LocalDataTable())
    {
    }


    void operator()(const tbb::blocked_range<size_t>& range) const {

        tree::ValueAccessor<const TreeType> distAcc(*mDistTree);
        tree::ValueAccessor<const Int32TreeType> idxAcc(*mIndexTree);

        ValueType nval;
        CoordBBox bbox;
        Index xPos(0), yPos(0);
        Coord ijk, nijk, nodeMin, nodeMax;
        Vec3d cp, xyz, nxyz, dir1, dir2;

        LocalData& localData = mLocalDataTable->local();

        PointArray& points = localData.first;
        if (!points) points.reset(new Vec3d[LeafNodeType::SIZE * 2]);

        MaskArray& mask = localData.second;
        if (!mask) mask.reset(new bool[LeafNodeType::SIZE]);


        typename LeafNodeType::ValueOnCIter it;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            LeafNodeType& node = *mDistNodes[n];
            ValueType* data = node.buffer().data();

            const Int32LeafNodeType* idxNode = idxAcc.probeConstLeaf(node.origin());
            const Int32* idxData = idxNode->buffer().data();

            nodeMin = node.origin();
            nodeMax = nodeMin.offsetBy(LeafNodeType::DIM - 1);

            // reset computed voxel mask.
            memset(mask.get(), 0, sizeof(bool) * LeafNodeType::SIZE);

            for (it = node.cbeginValueOn(); it; ++it) {
                Index pos = it.pos();

                ValueType& dist = data[pos];
                if (dist < 0.0 || dist > 0.75) continue;

                ijk = node.offsetToGlobalCoord(pos);

                xyz[0] = double(ijk[0]);
                xyz[1] = double(ijk[1]);
                xyz[2] = double(ijk[2]);


                bbox.min() = Coord::maxComponent(ijk.offsetBy(-1), nodeMin);
                bbox.max() = Coord::minComponent(ijk.offsetBy(1), nodeMax);

                bool flipSign = false;

                for (nijk[0] = bbox.min()[0]; nijk[0] <= bbox.max()[0] && !flipSign; ++nijk[0]) {
                    xPos = (nijk[0] & (LeafNodeType::DIM - 1u)) << (2 * LeafNodeType::LOG2DIM);
                    for (nijk[1]=bbox.min()[1]; nijk[1] <= bbox.max()[1] && !flipSign; ++nijk[1]) {
                        yPos = xPos + ((nijk[1] & (LeafNodeType::DIM-1u)) << LeafNodeType::LOG2DIM);
                        for (nijk[2] = bbox.min()[2]; nijk[2] <= bbox.max()[2]; ++nijk[2]) {
                            pos = yPos + (nijk[2] & (LeafNodeType::DIM - 1u));

                            const Int32& polyIdx = idxData[pos];

                            if (polyIdx == Int32(util::INVALID_IDX) || !(data[pos] < -0.75))
                                continue;

                            const Index pointIndex = pos * 2;

                            if (!mask[pos]) {

                                mask[pos] = true;

                                nxyz[0] = double(nijk[0]);
                                nxyz[1] = double(nijk[1]);
                                nxyz[2] = double(nijk[2]);

                                Vec3d& point = points[pointIndex];

                                point = closestPoint(nxyz, polyIdx);

                                Vec3d& direction = points[pointIndex + 1];
                                direction = nxyz - point;
                                direction.normalize();
                            }

                            dir1 = xyz - points[pointIndex];
                            dir1.normalize();

                            if (points[pointIndex + 1].dot(dir1) > 0.0) {
                                flipSign = true;
                                break;
                            }
                        }
                    }
                }

                if (flipSign) {
                    dist = -dist;
                } else {
                    for (Int32 m = 0; m < 26; ++m) {
                        nijk = ijk + util::COORD_OFFSETS[m];

                        if (!bbox.isInside(nijk) && distAcc.probeValue(nijk, nval) && nval<-0.75) {
                            nxyz[0] = double(nijk[0]);
                            nxyz[1] = double(nijk[1]);
                            nxyz[2] = double(nijk[2]);

                            cp = closestPoint(nxyz, idxAcc.getValue(nijk));

                            dir1 = xyz - cp;
                            dir1.normalize();

                            dir2 = nxyz - cp;
                            dir2.normalize();

                            if (dir2.dot(dir1) > 0.0) {
                                dist = -dist;
                                break;
                            }
                        }
                    }
                }

            } // active voxel loop
        } // leaf node loop
    }

private:

    Vec3d closestPoint(const Vec3d& center, Int32 polyIdx) const
    {
        Vec3d a, b, c, cp, uvw;

        const size_t polygon = size_t(polyIdx);
        mMesh->getIndexSpacePoint(polygon, 0, a);
        mMesh->getIndexSpacePoint(polygon, 1, b);
        mMesh->getIndexSpacePoint(polygon, 2, c);

        cp = closestPointOnTriangleToPoint(a, c, b, center, uvw);

        if (4 == mMesh->vertexCount(polygon)) {

            mMesh->getIndexSpacePoint(polygon, 3, b);

            c = closestPointOnTriangleToPoint(a, b, c, center, uvw);

            if ((center - c).lengthSqr() < (center - cp).lengthSqr()) {
                cp = c;
            }
        }

        return cp;
    }


    LeafNodeType         ** const mDistNodes;
    TreeType        const * const mDistTree;
    Int32TreeType   const * const mIndexTree;
    MeshDataAdapter const * const mMesh;

    SharedPtr<LocalDataTable> mLocalDataTable;
}; // ComputeIntersectingVoxelSign


////////////////////////////////////////


template<typename LeafNodeType>
inline void
maskNodeInternalNeighbours(const Index pos, bool (&mask)[26])
{
    using NodeT = LeafNodeType;

    const Coord ijk = NodeT::offsetToLocalCoord(pos);

    // Face adjacent neighbours
    // i+1, j, k
    mask[0] = ijk[0] != (NodeT::DIM - 1);
    // i-1, j, k
    mask[1] = ijk[0] != 0;
    // i, j+1, k
    mask[2] = ijk[1] != (NodeT::DIM - 1);
    // i, j-1, k
    mask[3] = ijk[1] != 0;
    // i, j, k+1
    mask[4] = ijk[2] != (NodeT::DIM - 1);
    // i, j, k-1
    mask[5] = ijk[2] != 0;

    // Edge adjacent neighbour
    // i+1, j, k-1
    mask[6] = mask[0] && mask[5];
    // i-1, j, k-1
    mask[7] = mask[1] && mask[5];
    // i+1, j, k+1
    mask[8] = mask[0] && mask[4];
    // i-1, j, k+1
    mask[9] = mask[1] && mask[4];
    // i+1, j+1, k
    mask[10] = mask[0] && mask[2];
    // i-1, j+1, k
    mask[11] = mask[1] && mask[2];
    // i+1, j-1, k
    mask[12] = mask[0] && mask[3];
    // i-1, j-1, k
    mask[13] = mask[1] && mask[3];
    // i, j-1, k+1
    mask[14] = mask[3] && mask[4];
    // i, j-1, k-1
    mask[15] = mask[3] && mask[5];
    // i, j+1, k+1
    mask[16] = mask[2] && mask[4];
    // i, j+1, k-1
    mask[17] = mask[2] && mask[5];

    // Corner adjacent neighbours
    // i-1, j-1, k-1
    mask[18] = mask[1] && mask[3] && mask[5];
    // i-1, j-1, k+1
    mask[19] = mask[1] && mask[3] && mask[4];
    // i+1, j-1, k+1
    mask[20] = mask[0] && mask[3] && mask[4];
    // i+1, j-1, k-1
    mask[21] = mask[0] && mask[3] && mask[5];
    // i-1, j+1, k-1
    mask[22] = mask[1] && mask[2] && mask[5];
    // i-1, j+1, k+1
    mask[23] = mask[1] && mask[2] && mask[4];
    // i+1, j+1, k+1
    mask[24] = mask[0] && mask[2] && mask[4];
    // i+1, j+1, k-1
    mask[25] = mask[0] && mask[2] && mask[5];
}


template<typename Compare, typename LeafNodeType>
inline bool
checkNeighbours(const Index pos, const typename LeafNodeType::ValueType * data, bool (&mask)[26])
{
    using NodeT = LeafNodeType;

    // i, j, k - 1
    if (mask[5] && Compare::check(data[pos - 1]))                                         return true;
    // i, j, k + 1
    if (mask[4] && Compare::check(data[pos + 1]))                                         return true;
    // i, j - 1, k
    if (mask[3] && Compare::check(data[pos - NodeT::DIM]))                                return true;
    // i, j + 1, k
    if (mask[2] && Compare::check(data[pos + NodeT::DIM]))                                return true;
    // i - 1, j, k
    if (mask[1] && Compare::check(data[pos - NodeT::DIM * NodeT::DIM]))                   return true;
    // i + 1, j, k
    if (mask[0] && Compare::check(data[pos + NodeT::DIM * NodeT::DIM]))                   return true;
    // i+1, j, k-1
    if (mask[6] && Compare::check(data[pos + NodeT::DIM * NodeT::DIM]))                   return true;
    // i-1, j, k-1
    if (mask[7] && Compare::check(data[pos - NodeT::DIM * NodeT::DIM - 1]))               return true;
    // i+1, j, k+1
    if (mask[8] && Compare::check(data[pos + NodeT::DIM * NodeT::DIM + 1]))               return true;
    // i-1, j, k+1
    if (mask[9] && Compare::check(data[pos - NodeT::DIM * NodeT::DIM + 1]))               return true;
    // i+1, j+1, k
    if (mask[10] && Compare::check(data[pos + NodeT::DIM * NodeT::DIM + NodeT::DIM]))     return true;
    // i-1, j+1, k
    if (mask[11] && Compare::check(data[pos - NodeT::DIM * NodeT::DIM + NodeT::DIM]))     return true;
    // i+1, j-1, k
    if (mask[12] && Compare::check(data[pos + NodeT::DIM * NodeT::DIM - NodeT::DIM]))     return true;
    // i-1, j-1, k
    if (mask[13] && Compare::check(data[pos - NodeT::DIM * NodeT::DIM - NodeT::DIM]))     return true;
    // i, j-1, k+1
    if (mask[14] && Compare::check(data[pos - NodeT::DIM + 1]))                           return true;
    // i, j-1, k-1
    if (mask[15] && Compare::check(data[pos - NodeT::DIM - 1]))                           return true;
    // i, j+1, k+1
    if (mask[16] && Compare::check(data[pos + NodeT::DIM + 1]))                           return true;
    // i, j+1, k-1
    if (mask[17] && Compare::check(data[pos + NodeT::DIM - 1]))                           return true;
    // i-1, j-1, k-1
    if (mask[18] && Compare::check(data[pos - NodeT::DIM * NodeT::DIM - NodeT::DIM - 1])) return true;
    // i-1, j-1, k+1
    if (mask[19] && Compare::check(data[pos - NodeT::DIM * NodeT::DIM - NodeT::DIM + 1])) return true;
    // i+1, j-1, k+1
    if (mask[20] && Compare::check(data[pos + NodeT::DIM * NodeT::DIM - NodeT::DIM + 1])) return true;
    // i+1, j-1, k-1
    if (mask[21] && Compare::check(data[pos + NodeT::DIM * NodeT::DIM - NodeT::DIM - 1])) return true;
    // i-1, j+1, k-1
    if (mask[22] && Compare::check(data[pos - NodeT::DIM * NodeT::DIM + NodeT::DIM - 1])) return true;
    // i-1, j+1, k+1
    if (mask[23] && Compare::check(data[pos - NodeT::DIM * NodeT::DIM + NodeT::DIM + 1])) return true;
    // i+1, j+1, k+1
    if (mask[24] && Compare::check(data[pos + NodeT::DIM * NodeT::DIM + NodeT::DIM + 1])) return true;
    // i+1, j+1, k-1
    if (mask[25] && Compare::check(data[pos + NodeT::DIM * NodeT::DIM + NodeT::DIM - 1])) return true;

    return false;
}


template<typename Compare, typename AccessorType>
inline bool
checkNeighbours(const Coord& ijk, AccessorType& acc, bool (&mask)[26])
{
    for (Int32 m = 0; m < 26; ++m) {
        if (!mask[m] && Compare::check(acc.getValue(ijk + util::COORD_OFFSETS[m]))) {
            return true;
        }
    }

    return false;
}


template<typename TreeType>
struct ValidateIntersectingVoxels
{
    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;

    struct IsNegative { static bool check(const ValueType v) { return v < ValueType(0.0); } };

    ValidateIntersectingVoxels(TreeType& tree, std::vector<LeafNodeType*>& nodes)
        : mTree(&tree)
        , mNodes(nodes.empty() ? nullptr : &nodes[0])
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        tree::ValueAccessor<const TreeType> acc(*mTree);
        bool neighbourMask[26];

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            LeafNodeType& node = *mNodes[n];
            ValueType* data = node.buffer().data();

            typename LeafNodeType::ValueOnCIter it;
            for (it = node.cbeginValueOn(); it; ++it) {

                const Index pos = it.pos();

                ValueType& dist = data[pos];
                if (dist < 0.0 || dist > 0.75) continue;

                // Mask node internal neighbours
                maskNodeInternalNeighbours<LeafNodeType>(pos, neighbourMask);

                const bool hasNegativeNeighbour =
                    checkNeighbours<IsNegative, LeafNodeType>(pos, data, neighbourMask) ||
                    checkNeighbours<IsNegative>(node.offsetToGlobalCoord(pos), acc, neighbourMask);

                if (!hasNegativeNeighbour) {
                    // push over boundary voxel distance
                    dist = ValueType(0.75) + Tolerance<ValueType>::epsilon();
                }
            }
        }
    }

    TreeType         * const mTree;
    LeafNodeType    ** const mNodes;
}; // ValidateIntersectingVoxels


template<typename TreeType>
struct RemoveSelfIntersectingSurface
{
    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using Int32TreeType = typename TreeType::template ValueConverter<Int32>::Type;

    struct Comp { static bool check(const ValueType v) { return !(v > ValueType(0.75)); } };

    RemoveSelfIntersectingSurface(std::vector<LeafNodeType*>& nodes,
        TreeType& distTree, Int32TreeType& indexTree)
        : mNodes(nodes.empty() ? nullptr : &nodes[0])
        , mDistTree(&distTree)
        , mIndexTree(&indexTree)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        tree::ValueAccessor<const TreeType> distAcc(*mDistTree);
        tree::ValueAccessor<Int32TreeType> idxAcc(*mIndexTree);
        bool neighbourMask[26];

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            LeafNodeType& distNode = *mNodes[n];
            ValueType* data = distNode.buffer().data();

            typename Int32TreeType::LeafNodeType* idxNode =
                idxAcc.probeLeaf(distNode.origin());

            typename LeafNodeType::ValueOnCIter it;
            for (it = distNode.cbeginValueOn(); it; ++it) {

                const Index pos = it.pos();

                if (!(data[pos] > 0.75)) continue;

                // Mask node internal neighbours
                maskNodeInternalNeighbours<LeafNodeType>(pos, neighbourMask);

                const bool hasBoundaryNeighbour =
                    checkNeighbours<Comp, LeafNodeType>(pos, data, neighbourMask) ||
                    checkNeighbours<Comp>(distNode.offsetToGlobalCoord(pos),distAcc,neighbourMask);

                if (!hasBoundaryNeighbour) {
                    distNode.setValueOff(pos);
                    idxNode->setValueOff(pos);
                }
            }
        }
    }

    LeafNodeType   * * const mNodes;
    TreeType         * const mDistTree;
    Int32TreeType    * const mIndexTree;
}; // RemoveSelfIntersectingSurface


////////////////////////////////////////


template<typename NodeType>
struct ReleaseChildNodes
{
    ReleaseChildNodes(NodeType ** nodes) : mNodes(nodes) {}

    void operator()(const tbb::blocked_range<size_t>& range) const {

        using NodeMaskType = typename NodeType::NodeMaskType;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            const_cast<NodeMaskType&>(mNodes[n]->getChildMask()).setOff();
        }
    }

    NodeType ** const mNodes;
};


template<typename TreeType>
inline void
releaseLeafNodes(TreeType& tree)
{
    using RootNodeType = typename TreeType::RootNodeType;
    using NodeChainType = typename RootNodeType::NodeChainType;
    using InternalNodeType = typename boost::mpl::at<NodeChainType, boost::mpl::int_<1> >::type;

    std::vector<InternalNodeType*> nodes;
    tree.getNodes(nodes);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, nodes.size()),
        ReleaseChildNodes<InternalNodeType>(nodes.empty() ? nullptr : &nodes[0]));
}


template<typename TreeType>
struct StealUniqueLeafNodes
{
    using LeafNodeType = typename TreeType::LeafNodeType;

    StealUniqueLeafNodes(TreeType& lhsTree, TreeType& rhsTree,
        std::vector<LeafNodeType*>& overlappingNodes)
        : mLhsTree(&lhsTree)
        , mRhsTree(&rhsTree)
        , mNodes(&overlappingNodes)
    {
    }

    void operator()() const {

        std::vector<LeafNodeType*> rhsLeafNodes;

        rhsLeafNodes.reserve(mRhsTree->leafCount());
        //mRhsTree->getNodes(rhsLeafNodes);
        //releaseLeafNodes(*mRhsTree);
        mRhsTree->stealNodes(rhsLeafNodes);

        tree::ValueAccessor<TreeType> acc(*mLhsTree);

        for (size_t n = 0, N = rhsLeafNodes.size(); n < N; ++n) {
            if (!acc.probeLeaf(rhsLeafNodes[n]->origin())) {
                acc.addLeaf(rhsLeafNodes[n]);
            } else {
                mNodes->push_back(rhsLeafNodes[n]);
            }
        }
    }

private:
    TreeType * const mLhsTree;
    TreeType * const mRhsTree;
    std::vector<LeafNodeType*> * const mNodes;
};


template<typename DistTreeType, typename IndexTreeType>
inline void
combineData(DistTreeType& lhsDist, IndexTreeType& lhsIdx,
    DistTreeType& rhsDist, IndexTreeType& rhsIdx)
{
    using DistLeafNodeType = typename DistTreeType::LeafNodeType;
    using IndexLeafNodeType = typename IndexTreeType::LeafNodeType;

    std::vector<DistLeafNodeType*>  overlappingDistNodes;
    std::vector<IndexLeafNodeType*> overlappingIdxNodes;

    // Steal unique leafnodes
    tbb::task_group tasks;
    tasks.run(StealUniqueLeafNodes<DistTreeType>(lhsDist, rhsDist, overlappingDistNodes));
    tasks.run(StealUniqueLeafNodes<IndexTreeType>(lhsIdx, rhsIdx, overlappingIdxNodes));
    tasks.wait();

    // Combine overlapping leaf nodes
    if (!overlappingDistNodes.empty() && !overlappingIdxNodes.empty()) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, overlappingDistNodes.size()),
            CombineLeafNodes<DistTreeType>(lhsDist, lhsIdx,
                &overlappingDistNodes[0], &overlappingIdxNodes[0]));
    }
}

/// @brief TBB body object to voxelize a mesh of triangles and/or quads into a collection
/// of VDB grids, namely a squared distance grid, a closest primitive grid and an
/// intersecting voxels grid (masks the mesh intersecting voxels)
/// @note Only the leaf nodes that intersect the mesh are allocated, and only voxels in
/// a narrow band (of two to three voxels in proximity to the mesh's surface) are activated.
/// They are populated with distance values and primitive indices.
template<typename TreeType>
struct VoxelizationData {

    using Ptr = std::unique_ptr<VoxelizationData>;
    using ValueType = typename TreeType::ValueType;

    using Int32TreeType = typename TreeType::template ValueConverter<Int32>::Type;
    using UCharTreeType = typename TreeType::template ValueConverter<unsigned char>::Type;

    using FloatTreeAcc = tree::ValueAccessor<TreeType>;
    using Int32TreeAcc = tree::ValueAccessor<Int32TreeType>;
    using UCharTreeAcc = tree::ValueAccessor<UCharTreeType>;


    VoxelizationData()
        : distTree(std::numeric_limits<ValueType>::max())
        , distAcc(distTree)
        , indexTree(Int32(util::INVALID_IDX))
        , indexAcc(indexTree)
        , primIdTree(MaxPrimId)
        , primIdAcc(primIdTree)
        , mPrimCount(0)
    {
    }

    TreeType        distTree;
    FloatTreeAcc    distAcc;

    Int32TreeType   indexTree;
    Int32TreeAcc    indexAcc;

    UCharTreeType   primIdTree;
    UCharTreeAcc    primIdAcc;

    unsigned char getNewPrimId() {

        if (mPrimCount == MaxPrimId || primIdTree.leafCount() > 1000) {
            mPrimCount = 0;
            primIdTree.clear();
        }

        return mPrimCount++;
    }

private:

    enum { MaxPrimId = 100 };

    unsigned char mPrimCount;
};


template<typename TreeType, typename MeshDataAdapter, typename Interrupter = util::NullInterrupter>
class VoxelizePolygons
{
public:

    using VoxelizationDataType = VoxelizationData<TreeType>;
    using DataTable = tbb::enumerable_thread_specific<typename VoxelizationDataType::Ptr>;

    VoxelizePolygons(DataTable& dataTable,
        const MeshDataAdapter& mesh,
        Interrupter* interrupter = nullptr)
        : mDataTable(&dataTable)
        , mMesh(&mesh)
        , mInterrupter(interrupter)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        typename VoxelizationDataType::Ptr& dataPtr = mDataTable->local();
        if (!dataPtr) dataPtr.reset(new VoxelizationDataType());

        Triangle prim;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            if (this->wasInterrupted()) {
                tbb::task::self().cancel_group_execution();
                break;
            }

            const size_t numVerts = mMesh->vertexCount(n);

            // rasterize triangles and quads.
            if (numVerts == 3 || numVerts == 4) {

                prim.index = Int32(n);

                mMesh->getIndexSpacePoint(n, 0, prim.a);
                mMesh->getIndexSpacePoint(n, 1, prim.b);
                mMesh->getIndexSpacePoint(n, 2, prim.c);

                evalTriangle(prim, *dataPtr);

                if (numVerts == 4) {
                    mMesh->getIndexSpacePoint(n, 3, prim.b);
                    evalTriangle(prim, *dataPtr);
                }
            }
        }
    }

private:

    bool wasInterrupted() const { return mInterrupter && mInterrupter->wasInterrupted(); }

    struct Triangle { Vec3d a, b, c; Int32 index; };

    struct SubTask
    {
        enum { POLYGON_LIMIT = 1000 };

        SubTask(const Triangle& prim, DataTable& dataTable,
            int subdivisionCount, size_t polygonCount,
            Interrupter* interrupter = nullptr)
            : mLocalDataTable(&dataTable)
            , mPrim(prim)
            , mSubdivisionCount(subdivisionCount)
            , mPolygonCount(polygonCount)
            , mInterrupter(interrupter)
        {
        }

        void operator()() const
        {
            if (mSubdivisionCount <= 0 || mPolygonCount >= POLYGON_LIMIT) {

                typename VoxelizationDataType::Ptr& dataPtr = mLocalDataTable->local();
                if (!dataPtr) dataPtr.reset(new VoxelizationDataType());

                voxelizeTriangle(mPrim, *dataPtr);

            } else if (!(mInterrupter && mInterrupter->wasInterrupted())) {
                spawnTasks(mPrim, *mLocalDataTable, mSubdivisionCount, mPolygonCount, mInterrupter);
            }
        }

        DataTable   * const mLocalDataTable;
        Triangle      const mPrim;
        int           const mSubdivisionCount;
        size_t        const mPolygonCount;
        Interrupter * const mInterrupter;
    }; // struct SubTask

    inline static int evalSubdivisionCount(const Triangle& prim)
    {
        const double ax = prim.a[0], bx = prim.b[0], cx = prim.c[0];
        const double dx = std::max(ax, std::max(bx, cx)) - std::min(ax, std::min(bx, cx));

        const double ay = prim.a[1], by = prim.b[1], cy = prim.c[1];
        const double dy = std::max(ay, std::max(by, cy)) - std::min(ay, std::min(by, cy));

        const double az = prim.a[2], bz = prim.b[2], cz = prim.c[2];
        const double dz = std::max(az, std::max(bz, cz)) - std::min(az, std::min(bz, cz));

        return int(std::max(dx, std::max(dy, dz)) / double(TreeType::LeafNodeType::DIM * 2));
    }

    void evalTriangle(const Triangle& prim, VoxelizationDataType& data) const
    {
        const size_t polygonCount = mMesh->polygonCount();
        const int subdivisionCount =
            polygonCount < SubTask::POLYGON_LIMIT ? evalSubdivisionCount(prim) : 0;

        if (subdivisionCount <= 0) {
            voxelizeTriangle(prim, data);
        } else {
            spawnTasks(prim, *mDataTable, subdivisionCount, polygonCount, mInterrupter);
        }
    }

    static void spawnTasks(
        const Triangle& mainPrim,
        DataTable& dataTable,
        int subdivisionCount,
        size_t polygonCount,
        Interrupter* const interrupter)
    {
        subdivisionCount -= 1;
        polygonCount *= 4;

        tbb::task_group tasks;

        const Vec3d ac = (mainPrim.a + mainPrim.c) * 0.5;
        const Vec3d bc = (mainPrim.b + mainPrim.c) * 0.5;
        const Vec3d ab = (mainPrim.a + mainPrim.b) * 0.5;

        Triangle prim;
        prim.index = mainPrim.index;

        prim.a = mainPrim.a;
        prim.b = ab;
        prim.c = ac;
        tasks.run(SubTask(prim, dataTable, subdivisionCount, polygonCount, interrupter));

        prim.a = ab;
        prim.b = bc;
        prim.c = ac;
        tasks.run(SubTask(prim, dataTable, subdivisionCount, polygonCount, interrupter));

        prim.a = ab;
        prim.b = mainPrim.b;
        prim.c = bc;
        tasks.run(SubTask(prim, dataTable, subdivisionCount, polygonCount, interrupter));

        prim.a = ac;
        prim.b = bc;
        prim.c = mainPrim.c;
        tasks.run(SubTask(prim, dataTable, subdivisionCount, polygonCount, interrupter));

        tasks.wait();
    }

    static void voxelizeTriangle(const Triangle& prim, VoxelizationDataType& data)
    {
        std::deque<Coord> coordList;
        Coord ijk, nijk;

        ijk = Coord::floor(prim.a);
        coordList.push_back(ijk);

        computeDistance(ijk, prim, data);

        unsigned char primId = data.getNewPrimId();
        data.primIdAcc.setValueOnly(ijk, primId);

        while (!coordList.empty()) {
            ijk = coordList.back();
            coordList.pop_back();

            for (Int32 i = 0; i < 26; ++i) {
                nijk = ijk + util::COORD_OFFSETS[i];
                if (primId != data.primIdAcc.getValue(nijk)) {
                    data.primIdAcc.setValueOnly(nijk, primId);
                    if(computeDistance(nijk, prim, data)) coordList.push_back(nijk);
                }
            }
        }
    }

    static bool computeDistance(const Coord& ijk, const Triangle& prim, VoxelizationDataType& data)
    {
        Vec3d uvw, voxelCenter(ijk[0], ijk[1], ijk[2]);

        using ValueType = typename TreeType::ValueType;

        const ValueType dist = ValueType((voxelCenter -
            closestPointOnTriangleToPoint(prim.a, prim.c, prim.b, voxelCenter, uvw)).lengthSqr());

        const ValueType oldDist = data.distAcc.getValue(ijk);

        if (dist < oldDist) {
            data.distAcc.setValue(ijk, dist);
            data.indexAcc.setValue(ijk, prim.index);
        } else if (math::isExactlyEqual(dist, oldDist)) {
            // makes reduction deterministic when different polygons
            // produce the same distance value.
            data.indexAcc.setValueOnly(ijk, std::min(prim.index, data.indexAcc.getValue(ijk)));
        }

        return !(dist > 0.75); // true if the primitive intersects the voxel.
    }

    DataTable                 * const mDataTable;
    MeshDataAdapter     const * const mMesh;
    Interrupter               * const mInterrupter;
}; // VoxelizePolygons


////////////////////////////////////////


template<typename TreeType>
struct DiffLeafNodeMask
{
    using AccessorType = typename tree::ValueAccessor<TreeType>;
    using LeafNodeType = typename TreeType::LeafNodeType;

    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    DiffLeafNodeMask(const TreeType& rhsTree,
        std::vector<BoolLeafNodeType*>& lhsNodes)
        : mRhsTree(&rhsTree), mLhsNodes(lhsNodes.empty() ? nullptr : &lhsNodes[0])
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        tree::ValueAccessor<const TreeType> acc(*mRhsTree);

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            BoolLeafNodeType* lhsNode = mLhsNodes[n];
            const LeafNodeType* rhsNode = acc.probeConstLeaf(lhsNode->origin());

            if (rhsNode) lhsNode->topologyDifference(*rhsNode, false);
        }
    }

private:
    TreeType            const * const mRhsTree;
    BoolLeafNodeType         ** const mLhsNodes;
};


template<typename LeafNodeTypeA, typename LeafNodeTypeB>
struct UnionValueMasks
{
    UnionValueMasks(std::vector<LeafNodeTypeA*>& nodesA, std::vector<LeafNodeTypeB*>& nodesB)
        : mNodesA(nodesA.empty() ? nullptr : &nodesA[0])
        , mNodesB(nodesB.empty() ? nullptr : &nodesB[0])
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            mNodesA[n]->topologyUnion(*mNodesB[n]);
        }
    }

private:
    LeafNodeTypeA ** const  mNodesA;
    LeafNodeTypeB ** const  mNodesB;
};


template<typename TreeType>
struct ConstructVoxelMask
{
    using LeafNodeType = typename TreeType::LeafNodeType;

    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    ConstructVoxelMask(BoolTreeType& maskTree, const TreeType& tree,
        std::vector<LeafNodeType*>& nodes)
        : mTree(&tree)
        , mNodes(nodes.empty() ? nullptr : &nodes[0])
        , mLocalMaskTree(false)
        , mMaskTree(&maskTree)
    {
    }

    ConstructVoxelMask(ConstructVoxelMask& rhs, tbb::split)
        : mTree(rhs.mTree)
        , mNodes(rhs.mNodes)
        , mLocalMaskTree(false)
        , mMaskTree(&mLocalMaskTree)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range)
    {
        using Iterator = typename LeafNodeType::ValueOnCIter;

        tree::ValueAccessor<const TreeType> acc(*mTree);
        tree::ValueAccessor<BoolTreeType> maskAcc(*mMaskTree);

        Coord ijk, nijk, localCorod;
        Index pos, npos;

        for (size_t n = range.begin(); n != range.end(); ++n) {

            LeafNodeType& node = *mNodes[n];

            CoordBBox bbox = node.getNodeBoundingBox();
            bbox.expand(-1);

            BoolLeafNodeType& maskNode = *maskAcc.touchLeaf(node.origin());

            for (Iterator it = node.cbeginValueOn(); it; ++it) {
                ijk = it.getCoord();
                pos = it.pos();

                localCorod = LeafNodeType::offsetToLocalCoord(pos);

                if (localCorod[2] < int(LeafNodeType::DIM - 1)) {
                    npos = pos + 1;
                    if (!node.isValueOn(npos)) maskNode.setValueOn(npos);
                } else {
                    nijk = ijk.offsetBy(0, 0, 1);
                    if (!acc.isValueOn(nijk)) maskAcc.setValueOn(nijk);
                }

                if (localCorod[2] > 0) {
                    npos = pos - 1;
                    if (!node.isValueOn(npos)) maskNode.setValueOn(npos);
                } else {
                    nijk = ijk.offsetBy(0, 0, -1);
                    if (!acc.isValueOn(nijk)) maskAcc.setValueOn(nijk);
                }

                if (localCorod[1] < int(LeafNodeType::DIM - 1)) {
                    npos = pos + LeafNodeType::DIM;
                    if (!node.isValueOn(npos)) maskNode.setValueOn(npos);
                } else {
                    nijk = ijk.offsetBy(0, 1, 0);
                    if (!acc.isValueOn(nijk)) maskAcc.setValueOn(nijk);
                }

                if (localCorod[1] > 0) {
                    npos = pos - LeafNodeType::DIM;
                    if (!node.isValueOn(npos)) maskNode.setValueOn(npos);
                } else {
                    nijk = ijk.offsetBy(0, -1, 0);
                    if (!acc.isValueOn(nijk)) maskAcc.setValueOn(nijk);
                }

                if (localCorod[0] < int(LeafNodeType::DIM - 1)) {
                    npos = pos + LeafNodeType::DIM * LeafNodeType::DIM;
                    if (!node.isValueOn(npos)) maskNode.setValueOn(npos);
                } else {
                    nijk = ijk.offsetBy(1, 0, 0);
                    if (!acc.isValueOn(nijk)) maskAcc.setValueOn(nijk);
                }

                if (localCorod[0] > 0) {
                    npos = pos - LeafNodeType::DIM * LeafNodeType::DIM;
                    if (!node.isValueOn(npos)) maskNode.setValueOn(npos);
                } else {
                    nijk = ijk.offsetBy(-1, 0, 0);
                    if (!acc.isValueOn(nijk)) maskAcc.setValueOn(nijk);
                }
            }
        }
    }

    void join(ConstructVoxelMask& rhs) { mMaskTree->merge(*rhs.mMaskTree); }

private:
    TreeType        const   * const mTree;
    LeafNodeType           ** const mNodes;

    BoolTreeType         mLocalMaskTree;
    BoolTreeType * const mMaskTree;
};


/// @note The interior and exterior widths should be in world space units and squared.
template<typename TreeType, typename MeshDataAdapter>
struct ExpandNarrowband
{
    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using NodeMaskType = typename LeafNodeType::NodeMaskType;
    using Int32TreeType = typename TreeType::template ValueConverter<Int32>::Type;
    using Int32LeafNodeType = typename Int32TreeType::LeafNodeType;
    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    struct Fragment
    {
        Int32 idx, x, y, z;
        ValueType dist;

        Fragment() : idx(0), x(0), y(0), z(0), dist(0.0) {}

        Fragment(Int32 idx_, Int32 x_, Int32 y_, Int32 z_, ValueType dist_)
            : idx(idx_), x(x_), y(y_), z(z_), dist(dist_)
        {
        }

        bool operator<(const Fragment& rhs) const { return idx < rhs.idx; }
    }; // struct Fragment

    ////////////////////

    ExpandNarrowband(
        std::vector<BoolLeafNodeType*>& maskNodes,
        BoolTreeType& maskTree,
        TreeType& distTree,
        Int32TreeType& indexTree,
        const MeshDataAdapter& mesh,
        ValueType exteriorBandWidth,
        ValueType interiorBandWidth,
        ValueType voxelSize)
        : mMaskNodes(maskNodes.empty() ? nullptr : &maskNodes[0])
        , mMaskTree(&maskTree)
        , mDistTree(&distTree)
        , mIndexTree(&indexTree)
        , mMesh(&mesh)
        , mNewMaskTree(false)
        , mDistNodes()
        , mUpdatedDistNodes()
        , mIndexNodes()
        , mUpdatedIndexNodes()
        , mExteriorBandWidth(exteriorBandWidth)
        , mInteriorBandWidth(interiorBandWidth)
        , mVoxelSize(voxelSize)
    {
    }

    ExpandNarrowband(const ExpandNarrowband& rhs, tbb::split)
        : mMaskNodes(rhs.mMaskNodes)
        , mMaskTree(rhs.mMaskTree)
        , mDistTree(rhs.mDistTree)
        , mIndexTree(rhs.mIndexTree)
        , mMesh(rhs.mMesh)
        , mNewMaskTree(false)
        , mDistNodes()
        , mUpdatedDistNodes()
        , mIndexNodes()
        , mUpdatedIndexNodes()
        , mExteriorBandWidth(rhs.mExteriorBandWidth)
        , mInteriorBandWidth(rhs.mInteriorBandWidth)
        , mVoxelSize(rhs.mVoxelSize)
    {
    }

    void join(ExpandNarrowband& rhs)
    {
        mDistNodes.insert(mDistNodes.end(), rhs.mDistNodes.begin(), rhs.mDistNodes.end());
        mIndexNodes.insert(mIndexNodes.end(), rhs.mIndexNodes.begin(), rhs.mIndexNodes.end());

        mUpdatedDistNodes.insert(mUpdatedDistNodes.end(),
            rhs.mUpdatedDistNodes.begin(), rhs.mUpdatedDistNodes.end());

        mUpdatedIndexNodes.insert(mUpdatedIndexNodes.end(),
            rhs.mUpdatedIndexNodes.begin(), rhs.mUpdatedIndexNodes.end());

        mNewMaskTree.merge(rhs.mNewMaskTree);
    }

    void operator()(const tbb::blocked_range<size_t>& range)
    {
        tree::ValueAccessor<BoolTreeType>   newMaskAcc(mNewMaskTree);
        tree::ValueAccessor<TreeType>       distAcc(*mDistTree);
        tree::ValueAccessor<Int32TreeType>  indexAcc(*mIndexTree);

        std::vector<Fragment> fragments;
        fragments.reserve(256);

        std::unique_ptr<LeafNodeType> newDistNodePt;
        std::unique_ptr<Int32LeafNodeType> newIndexNodePt;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            BoolLeafNodeType& maskNode = *mMaskNodes[n];
            if (maskNode.isEmpty()) continue;

            // Setup local caches

            const Coord& origin = maskNode.origin();

            LeafNodeType      * distNodePt = distAcc.probeLeaf(origin);
            Int32LeafNodeType * indexNodePt = indexAcc.probeLeaf(origin);

            assert(!distNodePt == !indexNodePt);

            bool usingNewNodes = false;

            if (!distNodePt && !indexNodePt) {

                const ValueType backgroundDist = distAcc.getValue(origin);

                if (!newDistNodePt.get() && !newIndexNodePt.get()) {
                    newDistNodePt.reset(new LeafNodeType(origin, backgroundDist));
                    newIndexNodePt.reset(new Int32LeafNodeType(origin, indexAcc.getValue(origin)));
                } else {

                    if ((backgroundDist < ValueType(0.0)) !=
                            (newDistNodePt->getValue(0) < ValueType(0.0))) {
                        newDistNodePt->buffer().fill(backgroundDist);
                    }

                    newDistNodePt->setOrigin(origin);
                    newIndexNodePt->setOrigin(origin);
                }

                distNodePt = newDistNodePt.get();
                indexNodePt = newIndexNodePt.get();

                usingNewNodes = true;
            }


            // Gather neighbour information

            CoordBBox bbox(Coord::max(), Coord::min());
            for (typename BoolLeafNodeType::ValueOnIter it = maskNode.beginValueOn(); it; ++it) {
                bbox.expand(it.getCoord());
            }

            bbox.expand(1);

            gatherFragments(fragments, bbox, distAcc, indexAcc);


            // Compute first voxel layer

            bbox = maskNode.getNodeBoundingBox();
            NodeMaskType mask;
            bool updatedLeafNodes = false;

            for (typename BoolLeafNodeType::ValueOnIter it = maskNode.beginValueOn(); it; ++it) {

                const Coord ijk = it.getCoord();

                if (updateVoxel(ijk, 5, fragments, *distNodePt, *indexNodePt, &updatedLeafNodes)) {

                    for (Int32 i = 0; i < 6; ++i) {
                        const Coord nijk = ijk + util::COORD_OFFSETS[i];
                        if (bbox.isInside(nijk)) {
                            mask.setOn(BoolLeafNodeType::coordToOffset(nijk));
                        } else  {
                            newMaskAcc.setValueOn(nijk);
                        }
                    }

                    for (Int32 i = 6; i < 26; ++i) {
                        const Coord nijk = ijk + util::COORD_OFFSETS[i];
                        if (bbox.isInside(nijk)) {
                            mask.setOn(BoolLeafNodeType::coordToOffset(nijk));
                        }
                    }
                }
            }

            if (updatedLeafNodes) {

                // Compute second voxel layer
                mask -= indexNodePt->getValueMask();

                for (typename NodeMaskType::OnIterator it = mask.beginOn(); it; ++it) {

                    const Index pos = it.pos();
                    const Coord ijk = maskNode.origin() + LeafNodeType::offsetToLocalCoord(pos);

                    if (updateVoxel(ijk, 6, fragments, *distNodePt, *indexNodePt)) {
                        for (Int32 i = 0; i < 6; ++i) {
                            newMaskAcc.setValueOn(ijk + util::COORD_OFFSETS[i]);
                        }
                    }
                }

                // Export new distance values
                if (usingNewNodes) {
                    newDistNodePt->topologyUnion(*newIndexNodePt);
                    mDistNodes.push_back(newDistNodePt.release());
                    mIndexNodes.push_back(newIndexNodePt.release());
                } else {
                    mUpdatedDistNodes.push_back(distNodePt);
                    mUpdatedIndexNodes.push_back(indexNodePt);
                }
            }
        } // end leafnode loop
    }

    //////////

    BoolTreeType& newMaskTree() { return mNewMaskTree; }

    std::vector<LeafNodeType*>& newDistNodes() { return mDistNodes; }
    std::vector<LeafNodeType*>& updatedDistNodes() { return mUpdatedDistNodes; }

    std::vector<Int32LeafNodeType*>& newIndexNodes() { return mIndexNodes; }
    std::vector<Int32LeafNodeType*>& updatedIndexNodes() { return mUpdatedIndexNodes; }

private:

    /// @note   The output fragment list is ordered by the primitive index
    void
    gatherFragments(std::vector<Fragment>& fragments, const CoordBBox& bbox,
        tree::ValueAccessor<TreeType>& distAcc, tree::ValueAccessor<Int32TreeType>& indexAcc)
    {
        fragments.clear();
        const Coord nodeMin = bbox.min() & ~(LeafNodeType::DIM - 1);
        const Coord nodeMax = bbox.max() & ~(LeafNodeType::DIM - 1);

        CoordBBox region;
        Coord ijk;

        for (ijk[0] = nodeMin[0]; ijk[0] <= nodeMax[0]; ijk[0] += LeafNodeType::DIM) {
            for (ijk[1] = nodeMin[1]; ijk[1] <= nodeMax[1]; ijk[1] += LeafNodeType::DIM) {
                for (ijk[2] = nodeMin[2]; ijk[2] <= nodeMax[2]; ijk[2] += LeafNodeType::DIM) {
                    if (LeafNodeType* distleaf = distAcc.probeLeaf(ijk)) {
                        region.min() = Coord::maxComponent(bbox.min(), ijk);
                        region.max() = Coord::minComponent(bbox.max(),
                            ijk.offsetBy(LeafNodeType::DIM - 1));
                        gatherFragments(fragments, region, *distleaf, *indexAcc.probeLeaf(ijk));
                    }
                }
            }
        }

        std::sort(fragments.begin(), fragments.end());
    }

    void
    gatherFragments(std::vector<Fragment>& fragments, const CoordBBox& bbox,
        const LeafNodeType& distLeaf, const Int32LeafNodeType& idxLeaf) const
    {
        const typename LeafNodeType::NodeMaskType& mask = distLeaf.getValueMask();
        const ValueType* distData = distLeaf.buffer().data();
        const Int32* idxData = idxLeaf.buffer().data();

        for (int x = bbox.min()[0]; x <= bbox.max()[0]; ++x) {
            const Index xPos = (x & (LeafNodeType::DIM - 1u)) << (2 * LeafNodeType::LOG2DIM);
            for (int y = bbox.min()[1]; y <= bbox.max()[1]; ++y) {
                const Index yPos = xPos + ((y & (LeafNodeType::DIM - 1u)) << LeafNodeType::LOG2DIM);
                for (int z = bbox.min()[2]; z <= bbox.max()[2]; ++z) {
                    const Index pos = yPos + (z & (LeafNodeType::DIM - 1u));
                    if (mask.isOn(pos)) {
                        fragments.push_back(Fragment(idxData[pos],x,y,z, std::abs(distData[pos])));
                    }
                }
            }
        }
    }

    /// @note   This method expects the fragment list to be ordered by the primitive index
    ///         to avoid redundant distance computations.
    ValueType
    computeDistance(const Coord& ijk, const Int32 manhattanLimit,
        const std::vector<Fragment>& fragments, Int32& closestPrimIdx) const
    {
        Vec3d a, b, c, uvw, voxelCenter(ijk[0], ijk[1], ijk[2]);
        double primDist, tmpDist, dist = std::numeric_limits<double>::max();
        Int32 lastIdx = Int32(util::INVALID_IDX);

        for (size_t n = 0, N = fragments.size(); n < N; ++n) {

            const Fragment& fragment = fragments[n];
            if (lastIdx == fragment.idx) continue;

            const Int32 dx = std::abs(fragment.x - ijk[0]);
            const Int32 dy = std::abs(fragment.y - ijk[1]);
            const Int32 dz = std::abs(fragment.z - ijk[2]);

            const Int32 manhattan = dx + dy + dz;
            if (manhattan > manhattanLimit) continue;

            lastIdx = fragment.idx;

            const size_t polygon = size_t(lastIdx);

            mMesh->getIndexSpacePoint(polygon, 0, a);
            mMesh->getIndexSpacePoint(polygon, 1, b);
            mMesh->getIndexSpacePoint(polygon, 2, c);

            primDist = (voxelCenter -
                closestPointOnTriangleToPoint(a, c, b, voxelCenter, uvw)).lengthSqr();

            // Split quad into a second triangle
            if (4 == mMesh->vertexCount(polygon)) {

                mMesh->getIndexSpacePoint(polygon, 3, b);

                tmpDist = (voxelCenter - closestPointOnTriangleToPoint(
                    a, b, c, voxelCenter, uvw)).lengthSqr();

                if (tmpDist < primDist) primDist = tmpDist;
            }

            if (primDist < dist) {
                dist = primDist;
                closestPrimIdx = lastIdx;
            }
        }

        return ValueType(std::sqrt(dist)) * mVoxelSize;
    }

    /// @note   Returns true if the current voxel was updated and neighbouring
    ///         voxels need to be evaluated.
    bool
    updateVoxel(const Coord& ijk, const Int32 manhattanLimit,
        const std::vector<Fragment>& fragments,
        LeafNodeType& distLeaf, Int32LeafNodeType& idxLeaf, bool* updatedLeafNodes = nullptr)
    {
        Int32 closestPrimIdx = 0;
        const ValueType distance = computeDistance(ijk, manhattanLimit, fragments, closestPrimIdx);

        const Index pos = LeafNodeType::coordToOffset(ijk);
        const bool inside = distLeaf.getValue(pos) < ValueType(0.0);

        bool activateNeighbourVoxels = false;

        if (!inside && distance < mExteriorBandWidth) {
            if (updatedLeafNodes) *updatedLeafNodes = true;
            activateNeighbourVoxels = (distance + mVoxelSize) < mExteriorBandWidth;
            distLeaf.setValueOnly(pos, distance);
            idxLeaf.setValueOn(pos, closestPrimIdx);
        } else if (inside && distance < mInteriorBandWidth) {
            if (updatedLeafNodes) *updatedLeafNodes = true;
            activateNeighbourVoxels = (distance + mVoxelSize) < mInteriorBandWidth;
            distLeaf.setValueOnly(pos, -distance);
            idxLeaf.setValueOn(pos, closestPrimIdx);
        }

        return activateNeighbourVoxels;
    }

    //////////

    BoolLeafNodeType     ** const mMaskNodes;
    BoolTreeType          * const mMaskTree;
    TreeType              * const mDistTree;
    Int32TreeType         * const mIndexTree;

    MeshDataAdapter const * const mMesh;

    BoolTreeType mNewMaskTree;

    std::vector<LeafNodeType*> mDistNodes, mUpdatedDistNodes;
    std::vector<Int32LeafNodeType*> mIndexNodes, mUpdatedIndexNodes;

    const ValueType mExteriorBandWidth, mInteriorBandWidth, mVoxelSize;
}; // struct ExpandNarrowband


template<typename TreeType>
struct AddNodes {
    using LeafNodeType = typename TreeType::LeafNodeType;

    AddNodes(TreeType& tree, std::vector<LeafNodeType*>& nodes)
        : mTree(&tree) , mNodes(&nodes)
    {
    }

    void operator()() const {
        tree::ValueAccessor<TreeType> acc(*mTree);
        std::vector<LeafNodeType*>& nodes = *mNodes;
        for (size_t n = 0, N = nodes.size(); n < N; ++n) {
            acc.addLeaf(nodes[n]);
        }
    }

    TreeType                   * const mTree;
    std::vector<LeafNodeType*> * const mNodes;
}; // AddNodes


template<typename TreeType, typename Int32TreeType, typename BoolTreeType, typename MeshDataAdapter>
inline void
expandNarrowband(
    TreeType& distTree,
    Int32TreeType& indexTree,
    BoolTreeType& maskTree,
    std::vector<typename BoolTreeType::LeafNodeType*>& maskNodes,
    const MeshDataAdapter& mesh,
    typename TreeType::ValueType exteriorBandWidth,
    typename TreeType::ValueType interiorBandWidth,
    typename TreeType::ValueType voxelSize)
{
    ExpandNarrowband<TreeType, MeshDataAdapter> expandOp(maskNodes, maskTree,
        distTree, indexTree, mesh, exteriorBandWidth, interiorBandWidth, voxelSize);

    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, maskNodes.size()), expandOp);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, expandOp.updatedIndexNodes().size()),
        UnionValueMasks<typename TreeType::LeafNodeType, typename Int32TreeType::LeafNodeType>(
            expandOp.updatedDistNodes(), expandOp.updatedIndexNodes()));

    tbb::task_group tasks;
    tasks.run(AddNodes<TreeType>(distTree, expandOp.newDistNodes()));
    tasks.run(AddNodes<Int32TreeType>(indexTree, expandOp.newIndexNodes()));
    tasks.wait();

    maskTree.clear();
    maskTree.merge(expandOp.newMaskTree());
}


////////////////////////////////////////


// Transform values (sqrt, world space scaling and sign flip if sdf)
template<typename TreeType>
struct TransformValues
{
    using LeafNodeType = typename TreeType::LeafNodeType;
    using ValueType = typename TreeType::ValueType;

    TransformValues(std::vector<LeafNodeType*>& nodes,
        ValueType voxelSize, bool unsignedDist)
        : mNodes(&nodes[0])
        , mVoxelSize(voxelSize)
        , mUnsigned(unsignedDist)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        typename LeafNodeType::ValueOnIter iter;

        const bool udf = mUnsigned;
        const ValueType w[2] = { -mVoxelSize, mVoxelSize };

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            for (iter = mNodes[n]->beginValueOn(); iter; ++iter) {
                ValueType& val = const_cast<ValueType&>(iter.getValue());
                val = w[udf || (val < ValueType(0.0))] * std::sqrt(std::abs(val));
            }
        }
    }

private:
    LeafNodeType * * const  mNodes;
    const ValueType         mVoxelSize;
    const bool              mUnsigned;
};


// Inactivate values outside the (exBandWidth, inBandWidth) range.
template<typename TreeType>
struct InactivateValues
{
    using LeafNodeType = typename TreeType::LeafNodeType;
    using ValueType = typename TreeType::ValueType;

    InactivateValues(std::vector<LeafNodeType*>& nodes,
        ValueType exBandWidth, ValueType inBandWidth)
        : mNodes(nodes.empty() ? nullptr : &nodes[0])
        , mExBandWidth(exBandWidth)
        , mInBandWidth(inBandWidth)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        typename LeafNodeType::ValueOnIter iter;
        const ValueType exVal = mExBandWidth;
        const ValueType inVal = -mInBandWidth;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            for (iter = mNodes[n]->beginValueOn(); iter; ++iter) {

                ValueType& val = const_cast<ValueType&>(iter.getValue());

                const bool inside = val < ValueType(0.0);

                if (inside && !(val > inVal)) {
                    val = inVal;
                    iter.setValueOff();
                } else if (!inside && !(val < exVal)) {
                    val = exVal;
                    iter.setValueOff();
                }
            }
        }
    }

private:
    LeafNodeType * * const mNodes;
    const ValueType mExBandWidth, mInBandWidth;
};


template<typename TreeType>
struct OffsetValues
{
    using LeafNodeType = typename TreeType::LeafNodeType;
    using ValueType = typename TreeType::ValueType;

    OffsetValues(std::vector<LeafNodeType*>& nodes, ValueType offset)
        : mNodes(nodes.empty() ? nullptr : &nodes[0]), mOffset(offset)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        const ValueType offset = mOffset;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            typename LeafNodeType::ValueOnIter iter = mNodes[n]->beginValueOn();

            for (; iter; ++iter) {
                ValueType& val = const_cast<ValueType&>(iter.getValue());
                val += offset;
            }
        }
    }

private:
    LeafNodeType * * const mNodes;
    const ValueType mOffset;
};


template<typename TreeType>
struct Renormalize
{
    using LeafNodeType = typename TreeType::LeafNodeType;
    using ValueType = typename TreeType::ValueType;

    Renormalize(const TreeType& tree, const std::vector<LeafNodeType*>& nodes,
        ValueType* buffer, ValueType voxelSize)
        : mTree(&tree)
        , mNodes(nodes.empty() ? nullptr : &nodes[0])
        , mBuffer(buffer)
        , mVoxelSize(voxelSize)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        using Vec3Type = math::Vec3<ValueType>;

        tree::ValueAccessor<const TreeType> acc(*mTree);

        Coord ijk;
        Vec3Type up, down;

        const ValueType dx = mVoxelSize, invDx = ValueType(1.0) / mVoxelSize;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            ValueType* bufferData = &mBuffer[n * LeafNodeType::SIZE];

            typename LeafNodeType::ValueOnCIter iter = mNodes[n]->cbeginValueOn();
            for (; iter; ++iter) {

                const ValueType phi0 = *iter;

                ijk = iter.getCoord();

                up[0] = acc.getValue(ijk.offsetBy(1, 0, 0)) - phi0;
                up[1] = acc.getValue(ijk.offsetBy(0, 1, 0)) - phi0;
                up[2] = acc.getValue(ijk.offsetBy(0, 0, 1)) - phi0;

                down[0] = phi0 - acc.getValue(ijk.offsetBy(-1, 0, 0));
                down[1] = phi0 - acc.getValue(ijk.offsetBy(0, -1, 0));
                down[2] = phi0 - acc.getValue(ijk.offsetBy(0, 0, -1));

                const ValueType normSqGradPhi = math::GodunovsNormSqrd(phi0 > 0.0, down, up);

                const ValueType diff = math::Sqrt(normSqGradPhi) * invDx - ValueType(1.0);
                const ValueType S = phi0 / (math::Sqrt(math::Pow2(phi0) + normSqGradPhi));

                bufferData[iter.pos()] = phi0 - dx * S * diff;
            }
        }
    }

private:
    TreeType             const * const mTree;
    LeafNodeType const * const * const mNodes;
    ValueType                  * const mBuffer;

    const ValueType mVoxelSize;
};


template<typename TreeType>
struct MinCombine
{
    using LeafNodeType = typename TreeType::LeafNodeType;
    using ValueType = typename TreeType::ValueType;

    MinCombine(std::vector<LeafNodeType*>& nodes, const ValueType* buffer)
        : mNodes(nodes.empty() ? nullptr : &nodes[0]), mBuffer(buffer)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            const ValueType* bufferData = &mBuffer[n * LeafNodeType::SIZE];

            typename LeafNodeType::ValueOnIter iter = mNodes[n]->beginValueOn();

            for (; iter; ++iter) {
                ValueType& val = const_cast<ValueType&>(iter.getValue());
                val = std::min(val, bufferData[iter.pos()]);
            }
        }
    }

private:
    LeafNodeType * * const mNodes;
    ValueType const * const mBuffer;
};


} // mesh_to_volume_internal namespace


////////////////////////////////////////

// Utility method implementation


template <typename FloatTreeT>
inline void
traceExteriorBoundaries(FloatTreeT& tree)
{
    using ConnectivityTable = mesh_to_volume_internal::LeafNodeConnectivityTable<FloatTreeT>;

    ConnectivityTable nodeConnectivity(tree);

    std::vector<size_t> zStartNodes, yStartNodes, xStartNodes;

    for (size_t n = 0; n < nodeConnectivity.size(); ++n) {
        if (ConnectivityTable::INVALID_OFFSET == nodeConnectivity.offsetsPrevX()[n]) {
            xStartNodes.push_back(n);
        }

        if (ConnectivityTable::INVALID_OFFSET == nodeConnectivity.offsetsPrevY()[n]) {
            yStartNodes.push_back(n);
        }

        if (ConnectivityTable::INVALID_OFFSET == nodeConnectivity.offsetsPrevZ()[n]) {
            zStartNodes.push_back(n);
        }
    }

    using SweepingOp = mesh_to_volume_internal::SweepExteriorSign<FloatTreeT>;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, zStartNodes.size()),
        SweepingOp(SweepingOp::Z_AXIS, zStartNodes, nodeConnectivity));

    tbb::parallel_for(tbb::blocked_range<size_t>(0, yStartNodes.size()),
        SweepingOp(SweepingOp::Y_AXIS, yStartNodes, nodeConnectivity));

    tbb::parallel_for(tbb::blocked_range<size_t>(0, xStartNodes.size()),
        SweepingOp(SweepingOp::X_AXIS, xStartNodes, nodeConnectivity));

    const size_t numLeafNodes = nodeConnectivity.size();
    const size_t numVoxels = numLeafNodes * FloatTreeT::LeafNodeType::SIZE;

    std::unique_ptr<bool[]> changedNodeMaskA{new bool[numLeafNodes]};
    std::unique_ptr<bool[]> changedNodeMaskB{new bool[numLeafNodes]};
    std::unique_ptr<bool[]> changedVoxelMask{new bool[numVoxels]};

    mesh_to_volume_internal::fillArray(changedNodeMaskA.get(), true, numLeafNodes);
    mesh_to_volume_internal::fillArray(changedNodeMaskB.get(), false, numLeafNodes);
    mesh_to_volume_internal::fillArray(changedVoxelMask.get(), false, numVoxels);

    const tbb::blocked_range<size_t> nodeRange(0, numLeafNodes);

    bool nodesUpdated = false;
    do {
        tbb::parallel_for(nodeRange, mesh_to_volume_internal::SeedFillExteriorSign<FloatTreeT>(
            nodeConnectivity.nodes(), changedNodeMaskA.get()));

        tbb::parallel_for(nodeRange, mesh_to_volume_internal::SeedPoints<FloatTreeT>(
            nodeConnectivity, changedNodeMaskA.get(), changedNodeMaskB.get(),
            changedVoxelMask.get()));

        changedNodeMaskA.swap(changedNodeMaskB);

        nodesUpdated = false;
        for (size_t n = 0; n < numLeafNodes; ++n) {
            nodesUpdated |= changedNodeMaskA[n];
            if (nodesUpdated) break;
        }

        if (nodesUpdated) {
            tbb::parallel_for(nodeRange, mesh_to_volume_internal::SyncVoxelMask<FloatTreeT>(
                nodeConnectivity.nodes(), changedNodeMaskA.get(), changedVoxelMask.get()));
        }
    } while (nodesUpdated);

} // void traceExteriorBoundaries()


////////////////////////////////////////


template <typename GridType, typename MeshDataAdapter, typename Interrupter>
inline typename GridType::Ptr
meshToVolume(
  Interrupter& interrupter,
  const MeshDataAdapter& mesh,
  const math::Transform& transform,
  float exteriorBandWidth,
  float interiorBandWidth,
  int flags,
  typename GridType::template ValueConverter<Int32>::Type * polygonIndexGrid)
{
    using GridTypePtr = typename GridType::Ptr;
    using TreeType = typename GridType::TreeType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using ValueType = typename GridType::ValueType;

    using Int32GridType = typename GridType::template ValueConverter<Int32>::Type;
    using Int32TreeType = typename Int32GridType::TreeType;

    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;

    //////////

    // Setup

    GridTypePtr distGrid(new GridType(std::numeric_limits<ValueType>::max()));
    distGrid->setTransform(transform.copy());

    ValueType exteriorWidth = ValueType(exteriorBandWidth);
    ValueType interiorWidth = ValueType(interiorBandWidth);

    // Note: inf interior width is all right, this value makes the converter fill
    // interior regions with distance values.
    if (!std::isfinite(exteriorWidth) || std::isnan(interiorWidth)) {
        std::stringstream msg;
        msg << "Illegal narrow band width: exterior = " << exteriorWidth
            << ", interior = " << interiorWidth;
        OPENVDB_LOG_DEBUG(msg.str());
        return distGrid;
    }

    const ValueType voxelSize = ValueType(transform.voxelSize()[0]);

    if (!std::isfinite(voxelSize) || math::isZero(voxelSize)) {
        std::stringstream msg;
        msg << "Illegal transform, voxel size = " << voxelSize;
        OPENVDB_LOG_DEBUG(msg.str());
        return distGrid;
    }

    // Convert narrow band width from voxel units to world space units.
    exteriorWidth *= voxelSize;
    // Avoid the unit conversion if the interior band width is set to
    // inf or std::numeric_limits<float>::max().
    if (interiorWidth < std::numeric_limits<ValueType>::max()) {
        interiorWidth *= voxelSize;
    }

    const bool computeSignedDistanceField = (flags & UNSIGNED_DISTANCE_FIELD) == 0;
    const bool removeIntersectingVoxels = (flags & DISABLE_INTERSECTING_VOXEL_REMOVAL) == 0;
    const bool renormalizeValues = (flags & DISABLE_RENORMALIZATION) == 0;
    const bool trimNarrowBand = (flags & DISABLE_NARROW_BAND_TRIMMING) == 0;

    Int32GridType* indexGrid = nullptr;

    typename Int32GridType::Ptr temporaryIndexGrid;

    if (polygonIndexGrid) {
        indexGrid = polygonIndexGrid;
    } else {
        temporaryIndexGrid.reset(new Int32GridType(Int32(util::INVALID_IDX)));
        indexGrid = temporaryIndexGrid.get();
    }

    indexGrid->newTree();
    indexGrid->setTransform(transform.copy());

    if (computeSignedDistanceField) {
        distGrid->setGridClass(GRID_LEVEL_SET);
    } else {
        distGrid->setGridClass(GRID_UNKNOWN);
        interiorWidth = ValueType(0.0);
    }

    TreeType& distTree = distGrid->tree();
    Int32TreeType& indexTree = indexGrid->tree();


    //////////

    // Voxelize mesh

    {
        using VoxelizationDataType = mesh_to_volume_internal::VoxelizationData<TreeType>;
        using DataTable = tbb::enumerable_thread_specific<typename VoxelizationDataType::Ptr>;

        DataTable data;
        using Voxelizer =
            mesh_to_volume_internal::VoxelizePolygons<TreeType, MeshDataAdapter, Interrupter>;

        const tbb::blocked_range<size_t> polygonRange(0, mesh.polygonCount());

        tbb::parallel_for(polygonRange, Voxelizer(data, mesh, &interrupter));

        for (typename DataTable::iterator i = data.begin(); i != data.end(); ++i) {
            VoxelizationDataType& dataItem = **i;
            mesh_to_volume_internal::combineData(
                distTree, indexTree, dataItem.distTree, dataItem.indexTree);
        }
    }

    // The progress estimates are based on the observed average time for a few different
    // test cases and is only intended to provide some rough progression feedback to the user.
    if (interrupter.wasInterrupted(30)) return distGrid;


    //////////

    // Classify interior and exterior regions

    if (computeSignedDistanceField) {

        // Determines the inside/outside state for the narrow band of voxels.
        traceExteriorBoundaries(distTree);

        std::vector<LeafNodeType*> nodes;
        nodes.reserve(distTree.leafCount());
        distTree.getNodes(nodes);

        const tbb::blocked_range<size_t> nodeRange(0, nodes.size());

        using SignOp =
            mesh_to_volume_internal::ComputeIntersectingVoxelSign<TreeType, MeshDataAdapter>;

        tbb::parallel_for(nodeRange, SignOp(nodes, distTree, indexTree, mesh));

        if (interrupter.wasInterrupted(45)) return distGrid;

        // Remove voxels created by self intersecting portions of the mesh.
        if (removeIntersectingVoxels) {

            tbb::parallel_for(nodeRange,
                mesh_to_volume_internal::ValidateIntersectingVoxels<TreeType>(distTree, nodes));

            tbb::parallel_for(nodeRange,
                mesh_to_volume_internal::RemoveSelfIntersectingSurface<TreeType>(
                    nodes, distTree, indexTree));

            tools::pruneInactive(distTree,  /*threading=*/true);
            tools::pruneInactive(indexTree, /*threading=*/true);
        }
    }

    if (interrupter.wasInterrupted(50)) return distGrid;

    if (distTree.activeVoxelCount() == 0) {
        distTree.clear();
        distTree.root().setBackground(exteriorWidth, /*updateChildNodes=*/false);
        return distGrid;
    }

    // Transform values (world space scaling etc.).
    {
        std::vector<LeafNodeType*> nodes;
        nodes.reserve(distTree.leafCount());
        distTree.getNodes(nodes);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, nodes.size()),
            mesh_to_volume_internal::TransformValues<TreeType>(
                nodes, voxelSize, !computeSignedDistanceField));
    }

    // Propagate sign information into tile regions.
    if (computeSignedDistanceField) {
        distTree.root().setBackground(exteriorWidth, /*updateChildNodes=*/false);
        tools::signedFloodFillWithValues(distTree, exteriorWidth, -interiorWidth);
    } else {
        tools::changeBackground(distTree, exteriorWidth);
    }

    if (interrupter.wasInterrupted(54)) return distGrid;


    //////////

    // Expand the narrow band region

    const ValueType minBandWidth = voxelSize * ValueType(2.0);

    if (interiorWidth > minBandWidth || exteriorWidth > minBandWidth) {

        // Create the initial voxel mask.
        BoolTreeType maskTree(false);

        {
            std::vector<LeafNodeType*> nodes;
            nodes.reserve(distTree.leafCount());
            distTree.getNodes(nodes);

            mesh_to_volume_internal::ConstructVoxelMask<TreeType> op(maskTree, distTree, nodes);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nodes.size()), op);
        }

        // Progress estimation
        unsigned maxIterations = std::numeric_limits<unsigned>::max();

        float progress = 54.0f, step = 0.0f;
        double estimated =
            2.0 * std::ceil((std::max(interiorWidth, exteriorWidth) - minBandWidth) / voxelSize);

        if (estimated < double(maxIterations)) {
            maxIterations = unsigned(estimated);
            step = 40.0f / float(maxIterations);
        }

        std::vector<typename BoolTreeType::LeafNodeType*> maskNodes;

        unsigned count = 0;
        while (true) {

            if (interrupter.wasInterrupted(int(progress))) return distGrid;

            const size_t maskNodeCount = maskTree.leafCount();
            if (maskNodeCount == 0) break;

            maskNodes.clear();
            maskNodes.reserve(maskNodeCount);
            maskTree.getNodes(maskNodes);

            const tbb::blocked_range<size_t> range(0, maskNodes.size());

            tbb::parallel_for(range,
                mesh_to_volume_internal::DiffLeafNodeMask<TreeType>(distTree, maskNodes));

            mesh_to_volume_internal::expandNarrowband(distTree, indexTree, maskTree, maskNodes,
                mesh, exteriorWidth, interiorWidth, voxelSize);

            if ((++count) >= maxIterations) break;
            progress += step;
        }
    }

    if (interrupter.wasInterrupted(94)) return distGrid;

    if (!polygonIndexGrid) indexGrid->clear();


    /////////

    // Renormalize distances to smooth out bumps caused by self intersecting
    // and overlapping portions of the mesh and renormalize the level set.

    if (computeSignedDistanceField && renormalizeValues) {

        std::vector<LeafNodeType*> nodes;
        nodes.reserve(distTree.leafCount());
        distTree.getNodes(nodes);

        std::unique_ptr<ValueType[]> buffer{new ValueType[LeafNodeType::SIZE * nodes.size()]};

        const ValueType offset = ValueType(0.8 * voxelSize);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, nodes.size()),
            mesh_to_volume_internal::OffsetValues<TreeType>(nodes, -offset));

        tbb::parallel_for(tbb::blocked_range<size_t>(0, nodes.size()),
            mesh_to_volume_internal::Renormalize<TreeType>(
                distTree, nodes, buffer.get(), voxelSize));

        tbb::parallel_for(tbb::blocked_range<size_t>(0, nodes.size()),
            mesh_to_volume_internal::MinCombine<TreeType>(nodes, buffer.get()));

        tbb::parallel_for(tbb::blocked_range<size_t>(0, nodes.size()),
            mesh_to_volume_internal::OffsetValues<TreeType>(
                nodes, offset - mesh_to_volume_internal::Tolerance<ValueType>::epsilon()));
    }

    if (interrupter.wasInterrupted(99)) return distGrid;


    /////////

    // Remove active voxels that exceed the narrow band limits

    if (trimNarrowBand && std::min(interiorWidth, exteriorWidth) < voxelSize * ValueType(4.0)) {

        std::vector<LeafNodeType*> nodes;
        nodes.reserve(distTree.leafCount());
        distTree.getNodes(nodes);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, nodes.size()),
            mesh_to_volume_internal::InactivateValues<TreeType>(
                nodes, exteriorWidth, computeSignedDistanceField ? interiorWidth : exteriorWidth));

        tools::pruneLevelSet(
            distTree, exteriorWidth, computeSignedDistanceField ? -interiorWidth : -exteriorWidth);
    }

    return distGrid;
}


template <typename GridType, typename MeshDataAdapter>
inline typename GridType::Ptr
meshToVolume(
  const MeshDataAdapter& mesh,
  const math::Transform& transform,
  float exteriorBandWidth,
  float interiorBandWidth,
  int flags,
  typename GridType::template ValueConverter<Int32>::Type * polygonIndexGrid)
{
    util::NullInterrupter nullInterrupter;
    return meshToVolume<GridType>(nullInterrupter, mesh, transform,
        exteriorBandWidth, interiorBandWidth, flags, polygonIndexGrid);
}


////////////////////////////////////////


//{
/// @cond OPENVDB_MESH_TO_VOLUME_INTERNAL

/// @internal This overload is enabled only for grids with a scalar, floating-point ValueType.
template<typename GridType, typename Interrupter>
inline typename std::enable_if<std::is_floating_point<typename GridType::ValueType>::value,
    typename GridType::Ptr>::type
doMeshConversion(
    Interrupter& interrupter,
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float exBandWidth,
    float inBandWidth,
    bool unsignedDistanceField = false)
{
    if (points.empty()) {
        return typename GridType::Ptr(new GridType(typename GridType::ValueType(exBandWidth)));
    }

    const size_t numPoints = points.size();
    std::unique_ptr<Vec3s[]> indexSpacePoints{new Vec3s[numPoints]};

    // transform points to local grid index space
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numPoints),
        mesh_to_volume_internal::TransformPoints<Vec3s>(
                &points[0], indexSpacePoints.get(), xform));

    const int conversionFlags = unsignedDistanceField ? UNSIGNED_DISTANCE_FIELD : 0;

    if (quads.empty()) {

        QuadAndTriangleDataAdapter<Vec3s, Vec3I>
            mesh(indexSpacePoints.get(), numPoints, &triangles[0], triangles.size());

        return meshToVolume<GridType>(
            interrupter, mesh, xform, exBandWidth, inBandWidth, conversionFlags);

    } else if (triangles.empty()) {

        QuadAndTriangleDataAdapter<Vec3s, Vec4I>
            mesh(indexSpacePoints.get(), numPoints, &quads[0], quads.size());

        return meshToVolume<GridType>(
            interrupter, mesh, xform, exBandWidth, inBandWidth, conversionFlags);
    }

    // pack primitives

    const size_t numPrimitives = triangles.size() + quads.size();
    std::unique_ptr<Vec4I[]> prims{new Vec4I[numPrimitives]};

    for (size_t n = 0, N = triangles.size(); n < N; ++n) {
        const Vec3I& triangle = triangles[n];
        Vec4I& prim = prims[n];
        prim[0] = triangle[0];
        prim[1] = triangle[1];
        prim[2] = triangle[2];
        prim[3] = util::INVALID_IDX;
    }

    const size_t offset = triangles.size();
    for (size_t n = 0, N = quads.size(); n < N; ++n) {
        prims[offset + n] = quads[n];
    }

    QuadAndTriangleDataAdapter<Vec3s, Vec4I>
        mesh(indexSpacePoints.get(), numPoints, prims.get(), numPrimitives);

    return meshToVolume<GridType>(interrupter, mesh, xform,
        exBandWidth, inBandWidth, conversionFlags);
}


/// @internal This overload is enabled only for grids that do not have a scalar,
/// floating-point ValueType.
template<typename GridType, typename Interrupter>
inline typename std::enable_if<!std::is_floating_point<typename GridType::ValueType>::value,
    typename GridType::Ptr>::type
doMeshConversion(
    Interrupter&,
    const math::Transform& /*xform*/,
    const std::vector<Vec3s>& /*points*/,
    const std::vector<Vec3I>& /*triangles*/,
    const std::vector<Vec4I>& /*quads*/,
    float /*exBandWidth*/,
    float /*inBandWidth*/,
    bool /*unsignedDistanceField*/ = false)
{
    OPENVDB_THROW(TypeError,
        "mesh to volume conversion is supported only for scalar floating-point grids");
}

/// @endcond
//}


////////////////////////////////////////


template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    float halfWidth)
{
    util::NullInterrupter nullInterrupter;
    std::vector<Vec4I> quads(0);
    return doMeshConversion<GridType>(nullInterrupter, xform, points, triangles, quads,
        halfWidth, halfWidth);
}


template<typename GridType, typename Interrupter>
inline typename GridType::Ptr
meshToLevelSet(
    Interrupter& interrupter,
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    float halfWidth)
{
    std::vector<Vec4I> quads(0);
    return doMeshConversion<GridType>(interrupter, xform, points, triangles, quads,
        halfWidth, halfWidth);
}


template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec4I>& quads,
    float halfWidth)
{
    util::NullInterrupter nullInterrupter;
    std::vector<Vec3I> triangles(0);
    return doMeshConversion<GridType>(nullInterrupter, xform, points, triangles, quads,
        halfWidth, halfWidth);
}


template<typename GridType, typename Interrupter>
inline typename GridType::Ptr
meshToLevelSet(
    Interrupter& interrupter,
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec4I>& quads,
    float halfWidth)
{
    std::vector<Vec3I> triangles(0);
    return doMeshConversion<GridType>(interrupter, xform, points, triangles, quads,
        halfWidth, halfWidth);
}


template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float halfWidth)
{
    util::NullInterrupter nullInterrupter;
    return doMeshConversion<GridType>(nullInterrupter, xform, points, triangles, quads,
        halfWidth, halfWidth);
}


template<typename GridType, typename Interrupter>
inline typename GridType::Ptr
meshToLevelSet(
    Interrupter& interrupter,
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float halfWidth)
{
    return doMeshConversion<GridType>(interrupter, xform, points, triangles, quads,
        halfWidth, halfWidth);
}


template<typename GridType>
inline typename GridType::Ptr
meshToSignedDistanceField(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float exBandWidth,
    float inBandWidth)
{
    util::NullInterrupter nullInterrupter;
    return doMeshConversion<GridType>(nullInterrupter, xform, points, triangles,
        quads, exBandWidth, inBandWidth);
}


template<typename GridType, typename Interrupter>
inline typename GridType::Ptr
meshToSignedDistanceField(
    Interrupter& interrupter,
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float exBandWidth,
    float inBandWidth)
{
    return doMeshConversion<GridType>(interrupter, xform, points, triangles,
        quads, exBandWidth, inBandWidth);
}


template<typename GridType>
inline typename GridType::Ptr
meshToUnsignedDistanceField(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float bandWidth)
{
    util::NullInterrupter nullInterrupter;
    return doMeshConversion<GridType>(nullInterrupter, xform, points, triangles, quads,
        bandWidth, bandWidth, true);
}


template<typename GridType, typename Interrupter>
inline typename GridType::Ptr
meshToUnsignedDistanceField(
    Interrupter& interrupter,
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float bandWidth)
{
    return doMeshConversion<GridType>(interrupter, xform, points, triangles, quads,
        bandWidth, bandWidth, true);
}


////////////////////////////////////////////////////////////////////////////////


// Required by several of the tree nodes
inline std::ostream&
operator<<(std::ostream& ostr, const MeshToVoxelEdgeData::EdgeData& rhs)
{
    ostr << "{[ " << rhs.mXPrim << ", " << rhs.mXDist << "]";
    ostr << " [ " << rhs.mYPrim << ", " << rhs.mYDist << "]";
    ostr << " [ " << rhs.mZPrim << ", " << rhs.mZDist << "]}";
    return ostr;
}

// Required by math::Abs
inline MeshToVoxelEdgeData::EdgeData
Abs(const MeshToVoxelEdgeData::EdgeData& x)
{
    return x;
}


////////////////////////////////////////


class MeshToVoxelEdgeData::GenEdgeData
{
public:

    GenEdgeData(
        const std::vector<Vec3s>& pointList,
        const std::vector<Vec4I>& polygonList);

    void run(bool threaded = true);

    GenEdgeData(GenEdgeData& rhs, tbb::split);
    inline void operator() (const tbb::blocked_range<size_t> &range);
    inline void join(GenEdgeData& rhs);

    inline TreeType& tree() { return mTree; }

private:
    void operator=(const GenEdgeData&) {}

    struct Primitive { Vec3d a, b, c, d; Int32 index; };

    template<bool IsQuad>
    inline void voxelize(const Primitive&);

    template<bool IsQuad>
    inline bool evalPrimitive(const Coord&, const Primitive&);

    inline bool rayTriangleIntersection( const Vec3d& origin, const Vec3d& dir,
        const Vec3d& a, const Vec3d& b, const Vec3d& c, double& t);


    TreeType mTree;
    Accessor mAccessor;

    const std::vector<Vec3s>& mPointList;
    const std::vector<Vec4I>& mPolygonList;

    // Used internally for acceleration
    using IntTreeT = TreeType::ValueConverter<Int32>::Type;
    IntTreeT mLastPrimTree;
    tree::ValueAccessor<IntTreeT> mLastPrimAccessor;
}; // class MeshToVoxelEdgeData::GenEdgeData


inline
MeshToVoxelEdgeData::GenEdgeData::GenEdgeData(
    const std::vector<Vec3s>& pointList,
    const std::vector<Vec4I>& polygonList)
    : mTree(EdgeData())
    , mAccessor(mTree)
    , mPointList(pointList)
    , mPolygonList(polygonList)
    , mLastPrimTree(Int32(util::INVALID_IDX))
    , mLastPrimAccessor(mLastPrimTree)
{
}


inline
MeshToVoxelEdgeData::GenEdgeData::GenEdgeData(GenEdgeData& rhs, tbb::split)
    : mTree(EdgeData())
    , mAccessor(mTree)
    , mPointList(rhs.mPointList)
    , mPolygonList(rhs.mPolygonList)
    , mLastPrimTree(Int32(util::INVALID_IDX))
    , mLastPrimAccessor(mLastPrimTree)
{
}


inline void
MeshToVoxelEdgeData::GenEdgeData::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, mPolygonList.size()), *this);
    } else {
        (*this)(tbb::blocked_range<size_t>(0, mPolygonList.size()));
    }
}


inline void
MeshToVoxelEdgeData::GenEdgeData::join(GenEdgeData& rhs)
{
    using RootNodeType = TreeType::RootNodeType;
    using NodeChainType = RootNodeType::NodeChainType;
    static_assert(boost::mpl::size<NodeChainType>::value > 1, "expected tree height > 1");
    using InternalNodeType = boost::mpl::at<NodeChainType, boost::mpl::int_<1> >::type;

    Coord ijk;
    Index offset;

    rhs.mTree.clearAllAccessors();

    TreeType::LeafIter leafIt = rhs.mTree.beginLeaf();
    for ( ; leafIt; ++leafIt) {
        ijk = leafIt->origin();

        TreeType::LeafNodeType* lhsLeafPt = mTree.probeLeaf(ijk);

        if (!lhsLeafPt) {

            mAccessor.addLeaf(rhs.mAccessor.probeLeaf(ijk));
            InternalNodeType* node = rhs.mAccessor.getNode<InternalNodeType>();
            node->stealNode<TreeType::LeafNodeType>(ijk, EdgeData(), false);
            rhs.mAccessor.clear();

        } else {

            TreeType::LeafNodeType::ValueOnCIter it = leafIt->cbeginValueOn();
            for ( ; it; ++it) {

                offset = it.pos();
                const EdgeData& rhsValue = it.getValue();

                if (!lhsLeafPt->isValueOn(offset)) {
                    lhsLeafPt->setValueOn(offset, rhsValue);
                } else {

                    EdgeData& lhsValue = const_cast<EdgeData&>(lhsLeafPt->getValue(offset));

                    if (rhsValue.mXDist < lhsValue.mXDist) {
                        lhsValue.mXDist = rhsValue.mXDist;
                        lhsValue.mXPrim = rhsValue.mXPrim;
                    }

                    if (rhsValue.mYDist < lhsValue.mYDist) {
                        lhsValue.mYDist = rhsValue.mYDist;
                        lhsValue.mYPrim = rhsValue.mYPrim;
                    }

                    if (rhsValue.mZDist < lhsValue.mZDist) {
                        lhsValue.mZDist = rhsValue.mZDist;
                        lhsValue.mZPrim = rhsValue.mZPrim;
                    }

                }
            } // end value iteration
        }
    } // end leaf iteration
}


inline void
MeshToVoxelEdgeData::GenEdgeData::operator()(const tbb::blocked_range<size_t> &range)
{
    Primitive prim;

    for (size_t n = range.begin(); n < range.end(); ++n) {

        const Vec4I& verts = mPolygonList[n];

        prim.index = Int32(n);
        prim.a = Vec3d(mPointList[verts[0]]);
        prim.b = Vec3d(mPointList[verts[1]]);
        prim.c = Vec3d(mPointList[verts[2]]);

        if (util::INVALID_IDX != verts[3]) {
            prim.d = Vec3d(mPointList[verts[3]]);
            voxelize<true>(prim);
        } else {
            voxelize<false>(prim);
        }
    }
}


template<bool IsQuad>
inline void
MeshToVoxelEdgeData::GenEdgeData::voxelize(const Primitive& prim)
{
    std::deque<Coord> coordList;
    Coord ijk, nijk;

    ijk = Coord::floor(prim.a);
    coordList.push_back(ijk);

    evalPrimitive<IsQuad>(ijk, prim);

    while (!coordList.empty()) {

        ijk = coordList.back();
        coordList.pop_back();

        for (Int32 i = 0; i < 26; ++i) {
            nijk = ijk + util::COORD_OFFSETS[i];

            if (prim.index != mLastPrimAccessor.getValue(nijk)) {
                mLastPrimAccessor.setValue(nijk, prim.index);
                if(evalPrimitive<IsQuad>(nijk, prim)) coordList.push_back(nijk);
            }
        }
    }
}


template<bool IsQuad>
inline bool
MeshToVoxelEdgeData::GenEdgeData::evalPrimitive(const Coord& ijk, const Primitive& prim)
{
    Vec3d uvw, org(ijk[0], ijk[1], ijk[2]);
    bool intersecting = false;
    double t;

    EdgeData edgeData;
    mAccessor.probeValue(ijk, edgeData);

    // Evaluate first triangle
    double dist = (org -
        closestPointOnTriangleToPoint(prim.a, prim.c, prim.b, org, uvw)).lengthSqr();

    if (rayTriangleIntersection(org, Vec3d(1.0, 0.0, 0.0), prim.a, prim.c, prim.b, t)) {
        if (t < edgeData.mXDist) {
            edgeData.mXDist = float(t);
            edgeData.mXPrim = prim.index;
            intersecting = true;
        }
    }

    if (rayTriangleIntersection(org, Vec3d(0.0, 1.0, 0.0), prim.a, prim.c, prim.b, t)) {
        if (t < edgeData.mYDist) {
            edgeData.mYDist = float(t);
            edgeData.mYPrim = prim.index;
            intersecting = true;
        }
    }

    if (rayTriangleIntersection(org, Vec3d(0.0, 0.0, 1.0), prim.a, prim.c, prim.b, t)) {
        if (t < edgeData.mZDist) {
            edgeData.mZDist = float(t);
            edgeData.mZPrim = prim.index;
            intersecting = true;
        }
    }

    if (IsQuad) {
        // Split quad into a second triangle and calculate distance.
        double secondDist = (org -
            closestPointOnTriangleToPoint(prim.a, prim.d, prim.c, org, uvw)).lengthSqr();

        if (secondDist < dist) dist = secondDist;

        if (rayTriangleIntersection(org, Vec3d(1.0, 0.0, 0.0), prim.a, prim.d, prim.c, t)) {
            if (t < edgeData.mXDist) {
                edgeData.mXDist = float(t);
                edgeData.mXPrim = prim.index;
                intersecting = true;
            }
        }

        if (rayTriangleIntersection(org, Vec3d(0.0, 1.0, 0.0), prim.a, prim.d, prim.c, t)) {
            if (t < edgeData.mYDist) {
                edgeData.mYDist = float(t);
                edgeData.mYPrim = prim.index;
                intersecting = true;
            }
        }

        if (rayTriangleIntersection(org, Vec3d(0.0, 0.0, 1.0), prim.a, prim.d, prim.c, t)) {
            if (t < edgeData.mZDist) {
                edgeData.mZDist = float(t);
                edgeData.mZPrim = prim.index;
                intersecting = true;
            }
        }
    }

    if (intersecting) mAccessor.setValue(ijk, edgeData);

    return (dist < 0.86602540378443861);
}


inline bool
MeshToVoxelEdgeData::GenEdgeData::rayTriangleIntersection(
    const Vec3d& origin, const Vec3d& dir,
    const Vec3d& a, const Vec3d& b, const Vec3d& c,
    double& t)
{
    // Check if ray is parallel with triangle

    Vec3d e1 = b - a;
    Vec3d e2 = c - a;
    Vec3d s1 = dir.cross(e2);

    double divisor = s1.dot(e1);
    if (!(std::abs(divisor) > 0.0)) return false;

    // Compute barycentric coordinates

    double inv_divisor = 1.0 / divisor;
    Vec3d d = origin - a;
    double b1 = d.dot(s1) * inv_divisor;

    if (b1 < 0.0 || b1 > 1.0) return false;

    Vec3d s2 = d.cross(e1);
    double b2 = dir.dot(s2) * inv_divisor;

    if (b2 < 0.0 || (b1 + b2) > 1.0) return false;

    // Compute distance to intersection point

    t = e2.dot(s2) * inv_divisor;
    return (t < 0.0) ? false : true;
}


////////////////////////////////////////


inline
MeshToVoxelEdgeData::MeshToVoxelEdgeData()
    : mTree(EdgeData())
{
}


inline void
MeshToVoxelEdgeData::convert(
    const std::vector<Vec3s>& pointList,
    const std::vector<Vec4I>& polygonList)
{
    GenEdgeData converter(pointList, polygonList);
    converter.run();

    mTree.clear();
    mTree.merge(converter.tree());
}


inline void
MeshToVoxelEdgeData::getEdgeData(
    Accessor& acc,
    const Coord& ijk,
    std::vector<Vec3d>& points,
    std::vector<Index32>& primitives)
{
    EdgeData data;
    Vec3d point;

    Coord coord = ijk;

    if (acc.probeValue(coord, data)) {

        if (data.mXPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]) + data.mXDist;
            point[1] = double(coord[1]);
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mXPrim);
        }

        if (data.mYPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]) + data.mYDist;
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mYPrim);
        }

        if (data.mZPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]);
            point[2] = double(coord[2]) + data.mZDist;

            points.push_back(point);
            primitives.push_back(data.mZPrim);
        }

    }

    coord[0] += 1;

    if (acc.probeValue(coord, data)) {

        if (data.mYPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]) + data.mYDist;
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mYPrim);
        }

        if (data.mZPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]);
            point[2] = double(coord[2]) + data.mZDist;

            points.push_back(point);
            primitives.push_back(data.mZPrim);
        }
    }

    coord[2] += 1;

    if (acc.probeValue(coord, data)) {
        if (data.mYPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]) + data.mYDist;
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mYPrim);
        }
    }

    coord[0] -= 1;

    if (acc.probeValue(coord, data)) {

        if (data.mXPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]) + data.mXDist;
            point[1] = double(coord[1]);
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mXPrim);
        }

        if (data.mYPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]) + data.mYDist;
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mYPrim);
        }
    }


    coord[1] += 1;

    if (acc.probeValue(coord, data)) {

        if (data.mXPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]) + data.mXDist;
            point[1] = double(coord[1]);
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mXPrim);
        }
    }

    coord[2] -= 1;

    if (acc.probeValue(coord, data)) {

        if (data.mXPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]) + data.mXDist;
            point[1] = double(coord[1]);
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mXPrim);
        }

        if (data.mZPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]);
            point[2] = double(coord[2]) + data.mZDist;

            points.push_back(point);
            primitives.push_back(data.mZPrim);
        }
    }

    coord[0] += 1;

    if (acc.probeValue(coord, data)) {

        if (data.mZPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]);
            point[2] = double(coord[2]) + data.mZDist;

            points.push_back(point);
            primitives.push_back(data.mZPrim);
        }
    }
}


template<typename GridType, typename VecType>
inline typename GridType::Ptr
createLevelSetBox(const math::BBox<VecType>& bbox,
    const openvdb::math::Transform& xform,
    typename VecType::ValueType halfWidth)
{
    const Vec3s pmin = Vec3s(xform.worldToIndex(bbox.min()));
    const Vec3s pmax = Vec3s(xform.worldToIndex(bbox.max()));

    Vec3s points[8];
    points[0] = Vec3s(pmin[0], pmin[1], pmin[2]);
    points[1] = Vec3s(pmin[0], pmin[1], pmax[2]);
    points[2] = Vec3s(pmax[0], pmin[1], pmax[2]);
    points[3] = Vec3s(pmax[0], pmin[1], pmin[2]);
    points[4] = Vec3s(pmin[0], pmax[1], pmin[2]);
    points[5] = Vec3s(pmin[0], pmax[1], pmax[2]);
    points[6] = Vec3s(pmax[0], pmax[1], pmax[2]);
    points[7] = Vec3s(pmax[0], pmax[1], pmin[2]);

    Vec4I faces[6];
    faces[0] = Vec4I(0, 1, 2, 3); // bottom
    faces[1] = Vec4I(7, 6, 5, 4); // top
    faces[2] = Vec4I(4, 5, 1, 0); // front
    faces[3] = Vec4I(6, 7, 3, 2); // back
    faces[4] = Vec4I(0, 3, 7, 4); // left
    faces[5] = Vec4I(1, 5, 6, 2); // right

    QuadAndTriangleDataAdapter<Vec3s, Vec4I> mesh(points, 8, faces, 6);

    return meshToVolume<GridType>(mesh, xform, halfWidth, halfWidth);
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MESH_TO_VOLUME_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
