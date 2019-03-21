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

/// @file tools/VolumeToSpheres.h
///
/// @brief Fill a closed level set or fog volume with adaptively-sized spheres.

#ifndef OPENVDB_TOOLS_VOLUME_TO_SPHERES_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VOLUME_TO_SPHERES_HAS_BEEN_INCLUDED

#include <openvdb/tree/LeafManager.h>
#include <openvdb/math/Math.h>
#include "Morphology.h" // for erodeVoxels()
#include "PointScatter.h"
#include "LevelSetRebuild.h"
#include "LevelSetUtil.h"
#include "VolumeToMesh.h"

#include <boost/mpl/at.hpp>
#include <boost/mpl/int.hpp>
#include <boost/scoped_array.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <algorithm> // for std::min(), std::max()
#include <cmath> // for std::sqrt()
#include <limits> // for std::numeric_limits
#include <memory>
#include <random>
#include <utility> // for std::pair
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Fill a closed level set or fog volume with adaptively-sized spheres.
///
/// @param grid             a scalar grid that defines the surface to be filled with spheres
/// @param spheres          an output array of 4-tuples representing the fitted spheres<BR>
///                         The first three components of each tuple specify the sphere center,
///                         and the fourth specifies the radius.
///                         The spheres are ordered by radius, from largest to smallest.
/// @param sphereCount      lower and upper bounds on the number of spheres to be generated<BR>
///                         The actual number will be somewhere within the bounds.
/// @param overlapping      toggle to allow spheres to overlap/intersect
/// @param minRadius        the smallest allowable sphere size, in voxel units<BR>
/// @param maxRadius        the largest allowable sphere size, in voxel units
/// @param isovalue         the voxel value that determines the surface of the volume<BR>
///                         The default value of zero works for signed distance fields,
///                         while fog volumes require a larger positive value
///                         (0.5 is a good initial guess).
/// @param instanceCount    the number of interior points to consider for the sphere placement<BR>
///                         Increasing this count increases the chances of finding optimal
///                         sphere sizes.
/// @param interrupter      pointer to an object adhering to the util::NullInterrupter interface
///
/// @note The minimum sphere count takes precedence over the minimum radius.
template<typename GridT, typename InterrupterT = util::NullInterrupter>
inline void
fillWithSpheres(
    const GridT& grid,
    std::vector<openvdb::Vec4s>& spheres,
    const Vec2i& sphereCount = Vec2i(1, 50),
    bool overlapping = false,
    float minRadius = 1.0,
    float maxRadius = std::numeric_limits<float>::max(),
    float isovalue = 0.0,
    int instanceCount = 10000,
    InterrupterT* interrupter = nullptr);


/// @deprecated Use the @a sphereCount overload instead.
template<typename GridT, typename InterrupterT = util::NullInterrupter>
OPENVDB_DEPRECATED
inline void
fillWithSpheres(
    const GridT& grid,
    std::vector<openvdb::Vec4s>& spheres,
    int maxSphereCount,
    bool overlapping = false,
    float minRadius = 1.0,
    float maxRadius = std::numeric_limits<float>::max(),
    float isovalue = 0.0,
    int instanceCount = 10000,
    InterrupterT* interrupter = nullptr);


////////////////////////////////////////


/// @brief  Accelerated closest surface point queries for narrow band level sets
/// @details Supports queries that originate at arbitrary world-space locations,
/// is not confined to the narrow band region of the input volume geometry.
template<typename GridT>
class ClosestSurfacePoint
{
public:
    using Ptr = std::unique_ptr<ClosestSurfacePoint>;
    using TreeT = typename GridT::TreeType;
    using BoolTreeT = typename TreeT::template ValueConverter<bool>::Type;
    using Index32TreeT = typename TreeT::template ValueConverter<Index32>::Type;
    using Int16TreeT = typename TreeT::template ValueConverter<Int16>::Type;

    /// @brief Extract surface points and construct a spatial acceleration structure.
    ///
    /// @return a null pointer if the initialization fails for any reason,
    /// otherwise a unique pointer to a newly-allocated ClosestSurfacePoint object.
    ///
    /// @param grid         a scalar level set or fog volume
    /// @param isovalue     the voxel value that determines the surface of the volume
    ///                     The default value of zero works for signed distance fields,
    ///                     while fog volumes require a larger positive value
    ///                     (0.5 is a good initial guess).
    /// @param interrupter  pointer to an object adhering to the util::NullInterrupter interface.
    template<typename InterrupterT = util::NullInterrupter>
    static inline Ptr create(const GridT& grid, float isovalue = 0.0,
        InterrupterT* interrupter = nullptr);

    /// @brief Compute the distance from each input point to its closest surface point.
    /// @param points       input list of points in world space
    /// @param distances    output list of closest surface point distances
    inline bool search(const std::vector<Vec3R>& points, std::vector<float>& distances);

    /// @brief Overwrite each input point with its closest surface point.
    /// @param points       input/output list of points in world space
    /// @param distances    output list of closest surface point distances
    inline bool searchAndReplace(std::vector<Vec3R>& points, std::vector<float>& distances);

    /// @brief Tree accessor
    const Index32TreeT& indexTree() const { return *mIdxTreePt; }
    /// @brief Tree accessor
    const Int16TreeT& signTree() const { return *mSignTreePt; }

private:
    using Index32LeafT = typename Index32TreeT::LeafNodeType;
    using IndexRange = std::pair<size_t, size_t>;

    std::vector<Vec4R> mLeafBoundingSpheres, mNodeBoundingSpheres;
    std::vector<IndexRange> mLeafRanges;
    std::vector<const Index32LeafT*> mLeafNodes;
    PointList mSurfacePointList;
    size_t mPointListSize = 0, mMaxNodeLeafs = 0;
    typename Index32TreeT::Ptr mIdxTreePt;
    typename Int16TreeT::Ptr mSignTreePt;

    ClosestSurfacePoint() = default;
    template<typename InterrupterT = util::NullInterrupter>
    inline bool initialize(const GridT&, float isovalue, InterrupterT*);
    inline bool search(std::vector<Vec3R>&, std::vector<float>&, bool transformPoints);
};


////////////////////////////////////////


// Internal utility methods

namespace v2s_internal {

struct PointAccessor
{
    PointAccessor(std::vector<Vec3R>& points)
        : mPoints(points)
    {
    }

    void add(const Vec3R &pos)
    {
        mPoints.push_back(pos);
    }
private:
    std::vector<Vec3R>& mPoints;
};


template<typename Index32LeafT>
class LeafOp
{
public:

    LeafOp(std::vector<Vec4R>& leafBoundingSpheres,
        const std::vector<const Index32LeafT*>& leafNodes,
        const math::Transform& transform,
        const PointList& surfacePointList);

    void run(bool threaded = true);


    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    std::vector<Vec4R>& mLeafBoundingSpheres;
    const std::vector<const Index32LeafT*>& mLeafNodes;
    const math::Transform& mTransform;
    const PointList& mSurfacePointList;
};

template<typename Index32LeafT>
LeafOp<Index32LeafT>::LeafOp(
    std::vector<Vec4R>& leafBoundingSpheres,
    const std::vector<const Index32LeafT*>& leafNodes,
    const math::Transform& transform,
    const PointList& surfacePointList)
    : mLeafBoundingSpheres(leafBoundingSpheres)
    , mLeafNodes(leafNodes)
    , mTransform(transform)
    , mSurfacePointList(surfacePointList)
{
}

template<typename Index32LeafT>
void
LeafOp<Index32LeafT>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mLeafNodes.size()), *this);
    } else {
        (*this)(tbb::blocked_range<size_t>(0, mLeafNodes.size()));
    }
}

template<typename Index32LeafT>
void
LeafOp<Index32LeafT>::operator()(const tbb::blocked_range<size_t>& range) const
{
    typename Index32LeafT::ValueOnCIter iter;
    Vec3s avg;

    for (size_t n = range.begin(); n != range.end(); ++n) {
        avg[0] = 0.0;
        avg[1] = 0.0;
        avg[2] = 0.0;

        int count = 0;
        for (iter = mLeafNodes[n]->cbeginValueOn(); iter; ++iter) {
            avg += mSurfacePointList[iter.getValue()];
            ++count;
        }
        if (count > 1) avg *= float(1.0 / double(count));

        float maxDist = 0.0;
        for (iter = mLeafNodes[n]->cbeginValueOn(); iter; ++iter) {
            float tmpDist = (mSurfacePointList[iter.getValue()] - avg).lengthSqr();
            if (tmpDist > maxDist) maxDist = tmpDist;
        }

        Vec4R& sphere = mLeafBoundingSpheres[n];
        sphere[0] = avg[0];
        sphere[1] = avg[1];
        sphere[2] = avg[2];
        sphere[3] = std::sqrt(maxDist);
    }
}


class NodeOp
{
public:
    using IndexRange = std::pair<size_t, size_t>;

    NodeOp(std::vector<Vec4R>& nodeBoundingSpheres,
        const std::vector<IndexRange>& leafRanges,
        const std::vector<Vec4R>& leafBoundingSpheres);

    inline void run(bool threaded = true);

    inline void operator()(const tbb::blocked_range<size_t>&) const;

private:
    std::vector<Vec4R>& mNodeBoundingSpheres;
    const std::vector<IndexRange>& mLeafRanges;
    const std::vector<Vec4R>& mLeafBoundingSpheres;
};

inline
NodeOp::NodeOp(std::vector<Vec4R>& nodeBoundingSpheres,
    const std::vector<IndexRange>& leafRanges,
    const std::vector<Vec4R>& leafBoundingSpheres)
    : mNodeBoundingSpheres(nodeBoundingSpheres)
    , mLeafRanges(leafRanges)
    , mLeafBoundingSpheres(leafBoundingSpheres)
{
}

inline void
NodeOp::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mLeafRanges.size()), *this);
    } else {
        (*this)(tbb::blocked_range<size_t>(0, mLeafRanges.size()));
    }
}

inline void
NodeOp::operator()(const tbb::blocked_range<size_t>& range) const
{
    Vec3d avg, pos;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        avg[0] = 0.0;
        avg[1] = 0.0;
        avg[2] = 0.0;

        int count = int(mLeafRanges[n].second) - int(mLeafRanges[n].first);

        for (size_t i = mLeafRanges[n].first; i < mLeafRanges[n].second; ++i) {
            avg[0] += mLeafBoundingSpheres[i][0];
            avg[1] += mLeafBoundingSpheres[i][1];
            avg[2] += mLeafBoundingSpheres[i][2];
        }

        if (count > 1) avg *= float(1.0 / double(count));


        double maxDist = 0.0;

        for (size_t i = mLeafRanges[n].first; i < mLeafRanges[n].second; ++i) {
            pos[0] = mLeafBoundingSpheres[i][0];
            pos[1] = mLeafBoundingSpheres[i][1];
            pos[2] = mLeafBoundingSpheres[i][2];

            double tmpDist = (pos - avg).length() + mLeafBoundingSpheres[i][3];
            if (tmpDist > maxDist) maxDist = tmpDist;
        }

        Vec4R& sphere = mNodeBoundingSpheres[n];

        sphere[0] = avg[0];
        sphere[1] = avg[1];
        sphere[2] = avg[2];
        sphere[3] = maxDist;
    }
}


////////////////////////////////////////


template<typename Index32LeafT>
class ClosestPointDist
{
public:
    using IndexRange = std::pair<size_t, size_t>;

    ClosestPointDist(
        std::vector<Vec3R>& instancePoints,
        std::vector<float>& instanceDistances,
        const PointList& surfacePointList,
        const std::vector<const Index32LeafT*>& leafNodes,
        const std::vector<IndexRange>& leafRanges,
        const std::vector<Vec4R>& leafBoundingSpheres,
        const std::vector<Vec4R>& nodeBoundingSpheres,
        size_t maxNodeLeafs,
        bool transformPoints = false);


    void run(bool threaded = true);


    void operator()(const tbb::blocked_range<size_t>&) const;

private:

    void evalLeaf(size_t index, const Index32LeafT& leaf) const;
    void evalNode(size_t pointIndex, size_t nodeIndex) const;


    std::vector<Vec3R>& mInstancePoints;
    std::vector<float>& mInstanceDistances;

    const PointList& mSurfacePointList;

    const std::vector<const Index32LeafT*>& mLeafNodes;
    const std::vector<IndexRange>& mLeafRanges;
    const std::vector<Vec4R>& mLeafBoundingSpheres;
    const std::vector<Vec4R>& mNodeBoundingSpheres;

    std::vector<float> mLeafDistances, mNodeDistances;

    const bool mTransformPoints;
    size_t mClosestPointIndex;
};// ClosestPointDist


template<typename Index32LeafT>
ClosestPointDist<Index32LeafT>::ClosestPointDist(
    std::vector<Vec3R>& instancePoints,
    std::vector<float>& instanceDistances,
    const PointList& surfacePointList,
    const std::vector<const Index32LeafT*>& leafNodes,
    const std::vector<IndexRange>& leafRanges,
    const std::vector<Vec4R>& leafBoundingSpheres,
    const std::vector<Vec4R>& nodeBoundingSpheres,
    size_t maxNodeLeafs,
    bool transformPoints)
    : mInstancePoints(instancePoints)
    , mInstanceDistances(instanceDistances)
    , mSurfacePointList(surfacePointList)
    , mLeafNodes(leafNodes)
    , mLeafRanges(leafRanges)
    , mLeafBoundingSpheres(leafBoundingSpheres)
    , mNodeBoundingSpheres(nodeBoundingSpheres)
    , mLeafDistances(maxNodeLeafs, 0.0)
    , mNodeDistances(leafRanges.size(), 0.0)
    , mTransformPoints(transformPoints)
    , mClosestPointIndex(0)
{
}


template<typename Index32LeafT>
void
ClosestPointDist<Index32LeafT>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mInstancePoints.size()), *this);
    } else {
        (*this)(tbb::blocked_range<size_t>(0, mInstancePoints.size()));
    }
}

template<typename Index32LeafT>
void
ClosestPointDist<Index32LeafT>::evalLeaf(size_t index, const Index32LeafT& leaf) const
{
    typename Index32LeafT::ValueOnCIter iter;
    const Vec3s center = mInstancePoints[index];
    size_t& closestPointIndex = const_cast<size_t&>(mClosestPointIndex);

    for (iter = leaf.cbeginValueOn(); iter; ++iter) {

        const Vec3s& point = mSurfacePointList[iter.getValue()];
        float tmpDist = (point - center).lengthSqr();

        if (tmpDist < mInstanceDistances[index]) {
            mInstanceDistances[index] = tmpDist;
            closestPointIndex = iter.getValue();
        }
    }
}


template<typename Index32LeafT>
void
ClosestPointDist<Index32LeafT>::evalNode(size_t pointIndex, size_t nodeIndex) const
{
    if (nodeIndex >= mLeafRanges.size()) return;

    const Vec3R& pos = mInstancePoints[pointIndex];
    float minDist = mInstanceDistances[pointIndex];
    size_t minDistIdx = 0;
    Vec3R center;
    bool updatedDist = false;

    for (size_t i = mLeafRanges[nodeIndex].first, n = 0;
        i < mLeafRanges[nodeIndex].second; ++i, ++n)
    {
        float& distToLeaf = const_cast<float&>(mLeafDistances[n]);

        center[0] = mLeafBoundingSpheres[i][0];
        center[1] = mLeafBoundingSpheres[i][1];
        center[2] = mLeafBoundingSpheres[i][2];
        const auto radius = mLeafBoundingSpheres[i][3];

        distToLeaf = float(std::max(0.0, (pos - center).length() - radius));

        if (distToLeaf < minDist) {
            minDist = distToLeaf;
            minDistIdx = i;
            updatedDist = true;
        }
    }

    if (!updatedDist) return;

    evalLeaf(pointIndex, *mLeafNodes[minDistIdx]);

    for (size_t i = mLeafRanges[nodeIndex].first, n = 0;
        i < mLeafRanges[nodeIndex].second; ++i, ++n)
    {
        if (mLeafDistances[n] < mInstanceDistances[pointIndex] && i != minDistIdx) {
            evalLeaf(pointIndex, *mLeafNodes[i]);
        }
    }
}


template<typename Index32LeafT>
void
ClosestPointDist<Index32LeafT>::operator()(const tbb::blocked_range<size_t>& range) const
{
    Vec3R center;
    for (size_t n = range.begin(); n != range.end(); ++n) {

        const Vec3R& pos = mInstancePoints[n];
        float minDist = mInstanceDistances[n];
        size_t minDistIdx = 0;

        for (size_t i = 0, I = mNodeDistances.size(); i < I; ++i) {
            float& distToNode = const_cast<float&>(mNodeDistances[i]);

            center[0] = mNodeBoundingSpheres[i][0];
            center[1] = mNodeBoundingSpheres[i][1];
            center[2] = mNodeBoundingSpheres[i][2];
            const auto radius = mNodeBoundingSpheres[i][3];

            distToNode = float(std::max(0.0, (pos - center).length() - radius));

            if (distToNode < minDist) {
                minDist = distToNode;
                minDistIdx = i;
            }
        }

        evalNode(n, minDistIdx);

        for (size_t i = 0, I = mNodeDistances.size(); i < I; ++i) {
            if (mNodeDistances[i] < mInstanceDistances[n] && i != minDistIdx) {
                evalNode(n, i);
            }
        }

        mInstanceDistances[n] = std::sqrt(mInstanceDistances[n]);

        if (mTransformPoints) mInstancePoints[n] = mSurfacePointList[mClosestPointIndex];
    }
}


class UpdatePoints
{
public:
    UpdatePoints(
        const Vec4s& sphere,
        const std::vector<Vec3R>& points,
        std::vector<float>& distances,
        std::vector<unsigned char>& mask,
        bool overlapping);

    float radius() const { return mRadius; }
    int index() const { return mIndex; }

    inline void run(bool threaded = true);


    UpdatePoints(UpdatePoints&, tbb::split);
    inline void operator()(const tbb::blocked_range<size_t>& range);
    void join(const UpdatePoints& rhs)
    {
        if (rhs.mRadius > mRadius) {
            mRadius = rhs.mRadius;
            mIndex = rhs.mIndex;
        }
    }

private:
    const Vec4s& mSphere;
    const std::vector<Vec3R>& mPoints;
    std::vector<float>& mDistances;
    std::vector<unsigned char>& mMask;
    bool mOverlapping;
    float mRadius;
    int mIndex;
};

inline
UpdatePoints::UpdatePoints(
    const Vec4s& sphere,
    const std::vector<Vec3R>& points,
    std::vector<float>& distances,
    std::vector<unsigned char>& mask,
    bool overlapping)
    : mSphere(sphere)
    , mPoints(points)
    , mDistances(distances)
    , mMask(mask)
    , mOverlapping(overlapping)
    , mRadius(0.0)
    , mIndex(0)
{
}

inline
UpdatePoints::UpdatePoints(UpdatePoints& rhs, tbb::split)
    : mSphere(rhs.mSphere)
    , mPoints(rhs.mPoints)
    , mDistances(rhs.mDistances)
    , mMask(rhs.mMask)
    , mOverlapping(rhs.mOverlapping)
    , mRadius(rhs.mRadius)
    , mIndex(rhs.mIndex)
{
}

inline void
UpdatePoints::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, mPoints.size()), *this);
    } else {
        (*this)(tbb::blocked_range<size_t>(0, mPoints.size()));
    }
}

inline void
UpdatePoints::operator()(const tbb::blocked_range<size_t>& range)
{
    Vec3s pos;
    for (size_t n = range.begin(); n != range.end(); ++n) {
        if (mMask[n]) continue;

        pos.x() = float(mPoints[n].x()) - mSphere[0];
        pos.y() = float(mPoints[n].y()) - mSphere[1];
        pos.z() = float(mPoints[n].z()) - mSphere[2];

        float dist = pos.length();

        if (dist < mSphere[3]) {
            mMask[n] = 1;
            continue;
        }

        if (!mOverlapping) {
            mDistances[n] = std::min(mDistances[n], (dist - mSphere[3]));
        }

        if (mDistances[n] > mRadius) {
            mRadius = mDistances[n];
            mIndex = int(n);
        }
    }
}


} // namespace v2s_internal


////////////////////////////////////////


template<typename GridT, typename InterrupterT>
inline void
fillWithSpheres(
    const GridT& grid,
    std::vector<openvdb::Vec4s>& spheres,
    int maxSphereCount,
    bool overlapping,
    float minRadius,
    float maxRadius,
    float isovalue,
    int instanceCount,
    InterrupterT* interrupter)
{
    fillWithSpheres(grid, spheres, Vec2i(1, maxSphereCount), overlapping,
        minRadius, maxRadius, isovalue, instanceCount, interrupter);
}


template<typename GridT, typename InterrupterT>
inline void
fillWithSpheres(
    const GridT& grid,
    std::vector<openvdb::Vec4s>& spheres,
    const Vec2i& sphereCount,
    bool overlapping,
    float minRadius,
    float maxRadius,
    float isovalue,
    int instanceCount,
    InterrupterT* interrupter)
{
    spheres.clear();

    if (grid.empty()) return;

    const int
        minSphereCount = sphereCount[0],
        maxSphereCount = sphereCount[1];
    if ((minSphereCount > maxSphereCount) || (maxSphereCount < 1)) {
        OPENVDB_LOG_WARN("fillWithSpheres: minimum sphere count ("
            << minSphereCount << ") exceeds maximum count (" << maxSphereCount << ")");
        return;
    }
    spheres.reserve(maxSphereCount);

    auto gridPtr = grid.copy(); // shallow copy

    if (gridPtr->getGridClass() == GRID_LEVEL_SET) {
        // Clamp the isovalue to the level set's background value minus epsilon.
        // (In a valid narrow-band level set, all voxels, including background voxels,
        // have values less than or equal to the background value, so an isovalue
        // greater than or equal to the background value would produce a mask with
        // effectively infinite extent.)
        isovalue = std::min(isovalue,
            static_cast<float>(gridPtr->background() - math::Tolerance<float>::value()));
    } else if (gridPtr->getGridClass() == GRID_FOG_VOLUME) {
        // Clamp the isovalue of a fog volume between epsilon and one,
        // again to avoid a mask with infinite extent.  (Recall that
        // fog volume voxel values vary from zero outside to one inside.)
        isovalue = math::Clamp(isovalue, math::Tolerance<float>::value(), 1.f);
    }

    // ClosestSurfacePoint is inaccurate for small grids.
    // Resample the input grid if it is too small.
    auto numVoxels = gridPtr->activeVoxelCount();
    if (numVoxels < 10000) {
        const auto scale = 1.0 / math::Cbrt(2.0 * 10000.0 / double(numVoxels));
        auto scaledXform = gridPtr->transform().copy();
        scaledXform->preScale(scale);

        auto newGridPtr = levelSetRebuild(*gridPtr, isovalue,
            LEVEL_SET_HALF_WIDTH, LEVEL_SET_HALF_WIDTH, scaledXform.get(), interrupter);

        const auto newNumVoxels = newGridPtr->activeVoxelCount();
        if (newNumVoxels > numVoxels) {
            OPENVDB_LOG_DEBUG_RUNTIME("fillWithSpheres: resampled input grid from "
                << numVoxels << " voxel" << (numVoxels == 1 ? "" : "s")
                << " to " << newNumVoxels << " voxel" << (newNumVoxels == 1 ? "" : "s"));
            gridPtr = newGridPtr;
            numVoxels = newNumVoxels;
        }
    }

    const bool addNarrowBandPoints = (numVoxels < 10000);
    int instances = std::max(instanceCount, maxSphereCount);

    using TreeT = typename GridT::TreeType;
    using BoolTreeT = typename TreeT::template ValueConverter<bool>::Type;
    using Int16TreeT = typename TreeT::template ValueConverter<Int16>::Type;

    using RandGen = std::mersenne_twister_engine<uint32_t, 32, 351, 175, 19,
        0xccab8ee7, 11, 0xffffffff, 7, 0x31b6ab00, 15, 0xffe50000, 17, 1812433253>; // mt11213b
    RandGen mtRand(/*seed=*/0);

    const TreeT& tree = gridPtr->tree();
    math::Transform transform = gridPtr->transform();

    std::vector<Vec3R> instancePoints;
    {
        // Compute a mask of the voxels enclosed by the isosurface.
        typename Grid<BoolTreeT>::Ptr interiorMaskPtr;
        if (gridPtr->getGridClass() == GRID_LEVEL_SET) {
            interiorMaskPtr = sdfInteriorMask(*gridPtr, isovalue);
        } else {
            // For non-level-set grids, the interior mask comprises the active voxels.
            interiorMaskPtr = typename Grid<BoolTreeT>::Ptr(Grid<BoolTreeT>::create(false));
            interiorMaskPtr->setTransform(transform.copy());
            interiorMaskPtr->tree().topologyUnion(tree);
        }

        if (interrupter && interrupter->wasInterrupted()) return;

        // If the interior mask is small and eroding it results in an empty grid,
        // use the uneroded mask instead.  (But if the minimum sphere count is zero,
        // then eroding away the mask is acceptable.)
        if (!addNarrowBandPoints || (minSphereCount <= 0)) {
            erodeVoxels(interiorMaskPtr->tree(), 1);
        } else {
            auto& maskTree = interiorMaskPtr->tree();
            auto copyOfTree = StaticPtrCast<BoolTreeT>(maskTree.copy());
            erodeVoxels(maskTree, 1);
            if (maskTree.empty()) { interiorMaskPtr->setTree(copyOfTree); }
        }

        // Scatter candidate sphere centroids (instancePoints)
        instancePoints.reserve(instances);
        v2s_internal::PointAccessor ptnAcc(instancePoints);

        const auto scatterCount = Index64(addNarrowBandPoints ? (instances / 2) : instances);

        UniformPointScatter<v2s_internal::PointAccessor, RandGen, InterrupterT> scatter(
            ptnAcc, scatterCount, mtRand, 1.0, interrupter);
        scatter(*interiorMaskPtr);
    }

    if (interrupter && interrupter->wasInterrupted()) return;

    auto csp = ClosestSurfacePoint<GridT>::create(*gridPtr, isovalue, interrupter);
    if (!csp) return;

    // Add extra instance points in the interior narrow band.
    if (instancePoints.size() < size_t(instances)) {
        const Int16TreeT& signTree = csp->signTree();
        for (auto leafIt = signTree.cbeginLeaf(); leafIt; ++leafIt) {
            for (auto it = leafIt->cbeginValueOn(); it; ++it) {
                const int flags = int(it.getValue());
                if (!(volume_to_mesh_internal::EDGES & flags)
                    && (volume_to_mesh_internal::INSIDE & flags))
                {
                    instancePoints.push_back(transform.indexToWorld(it.getCoord()));
                }
                if (instancePoints.size() == size_t(instances)) break;
            }
            if (instancePoints.size() == size_t(instances)) break;
        }
    }

    if (interrupter && interrupter->wasInterrupted()) return;

    // Assign a radius to each candidate sphere.  The radius is the world-space
    // distance from the sphere's center to the closest surface point.
    std::vector<float> instanceRadius;
    if (!csp->search(instancePoints, instanceRadius)) return;

    float largestRadius = 0.0;
    int largestRadiusIdx = 0;
    for (size_t n = 0, N = instancePoints.size(); n < N; ++n) {
        if (instanceRadius[n] > largestRadius) {
            largestRadius = instanceRadius[n];
            largestRadiusIdx = int(n);
        }
    }

    std::vector<unsigned char> instanceMask(instancePoints.size(), 0);

    minRadius = float(minRadius * transform.voxelSize()[0]);
    maxRadius = float(maxRadius * transform.voxelSize()[0]);

    for (size_t s = 0, S = std::min(size_t(maxSphereCount), instancePoints.size()); s < S; ++s) {

        if (interrupter && interrupter->wasInterrupted()) return;

        largestRadius = std::min(maxRadius, largestRadius);

        if ((int(s) >= minSphereCount) && (largestRadius < minRadius)) break;

        const Vec4s sphere(
            float(instancePoints[largestRadiusIdx].x()),
            float(instancePoints[largestRadiusIdx].y()),
            float(instancePoints[largestRadiusIdx].z()),
            largestRadius);

        spheres.push_back(sphere);
        instanceMask[largestRadiusIdx] = 1;

        v2s_internal::UpdatePoints op(
            sphere, instancePoints, instanceRadius, instanceMask, overlapping);
        op.run();

        largestRadius = op.radius();
        largestRadiusIdx = op.index();
    }
} // fillWithSpheres


////////////////////////////////////////


template<typename GridT>
template<typename InterrupterT>
inline typename ClosestSurfacePoint<GridT>::Ptr
ClosestSurfacePoint<GridT>::create(const GridT& grid, float isovalue, InterrupterT* interrupter)
{
    auto csp = Ptr{new ClosestSurfacePoint};
    if (!csp->initialize(grid, isovalue, interrupter)) csp.reset();
    return csp;
}


template<typename GridT>
template<typename InterrupterT>
inline bool
ClosestSurfacePoint<GridT>::initialize(
    const GridT& grid, float isovalue, InterrupterT* interrupter)
{
    using Index32LeafManagerT = tree::LeafManager<Index32TreeT>;
    using ValueT = typename GridT::ValueType;

    const TreeT& tree = grid.tree();
    const math::Transform& transform = grid.transform();

    { // Extract surface point cloud

        BoolTreeT mask(false);
        volume_to_mesh_internal::identifySurfaceIntersectingVoxels(mask, tree, ValueT(isovalue));

        mSignTreePt.reset(new Int16TreeT(0));
        mIdxTreePt.reset(new Index32TreeT(std::numeric_limits<Index32>::max()));


        volume_to_mesh_internal::computeAuxiliaryData(
            *mSignTreePt, *mIdxTreePt, mask, tree, ValueT(isovalue));

        if (interrupter && interrupter->wasInterrupted()) return false;

        // count unique points

        using Int16LeafNodeType = typename Int16TreeT::LeafNodeType;
        using Index32LeafNodeType = typename Index32TreeT::LeafNodeType;

        std::vector<Int16LeafNodeType*> signFlagsLeafNodes;
        mSignTreePt->getNodes(signFlagsLeafNodes);

        const tbb::blocked_range<size_t> auxiliaryLeafNodeRange(0, signFlagsLeafNodes.size());

        boost::scoped_array<Index32> leafNodeOffsets(new Index32[signFlagsLeafNodes.size()]);

        tbb::parallel_for(auxiliaryLeafNodeRange,
            volume_to_mesh_internal::LeafNodePointCount<Int16LeafNodeType::LOG2DIM>
                (signFlagsLeafNodes, leafNodeOffsets));

        {
            Index32 pointCount = 0;
            for (size_t n = 0, N = signFlagsLeafNodes.size(); n < N; ++n) {
                const Index32 tmp = leafNodeOffsets[n];
                leafNodeOffsets[n] = pointCount;
                pointCount += tmp;
            }

            mPointListSize = size_t(pointCount);
            mSurfacePointList.reset(new Vec3s[mPointListSize]);
        }


        std::vector<Index32LeafNodeType*> pointIndexLeafNodes;
        mIdxTreePt->getNodes(pointIndexLeafNodes);

        tbb::parallel_for(auxiliaryLeafNodeRange, volume_to_mesh_internal::ComputePoints<TreeT>(
            mSurfacePointList.get(), tree, pointIndexLeafNodes,
            signFlagsLeafNodes, leafNodeOffsets, transform, ValueT(isovalue)));
    }

    if (interrupter && interrupter->wasInterrupted()) return false;

    Index32LeafManagerT idxLeafs(*mIdxTreePt);

    using Index32RootNodeT = typename Index32TreeT::RootNodeType;
    using Index32NodeChainT = typename Index32RootNodeT::NodeChainType;
    static_assert(boost::mpl::size<Index32NodeChainT>::value > 1,
        "expected tree depth greater than one");
    using Index32InternalNodeT =
        typename boost::mpl::at<Index32NodeChainT, boost::mpl::int_<1> >::type;

    typename Index32TreeT::NodeCIter nIt = mIdxTreePt->cbeginNode();
    nIt.setMinDepth(Index32TreeT::NodeCIter::LEAF_DEPTH - 1);
    nIt.setMaxDepth(Index32TreeT::NodeCIter::LEAF_DEPTH - 1);

    std::vector<const Index32InternalNodeT*> internalNodes;

    const Index32InternalNodeT* node = nullptr;
    for (; nIt; ++nIt) {
        nIt.getNode(node);
        if (node) internalNodes.push_back(node);
    }

    std::vector<IndexRange>().swap(mLeafRanges);
    mLeafRanges.resize(internalNodes.size());

    std::vector<const Index32LeafT*>().swap(mLeafNodes);
    mLeafNodes.reserve(idxLeafs.leafCount());

    typename Index32InternalNodeT::ChildOnCIter leafIt;
    mMaxNodeLeafs = 0;
    for (size_t n = 0, N = internalNodes.size(); n < N; ++n) {

        mLeafRanges[n].first = mLeafNodes.size();

        size_t leafCount = 0;
        for (leafIt = internalNodes[n]->cbeginChildOn(); leafIt; ++leafIt) {
            mLeafNodes.push_back(&(*leafIt));
            ++leafCount;
        }

        mMaxNodeLeafs = std::max(leafCount, mMaxNodeLeafs);

        mLeafRanges[n].second = mLeafNodes.size();
    }

    std::vector<Vec4R>().swap(mLeafBoundingSpheres);
    mLeafBoundingSpheres.resize(mLeafNodes.size());

    v2s_internal::LeafOp<Index32LeafT> leafBS(
        mLeafBoundingSpheres, mLeafNodes, transform, mSurfacePointList);
    leafBS.run();


    std::vector<Vec4R>().swap(mNodeBoundingSpheres);
    mNodeBoundingSpheres.resize(internalNodes.size());

    v2s_internal::NodeOp nodeBS(mNodeBoundingSpheres, mLeafRanges, mLeafBoundingSpheres);
    nodeBS.run();
    return true;
} // ClosestSurfacePoint::initialize


template<typename GridT>
inline bool
ClosestSurfacePoint<GridT>::search(std::vector<Vec3R>& points,
    std::vector<float>& distances, bool transformPoints)
{
    distances.clear();
    distances.resize(points.size(), std::numeric_limits<float>::infinity());

    v2s_internal::ClosestPointDist<Index32LeafT> cpd(points, distances, mSurfacePointList,
        mLeafNodes, mLeafRanges, mLeafBoundingSpheres, mNodeBoundingSpheres,
        mMaxNodeLeafs, transformPoints);

    cpd.run();

    return true;
}


template<typename GridT>
inline bool
ClosestSurfacePoint<GridT>::search(const std::vector<Vec3R>& points, std::vector<float>& distances)
{
    return search(const_cast<std::vector<Vec3R>& >(points), distances, false);
}


template<typename GridT>
inline bool
ClosestSurfacePoint<GridT>::searchAndReplace(std::vector<Vec3R>& points,
    std::vector<float>& distances)
{
    return search(points, distances, true);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_VOLUME_TO_MESH_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
