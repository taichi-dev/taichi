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

/// @file   PointIndexGrid.h
///
/// @brief  Space-partitioning acceleration structure for points. Partitions
///         the points into voxels to accelerate range and nearest neighbor
///         searches.
///
/// @note   Leaf nodes store a single point-index array and the voxels are only
///         integer offsets into that array. The actual points are never stored
///         in the acceleration structure, only offsets into an external array.
///
/// @author Mihai Alden

#ifndef OPENVDB_TOOLS_POINT_INDEX_GRID_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_INDEX_GRID_HAS_BEEN_INCLUDED

#include "PointPartitioner.h"

#include <openvdb/version.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/tree/Tree.h>

#include <tbb/atomic.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <algorithm> // for std::min(), std::max()
#include <cmath> // for std::sqrt()
#include <deque>
#include <iostream>
#include <type_traits> // for std::is_same
#include <utility> // for std::pair
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace tree {
template<Index, typename> struct SameLeafConfig; // forward declaration
}

namespace tools {

template<typename T, Index Log2Dim> struct PointIndexLeafNode; // forward declaration

/// Point index tree configured to match the default OpenVDB tree configuration
using PointIndexTree = tree::Tree<tree::RootNode<tree::InternalNode<tree::InternalNode
    <PointIndexLeafNode<PointIndex32, 3>, 4>, 5>>>;

/// Point index grid
using PointIndexGrid = Grid<PointIndexTree>;


////////////////////////////////////////


/// @interface PointArray
/// Expected interface for the PointArray container:
/// @code
/// template<typename VectorType>
/// struct PointArray
/// {
///     // The type used to represent world-space point positions
///     using PosType = VectorType;
///
///     // Return the number of points in the array
///     size_t size() const;
///
///     // Return the world-space position of the nth point in the array.
///     void getPos(size_t n, PosType& xyz) const;
/// };
/// @endcode


////////////////////////////////////////


/// @brief  Partition points into a point index grid to accelerate range and
///         nearest-neighbor searches.
///
/// @param points       world-space point array conforming to the PointArray interface
/// @param voxelSize    voxel size in world units
template<typename GridT, typename PointArrayT>
inline typename GridT::Ptr
createPointIndexGrid(const PointArrayT& points, double voxelSize);


/// @brief  Partition points into a point index grid to accelerate range and
///         nearest-neighbor searches.
///
/// @param points   world-space point array conforming to the PointArray interface
/// @param xform    world-to-index-space transform
template<typename GridT, typename PointArrayT>
inline typename GridT::Ptr
createPointIndexGrid(const PointArrayT& points, const math::Transform& xform);


/// @brief  Return @c true if the given point index grid represents a valid partitioning
///         of the given point array.
///
/// @param points   world-space point array conforming to the PointArray interface
/// @param grid     point index grid to validate
template<typename PointArrayT, typename GridT>
inline bool
isValidPartition(const PointArrayT& points, const GridT& grid);


/// Repartition the @a points if needed, otherwise return the input @a grid.
template<typename GridT, typename PointArrayT>
inline typename GridT::ConstPtr
getValidPointIndexGrid(const PointArrayT& points, const typename GridT::ConstPtr& grid);

/// Repartition the @a points if needed, otherwise return the input @a grid.
template<typename GridT, typename PointArrayT>
inline typename GridT::Ptr
getValidPointIndexGrid(const PointArrayT& points, const typename GridT::Ptr& grid);


////////////////////////////////////////


/// Accelerated range and nearest-neighbor searches for point index grids
template<typename TreeType = PointIndexTree>
struct PointIndexIterator
{
    using ConstAccessor = tree::ValueAccessor<const TreeType>;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using ValueType = typename TreeType::ValueType;


    PointIndexIterator();
    PointIndexIterator(const PointIndexIterator& rhs);
    PointIndexIterator& operator=(const PointIndexIterator& rhs);


    /// @brief Construct an iterator over the indices of the points contained in voxel (i, j, k).
    /// @param ijk  the voxel containing the points over which to iterate
    /// @param acc  an accessor for the grid or tree that holds the point indices
    PointIndexIterator(const Coord& ijk, ConstAccessor& acc);


    /// @brief Construct an iterator over the indices of the points contained in
    ///        the given bounding box.
    /// @param bbox  the bounding box of the voxels containing the points over which to iterate
    /// @param acc   an accessor for the grid or tree that holds the point indices
    /// @note  The range of the @a bbox is inclusive. Thus, a bounding box with
    ///        min = max is not empty but rather encloses a single voxel.
    PointIndexIterator(const CoordBBox& bbox, ConstAccessor& acc);


    /// @brief Clear the iterator and update it with the result of the given voxel query.
    /// @param ijk  the voxel containing the points over which to iterate
    /// @param acc  an accessor for the grid or tree that holds the point indices
    void searchAndUpdate(const Coord& ijk, ConstAccessor& acc);


    /// @brief Clear the iterator and update it with the result of the given voxel region query.
    /// @param bbox  the bounding box of the voxels containing the points over which to iterate
    /// @param acc   an accessor for the grid or tree that holds the point indices
    /// @note  The range of the @a bbox is inclusive. Thus, a bounding box with
    ///        min = max is not empty but rather encloses a single voxel.
    void searchAndUpdate(const CoordBBox& bbox, ConstAccessor& acc);


    /// @brief Clear the iterator and update it with the result of the given
    ///        index-space bounding box query.
    /// @param bbox     index-space bounding box
    /// @param acc      an accessor for the grid or tree that holds the point indices
    /// @param points   world-space point array conforming to the PointArray interface
    /// @param xform    linear, uniform-scale transform (i.e., cubical voxels)
    template<typename PointArray>
    void searchAndUpdate(const BBoxd& bbox, ConstAccessor& acc,
        const PointArray& points, const math::Transform& xform);


    /// @brief Clear the iterator and update it with the result of the given
    ///        index-space radial query.
    /// @param center   index-space center
    /// @param radius   index-space radius
    /// @param acc      an accessor for the grid or tree that holds the point indices
    /// @param points   world-space point array conforming to the PointArray interface
    /// @param xform    linear, uniform-scale transform (i.e., cubical voxels)
    /// @param subvoxelAccuracy  if true, check individual points against the search region,
    ///                 otherwise return all points that reside in voxels that are inside
    ///                 or intersect the search region
    template<typename PointArray>
    void searchAndUpdate(const Vec3d& center, double radius, ConstAccessor& acc,
        const PointArray& points, const math::Transform& xform, bool subvoxelAccuracy = true);


    /// @brief Clear the iterator and update it with the result of the given
    ///        world-space bounding box query.
    /// @param bbox     world-space bounding box
    /// @param acc      an accessor for the grid or tree that holds the point indices
    /// @param points   world-space point array conforming to the PointArray interface
    /// @param xform    linear, uniform-scale transform (i.e., cubical voxels)
    template<typename PointArray>
    void worldSpaceSearchAndUpdate(const BBoxd& bbox, ConstAccessor& acc,
        const PointArray& points, const math::Transform& xform);


    /// @brief Clear the iterator and update it with the result of the given
    ///        world-space radial query.
    /// @param center   world-space center
    /// @param radius   world-space radius
    /// @param acc      an accessor for the grid or tree that holds the point indices
    /// @param points   world-space point array conforming to the PointArray interface
    /// @param xform    linear, uniform-scale transform (i.e., cubical voxels)
    /// @param subvoxelAccuracy  if true, check individual points against the search region,
    ///                 otherwise return all points that reside in voxels that are inside
    ///                 or intersect the search region
    template<typename PointArray>
    void worldSpaceSearchAndUpdate(const Vec3d& center, double radius, ConstAccessor& acc,
        const PointArray& points, const math::Transform& xform, bool subvoxelAccuracy = true);


    /// Reset the iterator to point to the first item.
    void reset();

    /// Return a const reference to the item to which this iterator is pointing.
    const ValueType& operator*() const { return *mRange.first; }

    /// @{
    /// @brief  Return @c true if this iterator is not yet exhausted.
    bool test() const { return mRange.first < mRange.second || mIter != mRangeList.end(); }
    operator bool() const { return this->test(); }
    /// @}

    /// Advance iterator to next item.
    void increment();

    /// Advance iterator to next item.
    void operator++() { this->increment(); }


    /// @brief Advance iterator to next item.
    /// @return @c true if this iterator is not yet exhausted.
    bool next();

    /// Return the number of point indices in the iterator range.
    size_t size() const;

    /// Return @c true if both iterators point to the same element.
    bool operator==(const PointIndexIterator& p) const { return mRange.first == p.mRange.first; }
    bool operator!=(const PointIndexIterator& p) const { return !this->operator==(p); }


private:
    using Range = std::pair<const ValueType*, const ValueType*>;
    using RangeDeque = std::deque<Range>;
    using RangeDequeCIter = typename RangeDeque::const_iterator;
    using IndexArray = std::unique_ptr<ValueType[]>;

    void clear();

    // Primary index collection
    Range           mRange;
    RangeDeque      mRangeList;
    RangeDequeCIter mIter;
    // Secondary index collection
    IndexArray      mIndexArray;
    size_t          mIndexArraySize;
}; // struct PointIndexIterator


/// @brief Selectively extract and filter point data using a custom filter operator.
///
/// @par FilterType example:
/// @interface FilterType
/// @code
/// template<typename T>
/// struct WeightedAverageAccumulator {
///   using ValueType = T;
///
///   WeightedAverageAccumulator(T const * const array, const T radius)
///     : mValues(array), mInvRadius(1.0/radius), mWeightSum(0.0), mValueSum(0.0) {}
///
///   void reset() { mWeightSum = mValueSum = T(0.0); }
///
///   // the following method is invoked by the PointIndexFilter
///   void operator()(const T distSqr, const size_t pointIndex) {
///     const T weight = T(1.0) - openvdb::math::Sqrt(distSqr) * mInvRadius;
///     mWeightSum += weight;
///     mValueSum += weight * mValues[pointIndex];
///   }
///
///   T result() const { return mWeightSum > T(0.0) ? mValueSum / mWeightSum : T(0.0); }
///
/// private:
///   T const * const mValues;
///   const T mInvRadius;
///   T mWeightSum, mValueSum;
/// }; // struct WeightedAverageAccumulator
/// @endcode
template<typename PointArray, typename TreeType = PointIndexTree>
struct PointIndexFilter
{
    using PosType = typename PointArray::PosType;
    using ScalarType = typename PosType::value_type;
    using ConstAccessor = tree::ValueAccessor<const TreeType>;

    /// @brief Constructor
    /// @param points   world-space point array conforming to the PointArray interface
    /// @param tree     a point index tree
    /// @param xform    linear, uniform-scale transform (i.e., cubical voxels)
    PointIndexFilter(const PointArray& points, const TreeType& tree, const math::Transform& xform);

    /// Thread safe copy constructor
    PointIndexFilter(const PointIndexFilter& rhs);

    /// @brief  Perform a radial search query and apply the given filter
    ///         operator to the selected points.
    /// @param center  world-space center
    /// @param radius  world-space radius
    /// @param op      custom filter operator (see the FilterType example for interface details)
    template<typename FilterType>
    void searchAndApply(const PosType& center, ScalarType radius, FilterType& op);

private:
    PointArray const * const mPoints;
    ConstAccessor mAcc;
    const math::Transform mXform;
    const ScalarType mInvVoxelSize;
    PointIndexIterator<TreeType> mIter;
}; // struct PointIndexFilter


////////////////////////////////////////

// Internal operators and implementation details


namespace point_index_grid_internal {

template<typename PointArrayT>
struct ValidPartitioningOp
{
    ValidPartitioningOp(tbb::atomic<bool>& hasChanged,
        const PointArrayT& points, const math::Transform& xform)
        : mPoints(&points)
        , mTransform(&xform)
        , mHasChanged(&hasChanged)
    {
    }

    template <typename LeafT>
    void operator()(LeafT &leaf, size_t /*leafIndex*/) const
    {
        if ((*mHasChanged)) {
            tbb::task::self().cancel_group_execution();
            return;
        }

        using IndexArrayT = typename LeafT::IndexArray;
        using IndexT = typename IndexArrayT::value_type;
        using PosType = typename PointArrayT::PosType;

        typename LeafT::ValueOnCIter iter;
        Coord voxelCoord;
        PosType point;

        const IndexT
            *begin = static_cast<IndexT*>(nullptr),
            *end = static_cast<IndexT*>(nullptr);

        for (iter = leaf.cbeginValueOn(); iter; ++iter) {

            if ((*mHasChanged)) break;

            voxelCoord = iter.getCoord();
            leaf.getIndices(iter.pos(), begin, end);

            while (begin < end) {

                mPoints->getPos(*begin, point);
                if (voxelCoord != mTransform->worldToIndexCellCentered(point)) {
                    mHasChanged->fetch_and_store(true);
                    break;
                }

                ++begin;
            }
        }
    }

private:
    PointArrayT         const * const mPoints;
    math::Transform     const * const mTransform;
    tbb::atomic<bool>         * const mHasChanged;
};


template<typename LeafNodeT>
struct PopulateLeafNodesOp
{
    using IndexT = uint32_t;
    using Partitioner = PointPartitioner<IndexT, LeafNodeT::LOG2DIM>;

    PopulateLeafNodesOp(std::unique_ptr<LeafNodeT*[]>& leafNodes,
        const Partitioner& partitioner)
        : mLeafNodes(leafNodes.get())
        , mPartitioner(&partitioner)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        using VoxelOffsetT = typename Partitioner::VoxelOffsetType;

        size_t maxPointCount = 0;
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            maxPointCount = std::max(maxPointCount, mPartitioner->indices(n).size());
        }

        const IndexT voxelCount = LeafNodeT::SIZE;

        // allocate histogram buffers
        std::unique_ptr<VoxelOffsetT[]> offsets{new VoxelOffsetT[maxPointCount]};
        std::unique_ptr<IndexT[]> histogram{new IndexT[voxelCount]};

        VoxelOffsetT const * const voxelOffsets = mPartitioner->voxelOffsets().get();

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            LeafNodeT* node = new LeafNodeT();
            node->setOrigin(mPartitioner->origin(n));

            typename Partitioner::IndexIterator it = mPartitioner->indices(n);

            const size_t pointCount = it.size();
            IndexT const * const indices = &*it;

            // local copy of voxel offsets.
            for (IndexT i = 0; i < pointCount; ++i) {
                offsets[i] = voxelOffsets[ indices[i] ];
            }

            // compute voxel-offset histogram
            memset(&histogram[0], 0, voxelCount * sizeof(IndexT));
            for (IndexT i = 0; i < pointCount; ++i) {
                ++histogram[ offsets[i] ];
            }

            typename LeafNodeT::NodeMaskType& mask = node->getValueMask();
            typename LeafNodeT::Buffer& buffer = node->buffer();

            // scan histogram (all-prefix-sums)
            IndexT count = 0, startOffset;
            for (int i = 0; i < int(voxelCount); ++i) {
                if (histogram[i] > 0) {
                    startOffset = count;
                    count += histogram[i];
                    histogram[i] = startOffset;
                    mask.setOn(i);
                }
                buffer.setValue(i, count);
            }

            // allocate point-index array
            node->indices().resize(pointCount);
            typename LeafNodeT::ValueType * const orderedIndices = node->indices().data();

            // rank and permute
            for (IndexT i = 0; i < pointCount; ++i) {
                orderedIndices[ histogram[ offsets[i] ]++ ] = indices[i];
            }

            mLeafNodes[n] = node;
        }
    }

    //////////

    LeafNodeT*        * const mLeafNodes;
    Partitioner const * const mPartitioner;
};


/// Construct a @c PointIndexTree
template<typename TreeType, typename PointArray>
inline void
constructPointTree(TreeType& tree, const math::Transform& xform, const PointArray& points)
{
    using LeafType = typename TreeType::LeafNodeType;

    std::unique_ptr<LeafType*[]> leafNodes;
    size_t leafNodeCount = 0;

    {
        // Important:  Do not disable the cell-centered transform in the PointPartitioner.
        //             This interpretation is assumed in the PointIndexGrid and all related
        //             search algorithms.
        PointPartitioner<uint32_t, LeafType::LOG2DIM> partitioner;
        partitioner.construct(points, xform, /*voxelOrder=*/false, /*recordVoxelOffsets=*/true);

        if (!partitioner.usingCellCenteredTransform()) {
            OPENVDB_THROW(LookupError, "The PointIndexGrid requires a "
                "cell-centered transform.");
        }

        leafNodeCount = partitioner.size();
        leafNodes.reset(new LeafType*[leafNodeCount]);

        const tbb::blocked_range<size_t> range(0, leafNodeCount);
        tbb::parallel_for(range, PopulateLeafNodesOp<LeafType>(leafNodes, partitioner));
    }

    tree::ValueAccessor<TreeType> acc(tree);
    for (size_t n = 0; n < leafNodeCount; ++n) {
        acc.addLeaf(leafNodes[n]);
    }
}


////////////////////////////////////////


template<typename T>
inline void
dequeToArray(const std::deque<T>& d, std::unique_ptr<T[]>& a, size_t& size)
{
    size = d.size();
    a.reset(new T[size]);
    typename std::deque<T>::const_iterator it = d.begin(), itEnd = d.end();
    T* item = a.get();
    for ( ; it != itEnd; ++it, ++item) *item = *it;
}


inline void
constructExclusiveRegions(std::vector<CoordBBox>& regions,
    const CoordBBox& bbox, const CoordBBox& ibox)
{
    regions.clear();
    regions.reserve(6);
    Coord cmin = ibox.min();
    Coord cmax = ibox.max();

    // left-face bbox
    regions.push_back(bbox);
    regions.back().max().z() = cmin.z();

    // right-face bbox
    regions.push_back(bbox);
    regions.back().min().z() = cmax.z();

    --cmax.z(); // accounting for cell centered bucketing.
    ++cmin.z();

    // front-face bbox
    regions.push_back(bbox);
    CoordBBox* lastRegion = &regions.back();
    lastRegion->min().z() = cmin.z();
    lastRegion->max().z() = cmax.z();
    lastRegion->max().x() = cmin.x();

    // back-face bbox
    regions.push_back(*lastRegion);
    lastRegion = &regions.back();
    lastRegion->min().x() = cmax.x();
    lastRegion->max().x() = bbox.max().x();

    --cmax.x();
    ++cmin.x();

    // bottom-face bbox
    regions.push_back(*lastRegion);
    lastRegion = &regions.back();
    lastRegion->min().x() = cmin.x();
    lastRegion->max().x() = cmax.x();
    lastRegion->max().y() = cmin.y();

    // top-face bbox
    regions.push_back(*lastRegion);
    lastRegion = &regions.back();
    lastRegion->min().y() = cmax.y();
    lastRegion->max().y() = bbox.max().y();
}


template<typename PointArray, typename IndexT>
struct BBoxFilter
{
    using PosType = typename PointArray::PosType;
    using ScalarType = typename PosType::value_type;
    using Range = std::pair<const IndexT*, const IndexT*>;
    using RangeDeque = std::deque<Range>;
    using IndexDeque = std::deque<IndexT>;

    BBoxFilter(RangeDeque& ranges, IndexDeque& indices, const BBoxd& bbox,
        const PointArray& points, const math::Transform& xform)
        : mRanges(ranges)
        , mIndices(indices)
        , mRegion(bbox)
        , mPoints(points)
        , mMap(*xform.baseMap())
    {
    }

    template <typename LeafNodeType>
    void filterLeafNode(const LeafNodeType& leaf)
    {
        typename LeafNodeType::ValueOnCIter iter;
        const IndexT
            *begin = static_cast<IndexT*>(nullptr),
            *end = static_cast<IndexT*>(nullptr);
        for (iter = leaf.cbeginValueOn(); iter; ++iter) {
            leaf.getIndices(iter.pos(), begin, end);
            filterVoxel(iter.getCoord(), begin, end);
        }
    }

    void filterVoxel(const Coord&, const IndexT* begin, const IndexT* end)
    {
        PosType vec;

        for (; begin < end; ++begin) {
            mPoints.getPos(*begin, vec);

            if (mRegion.isInside(mMap.applyInverseMap(vec))) {
                mIndices.push_back(*begin);
            }
        }
    }

private:
    RangeDeque& mRanges;
    IndexDeque& mIndices;
    const BBoxd mRegion;
    const PointArray& mPoints;
    const math::MapBase& mMap;
};


template<typename PointArray, typename IndexT>
struct RadialRangeFilter
{
    using PosType = typename PointArray::PosType;
    using ScalarType = typename PosType::value_type;
    using Range = std::pair<const IndexT*, const IndexT*>;
    using RangeDeque = std::deque<Range>;
    using IndexDeque = std::deque<IndexT>;

    RadialRangeFilter(RangeDeque& ranges, IndexDeque& indices, const Vec3d& xyz, double radius,
        const PointArray& points, const math::Transform& xform,
        const double leafNodeDim, const bool subvoxelAccuracy)
        : mRanges(ranges)
        , mIndices(indices)
        , mCenter(xyz)
        , mWSCenter(xform.indexToWorld(xyz))
        , mVoxelDist1(ScalarType(0.0))
        , mVoxelDist2(ScalarType(0.0))
        , mLeafNodeDist1(ScalarType(0.0))
        , mLeafNodeDist2(ScalarType(0.0))
        , mWSRadiusSqr(ScalarType(radius * xform.voxelSize()[0]))
        , mPoints(points)
        , mSubvoxelAccuracy(subvoxelAccuracy)
    {
        const ScalarType voxelRadius = ScalarType(std::sqrt(3.0) * 0.5);
        mVoxelDist1 = voxelRadius + ScalarType(radius);
        mVoxelDist1 *= mVoxelDist1;

        if (radius > voxelRadius) {
            mVoxelDist2 = ScalarType(radius) - voxelRadius;
            mVoxelDist2 *= mVoxelDist2;
        }

        const ScalarType leafNodeRadius = ScalarType(leafNodeDim * std::sqrt(3.0) * 0.5);
        mLeafNodeDist1 = leafNodeRadius + ScalarType(radius);
        mLeafNodeDist1 *= mLeafNodeDist1;

        if (radius > leafNodeRadius) {
            mLeafNodeDist2 = ScalarType(radius) - leafNodeRadius;
            mLeafNodeDist2 *= mLeafNodeDist2;
        }

        mWSRadiusSqr *= mWSRadiusSqr;
    }

    template <typename LeafNodeType>
    void filterLeafNode(const LeafNodeType& leaf)
    {
        {
            const Coord& ijk = leaf.origin();
            PosType vec;
            vec[0] = ScalarType(ijk[0]);
            vec[1] = ScalarType(ijk[1]);
            vec[2] = ScalarType(ijk[2]);
            vec += ScalarType(LeafNodeType::DIM - 1) * 0.5;
            vec -= mCenter;

            const ScalarType dist = vec.lengthSqr();
            if (dist > mLeafNodeDist1) return;

            if (mLeafNodeDist2 > 0.0 && dist < mLeafNodeDist2) {
                const IndexT* begin = &leaf.indices().front();
                mRanges.push_back(Range(begin, begin + leaf.indices().size()));
                return;
            }
        }

        typename LeafNodeType::ValueOnCIter iter;
        const IndexT
            *begin = static_cast<IndexT*>(nullptr),
            *end = static_cast<IndexT*>(nullptr);
        for (iter = leaf.cbeginValueOn(); iter; ++iter) {
            leaf.getIndices(iter.pos(), begin, end);
            filterVoxel(iter.getCoord(), begin, end);
        }
    }

    void filterVoxel(const Coord& ijk, const IndexT* begin, const IndexT* end)
    {
        PosType vec;

        {
            vec[0] = mCenter[0] - ScalarType(ijk[0]);
            vec[1] = mCenter[1] - ScalarType(ijk[1]);
            vec[2] = mCenter[2] - ScalarType(ijk[2]);

            const ScalarType dist = vec.lengthSqr();
            if (dist > mVoxelDist1) return;

            if (!mSubvoxelAccuracy || (mVoxelDist2 > 0.0 && dist < mVoxelDist2)) {
                if (!mRanges.empty() && mRanges.back().second == begin) {
                    mRanges.back().second = end;
                } else {
                    mRanges.push_back(Range(begin, end));
                }
                return;
            }
        }


        while (begin < end) {
            mPoints.getPos(*begin, vec);
            vec = mWSCenter - vec;

            if (vec.lengthSqr() < mWSRadiusSqr) {
                mIndices.push_back(*begin);
            }
            ++begin;
        }
    }

private:
    RangeDeque& mRanges;
    IndexDeque& mIndices;
    const PosType mCenter, mWSCenter;
    ScalarType mVoxelDist1, mVoxelDist2, mLeafNodeDist1, mLeafNodeDist2, mWSRadiusSqr;
    const PointArray& mPoints;
    const bool mSubvoxelAccuracy;
}; // struct RadialRangeFilter


////////////////////////////////////////


template<typename RangeFilterType, typename LeafNodeType>
inline void
filteredPointIndexSearchVoxels(RangeFilterType& filter,
    const LeafNodeType& leaf, const Coord& min, const Coord& max)
{
    using PointIndexT = typename LeafNodeType::ValueType;
    Index xPos(0), yPos(0), pos(0);
    Coord ijk(0);

    const PointIndexT* dataPtr = &leaf.indices().front();
    PointIndexT beginOffset, endOffset;

    for (ijk[0] = min[0]; ijk[0] <= max[0]; ++ijk[0]) {
        xPos = (ijk[0] & (LeafNodeType::DIM - 1u)) << (2 * LeafNodeType::LOG2DIM);
        for (ijk[1] = min[1]; ijk[1] <= max[1]; ++ijk[1]) {
            yPos = xPos + ((ijk[1] & (LeafNodeType::DIM - 1u)) << LeafNodeType::LOG2DIM);
            for (ijk[2] = min[2]; ijk[2] <= max[2]; ++ijk[2]) {
                pos = yPos + (ijk[2] & (LeafNodeType::DIM - 1u));

                beginOffset = (pos == 0 ? PointIndexT(0) : leaf.getValue(pos - 1));
                endOffset = leaf.getValue(pos);

                if (endOffset > beginOffset) {
                    filter.filterVoxel(ijk, dataPtr + beginOffset, dataPtr + endOffset);
                }
            }
        }
    }
}


template<typename RangeFilterType, typename ConstAccessor>
inline void
filteredPointIndexSearch(RangeFilterType& filter, ConstAccessor& acc, const CoordBBox& bbox)
{
    using LeafNodeType = typename ConstAccessor::TreeType::LeafNodeType;
    Coord ijk(0), ijkMax(0), ijkA(0), ijkB(0);
    const Coord leafMin = bbox.min() & ~(LeafNodeType::DIM - 1);
    const Coord leafMax = bbox.max() & ~(LeafNodeType::DIM - 1);

    for (ijk[0] = leafMin[0]; ijk[0] <= leafMax[0]; ijk[0] += LeafNodeType::DIM) {
        for (ijk[1] = leafMin[1]; ijk[1] <= leafMax[1]; ijk[1] += LeafNodeType::DIM) {
            for (ijk[2] = leafMin[2]; ijk[2] <= leafMax[2]; ijk[2] += LeafNodeType::DIM) {

                if (const LeafNodeType* leaf = acc.probeConstLeaf(ijk)) {
                    ijkMax = ijk;
                    ijkMax.offset(LeafNodeType::DIM - 1);

                    // intersect leaf bbox with search region.
                    ijkA = Coord::maxComponent(bbox.min(), ijk);
                    ijkB = Coord::minComponent(bbox.max(), ijkMax);

                    if (ijkA != ijk || ijkB != ijkMax) {
                        filteredPointIndexSearchVoxels(filter, *leaf, ijkA, ijkB);
                    } else { // leaf bbox is inside the search region
                        filter.filterLeafNode(*leaf);
                    }
                }
            }
        }
    }
}


////////////////////////////////////////


template<typename RangeDeque, typename LeafNodeType>
inline void
pointIndexSearchVoxels(RangeDeque& rangeList,
    const LeafNodeType& leaf, const Coord& min, const Coord& max)
{
    using PointIndexT = typename LeafNodeType::ValueType;
    using IntT = typename PointIndexT::IntType;
    using Range = typename RangeDeque::value_type;

    Index xPos(0), pos(0), zStride = Index(max[2] - min[2]);
    const PointIndexT* dataPtr = &leaf.indices().front();
    PointIndexT beginOffset(0), endOffset(0),
        previousOffset(static_cast<IntT>(leaf.indices().size() + 1u));
    Coord ijk(0);

    for (ijk[0] = min[0]; ijk[0] <= max[0]; ++ijk[0]) {
        xPos = (ijk[0] & (LeafNodeType::DIM - 1u)) << (2 * LeafNodeType::LOG2DIM);

        for (ijk[1] = min[1]; ijk[1] <= max[1]; ++ijk[1]) {
            pos = xPos + ((ijk[1] & (LeafNodeType::DIM - 1u)) << LeafNodeType::LOG2DIM);
            pos += (min[2] & (LeafNodeType::DIM - 1u));

            beginOffset = (pos == 0 ? PointIndexT(0) : leaf.getValue(pos - 1));
            endOffset = leaf.getValue(pos+zStride);

            if (endOffset > beginOffset) {

                if (beginOffset == previousOffset) {
                    rangeList.back().second = dataPtr + endOffset;
                } else {
                    rangeList.push_back(Range(dataPtr + beginOffset, dataPtr + endOffset));
                }

                previousOffset = endOffset;
            }
        }
    }
}


template<typename RangeDeque, typename ConstAccessor>
inline void
pointIndexSearch(RangeDeque& rangeList, ConstAccessor& acc, const CoordBBox& bbox)
{
    using LeafNodeType = typename ConstAccessor::TreeType::LeafNodeType;
    using PointIndexT = typename LeafNodeType::ValueType;
    using Range = typename RangeDeque::value_type;

    Coord ijk(0), ijkMax(0), ijkA(0), ijkB(0);
    const Coord leafMin = bbox.min() & ~(LeafNodeType::DIM - 1);
    const Coord leafMax = bbox.max() & ~(LeafNodeType::DIM - 1);

    for (ijk[0] = leafMin[0]; ijk[0] <= leafMax[0]; ijk[0] += LeafNodeType::DIM) {
        for (ijk[1] = leafMin[1]; ijk[1] <= leafMax[1]; ijk[1] += LeafNodeType::DIM) {
            for (ijk[2] = leafMin[2]; ijk[2] <= leafMax[2]; ijk[2] += LeafNodeType::DIM) {

                if (const LeafNodeType* leaf = acc.probeConstLeaf(ijk)) {
                    ijkMax = ijk;
                    ijkMax.offset(LeafNodeType::DIM - 1);

                    // intersect leaf bbox with search region.
                    ijkA = Coord::maxComponent(bbox.min(), ijk);
                    ijkB = Coord::minComponent(bbox.max(), ijkMax);

                    if (ijkA != ijk || ijkB != ijkMax) {
                        pointIndexSearchVoxels(rangeList, *leaf, ijkA, ijkB);
                    } else {
                        // leaf bbox is inside the search region, add all indices.
                        const PointIndexT* begin = &leaf->indices().front();
                        rangeList.push_back(Range(begin, (begin + leaf->indices().size())));
                    }
                }
            }
        }
    }
}


} // namespace point_index_grid_internal


// PointIndexIterator implementation

template<typename TreeType>
inline
PointIndexIterator<TreeType>::PointIndexIterator()
    : mRange(static_cast<ValueType*>(nullptr), static_cast<ValueType*>(nullptr))
    , mRangeList()
    , mIter(mRangeList.begin())
    , mIndexArray()
    , mIndexArraySize(0)
{
}


template<typename TreeType>
inline
PointIndexIterator<TreeType>::PointIndexIterator(const PointIndexIterator& rhs)
    : mRange(rhs.mRange)
    , mRangeList(rhs.mRangeList)
    , mIter(mRangeList.begin())
    , mIndexArray()
    , mIndexArraySize(rhs.mIndexArraySize)
{
    if (rhs.mIndexArray) {
        mIndexArray.reset(new ValueType[mIndexArraySize]);
        memcpy(mIndexArray.get(), rhs.mIndexArray.get(), mIndexArraySize * sizeof(ValueType));
    }
}


template<typename TreeType>
inline PointIndexIterator<TreeType>&
PointIndexIterator<TreeType>::operator=(const PointIndexIterator& rhs)
{
    if (&rhs != this) {
        mRange = rhs.mRange;
        mRangeList = rhs.mRangeList;
        mIter = mRangeList.begin();
        mIndexArray.reset();
        mIndexArraySize = rhs.mIndexArraySize;

        if (rhs.mIndexArray) {
            mIndexArray.reset(new ValueType[mIndexArraySize]);
            memcpy(mIndexArray.get(), rhs.mIndexArray.get(), mIndexArraySize * sizeof(ValueType));
        }
    }
    return *this;
}


template<typename TreeType>
inline
PointIndexIterator<TreeType>::PointIndexIterator(const Coord& ijk, ConstAccessor& acc)
    : mRange(static_cast<ValueType*>(nullptr), static_cast<ValueType*>(nullptr))
    , mRangeList()
    , mIter(mRangeList.begin())
    , mIndexArray()
    , mIndexArraySize(0)
{
    const LeafNodeType* leaf = acc.probeConstLeaf(ijk);
    if (leaf && leaf->getIndices(ijk, mRange.first, mRange.second)) {
        mRangeList.push_back(mRange);
        mIter = mRangeList.begin();
    }
}


template<typename TreeType>
inline
PointIndexIterator<TreeType>::PointIndexIterator(const CoordBBox& bbox, ConstAccessor& acc)
    : mRange(static_cast<ValueType*>(nullptr), static_cast<ValueType*>(nullptr))
    , mRangeList()
    , mIter(mRangeList.begin())
    , mIndexArray()
    , mIndexArraySize(0)
{
    point_index_grid_internal::pointIndexSearch(mRangeList, acc, bbox);

    if (!mRangeList.empty()) {
        mIter = mRangeList.begin();
        mRange = mRangeList.front();
    }
}


template<typename TreeType>
inline void
PointIndexIterator<TreeType>::reset()
{
    mIter = mRangeList.begin();
    if (!mRangeList.empty()) {
        mRange = mRangeList.front();
    } else if (mIndexArray) {
        mRange.first = mIndexArray.get();
        mRange.second = mRange.first + mIndexArraySize;
    } else {
        mRange.first = static_cast<ValueType*>(nullptr);
        mRange.second = static_cast<ValueType*>(nullptr);
    }
}


template<typename TreeType>
inline void
PointIndexIterator<TreeType>::increment()
{
    ++mRange.first;
    if (mRange.first >= mRange.second && mIter != mRangeList.end()) {
        ++mIter;
        if (mIter != mRangeList.end()) {
            mRange = *mIter;
        } else if (mIndexArray) {
            mRange.first = mIndexArray.get();
            mRange.second = mRange.first + mIndexArraySize;
        }
    }
}


template<typename TreeType>
inline bool
PointIndexIterator<TreeType>::next()
{
    if (!this->test()) return false;
    this->increment();
    return this->test();
}


template<typename TreeType>
inline size_t
PointIndexIterator<TreeType>::size() const
{
    size_t count = 0;
    typename RangeDeque::const_iterator it = mRangeList.begin();

    for ( ; it != mRangeList.end(); ++it) {
        count += it->second - it->first;
    }

    return count + mIndexArraySize;
}


template<typename TreeType>
inline void
PointIndexIterator<TreeType>::clear()
{
    mRange.first = static_cast<ValueType*>(nullptr);
    mRange.second = static_cast<ValueType*>(nullptr);
    mRangeList.clear();
    mIter = mRangeList.end();
    mIndexArray.reset();
    mIndexArraySize = 0;
}


template<typename TreeType>
inline void
PointIndexIterator<TreeType>::searchAndUpdate(const Coord& ijk, ConstAccessor& acc)
{
    this->clear();
    const LeafNodeType* leaf = acc.probeConstLeaf(ijk);
    if (leaf && leaf->getIndices(ijk, mRange.first, mRange.second)) {
        mRangeList.push_back(mRange);
        mIter = mRangeList.begin();
    }
}


template<typename TreeType>
inline void
PointIndexIterator<TreeType>::searchAndUpdate(const CoordBBox& bbox, ConstAccessor& acc)
{
    this->clear();
    point_index_grid_internal::pointIndexSearch(mRangeList, acc, bbox);

    if (!mRangeList.empty()) {
        mIter = mRangeList.begin();
        mRange = mRangeList.front();
    }
}


template<typename TreeType>
template<typename PointArray>
inline void
PointIndexIterator<TreeType>::searchAndUpdate(const BBoxd& bbox, ConstAccessor& acc,
    const PointArray& points, const math::Transform& xform)
{
    this->clear();

    std::vector<CoordBBox> searchRegions;
    CoordBBox region(Coord::round(bbox.min()), Coord::round(bbox.max()));

    const Coord dim = region.dim();
    const int minExtent = std::min(dim[0], std::min(dim[1], dim[2]));

    if (minExtent > 2) {
        // collect indices that don't need to be tested
        CoordBBox ibox = region;
        ibox.expand(-1);

        point_index_grid_internal::pointIndexSearch(mRangeList, acc, ibox);

        // define regions for the filtered search
        ibox.expand(1);
        point_index_grid_internal::constructExclusiveRegions(searchRegions, region, ibox);
    } else {
        searchRegions.push_back(region);
    }

    // filtered search
    std::deque<ValueType> filteredIndices;
    point_index_grid_internal::BBoxFilter<PointArray, ValueType>
        filter(mRangeList, filteredIndices, bbox, points, xform);

    for (size_t n = 0, N = searchRegions.size(); n < N; ++n) {
        point_index_grid_internal::filteredPointIndexSearch(filter, acc, searchRegions[n]);
    }

    point_index_grid_internal::dequeToArray(filteredIndices, mIndexArray, mIndexArraySize);

    this->reset();
}


template<typename TreeType>
template<typename PointArray>
inline void
PointIndexIterator<TreeType>::searchAndUpdate(const Vec3d& center, double radius,
    ConstAccessor& acc, const PointArray& points, const math::Transform& xform,
    bool subvoxelAccuracy)
{
    this->clear();
    std::vector<CoordBBox> searchRegions;

    // bounding box
    CoordBBox bbox(
        Coord::round(Vec3d(center[0] - radius, center[1] - radius, center[2] - radius)),
        Coord::round(Vec3d(center[0] + radius, center[1] + radius, center[2] + radius)));
    bbox.expand(1);

    const double iRadius = radius * double(1.0 / std::sqrt(3.0));
    if (iRadius > 2.0) {
        // inscribed box
        CoordBBox ibox(
            Coord::round(Vec3d(center[0] - iRadius, center[1] - iRadius, center[2] - iRadius)),
            Coord::round(Vec3d(center[0] + iRadius, center[1] + iRadius, center[2] + iRadius)));
        ibox.expand(-1);

        // collect indices that don't need to be tested
        point_index_grid_internal::pointIndexSearch(mRangeList, acc, ibox);

        ibox.expand(1);
        point_index_grid_internal::constructExclusiveRegions(searchRegions, bbox, ibox);
    } else {
        searchRegions.push_back(bbox);
    }

    // filtered search
    std::deque<ValueType> filteredIndices;
    const double leafNodeDim = double(TreeType::LeafNodeType::DIM);

    using FilterT = point_index_grid_internal::RadialRangeFilter<PointArray, ValueType>;

    FilterT filter(mRangeList, filteredIndices,
        center, radius, points, xform, leafNodeDim, subvoxelAccuracy);

    for (size_t n = 0, N = searchRegions.size(); n < N; ++n) {
        point_index_grid_internal::filteredPointIndexSearch(filter, acc, searchRegions[n]);
    }

    point_index_grid_internal::dequeToArray(filteredIndices, mIndexArray, mIndexArraySize);

    this->reset();
}


template<typename TreeType>
template<typename PointArray>
inline void
PointIndexIterator<TreeType>::worldSpaceSearchAndUpdate(const BBoxd& bbox, ConstAccessor& acc,
    const PointArray& points, const math::Transform& xform)
{
    this->searchAndUpdate(
        BBoxd(xform.worldToIndex(bbox.min()), xform.worldToIndex(bbox.max())), acc, points, xform);
}


template<typename TreeType>
template<typename PointArray>
inline void
PointIndexIterator<TreeType>::worldSpaceSearchAndUpdate(const Vec3d& center, double radius,
    ConstAccessor& acc, const PointArray& points, const math::Transform& xform,
    bool subvoxelAccuracy)
{
    this->searchAndUpdate(xform.worldToIndex(center),
        (radius / xform.voxelSize()[0]), acc, points, xform, subvoxelAccuracy);
}


////////////////////////////////////////

// PointIndexFilter implementation

template<typename PointArray, typename TreeType>
inline
PointIndexFilter<PointArray, TreeType>::PointIndexFilter(
    const PointArray& points, const TreeType& tree, const math::Transform& xform)
    : mPoints(&points), mAcc(tree), mXform(xform), mInvVoxelSize(1.0/xform.voxelSize()[0])
{
}


template<typename PointArray, typename TreeType>
inline
PointIndexFilter<PointArray, TreeType>::PointIndexFilter(const PointIndexFilter& rhs)
    : mPoints(rhs.mPoints)
    , mAcc(rhs.mAcc.tree())
    , mXform(rhs.mXform)
    , mInvVoxelSize(rhs.mInvVoxelSize)
{
}


template<typename PointArray, typename TreeType>
template<typename FilterType>
inline void
PointIndexFilter<PointArray, TreeType>::searchAndApply(
    const PosType& center, ScalarType radius, FilterType& op)
{
    if (radius * mInvVoxelSize < ScalarType(8.0)) {
        mIter.searchAndUpdate(openvdb::CoordBBox(
            mXform.worldToIndexCellCentered(center - radius),
            mXform.worldToIndexCellCentered(center + radius)), mAcc);
    } else {
        mIter.worldSpaceSearchAndUpdate(
            center, radius, mAcc, *mPoints, mXform, /*subvoxelAccuracy=*/false);
    }

    const ScalarType radiusSqr = radius * radius;
    ScalarType distSqr = 0.0;
    PosType pos;
    for (; mIter; ++mIter) {
        mPoints->getPos(*mIter, pos);
        pos -= center;
        distSqr = pos.lengthSqr();

        if (distSqr < radiusSqr) {
            op(distSqr, *mIter);
        }
    }
}


////////////////////////////////////////


template<typename GridT, typename PointArrayT>
inline typename GridT::Ptr
createPointIndexGrid(const PointArrayT& points, const math::Transform& xform)
{
    typename GridT::Ptr grid = GridT::create(typename GridT::ValueType(0));
    grid->setTransform(xform.copy());

    if (points.size() > 0) {
        point_index_grid_internal::constructPointTree(
            grid->tree(), grid->transform(), points);
    }

    return grid;
}


template<typename GridT, typename PointArrayT>
inline typename GridT::Ptr
createPointIndexGrid(const PointArrayT& points, double voxelSize)
{
    math::Transform::Ptr xform = math::Transform::createLinearTransform(voxelSize);
    return createPointIndexGrid<GridT>(points, *xform);
}


template<typename PointArrayT, typename GridT>
inline bool
isValidPartition(const PointArrayT& points, const GridT& grid)
{
    tree::LeafManager<const typename GridT::TreeType> leafs(grid.tree());

    size_t pointCount = 0;
    for (size_t n = 0, N = leafs.leafCount(); n < N; ++n) {
        pointCount += leafs.leaf(n).indices().size();
    }

    if (points.size() != pointCount) {
        return false;
    }

    tbb::atomic<bool> changed;
    changed = false;

    point_index_grid_internal::ValidPartitioningOp<PointArrayT>
        op(changed, points, grid.transform());

    leafs.foreach(op);

    return !bool(changed);
}


template<typename GridT, typename PointArrayT>
inline typename GridT::ConstPtr
getValidPointIndexGrid(const PointArrayT& points, const typename GridT::ConstPtr& grid)
{
    if (isValidPartition(points, *grid)) {
        return grid;
    }

    return createPointIndexGrid<GridT>(points, grid->transform());
}


template<typename GridT, typename PointArrayT>
inline typename GridT::Ptr
getValidPointIndexGrid(const PointArrayT& points, const typename GridT::Ptr& grid)
{
    if (isValidPartition(points, *grid)) {
        return grid;
    }

    return createPointIndexGrid<GridT>(points, grid->transform());
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
struct PointIndexLeafNode : public tree::LeafNode<T, Log2Dim>
{
    using LeafNodeType = PointIndexLeafNode<T, Log2Dim>;
    using Ptr = SharedPtr<PointIndexLeafNode>;

    using ValueType = T;
    using IndexArray = std::vector<ValueType>;


    IndexArray& indices() { return mIndices; }
    const IndexArray& indices() const { return mIndices; }

    bool getIndices(const Coord& ijk, const ValueType*& begin, const ValueType*& end) const;
    bool getIndices(Index offset, const ValueType*& begin, const ValueType*& end) const;

    void setOffsetOn(Index offset, const ValueType& val);
    void setOffsetOnly(Index offset, const ValueType& val);

    bool isEmpty(const CoordBBox& bbox) const;

private:
    IndexArray mIndices;

    ////////////////////////////////////////

    // The following methods had to be copied from the LeafNode class
    // to make the derived PointIndexLeafNode class compatible with the tree structure.

public:
    using BaseLeaf = tree::LeafNode<T, Log2Dim>;
    using NodeMaskType = util::NodeMask<Log2Dim>;

    using BaseLeaf::LOG2DIM;
    using BaseLeaf::TOTAL;
    using BaseLeaf::DIM;
    using BaseLeaf::NUM_VALUES;
    using BaseLeaf::NUM_VOXELS;
    using BaseLeaf::SIZE;
    using BaseLeaf::LEVEL;

    /// Default constructor
    PointIndexLeafNode() : BaseLeaf(), mIndices() {}

    explicit
    PointIndexLeafNode(const Coord& coords, const T& value = zeroVal<T>(), bool active = false)
        : BaseLeaf(coords, value, active)
        , mIndices()
    {
    }

#if OPENVDB_ABI_VERSION_NUMBER >= 3
    PointIndexLeafNode(PartialCreate, const Coord& coords,
        const T& value = zeroVal<T>(), bool active = false)
        : BaseLeaf(PartialCreate(), coords, value, active)
        , mIndices()
    {
    }
#endif

    /// Deep copy constructor
    PointIndexLeafNode(const PointIndexLeafNode& rhs) : BaseLeaf(rhs), mIndices(rhs.mIndices) {}

    /// @brief Return @c true if the given node (which may have a different @c ValueType
    /// than this node) has the same active value topology as this node.
    template<typename OtherType, Index OtherLog2Dim>
    bool hasSameTopology(const PointIndexLeafNode<OtherType, OtherLog2Dim>* other) const {
        return BaseLeaf::hasSameTopology(other);
    }

    /// Check for buffer, state and origin equivalence.
    bool operator==(const PointIndexLeafNode& other) const { return BaseLeaf::operator==(other); }

    bool operator!=(const PointIndexLeafNode& other) const { return !(other == *this); }

    template<MergePolicy Policy> void merge(const PointIndexLeafNode& rhs) {
        BaseLeaf::merge<Policy>(rhs);
    }
    template<MergePolicy Policy> void merge(const ValueType& tileValue, bool tileActive) {
         BaseLeaf::template merge<Policy>(tileValue, tileActive);
    }

    template<MergePolicy Policy>
    void merge(const PointIndexLeafNode& other,
        const ValueType& /*bg*/, const ValueType& /*otherBG*/)
    {
         BaseLeaf::template merge<Policy>(other);
    }

    void addLeaf(PointIndexLeafNode*) {}
    template<typename AccessorT>
    void addLeafAndCache(PointIndexLeafNode*, AccessorT&) {}

    //@{
    /// @brief Return a pointer to this node.
    PointIndexLeafNode* touchLeaf(const Coord&) { return this; }
    template<typename AccessorT>
    PointIndexLeafNode* touchLeafAndCache(const Coord&, AccessorT&) { return this; }

    template<typename NodeT, typename AccessorT>
    NodeT* probeNodeAndCache(const Coord&, AccessorT&)
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (!(std::is_same<NodeT, PointIndexLeafNode>::value)) return nullptr;
        return reinterpret_cast<NodeT*>(this);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    PointIndexLeafNode* probeLeaf(const Coord&) { return this; }
    template<typename AccessorT>
    PointIndexLeafNode* probeLeafAndCache(const Coord&, AccessorT&) { return this; }
    //@}

    //@{
    /// @brief Return a @const pointer to this node.
    const PointIndexLeafNode* probeConstLeaf(const Coord&) const { return this; }
    template<typename AccessorT>
    const PointIndexLeafNode* probeConstLeafAndCache(const Coord&, AccessorT&) const {return this;}
    template<typename AccessorT>
    const PointIndexLeafNode* probeLeafAndCache(const Coord&, AccessorT&) const { return this; }
    const PointIndexLeafNode* probeLeaf(const Coord&) const { return this; }
    template<typename NodeT, typename AccessorT>
    const NodeT* probeConstNodeAndCache(const Coord&, AccessorT&) const
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (!(std::is_same<NodeT, PointIndexLeafNode>::value)) return nullptr;
        return reinterpret_cast<const NodeT*>(this);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    //@}


    // I/O methods

    void readBuffers(std::istream& is, bool fromHalf = false);
    void readBuffers(std::istream& is, const CoordBBox&, bool fromHalf = false);
    void writeBuffers(std::ostream& os, bool toHalf = false) const;


    Index64 memUsage() const;


    ////////////////////////////////////////

    // Disable all write methods to avoid unintentional changes
    // to the point-array offsets.

    void assertNonmodifiable() {
        assert(false && "Cannot modify voxel values in a PointIndexTree.");
    }

    void setActiveState(const Coord&, bool) { assertNonmodifiable(); }
    void setActiveState(Index, bool) { assertNonmodifiable(); }

    void setValueOnly(const Coord&, const ValueType&) { assertNonmodifiable(); }
    void setValueOnly(Index, const ValueType&) { assertNonmodifiable(); }

    void setValueOff(const Coord&) { assertNonmodifiable(); }
    void setValueOff(Index) { assertNonmodifiable(); }

    void setValueOff(const Coord&, const ValueType&) { assertNonmodifiable(); }
    void setValueOff(Index, const ValueType&) { assertNonmodifiable(); }

    void setValueOn(const Coord&) { assertNonmodifiable(); }
    void setValueOn(Index) { assertNonmodifiable(); }

    void setValueOn(const Coord&, const ValueType&) { assertNonmodifiable(); }
    void setValueOn(Index, const ValueType&) { assertNonmodifiable(); }

    void setValue(const Coord&, const ValueType&) { assertNonmodifiable(); }

    void setValuesOn() { assertNonmodifiable(); }
    void setValuesOff() { assertNonmodifiable(); }

    template<typename ModifyOp>
    void modifyValue(Index, const ModifyOp&) { assertNonmodifiable(); }

    template<typename ModifyOp>
    void modifyValue(const Coord&, const ModifyOp&) { assertNonmodifiable(); }

    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord&, const ModifyOp&) { assertNonmodifiable(); }

    void clip(const CoordBBox&, const ValueType&) { assertNonmodifiable(); }

    void fill(const CoordBBox&, const ValueType&, bool) { assertNonmodifiable(); }
    void fill(const ValueType&) {}
    void fill(const ValueType&, bool) { assertNonmodifiable(); }

    template<typename AccessorT>
    void setValueOnlyAndCache(const Coord&, const ValueType&, AccessorT&) {assertNonmodifiable();}

    template<typename ModifyOp, typename AccessorT>
    void modifyValueAndActiveStateAndCache(const Coord&, const ModifyOp&, AccessorT&) {
        assertNonmodifiable();
    }

    template<typename AccessorT>
    void setValueOffAndCache(const Coord&, const ValueType&, AccessorT&) { assertNonmodifiable(); }

    template<typename AccessorT>
    void setActiveStateAndCache(const Coord&, bool, AccessorT&) { assertNonmodifiable(); }

    void resetBackground(const ValueType&, const ValueType&) { assertNonmodifiable(); }

    void signedFloodFill(const ValueType&) { assertNonmodifiable(); }
    void signedFloodFill(const ValueType&, const ValueType&) { assertNonmodifiable(); }

    void negate() { assertNonmodifiable(); }

protected:
    using ValueOn = typename BaseLeaf::ValueOn;
    using ValueOff = typename BaseLeaf::ValueOff;
    using ValueAll = typename BaseLeaf::ValueAll;
    using ChildOn = typename BaseLeaf::ChildOn;
    using ChildOff = typename BaseLeaf::ChildOff;
    using ChildAll = typename BaseLeaf::ChildAll;

    using MaskOnIterator = typename NodeMaskType::OnIterator;
    using MaskOffIterator = typename NodeMaskType::OffIterator;
    using MaskDenseIterator = typename NodeMaskType::DenseIterator;

    // During topology-only construction, access is needed
    // to protected/private members of other template instances.
    template<typename, Index> friend struct PointIndexLeafNode;

    friend class tree::IteratorBase<MaskOnIterator, PointIndexLeafNode>;
    friend class tree::IteratorBase<MaskOffIterator, PointIndexLeafNode>;
    friend class tree::IteratorBase<MaskDenseIterator, PointIndexLeafNode>;

public:
    using ValueOnIter = typename BaseLeaf::template ValueIter<
        MaskOnIterator, PointIndexLeafNode, const ValueType, ValueOn>;
    using ValueOnCIter = typename BaseLeaf::template ValueIter<
        MaskOnIterator, const PointIndexLeafNode, const ValueType, ValueOn>;
    using ValueOffIter = typename BaseLeaf::template ValueIter<
        MaskOffIterator, PointIndexLeafNode, const ValueType, ValueOff>;
    using ValueOffCIter = typename BaseLeaf::template ValueIter<
        MaskOffIterator,const PointIndexLeafNode,const ValueType, ValueOff>;
    using ValueAllIter = typename BaseLeaf::template ValueIter<
        MaskDenseIterator, PointIndexLeafNode, const ValueType, ValueAll>;
    using ValueAllCIter = typename BaseLeaf::template ValueIter<
        MaskDenseIterator,const PointIndexLeafNode,const ValueType, ValueAll>;
    using ChildOnIter = typename BaseLeaf::template ChildIter<
        MaskOnIterator, PointIndexLeafNode, ChildOn>;
    using ChildOnCIter = typename BaseLeaf::template ChildIter<
        MaskOnIterator, const PointIndexLeafNode, ChildOn>;
    using ChildOffIter = typename BaseLeaf::template ChildIter<
        MaskOffIterator, PointIndexLeafNode, ChildOff>;
    using ChildOffCIter = typename BaseLeaf::template ChildIter<
        MaskOffIterator, const PointIndexLeafNode, ChildOff>;
    using ChildAllIter = typename BaseLeaf::template DenseIter<
        PointIndexLeafNode, ValueType, ChildAll>;
    using ChildAllCIter = typename BaseLeaf::template DenseIter<
        const PointIndexLeafNode, const ValueType, ChildAll>;

#define VMASK_ this->getValueMask()
    ValueOnCIter  cbeginValueOn() const  { return ValueOnCIter(VMASK_.beginOn(), this); }
    ValueOnCIter   beginValueOn() const  { return ValueOnCIter(VMASK_.beginOn(), this); }
    ValueOnIter    beginValueOn()        { return ValueOnIter(VMASK_.beginOn(), this); }
    ValueOffCIter cbeginValueOff() const { return ValueOffCIter(VMASK_.beginOff(), this); }
    ValueOffCIter  beginValueOff() const { return ValueOffCIter(VMASK_.beginOff(), this); }
    ValueOffIter   beginValueOff()       { return ValueOffIter(VMASK_.beginOff(), this); }
    ValueAllCIter cbeginValueAll() const { return ValueAllCIter(VMASK_.beginDense(), this); }
    ValueAllCIter  beginValueAll() const { return ValueAllCIter(VMASK_.beginDense(), this); }
    ValueAllIter   beginValueAll()       { return ValueAllIter(VMASK_.beginDense(), this); }

    ValueOnCIter  cendValueOn() const    { return ValueOnCIter(VMASK_.endOn(), this); }
    ValueOnCIter   endValueOn() const    { return ValueOnCIter(VMASK_.endOn(), this); }
    ValueOnIter    endValueOn()          { return ValueOnIter(VMASK_.endOn(), this); }
    ValueOffCIter cendValueOff() const   { return ValueOffCIter(VMASK_.endOff(), this); }
    ValueOffCIter  endValueOff() const   { return ValueOffCIter(VMASK_.endOff(), this); }
    ValueOffIter   endValueOff()         { return ValueOffIter(VMASK_.endOff(), this); }
    ValueAllCIter cendValueAll() const   { return ValueAllCIter(VMASK_.endDense(), this); }
    ValueAllCIter  endValueAll() const   { return ValueAllCIter(VMASK_.endDense(), this); }
    ValueAllIter   endValueAll()         { return ValueAllIter(VMASK_.endDense(), this); }

    ChildOnCIter  cbeginChildOn() const  { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnCIter   beginChildOn() const  { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnIter    beginChildOn()        { return ChildOnIter(VMASK_.endOn(), this); }
    ChildOffCIter cbeginChildOff() const { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffCIter  beginChildOff() const { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffIter   beginChildOff()       { return ChildOffIter(VMASK_.endOff(), this); }
    ChildAllCIter cbeginChildAll() const { return ChildAllCIter(VMASK_.beginDense(), this); }
    ChildAllCIter  beginChildAll() const { return ChildAllCIter(VMASK_.beginDense(), this); }
    ChildAllIter   beginChildAll()       { return ChildAllIter(VMASK_.beginDense(), this); }

    ChildOnCIter  cendChildOn() const    { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnCIter   endChildOn() const    { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnIter    endChildOn()          { return ChildOnIter(VMASK_.endOn(), this); }
    ChildOffCIter cendChildOff() const   { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffCIter  endChildOff() const   { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffIter   endChildOff()         { return ChildOffIter(VMASK_.endOff(), this); }
    ChildAllCIter cendChildAll() const   { return ChildAllCIter(VMASK_.endDense(), this); }
    ChildAllCIter  endChildAll() const   { return ChildAllCIter(VMASK_.endDense(), this); }
    ChildAllIter   endChildAll()         { return ChildAllIter(VMASK_.endDense(), this); }
#undef VMASK_
}; // struct PointIndexLeafNode


template<typename T, Index Log2Dim>
inline bool
PointIndexLeafNode<T, Log2Dim>::getIndices(const Coord& ijk,
    const ValueType*& begin, const ValueType*& end) const
{
    return getIndices(LeafNodeType::coordToOffset(ijk), begin, end);
}


template<typename T, Index Log2Dim>
inline bool
PointIndexLeafNode<T, Log2Dim>::getIndices(Index offset,
    const ValueType*& begin, const ValueType*& end) const
{
    if (this->isValueMaskOn(offset)) {
        const ValueType* dataPtr = &mIndices.front();
        begin = dataPtr + (offset == 0 ? ValueType(0) : this->buffer()[offset - 1]);
        end = dataPtr + this->buffer()[offset];
        return true;
    }
    return false;
}


template<typename T, Index Log2Dim>
inline void
PointIndexLeafNode<T, Log2Dim>::setOffsetOn(Index offset, const ValueType& val)
{
    this->buffer().setValue(offset, val);
    this->setValueMaskOn(offset);
}


template<typename T, Index Log2Dim>
inline void
PointIndexLeafNode<T, Log2Dim>::setOffsetOnly(Index offset, const ValueType& val)
{
    this->buffer().setValue(offset, val);
}


template<typename T, Index Log2Dim>
inline bool
PointIndexLeafNode<T, Log2Dim>::isEmpty(const CoordBBox& bbox) const
{
    Index xPos, pos, zStride = Index(bbox.max()[2] - bbox.min()[2]);
    Coord ijk;

    for (ijk[0] = bbox.min()[0]; ijk[0] <= bbox.max()[0]; ++ijk[0]) {
        xPos = (ijk[0] & (DIM - 1u)) << (2 * LOG2DIM);

        for (ijk[1] = bbox.min()[1]; ijk[1] <= bbox.max()[1]; ++ijk[1]) {
            pos = xPos + ((ijk[1] & (DIM - 1u)) << LOG2DIM);
            pos += (bbox.min()[2] & (DIM - 1u));

            if (this->buffer()[pos+zStride] > (pos == 0 ? T(0) : this->buffer()[pos - 1])) {
                return false;
            }
        }
    }

    return true;
}


template<typename T, Index Log2Dim>
inline void
PointIndexLeafNode<T, Log2Dim>::readBuffers(std::istream& is, bool fromHalf)
{
    BaseLeaf::readBuffers(is, fromHalf);

    Index64 numIndices = Index64(0);
    is.read(reinterpret_cast<char*>(&numIndices), sizeof(Index64));

    mIndices.resize(size_t(numIndices));
    is.read(reinterpret_cast<char*>(mIndices.data()), numIndices * sizeof(T));
}


template<typename T, Index Log2Dim>
inline void
PointIndexLeafNode<T, Log2Dim>::readBuffers(std::istream& is, const CoordBBox& bbox, bool fromHalf)
{
    // Read and clip voxel values.
    BaseLeaf::readBuffers(is, bbox, fromHalf);

    Index64 numIndices = Index64(0);
    is.read(reinterpret_cast<char*>(&numIndices), sizeof(Index64));

    const Index64 numBytes = numIndices * sizeof(T);

    if (bbox.hasOverlap(this->getNodeBoundingBox())) {
        mIndices.resize(size_t(numIndices));
        is.read(reinterpret_cast<char*>(mIndices.data()), numBytes);

        /// @todo If any voxels were deactivated as a result of clipping in the call to
        /// BaseLeaf::readBuffers(), the point index list will need to be regenerated.
    } else {
        // Read and discard voxel values.
        std::unique_ptr<char[]> buf{new char[numBytes]};
        is.read(buf.get(), numBytes);
    }

    // Reserved for future use
    Index64 auxDataBytes = Index64(0);
    is.read(reinterpret_cast<char*>(&auxDataBytes), sizeof(Index64));
    if (auxDataBytes > 0) {
        // For now, read and discard any auxiliary data.
        std::unique_ptr<char[]> auxData{new char[auxDataBytes]};
        is.read(auxData.get(), auxDataBytes);
    }
}


template<typename T, Index Log2Dim>
inline void
PointIndexLeafNode<T, Log2Dim>::writeBuffers(std::ostream& os, bool toHalf) const
{
    BaseLeaf::writeBuffers(os, toHalf);

    Index64 numIndices = Index64(mIndices.size());
    os.write(reinterpret_cast<const char*>(&numIndices), sizeof(Index64));
    os.write(reinterpret_cast<const char*>(mIndices.data()), numIndices * sizeof(T));

    // Reserved for future use
    const Index64 auxDataBytes = Index64(0);
    os.write(reinterpret_cast<const char*>(&auxDataBytes), sizeof(Index64));
}


template<typename T, Index Log2Dim>
inline Index64
PointIndexLeafNode<T, Log2Dim>::memUsage() const
{
    return BaseLeaf::memUsage() + Index64((sizeof(T)*mIndices.capacity()) + sizeof(mIndices));
}

} // namespace tools


////////////////////////////////////////


namespace tree {

/// Helper metafunction used to implement LeafNode::SameConfiguration
/// (which, as an inner class, can't be independently specialized)
template<Index Dim1, typename T2>
struct SameLeafConfig<Dim1, openvdb::tools::PointIndexLeafNode<T2, Dim1> >
{
    static const bool value = true;
};

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_POINT_INDEX_GRID_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
