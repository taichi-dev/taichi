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

/// @author Dan Bailey
///
/// @file PointMove.h
///
/// @brief Ability to move VDB Points using a custom deformer.
///
/// Deformers used when moving points are in world space by default and must adhere
/// to the interface described in the example below:
/// @code
/// struct MyDeformer
/// {
///     // A reset is performed on each leaf in turn before the points in that leaf are
///     // deformed. A leaf and leaf index (standard leaf traversal order) are supplied as
///     // the arguments, which matches the functor interface for LeafManager::foreach().
///     template <typename LeafNoteType>
///     void reset(LeafNoteType& leaf, size_t idx);
///
///     // Evaluate the deformer and modify the given position to generate the deformed
///     // position. An index iterator is supplied as the argument to allow querying the
///     // point offset or containing voxel coordinate.
///     template <typename IndexIterT>
///     void apply(Vec3d& position, const IndexIterT& iter) const;
/// };
/// @endcode
///
/// @note The DeformerTraits struct (defined in PointMask.h) can be used to configure
/// a deformer to evaluate in index space.

#ifndef OPENVDB_POINTS_POINT_MOVE_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_MOVE_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointMask.h>

#include <tbb/concurrent_vector.h>

#include <algorithm>
#include <iterator> // for std::begin(), std::end()
#include <map>
#include <numeric> // for std::iota()
#include <tuple>
#include <unordered_map>
#include <vector>

class TestPointMove;


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

// dummy object for future use
namespace future { struct Advect { }; }


/// @brief Move points in a PointDataGrid using a custom deformer
/// @param points           the PointDataGrid containing the points to be moved.
/// @param deformer         a custom deformer that defines how to move the points.
/// @param filter           an optional index filter
/// @param objectNotInUse   for future use, this object is currently ignored
/// @param threaded         enable or disable threading  (threading is enabled by default)
template <typename PointDataGridT, typename DeformerT, typename FilterT = NullFilter>
inline void movePoints(PointDataGridT& points,
                       DeformerT& deformer,
                       const FilterT& filter = NullFilter(),
                       future::Advect* objectNotInUse = nullptr,
                       bool threaded = true);


/// @brief Move points in a PointDataGrid using a custom deformer and a new transform
/// @param points           the PointDataGrid containing the points to be moved.
/// @param transform        target transform to use for the resulting points.
/// @param deformer         a custom deformer that defines how to move the points.
/// @param filter           an optional index filter
/// @param objectNotInUse   for future use, this object is currently ignored
/// @param threaded         enable or disable threading  (threading is enabled by default)
template <typename PointDataGridT, typename DeformerT, typename FilterT = NullFilter>
inline void movePoints(PointDataGridT& points,
                       const math::Transform& transform,
                       DeformerT& deformer,
                       const FilterT& filter = NullFilter(),
                       future::Advect* objectNotInUse = nullptr,
                       bool threaded = true);


// define leaf index in use as 32-bit
namespace point_move_internal { using LeafIndex = Index32; }


/// @brief A Deformer that caches the resulting positions from evaluating another Deformer
template <typename T>
class CachedDeformer
{
public:
    using LeafIndex = point_move_internal::LeafIndex;
    using Vec3T = typename math::Vec3<T>;
    using LeafVecT = std::vector<Vec3T>;
    using LeafMapT = std::unordered_map<LeafIndex, Vec3T>;

    // Internal data cache to allow the deformer to offer light-weight copying
    struct Cache
    {
        struct Leaf
        {
            /// @brief clear data buffers and reset counter
            void clear() {
                vecData.clear();
                mapData.clear();
                totalSize = 0;
            }

            LeafVecT vecData;
            LeafMapT mapData;
            Index totalSize = 0;
        }; // struct Leaf

        std::vector<Leaf> leafs;
    }; // struct Cache

    /// Cache is expected to be persistent for the lifetime of the CachedDeformer
    explicit CachedDeformer(Cache& cache);

    /// Caches the result of evaluating the supplied point grid using the deformer and filter
    /// @param grid         the points to be moved
    /// @param deformer     the deformer to apply to the points
    /// @param filter       the point filter to use when evaluating the points
    /// @param threaded     enable or disable threading  (threading is enabled by default)
    template <typename PointDataGridT, typename DeformerT, typename FilterT>
    void evaluate(PointDataGridT& grid, DeformerT& deformer, const FilterT& filter,
        bool threaded = true);

    /// Stores pointers to the vector or map and optionally expands the map into a vector
    /// @throw IndexError if idx is out-of-range of the leafs in the cache
    template <typename LeafT>
    void reset(const LeafT& leaf, size_t idx);

    /// Retrieve the new position from the cache
    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT& iter) const;

private:
    friend class ::TestPointMove;

    Cache& mCache;
    LeafVecT mLocalLeafVec;
    const LeafVecT* mLeafVec = nullptr;
    const LeafMapT* mLeafMap = nullptr;
}; // class CachedDeformer


////////////////////////////////////////


namespace point_move_internal {


using IndexTriple = std::tuple<LeafIndex, Index, Index>;
using IndexTripleArray = tbb::concurrent_vector<IndexTriple>;
using GlobalPointIndexMap = std::vector<IndexTripleArray>;

using IndexPair = std::pair<Index, Index>;
using IndexPairArray = std::vector<IndexPair>;
using LocalPointIndexMap = std::vector<IndexPairArray>;

using IndexArray = std::vector<Index>;

using LeafIndexArray = std::vector<LeafIndex>;
using LeafOffsetArray = std::vector<LeafIndexArray>;
using LeafMap = std::map<Coord, LeafIndex>;

// TODO: The following infrastructure - ArrayProcessor, PerformTypedMoveOp, processTypedArray()
// is required to improve AttributeArray copying performance beyond using the virtual function
// AttributeArray::set(Index, AttributeArray&, Index), however an ABI=6+ addition to AttributeArray
// could eliminate this cost for the typical case where ValueType and CodecType are the same in
// source and target arrays, as they are here.


/// Helper class used internally by processTypedArray()
template<typename ValueType, typename OpType>
struct ArrayProcessor {
    static inline void call(OpType& op, const AttributeArray& array) {
#ifdef _MSC_VER
        op.operator()<ValueType>(array);
#else
        op.template operator()<ValueType>(array);
#endif
    }
};

/// @brief Utility function that, given a generic attribute array,
/// calls a functor with the fully-resolved value type of the array
template<typename ArrayType, typename OpType>
bool
processTypedArray(const ArrayType& array, OpType& op)
{
    using namespace openvdb;
    using namespace openvdb::math;
    if (array.template hasValueType<bool>())                    ArrayProcessor<bool, OpType>::call(op, array);
    else if (array.template hasValueType<int16_t>())            ArrayProcessor<int16_t, OpType>::call(op, array);
    else if (array.template hasValueType<int32_t>())            ArrayProcessor<int32_t, OpType>::call(op, array);
    else if (array.template hasValueType<int64_t>())            ArrayProcessor<int64_t, OpType>::call(op, array);
    else if (array.template hasValueType<float>())              ArrayProcessor<float, OpType>::call(op, array);
    else if (array.template hasValueType<double>())             ArrayProcessor<double, OpType>::call(op, array);
    else if (array.template hasValueType<Vec3<int32_t>>())      ArrayProcessor<Vec3<int32_t>, OpType>::call(op, array);
    else if (array.template hasValueType<Vec3<float>>())        ArrayProcessor<Vec3<float>, OpType>::call(op, array);
    else if (array.template hasValueType<Vec3<double>>())       ArrayProcessor<Vec3<double>, OpType>::call(op, array);
    else if (array.template hasValueType<GroupType>())          ArrayProcessor<GroupType, OpType>::call(op, array);
    else if (array.template hasValueType<StringIndexType>())    ArrayProcessor<StringIndexType, OpType>::call(op, array);
    else if (array.template hasValueType<Mat3<float>>())        ArrayProcessor<Mat3<float>, OpType>::call(op, array);
    else if (array.template hasValueType<Mat3<double>>())       ArrayProcessor<Mat3<double>, OpType>::call(op, array);
    else if (array.template hasValueType<Mat4<float>>())        ArrayProcessor<Mat4<float>, OpType>::call(op, array);
    else if (array.template hasValueType<Mat4<double>>())       ArrayProcessor<Mat4<double>, OpType>::call(op, array);
    else if (array.template hasValueType<Quat<float>>())        ArrayProcessor<Quat<float>, OpType>::call(op, array);
    else if (array.template hasValueType<Quat<double>>())       ArrayProcessor<Quat<double>, OpType>::call(op, array);
    else                                                        return false;
    return true;
}


/// Cache read and write attribute handles to amortize construction cost
struct AttributeHandles
{
    using HandleArray = std::vector<AttributeHandle<int>::Ptr>;

    AttributeHandles(const size_t size)
        : mHandles() { mHandles.reserve(size); }

    AttributeArray& getArray(const Index leafOffset)
    {
        auto* handle = reinterpret_cast<AttributeWriteHandle<int>*>(mHandles[leafOffset].get());
        assert(handle);
        return handle->array();
    }

    const AttributeArray& getConstArray(const Index leafOffset) const
    {
        const auto* handle = mHandles[leafOffset].get();
        assert(handle);
        return handle->array();
    }

    template <typename ValueT>
    AttributeHandle<ValueT>& getHandle(const Index leafOffset)
    {
        auto* handle = reinterpret_cast<AttributeHandle<ValueT>*>(mHandles[leafOffset].get());
        assert(handle);
        return *handle;
    }

    template <typename ValueT>
    AttributeWriteHandle<ValueT>& getWriteHandle(const Index leafOffset)
    {
        auto* handle = reinterpret_cast<AttributeWriteHandle<ValueT>*>(mHandles[leafOffset].get());
        assert(handle);
        return *handle;
    }

    /// Create a handle and reinterpret cast as an int handle to store
    struct CacheHandleOp
    {
        CacheHandleOp(HandleArray& handles)
            : mHandles(handles) { }

        template<typename ValueT>
        void operator()(const AttributeArray& array) const
        {
            auto* handleAsInt = reinterpret_cast<AttributeHandle<int>*>(
                new AttributeHandle<ValueT>(array));
            mHandles.emplace_back(handleAsInt);
        }

    private:
        HandleArray& mHandles;
    }; // struct CacheHandleOp

    template <typename LeafRangeT>
    void cache(const LeafRangeT& range, const Index attributeIndex)
    {
        using namespace openvdb::math;

        mHandles.clear();
        CacheHandleOp op(mHandles);

        for (auto leaf = range.begin(); leaf; ++leaf) {
            auto& array = leaf->attributeArray(attributeIndex);
            processTypedArray(array, op);
        }
    }

private:
    HandleArray mHandles;
}; // struct AttributeHandles


template <typename DeformerT, typename TreeT, typename FilterT>
struct BuildMoveMapsOp
{
    using LeafT = typename TreeT::LeafNodeType;
    using LeafArrayT = std::vector<LeafT*>;
    using LeafManagerT = typename tree::LeafManager<TreeT>;

    BuildMoveMapsOp(const DeformerT& deformer,
                    GlobalPointIndexMap& globalMoveLeafMap,
                    LocalPointIndexMap& localMoveLeafMap,
                    const LeafMap& targetLeafMap,
                    const math::Transform& targetTransform,
                    const math::Transform& sourceTransform,
                    const FilterT& filter)
        : mDeformer(deformer)
        , mGlobalMoveLeafMap(globalMoveLeafMap)
        , mLocalMoveLeafMap(localMoveLeafMap)
        , mTargetLeafMap(targetLeafMap)
        , mTargetTransform(targetTransform)
        , mSourceTransform(sourceTransform)
        , mFilter(filter) { }

    void operator()(LeafT& leaf, size_t idx) const
    {
        DeformerT deformer(mDeformer);
        deformer.reset(leaf, idx);

        // determine source leaf node origin and offset in the source leaf vector

        Coord sourceLeafOrigin = leaf.origin();

        auto sourceHandle = AttributeWriteHandle<Vec3f>::create(leaf.attributeArray("P"));

        for (auto iter = leaf.beginIndexOn(mFilter); iter; iter++) {

            const bool useIndexSpace = DeformerTraits<DeformerT>::IndexSpace;

            // extract index-space position and apply index-space deformation (if applicable)

            Vec3d positionIS = sourceHandle->get(*iter) + iter.getCoord().asVec3d();
            if (useIndexSpace) {
               deformer.apply(positionIS, iter);
            }

            // transform to world-space position and apply world-space deformation (if applicable)

            Vec3d positionWS = mSourceTransform.indexToWorld(positionIS);
            if (!useIndexSpace) {
                deformer.apply(positionWS, iter);
            }

            // transform to index-space position of target grid

            positionIS = mTargetTransform.worldToIndex(positionWS);

            // determine target voxel and offset

            Coord targetVoxel = Coord::round(positionIS);
            Index targetOffset = LeafT::coordToOffset(targetVoxel);

            // set new local position in source transform space (if point has been deformed)

            Vec3d voxelPosition(positionIS - targetVoxel.asVec3d());
            sourceHandle->set(*iter, voxelPosition);

            // determine target leaf node origin and offset in the target leaf vector

            Coord targetLeafOrigin = targetVoxel & ~(LeafT::DIM - 1);
            assert(mTargetLeafMap.find(targetLeafOrigin) != mTargetLeafMap.end());
            const LeafIndex targetLeafOffset(mTargetLeafMap.at(targetLeafOrigin));

            // insert into move map based on whether point ends up in a new leaf node or not

            if (targetLeafOrigin == sourceLeafOrigin) {
                mLocalMoveLeafMap[targetLeafOffset].emplace_back(targetOffset, *iter);
            }
            else {
                mGlobalMoveLeafMap[targetLeafOffset].push_back(IndexTriple(
                    LeafIndex(static_cast<LeafIndex>(idx)), targetOffset, *iter));
            }
        }
    }

private:
    const DeformerT& mDeformer;
    GlobalPointIndexMap& mGlobalMoveLeafMap;
    LocalPointIndexMap& mLocalMoveLeafMap;
    const LeafMap& mTargetLeafMap;
    const math::Transform& mTargetTransform;
    const math::Transform& mSourceTransform;
    const FilterT& mFilter;
}; // struct BuildMoveMapsOp

template <typename LeafT>
inline Index
indexOffsetFromVoxel(const Index voxelOffset, const LeafT& leaf, IndexArray& offsets)
{
    // compute the target point index by summing the point index of the previous
    // voxel with the current number of points added to this voxel, tracked by the
    // offsets array

    Index targetOffset = offsets[voxelOffset]++;
    if (voxelOffset > 0) {
        targetOffset += static_cast<Index>(leaf.getValue(voxelOffset - 1));
    }
    return targetOffset;
}

template <typename TreeT>
struct GlobalMovePointsOp
{
    using LeafT = typename TreeT::LeafNodeType;
    using LeafArrayT = std::vector<LeafT*>;
    using LeafManagerT = typename tree::LeafManager<TreeT>;

    GlobalMovePointsOp(LeafOffsetArray& offsetMap,
                       AttributeHandles& targetHandles,
                       AttributeHandles& sourceHandles,
                       const Index attributeIndex,
                       const GlobalPointIndexMap& moveLeafMap)
        : mOffsetMap(offsetMap)
        , mTargetHandles(targetHandles)
        , mSourceHandles(sourceHandles)
        , mAttributeIndex(attributeIndex)
        , mMoveLeafMap(moveLeafMap) { }

    struct PerformTypedMoveOp
    {
        PerformTypedMoveOp(AttributeHandles& targetHandles, AttributeHandles& sourceHandles,
            Index targetOffset, const LeafT& targetLeaf,
            IndexArray& offsets, const IndexTripleArray& indices)
            : mTargetHandles(targetHandles)
            , mSourceHandles(sourceHandles)
            , mTargetOffset(targetOffset)
            , mTargetLeaf(targetLeaf)
            , mOffsets(offsets)
            , mIndices(indices) { }

        template<typename ValueT>
        void operator()(const AttributeArray&) const
        {
            auto& targetHandle = mTargetHandles.getWriteHandle<ValueT>(mTargetOffset);
            targetHandle.expand();

            // build a sorted index vector that references the indices in order of their source
            // leafs and voxels to ensure determinism in the resulting point orders

            std::vector<int> sortedIndices(mIndices.size());
            std::iota(std::begin(sortedIndices), std::end(sortedIndices), 0);
            std::sort(std::begin(sortedIndices), std::end(sortedIndices),
                [&](int i, int j)
                {
                    const Index& indexI0(std::get<0>(mIndices[i]));
                    const Index& indexJ0(std::get<0>(mIndices[j]));
                    if (indexI0 < indexJ0)          return true;
                    if (indexI0 > indexJ0)          return false;
                    return std::get<2>(mIndices[i]) < std::get<2>(mIndices[j]);
                }
            );

            for (const auto& index : sortedIndices) {
                const auto& it = mIndices[index];
                const auto& sourceHandle = mSourceHandles.getHandle<ValueT>(std::get<0>(it));
                const Index targetIndex = indexOffsetFromVoxel(std::get<1>(it), mTargetLeaf, mOffsets);
                for (Index i = 0; i < sourceHandle.stride(); i++) {
                    ValueT sourceValue = sourceHandle.get(std::get<2>(it), i);
                    targetHandle.set(targetIndex, i, sourceValue);
                }
            }
        }

    private:
        AttributeHandles& mTargetHandles;
        AttributeHandles& mSourceHandles;
        Index mTargetOffset;
        const LeafT& mTargetLeaf;
        IndexArray& mOffsets;
        const IndexTripleArray& mIndices;
    }; // struct PerformTypedMoveOp

    void performMove(Index targetOffset, const LeafT& targetLeaf,
        IndexArray& offsets, const IndexTripleArray& indices) const
    {
        auto& targetArray = mTargetHandles.getArray(targetOffset);

        for (const auto& it : indices) {
            const auto& sourceArray = mSourceHandles.getConstArray(std::get<0>(it));
            const Index tgtOffset = indexOffsetFromVoxel(std::get<1>(it), targetLeaf, offsets);
            targetArray.set(tgtOffset, sourceArray, std::get<2>(it));
        }
    }

    void operator()(LeafT& leaf, size_t aIdx) const
    {
        const Index idx(static_cast<Index>(aIdx));
        const auto& moveIndices = mMoveLeafMap.at(aIdx);
        if (moveIndices.empty())  return;

        // extract per-voxel offsets for this leaf

        auto& offsets = mOffsetMap[aIdx];

        const auto& array = leaf.constAttributeArray(mAttributeIndex);

        PerformTypedMoveOp op(mTargetHandles, mSourceHandles, idx, leaf, offsets, moveIndices);
        if (!processTypedArray(array, op)) {
            this->performMove(idx, leaf, offsets, moveIndices);
        }
    }

private:
    LeafOffsetArray& mOffsetMap;
    AttributeHandles& mTargetHandles;
    AttributeHandles& mSourceHandles;
    const Index mAttributeIndex;
    const GlobalPointIndexMap& mMoveLeafMap;
}; // struct GlobalMovePointsOp

template <typename TreeT>
struct LocalMovePointsOp
{
    using LeafT = typename TreeT::LeafNodeType;
    using LeafArrayT = std::vector<LeafT*>;
    using LeafManagerT = typename tree::LeafManager<TreeT>;

    LocalMovePointsOp( LeafOffsetArray& offsetMap,
                       AttributeHandles& targetHandles,
                       const LeafIndexArray& sourceIndices,
                       AttributeHandles& sourceHandles,
                       const Index attributeIndex,
                       const LocalPointIndexMap& moveLeafMap)
        : mOffsetMap(offsetMap)
        , mTargetHandles(targetHandles)
        , mSourceIndices(sourceIndices)
        , mSourceHandles(sourceHandles)
        , mAttributeIndex(attributeIndex)
        , mMoveLeafMap(moveLeafMap) { }

    struct PerformTypedMoveOp
    {
        PerformTypedMoveOp(AttributeHandles& targetHandles, AttributeHandles& sourceHandles,
            Index targetOffset, Index sourceOffset, const LeafT& targetLeaf,
            IndexArray& offsets, const IndexPairArray& indices)
            : mTargetHandles(targetHandles)
            , mSourceHandles(sourceHandles)
            , mTargetOffset(targetOffset)
            , mSourceOffset(sourceOffset)
            , mTargetLeaf(targetLeaf)
            , mOffsets(offsets)
            , mIndices(indices) { }

        template<typename ValueT>
        void operator()(const AttributeArray&) const
        {
            auto& targetHandle = mTargetHandles.getWriteHandle<ValueT>(mTargetOffset);
            const auto& sourceHandle = mSourceHandles.getHandle<ValueT>(mSourceOffset);

            targetHandle.expand();

            for (const auto& it : mIndices) {
                const Index targetIndex = indexOffsetFromVoxel(it.first, mTargetLeaf, mOffsets);
                for (Index i = 0; i < sourceHandle.stride(); i++) {
                    ValueT sourceValue = sourceHandle.get(it.second, i);
                    targetHandle.set(targetIndex, i, sourceValue);
                }
            }
        }

    private:
        AttributeHandles& mTargetHandles;
        AttributeHandles& mSourceHandles;
        Index mTargetOffset;
        Index mSourceOffset;
        const LeafT& mTargetLeaf;
        IndexArray& mOffsets;
        const IndexPairArray& mIndices;
    }; // struct PerformTypedMoveOp

    template <typename ValueT>
    void performTypedMove(Index sourceOffset, Index targetOffset, const LeafT& targetLeaf,
        IndexArray& offsets, const IndexPairArray& indices) const
    {
        auto& targetHandle = mTargetHandles.getWriteHandle<ValueT>(targetOffset);
        const auto& sourceHandle = mSourceHandles.getHandle<ValueT>(sourceOffset);

        targetHandle.expand();

        for (const auto& it : indices) {
            const Index tgtOffset = indexOffsetFromVoxel(it.first, targetLeaf, offsets);
            for (Index i = 0; i < sourceHandle.stride(); i++) {
                ValueT sourceValue = sourceHandle.get(it.second, i);
                targetHandle.set(tgtOffset, i, sourceValue);
            }
        }
    }

    void performMove(Index targetOffset, Index sourceOffset, const LeafT& targetLeaf,
        IndexArray& offsets, const IndexPairArray& indices) const
    {
        auto& targetArray = mTargetHandles.getArray(targetOffset);
        const auto& sourceArray = mSourceHandles.getConstArray(sourceOffset);

        for (const auto& it : indices) {
            const Index tgtOffset = indexOffsetFromVoxel(it.first, targetLeaf, offsets);
            targetArray.set(tgtOffset, sourceArray, it.second);
        }
    }

    void operator()(const LeafT& leaf, size_t aIdx) const
    {
        const Index idx(static_cast<Index>(aIdx));
        const auto& moveIndices = mMoveLeafMap.at(aIdx);
        if (moveIndices.empty())  return;

        // extract target leaf and per-voxel offsets for this leaf

        auto& offsets = mOffsetMap[aIdx];

        // extract source leaf that has the same origin as the target leaf (if any)

        assert(aIdx < mSourceIndices.size());
        const Index sourceOffset(mSourceIndices[aIdx]);

        const auto& array = leaf.constAttributeArray(mAttributeIndex);

        PerformTypedMoveOp op(mTargetHandles, mSourceHandles,
            idx, sourceOffset, leaf, offsets, moveIndices);
        if (!processTypedArray(array, op)) {
            this->performMove(idx, sourceOffset, leaf, offsets, moveIndices);
        }
    }

private:
    LeafOffsetArray& mOffsetMap;
    AttributeHandles& mTargetHandles;
    const LeafIndexArray& mSourceIndices;
    AttributeHandles& mSourceHandles;
    const Index mAttributeIndex;
    const LocalPointIndexMap& mMoveLeafMap;
}; // struct LocalMovePointsOp

} // namespace point_move_internal


////////////////////////////////////////


template <typename PointDataGridT, typename DeformerT, typename FilterT>
inline void movePoints( PointDataGridT& points,
                        const math::Transform& transform,
                        DeformerT& deformer,
                        const FilterT& filter,
                        future::Advect* objectNotInUse,
                        bool threaded)
{
    using LeafIndex = point_move_internal::LeafIndex;
    using PointDataTreeT = typename PointDataGridT::TreeType;
    using LeafT = typename PointDataTreeT::LeafNodeType;
    using LeafManagerT = typename tree::LeafManager<PointDataTreeT>;

    using namespace point_move_internal;

    // this object is for future use only
    assert(!objectNotInUse);
    (void)objectNotInUse;

    PointDataTreeT& tree = points.tree();

    // early exit if no LeafNodes

    PointDataTree::LeafCIter iter = tree.cbeginLeaf();

    if (!iter)      return;

    // build voxel topology taking into account any point group deletion

    auto newPoints = point_mask_internal::convertPointsToScalar<PointDataGrid>(
        points, transform, filter, deformer, threaded);
    auto& newTree = newPoints->tree();

    // create leaf managers for both trees

    LeafManagerT sourceLeafManager(tree);
    LeafManagerT targetLeafManager(newTree);

    // initialize attribute handles
    AttributeHandles sourceHandles(sourceLeafManager.leafCount());
    AttributeHandles targetHandles(targetLeafManager.leafCount());

    // extract the existing attribute set
    const auto& existingAttributeSet = points.tree().cbeginLeaf()->attributeSet();

    // build a coord -> index map for looking up target leafs by origin and a faster
    // unordered map for finding the source index from a target index

    LeafMap targetLeafMap;
    LeafIndexArray sourceIndices(targetLeafManager.leafCount(),
        std::numeric_limits<LeafIndex>::max());

    LeafOffsetArray offsetMap(targetLeafManager.leafCount());

    {
        LeafMap sourceLeafMap;
        auto sourceRange = sourceLeafManager.leafRange();
        for (auto leaf = sourceRange.begin(); leaf; ++leaf) {
            sourceLeafMap.insert({leaf->origin(), LeafIndex(static_cast<LeafIndex>(leaf.pos()))});
        }
        auto targetRange = targetLeafManager.leafRange();
        for (auto leaf = targetRange.begin(); leaf; ++leaf) {
            targetLeafMap.insert({leaf->origin(), LeafIndex(static_cast<LeafIndex>(leaf.pos()))});
        }

        // perform four independent per-leaf operations in parallel
        targetLeafManager.foreach(
            [&](LeafT& leaf, size_t idx) {
                // map frequency => cumulative histogram
                auto* buffer = leaf.buffer().data();
                for (Index i = 1; i < leaf.buffer().size(); i++) {
                    buffer[i] = buffer[i-1] + buffer[i];
                }
                // replace attribute set with a copy of the existing one
                leaf.replaceAttributeSet(new AttributeSet(existingAttributeSet, leaf.getLastValue()),
                    /*allowMismatchingDescriptors=*/true);
                // store the index of the source leaf in a corresponding target leaf array
                const auto it = sourceLeafMap.find(leaf.origin());
                if (it != sourceLeafMap.end()) {
                    sourceIndices[idx] = it->second;
                }
                // allocate offset maps
                offsetMap[idx].resize(LeafT::SIZE);
            },
        threaded);
    }

    // moving leaf

    GlobalPointIndexMap globalMoveLeafMap(targetLeafManager.leafCount());
    LocalPointIndexMap localMoveLeafMap(targetLeafManager.leafCount());

    // build global and local move leaf maps and update local positions

    if (filter.state() == index::ALL) {
        NullFilter nullFilter;
        BuildMoveMapsOp<DeformerT, PointDataTreeT, NullFilter> op(deformer,
            globalMoveLeafMap, localMoveLeafMap, targetLeafMap,
            transform, points.transform(), nullFilter);
        sourceLeafManager.foreach(op, threaded);
    } else {
        BuildMoveMapsOp<DeformerT, PointDataTreeT, FilterT> op(deformer,
            globalMoveLeafMap, localMoveLeafMap, targetLeafMap,
            transform, points.transform(), filter);
        sourceLeafManager.foreach(op, threaded);
    }

    for (const auto& it : existingAttributeSet.descriptor().map()) {

        const Index attributeIndex = static_cast<Index>(it.second);

        // zero offsets
        targetLeafManager.foreach(
            [&offsetMap](const LeafT& /*leaf*/, size_t idx) {
                std::fill(offsetMap[idx].begin(), offsetMap[idx].end(), 0);
            },
        threaded);

        // cache attribute handles

        sourceHandles.cache(sourceLeafManager.leafRange(), attributeIndex);
        targetHandles.cache(targetLeafManager.leafRange(), attributeIndex);

        // move points between leaf nodes

        GlobalMovePointsOp<PointDataTreeT> globalMoveOp(offsetMap, targetHandles,
            sourceHandles, attributeIndex, globalMoveLeafMap);
        targetLeafManager.foreach(globalMoveOp, threaded);

        // move points within leaf nodes

        LocalMovePointsOp<PointDataTreeT> localMoveOp(offsetMap, targetHandles,
            sourceIndices, sourceHandles,
            attributeIndex, localMoveLeafMap);
        targetLeafManager.foreach(localMoveOp, threaded);
    }

    points.setTree(newPoints->treePtr());
}


template <typename PointDataGridT, typename DeformerT, typename FilterT>
inline void movePoints( PointDataGridT& points,
                        DeformerT& deformer,
                        const FilterT& filter,
                        future::Advect* objectNotInUse,
                        bool threaded)
{
    movePoints(points, points.transform(), deformer, filter, objectNotInUse, threaded);
}


////////////////////////////////////////


template <typename T>
CachedDeformer<T>::CachedDeformer(Cache& cache)
    : mCache(cache) { }


template <typename T>
template <typename PointDataGridT, typename DeformerT, typename FilterT>
void CachedDeformer<T>::evaluate(PointDataGridT& grid, DeformerT& deformer, const FilterT& filter,
    bool threaded)
{
    using TreeT = typename PointDataGridT::TreeType;
    using LeafT = typename TreeT::LeafNodeType;
    using LeafManagerT = typename tree::LeafManager<TreeT>;
    LeafManagerT leafManager(grid.tree());

    // initialize cache
    auto& leafs = mCache.leafs;
    leafs.resize(leafManager.leafCount());

    const auto& transform = grid.transform();

    // insert deformed positions into the cache

    auto cachePositionsOp = [&](const LeafT& leaf, size_t idx) {

        const Index64 totalPointCount = leaf.pointCount();
        if (totalPointCount == 0)   return;

        // deformer is copied to ensure that it is unique per-thread

        DeformerT newDeformer(deformer);

        // if more than half the number of total points are evaluated by the filter, prefer
        // accessing the data from a vector instead of a hash map for faster performance
        const Index64 vectorThreshold = totalPointCount / 2;

        newDeformer.reset(leaf, idx);

        auto handle = AttributeHandle<Vec3f>::create(leaf.constAttributeArray("P"));

        auto& cache = leafs[idx];
        cache.clear();

        // only insert into a vector directly if the filter evaluates all points and the
        // number of active points is greater than the vector threshold
        const bool useVector = filter.state() == index::ALL &&
            (leaf.isDense() || (leaf.onPointCount() > vectorThreshold));
        if (useVector) {
            cache.vecData.resize(totalPointCount);
        }

        for (auto iter = leaf.beginIndexOn(filter); iter; iter++) {

            // extract index-space position and apply index-space deformation (if defined)

            Vec3d position = handle->get(*iter) + iter.getCoord().asVec3d();

            // if deformer is designed to be used in index-space, perform deformation prior
            // to transforming position to world-space, otherwise perform deformation afterwards

            if (DeformerTraits<DeformerT>::IndexSpace) {
                newDeformer.apply(position, iter);
                position = transform.indexToWorld(position);
            }
            else {
                position = transform.indexToWorld(position);
                newDeformer.apply(position, iter);
            }

            // insert new position into the cache

            if (useVector) {
                cache.vecData[*iter] = static_cast<Vec3T>(position);
            }
            else {
                cache.mapData.insert({*iter, static_cast<Vec3T>(position)});
            }
        }

        // after insertion, move the data into a vector if the threshold is reached

        if (!useVector && cache.mapData.size() > vectorThreshold) {
            cache.vecData.resize(totalPointCount);
            for (const auto& it : cache.mapData) {
                cache.vecData[it.first] = it.second;
            }
            cache.mapData.clear();
        }

        // store the total number of points to allow use of an expanded vector on access

        if (!cache.mapData.empty()) {
            cache.totalSize = static_cast<Index>(totalPointCount);
        }
    };

    leafManager.foreach(cachePositionsOp, threaded);
}


template <typename T>
template <typename LeafT>
void CachedDeformer<T>::reset(const LeafT& /*leaf*/, size_t idx)
{
    if (idx >= mCache.leafs.size()) {
        if (mCache.leafs.empty()) {
            throw IndexError("No leafs in cache, perhaps CachedDeformer has not been evaluated?");
        } else {
            throw IndexError("Leaf index is out-of-range of cache leafs.");
        }
    }
    auto& cache = mCache.leafs[idx];
    if (!cache.mapData.empty()) {
        // expand into a local vector if there are greater than 16 values in the hash map
        // and the expanded vector would contain fewer values than 256 times those in the
        // hash map, this trades a little extra storage for faster random access performance
        if (cache.mapData.size() > 16 &&
            cache.totalSize < (cache.mapData.size() * 256)) {
            if (cache.totalSize < cache.mapData.size()) {
                throw ValueError("Cache total size is not valid.");
            }
            mLocalLeafVec.resize(cache.totalSize);
            for (const auto& it : cache.mapData) {
                assert(it.first < cache.totalSize);
                mLocalLeafVec[it.first] = it.second;
            }
            mLeafVec = &mLocalLeafVec;
            mLeafMap = nullptr;
        }
        else {
            mLeafMap = &cache.mapData;
            mLeafVec = nullptr;
        }
    }
    else {
        mLeafVec = &cache.vecData;
        mLeafMap = nullptr;
    }
}


template <typename T>
template <typename IndexIterT>
void CachedDeformer<T>::apply(Vec3d& position, const IndexIterT& iter) const
{
    assert(*iter >= 0);

    if (mLeafMap) {
        auto it = mLeafMap->find(*iter);
        if (it == mLeafMap->end())      return;
        position = static_cast<openvdb::Vec3d>(it->second);
    }
    else {
        assert(mLeafVec);

        if (mLeafVec->empty())          return;
        assert(*iter < mLeafVec->size());
        position = static_cast<openvdb::Vec3d>((*mLeafVec)[*iter]);
    }
}


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_MOVE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
