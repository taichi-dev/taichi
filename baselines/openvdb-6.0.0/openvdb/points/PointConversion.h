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

/// @author Dan Bailey, Nick Avramoussis
///
/// @file points/PointConversion.h
///
/// @brief  Convert points and attributes to and from VDB Point Data grids.

#ifndef OPENVDB_POINTS_POINT_CONVERSION_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_CONVERSION_HAS_BEEN_INCLUDED

#include <openvdb/math/Transform.h>

#include <openvdb/tools/PointIndexGrid.h>
#include <openvdb/tools/PointsToMask.h>
#include <openvdb/util/NullInterrupter.h>

#include "AttributeArrayString.h"
#include "AttributeSet.h"
#include "IndexFilter.h"
#include "PointAttribute.h"
#include "PointDataGrid.h"
#include "PointGroup.h"

#include <tbb/parallel_reduce.h>

#include <type_traits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


/// @brief  Localises points with position into a @c PointDataGrid into two stages:
///         allocation of the leaf attribute data and population of the positions.
///
/// @param  pointIndexGrid  a PointIndexGrid into the points.
/// @param  positions       list of world space point positions.
/// @param  xform           world to index space transform.
/// @param  positionDefaultValue metadata default position value
///
/// @note   The position data must be supplied in a Point-Partitioner compatible
///         data structure. A convenience PointAttributeVector class is offered.
///
/// @note   The position data is populated separately to perform world space to
///         voxel space conversion and apply quantisation.
///
/// @note   A @c PointIndexGrid to the points must be supplied to perform this
///         operation. Typically this is built implicitly by the PointDataGrid constructor.

template<
    typename CompressionT,
    typename PointDataGridT,
    typename PositionArrayT,
    typename PointIndexGridT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const PointIndexGridT& pointIndexGrid,
                    const PositionArrayT& positions,
                    const math::Transform& xform,
                    Metadata::Ptr positionDefaultValue = Metadata::Ptr());


/// @brief  Convenience method to create a @c PointDataGrid from a std::vector of
///         point positions.
///
/// @param  positions     list of world space point positions.
/// @param  xform         world to index space transform.
/// @param  positionDefaultValue metadata default position value
///
/// @note   This method implicitly wraps the std::vector for a Point-Partitioner compatible
///         data structure and creates the required @c PointIndexGrid to the points.

template <typename CompressionT, typename PointDataGridT, typename ValueT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const std::vector<ValueT>& positions,
                    const math::Transform& xform,
                    Metadata::Ptr positionDefaultValue = Metadata::Ptr());


/// @brief  Stores point attribute data in an existing @c PointDataGrid attribute.
///
/// @param  tree            the PointDataGrid to be populated.
/// @param  pointIndexTree  a PointIndexTree into the points.
/// @param  attributeName   the name of the VDB Points attribute to be populated.
/// @param  data            a wrapper to the attribute data.
/// @param  stride          the stride of the attribute
/// @param  insertMetadata  true if strings are to be automatically inserted as metadata.
///
/// @note   A @c PointIndexGrid to the points must be supplied to perform this
///         operation. This is required to ensure the same point index ordering.
template <typename PointDataTreeT, typename PointIndexTreeT, typename PointArrayT>
inline void
populateAttribute(  PointDataTreeT& tree,
                    const PointIndexTreeT& pointIndexTree,
                    const openvdb::Name& attributeName,
                    const PointArrayT& data,
                    const Index stride = 1,
                    const bool insertMetadata = true);

/// @brief Convert the position attribute from a Point Data Grid
///
/// @param positionAttribute    the position attribute to be populated.
/// @param grid                 the PointDataGrid to be converted.
/// @param pointOffsets         a vector of cumulative point offsets for each leaf
/// @param startOffset          a value to shift all the point offsets by
/// @param filter               an index filter
/// @param inCoreOnly           true if out-of-core leaf nodes are to be ignored
///

template <typename PositionAttribute, typename PointDataGridT, typename FilterT = NullFilter>
inline void
convertPointDataGridPosition(   PositionAttribute& positionAttribute,
                                const PointDataGridT& grid,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const FilterT& filter = NullFilter(),
                                const bool inCoreOnly = false);


/// @brief Convert the attribute from a PointDataGrid
///
/// @param attribute            the attribute to be populated.
/// @param tree                 the PointDataTree to be converted.
/// @param pointOffsets         a vector of cumulative point offsets for each leaf.
/// @param startOffset          a value to shift all the point offsets by
/// @param arrayIndex           the index in the Descriptor of the array to be converted.
/// @param stride               the stride of the attribute
/// @param filter               an index filter
/// @param inCoreOnly           true if out-of-core leaf nodes are to be ignored
template <typename TypedAttribute, typename PointDataTreeT, typename FilterT = NullFilter>
inline void
convertPointDataGridAttribute(  TypedAttribute& attribute,
                                const PointDataTreeT& tree,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const unsigned arrayIndex,
                                const Index stride = 1,
                                const FilterT& filter = NullFilter(),
                                const bool inCoreOnly = false);


/// @brief Convert the group from a PointDataGrid
///
/// @param group                the group to be populated.
/// @param tree                 the PointDataTree to be converted.
/// @param pointOffsets         a vector of cumulative point offsets for each leaf
/// @param startOffset          a value to shift all the point offsets by
/// @param index                the group index to be converted.
/// @param filter               an index filter
/// @param inCoreOnly           true if out-of-core leaf nodes are to be ignored
///

template <typename Group, typename PointDataTreeT, typename FilterT = NullFilter>
inline void
convertPointDataGridGroup(  Group& group,
                            const PointDataTreeT& tree,
                            const std::vector<Index64>& pointOffsets,
                            const Index64 startOffset,
                            const AttributeSet::Descriptor::GroupIndex index,
                            const FilterT& filter = NullFilter(),
                            const bool inCoreOnly = false);

/// @ brief Given a container of world space positions and a target points per voxel,
/// compute a uniform voxel size that would best represent the storage of the points in a grid.
/// This voxel size is typically used for conversion of the points into a PointDataGrid.
///
/// @param positions        array of world space positions
/// @param pointsPerVoxel   the target number of points per voxel, must be positive and non-zero
/// @param transform        voxel size will be computed using this optional transform if provided
/// @param decimalPlaces    for readability, truncate voxel size to this number of decimals
/// @param interrupter      an optional interrupter
///
/// @note if none or one point provided in positions, the default voxel size of 0.1 will be returned
///
template<typename PositionWrapper, typename InterrupterT = openvdb::util::NullInterrupter>
inline float
computeVoxelSize(  const PositionWrapper& positions,
                   const uint32_t pointsPerVoxel,
                   const math::Mat4d transform = math::Mat4d::identity(),
                   const Index decimalPlaces = 5,
                   InterrupterT* const interrupter = nullptr);


////////////////////////////////////////


/// @brief Point-partitioner compatible STL vector attribute wrapper for convenience
template<typename ValueType>
class PointAttributeVector {
public:
    using PosType = ValueType;
    using value_type= ValueType;

    PointAttributeVector(const std::vector<value_type>& data,
                         const Index stride = 1)
        : mData(data)
        , mStride(stride) { }

    size_t size() const { return mData.size(); }
    void getPos(size_t n, ValueType& xyz) const { xyz = mData[n]; }
    void get(ValueType& value, size_t n) const { value = mData[n]; }
    void get(ValueType& value, size_t n, openvdb::Index m) const { value = mData[n * mStride + m]; }

private:
    const std::vector<value_type>& mData;
    const Index mStride;
}; // PointAttributeVector


////////////////////////////////////////


namespace point_conversion_internal {


// ConversionTraits to create the relevant Attribute Handles from a LeafNode
template <typename T> struct ConversionTraits
{
    using Handle = AttributeHandle<T, UnknownCodec>;
    using WriteHandle = AttributeWriteHandle<T, UnknownCodec>;
    static T zero() { return zeroVal<T>(); }
    template <typename LeafT>
    static typename Handle::Ptr handleFromLeaf(LeafT& leaf, Index index) {
        const AttributeArray& array = leaf.constAttributeArray(index);
        return Handle::create(array);
    }
    template <typename LeafT>
    static typename WriteHandle::Ptr writeHandleFromLeaf(LeafT& leaf, Index index) {
        AttributeArray& array = leaf.attributeArray(index);
        return WriteHandle::create(array);
    }
}; // ConversionTraits
template <> struct ConversionTraits<openvdb::Name>
{
    using Handle = StringAttributeHandle;
    using WriteHandle = StringAttributeWriteHandle;
    static openvdb::Name zero() { return ""; }
    template <typename LeafT>
    static typename Handle::Ptr handleFromLeaf(LeafT& leaf, Index index) {
        const AttributeArray& array = leaf.constAttributeArray(index);
        const AttributeSet::Descriptor& descriptor = leaf.attributeSet().descriptor();
        return Handle::create(array, descriptor.getMetadata());
    }
    template <typename LeafT>
    static typename WriteHandle::Ptr writeHandleFromLeaf(LeafT& leaf, Index index) {
        AttributeArray& array = leaf.attributeArray(index);
        const AttributeSet::Descriptor& descriptor = leaf.attributeSet().descriptor();
        return WriteHandle::create(array, descriptor.getMetadata());
    }
}; // ConversionTraits<openvdb::Name>


template<typename PointDataTreeType, typename PointIndexTreeType>
struct InitialiseAttributesOp {

    using LeafManagerT          = typename tree::LeafManager<PointDataTreeType>;
    using LeafRangeT            = typename LeafManagerT::LeafRange;

    using PointIndexLeafNode    = typename PointIndexTreeType::LeafNodeType;
    using IndexArray            = typename PointIndexLeafNode::IndexArray;

    InitialiseAttributesOp( const PointIndexTreeType& pointIndexTree,
                            const AttributeSet::Descriptor::Ptr& attributeDescriptor)
        : mPointIndexTree(pointIndexTree)
        , mAttributeDescriptor(attributeDescriptor) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {
        for (auto leaf = range.begin(); leaf; ++leaf) {

            // obtain the PointIndexLeafNode (using the origin of the current leaf)

            const PointIndexLeafNode* pointIndexLeaf = mPointIndexTree.probeConstLeaf(leaf->origin());

            if (!pointIndexLeaf)    continue;

            // initialise the attribute storage

            const IndexArray& indices = pointIndexLeaf->indices();

            auto pointCount = static_cast<Index>(indices.size());

            leaf->initializeAttributes(mAttributeDescriptor, pointCount);
        }
    }

    //////////

    const PointIndexTreeType&               mPointIndexTree;
    const AttributeSet::Descriptor::Ptr&    mAttributeDescriptor;
};

template<   typename PointDataTreeType,
            typename PointIndexTreeType,
            typename PositionListType>
struct PopulatePositionAttributeOp {

    using LeafManagerT          = typename tree::LeafManager<PointDataTreeType>;
    using LeafRangeT            = typename LeafManagerT::LeafRange;

    using PointIndexLeafNode    = typename PointIndexTreeType::LeafNodeType;
    using IndexArray            = typename PointIndexLeafNode::IndexArray;

    using ValueType             = typename PositionListType::value_type;

    PopulatePositionAttributeOp(const PointIndexTreeType& pointIndexTree,
                                const math::Transform& transform,
                                const PositionListType& positions)
        : mPointIndexTree(pointIndexTree)
        , mTransform(transform)
        , mPositions(positions) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {

        for (auto leaf = range.begin(); leaf; ++leaf) {

            // obtain the PointIndexLeafNode (using the origin of the current leaf)

            const PointIndexLeafNode* pointIndexLeaf = mPointIndexTree.probeConstLeaf(leaf->origin());

            if (!pointIndexLeaf)    continue;

            auto attributeWriteHandle = AttributeWriteHandle<Vec3f>::create(leaf->attributeArray("P"));

            Index64 index = 0;

            const IndexArray& indices = pointIndexLeaf->indices();

            for (const Index64 i: indices) {
                ValueType positionWorldSpace;
                mPositions.getPos(i, positionWorldSpace);

                const ValueType positionIndexSpace = mTransform.worldToIndex(positionWorldSpace);

                const ValueType positionVoxelSpace = ValueType(
                            positionIndexSpace.x() - math::Round(positionIndexSpace.x()),
                            positionIndexSpace.y() - math::Round(positionIndexSpace.y()),
                            positionIndexSpace.z() - math::Round(positionIndexSpace.z()));

                attributeWriteHandle->set(static_cast<Index>(index), Vec3f(positionVoxelSpace));

                index++;
            }
        }
    }

    //////////

    const PointIndexTreeType&   mPointIndexTree;
    const math::Transform&      mTransform;
    const PositionListType&     mPositions;
};

template<   typename PointDataTreeType,
            typename PointIndexTreeType,
            typename AttributeListType>
struct PopulateAttributeOp {

    using LeafManagerT          = typename tree::LeafManager<PointDataTreeType>;
    using LeafRangeT            = typename LeafManagerT::LeafRange;
    using PointIndexLeafNode    = typename PointIndexTreeType::LeafNodeType;
    using IndexArray            = typename PointIndexLeafNode::IndexArray;
    using ValueType             = typename AttributeListType::value_type;
    using HandleT               = typename ConversionTraits<ValueType>::WriteHandle;

    PopulateAttributeOp(const PointIndexTreeType& pointIndexTree,
                        const AttributeListType& data,
                        const size_t index,
                        const Index stride = 1)
        : mPointIndexTree(pointIndexTree)
        , mData(data)
        , mIndex(index)
        , mStride(stride) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {

        for (auto leaf = range.begin(); leaf; ++leaf) {

            // obtain the PointIndexLeafNode (using the origin of the current leaf)

            const PointIndexLeafNode* pointIndexLeaf =
                mPointIndexTree.probeConstLeaf(leaf->origin());

            if (!pointIndexLeaf)    continue;

            typename HandleT::Ptr attributeWriteHandle =
                ConversionTraits<ValueType>::writeHandleFromLeaf(*leaf, static_cast<Index>(mIndex));

            Index64 index = 0;

            const IndexArray& indices = pointIndexLeaf->indices();

            for (const Index64 leafIndex: indices)
            {
                ValueType value;
                for (Index i = 0; i < mStride; i++) {
                    mData.get(value, leafIndex, i);
                    attributeWriteHandle->set(static_cast<Index>(index), i, value);
                }
                index++;
            }

            // attempt to compact the array

            attributeWriteHandle->compact();
        }
    }

    //////////

    const PointIndexTreeType&   mPointIndexTree;
    const AttributeListType&    mData;
    const size_t                mIndex;
    const Index                 mStride;
};

template<typename PointDataTreeType, typename Attribute, typename FilterT>
struct ConvertPointDataGridPositionOp {

    using LeafNode      = typename PointDataTreeType::LeafNodeType;
    using ValueType     = typename Attribute::ValueType;
    using HandleT       = typename Attribute::Handle;
    using SourceHandleT = AttributeHandle<ValueType>;
    using LeafManagerT  = typename tree::LeafManager<const PointDataTreeType>;
    using LeafRangeT    = typename LeafManagerT::LeafRange;

    ConvertPointDataGridPositionOp( Attribute& attribute,
                                    const std::vector<Index64>& pointOffsets,
                                    const Index64 startOffset,
                                    const math::Transform& transform,
                                    const size_t index,
                                    const FilterT& filter,
                                    const bool inCoreOnly)
        : mAttribute(attribute)
        , mPointOffsets(pointOffsets)
        , mStartOffset(startOffset)
        , mTransform(transform)
        , mIndex(index)
        , mFilter(filter)
        , mInCoreOnly(inCoreOnly)
    {
        // only accept Vec3f as ValueType
        static_assert(VecTraits<ValueType>::Size == 3 &&
                      std::is_floating_point<typename ValueType::ValueType>::value,
                      "ValueType is not Vec3f");
    }

    template <typename IterT>
    void convert(IterT& iter, HandleT& targetHandle,
        SourceHandleT& sourceHandle, Index64& offset) const
    {
        for (; iter; ++iter) {
            const Vec3d xyz = iter.getCoord().asVec3d();
            const Vec3d pos = sourceHandle.get(*iter);
            targetHandle.set(static_cast<Index>(offset++), /*stride=*/0,
                mTransform.indexToWorld(pos + xyz));
        }
    }

    void operator()(const LeafRangeT& range) const
    {
        HandleT pHandle(mAttribute);

        for (auto leaf = range.begin(); leaf; ++leaf) {

            assert(leaf.pos() < mPointOffsets.size());

            if (mInCoreOnly && leaf->buffer().isOutOfCore())    continue;

            Index64 offset = mStartOffset;

            if (leaf.pos() > 0) offset += mPointOffsets[leaf.pos() - 1];

            auto handle = SourceHandleT::create(leaf->constAttributeArray(mIndex));

            if (mFilter.state() == index::ALL) {
                auto iter = leaf->beginIndexOn();
                convert(iter, pHandle, *handle, offset);
            }
            else {
                auto iter = leaf->beginIndexOn(mFilter);
                convert(iter, pHandle, *handle, offset);
            }
        }
    }

    //////////

    Attribute&                              mAttribute;
    const std::vector<Index64>&             mPointOffsets;
    const Index64                           mStartOffset;
    const math::Transform&                  mTransform;
    const size_t                            mIndex;
    const FilterT&                          mFilter;
    const bool                              mInCoreOnly;
}; // ConvertPointDataGridPositionOp


template<typename PointDataTreeType, typename Attribute, typename FilterT>
struct ConvertPointDataGridAttributeOp {

    using LeafNode      = typename PointDataTreeType::LeafNodeType;
    using ValueType     = typename Attribute::ValueType;
    using HandleT       = typename Attribute::Handle;
    using SourceHandleT = typename ConversionTraits<ValueType>::Handle;
    using LeafManagerT  = typename tree::LeafManager<const PointDataTreeType>;
    using LeafRangeT    = typename LeafManagerT::LeafRange;

    ConvertPointDataGridAttributeOp(Attribute& attribute,
                                    const std::vector<Index64>& pointOffsets,
                                    const Index64 startOffset,
                                    const size_t index,
                                    const Index stride,
                                    const FilterT& filter,
                                    const bool inCoreOnly)
        : mAttribute(attribute)
        , mPointOffsets(pointOffsets)
        , mStartOffset(startOffset)
        , mIndex(index)
        , mStride(stride)
        , mFilter(filter)
        , mInCoreOnly(inCoreOnly) { }

    template <typename IterT>
    void convert(IterT& iter, HandleT& targetHandle,
        SourceHandleT& sourceHandle, Index64& offset) const
    {
        if (sourceHandle.isUniform()) {
            const ValueType uniformValue(sourceHandle.get(0));
            for (; iter; ++iter) {
                for (Index i = 0; i < mStride; i++) {
                    targetHandle.set(static_cast<Index>(offset), i, uniformValue);
                }
                offset++;
            }
        }
        else {
            for (; iter; ++iter) {
                for (Index i = 0; i < mStride; i++) {
                    targetHandle.set(static_cast<Index>(offset), i,
                        sourceHandle.get(*iter, /*stride=*/i));
                }
                offset++;
            }
        }
    }

    void operator()(const LeafRangeT& range) const
    {
        HandleT pHandle(mAttribute);

        for (auto leaf = range.begin(); leaf; ++leaf) {

            assert(leaf.pos() < mPointOffsets.size());

            if (mInCoreOnly && leaf->buffer().isOutOfCore())    continue;

            Index64 offset = mStartOffset;

            if (leaf.pos() > 0) offset += mPointOffsets[leaf.pos() - 1];

            typename SourceHandleT::Ptr handle = ConversionTraits<ValueType>::handleFromLeaf(
                *leaf, static_cast<Index>(mIndex));

            if (mFilter.state() == index::ALL) {
                auto iter = leaf->beginIndexOn();
                convert(iter, pHandle, *handle, offset);
            } else {
                auto iter = leaf->beginIndexOn(mFilter);
                convert(iter, pHandle, *handle, offset);
            }
        }
    }

    //////////

    Attribute&                              mAttribute;
    const std::vector<Index64>&             mPointOffsets;
    const Index64                           mStartOffset;
    const size_t                            mIndex;
    const Index                             mStride;
    const FilterT&                          mFilter;
    const bool                              mInCoreOnly;
}; // ConvertPointDataGridAttributeOp

template<typename PointDataTreeType, typename Group, typename FilterT>
struct ConvertPointDataGridGroupOp {

    using LeafNode      = typename PointDataTreeType::LeafNodeType;
    using GroupIndex    = AttributeSet::Descriptor::GroupIndex;
    using LeafManagerT  = typename tree::LeafManager<const PointDataTreeType>;
    using LeafRangeT    = typename LeafManagerT::LeafRange;

    ConvertPointDataGridGroupOp(Group& group,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const AttributeSet::Descriptor::GroupIndex index,
                                const FilterT& filter,
                                const bool inCoreOnly)
        : mGroup(group)
        , mPointOffsets(pointOffsets)
        , mStartOffset(startOffset)
        , mIndex(index)
        , mFilter(filter)
        , mInCoreOnly(inCoreOnly) { }

    template <typename IterT>
    void convert(IterT& iter, const GroupAttributeArray& groupArray, Index64& offset) const
    {
        const auto bitmask = static_cast<GroupType>(1 << mIndex.second);

        if (groupArray.isUniform()) {
            if (groupArray.get(0) & bitmask) {
                for (; iter; ++iter) {
                    mGroup.setOffsetOn(static_cast<Index>(offset));
                    offset++;
                }
            }
        }
        else {
            for (; iter; ++iter) {
                if (groupArray.get(*iter) & bitmask) {
                    mGroup.setOffsetOn(static_cast<Index>(offset));
                }
                offset++;
            }
        }
    }

    void operator()(const LeafRangeT& range) const
    {
        for (auto leaf = range.begin(); leaf; ++leaf) {

            assert(leaf.pos() < mPointOffsets.size());

            if (mInCoreOnly && leaf->buffer().isOutOfCore())    continue;

            Index64 offset = mStartOffset;

            if (leaf.pos() > 0)     offset += mPointOffsets[leaf.pos() - 1];

            const AttributeArray& array = leaf->constAttributeArray(mIndex.first);
            assert(isGroup(array));
            const GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

            if (mFilter.state() == index::ALL) {
                auto iter = leaf->beginIndexOn();
                convert(iter, groupArray, offset);
            }
            else {
                auto iter = leaf->beginIndexOn(mFilter);
                convert(iter, groupArray, offset);
            }
        }
    }

    //////////

    Group&                                  mGroup;
    const std::vector<Index64>&             mPointOffsets;
    const Index64                           mStartOffset;
    const GroupIndex                        mIndex;
    const FilterT&                          mFilter;
    const bool                              mInCoreOnly;
}; // ConvertPointDataGridGroupOp

template<typename PositionArrayT>
struct CalculatePositionBounds
{
    CalculatePositionBounds(const PositionArrayT& positions,
                            const math::Mat4d& inverse)
        : mPositions(positions)
        , mInverseMat(inverse)
        , mMin(std::numeric_limits<Real>::max())
        , mMax(-std::numeric_limits<Real>::max()) {}

    CalculatePositionBounds(const CalculatePositionBounds& other, tbb::split)
        : mPositions(other.mPositions)
        , mInverseMat(other.mInverseMat)
        , mMin(std::numeric_limits<Real>::max())
        , mMax(-std::numeric_limits<Real>::max()) {}

    void operator()(const tbb::blocked_range<size_t>& range) {
        Vec3R pos;
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            mPositions.getPos(n, pos);
            pos = mInverseMat.transform(pos);
            mMin = math::minComponent(mMin, pos);
            mMax = math::maxComponent(mMax, pos);
        }
    }

    void join(const CalculatePositionBounds& other) {
        mMin = math::minComponent(mMin, other.mMin);
        mMax = math::maxComponent(mMax, other.mMax);
    }

    BBoxd getBoundingBox() const {
        return BBoxd(mMin, mMax);
    }

private:
    const PositionArrayT& mPositions;
    const math::Mat4d&    mInverseMat;
    Vec3R mMin, mMax;
};

} // namespace point_conversion_internal


////////////////////////////////////////


template<typename CompressionT, typename PointDataGridT, typename PositionArrayT, typename PointIndexGridT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const PointIndexGridT& pointIndexGrid, const PositionArrayT& positions,
                    const math::Transform& xform, Metadata::Ptr positionDefaultValue)
{
    using PointDataTreeT        = typename PointDataGridT::TreeType;
    using PointIndexTreeT       = typename PointIndexGridT::TreeType;
    using LeafManagerT          = typename tree::LeafManager<PointDataTreeT>;
    using LeafRangeT            = typename LeafManagerT::LeafRange;
    using PositionAttributeT    = TypedAttributeArray<Vec3f, CompressionT>;

    using point_conversion_internal::InitialiseAttributesOp;
    using point_conversion_internal::PopulatePositionAttributeOp;

    const NamePair positionType = PositionAttributeT::attributeType();

    // construct the Tree using a topology copy of the PointIndexGrid

    const PointIndexTreeT& pointIndexTree(pointIndexGrid.tree());
    typename PointDataTreeT::Ptr treePtr(new PointDataTreeT(pointIndexTree));

    LeafManagerT leafManager = LeafManagerT(*treePtr);
    LeafRangeT leafRange = leafManager.leafRange();

    // create attribute descriptor from position type

    auto descriptor = AttributeSet::Descriptor::create(positionType);

    // add default value for position if provided

    if (positionDefaultValue)   descriptor->setDefaultValue("P", *positionDefaultValue);

    // create point attribute storage on each leaf

    InitialiseAttributesOp<PointDataTreeT, PointIndexTreeT> initialise(
                                pointIndexGrid.tree(), descriptor);
    tbb::parallel_for(leafRange, initialise);

    // populate position attribute

    PopulatePositionAttributeOp<PointDataTreeT,
                                PointIndexTreeT,
                                PositionArrayT> populate(pointIndexTree,
                                                        xform,
                                                        positions);

    tbb::parallel_for(leafRange, populate);

    auto grid = PointDataGridT::create(treePtr);
    grid->setTransform(xform.copy());
    return grid;
}


////////////////////////////////////////


template <typename CompressionT, typename PointDataGridT, typename ValueT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const std::vector<ValueT>& positions,
                    const math::Transform& xform,
                    Metadata::Ptr positionDefaultValue)
{
    const PointAttributeVector<ValueT> pointList(positions);

    tools::PointIndexGrid::Ptr pointIndexGrid = tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, xform);
    return createPointDataGrid<CompressionT, PointDataGridT>(*pointIndexGrid, pointList, xform, positionDefaultValue);
}


////////////////////////////////////////


template <typename PointDataTreeT, typename PointIndexTreeT, typename PointArrayT>
inline void
populateAttribute(PointDataTreeT& tree, const PointIndexTreeT& pointIndexTree,
    const openvdb::Name& attributeName, const PointArrayT& data, const Index stride,
    const bool insertMetadata)
{
    using point_conversion_internal::PopulateAttributeOp;
    using ValueType = typename PointArrayT::value_type;

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const size_t index = iter->attributeSet().find(attributeName);

    if (index == AttributeSet::INVALID_POS) {
        OPENVDB_THROW(KeyError, "Attribute not found to populate - " << attributeName << ".");
    }

    if (insertMetadata) {
        point_attribute_internal::MetadataStorage<PointDataTreeT, ValueType>::add(tree, data);
    }

    // populate attribute

    typename tree::LeafManager<PointDataTreeT> leafManager(tree);

    PopulateAttributeOp<PointDataTreeT,
                        PointIndexTreeT,
                        PointArrayT> populate(pointIndexTree, data, index, stride);
    tbb::parallel_for(leafManager.leafRange(), populate);
}


////////////////////////////////////////


template <typename PositionAttribute, typename PointDataGridT, typename FilterT>
inline void
convertPointDataGridPosition(   PositionAttribute& positionAttribute,
                                const PointDataGridT& grid,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const FilterT& filter,
                                const bool inCoreOnly)
{
    using TreeType      = typename PointDataGridT::TreeType;
    using LeafManagerT  = typename tree::LeafManager<const TreeType>;

    using point_conversion_internal::ConvertPointDataGridPositionOp;

    const TreeType& tree = grid.tree();
    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const size_t positionIndex = iter->attributeSet().find("P");

    positionAttribute.expand();
    LeafManagerT leafManager(tree);
    ConvertPointDataGridPositionOp<TreeType, PositionAttribute, FilterT> convert(
                    positionAttribute, pointOffsets, startOffset, grid.transform(), positionIndex,
                    filter, inCoreOnly);
    tbb::parallel_for(leafManager.leafRange(), convert);
    positionAttribute.compact();
}


////////////////////////////////////////


template <typename TypedAttribute, typename PointDataTreeT, typename FilterT>
inline void
convertPointDataGridAttribute(  TypedAttribute& attribute,
                                const PointDataTreeT& tree,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const unsigned arrayIndex,
                                const Index stride,
                                const FilterT& filter,
                                const bool inCoreOnly)
{
    using LeafManagerT = typename tree::LeafManager<const PointDataTreeT>;

    using point_conversion_internal::ConvertPointDataGridAttributeOp;

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    attribute.expand();
    LeafManagerT leafManager(tree);
    ConvertPointDataGridAttributeOp<PointDataTreeT, TypedAttribute, FilterT> convert(
                        attribute, pointOffsets, startOffset, arrayIndex, stride,
                        filter, inCoreOnly);
        tbb::parallel_for(leafManager.leafRange(), convert);
    attribute.compact();
}


////////////////////////////////////////


template <typename Group, typename PointDataTreeT, typename FilterT>
inline void
convertPointDataGridGroup(  Group& group,
                            const PointDataTreeT& tree,
                            const std::vector<Index64>& pointOffsets,
                            const Index64 startOffset,
                            const AttributeSet::Descriptor::GroupIndex index,
                            const FilterT& filter,
                            const bool inCoreOnly)
{
    using LeafManagerT= typename tree::LeafManager<const PointDataTreeT>;

    using point_conversion_internal::ConvertPointDataGridGroupOp;

    auto iter = tree.cbeginLeaf();
    if (!iter)  return;

    LeafManagerT leafManager(tree);
    ConvertPointDataGridGroupOp<PointDataTree, Group, FilterT> convert(
                    group, pointOffsets, startOffset, index,
                    filter, inCoreOnly);
    tbb::parallel_for(leafManager.leafRange(), convert);

    // must call this after modifying point groups in parallel

    group.finalize();
}

template<typename PositionWrapper, typename InterrupterT>
inline float
computeVoxelSize(  const PositionWrapper& positions,
                   const uint32_t pointsPerVoxel,
                   const math::Mat4d transform,
                   const Index decimalPlaces,
                   InterrupterT* const interrupter)
{
    using namespace point_conversion_internal;

    struct Local {

        static bool voxelSizeFromVolume(const double volume,
                                        const size_t estimatedVoxelCount,
                                        float& voxelSize)
        {
            // dictated by the math::ScaleMap limit
            static const double minimumVoxelVolume(3e-15);
            static const double maximumVoxelVolume(std::numeric_limits<float>::max());

            double voxelVolume = volume / static_cast<double>(estimatedVoxelCount);
            bool valid = true;

            if (voxelVolume < minimumVoxelVolume) {
                voxelVolume = minimumVoxelVolume;
                valid = false;
            }
            else if (voxelVolume > maximumVoxelVolume) {
                voxelVolume = maximumVoxelVolume;
                valid = false;
            }

            voxelSize = static_cast<float>(math::Pow(voxelVolume, 1.0/3.0));
            return valid;
        }

        static float truncate(const float voxelSize, Index decPlaces)
        {
            float truncatedVoxelSize = voxelSize;

            // attempt to truncate from decPlaces -> 11
            for (int i = decPlaces; i < 11; i++) {
                truncatedVoxelSize = static_cast<float>(math::Truncate(double(voxelSize), i));
                if (truncatedVoxelSize != 0.0f)     break;
            }

            return truncatedVoxelSize;
        }
    };

    if (pointsPerVoxel == 0) OPENVDB_THROW(ValueError, "Points per voxel cannot be zero.");

    // constructed with the default voxel size as specified by openvdb interface values

    float voxelSize(0.1f);

    const size_t numPoints = positions.size();

    // return the default voxel size if we have zero or only 1 point

    if (numPoints <= 1) return voxelSize;

    size_t targetVoxelCount(numPoints / size_t(pointsPerVoxel));
    if (targetVoxelCount == 0)   targetVoxelCount++;

    // calculate the world space, transform-oriented bounding box

    math::Mat4d inverseTransform = transform.inverse();
    inverseTransform = math::unit(inverseTransform);

    tbb::blocked_range<size_t> range(0, numPoints);
    CalculatePositionBounds<PositionWrapper> calculateBounds(positions, inverseTransform);
    tbb::parallel_reduce(range, calculateBounds);

    BBoxd bbox = calculateBounds.getBoundingBox();

    // return default size if points are coincident

    if (bbox.min() == bbox.max())  return voxelSize;

    double volume = bbox.volume();

    // handle points that are collinear or coplanar by expanding the volume

    if (math::isApproxZero(volume)) {
        Vec3d extents = bbox.extents().sorted().reversed();
        if (math::isApproxZero(extents[1])) {
            // colinear (maxExtent^3)
            volume = extents[0]*extents[0]*extents[0];
        }
        else {
            // coplanar (maxExtent*nextMaxExtent^2)
            volume = extents[0]*extents[1]*extents[1];
        }
    }

    double previousVolume = volume;

    if (!Local::voxelSizeFromVolume(volume, targetVoxelCount, voxelSize)) {
        OPENVDB_LOG_DEBUG("Out of range, clamping voxel size.");
        return voxelSize;
    }

    size_t previousVoxelCount(0);
    size_t voxelCount(1);

    if (interrupter) interrupter->start("Computing voxel size");

    while (voxelCount > previousVoxelCount)
    {
        math::Transform::Ptr newTransform;

        if (!math::isIdentity(transform))
        {
            // if using a custom transform, pre-scale by coefficients
            // which define the new voxel size

            math::Mat4d matrix(transform);
            matrix.preScale(Vec3d(voxelSize) / math::getScale(matrix));
            newTransform = math::Transform::createLinearTransform(matrix);
        }
        else
        {
            newTransform = math::Transform::createLinearTransform(voxelSize);
        }

        // create a mask grid of the points from the calculated voxel size
        // this is the same function call as tools::createPointMask() which has
        // been duplicated to provide an interrupter

        MaskGrid::Ptr mask = createGrid<MaskGrid>(false);
        mask->setTransform(newTransform);
        tools::PointsToMask<MaskGrid, InterrupterT> pointMaskOp(*mask, interrupter);
        pointMaskOp.addPoints(positions);

        if (interrupter && util::wasInterrupted(interrupter)) break;

        previousVoxelCount = voxelCount;
        voxelCount = mask->activeVoxelCount();
        volume = math::Pow3(voxelSize) * voxelCount;

        // stop if no change in the volume or the volume has increased

        if (volume >= previousVolume) break;
        previousVolume = volume;

        const float previousVoxelSize = voxelSize;

        // compute the new voxel size and if invalid return the previous value

        if (!Local::voxelSizeFromVolume(volume, targetVoxelCount, voxelSize)) {
            voxelSize = previousVoxelSize;
            break;
        }

        // halt convergence if the voxel size has decreased by less
        // than 10% in this iteration

        if (voxelSize / previousVoxelSize > 0.9f) break;
    }

    if (interrupter) interrupter->end();

    // truncate the voxel size for readability and return the value

    return Local::truncate(voxelSize, decimalPlaces);
}


////////////////////////////////////////


// deprecated functions


template <typename PositionAttribute, typename PointDataGridT>
OPENVDB_DEPRECATED
inline void
convertPointDataGridPosition(   PositionAttribute& positionAttribute,
                                const PointDataGridT& grid,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const std::vector<Name>& includeGroups,
                                const std::vector<Name>& excludeGroups,
                                const bool inCoreOnly = false)
{
    auto leaf = grid.tree().cbeginLeaf();
    if (!leaf)  return;
    MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
    convertPointDataGridPosition(positionAttribute, grid, pointOffsets, startOffset,
        filter, inCoreOnly);
}


template <typename TypedAttribute, typename PointDataTreeT>
OPENVDB_DEPRECATED
inline void
convertPointDataGridAttribute(  TypedAttribute& attribute,
                                const PointDataTreeT& tree,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const unsigned arrayIndex,
                                const Index stride,
                                const std::vector<Name>& includeGroups,
                                const std::vector<Name>& excludeGroups,
                                const bool inCoreOnly = false)
{
    auto leaf = tree.cbeginLeaf();
    if (!leaf)  return;
    MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
    convertPointDataGridAttribute(attribute, tree, pointOffsets, startOffset,
        arrayIndex, stride, filter, inCoreOnly);
}


template <typename Group, typename PointDataTreeT>
OPENVDB_DEPRECATED
inline void
convertPointDataGridGroup(  Group& group,
                            const PointDataTreeT& tree,
                            const std::vector<Index64>& pointOffsets,
                            const Index64 startOffset,
                            const AttributeSet::Descriptor::GroupIndex index,
                            const std::vector<Name>& includeGroups,
                            const std::vector<Name>& excludeGroups,
                            const bool inCoreOnly = false)
{
    auto leaf = tree.cbeginLeaf();
    if (!leaf)  return;
    MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
    convertPointDataGridGroup(group, tree, pointOffsets, startOffset,
        index, filter, inCoreOnly);
}


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_CONVERSION_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
