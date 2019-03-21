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

/// @author Nick Avramoussis, Francisco Gochez, Dan Bailey
///
/// @file PointDelete.h
///
/// @brief Methods for deleting points based on group membership

#ifndef OPENVDB_POINTS_POINT_DELETE_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_DELETE_HAS_BEEN_INCLUDED

#include "PointDataGrid.h"
#include "PointGroup.h"
#include "IndexIterator.h"
#include "IndexFilter.h"

#include <openvdb/tools/Prune.h>
#include <openvdb/tree/LeafManager.h>

#include <memory>
#include <string>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


/// @brief   Delete points that are members of specific groups
///
/// @details This method will delete points which are members of any of the supplied groups and
///          will optionally drop the groups from the tree. An invert flag can be used to
///          delete points that belong to none of the groups.
///
/// @param   pointTree    the point tree
/// @param   groups       the groups from which to delete points
/// @param   invert       if enabled, points not belonging to any of the groups will be deleted
/// @param   drop         if enabled and invert is disabled, the groups will be dropped from the tree
///
/// @note    If the invert flag is true, none of the groups will be dropped after deleting points
///          regardless of the value of the drop parameter.

template <typename PointDataTreeT>
inline void deleteFromGroups(PointDataTreeT& pointTree,
                             const std::vector<std::string>& groups,
                             bool invert = false,
                             bool drop = true);

/// @brief   Delete points that are members of a group
///
/// @details This method will delete points which are members of the supplied group and will
///          optionally drop the group from the tree. An invert flag can be used to
///          delete points that belong to none of the groups.
///
/// @param   pointTree    the point tree with the group to delete
/// @param   group        the name of the group to delete
/// @param   invert       if enabled, points not belonging to any of the groups will be deleted
/// @param   drop         if enabled and invert is disabled, the group will be dropped from the tree
///
/// @note    If the invert flag is true, the group will not be dropped after deleting points
///          regardless of the value of the drop parameter.

template <typename PointDataTreeT>
inline void deleteFromGroup(PointDataTreeT& pointTree,
                            const std::string& group,
                            bool invert = false,
                            bool drop = true);


////////////////////////////////////////


namespace point_delete_internal {


struct VectorWrapper
{
    using T = std::vector<std::pair<Index, Index>>;

    VectorWrapper(const T& _data) : data(_data) { }
    operator bool() const { return index < data.size(); }
    VectorWrapper& operator++() { index++; return *this; }
    Index sourceIndex() const { assert(*this); return data[index].first; }
    Index targetIndex() const { assert(*this); return data[index].second; }

private:
    const T& data;
    T::size_type index = 0;
}; // struct VectorWrapper


template <typename PointDataTreeT, typename FilterT>
struct DeleteByFilterOp
{
    using LeafManagerT = tree::LeafManager<PointDataTreeT>;
    using LeafRangeT = typename LeafManagerT::LeafRange;
    using LeafNodeT = typename PointDataTreeT::LeafNodeType;
    using ValueType = typename LeafNodeT::ValueType;

    DeleteByFilterOp(const FilterT& filter)
        : mFilter(filter) { }

    void operator()(const LeafRangeT& range) const
    {
        for (auto leaf = range.begin(); leaf != range.end(); ++leaf) {

            const size_t newSize =
                iterCount(leaf->template beginIndexAll<FilterT>(mFilter));

            // if all points are being deleted, clear the leaf attributes
            if (newSize == 0) {
                leaf->clearAttributes();
                continue;
            }

            // early exit if no points are being deleted

            const size_t currentSize = leaf->getLastValue();
            if (newSize == currentSize) continue;

            const AttributeSet& existingAttributeSet = leaf->attributeSet();
            AttributeSet* newAttributeSet = new AttributeSet(
                existingAttributeSet, static_cast<Index>(newSize));
            const size_t attributeSetSize = existingAttributeSet.size();

            // cache the attribute arrays for efficiency

            std::vector<AttributeArray*> newAttributeArrays;
            std::vector<const AttributeArray*> existingAttributeArrays;

            for (size_t i = 0; i < attributeSetSize; i++) {
                AttributeArray* newArray = newAttributeSet->get(i);
                const AttributeArray* existingArray = existingAttributeSet.getConst(i);

                if (!newArray->hasConstantStride() || !existingArray->hasConstantStride()) {
                    OPENVDB_THROW(openvdb::NotImplementedError,
                        "Transfer of attribute values for dynamic arrays not currently supported.");
                }

                if (newArray->stride() != existingArray->stride()) {
                    OPENVDB_THROW(openvdb::LookupError,
                        "Cannot transfer attribute values with mis-matching strides.");
                }

                newAttributeArrays.push_back(newArray);
                existingAttributeArrays.push_back(existingArray);
            }

            Index attributeIndex = 0;
            std::vector<ValueType> endOffsets;

            endOffsets.reserve(LeafNodeT::NUM_VALUES);

            // now construct new attribute arrays which exclude data from deleted points

#if OPENVDB_ABI_VERSION_NUMBER >= 6
            std::vector<std::pair<Index, Index>> indexMapping;
            indexMapping.reserve(newSize);

            for (auto voxel = leaf->cbeginValueAll(); voxel; ++voxel) {
                for (auto iter = leaf->beginIndexVoxel(voxel.getCoord(), mFilter);
                     iter; ++iter) {
                    indexMapping.emplace_back(*iter, attributeIndex++);
                }
                endOffsets.push_back(static_cast<ValueType>(attributeIndex));
            }

            for (size_t i = 0; i < attributeSetSize; i++) {
                VectorWrapper indexMappingWrapper(indexMapping);
                newAttributeArrays[i]->copyValues(*(existingAttributeArrays[i]), indexMappingWrapper);
            }
#else
            for (auto voxel = leaf->cbeginValueAll(); voxel; ++voxel) {
                for (auto iter = leaf->beginIndexVoxel(voxel.getCoord(), mFilter);
                     iter; ++iter) {
                    for (size_t i = 0; i < attributeSetSize; i++) {
                        newAttributeArrays[i]->set(attributeIndex, *(existingAttributeArrays[i]),
                            *iter);
                    }
                    ++attributeIndex;
                }
                endOffsets.push_back(static_cast<ValueType>(attributeIndex));
            }
#endif

            leaf->replaceAttributeSet(newAttributeSet);
            leaf->setOffsets(endOffsets);
        }
    }

private:
    const FilterT& mFilter;
}; // struct DeleteByFilterOp

} // namespace point_delete_internal


////////////////////////////////////////


template <typename PointDataTreeT>
inline void deleteFromGroups(PointDataTreeT& pointTree,
                             const std::vector<std::string>& groups,
                             bool invert,
                             bool drop)
{
    const typename PointDataTreeT::LeafCIter leafIter = pointTree.cbeginLeaf();

    if (!leafIter)    return;

    const openvdb::points::AttributeSet& attributeSet = leafIter->attributeSet();
    const AttributeSet::Descriptor& descriptor = attributeSet.descriptor();
    std::vector<std::string> availableGroups;

    // determine which of the requested groups exist, and early exit
    // if none are present in the tree

    for (const auto& groupName : groups) {
        if (descriptor.hasGroup(groupName)) {
            availableGroups.push_back(groupName);
        }
    }

    if (availableGroups.empty())    return;

    std::vector<std::string> empty;
    std::unique_ptr<MultiGroupFilter> filter;
    if (invert) {
        filter.reset(new MultiGroupFilter(groups, empty, leafIter->attributeSet()));
    }
    else {
        filter.reset(new MultiGroupFilter(empty, groups, leafIter->attributeSet()));
    }

    tree::LeafManager<PointDataTreeT> leafManager(pointTree);
    point_delete_internal::DeleteByFilterOp<PointDataTreeT, MultiGroupFilter> deleteOp(*filter);
    tbb::parallel_for(leafManager.leafRange(), deleteOp);

    // remove empty leaf nodes

    tools::pruneInactive(pointTree);

    // drop the now-empty groups if requested (unless invert = true)

    if (drop && !invert) {
        dropGroups(pointTree, availableGroups);
    }
}

template <typename PointDataTreeT>
inline void deleteFromGroup(PointDataTreeT& pointTree,
                            const std::string& group,
                            bool invert,
                            bool drop)
{
    std::vector<std::string> groups(1, group);

    deleteFromGroups(pointTree, groups, invert, drop);
}


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_DELETE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
