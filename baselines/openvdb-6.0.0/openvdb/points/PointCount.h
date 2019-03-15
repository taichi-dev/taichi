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

/// @file points/PointCount.h
///
/// @author Dan Bailey
///
/// @brief  Methods for counting points in VDB Point grids.

#ifndef OPENVDB_POINTS_POINT_COUNT_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_COUNT_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>

#include "PointDataGrid.h"
#include "PointMask.h"
#include "IndexFilter.h"

#include <tbb/parallel_reduce.h>

#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


/// @brief Count the total number of points in a PointDataTree
/// @param tree         the PointDataTree in which to count the points
/// @param filter       an optional index filter
/// @param inCoreOnly   if true, points in out-of-core leaf nodes are not counted
/// @param threaded     enable or disable threading  (threading is enabled by default)
template <typename PointDataTreeT, typename FilterT = NullFilter>
inline Index64 pointCount(  const PointDataTreeT& tree,
                            const FilterT& filter = NullFilter(),
                            const bool inCoreOnly = false,
                            const bool threaded = true);


/// @brief Populate an array of cumulative point offsets per leaf node.
/// @param pointOffsets     array of offsets to be populated
/// @param tree             the PointDataTree from which to populate the offsets
/// @param filter           an optional index filter
/// @param inCoreOnly       if true, points in out-of-core leaf nodes are ignored
/// @param threaded         enable or disable threading  (threading is enabled by default)
/// @return The final cumulative point offset.
template <typename PointDataTreeT, typename FilterT = NullFilter>
inline Index64 pointOffsets(std::vector<Index64>& pointOffsets,
                            const PointDataTreeT& tree,
                            const FilterT& filter = NullFilter(),
                            const bool inCoreOnly = false,
                            const bool threaded = true);


/// @brief Generate a new grid with voxel values to store the number of points per voxel
/// @param grid             the PointDataGrid to use to compute the count grid
/// @param filter           an optional index filter
/// @note The return type of the grid must be an integer or floating-point scalar grid.
template <typename PointDataGridT,
    typename GridT = typename PointDataGridT::template ValueConverter<Int32>::Type,
    typename FilterT = NullFilter>
inline typename GridT::Ptr
pointCountGrid( const PointDataGridT& grid,
                const FilterT& filter = NullFilter());


/// @brief Generate a new grid that uses the supplied transform with voxel values to store the
///        number of points per voxel.
/// @param grid             the PointDataGrid to use to compute the count grid
/// @param transform        the transform to use to compute the count grid
/// @param filter           an optional index filter
/// @note The return type of the grid must be an integer or floating-point scalar grid.
template <typename PointDataGridT,
    typename GridT = typename PointDataGridT::template ValueConverter<Int32>::Type,
    typename FilterT = NullFilter>
inline typename GridT::Ptr
pointCountGrid( const PointDataGridT& grid,
                const openvdb::math::Transform& transform,
                const FilterT& filter = NullFilter());


////////////////////////////////////////


template <typename PointDataTreeT, typename FilterT>
Index64 pointCount(const PointDataTreeT& tree,
                   const FilterT& filter,
                   const bool inCoreOnly,
                   const bool threaded)
{
    using LeafManagerT = tree::LeafManager<const PointDataTreeT>;
    using LeafRangeT = typename LeafManagerT::LeafRange;

    auto countLambda =
        [&filter, &inCoreOnly] (const LeafRangeT& range, Index64 sum) -> Index64 {
            for (const auto& leaf : range) {
                if (inCoreOnly && leaf.buffer().isOutOfCore())  continue;
                auto state = filter.state(leaf);
                if (state == index::ALL) {
                    sum += leaf.pointCount();
                } else if (state != index::NONE) {
                    sum += iterCount(leaf.beginIndexAll(filter));
                }
            }
            return sum;
        };

    LeafManagerT leafManager(tree);
    if (threaded) {
        return tbb::parallel_reduce(leafManager.leafRange(), Index64(0), countLambda,
            [] (Index64 n, Index64 m) -> Index64 { return n + m; });
    }
    else {
        return countLambda(leafManager.leafRange(), Index64(0));
    }
}


template <typename PointDataTreeT, typename FilterT>
Index64 pointOffsets(   std::vector<Index64>& pointOffsets,
                        const PointDataTreeT& tree,
                        const FilterT& filter,
                        const bool inCoreOnly,
                        const bool threaded)
{
    using LeafT = typename PointDataTreeT::LeafNodeType;
    using LeafManagerT = typename tree::LeafManager<const PointDataTreeT>;

    // allocate and zero values in point offsets array

    pointOffsets.assign(tree.leafCount(), Index64(0));

    // compute total points per-leaf

    LeafManagerT leafManager(tree);
    leafManager.foreach(
        [&pointOffsets, &filter, &inCoreOnly](const LeafT& leaf, size_t pos) {
            if (inCoreOnly && leaf.buffer().isOutOfCore())  return;
            auto state = filter.state(leaf);
            if (state == index::ALL) {
                pointOffsets[pos] = leaf.pointCount();
            } else if (state != index::NONE) {
                pointOffsets[pos] = iterCount(leaf.beginIndexAll(filter));
            }
        },
    threaded);

    // turn per-leaf totals into cumulative leaf totals

    Index64 pointOffset(pointOffsets[0]);
    for (size_t n = 1; n < pointOffsets.size(); n++) {
        pointOffset += pointOffsets[n];
        pointOffsets[n] = pointOffset;
    }

    return pointOffset;
}


template <typename PointDataGridT, typename GridT, typename FilterT>
typename GridT::Ptr
pointCountGrid( const PointDataGridT& points,
                const FilterT& filter)
{
    static_assert(  std::is_integral<typename GridT::ValueType>::value ||
                    std::is_floating_point<typename GridT::ValueType>::value,
        "openvdb::points::pointCountGrid must return an integer or floating-point scalar grid");

    // This is safe because the PointDataGrid can only be modified by the deformer
    using AdapterT = TreeAdapter<typename PointDataGridT::TreeType>;
    auto& nonConstPoints = const_cast<typename AdapterT::NonConstGridType&>(points);

    return point_mask_internal::convertPointsToScalar<GridT>(
        nonConstPoints, filter);
}


template <typename PointDataGridT, typename GridT, typename FilterT>
typename GridT::Ptr
pointCountGrid( const PointDataGridT& points,
                const openvdb::math::Transform& transform,
                const FilterT& filter)
{
    static_assert(  std::is_integral<typename GridT::ValueType>::value ||
                    std::is_floating_point<typename GridT::ValueType>::value,
        "openvdb::points::pointCountGrid must return an integer or floating-point scalar grid");

    // This is safe because the PointDataGrid can only be modified by the deformer
    using AdapterT = TreeAdapter<typename PointDataGridT::TreeType>;
    auto& nonConstPoints = const_cast<typename AdapterT::NonConstGridType&>(points);

    NullDeformer deformer;
    return point_mask_internal::convertPointsToScalar<GridT>(
        nonConstPoints, transform, filter, deformer);
}


////////////////////////////////////////


// deprecated functions


template <typename PointDataTreeT>
OPENVDB_DEPRECATED
inline Index64 pointCount(const PointDataTreeT& tree, const bool inCoreOnly)
{
    NullFilter filter;
    return pointCount(tree, filter, inCoreOnly);
}


template <typename PointDataTreeT>
OPENVDB_DEPRECATED
inline Index64 activePointCount(const PointDataTreeT& tree, const bool inCoreOnly = true)
{
    ActiveFilter filter;
    return pointCount(tree, filter, inCoreOnly);
}


template <typename PointDataTreeT>
OPENVDB_DEPRECATED
inline Index64 inactivePointCount(const PointDataTreeT& tree, const bool inCoreOnly = true)
{
    InactiveFilter filter;
    return pointCount(tree, filter, inCoreOnly);
}


template <typename PointDataTreeT>
OPENVDB_DEPRECATED
inline Index64 groupPointCount(const PointDataTreeT& tree, const Name& name,
    const bool inCoreOnly = true)
{
    auto iter = tree.cbeginLeaf();
    if (!iter || !iter->attributeSet().descriptor().hasGroup(name)) {
        return Index64(0);
    }
    GroupFilter filter(name, iter->attributeSet());
    return pointCount(tree, filter, inCoreOnly);
}


template <typename PointDataTreeT>
OPENVDB_DEPRECATED
inline Index64 activeGroupPointCount(const PointDataTreeT& tree, const Name& name,
    const bool inCoreOnly = true)
{
    auto iter = tree.cbeginLeaf();
    if (!iter || !iter->attributeSet().descriptor().hasGroup(name)) {
        return Index64(0);
    }
    BinaryFilter<GroupFilter, ActiveFilter> filter(GroupFilter(name, iter->attributeSet()), ActiveFilter());
    return pointCount(tree, filter, inCoreOnly);
}


template <typename PointDataTreeT>
OPENVDB_DEPRECATED
inline Index64 inactiveGroupPointCount(const PointDataTreeT& tree, const Name& name,
    const bool inCoreOnly = true)
{
    auto iter = tree.cbeginLeaf();
    if (!iter || !iter->attributeSet().descriptor().hasGroup(name)) {
        return Index64(0);
    }
    BinaryFilter<GroupFilter, InactiveFilter> filter(GroupFilter(name, iter->attributeSet()), InactiveFilter());
    return pointCount(tree, filter, inCoreOnly);
}


template <typename PointDataTreeT>
OPENVDB_DEPRECATED
inline Index64 getPointOffsets(std::vector<Index64>& offsets, const PointDataTreeT& tree,
                        const std::vector<Name>& includeGroups,
                        const std::vector<Name>& excludeGroups,
                        const bool inCoreOnly = false)
{
    MultiGroupFilter filter(includeGroups, excludeGroups, tree.cbeginLeaf()->attributeSet());
    return pointOffsets(offsets, tree, filter, inCoreOnly);
}


template <typename PointDataGridT,
    typename GridT = typename PointDataGridT::template ValueConverter<Int32>::Type>
OPENVDB_DEPRECATED
inline typename GridT::Ptr
pointCountGrid(const PointDataGridT& grid,
    const std::vector<Name>& includeGroups,
    const std::vector<Name>& excludeGroups)
{
    auto leaf = grid.tree().cbeginLeaf();
    if (!leaf)  return GridT::create(0);
    MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
    return pointCountGrid(grid, filter);
}


template <typename PointDataGridT,
    typename GridT = typename PointDataGridT::template ValueConverter<Int32>::Type>
OPENVDB_DEPRECATED
inline typename GridT::Ptr
pointCountGrid(const PointDataGridT& grid,
    const openvdb::math::Transform& transform,
    const std::vector<Name>& includeGroups,
    const std::vector<Name>& excludeGroups)
{
    auto leaf = grid.tree().cbeginLeaf();
    if (!leaf)  return GridT::create(0);
    MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
    return pointCountGrid(grid, transform, filter);
}


////////////////////////////////////////


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_COUNT_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
