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

/// @file Mask.h
///
/// @brief Construct boolean mask grids from grids of arbitrary type

#ifndef OPENVDB_TOOLS_MASK_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MASK_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include "LevelSetUtil.h" // for tools::sdfInteriorMask()
#include <type_traits> // for std::enable_if, std::is_floating_point


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Given an input grid of any type, return a new, boolean grid
/// whose active voxel topology matches the input grid's or,
/// if the input grid is a level set, matches the input grid's interior.
/// @param grid      the grid from which to construct a mask
/// @param isovalue  for a level set grid, the isovalue that defines the grid's interior
/// @sa tools::sdfInteriorMask()
template<typename GridType>
inline typename GridType::template ValueConverter<bool>::Type::Ptr
interiorMask(const GridType& grid, const double isovalue = 0.0);


////////////////////////////////////////


namespace mask_internal {

/// @private
template<typename GridType>
struct Traits {
    static const bool isBool = std::is_same<typename GridType::ValueType, bool>::value;
    using BoolGridType = typename GridType::template ValueConverter<bool>::Type;
    using BoolGridPtrType = typename BoolGridType::Ptr;
};


/// @private
template<typename GridType>
inline typename std::enable_if<std::is_floating_point<typename GridType::ValueType>::value,
    typename mask_internal::Traits<GridType>::BoolGridPtrType>::type
doLevelSetInteriorMask(const GridType& grid, const double isovalue)
{
    using GridValueT = typename GridType::ValueType;
    using MaskGridPtrT = typename mask_internal::Traits<GridType>::BoolGridPtrType;

    // If the input grid is a level set (and floating-point), return a mask of its interior.
    if (grid.getGridClass() == GRID_LEVEL_SET) {
        return tools::sdfInteriorMask(grid, static_cast<GridValueT>(isovalue));
    }
    return MaskGridPtrT{};
}


/// @private
// No-op specialization for non-floating-point grids
template<typename GridType>
inline typename std::enable_if<!std::is_floating_point<typename GridType::ValueType>::value,
    typename mask_internal::Traits<GridType>::BoolGridPtrType>::type
doLevelSetInteriorMask(const GridType&, const double /*isovalue*/)
{
    using MaskGridPtrT = typename mask_internal::Traits<GridType>::BoolGridPtrType;
    return MaskGridPtrT{};
}


/// @private
template<typename GridType>
inline typename std::enable_if<mask_internal::Traits<GridType>::isBool,
    typename mask_internal::Traits<GridType>::BoolGridPtrType>::type
doInteriorMask(const GridType& grid, const double /*isovalue*/)
{
    // If the input grid is already boolean, return a copy of it.
    return grid.deepCopy();
}


/// @private
template<typename GridType>
inline typename std::enable_if<!(mask_internal::Traits<GridType>::isBool),
    typename mask_internal::Traits<GridType>::BoolGridPtrType>::type
doInteriorMask(const GridType& grid, const double isovalue)
{
    using MaskGridT = typename mask_internal::Traits<GridType>::BoolGridType;

    // If the input grid is a level set, return a mask of its interior.
    if (auto maskGridPtr = doLevelSetInteriorMask(grid, isovalue)) {
        return maskGridPtr;
    }

    // For any other grid type, return a mask of its active voxels.
    auto maskGridPtr = MaskGridT::create(/*background=*/false);
    maskGridPtr->setTransform(grid.transform().copy());
    maskGridPtr->topologyUnion(grid);
    return maskGridPtr;
}

} // namespace mask_internal


template<typename GridType>
inline typename GridType::template ValueConverter<bool>::Type::Ptr
interiorMask(const GridType& grid, const double isovalue)
{
    return mask_internal::doInteriorMask(grid, isovalue);
}


////////////////////////////////////////

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MASK_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
