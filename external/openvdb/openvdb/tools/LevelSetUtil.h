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

/// @file tools/LevelSetUtil.h
///
/// @brief  Miscellaneous utility methods that operate primarily
///         or exclusively on level set grids.
///
/// @author Mihai Alden

#ifndef OPENVDB_TOOLS_LEVEL_SET_UTIL_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_UTIL_HAS_BEEN_INCLUDED

#include "MeshToVolume.h" // for traceExteriorBoundaries
#include "SignedFloodFill.h" // for signedFloodFillWithValues

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <boost/mpl/at.hpp>
#include <boost/mpl/int.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <memory>
#include <set>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

// MS Visual C++ requires this extra level of indirection in order to compile
// THIS MUST EXIST IN AN UNNAMED NAMESPACE IN ORDER TO COMPILE ON WINDOWS
namespace {

template<typename GridType>
inline typename GridType::ValueType lsutilGridMax()
{
    return std::numeric_limits<typename GridType::ValueType>::max();
}

template<typename GridType>
inline typename GridType::ValueType lsutilGridZero()
{
    return zeroVal<typename GridType::ValueType>();
}

} // unnamed namespace


////////////////////////////////////////


/// @brief Threaded method to convert a sparse level set/SDF into a sparse fog volume
///
/// @details For a level set, the active and negative-valued interior half of the
/// narrow band becomes a linear ramp from 0 to 1; the inactive interior becomes
/// active with a constant value of 1; and the exterior, including the background
/// and the active exterior half of the narrow band, becomes inactive with a constant
/// value of 0.  The interior, though active, remains sparse.
/// @details For a generic SDF, a specified cutoff distance determines the width
/// of the ramp, but otherwise the result is the same as for a level set.
///
/// @param grid            level set/SDF grid to transform
/// @param cutoffDistance  optional world space cutoff distance for the ramp
///                        (automatically clamped if greater than the interior
///                        narrow band width)
template<class GridType>
inline void
sdfToFogVolume(
    GridType& grid,
    typename GridType::ValueType cutoffDistance = lsutilGridMax<GridType>());


/// @brief Threaded method to construct a boolean mask that represents interior regions
///        in a signed distance field.
///
/// @return A shared pointer to either a boolean grid or tree with the same tree
///         configuration and potentially transform as the input @c volume and whose active
///         and @c true values correspond to the interior of the input signed distance field.
///
/// @param volume               Signed distance field / level set volume.
/// @param isovalue             Threshold below which values are considered part of the
///                             interior region.
template<class GridOrTreeType>
inline typename GridOrTreeType::template ValueConverter<bool>::Type::Ptr
sdfInteriorMask(
    const GridOrTreeType& volume,
    typename GridOrTreeType::ValueType isovalue = lsutilGridZero<GridOrTreeType>());


/// @brief  Extracts the interior regions of a signed distance field and topologically enclosed
///         (watertight) regions of value greater than the @a isovalue (cavities) that can arise
///         as the result of CSG union operations between different shapes where at least one of
///         the shapes has a concavity that is capped.
///
///         For example the enclosed region of a capped bottle would include the walls and
///         the interior cavity.
///
/// @return A shared pointer to either a boolean grid or tree with the same tree configuration
///         and potentially transform as the input @c volume and whose active and @c true values
///         correspond to the interior and enclosed regions in the input signed distance field.
///
/// @param volume       Signed distance field / level set volume.
/// @param isovalue     Threshold below which values are considered part of the interior region.
/// @param fillMask     Optional boolean tree, when provided enclosed cavity regions that are not
///                     completely filled by this mask are ignored.
///
///                     For instance if the fill mask does not completely fill the bottle in the
///                     previous example only the walls and cap are returned and the interior
///                     cavity will be ignored.
template<typename GridOrTreeType>
inline typename GridOrTreeType::template ValueConverter<bool>::Type::Ptr
extractEnclosedRegion(
    const GridOrTreeType& volume,
    typename GridOrTreeType::ValueType isovalue = lsutilGridZero<GridOrTreeType>(),
    const typename TreeAdapter<GridOrTreeType>::TreeType::template ValueConverter<bool>::Type*
        fillMask = nullptr);


/// @brief Return a mask of the voxels that intersect the implicit surface with
/// the given @a isovalue.
///
/// @param volume       Signed distance field / level set volume.
/// @param isovalue     The crossing point that is considered the surface.
template<typename GridOrTreeType>
inline typename GridOrTreeType::template ValueConverter<bool>::Type::Ptr
extractIsosurfaceMask(const GridOrTreeType& volume, typename GridOrTreeType::ValueType isovalue);


/// @brief Return a mask for each connected component of the given grid's active voxels.
///
/// @param volume   Input grid or tree
/// @param masks    Output set of disjoint active topology masks sorted in descending order
///                 based on the active voxel count.
template<typename GridOrTreeType>
inline void
extractActiveVoxelSegmentMasks(const GridOrTreeType& volume,
    std::vector<typename GridOrTreeType::template ValueConverter<bool>::Type::Ptr>& masks);


/// @brief  Separates disjoint active topology components into distinct grids or trees.
///
/// @details Supports volumes with active tiles.
///
/// @param volume       Input grid or tree
/// @param segments     Output set of disjoint active topology components sorted in
///                     descending order based on the active voxel count.
template<typename GridOrTreeType>
inline void
segmentActiveVoxels(const GridOrTreeType& volume,
    std::vector<typename GridOrTreeType::Ptr>& segments);


/// @brief  Separates disjoint SDF surfaces into distinct grids or trees.
///
/// @details Supports asymmetric interior / exterior narrowband widths and
///          SDF volumes with dense interior regions.
///
/// @param volume       Input signed distance field / level set volume
/// @param segments     Output set of disjoint SDF surfaces found in @a volume sorted in
///                     descending order based on the surface intersecting voxel count.
template<typename GridOrTreeType>
inline void
segmentSDF(const GridOrTreeType& volume, std::vector<typename GridOrTreeType::Ptr>& segments);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Internal utility objects and implementation details


namespace level_set_util_internal {


template<typename LeafNodeType>
struct MaskInteriorVoxels {

    using ValueType = typename LeafNodeType::ValueType;
    using BoolLeafNodeType = tree::LeafNode<bool, LeafNodeType::LOG2DIM>;

    MaskInteriorVoxels(
        ValueType isovalue, const LeafNodeType ** nodes, BoolLeafNodeType ** maskNodes)
        : mNodes(nodes), mMaskNodes(maskNodes), mIsovalue(isovalue)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        BoolLeafNodeType * maskNodePt = nullptr;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            mMaskNodes[n] = nullptr;
            const LeafNodeType& node = *mNodes[n];

            if (!maskNodePt) {
                maskNodePt = new BoolLeafNodeType(node.origin(), false);
            } else {
                maskNodePt->setOrigin(node.origin());
            }

            const ValueType* values = &node.getValue(0);
            for (Index i = 0; i < LeafNodeType::SIZE; ++i) {
                if (values[i] < mIsovalue) maskNodePt->setValueOn(i, true);
            }

            if (maskNodePt->onVoxelCount() > 0) {
                mMaskNodes[n] = maskNodePt;
                maskNodePt = nullptr;
            }
        }

        if (maskNodePt) delete maskNodePt;
    }

    LeafNodeType        const * const * const mNodes;
    BoolLeafNodeType                 ** const mMaskNodes;
    ValueType                           const mIsovalue;
}; // MaskInteriorVoxels


template<typename TreeType, typename InternalNodeType>
struct MaskInteriorTiles {

    using ValueType = typename TreeType::ValueType;

    MaskInteriorTiles(ValueType isovalue, const TreeType& tree, InternalNodeType ** maskNodes)
        : mTree(&tree), mMaskNodes(maskNodes), mIsovalue(isovalue) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        tree::ValueAccessor<const TreeType> acc(*mTree);
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            typename InternalNodeType::ValueAllIter it = mMaskNodes[n]->beginValueAll();
            for (; it; ++it) {
                if (acc.getValue(it.getCoord()) < mIsovalue) {
                    it.setValue(true);
                    it.setValueOn(true);
                }
            }
        }
    }

    TreeType            const * const mTree;
    InternalNodeType         ** const mMaskNodes;
    ValueType                   const mIsovalue;
}; // MaskInteriorTiles


template<typename TreeType>
struct PopulateTree {

    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;

    PopulateTree(TreeType& tree, LeafNodeType** leafnodes,
        const size_t * nodexIndexMap, ValueType background)
        : mNewTree(background)
        , mTreePt(&tree)
        , mNodes(leafnodes)
        , mNodeIndexMap(nodexIndexMap)
    {
    }

    PopulateTree(PopulateTree& rhs, tbb::split)
        : mNewTree(rhs.mNewTree.background())
        , mTreePt(&mNewTree)
        , mNodes(rhs.mNodes)
        , mNodeIndexMap(rhs.mNodeIndexMap)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) {

        tree::ValueAccessor<TreeType> acc(*mTreePt);

        if (mNodeIndexMap) {
            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
                for (size_t i = mNodeIndexMap[n], I = mNodeIndexMap[n + 1]; i < I; ++i) {
                    if (mNodes[i] != nullptr) acc.addLeaf(mNodes[i]);
                }
            }
        } else {
            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
                acc.addLeaf(mNodes[n]);
            }
        }
    }

    void join(PopulateTree& rhs) { mTreePt->merge(*rhs.mTreePt); }

private:
    TreeType                      mNewTree;
    TreeType              * const mTreePt;
    LeafNodeType         ** const mNodes;
    size_t          const * const mNodeIndexMap;
}; // PopulateTree


/// @brief Negative active values are set @c 0, everything else is set to @c 1.
template<typename LeafNodeType>
struct LabelBoundaryVoxels {

    using ValueType = typename LeafNodeType::ValueType;
    using CharLeafNodeType = tree::LeafNode<char, LeafNodeType::LOG2DIM>;

    LabelBoundaryVoxels(
        ValueType isovalue, const LeafNodeType ** nodes, CharLeafNodeType ** maskNodes)
        : mNodes(nodes), mMaskNodes(maskNodes), mIsovalue(isovalue)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        CharLeafNodeType * maskNodePt = nullptr;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            mMaskNodes[n] = nullptr;
            const LeafNodeType& node = *mNodes[n];

            if (!maskNodePt) {
                maskNodePt = new CharLeafNodeType(node.origin(), 1);
            } else {
                maskNodePt->setOrigin(node.origin());
            }

            typename LeafNodeType::ValueOnCIter it;
            for (it = node.cbeginValueOn(); it; ++it) {
                maskNodePt->setValueOn(it.pos(), ((*it - mIsovalue) < 0.0) ? 0 : 1);
            }

            if (maskNodePt->onVoxelCount() > 0) {
                mMaskNodes[n] = maskNodePt;
                maskNodePt = nullptr;
            }
        }

        if (maskNodePt) delete maskNodePt;
    }

    LeafNodeType        const * const * const mNodes;
    CharLeafNodeType                 ** const mMaskNodes;
    ValueType                           const mIsovalue;
}; // LabelBoundaryVoxels


template<typename LeafNodeType>
struct FlipRegionSign {
    using ValueType = typename LeafNodeType::ValueType;

    FlipRegionSign(LeafNodeType ** nodes) : mNodes(nodes) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            ValueType* values = const_cast<ValueType*>(&mNodes[n]->getValue(0));
            for (Index i = 0; i < LeafNodeType::SIZE; ++i) {
                values[i] = values[i] < 0 ? 1 : -1;
            }
        }
    }

    LeafNodeType ** const mNodes;
}; // FlipRegionSign


template<typename LeafNodeType>
struct FindMinVoxelValue {

    using ValueType = typename LeafNodeType::ValueType;

    FindMinVoxelValue(LeafNodeType const * const * const leafnodes)
        : minValue(std::numeric_limits<ValueType>::max())
        , mNodes(leafnodes)
    {
    }

    FindMinVoxelValue(FindMinVoxelValue& rhs, tbb::split)
        : minValue(std::numeric_limits<ValueType>::max())
        , mNodes(rhs.mNodes)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            const ValueType* data = mNodes[n]->buffer().data();
            for (Index i = 0; i < LeafNodeType::SIZE; ++i) {
                minValue = std::min(minValue, data[i]);
            }
        }
    }

    void join(FindMinVoxelValue& rhs) { minValue = std::min(minValue, rhs.minValue); }

    ValueType minValue;

    LeafNodeType const * const * const mNodes;
}; // FindMinVoxelValue


template<typename InternalNodeType>
struct FindMinTileValue {

    using ValueType = typename InternalNodeType::ValueType;

    FindMinTileValue(InternalNodeType const * const * const nodes)
        : minValue(std::numeric_limits<ValueType>::max())
        , mNodes(nodes)
    {
    }

    FindMinTileValue(FindMinTileValue& rhs, tbb::split)
        : minValue(std::numeric_limits<ValueType>::max())
        , mNodes(rhs.mNodes)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            typename InternalNodeType::ValueAllCIter it = mNodes[n]->beginValueAll();
            for (; it; ++it) {
                minValue = std::min(minValue, *it);
            }
        }
    }

    void join(FindMinTileValue& rhs) { minValue = std::min(minValue, rhs.minValue); }

    ValueType minValue;

    InternalNodeType const * const * const mNodes;
}; // FindMinTileValue


template<typename LeafNodeType>
struct SDFVoxelsToFogVolume {

    using ValueType = typename LeafNodeType::ValueType;

    SDFVoxelsToFogVolume(LeafNodeType ** nodes, ValueType cutoffDistance)
        : mNodes(nodes), mWeight(ValueType(1.0) / cutoffDistance)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            LeafNodeType& node = *mNodes[n];
            node.setValuesOff();

            ValueType* values = node.buffer().data();
            for (Index i = 0; i < LeafNodeType::SIZE; ++i) {
                values[i] = values[i] > ValueType(0.0) ? ValueType(0.0) : values[i] * mWeight;
                if (values[i] > ValueType(0.0)) node.setValueOn(i);
            }

            if (node.onVoxelCount() == 0) {
                delete mNodes[n];
                mNodes[n] = nullptr;
            }
        }
    }

    LeafNodeType    ** const mNodes;
    ValueType          const mWeight;
}; // SDFVoxelsToFogVolume


template<typename TreeType, typename InternalNodeType>
struct SDFTilesToFogVolume {

    SDFTilesToFogVolume(const TreeType& tree, InternalNodeType ** nodes)
        : mTree(&tree), mNodes(nodes) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        using ValueType = typename TreeType::ValueType;
        tree::ValueAccessor<const TreeType> acc(*mTree);

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            typename InternalNodeType::ValueAllIter it = mNodes[n]->beginValueAll();
            for (; it; ++it) {
                if (acc.getValue(it.getCoord()) < ValueType(0.0)) {
                    it.setValue(ValueType(1.0));
                    it.setValueOn(true);
                }
            }
        }
    }

    TreeType            const * const mTree;
    InternalNodeType         ** const mNodes;
}; // SDFTilesToFogVolume


template<typename TreeType>
struct FillMaskBoundary {

    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    FillMaskBoundary(const TreeType& tree, ValueType isovalue, const BoolTreeType& fillMask,
        const BoolLeafNodeType ** fillNodes, BoolLeafNodeType ** newNodes)
        : mTree(&tree)
        , mFillMask(&fillMask)
        , mFillNodes(fillNodes)
        , mNewNodes(newNodes)
        , mIsovalue(isovalue)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        tree::ValueAccessor<const BoolTreeType> maskAcc(*mFillMask);
        tree::ValueAccessor<const TreeType> distAcc(*mTree);

        std::unique_ptr<char[]> valueMask(new char[BoolLeafNodeType::SIZE]);

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            mNewNodes[n] = nullptr;
            const BoolLeafNodeType& node = *mFillNodes[n];
            const Coord& origin = node.origin();

            const bool denseNode = node.isDense();

            // possible early out if the fill mask is dense
            if (denseNode) {

                int denseNeighbors = 0;

                const BoolLeafNodeType* neighborNode =
                    maskAcc.probeConstLeaf(origin.offsetBy(-1, 0, 0));
                if (neighborNode && neighborNode->isDense()) ++denseNeighbors;

                neighborNode = maskAcc.probeConstLeaf(origin.offsetBy(BoolLeafNodeType::DIM, 0, 0));
                if (neighborNode && neighborNode->isDense()) ++denseNeighbors;

                neighborNode = maskAcc.probeConstLeaf(origin.offsetBy(0, -1, 0));
                if (neighborNode && neighborNode->isDense()) ++denseNeighbors;

                neighborNode = maskAcc.probeConstLeaf(origin.offsetBy(0, BoolLeafNodeType::DIM, 0));
                if (neighborNode && neighborNode->isDense()) ++denseNeighbors;

                neighborNode = maskAcc.probeConstLeaf(origin.offsetBy(0, 0, -1));
                if (neighborNode && neighborNode->isDense()) ++denseNeighbors;

                neighborNode = maskAcc.probeConstLeaf(origin.offsetBy(0, 0, BoolLeafNodeType::DIM));
                if (neighborNode && neighborNode->isDense()) ++denseNeighbors;

                if (denseNeighbors == 6) continue;
            }

            // rest value mask
            memset(valueMask.get(), 0, sizeof(char) * BoolLeafNodeType::SIZE);

            const typename TreeType::LeafNodeType* distNode = distAcc.probeConstLeaf(origin);

            // check internal voxel neighbors

            bool earlyTermination = false;

            if (!denseNode) {
                if (distNode) {
                    evalInternalNeighborsP(valueMask.get(), node, *distNode);
                    evalInternalNeighborsN(valueMask.get(), node, *distNode);
                } else if (distAcc.getValue(origin) > mIsovalue) {
                    earlyTermination = evalInternalNeighborsP(valueMask.get(), node);
                    if (!earlyTermination) {
                        earlyTermination = evalInternalNeighborsN(valueMask.get(), node);
                    }
                }
            }

            // check external voxel neighbors

            if (!earlyTermination) {
                evalExternalNeighborsX<true>(valueMask.get(), node, maskAcc, distAcc);
                evalExternalNeighborsX<false>(valueMask.get(), node, maskAcc, distAcc);
                evalExternalNeighborsY<true>(valueMask.get(), node, maskAcc, distAcc);
                evalExternalNeighborsY<false>(valueMask.get(), node, maskAcc, distAcc);
                evalExternalNeighborsZ<true>(valueMask.get(), node, maskAcc, distAcc);
                evalExternalNeighborsZ<false>(valueMask.get(), node, maskAcc, distAcc);
            }

            // Export marked boundary voxels.

            int numBoundaryValues = 0;
            for (Index i = 0, I = BoolLeafNodeType::SIZE; i < I; ++i) {
                numBoundaryValues += valueMask[i] == 1;
            }

            if (numBoundaryValues > 0) {
                mNewNodes[n] = new BoolLeafNodeType(origin, false);
                for (Index i = 0, I = BoolLeafNodeType::SIZE; i < I; ++i) {
                    if (valueMask[i] == 1) mNewNodes[n]->setValueOn(i);
                }
            }
        }
    }

private:
    // Check internal voxel neighbors in positive {x, y, z} directions.
    void evalInternalNeighborsP(char* valueMask, const BoolLeafNodeType& node,
        const LeafNodeType& distNode) const
    {
        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM - 1; ++z) {
                    const Index pos = yPos + z;

                    if (valueMask[pos] != 0 || !node.isValueOn(pos)) continue;

                    if (!node.isValueOn(pos + 1) && distNode.getValue(pos + 1)  > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM - 1; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (valueMask[pos] != 0 || !node.isValueOn(pos)) continue;

                    if (!node.isValueOn(pos + BoolLeafNodeType::DIM) &&
                        distNode.getValue(pos + BoolLeafNodeType::DIM)  > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }

        for (Index x = 0; x < BoolLeafNodeType::DIM - 1; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (valueMask[pos] != 0 || !node.isValueOn(pos)) continue;

                    if (!node.isValueOn(pos + BoolLeafNodeType::DIM * BoolLeafNodeType::DIM) &&
                        (distNode.getValue(pos + BoolLeafNodeType::DIM * BoolLeafNodeType::DIM)
                             > mIsovalue))
                    {
                        valueMask[pos] = 1;
                    }
                }
            }
        }
    }

    bool evalInternalNeighborsP(char* valueMask, const BoolLeafNodeType& node) const {

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM - 1; ++z) {
                    const Index pos = yPos + z;

                    if (node.isValueOn(pos) && !node.isValueOn(pos + 1)) {
                        valueMask[pos] = 1;
                        return true;
                    }
                }
            }
        }

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM - 1; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (node.isValueOn(pos) && !node.isValueOn(pos + BoolLeafNodeType::DIM)) {
                        valueMask[pos] = 1;
                        return true;
                    }
                }
            }
        }

        for (Index x = 0; x < BoolLeafNodeType::DIM - 1; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (node.isValueOn(pos) &&
                        !node.isValueOn(pos + BoolLeafNodeType::DIM * BoolLeafNodeType::DIM)) {
                        valueMask[pos] = 1;
                        return true;
                    }
                }
            }
        }

        return false;
    }

    // Check internal voxel neighbors in negative {x, y, z} directions.

    void evalInternalNeighborsN(char* valueMask, const BoolLeafNodeType& node,
        const LeafNodeType& distNode) const
    {
        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 1; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (valueMask[pos] != 0 || !node.isValueOn(pos)) continue;

                    if (!node.isValueOn(pos - 1) && distNode.getValue(pos - 1)  > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 1; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (valueMask[pos] != 0 || !node.isValueOn(pos)) continue;

                    if (!node.isValueOn(pos - BoolLeafNodeType::DIM) &&
                        distNode.getValue(pos - BoolLeafNodeType::DIM)  > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }

        for (Index x = 1; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (valueMask[pos] != 0 || !node.isValueOn(pos)) continue;

                    if (!node.isValueOn(pos - BoolLeafNodeType::DIM * BoolLeafNodeType::DIM) &&
                        (distNode.getValue(pos - BoolLeafNodeType::DIM * BoolLeafNodeType::DIM)
                             > mIsovalue))
                    {
                        valueMask[pos] = 1;
                    }
                }
            }
        }
    }


    bool evalInternalNeighborsN(char* valueMask, const BoolLeafNodeType& node) const {

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 1; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (node.isValueOn(pos) && !node.isValueOn(pos - 1)) {
                        valueMask[pos] = 1;
                        return true;
                    }
                }
            }
        }

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 1; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (node.isValueOn(pos) && !node.isValueOn(pos - BoolLeafNodeType::DIM)) {
                        valueMask[pos] = 1;
                        return true;
                    }
                }
            }
        }

        for (Index x = 1; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (node.isValueOn(pos) &&
                        !node.isValueOn(pos - BoolLeafNodeType::DIM * BoolLeafNodeType::DIM)) {
                        valueMask[pos] = 1;
                        return true;
                    }
                }
            }
        }

        return false;
    }


    // Check external voxel neighbors

    // If UpWind is true check the X+ oriented node face, else the X- oriented face.
    template<bool UpWind>
    void evalExternalNeighborsX(char* valueMask, const BoolLeafNodeType& node,
        const tree::ValueAccessor<const BoolTreeType>& maskAcc,
        const tree::ValueAccessor<const TreeType>& distAcc) const {

        const Coord& origin = node.origin();
        Coord ijk(0, 0, 0), nijk;
        int step = -1;

        if (UpWind) {
            step = 1;
            ijk[0] = int(BoolLeafNodeType::DIM) - 1;
        }

        const Index xPos = ijk[0] << (2 * int(BoolLeafNodeType::LOG2DIM));

        for (ijk[1] = 0; ijk[1] < int(BoolLeafNodeType::DIM); ++ijk[1]) {
            const Index yPos = xPos + (ijk[1] << int(BoolLeafNodeType::LOG2DIM));

            for (ijk[2] = 0; ijk[2] < int(BoolLeafNodeType::DIM); ++ijk[2]) {
                const Index pos = yPos + ijk[2];

                if (valueMask[pos] == 0 && node.isValueOn(pos)) {

                    nijk = origin + ijk.offsetBy(step, 0, 0);

                    if (!maskAcc.isValueOn(nijk) && distAcc.getValue(nijk) > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }
    }

    // If UpWind is true check the Y+ oriented node face, else the Y- oriented face.
    template<bool UpWind>
    void evalExternalNeighborsY(char* valueMask, const BoolLeafNodeType& node,
        const tree::ValueAccessor<const BoolTreeType>& maskAcc,
        const tree::ValueAccessor<const TreeType>& distAcc) const {

        const Coord& origin = node.origin();
        Coord ijk(0, 0, 0), nijk;
        int step = -1;

        if (UpWind) {
            step = 1;
            ijk[1] = int(BoolLeafNodeType::DIM) - 1;
        }

        const Index yPos = ijk[1] << int(BoolLeafNodeType::LOG2DIM);

        for (ijk[0] = 0;  ijk[0] < int(BoolLeafNodeType::DIM); ++ijk[0]) {
            const Index xPos = yPos + (ijk[0] << (2 * int(BoolLeafNodeType::LOG2DIM)));

            for (ijk[2] = 0; ijk[2] < int(BoolLeafNodeType::DIM); ++ijk[2]) {
                const Index pos = xPos + ijk[2];

                if (valueMask[pos] == 0 && node.isValueOn(pos)) {

                    nijk = origin + ijk.offsetBy(0, step, 0);
                    if (!maskAcc.isValueOn(nijk) && distAcc.getValue(nijk) > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }
    }

    // If UpWind is true check the Z+ oriented node face, else the Z- oriented face.
    template<bool UpWind>
    void evalExternalNeighborsZ(char* valueMask, const BoolLeafNodeType& node,
        const tree::ValueAccessor<const BoolTreeType>& maskAcc,
        const tree::ValueAccessor<const TreeType>& distAcc) const {

        const Coord& origin = node.origin();
        Coord ijk(0, 0, 0), nijk;
        int step = -1;

        if (UpWind) {
            step = 1;
            ijk[2] = int(BoolLeafNodeType::DIM) - 1;
        }

        for (ijk[0] = 0;  ijk[0] < int(BoolLeafNodeType::DIM); ++ijk[0]) {
            const Index xPos = ijk[0] << (2 * int(BoolLeafNodeType::LOG2DIM));

            for (ijk[1] = 0; ijk[1] < int(BoolLeafNodeType::DIM); ++ijk[1]) {
                const Index pos = ijk[2] + xPos + (ijk[1] << int(BoolLeafNodeType::LOG2DIM));

                if (valueMask[pos] == 0 && node.isValueOn(pos)) {

                    nijk = origin + ijk.offsetBy(0, 0, step);
                    if (!maskAcc.isValueOn(nijk) && distAcc.getValue(nijk) > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }
    }

    //////////

    TreeType                    const * const mTree;
    BoolTreeType                const * const mFillMask;
    BoolLeafNodeType    const * const * const mFillNodes;
    BoolLeafNodeType                 ** const mNewNodes;
    ValueType                           const mIsovalue;
}; // FillMaskBoundary


/// @brief Constructs a memory light char tree that represents the exterior region with @c +1
///        and the interior regions with @c -1.
template <class TreeType>
inline typename TreeType::template ValueConverter<char>::Type::Ptr
computeEnclosedRegionMask(const TreeType& tree, typename TreeType::ValueType isovalue,
    const typename TreeType::template ValueConverter<bool>::Type* fillMask)
{
    using LeafNodeType = typename TreeType::LeafNodeType;
    using RootNodeType = typename TreeType::RootNodeType;
    using NodeChainType = typename RootNodeType::NodeChainType;
    using InternalNodeType = typename boost::mpl::at<NodeChainType, boost::mpl::int_<1>>::type;

    using CharTreeType = typename TreeType::template ValueConverter<char>::Type;
    using CharLeafNodeType = typename CharTreeType::LeafNodeType;

    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    const TreeType* treePt = &tree;

    size_t numLeafNodes = 0, numInternalNodes = 0;

    std::vector<const LeafNodeType*> nodes;
    std::vector<size_t> leafnodeCount;

    {
        // compute the prefix sum of the leafnode count in each internal node.
        std::vector<const InternalNodeType*> internalNodes;
        treePt->getNodes(internalNodes);

        numInternalNodes = internalNodes.size();

        leafnodeCount.push_back(0);
        for (size_t n = 0; n < numInternalNodes; ++n) {
            leafnodeCount.push_back(leafnodeCount.back() + internalNodes[n]->leafCount());
        }

        numLeafNodes = leafnodeCount.back();

        // extract all leafnodes
        nodes.reserve(numLeafNodes);

        for (size_t n = 0; n < numInternalNodes; ++n) {
            internalNodes[n]->getNodes(nodes);
        }
    }

    // create mask leafnodes
    std::unique_ptr<CharLeafNodeType*[]> maskNodes(new CharLeafNodeType*[numLeafNodes]);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, numLeafNodes),
        LabelBoundaryVoxels<LeafNodeType>(isovalue, &nodes[0], maskNodes.get()));

    // create mask grid
    typename CharTreeType::Ptr maskTree(new CharTreeType(1));

    PopulateTree<CharTreeType> populate(*maskTree, maskNodes.get(), &leafnodeCount[0], 1);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, numInternalNodes), populate);

    // optionally evaluate the fill mask

    std::vector<CharLeafNodeType*> extraMaskNodes;

    if (fillMask) {

        std::vector<const BoolLeafNodeType*> fillMaskNodes;
        fillMask->getNodes(fillMaskNodes);

        std::unique_ptr<BoolLeafNodeType*[]> boundaryMaskNodes(
            new BoolLeafNodeType*[fillMaskNodes.size()]);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, fillMaskNodes.size()),
            FillMaskBoundary<TreeType>(tree, isovalue, *fillMask, &fillMaskNodes[0],
                boundaryMaskNodes.get()));

        tree::ValueAccessor<CharTreeType> maskAcc(*maskTree);

        for (size_t n = 0, N = fillMaskNodes.size(); n < N; ++n) {

            if (boundaryMaskNodes[n] == nullptr) continue;

            const BoolLeafNodeType& boundaryNode = *boundaryMaskNodes[n];
            const Coord& origin = boundaryNode.origin();

            CharLeafNodeType* maskNodePt = maskAcc.probeLeaf(origin);

            if (!maskNodePt) {
                maskNodePt = maskAcc.touchLeaf(origin);
                extraMaskNodes.push_back(maskNodePt);
            }

            char* data = maskNodePt->buffer().data();

            typename BoolLeafNodeType::ValueOnCIter it = boundaryNode.cbeginValueOn();
            for (; it; ++it) {
                if (data[it.pos()] != 0) data[it.pos()] = -1;
            }

            delete boundaryMaskNodes[n];
        }
    }

    // eliminate enclosed regions
    tools::traceExteriorBoundaries(*maskTree);

    // flip voxel sign to negative inside and positive outside.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numLeafNodes),
        FlipRegionSign<CharLeafNodeType>(maskNodes.get()));

    if (!extraMaskNodes.empty()) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, extraMaskNodes.size()),
            FlipRegionSign<CharLeafNodeType>(&extraMaskNodes[0]));
    }

    // propagate sign information into tile region
    tools::signedFloodFill(*maskTree);

    return maskTree;
} // computeEnclosedRegionMask()


template <class TreeType>
inline typename TreeType::template ValueConverter<bool>::Type::Ptr
computeInteriorMask(const TreeType& tree, typename TreeType::ValueType iso)
{
    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using RootNodeType = typename TreeType::RootNodeType;
    using NodeChainType = typename RootNodeType::NodeChainType;
    using InternalNodeType = typename boost::mpl::at<NodeChainType, boost::mpl::int_<1> >::type;

    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;
    using BoolRootNodeType = typename BoolTreeType::RootNodeType;
    using BoolNodeChainType = typename BoolRootNodeType::NodeChainType;
    using BoolInternalNodeType =
        typename boost::mpl::at<BoolNodeChainType, boost::mpl::int_<1>>::type;

    /////

    // Clamp the isovalue to the level set's background value minus epsilon.
    // (In a valid narrow-band level set, all voxels, including background voxels,
    // have values less than or equal to the background value, so an isovalue
    // greater than or equal to the background value would produce a mask with
    // effectively infinite extent.)
    iso = std::min(iso,
        static_cast<ValueType>(tree.background() - math::Tolerance<ValueType>::value()));

    size_t numLeafNodes = 0, numInternalNodes = 0;

    std::vector<const LeafNodeType*> nodes;
    std::vector<size_t> leafnodeCount;

    {
        // compute the prefix sum of the leafnode count in each internal node.
        std::vector<const InternalNodeType*> internalNodes;
        tree.getNodes(internalNodes);

        numInternalNodes = internalNodes.size();

        leafnodeCount.push_back(0);
        for (size_t n = 0; n < numInternalNodes; ++n) {
            leafnodeCount.push_back(leafnodeCount.back() + internalNodes[n]->leafCount());
        }

        numLeafNodes = leafnodeCount.back();

        // extract all leafnodes
        nodes.reserve(numLeafNodes);

        for (size_t n = 0; n < numInternalNodes; ++n) {
            internalNodes[n]->getNodes(nodes);
        }
    }

    // create mask leafnodes
    std::unique_ptr<BoolLeafNodeType*[]> maskNodes(new BoolLeafNodeType*[numLeafNodes]);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, numLeafNodes),
        MaskInteriorVoxels<LeafNodeType>(iso, &nodes[0], maskNodes.get()));


    // create mask grid
    typename BoolTreeType::Ptr maskTree(new BoolTreeType(false));

    PopulateTree<BoolTreeType> populate(*maskTree, maskNodes.get(), &leafnodeCount[0], false);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, numInternalNodes), populate);


    // evaluate tile values
    std::vector<BoolInternalNodeType*> internalMaskNodes;
    maskTree->getNodes(internalMaskNodes);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, internalMaskNodes.size()),
        MaskInteriorTiles<TreeType, BoolInternalNodeType>(iso, tree, &internalMaskNodes[0]));

    tree::ValueAccessor<const TreeType> acc(tree);

    typename BoolTreeType::ValueAllIter it(*maskTree);
    it.setMaxDepth(BoolTreeType::ValueAllIter::LEAF_DEPTH - 2);

    for ( ; it; ++it) {
        if (acc.getValue(it.getCoord()) < iso) {
            it.setValue(true);
            it.setActiveState(true);
        }
    }

    return maskTree;
} // computeInteriorMask()


template<typename InputTreeType>
struct MaskIsovalueCrossingVoxels
{
    using InputValueType = typename InputTreeType::ValueType;
    using InputLeafNodeType = typename InputTreeType::LeafNodeType;
    using BoolTreeType = typename InputTreeType::template ValueConverter<bool>::Type;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    MaskIsovalueCrossingVoxels(
        const InputTreeType& inputTree,
        const std::vector<const InputLeafNodeType*>& inputLeafNodes,
        BoolTreeType& maskTree,
        InputValueType iso)
        : mInputAccessor(inputTree)
        , mInputNodes(!inputLeafNodes.empty() ? &inputLeafNodes.front() : nullptr)
        , mMaskTree(false)
        , mMaskAccessor(maskTree)
        , mIsovalue(iso)
    {
    }

    MaskIsovalueCrossingVoxels(MaskIsovalueCrossingVoxels& rhs, tbb::split)
        : mInputAccessor(rhs.mInputAccessor.tree())
        , mInputNodes(rhs.mInputNodes)
        , mMaskTree(false)
        , mMaskAccessor(mMaskTree)
        , mIsovalue(rhs.mIsovalue)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) {

        const InputValueType iso = mIsovalue;
        Coord ijk(0, 0, 0);

        BoolLeafNodeType* maskNodePt = nullptr;

        for (size_t n = range.begin(); mInputNodes && (n != range.end()); ++n) {

            const InputLeafNodeType& node = *mInputNodes[n];

            if (!maskNodePt) maskNodePt = new BoolLeafNodeType(node.origin(), false);
            else maskNodePt->setOrigin(node.origin());

            bool collectedData = false;

            for (typename InputLeafNodeType::ValueOnCIter it = node.cbeginValueOn(); it; ++it) {

                bool isUnder = *it < iso;

                ijk = it.getCoord();

                ++ijk[2];
                bool signChange = isUnder != (mInputAccessor.getValue(ijk) < iso); // +z edge
                --ijk[2];

                if (!signChange) {
                    --ijk[2];
                    signChange = isUnder != (mInputAccessor.getValue(ijk) < iso); // -z edge
                    ++ijk[2];
                }

                if (!signChange) {
                    ++ijk[1];
                    signChange = isUnder != (mInputAccessor.getValue(ijk) < iso); // +y edge
                    --ijk[1];
                }

                if (!signChange) {
                    --ijk[1];
                    signChange = isUnder != (mInputAccessor.getValue(ijk) < iso); // -y edge
                    ++ijk[1];
                }

                if (!signChange) {
                    ++ijk[0];
                    signChange = isUnder != (mInputAccessor.getValue(ijk) < iso); // +x edge
                    --ijk[0];
                }

                if (!signChange) {
                    --ijk[0];
                    signChange = isUnder != (mInputAccessor.getValue(ijk) < iso); // -x edge
                    ++ijk[0];
                }

                if (signChange) {
                    collectedData = true;
                    maskNodePt->setValueOn(it.pos(), true);
                }
            }

            if (collectedData) {
                mMaskAccessor.addLeaf(maskNodePt);
                maskNodePt = nullptr;
            }
        }

        if (maskNodePt) delete maskNodePt;
    }

    void join(MaskIsovalueCrossingVoxels& rhs) {
        mMaskAccessor.tree().merge(rhs.mMaskAccessor.tree());
    }

private:
    tree::ValueAccessor<const InputTreeType>    mInputAccessor;
    InputLeafNodeType const * const * const     mInputNodes;

    BoolTreeType                                mMaskTree;
    tree::ValueAccessor<BoolTreeType>           mMaskAccessor;

    InputValueType                              mIsovalue;
}; // MaskIsovalueCrossingVoxels


////////////////////////////////////////


template<typename NodeType>
struct NodeMaskSegment
{
    using Ptr = SharedPtr<NodeMaskSegment>;
    using NodeMaskType = typename NodeType::NodeMaskType;

    NodeMaskSegment() : connections(), mask(false), origin(0,0,0), visited(false) {}

    std::vector<NodeMaskSegment*>   connections;
    NodeMaskType                    mask;
    Coord                           origin;
    bool                            visited;
}; // struct NodeMaskSegment


template<typename NodeType>
inline void
nodeMaskSegmentation(const NodeType& node,
    std::vector<typename NodeMaskSegment<NodeType>::Ptr>& segments)
{
    using NodeMaskType = typename NodeType::NodeMaskType;
    using NodeMaskSegmentType = NodeMaskSegment<NodeType>;
    using NodeMaskSegmentTypePtr = typename NodeMaskSegmentType::Ptr;

    NodeMaskType nodeMask(node.getValueMask());
    std::deque<Index> indexList;

    while (!nodeMask.isOff()) {

        NodeMaskSegmentTypePtr segment(new NodeMaskSegmentType());
        segment->origin = node.origin();

        NodeMaskType& mask = segment->mask;

        indexList.push_back(nodeMask.findFirstOn());
        nodeMask.setOff(indexList.back()); // mark as visited
        Coord ijk(0, 0, 0);

        while (!indexList.empty()) {

            const Index pos = indexList.back();
            indexList.pop_back();

            if (mask.isOn(pos)) continue;
            mask.setOn(pos);

            ijk = NodeType::offsetToLocalCoord(pos);

            Index npos = pos - 1;
            if (ijk[2] != 0 && nodeMask.isOn(npos)) {
                nodeMask.setOff(npos);
                indexList.push_back(npos);
            }

            npos = pos + 1;
            if (ijk[2] != (NodeType::DIM - 1) && nodeMask.isOn(npos)) {
                nodeMask.setOff(npos);
                indexList.push_back(npos);
            }

            npos = pos - NodeType::DIM;
            if (ijk[1] != 0 && nodeMask.isOn(npos)) {
                nodeMask.setOff(npos);
                indexList.push_back(npos);
            }

            npos = pos + NodeType::DIM;
            if (ijk[1] != (NodeType::DIM - 1) && nodeMask.isOn(npos)) {
                nodeMask.setOff(npos);
                indexList.push_back(npos);
            }

            npos = pos - NodeType::DIM * NodeType::DIM;
            if (ijk[0] != 0 && nodeMask.isOn(npos)) {
                nodeMask.setOff(npos);
                indexList.push_back(npos);
            }

            npos = pos + NodeType::DIM * NodeType::DIM;
            if (ijk[0] != (NodeType::DIM - 1) && nodeMask.isOn(npos)) {
                nodeMask.setOff(npos);
                indexList.push_back(npos);
            }

        }

        segments.push_back(segment);
    }
}


template<typename NodeType>
struct SegmentNodeMask
{
    using NodeMaskSegmentType = NodeMaskSegment<NodeType>;
    using NodeMaskSegmentTypePtr = typename NodeMaskSegmentType::Ptr;
    using NodeMaskSegmentVector = typename std::vector<NodeMaskSegmentTypePtr>;

    SegmentNodeMask(std::vector<NodeType*>& nodes, NodeMaskSegmentVector* nodeMaskArray)
        : mNodes(!nodes.empty() ? &nodes.front() : nullptr)
        , mNodeMaskArray(nodeMaskArray)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            NodeType& node = *mNodes[n];
            nodeMaskSegmentation(node, mNodeMaskArray[n]);

            // hack origin data to store array offset
            Coord& origin = const_cast<Coord&>(node.origin());
            origin[0] = static_cast<int>(n);
        }
    }

    NodeType                * const * const mNodes;
    NodeMaskSegmentVector           * const mNodeMaskArray;
}; // struct SegmentNodeMask


template<typename TreeType, typename NodeType>
struct ConnectNodeMaskSegments
{
    using NodeMaskType = typename NodeType::NodeMaskType;
    using NodeMaskSegmentType = NodeMaskSegment<NodeType>;
    using NodeMaskSegmentTypePtr = typename NodeMaskSegmentType::Ptr;
    using NodeMaskSegmentVector = typename std::vector<NodeMaskSegmentTypePtr>;

    ConnectNodeMaskSegments(const TreeType& tree, NodeMaskSegmentVector* nodeMaskArray)
        : mTree(&tree)
        , mNodeMaskArray(nodeMaskArray)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        tree::ValueAccessor<const TreeType> acc(*mTree);

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            NodeMaskSegmentVector& segments = mNodeMaskArray[n];
            if (segments.empty()) continue;

            std::vector<std::set<NodeMaskSegmentType*> > connections(segments.size());

            Coord ijk = segments[0]->origin;

            const NodeType* node = acc.template probeConstNode<NodeType>(ijk);
            if (!node) continue;

            // get neighbour nodes

            ijk[2] += NodeType::DIM;
            const NodeType* nodeZUp = acc.template probeConstNode<NodeType>(ijk);
            ijk[2] -= (NodeType::DIM + NodeType::DIM);
            const NodeType* nodeZDown = acc.template probeConstNode<NodeType>(ijk);
            ijk[2] += NodeType::DIM;

            ijk[1] += NodeType::DIM;
            const NodeType* nodeYUp = acc.template probeConstNode<NodeType>(ijk);
            ijk[1] -= (NodeType::DIM + NodeType::DIM);
            const NodeType* nodeYDown = acc.template probeConstNode<NodeType>(ijk);
            ijk[1] += NodeType::DIM;

            ijk[0] += NodeType::DIM;
            const NodeType* nodeXUp = acc.template probeConstNode<NodeType>(ijk);
            ijk[0] -= (NodeType::DIM + NodeType::DIM);
            const NodeType* nodeXDown = acc.template probeConstNode<NodeType>(ijk);
            ijk[0] += NodeType::DIM;

            const Index startPos = node->getValueMask().findFirstOn();
            for (Index pos = startPos; pos < NodeMaskType::SIZE; ++pos) {

                if (!node->isValueOn(pos)) continue;

                ijk = NodeType::offsetToLocalCoord(pos);

#ifdef _MSC_FULL_VER
  #if _MSC_FULL_VER >= 190000000 && _MSC_FULL_VER < 190024210
                // Visual Studio 2015 had a codegen bug that wasn't fixed until Update 3
                volatile Index npos = 0;
  #else
                Index npos = 0;
  #endif
#else
                Index npos = 0;
#endif

                if (ijk[2] == 0) {
                    npos = pos + (NodeType::DIM - 1);
                    if (nodeZDown && nodeZDown->isValueOn(npos)) {
                        NodeMaskSegmentType* nsegment =
                            findNodeMaskSegment(mNodeMaskArray[getNodeOffset(*nodeZDown)], npos);
                        const Index idx = findNodeMaskSegmentIndex(segments, pos);
                        connections[idx].insert(nsegment);
                    }
                } else if (ijk[2] == (NodeType::DIM - 1)) {
                    npos = pos - (NodeType::DIM - 1);
                    if (nodeZUp && nodeZUp->isValueOn(npos)) {
                        NodeMaskSegmentType* nsegment =
                            findNodeMaskSegment(mNodeMaskArray[getNodeOffset(*nodeZUp)], npos);
                        const Index idx = findNodeMaskSegmentIndex(segments, pos);
                        connections[idx].insert(nsegment);
                    }
                }

                if (ijk[1] == 0) {
                    npos = pos + (NodeType::DIM - 1) * NodeType::DIM;
                    if (nodeYDown && nodeYDown->isValueOn(npos)) {
                        NodeMaskSegmentType* nsegment =
                            findNodeMaskSegment(mNodeMaskArray[getNodeOffset(*nodeYDown)], npos);
                        const Index idx = findNodeMaskSegmentIndex(segments, pos);
                        connections[idx].insert(nsegment);
                    }
                } else if (ijk[1] == (NodeType::DIM - 1)) {
                    npos = pos - (NodeType::DIM - 1) * NodeType::DIM;
                    if (nodeYUp && nodeYUp->isValueOn(npos)) {
                        NodeMaskSegmentType* nsegment =
                            findNodeMaskSegment(mNodeMaskArray[getNodeOffset(*nodeYUp)], npos);
                        const Index idx = findNodeMaskSegmentIndex(segments, pos);
                        connections[idx].insert(nsegment);
                    }
                }

                if (ijk[0] == 0) {
                    npos = pos + (NodeType::DIM - 1) * NodeType::DIM * NodeType::DIM;
                    if (nodeXDown && nodeXDown->isValueOn(npos)) {
                        NodeMaskSegmentType* nsegment =
                            findNodeMaskSegment(mNodeMaskArray[getNodeOffset(*nodeXDown)], npos);
                        const Index idx = findNodeMaskSegmentIndex(segments, pos);
                        connections[idx].insert(nsegment);
                    }
                } else if (ijk[0] == (NodeType::DIM - 1)) {
                    npos = pos - (NodeType::DIM - 1) * NodeType::DIM * NodeType::DIM;
                    if (nodeXUp && nodeXUp->isValueOn(npos)) {
                        NodeMaskSegmentType* nsegment =
                            findNodeMaskSegment(mNodeMaskArray[getNodeOffset(*nodeXUp)], npos);
                        const Index idx = findNodeMaskSegmentIndex(segments, pos);
                        connections[idx].insert(nsegment);
                    }
                }
            }

            for (size_t i = 0, I = connections.size(); i < I; ++i) {

                typename std::set<NodeMaskSegmentType*>::iterator
                    it = connections[i].begin(), end =  connections[i].end();

                std::vector<NodeMaskSegmentType*>& segmentConnections = segments[i]->connections;
                segmentConnections.reserve(connections.size());
                for (; it != end; ++it) {
                    segmentConnections.push_back(*it);
                }
            }
        } // end range loop
    }

private:

    static inline size_t getNodeOffset(const NodeType& node) {
        return static_cast<size_t>(node.origin()[0]);
    }

    static inline NodeMaskSegmentType*
    findNodeMaskSegment(NodeMaskSegmentVector& segments, Index pos)
    {
        NodeMaskSegmentType* segment = nullptr;

        for (size_t n = 0, N = segments.size(); n < N; ++n) {
            if (segments[n]->mask.isOn(pos)) {
                segment = segments[n].get();
                break;
            }
        }

        return segment;
    }

    static inline Index
    findNodeMaskSegmentIndex(NodeMaskSegmentVector& segments, Index pos)
    {
        for (Index n = 0, N = Index(segments.size()); n < N; ++n) {
            if (segments[n]->mask.isOn(pos)) return n;
        }
        return Index(-1);
    }

    TreeType                const * const mTree;
    NodeMaskSegmentVector         * const mNodeMaskArray;
}; // struct ConnectNodeMaskSegments


template<typename TreeType>
struct MaskSegmentGroup
{
    using LeafNodeType = typename TreeType::LeafNodeType;
    using TreeTypePtr = typename TreeType::Ptr;
    using NodeMaskSegmentType = NodeMaskSegment<LeafNodeType>;

    MaskSegmentGroup(const std::vector<NodeMaskSegmentType*>& segments)
        : mSegments(!segments.empty() ? &segments.front() : nullptr)
        , mTree(new TreeType(false))
    {
    }

    MaskSegmentGroup(const MaskSegmentGroup& rhs, tbb::split)
        : mSegments(rhs.mSegments)
        , mTree(new TreeType(false))
    {
    }

    TreeTypePtr& mask() { return mTree; }

    void join(MaskSegmentGroup& rhs) { mTree->merge(*rhs.mTree); }

    void operator()(const tbb::blocked_range<size_t>& range) {

        tree::ValueAccessor<TreeType> acc(*mTree);

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            NodeMaskSegmentType& segment = *mSegments[n];
            LeafNodeType* node = acc.touchLeaf(segment.origin);
            node->getValueMask() |= segment.mask;
        }
    }

private:
    NodeMaskSegmentType * const * const mSegments;
    TreeTypePtr                         mTree;
}; // struct MaskSegmentGroup


////////////////////////////////////////


template<typename TreeType>
struct ExpandLeafNodeRegion
{
    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using NodeMaskType = typename LeafNodeType::NodeMaskType;

    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    /////

    ExpandLeafNodeRegion(const TreeType& distTree, BoolTreeType& maskTree,
        std::vector<BoolLeafNodeType*>& maskNodes)
        : mDistTree(&distTree)
        , mMaskTree(&maskTree)
        , mMaskNodes(!maskNodes.empty() ? &maskNodes.front() : nullptr)
        , mNewMaskTree(false)
    {
    }

    ExpandLeafNodeRegion(const ExpandLeafNodeRegion& rhs, tbb::split)
        : mDistTree(rhs.mDistTree)
        , mMaskTree(rhs.mMaskTree)
        , mMaskNodes(rhs.mMaskNodes)
        , mNewMaskTree(false)
    {
    }

    BoolTreeType& newMaskTree() { return mNewMaskTree; }

    void join(ExpandLeafNodeRegion& rhs) { mNewMaskTree.merge(rhs.mNewMaskTree); }

    void operator()(const tbb::blocked_range<size_t>& range) {

        using NodeType = LeafNodeType;

        tree::ValueAccessor<const TreeType>         distAcc(*mDistTree);
        tree::ValueAccessor<const BoolTreeType>     maskAcc(*mMaskTree);
        tree::ValueAccessor<BoolTreeType>           newMaskAcc(mNewMaskTree);

        NodeMaskType maskZUp, maskZDown, maskYUp, maskYDown, maskXUp, maskXDown;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            BoolLeafNodeType& maskNode = *mMaskNodes[n];
            if (maskNode.isEmpty()) continue;

            Coord ijk = maskNode.origin(), nijk;

            const LeafNodeType* distNode = distAcc.probeConstLeaf(ijk);
            if (!distNode) continue;

            const ValueType *dataZUp = nullptr, *dataZDown = nullptr,
                            *dataYUp = nullptr, *dataYDown = nullptr,
                            *dataXUp = nullptr, *dataXDown = nullptr;

            ijk[2] += NodeType::DIM;
            getData(ijk, distAcc, maskAcc, maskZUp, dataZUp);
            ijk[2] -= (NodeType::DIM + NodeType::DIM);
            getData(ijk, distAcc, maskAcc, maskZDown, dataZDown);
            ijk[2] += NodeType::DIM;

            ijk[1] += NodeType::DIM;
            getData(ijk, distAcc, maskAcc, maskYUp, dataYUp);
            ijk[1] -= (NodeType::DIM + NodeType::DIM);
            getData(ijk, distAcc, maskAcc, maskYDown, dataYDown);
            ijk[1] += NodeType::DIM;

            ijk[0] += NodeType::DIM;
            getData(ijk, distAcc, maskAcc, maskXUp, dataXUp);
            ijk[0] -= (NodeType::DIM + NodeType::DIM);
            getData(ijk, distAcc, maskAcc, maskXDown, dataXDown);
            ijk[0] += NodeType::DIM;

            for (typename BoolLeafNodeType::ValueOnIter it = maskNode.beginValueOn(); it; ++it) {

                const Index pos = it.pos();
                const ValueType val = std::abs(distNode->getValue(pos));

                ijk = BoolLeafNodeType::offsetToLocalCoord(pos);
                nijk = ijk + maskNode.origin();

                if (dataZUp && ijk[2] == (BoolLeafNodeType::DIM - 1)) {
                    const Index npos = pos - (NodeType::DIM - 1);
                    if (maskZUp.isOn(npos) && std::abs(dataZUp[npos]) > val) {
                        newMaskAcc.setValueOn(nijk.offsetBy(0, 0, 1));
                    }
                } else if (dataZDown && ijk[2] == 0) {
                    const Index npos = pos + (NodeType::DIM - 1);
                    if (maskZDown.isOn(npos) && std::abs(dataZDown[npos]) > val) {
                        newMaskAcc.setValueOn(nijk.offsetBy(0, 0, -1));
                    }
                }

                if (dataYUp && ijk[1] == (BoolLeafNodeType::DIM - 1)) {
                    const Index npos = pos - (NodeType::DIM - 1) * NodeType::DIM;
                    if (maskYUp.isOn(npos) && std::abs(dataYUp[npos]) > val) {
                        newMaskAcc.setValueOn(nijk.offsetBy(0, 1, 0));
                    }
                } else if (dataYDown && ijk[1] == 0) {
                    const Index npos = pos + (NodeType::DIM - 1) * NodeType::DIM;
                    if (maskYDown.isOn(npos) && std::abs(dataYDown[npos]) > val) {
                        newMaskAcc.setValueOn(nijk.offsetBy(0, -1, 0));
                    }
                }

                if (dataXUp && ijk[0] == (BoolLeafNodeType::DIM - 1)) {
                    const Index npos = pos - (NodeType::DIM - 1) * NodeType::DIM * NodeType::DIM;
                    if (maskXUp.isOn(npos) && std::abs(dataXUp[npos]) > val) {
                        newMaskAcc.setValueOn(nijk.offsetBy(1, 0, 0));
                    }
                } else if (dataXDown && ijk[0] == 0) {
                    const Index npos = pos + (NodeType::DIM - 1) * NodeType::DIM * NodeType::DIM;
                    if (maskXDown.isOn(npos) && std::abs(dataXDown[npos]) > val) {
                        newMaskAcc.setValueOn(nijk.offsetBy(-1, 0, 0));
                    }
                }

            } // end value on loop
        } // end range loop
    }

private:

    static inline void
    getData(const Coord& ijk, tree::ValueAccessor<const TreeType>& distAcc,
        tree::ValueAccessor<const BoolTreeType>& maskAcc, NodeMaskType& mask,
        const ValueType*& data)
    {
        const LeafNodeType* node = distAcc.probeConstLeaf(ijk);
        if (node) {
            data = node->buffer().data();
            mask = node->getValueMask();
            const BoolLeafNodeType* maskNodePt = maskAcc.probeConstLeaf(ijk);
            if (maskNodePt) mask -= maskNodePt->getValueMask();
        }
    }

    TreeType        const * const mDistTree;
    BoolTreeType          * const mMaskTree;
    BoolLeafNodeType     ** const mMaskNodes;

    BoolTreeType mNewMaskTree;
}; // struct ExpandLeafNodeRegion


template<typename TreeType>
struct FillLeafNodeVoxels
{
    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using NodeMaskType = typename LeafNodeType::NodeMaskType;
    using BoolLeafNodeType = tree::LeafNode<bool, LeafNodeType::LOG2DIM>;

    FillLeafNodeVoxels(const TreeType& tree, std::vector<BoolLeafNodeType*>& maskNodes)
        : mTree(&tree), mMaskNodes(!maskNodes.empty() ? &maskNodes.front() : nullptr)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        tree::ValueAccessor<const TreeType> distAcc(*mTree);

        std::vector<Index> indexList;
        indexList.reserve(NodeMaskType::SIZE);

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            BoolLeafNodeType& maskNode = *mMaskNodes[n];

            const LeafNodeType * distNode = distAcc.probeConstLeaf(maskNode.origin());
            if (!distNode) continue;

            NodeMaskType mask(distNode->getValueMask());
            NodeMaskType& narrowbandMask = maskNode.getValueMask();

            for (Index pos = narrowbandMask.findFirstOn(); pos < NodeMaskType::SIZE; ++pos) {
                if (narrowbandMask.isOn(pos)) indexList.push_back(pos);
            }

            mask -= narrowbandMask; // bitwise difference
            narrowbandMask.setOff();

            const ValueType* data = distNode->buffer().data();
            Coord ijk(0, 0, 0);

            while (!indexList.empty()) {

                const Index pos = indexList.back();
                indexList.pop_back();

                if (narrowbandMask.isOn(pos)) continue;
                narrowbandMask.setOn(pos);

                const ValueType dist = std::abs(data[pos]);

                ijk = LeafNodeType::offsetToLocalCoord(pos);

                Index npos = pos - 1;
                if (ijk[2] != 0 && mask.isOn(npos) && std::abs(data[npos]) > dist) {
                    mask.setOff(npos);
                    indexList.push_back(npos);
                }

                npos = pos + 1;
                if ((ijk[2] != (LeafNodeType::DIM - 1)) && mask.isOn(npos)
                    && std::abs(data[npos]) > dist)
                {
                    mask.setOff(npos);
                    indexList.push_back(npos);
                }

                npos = pos - LeafNodeType::DIM;
                if (ijk[1] != 0 && mask.isOn(npos) && std::abs(data[npos]) > dist) {
                    mask.setOff(npos);
                    indexList.push_back(npos);
                }

                npos = pos + LeafNodeType::DIM;
                if ((ijk[1] != (LeafNodeType::DIM - 1)) && mask.isOn(npos)
                    && std::abs(data[npos]) > dist)
                {
                    mask.setOff(npos);
                    indexList.push_back(npos);
                }

                npos = pos - LeafNodeType::DIM * LeafNodeType::DIM;
                if (ijk[0] != 0 && mask.isOn(npos) && std::abs(data[npos]) > dist) {
                    mask.setOff(npos);
                    indexList.push_back(npos);
                }

                npos = pos + LeafNodeType::DIM * LeafNodeType::DIM;
                if ((ijk[0] != (LeafNodeType::DIM - 1)) && mask.isOn(npos)
                    && std::abs(data[npos]) > dist)
                {
                    mask.setOff(npos);
                    indexList.push_back(npos);
                }
            } // end flood fill loop
        } // end range loop
    }

    TreeType            const * const mTree;
    BoolLeafNodeType         ** const mMaskNodes;
}; // FillLeafNodeVoxels


template<typename TreeType>
struct ExpandNarrowbandMask
{
    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;
    using BoolTreeTypePtr = typename BoolTreeType::Ptr;

    ExpandNarrowbandMask(const TreeType& tree, std::vector<BoolTreeTypePtr>& segments)
        : mTree(&tree), mSegments(!segments.empty() ? &segments.front() : nullptr)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        const TreeType& distTree = *mTree;
        std::vector<BoolLeafNodeType*> nodes;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            BoolTreeType& narrowBandMask = *mSegments[n];

            BoolTreeType candidateMask(narrowBandMask, false, TopologyCopy());

            while (true) {

                nodes.clear();
                candidateMask.getNodes(nodes);
                if (nodes.empty()) break;

                const tbb::blocked_range<size_t> nodeRange(0, nodes.size());

                tbb::parallel_for(nodeRange, FillLeafNodeVoxels<TreeType>(distTree, nodes));

                narrowBandMask.topologyUnion(candidateMask);

                ExpandLeafNodeRegion<TreeType> op(distTree, narrowBandMask, nodes);
                tbb::parallel_reduce(nodeRange, op);

                if (op.newMaskTree().empty()) break;

                candidateMask.clear();
                candidateMask.merge(op.newMaskTree());
            } // end expand loop
        } // end range loop
    }

    TreeType            const * const mTree;
    BoolTreeTypePtr           * const mSegments;
}; // ExpandNarrowbandMask


template<typename TreeType>
struct FloodFillSign
{
    using TreeTypePtr = typename TreeType::Ptr;
    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using RootNodeType = typename TreeType::RootNodeType;
    using NodeChainType = typename RootNodeType::NodeChainType;
    using InternalNodeType = typename boost::mpl::at<NodeChainType, boost::mpl::int_<1> >::type;

    FloodFillSign(const TreeType& tree, std::vector<TreeTypePtr>& segments)
        : mTree(&tree)
        , mSegments(!segments.empty() ? &segments.front() : nullptr)
        , mMinValue(ValueType(0.0))
    {
        ValueType minSDFValue = std::numeric_limits<ValueType>::max();

        {
            std::vector<const InternalNodeType*> nodes;
            tree.getNodes(nodes);

            if (!nodes.empty()) {
                FindMinTileValue<InternalNodeType> minOp(&nodes[0]);
                tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nodes.size()), minOp);
                minSDFValue = std::min(minSDFValue, minOp.minValue);
            }
        }

        if (minSDFValue > ValueType(0.0)) {
            std::vector<const LeafNodeType*> nodes;
            tree.getNodes(nodes);
            if (!nodes.empty()) {
                FindMinVoxelValue<LeafNodeType> minOp(&nodes[0]);
                tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nodes.size()), minOp);
                minSDFValue = std::min(minSDFValue, minOp.minValue);
            }
        }

        mMinValue = minSDFValue;
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        const ValueType interiorValue = -std::abs(mMinValue);
        const ValueType exteriorValue = std::abs(mTree->background());
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            tools::signedFloodFillWithValues(*mSegments[n], exteriorValue, interiorValue);
        }
    }

private:

    TreeType    const * const mTree;
    TreeTypePtr       * const mSegments;
    ValueType                 mMinValue;
}; // FloodFillSign


template<typename TreeType>
struct MaskedCopy
{
    using TreeTypePtr = typename TreeType::Ptr;
    using ValueType = typename TreeType::ValueType;
    using LeafNodeType = typename TreeType::LeafNodeType;

    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    using BoolTreeTypePtr = typename BoolTreeType::Ptr;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    MaskedCopy(const TreeType& tree, std::vector<TreeTypePtr>& segments,
        std::vector<BoolTreeTypePtr>& masks)
        : mTree(&tree)
        , mSegments(!segments.empty() ? &segments.front() : nullptr)
        , mMasks(!masks.empty() ? &masks.front() : nullptr)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        std::vector<const BoolLeafNodeType*> nodes;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            const BoolTreeType& mask = *mMasks[n];

            nodes.clear();
            mask.getNodes(nodes);

            Copy op(*mTree, nodes);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nodes.size()), op);
            mSegments[n] = op.outputTree();
        }
    }

private:

    struct Copy {
        Copy(const TreeType& inputTree, std::vector<const BoolLeafNodeType*>& maskNodes)
            : mInputTree(&inputTree)
            , mMaskNodes(!maskNodes.empty() ? &maskNodes.front() : nullptr)
            , mOutputTreePtr(new TreeType(inputTree.background()))
        {
        }

        Copy(const Copy& rhs, tbb::split)
            : mInputTree(rhs.mInputTree)
            , mMaskNodes(rhs.mMaskNodes)
            , mOutputTreePtr(new TreeType(mInputTree->background()))
        {
        }

        TreeTypePtr& outputTree() { return mOutputTreePtr; }

        void join(Copy& rhs) { mOutputTreePtr->merge(*rhs.mOutputTreePtr); }

        void operator()(const tbb::blocked_range<size_t>& range) {

            tree::ValueAccessor<const TreeType> inputAcc(*mInputTree);
            tree::ValueAccessor<TreeType>       outputAcc(*mOutputTreePtr);

            for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

                const BoolLeafNodeType& maskNode = *mMaskNodes[n];
                if (maskNode.isEmpty()) continue;

                const Coord& ijk = maskNode.origin();

                const LeafNodeType* inputNode = inputAcc.probeConstLeaf(ijk);
                if (inputNode) {

                    LeafNodeType* outputNode = outputAcc.touchLeaf(ijk);

                    for (typename BoolLeafNodeType::ValueOnCIter it = maskNode.cbeginValueOn();
                        it; ++it)
                    {
                        const Index idx = it.pos();
                        outputNode->setValueOn(idx, inputNode->getValue(idx));
                    }
                } else {
                    const int valueDepth = inputAcc.getValueDepth(ijk);
                    if (valueDepth >= 0) {
                        outputAcc.addTile(TreeType::RootNodeType::LEVEL - valueDepth,
                            ijk, inputAcc.getValue(ijk), true);
                    }
                }
            }
        }

    private:
        TreeType                 const * const mInputTree;
        BoolLeafNodeType const * const * const mMaskNodes;
        TreeTypePtr                            mOutputTreePtr;
    }; // struct Copy

    TreeType            const * const mTree;
    TreeTypePtr               * const mSegments;
    BoolTreeTypePtr           * const mMasks;
}; // MaskedCopy


////////////////////////////////////////


template<typename VolumePtrType>
struct ComputeActiveVoxelCount
{
    ComputeActiveVoxelCount(std::vector<VolumePtrType>& segments, size_t *countArray)
        : mSegments(!segments.empty() ? &segments.front() : nullptr)
        , mCountArray(countArray)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            mCountArray[n] = mSegments[n]->activeVoxelCount();
        }
    }

    VolumePtrType   * const mSegments;
    size_t          * const mCountArray;
};


struct GreaterCount
{
    GreaterCount(const size_t *countArray) : mCountArray(countArray) {}

    inline bool operator() (const size_t& lhs, const size_t& rhs) const
    {
        return (mCountArray[lhs] > mCountArray[rhs]);
    }

    size_t const * const mCountArray;
};

////////////////////////////////////////


template<typename TreeType>
struct GridOrTreeConstructor
{
    using TreeTypePtr = typename TreeType::Ptr;
    using BoolTreePtrType = typename TreeType::template ValueConverter<bool>::Type::Ptr;

    static BoolTreePtrType constructMask(const TreeType&, BoolTreePtrType& maskTree)
        { return maskTree; }
    static TreeTypePtr construct(const TreeType&, TreeTypePtr& tree) { return tree; }
};


template<typename TreeType>
struct GridOrTreeConstructor<Grid<TreeType> >
{
    using GridType = Grid<TreeType>;
    using GridTypePtr = typename Grid<TreeType>::Ptr;
    using TreeTypePtr = typename TreeType::Ptr;

    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    using BoolTreePtrType = typename BoolTreeType::Ptr;
    using BoolGridType = Grid<BoolTreeType>;
    using BoolGridPtrType = typename BoolGridType::Ptr;

    static BoolGridPtrType constructMask(const GridType& grid, BoolTreePtrType& maskTree) {
        BoolGridPtrType maskGrid(BoolGridType::create(maskTree));
        maskGrid->setTransform(grid.transform().copy());
        return maskGrid;
    }

    static GridTypePtr construct(const GridType& grid, TreeTypePtr& maskTree) {
        GridTypePtr maskGrid(GridType::create(maskTree));
        maskGrid->setTransform(grid.transform().copy());
        maskGrid->insertMeta(grid);
        return maskGrid;
    }
};


} // namespace level_set_util_internal


////////////////////////////////////////


template <class GridType>
inline void
sdfToFogVolume(GridType& grid, typename GridType::ValueType cutoffDistance)
{
    using ValueType = typename GridType::ValueType;
    using TreeType = typename GridType::TreeType;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using RootNodeType = typename TreeType::RootNodeType;
    using NodeChainType = typename RootNodeType::NodeChainType;
    using InternalNodeType = typename boost::mpl::at<NodeChainType, boost::mpl::int_<1>>::type;

    //////////

    TreeType& tree = grid.tree();

    size_t numLeafNodes = 0, numInternalNodes = 0;

    std::vector<LeafNodeType*> nodes;
    std::vector<size_t> leafnodeCount;

    {
        // Compute the prefix sum of the leafnode count in each internal node.
        std::vector<InternalNodeType*> internalNodes;
        tree.getNodes(internalNodes);

        numInternalNodes = internalNodes.size();

        leafnodeCount.push_back(0);
        for (size_t n = 0; n < numInternalNodes; ++n) {
            leafnodeCount.push_back(leafnodeCount.back() + internalNodes[n]->leafCount());
        }

        numLeafNodes = leafnodeCount.back();

        // Steal all leafnodes (Removes them from the tree and transfers ownership.)
        nodes.reserve(numLeafNodes);

        for (size_t n = 0; n < numInternalNodes; ++n) {
            internalNodes[n]->stealNodes(nodes, tree.background(), false);
        }

        // Clamp cutoffDistance to min sdf value
        ValueType minSDFValue = std::numeric_limits<ValueType>::max();

        {
            level_set_util_internal::FindMinTileValue<InternalNodeType> minOp(&internalNodes[0]);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, internalNodes.size()), minOp);
            minSDFValue = std::min(minSDFValue, minOp.minValue);
        }

        if (minSDFValue > ValueType(0.0)) {
            level_set_util_internal::FindMinVoxelValue<LeafNodeType> minOp(&nodes[0]);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nodes.size()), minOp);
            minSDFValue = std::min(minSDFValue, minOp.minValue);
        }

        cutoffDistance = -std::abs(cutoffDistance);
        cutoffDistance = minSDFValue > cutoffDistance ? minSDFValue : cutoffDistance;
    }

    // Transform voxel values and delete leafnodes that are uniformly zero after the transformation.
    // (Positive values are set to zero with inactive state and negative values are remapped
    // from zero to one with active state.)
    tbb::parallel_for(tbb::blocked_range<size_t>(0, nodes.size()),
        level_set_util_internal::SDFVoxelsToFogVolume<LeafNodeType>(&nodes[0], cutoffDistance));

    // Populate a new tree with the remaining leafnodes
    typename TreeType::Ptr newTree(new TreeType(ValueType(0.0)));

    level_set_util_internal::PopulateTree<TreeType> populate(
        *newTree, &nodes[0], &leafnodeCount[0], 0);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, numInternalNodes), populate);

    // Transform tile values (Negative valued tiles are set to 1.0 with active state.)
    std::vector<InternalNodeType*> internalNodes;
    newTree->getNodes(internalNodes);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, internalNodes.size()),
        level_set_util_internal::SDFTilesToFogVolume<TreeType, InternalNodeType>(
            tree, &internalNodes[0]));

    {
        tree::ValueAccessor<const TreeType> acc(tree);

        typename TreeType::ValueAllIter it(*newTree);
        it.setMaxDepth(TreeType::ValueAllIter::LEAF_DEPTH - 2);

        for ( ; it; ++it) {
            if (acc.getValue(it.getCoord()) < ValueType(0.0)) {
                it.setValue(ValueType(1.0));
                it.setActiveState(true);
            }
        }
    }

    // Insert missing root level tiles. (The new tree is constructed from the remaining leafnodes
    // and will therefore not contain any root level tiles that may exist in the original tree.)
    {
        typename TreeType::ValueAllIter it(tree);
        it.setMaxDepth(TreeType::ValueAllIter::ROOT_DEPTH);
        for ( ; it; ++it) {
            if (it.getValue() <  ValueType(0.0)) {
                newTree->addTile(TreeType::ValueAllIter::ROOT_LEVEL, it.getCoord(),
                    ValueType(1.0), true);
            }
        }
    }

    grid.setTree(newTree);
    grid.setGridClass(GRID_FOG_VOLUME);
}


////////////////////////////////////////


template <class GridOrTreeType>
inline typename GridOrTreeType::template ValueConverter<bool>::Type::Ptr
sdfInteriorMask(const GridOrTreeType& volume, typename GridOrTreeType::ValueType isovalue)
{
    using TreeType = typename TreeAdapter<GridOrTreeType>::TreeType;
    const TreeType& tree = TreeAdapter<GridOrTreeType>::tree(volume);

    using BoolTreePtrType = typename TreeType::template ValueConverter<bool>::Type::Ptr;
    BoolTreePtrType mask = level_set_util_internal::computeInteriorMask(tree, isovalue);

    return level_set_util_internal::GridOrTreeConstructor<GridOrTreeType>::constructMask(
        volume, mask);
}


template<typename GridOrTreeType>
inline typename GridOrTreeType::template ValueConverter<bool>::Type::Ptr
extractEnclosedRegion(const GridOrTreeType& volume,
    typename GridOrTreeType::ValueType isovalue,
    const typename TreeAdapter<GridOrTreeType>::TreeType::template ValueConverter<bool>::Type*
        fillMask)
{
    using TreeType = typename TreeAdapter<GridOrTreeType>::TreeType;
    const TreeType& tree = TreeAdapter<GridOrTreeType>::tree(volume);

    using CharTreePtrType = typename TreeType::template ValueConverter<char>::Type::Ptr;
    CharTreePtrType regionMask = level_set_util_internal::computeEnclosedRegionMask(
        tree, isovalue, fillMask);

    using BoolTreePtrType = typename TreeType::template ValueConverter<bool>::Type::Ptr;
    BoolTreePtrType mask = level_set_util_internal::computeInteriorMask(*regionMask, 0);

    return level_set_util_internal::GridOrTreeConstructor<GridOrTreeType>::constructMask(
        volume, mask);
}


////////////////////////////////////////


template<typename GridOrTreeType>
inline typename GridOrTreeType::template ValueConverter<bool>::Type::Ptr
extractIsosurfaceMask(const GridOrTreeType& volume, typename GridOrTreeType::ValueType isovalue)
{
    using TreeType = typename TreeAdapter<GridOrTreeType>::TreeType;
    const TreeType& tree = TreeAdapter<GridOrTreeType>::tree(volume);

    std::vector<const typename TreeType::LeafNodeType*> nodes;
    tree.getNodes(nodes);

    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    typename BoolTreeType::Ptr mask(new BoolTreeType(false));

    level_set_util_internal::MaskIsovalueCrossingVoxels<TreeType> op(tree, nodes, *mask, isovalue);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nodes.size()), op);

    return level_set_util_internal::GridOrTreeConstructor<GridOrTreeType>::constructMask(
        volume, mask);
}


////////////////////////////////////////


template<typename GridOrTreeType>
inline void
extractActiveVoxelSegmentMasks(const GridOrTreeType& volume,
    std::vector<typename GridOrTreeType::template ValueConverter<bool>::Type::Ptr>& masks)
{
    using TreeType = typename TreeAdapter<GridOrTreeType>::TreeType;
    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    using BoolTreePtrType = typename BoolTreeType::Ptr;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    using NodeMaskSegmentType = level_set_util_internal::NodeMaskSegment<BoolLeafNodeType>;
    using NodeMaskSegmentPtrType = typename NodeMaskSegmentType::Ptr;
    using NodeMaskSegmentPtrVector = typename std::vector<NodeMaskSegmentPtrType>;
    using NodeMaskSegmentRawPtrVector = typename std::vector<NodeMaskSegmentType*>;

    /////

    const TreeType& tree = TreeAdapter<GridOrTreeType>::tree(volume);

    BoolTreeType topologyMask(tree, false, TopologyCopy());

    if (topologyMask.hasActiveTiles()) {
        topologyMask.voxelizeActiveTiles();
    }

    std::vector<BoolLeafNodeType*> leafnodes;
    topologyMask.getNodes(leafnodes);

    if (leafnodes.empty()) return;

    // 1. Split node masks into disjoint segments
    // Note: The LeafNode origin coord is modified to record the 'leafnodes' array offset.

    std::unique_ptr<NodeMaskSegmentPtrVector[]> nodeSegmentArray(
        new NodeMaskSegmentPtrVector[leafnodes.size()]);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, leafnodes.size()),
        level_set_util_internal::SegmentNodeMask<BoolLeafNodeType>(
            leafnodes, nodeSegmentArray.get()));


    // 2. Compute segment connectivity

    tbb::parallel_for(tbb::blocked_range<size_t>(0, leafnodes.size()),
        level_set_util_internal::ConnectNodeMaskSegments<BoolTreeType, BoolLeafNodeType>(
            topologyMask, nodeSegmentArray.get()));

    topologyMask.clear();

    size_t nodeSegmentCount = 0;
    for (size_t n = 0, N = leafnodes.size(); n < N; ++n) {
        nodeSegmentCount += nodeSegmentArray[n].size();
    }

    // 3. Group connected segments

    std::deque<NodeMaskSegmentRawPtrVector> nodeSegmentGroups;

    NodeMaskSegmentType* nextSegment = nodeSegmentArray[0][0].get();
    while (nextSegment) {

        nodeSegmentGroups.push_back(NodeMaskSegmentRawPtrVector());

        std::vector<NodeMaskSegmentType*>& segmentGroup = nodeSegmentGroups.back();
        segmentGroup.reserve(nodeSegmentCount);

        std::deque<NodeMaskSegmentType*> segmentQueue;
        segmentQueue.push_back(nextSegment);
        nextSegment = nullptr;

        while (!segmentQueue.empty()) {

            NodeMaskSegmentType* segment = segmentQueue.back();
            segmentQueue.pop_back();

            if (segment->visited) continue;
            segment->visited = true;

            segmentGroup.push_back(segment);

            // queue connected segments
            std::vector<NodeMaskSegmentType*>& connections = segment->connections;
            for (size_t n = 0, N = connections.size(); n < N; ++n) {
                if (!connections[n]->visited) segmentQueue.push_back(connections[n]);
            }
        }

        // find first unvisited segment
        for (size_t n = 0, N = leafnodes.size(); n < N; ++n) {
            NodeMaskSegmentPtrVector& nodeSegments = nodeSegmentArray[n];
            for (size_t i = 0, I = nodeSegments.size(); i < I; ++i) {
                if (!nodeSegments[i]->visited) nextSegment = nodeSegments[i].get();
            }
        }
    }

    // 4. Mask segment groups

    if (nodeSegmentGroups.size() == 1) {

        BoolTreePtrType mask(new BoolTreeType(tree, false, TopologyCopy()));

        if (mask->hasActiveTiles()) {
            mask->voxelizeActiveTiles();
        }

        masks.push_back(
            level_set_util_internal::GridOrTreeConstructor<GridOrTreeType>::constructMask(
                volume, mask));

    } else if (nodeSegmentGroups.size() > 1) {

        for (size_t n = 0, N = nodeSegmentGroups.size(); n < N; ++n) {

            NodeMaskSegmentRawPtrVector& segmentGroup = nodeSegmentGroups[n];

            level_set_util_internal::MaskSegmentGroup<BoolTreeType> op(segmentGroup);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, segmentGroup.size()), op);

            masks.push_back(
                level_set_util_internal::GridOrTreeConstructor<GridOrTreeType>::constructMask(
                    volume, op.mask()));
        }
    }

    // 5. Sort segments in descending order based on the active voxel count.

    if (masks.size() > 1) {
        const size_t segmentCount = masks.size();

        std::unique_ptr<size_t[]> segmentOrderArray(new size_t[segmentCount]);
        std::unique_ptr<size_t[]> voxelCountArray(new size_t[segmentCount]);

        for (size_t n = 0; n < segmentCount; ++n) {
            segmentOrderArray[n] = n;
        }

        tbb::parallel_for(tbb::blocked_range<size_t>(0, segmentCount),
            level_set_util_internal::ComputeActiveVoxelCount<BoolTreePtrType>(
                masks, voxelCountArray.get()));

        size_t *begin = segmentOrderArray.get();
        tbb::parallel_sort(begin, begin + masks.size(), level_set_util_internal::GreaterCount(
            voxelCountArray.get()));

        std::vector<BoolTreePtrType> orderedMasks;
        orderedMasks.reserve(masks.size());

        for (size_t n = 0; n < segmentCount; ++n) {
            orderedMasks.push_back(masks[segmentOrderArray[n]]);
        }

        masks.swap(orderedMasks);
    }

} // extractActiveVoxelSegmentMasks()


template<typename GridOrTreeType>
inline void
segmentActiveVoxels(const GridOrTreeType& volume,
    std::vector<typename GridOrTreeType::Ptr>& segments)
{
    using TreeType = typename TreeAdapter<GridOrTreeType>::TreeType;
    using TreePtrType = typename TreeType::Ptr;
    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    using BoolTreePtrType = typename BoolTreeType::Ptr;

    const TreeType& inputTree = TreeAdapter<GridOrTreeType>::tree(volume);

    // 1. Segment active topology mask
    std::vector<BoolTreePtrType> maskSegmentArray;
    extractActiveVoxelSegmentMasks(inputTree, maskSegmentArray);

    const size_t numSegments = maskSegmentArray.size();

    if (numSegments < 2) {
        // single segment early-out
        TreePtrType segment(new TreeType(inputTree));
        segments.push_back(
            level_set_util_internal::GridOrTreeConstructor<GridOrTreeType>::construct(
                volume, segment));
        return;
    }

    const tbb::blocked_range<size_t> segmentRange(0, numSegments);

    // 2. Export segments
    std::vector<TreePtrType> outputSegmentArray(numSegments);

    tbb::parallel_for(segmentRange,
        level_set_util_internal::MaskedCopy<TreeType>(inputTree, outputSegmentArray,
            maskSegmentArray));

    for (size_t n = 0, N = numSegments; n < N; ++n) {
        segments.push_back(
            level_set_util_internal::GridOrTreeConstructor<GridOrTreeType>::construct(
                volume, outputSegmentArray[n]));
    }
}


template<typename GridOrTreeType>
inline void
segmentSDF(const GridOrTreeType& volume, std::vector<typename GridOrTreeType::Ptr>& segments)
{
    using TreeType = typename TreeAdapter<GridOrTreeType>::TreeType;
    using TreePtrType = typename TreeType::Ptr;
    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    using BoolTreePtrType = typename BoolTreeType::Ptr;

    const TreeType& inputTree = TreeAdapter<GridOrTreeType>::tree(volume);

    // 1. Mask zero crossing voxels
    BoolTreePtrType mask = extractIsosurfaceMask(inputTree, lsutilGridZero<GridOrTreeType>());

    // 2. Segment the zero crossing mask
    std::vector<BoolTreePtrType> maskSegmentArray;
    extractActiveVoxelSegmentMasks(*mask, maskSegmentArray);

    const size_t numSegments = maskSegmentArray.size();

    if (numSegments < 2) {
        // single segment early-out
        TreePtrType segment(new TreeType(inputTree));
        segments.push_back(
            level_set_util_internal::GridOrTreeConstructor<GridOrTreeType>::construct(
                volume, segment));
        return;
    }

    const tbb::blocked_range<size_t> segmentRange(0, numSegments);


    // 3. Expand zero crossing mask to capture sdf narrow band
    tbb::parallel_for(segmentRange,
        level_set_util_internal::ExpandNarrowbandMask<TreeType>(inputTree, maskSegmentArray));

    // 4. Export sdf segments
    std::vector<TreePtrType> outputSegmentArray(numSegments);

    tbb::parallel_for(segmentRange, level_set_util_internal::MaskedCopy<TreeType>(
        inputTree, outputSegmentArray, maskSegmentArray));

    tbb::parallel_for(segmentRange,
        level_set_util_internal::FloodFillSign<TreeType>(inputTree, outputSegmentArray));


    for (size_t n = 0, N = numSegments; n < N; ++n) {
        segments.push_back(
            level_set_util_internal::GridOrTreeConstructor<GridOrTreeType>::construct(
                volume, outputSegmentArray[n]));
    }
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_UTIL_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
