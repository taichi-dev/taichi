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
//
/// @file SignedFloodFill.h
///
/// @brief Propagate the signs of distance values from the active voxels
/// in the narrow band to the inactive values outside the narrow band.
///
/// @author Ken Museth

#ifndef OPENVDB_TOOLS_SIGNEDFLOODFILL_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_SIGNEDFLOODFILL_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Types.h> // for Index typedef
#include <openvdb/math/Math.h> // for math::negative
#include <openvdb/tree/NodeManager.h>
#include <map>
#include <type_traits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Set the values of all inactive voxels and tiles of a narrow-band
/// level set from the signs of the active voxels, setting outside values to
/// +background and inside values to -background.
///
/// @warning This method should only be used on closed, symmetric narrow-band level sets.
///
/// @note If a LeafManager is used the cached leaf nodes are reused,
/// resulting in slightly better overall performance.
///
/// @param tree          Tree or LeafManager that will be flood filled.
/// @param threaded      enable or disable threading  (threading is enabled by default)
/// @param grainSize     used to control the threading granularity (default is 1)
/// @param minLevel      Specify the lowest tree level to process (leafnode level = 0)
///
/// @throw TypeError if the ValueType of @a tree is not floating-point.
template<typename TreeOrLeafManagerT>
inline void
signedFloodFill(TreeOrLeafManagerT& tree, bool threaded = true,
    size_t grainSize = 1, Index minLevel = 0);


/// @brief Set the values of all inactive voxels and tiles of a narrow-band
/// level set from the signs of the active voxels, setting exterior values to
/// @a outsideWidth and interior values to @a insideWidth.  Set the background value
/// of this tree to @a outsideWidth.
///
/// @warning This method should only be used on closed, narrow-band level sets.
///
/// @note If a LeafManager is used the cached leaf nodes are reused
/// resulting in slightly better overall performance.
///
/// @param tree          Tree or LeafManager that will be flood filled
/// @param outsideWidth  the width of the outside of the narrow band
/// @param insideWidth   the width of the inside of the narrow band
/// @param threaded      enable or disable threading  (threading is enabled by default)
/// @param grainSize     used to control the threading granularity (default is 1)
/// @param minLevel      Specify the lowest tree level to process (leafnode level = 0)
///
/// @throw TypeError if the ValueType of @a tree is not floating-point.
template<typename TreeOrLeafManagerT>
inline void
signedFloodFillWithValues(
    TreeOrLeafManagerT& tree,
    const typename TreeOrLeafManagerT::ValueType& outsideWidth,
    const typename TreeOrLeafManagerT::ValueType& insideWidth,
    bool threaded = true,
    size_t grainSize = 1,
    Index minLevel = 0);


////////////////////////// Implementation of SignedFloodFill ////////////////////////////


template<typename TreeOrLeafManagerT>
class SignedFloodFillOp
{
public:
    using ValueT = typename TreeOrLeafManagerT::ValueType;
    using RootT = typename TreeOrLeafManagerT::RootNodeType;
    using LeafT = typename TreeOrLeafManagerT::LeafNodeType;
    static_assert(std::is_signed<ValueT>::value,
        "signed flood fill is supported only for signed value grids");

    SignedFloodFillOp(const TreeOrLeafManagerT& tree, Index minLevel = 0)
        : mOutside(ValueT(math::Abs(tree.background())))
        , mInside(ValueT(math::negative(mOutside)))
        , mMinLevel(minLevel)
    {
    }

    SignedFloodFillOp(ValueT outsideValue, ValueT insideValue, Index minLevel = 0)
        : mOutside(ValueT(math::Abs(outsideValue)))
        , mInside(ValueT(math::negative(math::Abs(insideValue))))
        , mMinLevel(minLevel)
    {
    }

    // Nothing to do at the leaf node level
    void operator()(LeafT& leaf) const
    {
        if (LeafT::LEVEL < mMinLevel) return;

#if OPENVDB_ABI_VERSION_NUMBER >= 3
        if (!leaf.allocate()) return; // this assures that the buffer is allocated and in-memory
#endif
        const typename LeafT::NodeMaskType& valueMask = leaf.getValueMask();
        // WARNING: "Never do what you're about to see at home, we're what you call experts!"
        typename LeafT::ValueType* buffer =
            const_cast<typename LeafT::ValueType*>(&(leaf.getFirstValue()));

        const Index first = valueMask.findFirstOn();
        if (first < LeafT::SIZE) {
            bool xInside = buffer[first]<0, yInside = xInside, zInside = xInside;
            for (Index x = 0; x != (1 << LeafT::LOG2DIM); ++x) {
                const Index x00 = x << (2 * LeafT::LOG2DIM);
                if (valueMask.isOn(x00)) xInside = buffer[x00] < 0; // element(x, 0, 0)
                yInside = xInside;
                for (Index y = 0; y != (1 << LeafT::LOG2DIM); ++y) {
                    const Index xy0 = x00 + (y << LeafT::LOG2DIM);
                    if (valueMask.isOn(xy0)) yInside = buffer[xy0] < 0; // element(x, y, 0)
                    zInside = yInside;
                    for (Index z = 0; z != (1 << LeafT::LOG2DIM); ++z) {
                        const Index xyz = xy0 + z; // element(x, y, z)
                        if (valueMask.isOn(xyz)) {
                            zInside = buffer[xyz] < 0;
                        } else {
                            buffer[xyz] = zInside ? mInside : mOutside;
                        }
                    }
                }
            }
        } else {// if no active voxels exist simply use the sign of the first value
            leaf.fill(buffer[0] < 0 ? mInside : mOutside);
        }
    }

    // Prune the child nodes of the internal nodes
    template<typename NodeT>
    void operator()(NodeT& node) const
    {
        if (NodeT::LEVEL < mMinLevel) return;
        // We assume the child nodes have already been flood filled!
        const typename NodeT::NodeMaskType& childMask = node.getChildMask();
        // WARNING: "Never do what you're about to see at home, we're what you call experts!"
        typename NodeT::UnionType* table = const_cast<typename NodeT::UnionType*>(node.getTable());

        const Index first = childMask.findFirstOn();
        if (first < NodeT::NUM_VALUES) {
            bool xInside = table[first].getChild()->getFirstValue()<0;
            bool yInside = xInside, zInside = xInside;
            for (Index x = 0; x != (1 << NodeT::LOG2DIM); ++x) {
                const int x00 = x << (2 * NodeT::LOG2DIM); // offset for block(x, 0, 0)
                if (childMask.isOn(x00)) xInside = table[x00].getChild()->getLastValue()<0;
                yInside = xInside;
                for (Index y = 0; y != (1 << NodeT::LOG2DIM); ++y) {
                    const Index xy0 = x00 + (y << NodeT::LOG2DIM); // offset for block(x, y, 0)
                    if (childMask.isOn(xy0)) yInside = table[xy0].getChild()->getLastValue()<0;
                    zInside = yInside;
                    for (Index z = 0; z != (1 << NodeT::LOG2DIM); ++z) {
                        const Index xyz = xy0 + z; // offset for block(x, y, z)
                        if (childMask.isOn(xyz)) {
                            zInside = table[xyz].getChild()->getLastValue()<0;
                        } else {
                            table[xyz].setValue(zInside ? mInside : mOutside);
                        }
                    }
                }
            }
        } else {//no child nodes exist simply use the sign of the first tile value.
            const ValueT v =  table[0].getValue()<0 ? mInside : mOutside;
            for (Index i = 0; i < NodeT::NUM_VALUES; ++i) table[i].setValue(v);
        }
    }

    // Prune the child nodes of the root node
    void operator()(RootT& root) const
    {
        if (RootT::LEVEL < mMinLevel) return;
        using ChildT = typename RootT::ChildNodeType;
        // Insert the child nodes into a map sorted according to their origin
        std::map<Coord, ChildT*> nodeKeys;
        typename RootT::ChildOnIter it = root.beginChildOn();
        for (; it; ++it) nodeKeys.insert(std::pair<Coord, ChildT*>(it.getCoord(), &(*it)));
        static const Index DIM = RootT::ChildNodeType::DIM;

        // We employ a simple z-scanline algorithm that inserts inactive tiles with
        // the inside value if they are sandwiched between inside child nodes only!
        typename std::map<Coord, ChildT*>::const_iterator b = nodeKeys.begin(), e = nodeKeys.end();
        if ( b == e ) return;
        for (typename std::map<Coord, ChildT*>::const_iterator a = b++; b != e; ++a, ++b) {
            Coord d = b->first - a->first; // delta of neighboring coordinates
            if (d[0]!=0 || d[1]!=0 || d[2]==Int32(DIM)) continue;// not same z-scanline or neighbors
            const ValueT fill[] = { a->second->getLastValue(), b->second->getFirstValue() };
            if (!(fill[0] < 0) || !(fill[1] < 0)) continue; // scanline isn't inside
            Coord c = a->first + Coord(0u, 0u, DIM);
            for (; c[2] != b->first[2]; c[2] += DIM) root.addTile(c, mInside, false);
        }
        root.setBackground(mOutside, /*updateChildNodes=*/false);
    }

private:
    const ValueT mOutside, mInside;
    const Index mMinLevel;
};// SignedFloodFillOp


//{
/// @cond OPENVDB_SIGNED_FLOOD_FILL_INTERNAL

template<typename TreeOrLeafManagerT>
inline
typename std::enable_if<std::is_signed<typename TreeOrLeafManagerT::ValueType>::value, void>::type
doSignedFloodFill(TreeOrLeafManagerT& tree,
                  typename TreeOrLeafManagerT::ValueType outsideValue,
                  typename TreeOrLeafManagerT::ValueType insideValue,
                  bool threaded,
                  size_t grainSize,
                  Index minLevel)
{
    tree::NodeManager<TreeOrLeafManagerT> nodes(tree);
    SignedFloodFillOp<TreeOrLeafManagerT> op(outsideValue, insideValue, minLevel);
    nodes.foreachBottomUp(op, threaded, grainSize);
}

// Dummy (no-op) implementation for unsigned types
template <typename TreeOrLeafManagerT>
inline
typename std::enable_if<!std::is_signed<typename TreeOrLeafManagerT::ValueType>::value, void>::type
doSignedFloodFill(TreeOrLeafManagerT&,
                  const typename TreeOrLeafManagerT::ValueType&,
                  const typename TreeOrLeafManagerT::ValueType&,
                  bool,
                  size_t,
                  Index)
{
    OPENVDB_THROW(TypeError,
        "signedFloodFill is supported only for signed value grids");
}

/// @endcond
//}


// If the narrow-band is symmetric and unchanged
template <typename TreeOrLeafManagerT>
inline void
signedFloodFillWithValues(
    TreeOrLeafManagerT& tree,
    const typename TreeOrLeafManagerT::ValueType& outsideValue,
    const typename TreeOrLeafManagerT::ValueType& insideValue,
    bool threaded,
    size_t grainSize,
    Index minLevel)
{
    doSignedFloodFill(tree, outsideValue, insideValue, threaded, grainSize, minLevel);
}


template <typename TreeOrLeafManagerT>
inline void
signedFloodFill(TreeOrLeafManagerT& tree,
                bool threaded,
                size_t grainSize,
                Index minLevel)
{
    const typename TreeOrLeafManagerT::ValueType v = tree.root().background();
    doSignedFloodFill(tree, v, math::negative(v), threaded, grainSize, minLevel);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_RESETBACKGROUND_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
