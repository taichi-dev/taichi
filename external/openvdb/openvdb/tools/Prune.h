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
/// @file Prune.h
///
/// @brief Defined various multi-threaded utility functions for trees
///
/// @author Ken Museth

#ifndef OPENVDB_TOOLS_PRUNE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_PRUNE_HAS_BEEN_INCLUDED

#include <openvdb/math/Math.h> // for isNegative and negative
#include <openvdb/Types.h>
#include <openvdb/tree/NodeManager.h>
#include <algorithm> // for std::nth_element()
#include <type_traits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Reduce the memory footprint of a @a tree by replacing with tiles
/// any nodes whose values are all the same (optionally to within a tolerance)
/// and have the same active state.
///
/// @note For trees with non-boolean values a child node with (approximately)
/// constant values are replaced with a tile value corresponding to the median
/// of the values in said child node.
///
/// @param tree       the tree to be pruned
/// @param tolerance  tolerance within which values are considered to be equal
/// @param threaded   enable or disable threading (threading is enabled by default)
/// @param grainSize  used to control the threading granularity (default is 1)
template<typename TreeT>
inline void
prune(TreeT& tree,
      typename TreeT::ValueType tolerance = zeroVal<typename TreeT::ValueType>(),
      bool threaded = true,
      size_t grainSize = 1);


/// @brief Reduce the memory footprint of a @a tree by replacing with tiles
/// any non-leaf nodes whose values are all the same (optionally to within a tolerance)
/// and have the same active state.
///
/// @param tree       the tree to be pruned
/// @param tolerance  tolerance within which values are considered to be equal
/// @param threaded   enable or disable threading (threading is enabled by default)
/// @param grainSize  used to control the threading granularity (default is 1)
template<typename TreeT>
inline void
pruneTiles(TreeT& tree,
           typename TreeT::ValueType tolerance = zeroVal<typename TreeT::ValueType>(),
           bool threaded = true,
           size_t grainSize = 1);


/// @brief Reduce the memory footprint of a @a tree by replacing with
/// background tiles any nodes whose values are all inactive.
///
/// @param tree       the tree to be pruned
/// @param threaded   enable or disable threading (threading is enabled by default)
/// @param grainSize  used to control the threading granularity (default is 1)
template<typename TreeT>
inline void
pruneInactive(TreeT& tree, bool threaded = true, size_t grainSize = 1);


/// @brief Reduce the memory footprint of a @a tree by replacing any nodes
/// whose values are all inactive with tiles of the given @a value.
///
/// @param tree       the tree to be pruned
/// @param value      value assigned to inactive tiles created during pruning
/// @param threaded   enable or disable threading (threading is enabled by default)
/// @param grainSize  used to control the threading granularity (default is 1)
template<typename TreeT>
inline void
pruneInactiveWithValue(
    TreeT& tree,
    const typename TreeT::ValueType& value,
    bool threaded = true,
    size_t grainSize = 1);


/// @brief Reduce the memory footprint of a @a tree by replacing nodes
/// whose values are all inactive with inactive tiles having a value equal to
/// the first value encountered in the (inactive) child.
/// @details This method is faster than tolerance-based prune and
/// useful for narrow-band level set applications where inactive
/// values are limited to either an inside or an outside value.
///
/// @param tree       the tree to be pruned
/// @param threaded   enable or disable threading (threading is enabled by default)
/// @param grainSize  used to control the threading granularity (default is 1)
///
/// @throw ValueError if the background of the @a tree is negative (as defined by math::isNegative)
template<typename TreeT>
inline void
pruneLevelSet(TreeT& tree,
              bool threaded = true,
              size_t grainSize = 1);


/// @brief Reduce the memory footprint of a @a tree by replacing nodes whose voxel values
/// are all inactive with inactive tiles having the value -| @a insideWidth |
/// if the voxel values are negative and | @a outsideWidth | otherwise.
///
/// @details This method is faster than tolerance-based prune and
/// useful for narrow-band level set applications where inactive
/// values are limited to either an inside or an outside value.
///
/// @param tree          the tree to be pruned
/// @param outsideWidth  the width of the outside of the narrow band
/// @param insideWidth   the width of the inside of the narrow band
/// @param threaded      enable or disable threading (threading is enabled by default)
/// @param grainSize     used to control the threading granularity (default is 1)
///
/// @throw ValueError if @a outsideWidth is negative or @a insideWidth is
/// not negative (as defined by math::isNegative).
template<typename TreeT>
inline void
pruneLevelSet(TreeT& tree,
              const typename TreeT::ValueType& outsideWidth,
              const typename TreeT::ValueType& insideWidth,
              bool threaded = true,
              size_t grainSize = 1);


////////////////////////////////////////////////


template<typename TreeT, Index TerminationLevel = 0>
class InactivePruneOp
{
public:
    using ValueT = typename TreeT::ValueType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    static_assert(RootT::LEVEL > TerminationLevel, "TerminationLevel out of range");

    InactivePruneOp(TreeT& tree) : mValue(tree.background())
    {
        tree.clearAllAccessors();//clear cache of nodes that could be pruned
    }

    InactivePruneOp(TreeT& tree, const ValueT& v) : mValue(v)
    {
        tree.clearAllAccessors();//clear cache of nodes that could be pruned
    }

    // Nothing to do at the leaf node level
    void operator()(LeafT&) const {}

    // Prune the child nodes of the internal nodes
    template<typename NodeT>
    void operator()(NodeT& node) const
    {
        if (NodeT::LEVEL > TerminationLevel) {
            for (typename NodeT::ChildOnIter it=node.beginChildOn(); it; ++it) {
                if (it->isInactive()) node.addTile(it.pos(), mValue, false);
            }
        }
    }

    // Prune the child nodes of the root node
    void operator()(RootT& root) const
    {
        for (typename RootT::ChildOnIter it = root.beginChildOn(); it; ++it) {
            if (it->isInactive()) root.addTile(it.getCoord(), mValue, false);
        }
        root.eraseBackgroundTiles();
    }

private:
    const ValueT mValue;
};// InactivePruneOp


template<typename TreeT, Index TerminationLevel = 0>
class TolerancePruneOp
{
public:
    using ValueT = typename TreeT::ValueType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    static_assert(RootT::LEVEL > TerminationLevel, "TerminationLevel out of range");

    TolerancePruneOp(TreeT& tree, const ValueT& tol) : mTolerance(tol)
    {
        tree.clearAllAccessors();//clear cache of nodes that could be pruned
    }

    // Prune the child nodes of the root node
    inline void operator()(RootT& root) const
    {
        ValueT value;
        bool   state;
        for (typename RootT::ChildOnIter it = root.beginChildOn(); it; ++it) {
            if (this->isConstant(*it, value, state)) root.addTile(it.getCoord(), value, state);
        }
        root.eraseBackgroundTiles();
    }

    // Prune the child nodes of the internal nodes
    template<typename NodeT>
    inline void operator()(NodeT& node) const
    {
        if (NodeT::LEVEL > TerminationLevel) {
            ValueT value;
            bool   state;
            for (typename NodeT::ChildOnIter it=node.beginChildOn(); it; ++it) {
                if (this->isConstant(*it, value, state)) node.addTile(it.pos(), value, state);
            }
        }
    }

    // Nothing to do at the leaf node level
    inline void operator()(LeafT&) const {}

private:
    // Private method specialized for leaf nodes
    inline ValueT median(LeafT& leaf) const {return leaf.medianAll(leaf.buffer().data());}

    // Private method for internal nodes
    template<typename NodeT>
    inline typename NodeT::ValueType median(NodeT& node) const
    {
        using UnionT = typename NodeT::UnionType;
        UnionT* data = const_cast<UnionT*>(node.getTable());//never do this at home kids :)
        static const size_t midpoint = (NodeT::NUM_VALUES - 1) >> 1;
        auto op = [](const UnionT& a, const UnionT& b){return a.getValue() < b.getValue();};
        std::nth_element(data, data + midpoint, data + NodeT::NUM_VALUES, op);
        return data[midpoint].getValue();
    }

    // Specialization to nodes templated on booleans values
    template<typename NodeT>
    inline
    typename std::enable_if<std::is_same<bool, typename NodeT::ValueType>::value, bool>::type
    isConstant(NodeT& node, bool& value, bool& state) const
    {
        return node.isConstant(value, state, mTolerance);
    }

    // Nodes templated on non-boolean values
    template<typename NodeT>
    inline
    typename std::enable_if<!std::is_same<bool, typename NodeT::ValueType>::value, bool>::type
    isConstant(NodeT& node, ValueT& value, bool& state) const
    {
        ValueT tmp;
        const bool test = node.isConstant(value, tmp, state, mTolerance);
        if (test) value = this->median(node);
        return test;
    }

    const ValueT mTolerance;
};// TolerancePruneOp


template<typename TreeT, Index TerminationLevel = 0>
class LevelSetPruneOp
{
public:
    using ValueT = typename TreeT::ValueType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    static_assert(RootT::LEVEL > TerminationLevel, "TerminationLevel out of range");

    LevelSetPruneOp(TreeT& tree)
        : mOutside(tree.background())
        , mInside(math::negative(mOutside))
    {
        if (math::isNegative(mOutside)) {
            OPENVDB_THROW(ValueError,
                          "LevelSetPruneOp: the background value cannot be negative!");
        }
        tree.clearAllAccessors();//clear cache of nodes that could be pruned
    }

    LevelSetPruneOp(TreeT& tree, const ValueT& outside, const ValueT& inside)
        : mOutside(outside)
        , mInside(inside)
    {
        if (math::isNegative(mOutside)) {
            OPENVDB_THROW(ValueError,
                          "LevelSetPruneOp: the outside value cannot be negative!");
        }
        if (!math::isNegative(mInside)) {
            OPENVDB_THROW(ValueError,
                          "LevelSetPruneOp: the inside value must be negative!");
        }
        tree.clearAllAccessors();//clear cache of nodes that could be pruned
    }

    // Nothing to do at the leaf node level
    void operator()(LeafT&) const {}

    // Prune the child nodes of the internal nodes
    template<typename NodeT>
    void operator()(NodeT& node) const
    {
        if (NodeT::LEVEL > TerminationLevel) {
            for (typename NodeT::ChildOnIter it=node.beginChildOn(); it; ++it) {
                if (it->isInactive()) node.addTile(it.pos(), this->getTileValue(it), false);
            }
        }
    }

    // Prune the child nodes of the root node
    void operator()(RootT& root) const
    {
        for (typename RootT::ChildOnIter it = root.beginChildOn(); it; ++it) {
            if (it->isInactive()) root.addTile(it.getCoord(), this->getTileValue(it), false);
        }
        root.eraseBackgroundTiles();
    }

private:
    template <typename IterT>
    inline ValueT getTileValue(const IterT& iter) const
    {
        return  math::isNegative(iter->getFirstValue()) ? mInside : mOutside;
    }

    const ValueT mOutside, mInside;
};// LevelSetPruneOp


template<typename TreeT>
inline void
prune(TreeT& tree, typename TreeT::ValueType tol, bool threaded, size_t grainSize)
{
    tree::NodeManager<TreeT, TreeT::DEPTH-2> nodes(tree);
    TolerancePruneOp<TreeT> op(tree, tol);
    nodes.foreachBottomUp(op, threaded, grainSize);
}


template<typename TreeT>
inline void
pruneTiles(TreeT& tree, typename TreeT::ValueType tol, bool threaded, size_t grainSize)
{
    tree::NodeManager<TreeT, TreeT::DEPTH-3> nodes(tree);
    TolerancePruneOp<TreeT> op(tree, tol);
    nodes.foreachBottomUp(op, threaded, grainSize);
}


template<typename TreeT>
inline void
pruneInactive(TreeT& tree, bool threaded, size_t grainSize)
{
    tree::NodeManager<TreeT, TreeT::DEPTH-2> nodes(tree);
    InactivePruneOp<TreeT> op(tree);
    nodes.foreachBottomUp(op, threaded, grainSize);
}


template<typename TreeT>
inline void
pruneInactiveWithValue(TreeT& tree, const typename TreeT::ValueType& v,
    bool threaded, size_t grainSize)
{
    tree::NodeManager<TreeT, TreeT::DEPTH-2> nodes(tree);
    InactivePruneOp<TreeT> op(tree, v);
    nodes.foreachBottomUp(op, threaded, grainSize);
}


template<typename TreeT>
inline void
pruneLevelSet(TreeT& tree,
              const typename TreeT::ValueType& outside,
              const typename TreeT::ValueType& inside,
              bool threaded,
              size_t grainSize)
{
    tree::NodeManager<TreeT, TreeT::DEPTH-2> nodes(tree);
    LevelSetPruneOp<TreeT> op(tree, outside, inside);
    nodes.foreachBottomUp(op, threaded, grainSize);
}


template<typename TreeT>
inline void
pruneLevelSet(TreeT& tree, bool threaded, size_t grainSize)
{
    tree::NodeManager<TreeT, TreeT::DEPTH-2> nodes(tree);
    LevelSetPruneOp<TreeT> op(tree);
    nodes.foreachBottomUp(op, threaded, grainSize);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_PRUNE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
