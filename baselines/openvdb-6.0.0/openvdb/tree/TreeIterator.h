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
/// @file tree/TreeIterator.h

#ifndef OPENVDB_TREE_TREEITERATOR_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_TREEITERATOR_HAS_BEEN_INCLUDED

#include <boost/mpl/front.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <openvdb/version.h>
#include <openvdb/Types.h>
#include <algorithm>
#include <sstream>
#include <string>
#include <type_traits>

// Prior to 0.96.1, depth-bounded value iterators always descended to the leaf level
// and iterated past leaf nodes.  Now, they never descend past the maximum depth.
// Comment out the following line to restore the older, less-efficient behavior:
#define ENABLE_TREE_VALUE_DEPTH_BOUND_OPTIMIZATION


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

/// CopyConstness<T1, T2>::Type is either const T2 or T2 with no const qualifier,
/// depending on whether T1 is const.  For example,
/// - CopyConstness<int, int>::Type is int
/// - CopyConstness<int, const int>::Type is int
/// - CopyConstness<const int, int>::Type is const int
/// - CopyConstness<const int, const int>::Type is const int
template<typename FromType, typename ToType> struct CopyConstness {
    using Type = typename std::remove_const<ToType>::type;
};
template<typename FromType, typename ToType> struct CopyConstness<const FromType, ToType> {
    using Type = const ToType;
};


////////////////////////////////////////


namespace iter {

template<typename HeadT, int HeadLevel>
struct InvertedTree {
    using SubtreeT = typename InvertedTree<typename HeadT::ChildNodeType, HeadLevel-1>::Type;
    using Type = typename boost::mpl::push_back<SubtreeT, HeadT>::type;
};
template<typename HeadT>
struct InvertedTree<HeadT, /*HeadLevel=*/1> {
    using Type = typename boost::mpl::vector<typename HeadT::ChildNodeType, HeadT>::type;
};

} // namespace iter


////////////////////////////////////////


/// IterTraits provides the following for iterators of the standard types,
/// i.e., for {Child,Value}{On,Off,All}{Iter,CIter}:
/// - a NodeConverter template to convert an iterator for one type of node
///   to an iterator of the same type for another type of node; for example,
///   IterTraits<RootNode, RootNode::ValueOnIter>::NodeConverter<LeafNode>::Type
///   is synonymous with LeafNode::ValueOnIter.
/// - a begin(node) function that returns a begin iterator for a node of arbitrary type;
///   for example, IterTraits<LeafNode, LeafNode::ValueOnIter>::begin(leaf) returns
///   leaf.beginValueOn()
/// - a getChild() function that returns a pointer to the child node to which the iterator
///   is currently pointing (always null if the iterator is a Value iterator)
template<typename NodeT, typename IterT>
struct IterTraits
{
    template<typename ChildT> static ChildT* getChild(const IterT&) { return nullptr; }
};

template<typename NodeT>
struct IterTraits<NodeT, typename NodeT::ChildOnIter>
{
    using IterT = typename NodeT::ChildOnIter;
    static IterT begin(NodeT& node) { return node.beginChildOn(); }
    template<typename ChildT> static ChildT* getChild(const IterT& iter) {
        return &iter.getValue();
    }
    template<typename OtherNodeT> struct NodeConverter {
        using Type = typename OtherNodeT::ChildOnIter;
    };
};

template<typename NodeT>
struct IterTraits<NodeT, typename NodeT::ChildOnCIter>
{
    using IterT = typename NodeT::ChildOnCIter;
    static IterT begin(const NodeT& node) { return node.cbeginChildOn(); }
    template<typename ChildT> static const ChildT* getChild(const IterT& iter) {
        return &iter.getValue();
    }
    template<typename OtherNodeT> struct NodeConverter {
        using Type = typename OtherNodeT::ChildOnCIter;
    };
};

template<typename NodeT>
struct IterTraits<NodeT, typename NodeT::ChildOffIter>
{
    using IterT = typename NodeT::ChildOffIter;
    static IterT begin(NodeT& node) { return node.beginChildOff(); }
    template<typename OtherNodeT> struct NodeConverter {
        using Type = typename OtherNodeT::ChildOffIter;
    };
};

template<typename NodeT>
struct IterTraits<NodeT, typename NodeT::ChildOffCIter>
{
    using IterT = typename NodeT::ChildOffCIter;
    static IterT begin(const NodeT& node) { return node.cbeginChildOff(); }
    template<typename OtherNodeT> struct NodeConverter {
        using Type = typename OtherNodeT::ChildOffCIter;
    };
};

template<typename NodeT>
struct IterTraits<NodeT, typename NodeT::ChildAllIter>
{
    using IterT = typename NodeT::ChildAllIter;
    static IterT begin(NodeT& node) { return node.beginChildAll(); }
    template<typename ChildT> static ChildT* getChild(const IterT& iter) {
        typename IterT::NonConstValueType val;
        return iter.probeChild(val);
    }
    template<typename OtherNodeT> struct NodeConverter {
        using Type = typename OtherNodeT::ChildAllIter;
    };
};

template<typename NodeT>
struct IterTraits<NodeT, typename NodeT::ChildAllCIter>
{
    using IterT = typename NodeT::ChildAllCIter;
    static IterT begin(const NodeT& node) { return node.cbeginChildAll(); }
    template<typename ChildT> static ChildT* getChild(const IterT& iter) {
        typename IterT::NonConstValueType val;
        return iter.probeChild(val);
    }
    template<typename OtherNodeT> struct NodeConverter {
        using Type = typename OtherNodeT::ChildAllCIter;
    };
};

template<typename NodeT>
struct IterTraits<NodeT, typename NodeT::ValueOnIter>
{
    using IterT = typename NodeT::ValueOnIter;
    static IterT begin(NodeT& node) { return node.beginValueOn(); }
    template<typename OtherNodeT> struct NodeConverter {
        using Type = typename OtherNodeT::ValueOnIter;
    };
};

template<typename NodeT>
struct IterTraits<NodeT, typename NodeT::ValueOnCIter>
{
    using IterT = typename NodeT::ValueOnCIter;
    static IterT begin(const NodeT& node) { return node.cbeginValueOn(); }
    template<typename OtherNodeT> struct NodeConverter {
        using Type = typename OtherNodeT::ValueOnCIter;
    };
};

template<typename NodeT>
struct IterTraits<NodeT, typename NodeT::ValueOffIter>
{
    using IterT = typename NodeT::ValueOffIter;
    static IterT begin(NodeT& node) { return node.beginValueOff(); }
    template<typename OtherNodeT> struct NodeConverter {
        using Type = typename OtherNodeT::ValueOffIter;
    };
};

template<typename NodeT>
struct IterTraits<NodeT, typename NodeT::ValueOffCIter>
{
    using IterT = typename NodeT::ValueOffCIter;
    static IterT begin(const NodeT& node) { return node.cbeginValueOff(); }
    template<typename OtherNodeT> struct NodeConverter {
        using Type = typename OtherNodeT::ValueOffCIter;
    };
};

template<typename NodeT>
struct IterTraits<NodeT, typename NodeT::ValueAllIter>
{
    using IterT = typename NodeT::ValueAllIter;
    static IterT begin(NodeT& node) { return node.beginValueAll(); }
    template<typename OtherNodeT> struct NodeConverter {
        using Type = typename OtherNodeT::ValueAllIter;
    };
};

template<typename NodeT>
struct IterTraits<NodeT, typename NodeT::ValueAllCIter>
{
    using IterT = typename NodeT::ValueAllCIter;
    static IterT begin(const NodeT& node) { return node.cbeginValueAll(); }
    template<typename OtherNodeT> struct NodeConverter {
        using Type = typename OtherNodeT::ValueAllCIter;
    };
};


////////////////////////////////////////


/// @brief An IterListItem is an element of a compile-time linked list of iterators
/// to nodes of different types.
///
/// The list is constructed by traversing the template hierarchy of a Tree in reverse order,
/// so typically the elements will be a LeafNode iterator of some type (e.g., ValueOnCIter),
/// followed by one or more InternalNode iterators of the same type, followed by a RootNode
/// iterator of the same type.
///
/// The length of the list is fixed at compile time, and because it is implemented using
/// nested, templated classes, much of the list traversal logic can be optimized away.
template<typename PrevItemT, typename NodeVecT, size_t VecSize, Index _Level>
class IterListItem
{
public:
    /// The type of iterator stored in the previous list item
    using PrevIterT = typename PrevItemT::IterT;
    /// The type of node (non-const) whose iterator is stored in this list item
    using _NodeT = typename boost::mpl::front<NodeVecT>::type;
    /// The type of iterator stored in this list item (e.g., InternalNode::ValueOnCIter)
    using IterT = typename IterTraits<typename PrevIterT::NonConstNodeType, PrevIterT>::template
        NodeConverter<_NodeT>::Type;

    /// The type of node (const or non-const) over which IterT iterates (e.g., const RootNode<...>)
    using NodeT = typename IterT::NodeType;
    /// The type of the node with const qualifiers removed ("Non-Const")
    using NCNodeT = typename IterT::NonConstNodeType;
    /// The type of value (with const qualifiers removed) to which the iterator points
    using NCValueT = typename IterT::NonConstValueType;
    /// NodeT's child node type, with the same constness (e.g., const InternalNode<...>)
    using ChildT = typename CopyConstness<NodeT, typename NodeT::ChildNodeType>::Type;
    /// NodeT's child node type with const qualifiers removed
    using NCChildT = typename CopyConstness<NCNodeT, typename NCNodeT::ChildNodeType>::Type;
    using ITraits = IterTraits<NCNodeT, IterT>;
    /// NodeT's level in its tree (0 = LeafNode)
    static const Index Level = _Level;

    IterListItem(PrevItemT* prev): mNext(this), mPrev(prev) {}

    IterListItem(const IterListItem& other):
        mIter(other.mIter), mNext(other.mNext), mPrev(nullptr) {}
    IterListItem& operator=(const IterListItem& other)
    {
        if (&other != this) {
            mIter = other.mIter;
            mNext = other.mNext;
            mPrev = nullptr; ///< @note external call to updateBackPointers() required
        }
        return *this;
    }

    void updateBackPointers(PrevItemT* prev) { mPrev = prev; mNext.updateBackPointers(this); }

    void setIter(const IterT& iter) { mIter = iter; }
    template<typename OtherIterT>
    void setIter(const OtherIterT& iter) { mNext.setIter(iter); }

    /// Return the node over which this list element's iterator iterates.
    void getNode(Index lvl, NodeT*& node) const
    {
        node = (lvl <= Level) ? mIter.getParentNode() : nullptr;
    }
    /// Return the node over which one of the following list elements' iterator iterates.
    template<typename OtherNodeT>
    void getNode(Index lvl, OtherNodeT*& node) const { mNext.getNode(lvl, node); }

    /// @brief Initialize the iterator for level @a lvl of the tree with the node
    /// over which the corresponding iterator of @a otherListItem is iterating.
    ///
    /// For example, if @a otherListItem contains a LeafNode::ValueOnIter,
    /// initialize this list's leaf iterator with the same LeafNode.
    template<typename OtherIterListItemT>
    void initLevel(Index lvl, OtherIterListItemT& otherListItem)
    {
        if (lvl == Level) {
            const NodeT* node = nullptr;
            otherListItem.getNode(lvl, node);
            mIter = (node == nullptr) ? IterT() : ITraits::begin(*const_cast<NodeT*>(node));
        } else {
            // Forward to one of the following list elements.
            mNext.initLevel(lvl, otherListItem);
        }
    }

    /// Return The table offset of the iterator at level @a lvl of the tree.
    Index pos(Index lvl) const { return (lvl == Level) ? mIter.pos() : mNext.pos(lvl); }

    /// Return @c true if the iterator at level @a lvl of the tree has not yet reached its end.
    bool test(Index lvl) const { return (lvl == Level) ? mIter.test() : mNext.test(lvl); }

    /// Increment the iterator at level @a lvl of the tree.
    bool next(Index lvl) { return (lvl == Level) ? mIter.next() : mNext.next(lvl); }

    /// @brief If the iterator at level @a lvl of the tree points to a child node,
    /// initialize the next iterator in this list with that child node.
    bool down(Index lvl)
    {
        if (lvl == Level && mPrev != nullptr && mIter) {
            if (ChildT* child = ITraits::template getChild<ChildT>(mIter)) {
                mPrev->setIter(PrevItemT::ITraits::begin(*child));
                return true;
            }
        }
        return (lvl > Level) ? mNext.down(lvl) : false;
    }

    /// @brief Return the global coordinates of the voxel or tile to which the iterator
    /// at level @a lvl of the tree is currently pointing.
    Coord getCoord(Index lvl) const
    {
        return (lvl == Level) ? mIter.getCoord() : mNext.getCoord(lvl);
    }
    Index getChildDim(Index lvl) const
    {
        return (lvl == Level) ? NodeT::getChildDim() : mNext.getChildDim(lvl);
    }
    /// Return the number of (virtual) voxels spanned by a tile value or child node
    Index64 getVoxelCount(Index lvl) const
    {
        return (lvl == Level) ? ChildT::NUM_VOXELS : mNext.getVoxelCount(lvl);
    }

    /// Return @c true if the iterator at level @a lvl of the tree points to an active value.
    bool isValueOn(Index lvl) const
    {
        return (lvl == Level) ? mIter.isValueOn() : mNext.isValueOn(lvl);
    }

    /// Return the value to which the iterator at level @a lvl of the tree points.
    const NCValueT& getValue(Index lvl) const
    {
        if (lvl == Level) return mIter.getValue();
        return mNext.getValue(lvl);
    }

    /// @brief Set the value (to @a val) to which the iterator at level @a lvl
    /// of the tree points and mark the value as active.
    /// @note Not valid when @c IterT is a const iterator type
    void setValue(Index lvl, const NCValueT& val) const
    {
        if (lvl == Level) mIter.setValue(val); else mNext.setValue(lvl, val);
    }
    /// @brief Set the value (to @a val) to which the iterator at level @a lvl of the tree
    /// points and mark the value as active if @a on is @c true, or inactive otherwise.
    /// @note Not valid when @c IterT is a const iterator type
    void setValueOn(Index lvl, bool on = true) const
    {
        if (lvl == Level) mIter.setValueOn(on); else mNext.setValueOn(lvl, on);
    }
    /// @brief Mark the value to which the iterator at level @a lvl of the tree points
    /// as inactive.
    /// @note Not valid when @c IterT is a const iterator type
    void setValueOff(Index lvl) const
    {
        if (lvl == Level) mIter.setValueOff(); else mNext.setValueOff(lvl);
    }

    /// @brief Apply a functor to the item to which this iterator is pointing.
    /// @note Not valid when @c IterT is a const iterator type
    template<typename ModifyOp>
    void modifyValue(Index lvl, const ModifyOp& op) const
    {
        if (lvl == Level) mIter.modifyValue(op); else mNext.modifyValue(lvl, op);
    }

private:
    using RestT = typename boost::mpl::pop_front<NodeVecT>::type; // NodeVecT minus its first item
    using NextItem = IterListItem<IterListItem, RestT, VecSize - 1, Level + 1>;

    IterT mIter;
    NextItem mNext;
    PrevItemT* mPrev;
};


/// The initial element of a compile-time linked list of iterators to nodes of different types
template<typename PrevItemT, typename NodeVecT, size_t VecSize>
class IterListItem<PrevItemT, NodeVecT, VecSize, /*Level=*/0U>
{
public:
    /// The type of iterator stored in the previous list item
    using PrevIterT = typename PrevItemT::IterT;
    /// The type of node (non-const) whose iterator is stored in this list item
    using _NodeT = typename boost::mpl::front<NodeVecT>::type;
    /// The type of iterator stored in this list item (e.g., InternalNode::ValueOnCIter)
    using IterT = typename IterTraits<typename PrevIterT::NonConstNodeType, PrevIterT>::template
        NodeConverter<_NodeT>::Type;

    /// The type of node (const or non-const) over which IterT iterates (e.g., const RootNode<...>)
    using NodeT = typename IterT::NodeType;
    /// The type of the node with const qualifiers removed ("Non-Const")
    using NCNodeT = typename IterT::NonConstNodeType;
    /// The type of value (with const qualifiers removed) to which the iterator points
    using NCValueT = typename IterT::NonConstValueType;
    using ITraits = IterTraits<NCNodeT, IterT>;
    /// NodeT's level in its tree (0 = LeafNode)
    static const Index Level = 0;

    IterListItem(PrevItemT*): mNext(this), mPrev(nullptr) {}

    IterListItem(const IterListItem& other):
        mIter(other.mIter), mNext(other.mNext), mPrev(nullptr) {}
    IterListItem& operator=(const IterListItem& other)
    {
        if (&other != this) {
            mIter = other.mIter;
            mNext = other.mNext;
            mPrev = nullptr;
        }
        return *this;
    }

    void updateBackPointers(PrevItemT* = nullptr)
    {
        mPrev = nullptr; mNext.updateBackPointers(this);
    }

    void setIter(const IterT& iter) { mIter = iter; }
    template<typename OtherIterT>
    void setIter(const OtherIterT& iter) { mNext.setIter(iter); }

    void getNode(Index lvl, NodeT*& node) const
    {
        node = (lvl == 0) ? mIter.getParentNode() : nullptr;
    }
    template<typename OtherNodeT>
    void getNode(Index lvl, OtherNodeT*& node) const { mNext.getNode(lvl, node); }

    template<typename OtherIterListItemT>
    void initLevel(Index lvl, OtherIterListItemT& otherListItem)
    {
        if (lvl == 0) {
            const NodeT* node = nullptr;
            otherListItem.getNode(lvl, node);
            mIter = (node == nullptr) ? IterT() : ITraits::begin(*const_cast<NodeT*>(node));
        } else {
            mNext.initLevel(lvl, otherListItem);
        }
    }

    Index pos(Index lvl) const { return (lvl == 0) ? mIter.pos() : mNext.pos(lvl); }

    bool test(Index lvl) const { return (lvl == 0) ? mIter.test() : mNext.test(lvl); }

    bool next(Index lvl) { return (lvl == 0) ? mIter.next() : mNext.next(lvl); }

    bool down(Index lvl) { return (lvl == 0) ? false : mNext.down(lvl); }

    Coord getCoord(Index lvl) const
    {
        return (lvl == 0) ?  mIter.getCoord() : mNext.getCoord(lvl);
    }
    Index getChildDim(Index lvl) const
    {
        return (lvl == 0) ? NodeT::getChildDim() : mNext.getChildDim(lvl);
    }

    Index64 getVoxelCount(Index lvl) const
    {
        return (lvl == 0) ? 1 : mNext.getVoxelCount(lvl);
    }

    bool isValueOn(Index lvl) const
    {
        return (lvl == 0) ? mIter.isValueOn() : mNext.isValueOn(lvl);
    }

    const NCValueT& getValue(Index lvl) const
    {
        if (lvl == 0) return mIter.getValue();
        return mNext.getValue(lvl);
    }

    void setValue(Index lvl, const NCValueT& val) const
    {
        if (lvl == 0) mIter.setValue(val); else mNext.setValue(lvl, val);
    }
    void setValueOn(Index lvl, bool on = true) const
    {
        if (lvl == 0) mIter.setValueOn(on); else mNext.setValueOn(lvl, on);
    }
    void setValueOff(Index lvl) const
    {
        if (lvl == 0) mIter.setValueOff(); else mNext.setValueOff(lvl);
    }

    template<typename ModifyOp>
    void modifyValue(Index lvl, const ModifyOp& op) const
    {
        if (lvl == 0) mIter.modifyValue(op); else mNext.modifyValue(lvl, op);
    }

private:
    using RestT = typename boost::mpl::pop_front<NodeVecT>::type; // NodeVecT minus its first item
    using NextItem = IterListItem<IterListItem, RestT, VecSize - 1, /*Level=*/1>;

    IterT mIter;
    NextItem mNext;
    PrevItemT* mPrev;
};


/// The final element of a compile-time linked list of iterators to nodes of different types
template<typename PrevItemT, typename NodeVecT, Index _Level>
class IterListItem<PrevItemT, NodeVecT, /*VecSize=*/1, _Level>
{
public:
    using _NodeT = typename boost::mpl::front<NodeVecT>::type;
    /// The type of iterator stored in the previous list item
    using PrevIterT = typename PrevItemT::IterT;
    /// The type of iterator stored in this list item (e.g., RootNode::ValueOnCIter)
    using IterT = typename IterTraits<typename PrevIterT::NonConstNodeType, PrevIterT>::template
        NodeConverter<_NodeT>::Type;

    /// The type of node over which IterT iterates (e.g., const RootNode<...>)
    using NodeT = typename IterT::NodeType;
    /// The type of the node with const qualifiers removed ("Non-Const")
    using NCNodeT = typename IterT::NonConstNodeType;
    /// The type of value (with const qualifiers removed) to which the iterator points
    using NCValueT = typename IterT::NonConstValueType;
    /// NodeT's child node type, with the same constness (e.g., const InternalNode<...>)
    using ChildT = typename CopyConstness<NodeT, typename NodeT::ChildNodeType>::Type;
    /// NodeT's child node type with const qualifiers removed
    using NCChildT = typename CopyConstness<NCNodeT, typename NCNodeT::ChildNodeType>::Type;
    using ITraits = IterTraits<NCNodeT, IterT>;
    /// NodeT's level in its tree (0 = LeafNode)
    static const Index Level = _Level;

    IterListItem(PrevItemT* prev): mPrev(prev) {}

    IterListItem(const IterListItem& other): mIter(other.mIter), mPrev(nullptr) {}
    IterListItem& operator=(const IterListItem& other)
    {
        if (&other != this) {
            mIter = other.mIter;
            mPrev = nullptr; ///< @note external call to updateBackPointers() required
        }
        return *this;
    }

    void updateBackPointers(PrevItemT* prev) { mPrev = prev; }

    // The following method specializations differ from the default template
    // implementations mainly in that they don't forward.

    void setIter(const IterT& iter) { mIter = iter; }

    void getNode(Index lvl, NodeT*& node) const
    {
        node = (lvl <= Level) ? mIter.getParentNode() : nullptr;
    }

    template<typename OtherIterListItemT>
    void initLevel(Index lvl, OtherIterListItemT& otherListItem)
    {
        if (lvl == Level) {
            const NodeT* node = nullptr;
            otherListItem.getNode(lvl, node);
            mIter = (node == nullptr) ? IterT() : ITraits::begin(*const_cast<NodeT*>(node));
        }
    }

    Index pos(Index lvl) const { return (lvl == Level) ? mIter.pos() : Index(-1); }

    bool test(Index lvl) const { return (lvl == Level) ? mIter.test() : false; }

    bool next(Index lvl) { return (lvl == Level) ? mIter.next() : false; }

    bool down(Index lvl)
    {
        if (lvl == Level && mPrev != nullptr && mIter) {
            if (ChildT* child = ITraits::template getChild<ChildT>(mIter)) {
                mPrev->setIter(PrevItemT::ITraits::begin(*child));
                return true;
            }
        }
        return false;
    }

    Coord getCoord(Index lvl) const { return (lvl == Level) ? mIter.getCoord() : Coord(); }
    Index getChildDim(Index lvl) const { return (lvl == Level) ? NodeT::getChildDim() : 0; }
    Index64 getVoxelCount(Index lvl) const { return (lvl == Level) ? ChildT::NUM_VOXELS : 0; }

    bool isValueOn(Index lvl) const { return (lvl == Level) ? mIter.isValueOn() : false; }

    const NCValueT& getValue(Index lvl) const
    {
        assert(lvl == Level);
        (void)lvl; // avoid unused variable warning in optimized builds
        return mIter.getValue();
    }

    void setValue(Index lvl, const NCValueT& val) const { if (lvl == Level) mIter.setValue(val); }
    void setValueOn(Index lvl, bool on = true) const { if (lvl == Level) mIter.setValueOn(on); }
    void setValueOff(Index lvl) const { if (lvl == Level) mIter.setValueOff(); }

    template<typename ModifyOp>
    void modifyValue(Index lvl, const ModifyOp& op) const
    {
        if (lvl == Level) mIter.modifyValue(op);
    }

private:
    IterT mIter;
    PrevItemT* mPrev;
};


////////////////////////////////////////


//#define DEBUG_TREE_VALUE_ITERATOR

/// @brief Base class for tree-traversal iterators over tile and voxel values
template<typename _TreeT, typename _ValueIterT>
class TreeValueIteratorBase
{
public:
    using TreeT = _TreeT;
    using ValueIterT = _ValueIterT;
    using NodeT = typename ValueIterT::NodeType;
    using ValueT = typename ValueIterT::NonConstValueType;
    using ChildOnIterT = typename NodeT::ChildOnCIter;
    static const Index ROOT_LEVEL = NodeT::LEVEL;
    static_assert(ValueIterT::NodeType::LEVEL == ROOT_LEVEL, "invalid value iterator node type");
    static const Index LEAF_LEVEL = 0, ROOT_DEPTH = 0, LEAF_DEPTH = ROOT_LEVEL;

    TreeValueIteratorBase(TreeT&);

    TreeValueIteratorBase(const TreeValueIteratorBase& other);
    TreeValueIteratorBase& operator=(const TreeValueIteratorBase& other);

    /// Specify the depth of the highest level of the tree to which to ascend (depth 0 = root).
    void setMinDepth(Index minDepth);
    /// Return the depth of the highest level of the tree to which this iterator ascends.
    Index getMinDepth() const { return ROOT_LEVEL - Index(mMaxLevel); }
    /// Specify the depth of the lowest level of the tree to which to descend (depth 0 = root).
    void setMaxDepth(Index maxDepth);
    /// Return the depth of the lowest level of the tree to which this iterator ascends.
    Index getMaxDepth() const { return ROOT_LEVEL - Index(mMinLevel); }

    //@{
    /// Return @c true if this iterator is not yet exhausted.
    bool test() const { return mValueIterList.test(mLevel); }
    operator bool() const { return this->test(); }
    //@}

    /// @brief Advance to the next tile or voxel value.
    /// Return @c true if this iterator is not yet exhausted.
    bool next();
    /// Advance to the next tile or voxel value.
    TreeValueIteratorBase& operator++() { this->next(); return *this; }

    /// @brief Return the level in the tree (0 = leaf) of the node to which
    /// this iterator is currently pointing.
    Index getLevel() const { return mLevel; }
    /// @brief Return the depth in the tree (0 = root) of the node to which
    /// this iterator is currently pointing.
    Index getDepth() const { return ROOT_LEVEL - mLevel; }
    static Index getLeafDepth() { return LEAF_DEPTH; }

    /// @brief Return in @a node a pointer to the node over which this iterator is
    /// currently iterating or one of that node's parents, as determined by @a NodeType.
    /// @return a null pointer if @a NodeType specifies a node at a lower level
    /// of the tree than that given by getLevel().
    template<typename NodeType>
    void getNode(NodeType*& node) const { mValueIterList.getNode(mLevel, node); }

    /// @brief Return the global coordinates of the voxel or tile to which
    /// this iterator is currently pointing.
    Coord getCoord() const { return mValueIterList.getCoord(mLevel); }
    /// @brief Return in @a bbox the axis-aligned bounding box of
    /// the voxel or tile to which this iterator is currently pointing.
    /// @return false if the bounding box is empty.
    bool getBoundingBox(CoordBBox&) const;
    /// @brief Return the axis-aligned bounding box of the voxel or tile to which
    /// this iterator is currently pointing.
    CoordBBox getBoundingBox() const { CoordBBox b; this->getBoundingBox(b); return b; }

    /// Return the number of (virtual) voxels corresponding to the value
    Index64 getVoxelCount() const { return mValueIterList.getVoxelCount(mLevel);}

    /// Return @c true if this iterator is currently pointing to a (non-leaf) tile value.
    bool isTileValue() const { return mLevel != 0 && this->test(); }
    /// Return @c true if this iterator is currently pointing to a (leaf) voxel value.
    bool isVoxelValue() const { return mLevel == 0 && this->test(); }
    /// Return @c true if the value to which this iterator is currently pointing is active.
    bool isValueOn() const { return mValueIterList.isValueOn(mLevel); }

    //@{
    /// Return the tile or voxel value to which this iterator is currently pointing.
    const ValueT& getValue() const { return mValueIterList.getValue(mLevel); }
    const ValueT& operator*() const { return this->getValue(); }
    const ValueT* operator->() const { return &(this->operator*()); }
    //@}

    /// @brief Change the tile or voxel value to which this iterator is currently pointing
    /// and mark it as active.
    void setValue(const ValueT& val) const { mValueIterList.setValue(mLevel, val); }
    /// @brief Change the active/inactive state of the tile or voxel value to which
    /// this iterator is currently pointing.
    void setActiveState(bool on) const { mValueIterList.setValueOn(mLevel, on); }
    /// Mark the tile or voxel value to which this iterator is currently pointing as inactive.
    void setValueOff() const { mValueIterList.setValueOff(mLevel); }

    /// @brief Apply a functor to the item to which this iterator is pointing.
    /// (Not valid for const iterators.)
    /// @param op  a functor of the form <tt>void op(ValueType&) const</tt> that modifies
    ///            its argument in place
    /// @see Tree::modifyValue()
    template<typename ModifyOp>
    void modifyValue(const ModifyOp& op) const { mValueIterList.modifyValue(mLevel, op); }

    /// Return a pointer to the tree over which this iterator is iterating.
    TreeT* getTree() const { return mTree; }

    /// Return a string (for debugging, mainly) describing this iterator's current state.
    std::string summary() const;

private:
    bool advance(bool dontIncrement = false);

    using InvTreeT = typename iter::InvertedTree<NodeT, NodeT::LEVEL>::Type;
    struct PrevChildItem { using IterT = ChildOnIterT; };
    struct PrevValueItem { using IterT = ValueIterT; };

    IterListItem<PrevChildItem, InvTreeT, /*VecSize=*/ROOT_LEVEL+1, /*Level=*/0> mChildIterList;
    IterListItem<PrevValueItem, InvTreeT, /*VecSize=*/ROOT_LEVEL+1, /*Level=*/0> mValueIterList;
    Index mLevel;
    int mMinLevel, mMaxLevel;
    TreeT* mTree;
}; // class TreeValueIteratorBase


template<typename TreeT, typename ValueIterT>
inline
TreeValueIteratorBase<TreeT, ValueIterT>::TreeValueIteratorBase(TreeT& tree):
    mChildIterList(nullptr),
    mValueIterList(nullptr),
    mLevel(ROOT_LEVEL),
    mMinLevel(int(LEAF_LEVEL)),
    mMaxLevel(int(ROOT_LEVEL)),
    mTree(&tree)
{
    mChildIterList.setIter(IterTraits<NodeT, ChildOnIterT>::begin(tree.root()));
    mValueIterList.setIter(IterTraits<NodeT, ValueIterT>::begin(tree.root()));
    this->advance(/*dontIncrement=*/true);
}


template<typename TreeT, typename ValueIterT>
inline
TreeValueIteratorBase<TreeT, ValueIterT>::TreeValueIteratorBase(const TreeValueIteratorBase& other):
    mChildIterList(other.mChildIterList),
    mValueIterList(other.mValueIterList),
    mLevel(other.mLevel),
    mMinLevel(other.mMinLevel),
    mMaxLevel(other.mMaxLevel),
    mTree(other.mTree)
{
    mChildIterList.updateBackPointers();
    mValueIterList.updateBackPointers();
}


template<typename TreeT, typename ValueIterT>
inline TreeValueIteratorBase<TreeT, ValueIterT>&
TreeValueIteratorBase<TreeT, ValueIterT>::operator=(const TreeValueIteratorBase& other)
{
    if (&other != this) {
        mChildIterList = other.mChildIterList;
        mValueIterList = other.mValueIterList;
        mLevel = other.mLevel;
        mMinLevel = other.mMinLevel;
        mMaxLevel = other.mMaxLevel;
        mTree = other.mTree;
        mChildIterList.updateBackPointers();
        mValueIterList.updateBackPointers();
    }
    return *this;
}


template<typename TreeT, typename ValueIterT>
inline void
TreeValueIteratorBase<TreeT, ValueIterT>::setMinDepth(Index minDepth)
{
    mMaxLevel = int(ROOT_LEVEL - minDepth); // level = ROOT_LEVEL - depth
    if (int(mLevel) > mMaxLevel) this->next();
}


template<typename TreeT, typename ValueIterT>
inline void
TreeValueIteratorBase<TreeT, ValueIterT>::setMaxDepth(Index maxDepth)
{
    // level = ROOT_LEVEL - depth
    mMinLevel = int(ROOT_LEVEL - std::min(maxDepth, this->getLeafDepth()));
    if (int(mLevel) < mMinLevel) this->next();
}


template<typename TreeT, typename ValueIterT>
inline bool
TreeValueIteratorBase<TreeT, ValueIterT>::next()
{
    do {
        if (!this->advance()) return false;
    } while (int(mLevel) < mMinLevel || int(mLevel) > mMaxLevel);
    return true;
}


template<typename TreeT, typename ValueIterT>
inline bool
TreeValueIteratorBase<TreeT, ValueIterT>::advance(bool dontIncrement)
{
    bool recurse = false;
    do {
        recurse = false;
        Index
            vPos = mValueIterList.pos(mLevel),
            cPos = mChildIterList.pos(mLevel);
        if (vPos == cPos && mChildIterList.test(mLevel)) {
            /// @todo Once ValueOff iterators properly skip child pointers, remove this block.
            mValueIterList.next(mLevel);
            vPos = mValueIterList.pos(mLevel);
        }
        if (vPos < cPos) {
            if (dontIncrement) return true;
            if (mValueIterList.next(mLevel)) {
                if (mValueIterList.pos(mLevel) == cPos && mChildIterList.test(mLevel)) {
                    /// @todo Once ValueOff iterators properly skip child pointers,
                    /// remove this block.
                    mValueIterList.next(mLevel);
                }
                // If there is a next value and it precedes the next child, return.
                if (mValueIterList.pos(mLevel) < cPos) return true;
            }
        } else {
            // Advance to the next child, which may or may not precede the next value.
            if (!dontIncrement) mChildIterList.next(mLevel);
        }
#ifdef DEBUG_TREE_VALUE_ITERATOR
        std::cout << "\n" << this->summary() << std::flush;
#endif

        // Descend to the lowest level at which the next value precedes the next child.
        while (mChildIterList.pos(mLevel) < mValueIterList.pos(mLevel)) {
#ifdef ENABLE_TREE_VALUE_DEPTH_BOUND_OPTIMIZATION
            if (int(mLevel) == mMinLevel) {
                // If the current node lies at the lowest allowed level, none of its
                // children can be visited, so just advance its child iterator.
                mChildIterList.next(mLevel);
                if (mValueIterList.pos(mLevel) == mChildIterList.pos(mLevel)
                    && mChildIterList.test(mLevel))
                {
                    /// @todo Once ValueOff iterators properly skip child pointers,
                    /// remove this block.
                    mValueIterList.next(mLevel);
                }
            } else
#endif
                if (mChildIterList.down(mLevel)) {
                    --mLevel; // descend one level
                    mValueIterList.initLevel(mLevel, mChildIterList);
                    if (mValueIterList.pos(mLevel) == mChildIterList.pos(mLevel)
                        && mChildIterList.test(mLevel))
                    {
                        /// @todo Once ValueOff iterators properly skip child pointers,
                        /// remove this block.
                        mValueIterList.next(mLevel);
                    }
                } else break;
#ifdef DEBUG_TREE_VALUE_ITERATOR
            std::cout << "\n" << this->summary() << std::flush;
#endif
        }
        // Ascend to the nearest level at which one of the iterators is not yet exhausted.
        while (!mChildIterList.test(mLevel) && !mValueIterList.test(mLevel)) {
            if (mLevel == ROOT_LEVEL) return false;
            ++mLevel;
            mChildIterList.next(mLevel);
            dontIncrement = true;
            recurse = true;
        }
    } while (recurse);
    return true;
}


template<typename TreeT, typename ValueIterT>
inline bool
TreeValueIteratorBase<TreeT, ValueIterT>::getBoundingBox(CoordBBox& bbox) const
{
    if (!this->test()) {
        bbox = CoordBBox();
        return false;
    }
    bbox.min() = mValueIterList.getCoord(mLevel);
    bbox.max() = bbox.min().offsetBy(mValueIterList.getChildDim(mLevel) - 1);
    return true;
}


template<typename TreeT, typename ValueIterT>
inline std::string
TreeValueIteratorBase<TreeT, ValueIterT>::summary() const
{
    std::ostringstream ostr;
    for (int lvl = int(ROOT_LEVEL); lvl >= 0 && lvl >= int(mLevel); --lvl) {
        if (lvl == 0) ostr << "leaf";
        else if (lvl == int(ROOT_LEVEL)) ostr << "root";
        else ostr << "int" << (ROOT_LEVEL - lvl);
        ostr << " v" << mValueIterList.pos(lvl)
            << " c" << mChildIterList.pos(lvl);
        if (lvl > int(mLevel)) ostr << " / ";
    }
    if (this->test() && mValueIterList.pos(mLevel) < mChildIterList.pos(mLevel)) {
        if (mLevel == 0) {
            ostr << " " << this->getCoord();
        } else {
            ostr << " " << this->getBoundingBox();
        }
    }
    return ostr.str();
}


////////////////////////////////////////


/// @brief Base class for tree-traversal iterators over all nodes
template<typename _TreeT, typename RootChildOnIterT>
class NodeIteratorBase
{
public:
    using TreeT = _TreeT;
    using RootIterT = RootChildOnIterT;
    using RootNodeT = typename RootIterT::NodeType;
    using NCRootNodeT = typename RootIterT::NonConstNodeType;
    static const Index ROOT_LEVEL = RootNodeT::LEVEL;
    using InvTreeT = typename iter::InvertedTree<NCRootNodeT, ROOT_LEVEL>::Type;
    static const Index LEAF_LEVEL = 0, ROOT_DEPTH = 0, LEAF_DEPTH = ROOT_LEVEL;

    using RootIterTraits = IterTraits<NCRootNodeT, RootIterT>;

    NodeIteratorBase();
    NodeIteratorBase(TreeT&);

    NodeIteratorBase(const NodeIteratorBase& other);
    NodeIteratorBase& operator=(const NodeIteratorBase& other);

    /// Specify the depth of the highest level of the tree to which to ascend (depth 0 = root).
    void setMinDepth(Index minDepth);
    /// Return the depth of the highest level of the tree to which this iterator ascends.
    Index getMinDepth() const { return ROOT_LEVEL - Index(mMaxLevel); }
    /// Specify the depth of the lowest level of the tree to which to descend (depth 0 = root).
    void setMaxDepth(Index maxDepth);
    /// Return the depth of the lowest level of the tree to which this iterator ascends.
    Index getMaxDepth() const { return ROOT_LEVEL - Index(mMinLevel); }

    //@{
    /// Return @c true if this iterator is not yet exhausted.
    bool test() const { return !mDone; }
    operator bool() const { return this->test(); }
    //@}

    /// @brief Advance to the next tile or voxel value.
    /// @return @c true if this iterator is not yet exhausted.
    bool next();
    /// Advance the iterator to the next leaf node.
    void increment() { this->next(); }
    NodeIteratorBase& operator++() { this->increment(); return *this; }
    /// Increment the iterator n times.
    void increment(Index n) { for (Index i = 0; i < n && this->next(); ++i) {} }

    /// @brief Return the level in the tree (0 = leaf) of the node to which
    /// this iterator is currently pointing.
    Index getLevel() const { return mLevel; }
    /// @brief Return the depth in the tree (0 = root) of the node to which
    /// this iterator is currently pointing.
    Index getDepth() const { return ROOT_LEVEL - mLevel; }
    static Index getLeafDepth() { return LEAF_DEPTH; }

    /// @brief Return the global coordinates of the voxel or tile to which
    /// this iterator is currently pointing.
    Coord getCoord() const;
    /// @brief Return in @a bbox the axis-aligned bounding box of
    /// the voxel or tile to which this iterator is currently pointing.
    /// @return false if the bounding box is empty.
    bool getBoundingBox(CoordBBox& bbox) const;
    /// @brief Return the axis-aligned bounding box of the voxel or tile to which
    /// this iterator is currently pointing.
    CoordBBox getBoundingBox() const { CoordBBox b; this->getBoundingBox(b); return b; }

    //@{
    /// @brief Return the node to which the iterator is pointing.
    /// @note This iterator doesn't have the usual dereference operators (* and ->),
    /// because they would have to be overloaded by the returned node type.
    template<typename NodeT>
    void getNode(NodeT*& node) const { node = nullptr; mIterList.getNode(mLevel, node); }
    template<typename NodeT>
    void getNode(const NodeT*& node) const { node = nullptr; mIterList.getNode(mLevel, node); }
    //@}

    TreeT* getTree() const { return mTree; }

    std::string summary() const;

private:
    struct PrevItem { using IterT = RootIterT; };

    IterListItem<PrevItem, InvTreeT, /*VecSize=*/ROOT_LEVEL+1, LEAF_LEVEL> mIterList;
    Index mLevel;
    int mMinLevel, mMaxLevel;
    bool mDone;
    TreeT* mTree;
}; // class NodeIteratorBase


template<typename TreeT, typename RootChildOnIterT>
inline
NodeIteratorBase<TreeT, RootChildOnIterT>::NodeIteratorBase():
    mIterList(nullptr),
    mLevel(ROOT_LEVEL),
    mMinLevel(int(LEAF_LEVEL)),
    mMaxLevel(int(ROOT_LEVEL)),
    mDone(true),
    mTree(nullptr)
{
}


template<typename TreeT, typename RootChildOnIterT>
inline
NodeIteratorBase<TreeT, RootChildOnIterT>::NodeIteratorBase(TreeT& tree):
    mIterList(nullptr),
    mLevel(ROOT_LEVEL),
    mMinLevel(int(LEAF_LEVEL)),
    mMaxLevel(int(ROOT_LEVEL)),
    mDone(false),
    mTree(&tree)
{
    mIterList.setIter(RootIterTraits::begin(tree.root()));
}


template<typename TreeT, typename RootChildOnIterT>
inline
NodeIteratorBase<TreeT, RootChildOnIterT>::NodeIteratorBase(const NodeIteratorBase& other):
    mIterList(other.mIterList),
    mLevel(other.mLevel),
    mMinLevel(other.mMinLevel),
    mMaxLevel(other.mMaxLevel),
    mDone(other.mDone),
    mTree(other.mTree)
{
    mIterList.updateBackPointers();
}


template<typename TreeT, typename RootChildOnIterT>
inline NodeIteratorBase<TreeT, RootChildOnIterT>&
NodeIteratorBase<TreeT, RootChildOnIterT>::operator=(const NodeIteratorBase& other)
{
    if (&other != this) {
        mLevel = other.mLevel;
        mMinLevel = other.mMinLevel;
        mMaxLevel = other.mMaxLevel;
        mDone = other.mDone;
        mTree = other.mTree;
        mIterList = other.mIterList;
        mIterList.updateBackPointers();
    }
    return *this;
}


template<typename TreeT, typename RootChildOnIterT>
inline void
NodeIteratorBase<TreeT, RootChildOnIterT>::setMinDepth(Index minDepth)
{
    mMaxLevel = int(ROOT_LEVEL - minDepth); // level = ROOT_LEVEL - depth
    if (int(mLevel) > mMaxLevel) this->next();
}


template<typename TreeT, typename RootChildOnIterT>
inline void
NodeIteratorBase<TreeT, RootChildOnIterT>::setMaxDepth(Index maxDepth)
{
    // level = ROOT_LEVEL - depth
    mMinLevel = int(ROOT_LEVEL - std::min(maxDepth, this->getLeafDepth()));
    if (int(mLevel) < mMinLevel) this->next();
}


template<typename TreeT, typename RootChildOnIterT>
inline bool
NodeIteratorBase<TreeT, RootChildOnIterT>::next()
{
    do {
        if (mDone) return false;

        // If the iterator over the current node points to a child,
        // descend to the child (depth-first traversal).
        if (int(mLevel) > mMinLevel && mIterList.test(mLevel)) {
            if (!mIterList.down(mLevel)) return false;
            --mLevel;
        } else {
            // Ascend to the nearest ancestor that has other children.
            while (!mIterList.test(mLevel)) {
                if (mLevel == ROOT_LEVEL) {
                    // Can't ascend higher than the root.
                    mDone = true;
                    return false;
                }
                ++mLevel; // ascend one level
                mIterList.next(mLevel); // advance to the next child, if there is one
            }
            // Descend to the child.
            if (!mIterList.down(mLevel)) return false;
            --mLevel;
        }
    } while (int(mLevel) < mMinLevel || int(mLevel) > mMaxLevel);
    return true;
}


template<typename TreeT, typename RootChildOnIterT>
inline Coord
NodeIteratorBase<TreeT, RootChildOnIterT>::getCoord() const
{
    if (mLevel != ROOT_LEVEL) return  mIterList.getCoord(mLevel + 1);
    RootNodeT* root = nullptr;
    this->getNode(root);
    return root ? root->getMinIndex() : Coord::min();
}


template<typename TreeT, typename RootChildOnIterT>
inline bool
NodeIteratorBase<TreeT, RootChildOnIterT>::getBoundingBox(CoordBBox& bbox) const
{
    if (mLevel == ROOT_LEVEL) {
        RootNodeT* root = nullptr;
        this->getNode(root);
        if (root == nullptr) {
            bbox = CoordBBox();
            return false;
        }
        root->getIndexRange(bbox);
        return true;
    }
    bbox.min() = mIterList.getCoord(mLevel + 1);
    bbox.max() = bbox.min().offsetBy(mIterList.getChildDim(mLevel + 1) - 1);
    return true;
}


template<typename TreeT, typename RootChildOnIterT>
inline std::string
NodeIteratorBase<TreeT, RootChildOnIterT>::summary() const
{
    std::ostringstream ostr;
    for (int lvl = int(ROOT_LEVEL); lvl >= 0 && lvl >= int(mLevel); --lvl) {
        if (lvl == 0) ostr << "leaf";
        else if (lvl == int(ROOT_LEVEL)) ostr << "root";
        else ostr << "int" << (ROOT_LEVEL - lvl);
        ostr << " c" << mIterList.pos(lvl);
        if (lvl > int(mLevel)) ostr << " / ";
    }
    CoordBBox bbox;
    this->getBoundingBox(bbox);
    ostr << " " << bbox;
    return ostr.str();
}


////////////////////////////////////////


/// @brief Base class for tree-traversal iterators over all leaf nodes (but not leaf voxels)
template<typename TreeT, typename RootChildOnIterT>
class LeafIteratorBase
{
public:
    using RootIterT = RootChildOnIterT;
    using RootNodeT = typename RootIterT::NodeType;
    using NCRootNodeT = typename RootIterT::NonConstNodeType;
    static const Index ROOT_LEVEL = RootNodeT::LEVEL;
    using InvTreeT = typename iter::InvertedTree<NCRootNodeT, ROOT_LEVEL>::Type;
    using NCLeafNodeT = typename boost::mpl::front<InvTreeT>::type;
    using LeafNodeT = typename CopyConstness<RootNodeT, NCLeafNodeT>::Type;
    static const Index LEAF_LEVEL = 0, LEAF_PARENT_LEVEL = LEAF_LEVEL + 1;

    using RootIterTraits = IterTraits<NCRootNodeT, RootIterT>;

    LeafIteratorBase(): mIterList(nullptr), mTree(nullptr) {}

    LeafIteratorBase(TreeT& tree): mIterList(nullptr), mTree(&tree)
    {
        // Initialize the iterator list with a root node iterator.
        mIterList.setIter(RootIterTraits::begin(tree.root()));
        // Descend along the first branch, initializing the node iterator at each level.
        Index lvl = ROOT_LEVEL;
        for ( ; lvl > 0 && mIterList.down(lvl); --lvl) {}
        // If the first branch terminated above the leaf level, backtrack to the next branch.
        if (lvl > 0) this->next();
    }

    LeafIteratorBase(const LeafIteratorBase& other): mIterList(other.mIterList), mTree(other.mTree)
    {
        mIterList.updateBackPointers();
    }
    LeafIteratorBase& operator=(const LeafIteratorBase& other)
    {
        if (&other != this) {
            mTree = other.mTree;
            mIterList = other.mIterList;
            mIterList.updateBackPointers();
        }
        return *this;
    }

    //@{
    /// Return the leaf node to which the iterator is pointing.
    LeafNodeT* getLeaf() const
    {
        LeafNodeT* n = nullptr;
        mIterList.getNode(LEAF_LEVEL, n);
        return n;
    }
    LeafNodeT& operator*() const { return *this->getLeaf(); }
    LeafNodeT* operator->() const { return this->getLeaf(); }
    //@}

    bool test() const { return mIterList.test(LEAF_PARENT_LEVEL); }
    operator bool() const { return this->test(); }

    //@{
    /// Advance the iterator to the next leaf node.
    bool next();
    void increment() { this->next(); }
    LeafIteratorBase& operator++() { this->increment(); return *this; }
    //@}
    /// Increment the iterator n times.
    void increment(Index n) { for (Index i = 0; i < n && this->next(); ++i) {} }

    TreeT* getTree() const { return mTree; }

private:
    struct PrevItem { using IterT = RootIterT; };

    /// @note Even though a LeafIterator doesn't iterate over leaf voxels,
    /// the first item of this linked list of node iterators is a leaf node iterator,
    /// whose purpose is only to provide access to its parent leaf node.
    IterListItem<PrevItem, InvTreeT, /*VecSize=*/ROOT_LEVEL+1, LEAF_LEVEL> mIterList;
    TreeT* mTree;
}; // class LeafIteratorBase


template<typename TreeT, typename RootChildOnIterT>
inline bool
LeafIteratorBase<TreeT, RootChildOnIterT>::next()
{
    // If the iterator is valid for the current node one level above the leaf level,
    // advance the iterator to the node's next child.
    if (mIterList.test(LEAF_PARENT_LEVEL) && mIterList.next(LEAF_PARENT_LEVEL)) {
        mIterList.down(LEAF_PARENT_LEVEL); // initialize the leaf iterator
        return true;
    }

    Index lvl = LEAF_PARENT_LEVEL;
    while (!mIterList.test(LEAF_PARENT_LEVEL)) {
        if (mIterList.test(lvl)) {
            mIterList.next(lvl);
        } else {
            do {
                // Ascend to the nearest level at which
                // one of the iterators is not yet exhausted.
                if (lvl == ROOT_LEVEL) return false;
                ++lvl;
                if (mIterList.test(lvl)) mIterList.next(lvl);
            } while (!mIterList.test(lvl));
        }
        // Descend to the lowest child, but not as far as the leaf iterator.
        while (lvl > LEAF_PARENT_LEVEL && mIterList.down(lvl)) --lvl;
    }
    mIterList.down(LEAF_PARENT_LEVEL); // initialize the leaf iterator
    return true;
}


////////////////////////////////////////


/// An IteratorRange wraps a tree or node iterator, giving the iterator TBB
/// splittable range semantics.
template<typename IterT>
class IteratorRange
{
public:
    IteratorRange(const IterT& iter, size_t grainSize = 8):
        mIter(iter),
        mGrainSize(grainSize),
        mSize(0)
    {
        mSize = this->size();
    }
    IteratorRange(IteratorRange& other, tbb::split):
        mIter(other.mIter),
        mGrainSize(other.mGrainSize),
        mSize(other.mSize >> 1)
    {
        other.increment(mSize);
    }

    /// @brief Return a reference to this range's iterator.
    /// @note The reference is const, because the iterator should not be
    /// incremented directly.  Use this range object's increment() instead.
    const IterT& iterator() const { return mIter; }

    bool empty() const { return mSize == 0 || !mIter.test(); }
    bool test() const { return !this->empty(); }
    operator bool() const { return !this->empty(); }

    /// @brief Return @c true if this range is splittable (i.e., if the iterator
    /// can be advanced more than mGrainSize times).
    bool is_divisible() const { return mSize > mGrainSize; }

    /// Advance the iterator @a n times.
    void increment(Index n = 1) { for ( ; n > 0 && mSize > 0; --n, --mSize, ++mIter) {} }
    /// Advance the iterator to the next item.
    IteratorRange& operator++() { this->increment(); return *this; }
    /// @brief Advance the iterator to the next item.
    /// @return @c true if the iterator is not yet exhausted.
    bool next() { this->increment(); return this->test(); }

private:
    Index size() const { Index n = 0; for (IterT it(mIter); it.test(); ++n, ++it) {} return n; }

    IterT mIter;
    size_t mGrainSize;
    /// @note mSize is only an estimate of the number of times mIter can be incremented
    /// before it is exhausted (because the topology of the underlying tree could change
    /// during iteration).  For the purpose of range splitting, though, that should be
    /// sufficient, since the two halves need not be of exactly equal size.
    Index mSize;
};


////////////////////////////////////////


/// @brief Base class for tree-traversal iterators over real and virtual voxel values
/// @todo class TreeVoxelIteratorBase;

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_TREEITERATOR_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
