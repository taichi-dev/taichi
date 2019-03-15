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

/// @file tree/NodeManager.h
///
/// @author Ken Museth
///
/// @brief NodeManager produces linear arrays of all tree nodes
/// allowing for efficient threading and bottom-up processing.
///
/// @note A NodeManager can be constructed from a Tree or LeafManager.
/// The latter is slightly more efficient since the cached leaf nodes will be reused.

#ifndef OPENVDB_TREE_NODEMANAGER_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_NODEMANAGER_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <deque>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

// Produce linear arrays of all tree nodes, to facilitate efficient threading
// and bottom-up processing.
template<typename TreeOrLeafManagerT, Index LEVELS = TreeOrLeafManagerT::RootNodeType::LEVEL>
class NodeManager;


////////////////////////////////////////


/// @brief This class caches tree nodes of a specific type in a linear array.
///
/// @note It is for internal use and should rarely be used directly.
template<typename NodeT>
class NodeList
{
public:
    using value_type = NodeT*;
    using ListT = std::deque<value_type>;

    NodeList() {}

    void push_back(NodeT* node) { mList.push_back(node); }

    NodeT& operator()(size_t n) const { assert(n<mList.size()); return *(mList[n]); }

    NodeT*& operator[](size_t n) { assert(n<mList.size()); return mList[n]; }

    Index64 nodeCount() const { return mList.size(); }

    void clear() { mList.clear(); }

    void resize(size_t n) { mList.resize(n); }

    class NodeRange
    {
    public:

        NodeRange(size_t begin, size_t end, const NodeList& nodeList, size_t grainSize=1):
            mEnd(end), mBegin(begin), mGrainSize(grainSize), mNodeList(nodeList) {}

        NodeRange(NodeRange& r, tbb::split):
            mEnd(r.mEnd), mBegin(doSplit(r)), mGrainSize(r.mGrainSize),
            mNodeList(r.mNodeList) {}

        size_t size() const { return mEnd - mBegin; }

        size_t grainsize() const { return mGrainSize; }

        const NodeList& nodeList() const { return mNodeList; }

        bool empty() const {return !(mBegin < mEnd);}

        bool is_divisible() const {return mGrainSize < this->size();}

        class Iterator
        {
        public:
            Iterator(const NodeRange& range, size_t pos): mRange(range), mPos(pos)
            {
                assert(this->isValid());
            }
            Iterator(const Iterator&) = default;
            Iterator& operator=(const Iterator&) = default;
            /// Advance to the next node.
            Iterator& operator++() { ++mPos; return *this; }
            /// Return a reference to the node to which this iterator is pointing.
            NodeT& operator*() const { return mRange.mNodeList(mPos); }
            /// Return a pointer to the node to which this iterator is pointing.
            NodeT* operator->() const { return &(this->operator*()); }
            /// Return the index into the list of the current node.
            size_t pos() const { return mPos; }
            bool isValid() const { return mPos>=mRange.mBegin && mPos<=mRange.mEnd; }
            /// Return @c true if this iterator is not yet exhausted.
            bool test() const { return mPos < mRange.mEnd; }
            /// Return @c true if this iterator is not yet exhausted.
            operator bool() const { return this->test(); }
            /// Return @c true if this iterator is exhausted.
            bool empty() const { return !this->test(); }
            bool operator!=(const Iterator& other) const
            {
                return (mPos != other.mPos) || (&mRange != &other.mRange);
            }
            bool operator==(const Iterator& other) const { return !(*this != other); }
            const NodeRange& nodeRange() const { return mRange; }

        private:
            const NodeRange& mRange;
            size_t mPos;
        };// NodeList::NodeRange::Iterator

        Iterator begin() const {return Iterator(*this, mBegin);}

        Iterator end() const {return Iterator(*this, mEnd);}

    private:
        size_t mEnd, mBegin, mGrainSize;
        const NodeList& mNodeList;

        static size_t doSplit(NodeRange& r)
        {
            assert(r.is_divisible());
            size_t middle = r.mBegin + (r.mEnd - r.mBegin) / 2u;
            r.mEnd = middle;
            return middle;
        }
    };// NodeList::NodeRange

    /// Return a TBB-compatible NodeRange.
    NodeRange nodeRange(size_t grainsize = 1) const
    {
        return NodeRange(0, this->nodeCount(), *this, grainsize);
    }

    template<typename NodeOp>
    void foreach(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        NodeTransformer<NodeOp> transform(op);
        transform.run(this->nodeRange(grainSize), threaded);
    }

    template<typename NodeOp>
    void reduce(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        NodeReducer<NodeOp> transform(op);
        transform.run(this->nodeRange(grainSize), threaded);
    }

private:

    // Private struct of NodeList that performs parallel_for
    template<typename NodeOp>
    struct NodeTransformer
    {
        NodeTransformer(const NodeOp& nodeOp) : mNodeOp(nodeOp)
        {
        }
        void run(const NodeRange& range, bool threaded = true)
        {
            threaded ? tbb::parallel_for(range, *this) : (*this)(range);
        }
        void operator()(const NodeRange& range) const
        {
            for (typename NodeRange::Iterator it = range.begin(); it; ++it) mNodeOp(*it);
        }
        const NodeOp mNodeOp;
    };// NodeList::NodeTransformer

    // Private struct of NodeList that performs parallel_reduce
    template<typename NodeOp>
    struct NodeReducer
    {
        NodeReducer(NodeOp& nodeOp) : mNodeOp(&nodeOp), mOwnsOp(false)
        {
        }
        NodeReducer(const NodeReducer& other, tbb::split) :
            mNodeOp(new NodeOp(*(other.mNodeOp), tbb::split())), mOwnsOp(true)
        {
        }
        ~NodeReducer() { if (mOwnsOp) delete mNodeOp; }
        void run(const NodeRange& range, bool threaded = true)
        {
            threaded ? tbb::parallel_reduce(range, *this) : (*this)(range);
        }
        void operator()(const NodeRange& range)
        {
            NodeOp &op = *mNodeOp;
            for (typename NodeRange::Iterator it = range.begin(); it; ++it) op(*it);
        }
        void join(const NodeReducer& other)
        {
            mNodeOp->join(*(other.mNodeOp));
        }
        NodeOp *mNodeOp;
        const bool mOwnsOp;
    };// NodeList::NodeReducer


protected:
    ListT mList;
};// NodeList


/////////////////////////////////////////////


/// @brief This class is a link in a chain that each caches tree nodes
/// of a specific type in a linear array.
///
/// @note It is for internal use and should rarely be used directly.
template<typename NodeT, Index LEVEL>
class NodeManagerLink
{
public:
    NodeManagerLink() {}

    virtual ~NodeManagerLink() {}

    void clear() { mList.clear(); mNext.clear(); }

    template<typename ParentT, typename TreeOrLeafManagerT>
    void init(ParentT& parent, TreeOrLeafManagerT& tree)
    {
        parent.getNodes(mList);
        for (size_t i=0, n=mList.nodeCount(); i<n; ++i) mNext.init(mList(i), tree);
    }

    template<typename ParentT>
    void rebuild(ParentT& parent)
    {
        mList.clear();
        parent.getNodes(mList);
        for (size_t i=0, n=mList.nodeCount(); i<n; ++i) mNext.rebuild(mList(i));
    }

    Index64 nodeCount() const { return mList.nodeCount() + mNext.nodeCount(); }

    Index64 nodeCount(Index i) const
    {
        return i==NodeT::LEVEL ? mList.nodeCount() : mNext.nodeCount(i);
    }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded, size_t grainSize)
    {
        mNext.foreachBottomUp(op, threaded, grainSize);
        mList.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded, size_t grainSize)
    {
        mList.foreach(op, threaded, grainSize);
        mNext.foreachTopDown(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded, size_t grainSize)
    {
        mNext.reduceBottomUp(op, threaded, grainSize);
        mList.reduce(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded, size_t grainSize)
    {
        mList.reduce(op, threaded, grainSize);
        mNext.reduceTopDown(op, threaded, grainSize);
    }

protected:
    NodeList<NodeT> mList;
    NodeManagerLink<typename NodeT::ChildNodeType, LEVEL-1> mNext;
};// NodeManagerLink class


////////////////////////////////////////


/// @private
/// @brief Specialization that terminates the chain of cached tree nodes
/// @note It is for internal use and should rarely be used directly.
template<typename NodeT>
class NodeManagerLink<NodeT, 0>
{
public:
    NodeManagerLink() {}

    virtual ~NodeManagerLink() {}

    /// @brief Clear all the cached tree nodes
    void clear() { mList.clear(); }

    template<typename ParentT>
    void rebuild(ParentT& parent) { mList.clear(); parent.getNodes(mList); }

    Index64 nodeCount() const { return mList.nodeCount(); }

    Index64 nodeCount(Index) const { return mList.nodeCount(); }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded, size_t grainSize)
    {
        mList.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded, size_t grainSize)
    {
        mList.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded, size_t grainSize)
    {
        mList.reduce(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded, size_t grainSize)
    {
        mList.reduce(op, threaded, grainSize);
    }

    template<typename ParentT, typename TreeOrLeafManagerT>
    void init(ParentT& parent, TreeOrLeafManagerT& tree)
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (TreeOrLeafManagerT::DEPTH == 2 && NodeT::LEVEL == 0) {
            tree.getNodes(mList);
        } else {
            parent.getNodes(mList);
        }
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
protected:
    NodeList<NodeT> mList;
};// NodeManagerLink class


////////////////////////////////////////


/// @brief To facilitate threading over the nodes of a tree, cache
/// node pointers in linear arrays, one for each level of the tree.
///
/// @details This implementation works with trees of any depth, but
/// optimized specializations are provided for the most typical tree depths.
template<typename TreeOrLeafManagerT, Index _LEVELS>
class NodeManager
{
public:
    static const Index LEVELS = _LEVELS;
    static_assert(LEVELS > 0,
        "expected instantiation of template specialization"); // see specialization below
    using RootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    static_assert(RootNodeType::LEVEL >= LEVELS, "number of levels exceeds root node height");

    NodeManager(TreeOrLeafManagerT& tree) : mRoot(tree.root()) { mChain.init(mRoot, tree); }

    virtual ~NodeManager() {}

    /// @brief Clear all the cached tree nodes
    void clear() { mChain.clear(); }

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild() { mChain.rebuild(mRoot); }

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    /// @brief Return the total number of cached nodes (excluding the root node)
    Index64 nodeCount() const { return mChain.nodeCount(); }

    /// @brief Return the number of cached nodes at level @a i, where
    /// 0 corresponds to the lowest level.
    Index64 nodeCount(Index i) const { return mChain.nodeCount(i); }

    //@{
    /// @brief   Threaded method that applies a user-supplied functor
    ///          to all the nodes in the tree.
    ///
    /// @param op        user-supplied functor, see examples for interface details.
    /// @param threaded  optional toggle to disable threading, on by default.
    /// @param grainSize optional parameter to specify the grainsize
    ///                  for threading, one by default.
    ///
    /// @warning The functor object is deep-copied to create TBB tasks.
    ///
    /// @par Example:
    /// @code
    /// // Functor to offset all the inactive values of a tree. Note
    /// // this implementation also illustrates how different
    /// // computation can be applied to the different node types.
    /// template<typename TreeType>
    /// struct OffsetOp
    /// {
    ///     using ValueT = typename TreeT::ValueType;
    ///     using RootT = typename TreeT::RootNodeType;
    ///     using LeafT = typename TreeT::LeafNodeType;
    ///     OffsetOp(const ValueT& v) : mOffset(v) {}
    ///
    ///     // Processes the root node. Required by the NodeManager
    ///     void operator()(RootT& root) const
    ///     {
    ///         for (typename RootT::ValueOffIter i = root.beginValueOff(); i; ++i) *i += mOffset;
    ///     }
    ///     // Processes the leaf nodes. Required by the NodeManager
    ///     void operator()(LeafT& leaf) const
    ///     {
    ///         for (typename LeafT::ValueOffIter i = leaf.beginValueOff(); i; ++i) *i += mOffset;
    ///     }
    ///     // Processes the internal nodes. Required by the NodeManager
    ///     template<typename NodeT>
    ///     void operator()(NodeT& node) const
    ///     {
    ///         for (typename NodeT::ValueOffIter i = node.beginValueOff(); i; ++i) *i += mOffset;
    ///     }
    /// private:
    ///     const ValueT mOffset;
    /// };
    ///
    /// // usage:
    /// OffsetOp<FloatTree> op(3.0f);
    /// tree::NodeManager<FloatTree> nodes(tree);
    /// nodes.foreachBottomUp(op);
    ///
    /// // or if a LeafManager already exists
    /// using T = tree::LeafManager<FloatTree>;
    /// OffsetOp<T> op(3.0f);
    /// tree::NodeManager<T> nodes(leafManager);
    /// nodes.foreachBottomUp(op);
    ///
    /// @endcode
    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mChain.foreachBottomUp(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mChain.foreachTopDown(op, threaded, grainSize);
    }

    //@}

    //@{
    /// @brief   Threaded method that processes nodes with a user supplied functor
    ///
    /// @param op        user-supplied functor, see examples for interface details.
    /// @param threaded  optional toggle to disable threading, on by default.
    /// @param grainSize optional parameter to specify the grainsize
    ///                  for threading, one by default.
    ///
    /// @warning The functor object is deep-copied to create TBB tasks.
    ///
    /// @par Example:
    /// @code
    ///  // Functor to count nodes in a tree
    ///  template<typename TreeType>
    ///  struct NodeCountOp
    ///  {
    ///      NodeCountOp() : nodeCount(TreeType::DEPTH, 0), totalCount(0)
    ///      {
    ///      }
    ///      NodeCountOp(const NodeCountOp& other, tbb::split) :
    ///          nodeCount(TreeType::DEPTH, 0), totalCount(0)
    ///      {
    ///      }
    ///      void join(const NodeCountOp& other)
    ///      {
    ///          for (size_t i = 0; i < nodeCount.size(); ++i) {
    ///              nodeCount[i] += other.nodeCount[i];
    ///          }
    ///          totalCount += other.totalCount;
    ///      }
    ///      // do nothing for the root node
    ///      void operator()(const typename TreeT::RootNodeType& node)
    ///      {
    ///      }
    ///      // count the internal and leaf nodes
    ///      template<typename NodeT>
    ///      void operator()(const NodeT& node)
    ///      {
    ///          ++(nodeCount[NodeT::LEVEL]);
    ///          ++totalCount;
    ///      }
    ///      std::vector<openvdb::Index64> nodeCount;
    ///      openvdb::Index64 totalCount;
    /// };
    ///
    /// // usage:
    /// NodeCountOp<FloatTree> op;
    /// tree::NodeManager<FloatTree> nodes(tree);
    /// nodes.reduceBottomUp(op);
    ///
    /// // or if a LeafManager already exists
    /// NodeCountOp<FloatTree> op;
    /// using T = tree::LeafManager<FloatTree>;
    /// T leafManager(tree);
    /// tree::NodeManager<T> nodes(leafManager);
    /// nodes.reduceBottomUp(op);
    ///
    /// @endcode
    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mChain.reduceBottomUp(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mChain.reduceTopDown(op, threaded, grainSize);
    }
    //@}

protected:
    RootNodeType& mRoot;
    NodeManagerLink<typename RootNodeType::ChildNodeType, LEVELS-1> mChain;

private:
    NodeManager(const NodeManager&) {}//disallow copy-construction
};// NodeManager class


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with no caching of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 0>
{
public:
    using RootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    static const Index LEVELS = 0;

    NodeManager(TreeOrLeafManagerT& tree) : mRoot(tree.root()) {}

    virtual ~NodeManager() {}

    /// @brief Clear all the cached tree nodes
    void clear() {}

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild() {}

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    /// @brief Return the total number of cached nodes (excluding the root node)
    Index64 nodeCount() const { return 0; }

    Index64 nodeCount(Index) const { return 0; }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool, size_t) { op(mRoot); }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool, size_t) { op(mRoot); }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool, size_t) { op(mRoot); }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool, size_t) { op(mRoot); }

protected:
    RootNodeType& mRoot;

private:
    NodeManager(const NodeManager&) {} // disallow copy-construction
}; // NodeManager<0>


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with one level of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 1>
{
public:
    using RootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    static_assert(RootNodeType::LEVEL > 0, "expected instantiation of template specialization");
    static const Index LEVELS = 1;

    NodeManager(TreeOrLeafManagerT& tree) : mRoot(tree.root())
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (TreeOrLeafManagerT::DEPTH == 2 && NodeT0::LEVEL == 0) {
            tree.getNodes(mList0);
        } else {
            mRoot.getNodes(mList0);
        }
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }

    virtual ~NodeManager() {}

    /// @brief Clear all the cached tree nodes
    void clear() { mList0.clear(); }

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild() { mList0.clear(); mRoot.getNodes(mList0); }

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    /// @brief Return the total number of cached nodes (excluding the root node)
    Index64 nodeCount() const { return mList0.nodeCount(); }

    /// @brief Return the number of cached nodes at level @a i, where
    /// 0 corresponds to the lowest level.
    Index64 nodeCount(Index i) const { return i==0 ? mList0.nodeCount() : 0; }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.foreach(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList0.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.reduce(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList0.reduce(op, threaded, grainSize);
    }

protected:
    using NodeT1 = RootNodeType;
    using NodeT0 = typename NodeT1::ChildNodeType;
    using ListT0 = NodeList<NodeT0>;

    NodeT1& mRoot;
    ListT0 mList0;

private:
    NodeManager(const NodeManager&) {} // disallow copy-construction
}; // NodeManager<1>


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with two levels of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 2>
{
public:
    using RootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    static_assert(RootNodeType::LEVEL > 1, "expected instantiation of template specialization");
    static const Index LEVELS = 2;

    NodeManager(TreeOrLeafManagerT& tree) : mRoot(tree.root())
    {
        mRoot.getNodes(mList1);

        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (TreeOrLeafManagerT::DEPTH == 2 && NodeT0::LEVEL == 0) {
            tree.getNodes(mList0);
        } else {
            for (size_t i=0, n=mList1.nodeCount(); i<n; ++i) mList1(i).getNodes(mList0);
        }
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }

    virtual ~NodeManager() {}

    /// @brief Clear all the cached tree nodes
    void clear() { mList0.clear(); mList1.clear(); }

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild()
    {
        this->clear();
        mRoot.getNodes(mList1);
        for (size_t i=0, n=mList1.nodeCount(); i<n; ++i) mList1(i).getNodes(mList0);
    }

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    /// @brief Return the total number of cached nodes (excluding the root node)
    Index64 nodeCount() const { return mList0.nodeCount() + mList1.nodeCount(); }

    /// @brief Return the number of cached nodes at level @a i, where
    /// 0 corresponds to the lowest level.
    Index64 nodeCount(Index i) const
    {
        return i==0 ? mList0.nodeCount() : i==1 ? mList1.nodeCount() : 0;
    }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.foreach(op, threaded, grainSize);
        mList1.foreach(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList1.foreach(op, threaded, grainSize);
        mList0.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.reduce(op, threaded, grainSize);
        mList1.reduce(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList1.reduce(op, threaded, grainSize);
        mList0.reduce(op, threaded, grainSize);
    }

protected:
    using NodeT2 = RootNodeType;
    using NodeT1 = typename NodeT2::ChildNodeType; // upper level
    using NodeT0 = typename NodeT1::ChildNodeType; // lower level

    using ListT1 = NodeList<NodeT1>; // upper level
    using ListT0 = NodeList<NodeT0>; // lower level

    NodeT2& mRoot;
    ListT1 mList1;
    ListT0 mList0;

private:
    NodeManager(const NodeManager&) {} // disallow copy-construction
}; // NodeManager<2>


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with three levels of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 3>
{
public:
    using RootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    static_assert(RootNodeType::LEVEL > 2, "expected instantiation of template specialization");
    static const Index LEVELS = 3;

    NodeManager(TreeOrLeafManagerT& tree) : mRoot(tree.root())
    {
        mRoot.getNodes(mList2);
        for (size_t i=0, n=mList2.nodeCount(); i<n; ++i) mList2(i).getNodes(mList1);

        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (TreeOrLeafManagerT::DEPTH == 2 && NodeT0::LEVEL == 0) {
            tree.getNodes(mList0);
        } else {
            for (size_t i=0, n=mList1.nodeCount(); i<n; ++i) mList1(i).getNodes(mList0);
        }
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }

    virtual ~NodeManager() {}

    /// @brief Clear all the cached tree nodes
    void clear() { mList0.clear(); mList1.clear(); mList2.clear(); }

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild()
    {
        this->clear();
        mRoot.getNodes(mList2);
        for (size_t i=0, n=mList2.nodeCount(); i<n; ++i) mList2(i).getNodes(mList1);
        for (size_t i=0, n=mList1.nodeCount(); i<n; ++i) mList1(i).getNodes(mList0);
    }

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    /// @brief Return the total number of cached nodes (excluding the root node)
    Index64 nodeCount() const { return mList0.nodeCount()+mList1.nodeCount()+mList2.nodeCount(); }

    /// @brief Return the number of cached nodes at level @a i, where
    /// 0 corresponds to the lowest level.
    Index64 nodeCount(Index i) const
    {
        return i==0 ? mList0.nodeCount() : i==1 ? mList1.nodeCount()
             : i==2 ? mList2.nodeCount() : 0;
    }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.foreach(op, threaded, grainSize);
        mList1.foreach(op, threaded, grainSize);
        mList2.foreach(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList2.foreach(op, threaded, grainSize);
        mList1.foreach(op, threaded, grainSize);
        mList0.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.reduce(op, threaded, grainSize);
        mList1.reduce(op, threaded, grainSize);
        mList2.reduce(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList2.reduce(op, threaded, grainSize);
        mList1.reduce(op, threaded, grainSize);
        mList0.reduce(op, threaded, grainSize);
    }

protected:
    using NodeT3 = RootNodeType;
    using NodeT2 = typename NodeT3::ChildNodeType; // upper level
    using NodeT1 = typename NodeT2::ChildNodeType; // mid level
    using NodeT0 = typename NodeT1::ChildNodeType; // lower level

    using ListT2 = NodeList<NodeT2>; // upper level of internal nodes
    using ListT1 = NodeList<NodeT1>; // lower level of internal nodes
    using ListT0 = NodeList<NodeT0>; // lower level of internal nodes or leafs

    NodeT3& mRoot;
    ListT2 mList2;
    ListT1 mList1;
    ListT0 mList0;

private:
    NodeManager(const NodeManager&) {} // disallow copy-construction
}; // NodeManager<3>


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with four levels of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 4>
{
public:
    using RootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    static_assert(RootNodeType::LEVEL > 3, "expected instantiation of template specialization");
    static const Index LEVELS = 4;

    NodeManager(TreeOrLeafManagerT& tree) : mRoot(tree.root())
    {
        mRoot.getNodes(mList3);
        for (size_t i=0, n=mList3.nodeCount(); i<n; ++i) mList3(i).getNodes(mList2);
        for (size_t i=0, n=mList2.nodeCount(); i<n; ++i) mList2(i).getNodes(mList1);

        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (TreeOrLeafManagerT::DEPTH == 2 && NodeT0::LEVEL == 0) {
            tree.getNodes(mList0);
        } else {
            for (size_t i=0, n=mList1.nodeCount(); i<n; ++i) mList1(i).getNodes(mList0);
        }
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }

    virtual ~NodeManager() {}

    /// @brief Clear all the cached tree nodes
    void clear() { mList0.clear(); mList1.clear(); mList2.clear(); mList3.clear; }

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild()
    {
        this->clear();
        mRoot.getNodes(mList3);
        for (size_t i=0, n=mList3.nodeCount(); i<n; ++i) mList3(i).getNodes(mList2);
        for (size_t i=0, n=mList2.nodeCount(); i<n; ++i) mList2(i).getNodes(mList1);
        for (size_t i=0, n=mList1.nodeCount(); i<n; ++i) mList1(i).getNodes(mList0);
    }

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    /// @brief Return the total number of cached nodes (excluding the root node)
    Index64 nodeCount() const
    {
        return mList0.nodeCount() + mList1.nodeCount()
             + mList2.nodeCount() + mList3.nodeCount();
    }

    /// @brief Return the number of cached nodes at level @a i, where
    /// 0 corresponds to the lowest level.
    Index64 nodeCount(Index i) const
    {
        return i==0 ? mList0.nodeCount() : i==1 ? mList1.nodeCount() :
               i==2 ? mList2.nodeCount() : i==3 ? mList3.nodeCount() : 0;
    }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.foreach(op, threaded, grainSize);
        mList1.foreach(op, threaded, grainSize);
        mList2.foreach(op, threaded, grainSize);
        mList3.foreach(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList3.foreach(op, threaded, grainSize);
        mList2.foreach(op, threaded, grainSize);
        mList1.foreach(op, threaded, grainSize);
        mList0.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.reduce(op, threaded, grainSize);
        mList1.reduce(op, threaded, grainSize);
        mList2.reduce(op, threaded, grainSize);
        mList3.reduce(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList3.reduce(op, threaded, grainSize);
        mList2.reduce(op, threaded, grainSize);
        mList1.reduce(op, threaded, grainSize);
        mList0.reduce(op, threaded, grainSize);
    }

protected:
    using NodeT4 = RootNodeType;
    using NodeT3 = typename NodeT4::ChildNodeType; // upper level
    using NodeT2 = typename NodeT3::ChildNodeType; // upper mid level
    using NodeT1 = typename NodeT2::ChildNodeType; // lower mid level
    using NodeT0 = typename NodeT1::ChildNodeType; // lower level

    using ListT3 = NodeList<NodeT3>; // upper level of internal nodes
    using ListT2 = NodeList<NodeT2>; // upper mid level of internal nodes
    using ListT1 = NodeList<NodeT1>; // lower mid level of internal nodes
    using ListT0 = NodeList<NodeT0>; // lower level of internal nodes or leafs

    NodeT4& mRoot;
    ListT3  mList3;
    ListT2  mList2;
    ListT1  mList1;
    ListT0  mList0;

private:
    NodeManager(const NodeManager&) {} // disallow copy-construction
}; // NodeManager<4>

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_NODEMANAGER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
