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

/// @file tree/ValueAccessor.h
///
/// When traversing a grid in a spatially coherent pattern (e.g., iterating
/// over neighboring voxels), request a @c ValueAccessor from the grid
/// (with Grid::getAccessor()) and use the accessor's @c getValue() and
/// @c setValue() methods.  These will typically be significantly faster
/// than accessing voxels directly in the grid's tree.
///
/// @par Example:
///
/// @code
/// FloatGrid grid;
/// FloatGrid::Accessor acc = grid.getAccessor();
/// // First access is slow:
/// acc.setValue(Coord(0, 0, 0), 100);
/// // Subsequent nearby accesses are fast, since the accessor now holds pointers
/// // to nodes that contain (0, 0, 0) along the path from the root of the grid's
/// // tree to the leaf:
/// acc.setValue(Coord(0, 0, 1), 100);
/// acc.getValue(Coord(0, 2, 0), 100);
/// // Slow, because the accessor must be repopulated:
/// acc.getValue(Coord(-1, -1, -1));
/// // Fast:
/// acc.getValue(Coord(-1, -1, -2));
/// acc.setValue(Coord(-1, -2, 0), -100);
/// @endcode

#ifndef OPENVDB_TREE_VALUEACCESSOR_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_VALUEACCESSOR_HAS_BEEN_INCLUDED

#include <boost/mpl/front.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/comparison.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/erase.hpp>
#include <boost/mpl/find.hpp>
#include <tbb/null_mutex.h>
#include <tbb/spin_mutex.h>
#include <openvdb/version.h>
#include <openvdb/Types.h>
#include <cassert>
#include <limits>
#include <type_traits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

// Forward declarations of local classes that are not intended for general use
// The IsSafe template parameter is explained in the warning below.
template<typename TreeType, bool IsSafe = true>
class ValueAccessor0;
template<typename TreeType, bool IsSafe = true, Index L0 = 0>
class ValueAccessor1;
template<typename TreeType, bool IsSafe = true, Index L0 = 0, Index L1 = 1>
class ValueAccessor2;
template<typename TreeType, bool IsSafe = true, Index L0 = 0, Index L1 = 1, Index L2 = 2>
class ValueAccessor3;
template<typename TreeCacheT, typename NodeVecT, bool AtRoot> class CacheItem;


/// @brief This base class for ValueAccessors manages registration of an accessor
/// with a tree so that the tree can automatically clear the accessor whenever
/// one of its nodes is deleted.
///
/// @internal A base class is needed because ValueAccessor is templated on both
/// a Tree type and a mutex type.  The various instantiations of the template
/// are distinct, unrelated types, so they can't easily be stored in a container
/// such as the Tree's CacheRegistry.  This base class, in contrast, is templated
/// only on the Tree type, so for any given Tree, only two distinct instantiations
/// are possible, ValueAccessorBase<Tree> and ValueAccessorBase<const Tree>.
///
/// @warning If IsSafe = false then the ValueAccessor will not register itself
/// with the tree from which it is constructed. While in some rare cases this can
/// lead to better performance (since it avoids the small overhead of insertion
/// on creation and deletion on destruction) it is also unsafe if the tree is
/// modified. So unless you're an expert it is highly recommended to set
/// IsSafe = true, which is the default in all derived ValueAccessors defined
/// below. However if you know that the tree is no being modifed for the lifespan
/// of the ValueAccessor AND the work performed per ValueAccessor is small relative
/// to overhead of registering it you should consider setting IsSafe = false. If
/// this turns out to improve performance you should really rewrite your code so as
/// to better amortize the construction of the ValueAccessor, i.e. reuse it as much
/// as possible!
template<typename TreeType, bool IsSafe>
class ValueAccessorBase
{
public:
    static const bool IsConstTree = std::is_const<TreeType>::value;

    /// @brief Return true if this accessor is safe, i.e. registered
    /// by the tree from which it is constructed. Un-registered
    /// accessors can in rare cases be faster because it avoids the
    /// (small) overhead of registration, but they are unsafe if the
    /// tree is modified. So unless you're an expert it is highly
    /// recommended to set IsSafe = true (which is the default).
    static bool isSafe() { return IsSafe; }

    ValueAccessorBase(TreeType& tree): mTree(&tree)
    {
        if (IsSafe) tree.attachAccessor(*this);
    }

    virtual ~ValueAccessorBase() { if (IsSafe && mTree) mTree->releaseAccessor(*this); }

    /// @brief Return a pointer to the tree associated with this accessor.
    /// @details The pointer will be null only if the tree from which this accessor
    /// was constructed was subsequently deleted (which generally leaves the
    /// accessor in an unsafe state).
    TreeType* getTree() const { return mTree; }
    /// Return a reference to the tree associated with this accessor.
    TreeType& tree() const { assert(mTree); return *mTree; }

    ValueAccessorBase(const ValueAccessorBase& other): mTree(other.mTree)
    {
        if (IsSafe && mTree) mTree->attachAccessor(*this);
    }

    ValueAccessorBase& operator=(const ValueAccessorBase& other)
    {
        if (&other != this) {
            if (IsSafe && mTree) mTree->releaseAccessor(*this);
            mTree = other.mTree;
            if (IsSafe && mTree) mTree->attachAccessor(*this);
        }
        return *this;
    }

    virtual void clear() = 0;

protected:
    // Allow trees to deregister themselves.
    template<typename> friend class Tree;

    virtual void release() { mTree = nullptr; }

    TreeType* mTree;
}; // class ValueAccessorBase


////////////////////////////////////////


/// When traversing a grid in a spatially coherent pattern (e.g., iterating
/// over neighboring voxels), request a @c ValueAccessor from the grid
/// (with Grid::getAccessor()) and use the accessor's @c getValue() and
/// @c setValue() methods.  These will typically be significantly faster
/// than accessing voxels directly in the grid's tree.
///
/// A ValueAccessor caches pointers to tree nodes along the path to a voxel (x, y, z).
/// A subsequent access to voxel (x', y', z') starts from the cached leaf node and
/// moves up until a cached node that encloses (x', y', z') is found, then traverses
/// down the tree from that node to a leaf, updating the cache with the new path.
/// This leads to significant acceleration of spatially-coherent accesses.
///
/// @param _TreeType    the type of the tree to be accessed [required]
/// @param IsSafe       if IsSafe = false then the ValueAccessor will
///                     not register itself with the tree from which
///                     it is consturcted (see warning).
/// @param CacheLevels  the number of nodes to be cached, starting from the leaf level
///                     and not including the root (i.e., CacheLevels < DEPTH),
///                     and defaulting to all non-root nodes
/// @param MutexType    the type of mutex to use (see note)
///
/// @warning If IsSafe = false then the ValueAccessor will not register itself
/// with the tree from which it is constructed. While in some rare cases this can
/// lead to better performance (since it avoids the small overhead of insertion
/// on creation and deletion on destruction) it is also unsafe if the tree is
/// modified. So unless you're an expert it is highly recommended to set
/// IsSafe = true, which is the default. However if you know that the tree is no
/// being modifed for the lifespan of the ValueAccessor AND the work performed
/// per ValueAccessor is small relative to overhead of registering it you should
/// consider setting IsSafe = false. If this improves performance you should
/// really rewrite your code so as to better amortize the construction of the
/// ValueAccessor, i.e. reuse it as much as possible!
///
/// @note If @c MutexType is a TBB-compatible mutex, then multiple threads may
/// safely access a single, shared accessor.  However, it is highly recommended
/// that, instead, each thread be assigned its own, non-mutex-protected accessor.
template<typename _TreeType,
         bool IsSafe = true,
         Index CacheLevels = _TreeType::DEPTH-1,
         typename MutexType = tbb::null_mutex>
class ValueAccessor: public ValueAccessorBase<_TreeType, IsSafe>
{
public:
    static_assert(CacheLevels < _TreeType::DEPTH, "cache size exceeds tree depth");

    using TreeType = _TreeType;
    using RootNodeT = typename TreeType::RootNodeType;
    using LeafNodeT = typename TreeType::LeafNodeType;
    using ValueType = typename RootNodeT::ValueType;
    using BaseT = ValueAccessorBase<TreeType, IsSafe>;
    using LockT = typename MutexType::scoped_lock;
    using BaseT::IsConstTree;

    ValueAccessor(TreeType& tree): BaseT(tree), mCache(*this)
    {
        mCache.insert(Coord(), &tree.root());
    }

    ValueAccessor(const ValueAccessor& other): BaseT(other), mCache(*this, other.mCache) {}

    ValueAccessor& operator=(const ValueAccessor& other)
    {
        if (&other != this) {
            this->BaseT::operator=(other);
            mCache.copy(*this, other.mCache);
        }
        return *this;
    }
    ~ValueAccessor() override = default;

    /// Return the number of cache levels employed by this accessor.
    static Index numCacheLevels() { return CacheLevels; }

    /// Return @c true if nodes along the path to the given voxel have been cached.
    bool isCached(const Coord& xyz) const { LockT lock(mMutex); return mCache.isCached(xyz); }

    /// Return the value of the voxel at the given coordinates.
    const ValueType& getValue(const Coord& xyz) const
    {
        LockT lock(mMutex);
        return mCache.getValue(xyz);
    }

    /// Return the active state of the voxel at the given coordinates.
    bool isValueOn(const Coord& xyz) const { LockT lock(mMutex); return mCache.isValueOn(xyz); }

    /// Return the active state of the voxel as well as its value
    bool probeValue(const Coord& xyz, ValueType& value) const
    {
        LockT lock(mMutex);
        return mCache.probeValue(xyz,value);
    }

    /// Return the tree depth (0 = root) at which the value of voxel (x, y, z) resides,
    /// or -1 if (x, y, z) isn't explicitly represented in the tree (i.e., if it is
    /// implicitly a background voxel).
    int getValueDepth(const Coord& xyz) const
    {
        LockT lock(mMutex);
        return mCache.getValueDepth(xyz);
    }

    /// Return @c true if the value of voxel (x, y, z) resides at the leaf level
    /// of the tree, i.e., if it is not a tile value.
    bool isVoxel(const Coord& xyz) const { LockT lock(mMutex); return mCache.isVoxel(xyz); }

    //@{
    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValue(const Coord& xyz, const ValueType& value)
    {
        LockT lock(mMutex);
        mCache.setValue(xyz, value);
    }
    void setValueOn(const Coord& xyz, const ValueType& value) { this->setValue(xyz, value); }
    //@}

    /// Set the value of the voxel at the given coordinate but don't change its active state.
    void setValueOnly(const Coord& xyz, const ValueType& value)
    {
        LockT lock(mMutex);
        mCache.setValueOnly(xyz, value);
    }

    /// Set the value of the voxel at the given coordinates and mark the voxel as inactive.
    void setValueOff(const Coord& xyz, const ValueType& value)
    {
        LockT lock(mMutex);
        mCache.setValueOff(xyz, value);
    }

    /// @brief Apply a functor to the value of the voxel at the given coordinates
    /// and mark the voxel as active.
    /// @details See Tree::modifyValue() for details.
    template<typename ModifyOp>
    void modifyValue(const Coord& xyz, const ModifyOp& op)
    {
        LockT lock(mMutex);
        mCache.modifyValue(xyz, op);
    }

    /// @brief Apply a functor to the voxel at the given coordinates.
    /// @details See Tree::modifyValueAndActiveState() for details.
    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op)
    {
        LockT lock(mMutex);
        mCache.modifyValueAndActiveState(xyz, op);
    }

    /// Set the active state of the voxel at the given coordinates but don't change its value.
    void setActiveState(const Coord& xyz, bool on = true)
    {
        LockT lock(mMutex);
        mCache.setActiveState(xyz, on);
    }
    /// Mark the voxel at the given coordinates as active but don't change its value.
    void setValueOn(const Coord& xyz) { this->setActiveState(xyz, true); }
    /// Mark the voxel at the given coordinates as inactive but don't change its value.
    void setValueOff(const Coord& xyz) { this->setActiveState(xyz, false); }

    /// Return the cached node of type @a NodeType.  [Mainly for internal use]
    template<typename NodeType>
    NodeType* getNode()
    {
        LockT lock(mMutex);
        NodeType* node = nullptr;
        mCache.getNode(node);
        return node;
    }

    /// Cache the given node, which should lie along the path from the root node to
    /// the node containing voxel (x, y, z).  [Mainly for internal use]
    template<typename NodeType>
    void insertNode(const Coord& xyz, NodeType& node)
    {
        LockT lock(mMutex);
        mCache.insert(xyz, &node);
    }

    /// If a node of the given type exists in the cache, remove it, so that
    /// isCached(xyz) returns @c false for any voxel (x, y, z) contained in
    /// that node.  [Mainly for internal use]
    template<typename NodeType>
    void eraseNode() { LockT lock(mMutex); NodeType* node = nullptr; mCache.erase(node); }

    /// @brief Add the specified leaf to this tree, possibly creating a child branch
    /// in the process.  If the leaf node already exists, replace it.
    void addLeaf(LeafNodeT* leaf)
    {
        LockT lock(mMutex);
        mCache.addLeaf(leaf);
    }

    /// @brief Add a tile at the specified tree level that contains voxel (x, y, z),
    /// possibly deleting existing nodes or creating new nodes in the process.
    void addTile(Index level, const Coord& xyz, const ValueType& value, bool state)
    {
        LockT lock(mMutex);
        mCache.addTile(level, xyz, value, state);
    }

    /// @brief Return a pointer to the leaf node that contains voxel (x, y, z).
    /// If no such node exists, create one, but preserve the values and
    /// active states of all voxels.
    /// @details Use this method to preallocate a static tree topology
    /// over which to safely perform multithreaded processing.
    LeafNodeT* touchLeaf(const Coord& xyz)
    {
        LockT lock(mMutex);
        return mCache.touchLeaf(xyz);
    }

    //@{
    /// @brief Return a pointer to the node of the specified type that contains
    /// voxel (x, y, z), or @c nullptr if no such node exists.
    template<typename NodeT>
    NodeT* probeNode(const Coord& xyz)
    {
        LockT lock(mMutex);
        return mCache.template probeNode<NodeT>(xyz);
    }
    template<typename NodeT>
    const NodeT* probeConstNode(const Coord& xyz) const
    {
        LockT lock(mMutex);
        return mCache.template probeConstNode<NodeT>(xyz);
    }
    template<typename NodeT>
    const NodeT* probeNode(const Coord& xyz) const
    {
        return this->template probeConstNode<NodeT>(xyz);
    }
    //@}

    //@{
    /// @brief Return a pointer to the leaf node that contains voxel (x, y, z),
    /// or @c nullptr if no such node exists.
    LeafNodeT* probeLeaf(const Coord& xyz)
    {
        LockT lock(mMutex);
        return mCache.probeLeaf(xyz);
    }
    const LeafNodeT* probeConstLeaf(const Coord& xyz) const
    {
        LockT lock(mMutex);
        return mCache.probeConstLeaf(xyz);
    }
    const LeafNodeT* probeLeaf(const Coord& xyz) const { return this->probeConstLeaf(xyz); }
    //@}

    /// Remove all nodes from this cache, then reinsert the root node.
    void clear() override
    {
        LockT lock(mMutex);
        mCache.clear();
        if (this->mTree) mCache.insert(Coord(), &(this->mTree->root()));
    }

private:
    // Allow nodes to insert themselves into the cache.
    template<typename> friend class RootNode;
    template<typename, Index> friend class InternalNode;
    template<typename, Index> friend class LeafNode;
    // Allow trees to deregister themselves.
    template<typename> friend class Tree;

    /// Prevent this accessor from calling Tree::releaseCache() on a tree that
    /// no longer exists.  (Called by mTree when it is destroyed.)
    void release() override
    {
        LockT lock(mMutex);
        this->BaseT::release();
        mCache.clear();
    }

    /// Cache the given node, which should lie along the path from the root node to
    /// the node containing voxel (x, y, z).
    /// @note This operation is not mutex-protected and is intended to be called
    /// only by nodes and only in the context of a getValue() or setValue() call.
    template<typename NodeType>
    void insert(const Coord& xyz, NodeType* node) { mCache.insert(xyz, node); }

    // Define a list of all tree node types from LeafNode to RootNode
    using InvTreeT = typename RootNodeT::NodeChainType;
    // Remove all tree node types that are excluded from the cache
    using BeginT = typename boost::mpl::begin<InvTreeT>::type;
    using FirstT = typename boost::mpl::advance<BeginT, boost::mpl::int_<CacheLevels>>::type;
    using LastT = typename boost::mpl::find<InvTreeT, RootNodeT>::type;
    using SubtreeT = typename boost::mpl::erase<InvTreeT, FirstT, LastT>::type;
    using CacheItemT = CacheItem<ValueAccessor, SubtreeT, boost::mpl::size<SubtreeT>::value==1>;

    // Private member data
    mutable CacheItemT mCache;
    mutable MutexType  mMutex;

}; // class ValueAccessor


/// @brief Template specialization of the ValueAccessor with no mutex and no cache levels
/// @details This specialization is provided mainly for benchmarking.
/// Accessors with caching will almost always be faster.
template<typename TreeType, bool IsSafe>
class ValueAccessor<TreeType, IsSafe, 0, tbb::null_mutex>
    : public ValueAccessor0<TreeType, IsSafe>
{
public:
    ValueAccessor(TreeType& tree): ValueAccessor0<TreeType, IsSafe>(tree) {}
    ValueAccessor(const ValueAccessor& other): ValueAccessor0<TreeType, IsSafe>(other) {}
    ~ValueAccessor() override = default;
};


/// Template specialization of the ValueAccessor with no mutex and one cache level
template<typename TreeType, bool IsSafe>
class ValueAccessor<TreeType, IsSafe, 1, tbb::null_mutex>
    : public ValueAccessor1<TreeType, IsSafe>
{
public:
    ValueAccessor(TreeType& tree): ValueAccessor1<TreeType, IsSafe>(tree) {}
    ValueAccessor(const ValueAccessor& other): ValueAccessor1<TreeType, IsSafe>(other) {}
    ~ValueAccessor() override = default;
};


/// Template specialization of the ValueAccessor with no mutex and two cache levels
template<typename TreeType, bool IsSafe>
class ValueAccessor<TreeType, IsSafe, 2, tbb::null_mutex>
    : public ValueAccessor2<TreeType, IsSafe>
{
public:
    ValueAccessor(TreeType& tree): ValueAccessor2<TreeType, IsSafe>(tree) {}
    ValueAccessor(const ValueAccessor& other): ValueAccessor2<TreeType, IsSafe>(other) {}
    ~ValueAccessor() override = default;
};


/// Template specialization of the ValueAccessor with no mutex and three cache levels
template<typename TreeType, bool IsSafe>
class ValueAccessor<TreeType, IsSafe, 3, tbb::null_mutex>: public ValueAccessor3<TreeType, IsSafe>
{
public:
    ValueAccessor(TreeType& tree): ValueAccessor3<TreeType, IsSafe>(tree) {}
    ValueAccessor(const ValueAccessor&) = default;
    ValueAccessor& operator=(const ValueAccessor&) = default;
    ~ValueAccessor() override = default;
};


////////////////////////////////////////


/// @brief This accessor is thread-safe (at the cost of speed) for both reading and
/// writing to a tree.  That is, multiple threads may safely access a single,
/// shared ValueAccessorRW.
///
/// @warning Since the mutex-locking employed by the ValueAccessorRW
/// can seriously impair performance of multithreaded applications, it
/// is recommended that, instead, each thread be assigned its own
/// (non-mutex protected) accessor.
template<typename TreeType, bool IsSafe = true>
class ValueAccessorRW: public ValueAccessor<TreeType, IsSafe, TreeType::DEPTH-1, tbb::spin_mutex>
{
public:
    ValueAccessorRW(TreeType& tree)
        : ValueAccessor<TreeType, IsSafe, TreeType::DEPTH-1, tbb::spin_mutex>(tree)
    {
    }
};


////////////////////////////////////////


//
// The classes below are for internal use and should rarely be used directly.
//

// An element of a compile-time linked list of node pointers, ordered from LeafNode to RootNode
template<typename TreeCacheT, typename NodeVecT, bool AtRoot>
class CacheItem
{
public:
    using NodeType = typename boost::mpl::front<NodeVecT>::type;
    using ValueType = typename NodeType::ValueType;
    using LeafNodeType = typename NodeType::LeafNodeType;
    using CoordLimits = std::numeric_limits<Int32>;

    CacheItem(TreeCacheT& parent):
        mParent(&parent),
        mHash(CoordLimits::max()),
        mNode(nullptr),
        mNext(parent)
    {
    }

    //@{
    /// Copy another CacheItem's node pointers and hash keys, but not its parent pointer.
    CacheItem(TreeCacheT& parent, const CacheItem& other):
        mParent(&parent),
        mHash(other.mHash),
        mNode(other.mNode),
        mNext(parent, other.mNext)
    {
    }

    CacheItem& copy(TreeCacheT& parent, const CacheItem& other)
    {
        mParent = &parent;
        mHash = other.mHash;
        mNode = other.mNode;
        mNext.copy(parent, other.mNext);
        return *this;
    }
    //@}

    bool isCached(const Coord& xyz) const
    {
        return (this->isHashed(xyz) || mNext.isCached(xyz));
    }

    /// Cache the given node at this level.
    void insert(const Coord& xyz, const NodeType* node)
    {
        mHash = (node != nullptr) ? xyz & ~(NodeType::DIM-1) : Coord::max();
        mNode = node;
    }
    /// Forward the given node to another level of the cache.
    template<typename OtherNodeType>
    void insert(const Coord& xyz, const OtherNodeType* node) { mNext.insert(xyz, node); }

    /// Erase the node at this level.
    void erase(const NodeType*) { mHash = Coord::max(); mNode = nullptr; }
    /// Erase the node at another level of the cache.
    template<typename OtherNodeType>
    void erase(const OtherNodeType* node) { mNext.erase(node); }

    /// Erase the nodes at this and lower levels of the cache.
    void clear() { mHash = Coord::max(); mNode = nullptr; mNext.clear(); }

    /// Return the cached node (if any) at this level.
    void getNode(const NodeType*& node) const { node = mNode; }
    void getNode(const NodeType*& node) { node = mNode; }
    void getNode(NodeType*& node)
    {
        // This combination of a static assertion and a const_cast might not be elegant,
        // but it is a lot simpler than specializing TreeCache for const Trees.
        static_assert(!TreeCacheT::IsConstTree, "can't get a non-const node from a const tree");
        node = const_cast<NodeType*>(mNode);
    }
    /// Forward the request to another level of the cache.
    template<typename OtherNodeType>
    void getNode(OtherNodeType*& node) { mNext.getNode(node); }

    /// Return the value of the voxel at the given coordinates.
    const ValueType& getValue(const Coord& xyz)
    {
        if (this->isHashed(xyz)) {
            assert(mNode);
            return mNode->getValueAndCache(xyz, *mParent);
        }
        return mNext.getValue(xyz);
    }

    void addLeaf(LeafNodeType* leaf)
    {
        static_assert(!TreeCacheT::IsConstTree, "can't add a node to a const tree");
        if (NodeType::LEVEL == 0) return;
        if (this->isHashed(leaf->origin())) {
            assert(mNode);
            return const_cast<NodeType*>(mNode)->addLeafAndCache(leaf, *mParent);
        }
        mNext.addLeaf(leaf);
    }

    void addTile(Index level, const Coord& xyz, const ValueType& value, bool state)
    {
        static_assert(!TreeCacheT::IsConstTree, "can't add a tile to a const tree");
        if (NodeType::LEVEL < level) return;
        if (this->isHashed(xyz)) {
            assert(mNode);
            return const_cast<NodeType*>(mNode)->addTileAndCache(
                level, xyz, value, state, *mParent);
        }
        mNext.addTile(level, xyz, value, state);
    }

    LeafNodeType* touchLeaf(const Coord& xyz)
    {
        static_assert(!TreeCacheT::IsConstTree, "can't get a non-const node from a const tree");
        if (this->isHashed(xyz)) {
            assert(mNode);
            return const_cast<NodeType*>(mNode)->touchLeafAndCache(xyz, *mParent);
        }
        return mNext.touchLeaf(xyz);
    }

    LeafNodeType* probeLeaf(const Coord& xyz)
    {
        static_assert(!TreeCacheT::IsConstTree, "can't get a non-const node from a const tree");
        if (this->isHashed(xyz)) {
            assert(mNode);
            return const_cast<NodeType*>(mNode)->probeLeafAndCache(xyz, *mParent);
        }
        return mNext.probeLeaf(xyz);
    }

    const LeafNodeType* probeConstLeaf(const Coord& xyz)
    {
        if (this->isHashed(xyz)) {
            assert(mNode);
            return mNode->probeConstLeafAndCache(xyz, *mParent);
        }
        return mNext.probeConstLeaf(xyz);
    }

    template<typename NodeT>
    NodeT* probeNode(const Coord& xyz)
    {
        static_assert(!TreeCacheT::IsConstTree, "can't get a non-const node from a const tree");
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (this->isHashed(xyz)) {
            if ((std::is_same<NodeT, NodeType>::value)) {
                assert(mNode);
                return reinterpret_cast<NodeT*>(const_cast<NodeType*>(mNode));
            }
            return const_cast<NodeType*>(mNode)->template probeNodeAndCache<NodeT>(xyz, *mParent);
        }
        return mNext.template probeNode<NodeT>(xyz);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }

    template<typename NodeT>
    const NodeT* probeConstNode(const Coord& xyz)
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (this->isHashed(xyz)) {
            if ((std::is_same<NodeT, NodeType>::value)) {
                assert(mNode);
                return reinterpret_cast<const NodeT*>(mNode);
            }
            return mNode->template probeConstNodeAndCache<NodeT>(xyz, *mParent);
        }
        return mNext.template probeConstNode<NodeT>(xyz);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }

    /// Return the active state of the voxel at the given coordinates.
    bool isValueOn(const Coord& xyz)
    {
        if (this->isHashed(xyz)) {
            assert(mNode);
            return mNode->isValueOnAndCache(xyz, *mParent);
        }
        return mNext.isValueOn(xyz);
    }

    /// Return the active state and value of the voxel at the given coordinates.
    bool probeValue(const Coord& xyz, ValueType& value)
    {
        if (this->isHashed(xyz)) {
            assert(mNode);
            return mNode->probeValueAndCache(xyz, value, *mParent);
        }
        return mNext.probeValue(xyz, value);
    }

     int getValueDepth(const Coord& xyz)
    {
        if (this->isHashed(xyz)) {
            assert(mNode);
            return static_cast<int>(TreeCacheT::RootNodeT::LEVEL) -
                   static_cast<int>(mNode->getValueLevelAndCache(xyz, *mParent));
        } else {
            return mNext.getValueDepth(xyz);
        }
    }

    bool isVoxel(const Coord& xyz)
    {
        if (this->isHashed(xyz)) {
            assert(mNode);
            return mNode->getValueLevelAndCache(xyz, *mParent)==0;
        } else {
            return mNext.isVoxel(xyz);
        }
    }

    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValue(const Coord& xyz, const ValueType& value)
    {
        if (this->isHashed(xyz)) {
            assert(mNode);
            static_assert(!TreeCacheT::IsConstTree, "can't modify a const tree's values");
            const_cast<NodeType*>(mNode)->setValueAndCache(xyz, value, *mParent);
        } else {
            mNext.setValue(xyz, value);
        }
    }
    void setValueOnly(const Coord& xyz, const ValueType& value)
    {
        if (this->isHashed(xyz)) {
            assert(mNode);
            static_assert(!TreeCacheT::IsConstTree, "can't modify a const tree's values");
            const_cast<NodeType*>(mNode)->setValueOnlyAndCache(xyz, value, *mParent);
        } else {
            mNext.setValueOnly(xyz, value);
        }
    }
    void setValueOn(const Coord& xyz, const ValueType& value) { this->setValue(xyz, value); }

    /// @brief Apply a functor to the value of the voxel at the given coordinates
    /// and mark the voxel as active.
    /// @details See Tree::modifyValue() for details.
    template<typename ModifyOp>
    void modifyValue(const Coord& xyz, const ModifyOp& op)
    {
        if (this->isHashed(xyz)) {
            assert(mNode);
            static_assert(!TreeCacheT::IsConstTree, "can't modify a const tree's values");
            const_cast<NodeType*>(mNode)->modifyValueAndCache(xyz, op, *mParent);
        } else {
            mNext.modifyValue(xyz, op);
        }
    }

    /// @brief Apply a functor to the voxel at the given coordinates.
    /// @details See Tree::modifyValueAndActiveState() for details.
    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op)
    {
        if (this->isHashed(xyz)) {
            assert(mNode);
            static_assert(!TreeCacheT::IsConstTree, "can't modify a const tree's values");
            const_cast<NodeType*>(mNode)->modifyValueAndActiveStateAndCache(xyz, op, *mParent);
        } else {
            mNext.modifyValueAndActiveState(xyz, op);
        }
    }

    /// Set the value of the voxel at the given coordinates and mark the voxel as inactive.
    void setValueOff(const Coord& xyz, const ValueType& value)
    {
        if (this->isHashed(xyz)) {
            assert(mNode);
            static_assert(!TreeCacheT::IsConstTree, "can't modify a const tree's values");
            const_cast<NodeType*>(mNode)->setValueOffAndCache(xyz, value, *mParent);
        } else {
            mNext.setValueOff(xyz, value);
        }
    }

    /// Set the active state of the voxel at the given coordinates.
    void setActiveState(const Coord& xyz, bool on)
    {
        if (this->isHashed(xyz)) {
            assert(mNode);
            static_assert(!TreeCacheT::IsConstTree, "can't modify a const tree's values");
            const_cast<NodeType*>(mNode)->setActiveStateAndCache(xyz, on, *mParent);
        } else {
            mNext.setActiveState(xyz, on);
        }
    }

private:
    CacheItem(const CacheItem&);
    CacheItem& operator=(const CacheItem&);

    bool isHashed(const Coord& xyz) const
    {
        return (xyz[0] & ~Coord::ValueType(NodeType::DIM-1)) == mHash[0]
            && (xyz[1] & ~Coord::ValueType(NodeType::DIM-1)) == mHash[1]
            && (xyz[2] & ~Coord::ValueType(NodeType::DIM-1)) == mHash[2];
    }

    TreeCacheT* mParent;
    Coord mHash;
    const NodeType* mNode;
    using RestT = typename boost::mpl::pop_front<NodeVecT>::type; // NodeVecT minus its first item
    CacheItem<TreeCacheT, RestT, /*AtRoot=*/boost::mpl::size<RestT>::value == 1> mNext;
};// end of CacheItem


/// The tail of a compile-time list of cached node pointers, ordered from LeafNode to RootNode
template<typename TreeCacheT, typename NodeVecT>
class CacheItem<TreeCacheT, NodeVecT, /*AtRoot=*/true>
{
public:
    using RootNodeType = typename boost::mpl::front<NodeVecT>::type;
    using ValueType = typename RootNodeType::ValueType;
    using LeafNodeType = typename RootNodeType::LeafNodeType;

    CacheItem(TreeCacheT& parent): mParent(&parent), mRoot(nullptr) {}
    CacheItem(TreeCacheT& parent, const CacheItem& other): mParent(&parent), mRoot(other.mRoot) {}

    CacheItem& copy(TreeCacheT& parent, const CacheItem& other)
    {
        mParent = &parent;
        mRoot = other.mRoot;
        return *this;
    }

    bool isCached(const Coord& xyz) const { return this->isHashed(xyz); }

    void insert(const Coord&, const RootNodeType* root) { mRoot = root; }

    // Needed for node types that are not cached
    template<typename OtherNodeType>
    void insert(const Coord&, const OtherNodeType*) {}

    void erase(const RootNodeType*) { mRoot = nullptr; }

    void clear() { mRoot = nullptr; }

    void getNode(RootNodeType*& node)
    {
        static_assert(!TreeCacheT::IsConstTree, "can't get a non-const node from a const tree");
        node = const_cast<RootNodeType*>(mRoot);
    }
    void getNode(const RootNodeType*& node) const { node = mRoot; }

    void addLeaf(LeafNodeType* leaf)
    {
        assert(mRoot);
        static_assert(!TreeCacheT::IsConstTree, "can't add a node to a const tree");
        const_cast<RootNodeType*>(mRoot)->addLeafAndCache(leaf, *mParent);
    }

    void addTile(Index level, const Coord& xyz, const ValueType& value, bool state)
    {
        assert(mRoot);
        static_assert(!TreeCacheT::IsConstTree, "can't add a tile to a const tree");
        const_cast<RootNodeType*>(mRoot)->addTileAndCache(level, xyz, value, state, *mParent);
    }

    LeafNodeType* touchLeaf(const Coord& xyz)
    {
        assert(mRoot);
        static_assert(!TreeCacheT::IsConstTree, "can't get a non-const node from a const tree");
        return const_cast<RootNodeType*>(mRoot)->touchLeafAndCache(xyz, *mParent);
    }

    LeafNodeType* probeLeaf(const Coord& xyz)
    {
        assert(mRoot);
        static_assert(!TreeCacheT::IsConstTree, "can't get a non-const node from a const tree");
        return const_cast<RootNodeType*>(mRoot)->probeLeafAndCache(xyz, *mParent);
    }

    const LeafNodeType* probeConstLeaf(const Coord& xyz)
    {
        assert(mRoot);
        return mRoot->probeConstLeafAndCache(xyz, *mParent);
    }

    template<typename NodeType>
    NodeType* probeNode(const Coord& xyz)
    {
        assert(mRoot);
        static_assert(!TreeCacheT::IsConstTree, "can't get a non-const node from a const tree");
        return const_cast<RootNodeType*>(mRoot)->
            template probeNodeAndCache<NodeType>(xyz, *mParent);
    }

    template<typename NodeType>
    const NodeType* probeConstNode(const Coord& xyz)
    {
        assert(mRoot);
        return mRoot->template probeConstNodeAndCache<NodeType>(xyz, *mParent);
    }

    int getValueDepth(const Coord& xyz)
    {
        assert(mRoot);
        return mRoot->getValueDepthAndCache(xyz, *mParent);
    }
    bool isValueOn(const Coord& xyz)
    {
        assert(mRoot);
        return mRoot->isValueOnAndCache(xyz, *mParent);
    }

    bool probeValue(const Coord& xyz, ValueType& value)
    {
        assert(mRoot);
        return mRoot->probeValueAndCache(xyz, value, *mParent);
    }
    bool isVoxel(const Coord& xyz)
    {
        assert(mRoot);
        return mRoot->getValueDepthAndCache(xyz, *mParent) ==
               static_cast<int>(RootNodeType::LEVEL);
    }
    const ValueType& getValue(const Coord& xyz)
    {
        assert(mRoot);
        return mRoot->getValueAndCache(xyz, *mParent);
    }

    void setValue(const Coord& xyz, const ValueType& value)
    {
        assert(mRoot);
        static_assert(!TreeCacheT::IsConstTree, "can't modify a const tree's values");
        const_cast<RootNodeType*>(mRoot)->setValueAndCache(xyz, value, *mParent);
    }
    void setValueOnly(const Coord& xyz, const ValueType& value)
    {
        assert(mRoot);
        static_assert(!TreeCacheT::IsConstTree, "can't modify a const tree's values");
        const_cast<RootNodeType*>(mRoot)->setValueOnlyAndCache(xyz, value, *mParent);
    }
    void setValueOn(const Coord& xyz, const ValueType& value) { this->setValue(xyz, value); }

    template<typename ModifyOp>
    void modifyValue(const Coord& xyz, const ModifyOp& op)
    {
        assert(mRoot);
        static_assert(!TreeCacheT::IsConstTree, "can't modify a const tree's values");
        const_cast<RootNodeType*>(mRoot)->modifyValueAndCache(xyz, op, *mParent);
    }

    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op)
    {
        assert(mRoot);
        static_assert(!TreeCacheT::IsConstTree, "can't modify a const tree's values");
        const_cast<RootNodeType*>(mRoot)->modifyValueAndActiveStateAndCache(xyz, op, *mParent);
    }

    void setValueOff(const Coord& xyz, const ValueType& value)
    {
        assert(mRoot);
        static_assert(!TreeCacheT::IsConstTree, "can't modify a const tree's values");
        const_cast<RootNodeType*>(mRoot)->setValueOffAndCache(xyz, value, *mParent);
    }

    void setActiveState(const Coord& xyz, bool on)
    {
        assert(mRoot);
        static_assert(!TreeCacheT::IsConstTree, "can't modify a const tree's values");
        const_cast<RootNodeType*>(mRoot)->setActiveStateAndCache(xyz, on, *mParent);
    }

private:
    CacheItem(const CacheItem&);
    CacheItem& operator=(const CacheItem&);

    bool isHashed(const Coord&) const { return false; }

    TreeCacheT* mParent;
    const RootNodeType* mRoot;
};// end of CacheItem specialized for RootNode


////////////////////////////////////////


/// @brief ValueAccessor with no mutex and no node caching.
/// @details This specialization is provided mainly for benchmarking.
/// Accessors with caching will almost always be faster.
template<typename _TreeType, bool IsSafe>
class ValueAccessor0: public ValueAccessorBase<_TreeType, IsSafe>
{
public:
    using TreeType = _TreeType;
    using ValueType = typename TreeType::ValueType;
    using RootNodeT = typename TreeType::RootNodeType;
    using LeafNodeT = typename TreeType::LeafNodeType;
    using BaseT = ValueAccessorBase<TreeType, IsSafe>;

    ValueAccessor0(TreeType& tree): BaseT(tree) {}

    ValueAccessor0(const ValueAccessor0& other): BaseT(other) {}

    /// Return the number of cache levels employed by this accessor.
    static Index numCacheLevels() { return 0; }

    ValueAccessor0& operator=(const ValueAccessor0& other)
    {
        if (&other != this) this->BaseT::operator=(other);
        return *this;
    }

    ~ValueAccessor0() override = default;

    /// Return @c true if nodes along the path to the given voxel have been cached.
    bool isCached(const Coord&) const { return false; }

    /// Return the value of the voxel at the given coordinates.
    const ValueType& getValue(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        return BaseT::mTree->getValue(xyz);
    }

    /// Return the active state of the voxel at the given coordinates.
    bool isValueOn(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        return BaseT::mTree->isValueOn(xyz);
    }

    /// Return the active state and, in @a value, the value of the voxel at the given coordinates.
    bool probeValue(const Coord& xyz, ValueType& value) const
    {
        assert(BaseT::mTree);
        return BaseT::mTree->probeValue(xyz, value);
    }

    /// Return the tree depth (0 = root) at which the value of voxel (x, y, z) resides,
    /// or -1 if (x, y, z) isn't explicitly represented in the tree (i.e., if it is
    /// implicitly a background voxel).
    int getValueDepth(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        return BaseT::mTree->getValueDepth(xyz);
    }

    /// Return @c true if the value of voxel (x, y, z) resides at the leaf level
    /// of the tree, i.e., if it is not a tile value.
    bool isVoxel(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        return BaseT::mTree->getValueDepth(xyz) == static_cast<int>(RootNodeT::LEVEL);
    }

    //@{
    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValue(const Coord& xyz, const ValueType& value)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        BaseT::mTree->setValue(xyz, value);
    }
    void setValueOn(const Coord& xyz, const ValueType& value) { this->setValue(xyz, value); }
    //@}

    /// Set the value of the voxel at the given coordinate but don't change its active state.
    void setValueOnly(const Coord& xyz, const ValueType& value)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        BaseT::mTree->setValueOnly(xyz, value);
    }

    /// Set the value of the voxel at the given coordinates and mark the voxel as inactive.
    void setValueOff(const Coord& xyz, const ValueType& value)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        BaseT::mTree->root().setValueOff(xyz, value);
    }

    /// @brief Apply a functor to the value of the voxel at the given coordinates
    /// and mark the voxel as active.
    /// @details See Tree::modifyValue() for details.
    template<typename ModifyOp>
    void modifyValue(const Coord& xyz, const ModifyOp& op)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        BaseT::mTree->modifyValue(xyz, op);
    }

    /// @brief Apply a functor to the voxel at the given coordinates.
    /// @details See Tree::modifyValueAndActiveState() for details.
    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        BaseT::mTree->modifyValueAndActiveState(xyz, op);
    }

    /// Set the active state of the voxel at the given coordinates but don't change its value.
    void setActiveState(const Coord& xyz, bool on = true)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        BaseT::mTree->setActiveState(xyz, on);
    }
    /// Mark the voxel at the given coordinates as active but don't change its value.
    void setValueOn(const Coord& xyz) { this->setActiveState(xyz, true); }
    /// Mark the voxel at the given coordinates as inactive but don't change its value.
    void setValueOff(const Coord& xyz) { this->setActiveState(xyz, false); }

    /// Return the cached node of type @a NodeType.  [Mainly for internal use]
    template<typename NodeT> NodeT* getNode() { return nullptr; }

    /// Cache the given node, which should lie along the path from the root node to
    /// the node containing voxel (x, y, z).  [Mainly for internal use]
    template<typename NodeT> void insertNode(const Coord&, NodeT&) {}

    /// @brief Add the specified leaf to this tree, possibly creating a child branch
    /// in the process.  If the leaf node already exists, replace it.
    void addLeaf(LeafNodeT* leaf)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't add a node to a const tree");
        BaseT::mTree->root().addLeaf(leaf);
    }

    /// @brief Add a tile at the specified tree level that contains voxel (x, y, z),
    /// possibly deleting existing nodes or creating new nodes in the process.
    void addTile(Index level, const Coord& xyz, const ValueType& value, bool state)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't add a tile to a const tree");
        BaseT::mTree->root().addTile(level, xyz, value, state);
    }

    /// If a node of the given type exists in the cache, remove it, so that
    /// isCached(xyz) returns @c false for any voxel (x, y, z) contained in
    /// that node.  [Mainly for internal use]
    template<typename NodeT> void eraseNode() {}

    LeafNodeT* touchLeaf(const Coord& xyz)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't get a non-const node from a const tree");
        return BaseT::mTree->touchLeaf(xyz);
    }

    template<typename NodeT>
    NodeT* probeNode(const Coord& xyz)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't get a non-const node from a const tree");
        return BaseT::mTree->template probeNode<NodeT>(xyz);
    }

    template<typename NodeT>
    const NodeT* probeConstNode(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        return BaseT::mTree->template probeConstNode<NodeT>(xyz);
    }

    LeafNodeT* probeLeaf(const Coord& xyz)
    {
        return this->template probeNode<LeafNodeT>(xyz);
    }

    const LeafNodeT* probeConstLeaf(const Coord& xyz) const
    {
        return this->template probeConstNode<LeafNodeT>(xyz);
    }

    const LeafNodeT* probeLeaf(const Coord& xyz) const
    {
        return this->probeConstLeaf(xyz);
    }

    /// Remove all nodes from this cache, then reinsert the root node.
    void clear() override {}

private:
    // Allow trees to deregister themselves.
    template<typename> friend class Tree;

    /// Prevent this accessor from calling Tree::releaseCache() on a tree that
    /// no longer exists.  (Called by mTree when it is destroyed.)
    void release() override { this->BaseT::release(); }

}; // ValueAccessor0


/// @brief Value accessor with one level of node caching.
/// @details The node cache level is specified by L0 with the default value 0
/// (defined in the forward declaration) corresponding to a LeafNode.
///
/// @note This class is for experts only and should rarely be used
/// directly. Instead use ValueAccessor with its default template arguments.
template<typename _TreeType, bool IsSafe, Index L0>
class ValueAccessor1 : public ValueAccessorBase<_TreeType, IsSafe>
{
public:
    static_assert(_TreeType::DEPTH >= 2, "cache size exceeds tree depth");
    static_assert(L0 < _TreeType::RootNodeType::LEVEL, "invalid cache level");
    using TreeType = _TreeType;
    using ValueType = typename TreeType::ValueType;
    using RootNodeT = typename TreeType::RootNodeType;
    using LeafNodeT = typename TreeType::LeafNodeType;
    using BaseT = ValueAccessorBase<TreeType, IsSafe>;
    using InvTreeT = typename RootNodeT::NodeChainType;
    using NodeT0 = typename boost::mpl::at<InvTreeT, boost::mpl::int_<L0> >::type;

    /// Constructor from a tree
    ValueAccessor1(TreeType& tree) : BaseT(tree), mKey0(Coord::max()), mNode0(nullptr)
    {
    }

    /// Copy constructor
    ValueAccessor1(const ValueAccessor1& other) : BaseT(other) { this->copy(other); }

    /// Return the number of cache levels employed by this ValueAccessor
    static Index numCacheLevels() { return 1; }

    /// Asignment operator
    ValueAccessor1& operator=(const ValueAccessor1& other)
    {
        if (&other != this) {
            this->BaseT::operator=(other);
            this->copy(other);
        }
        return *this;
    }

    /// Virtual destructor
    ~ValueAccessor1() override = default;

    /// Return @c true if any of the nodes along the path to the given
    /// voxel have been cached.
    bool isCached(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        return this->isHashed(xyz);
    }

    /// Return the value of the voxel at the given coordinates.
    const ValueType& getValue(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        if (this->isHashed(xyz)) {
            assert(mNode0);
            return mNode0->getValueAndCache(xyz, this->self());
        }
        return BaseT::mTree->root().getValueAndCache(xyz, this->self());
    }

    /// Return the active state of the voxel at the given coordinates.
    bool isValueOn(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        if (this->isHashed(xyz)) {
            assert(mNode0);
            return mNode0->isValueOnAndCache(xyz, this->self());
        }
        return BaseT::mTree->root().isValueOnAndCache(xyz, this->self());
    }

    /// Return the active state of the voxel as well as its value
    bool probeValue(const Coord& xyz, ValueType& value) const
    {
        assert(BaseT::mTree);
        if (this->isHashed(xyz)) {
            assert(mNode0);
            return mNode0->probeValueAndCache(xyz, value, this->self());
        }
        return BaseT::mTree->root().probeValueAndCache(xyz, value, this->self());
    }

    /// Return the tree depth (0 = root) at which the value of voxel (x, y, z) resides,
    /// or -1 if (x, y, z) isn't explicitly represented in the tree (i.e., if it is
    /// implicitly a background voxel).
    int getValueDepth(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        if (this->isHashed(xyz)) {
            assert(mNode0);
            return RootNodeT::LEVEL - mNode0->getValueLevelAndCache(xyz, this->self());
        }
        return BaseT::mTree->root().getValueDepthAndCache(xyz, this->self());
    }

    /// Return @c true if the value of voxel (x, y, z) resides at the leaf level
    /// of the tree, i.e., if it is not a tile value.
    bool isVoxel(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        if (this->isHashed(xyz)) {
            assert(mNode0);
            return mNode0->getValueLevelAndCache(xyz, this->self()) == 0;
        }
        return BaseT::mTree->root().getValueDepthAndCache(xyz, this->self()) ==
               static_cast<int>(RootNodeT::LEVEL);
    }

    //@{
    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValue(const Coord& xyz, const ValueType& value)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->setValueAndCache(xyz, value, *this);
        } else {
            BaseT::mTree->root().setValueAndCache(xyz, value, *this);
        }
    }
    void setValueOn(const Coord& xyz, const ValueType& value) { this->setValue(xyz, value); }
    //@}

    /// Set the value of the voxel at the given coordinate but preserves its active state.
    void setValueOnly(const Coord& xyz, const ValueType& value)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->setValueOnlyAndCache(xyz, value, *this);
        } else {
            BaseT::mTree->root().setValueOnlyAndCache(xyz, value, *this);
        }
    }

    /// Set the value of the voxel at the given coordinates and mark the voxel as inactive.
    void setValueOff(const Coord& xyz, const ValueType& value)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->setValueOffAndCache(xyz, value, *this);
        } else {
            BaseT::mTree->root().setValueOffAndCache(xyz, value, *this);
        }
    }

    /// @brief Apply a functor to the value of the voxel at the given coordinates
    /// and mark the voxel as active.
    /// @details See Tree::modifyValue() for details.
    template<typename ModifyOp>
    void modifyValue(const Coord& xyz, const ModifyOp& op)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->modifyValueAndCache(xyz, op, *this);
        } else {
            BaseT::mTree->root().modifyValueAndCache(xyz, op, *this);
        }
    }

    /// @brief Apply a functor to the voxel at the given coordinates.
    /// @details See Tree::modifyValueAndActiveState() for details.
    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->modifyValueAndActiveStateAndCache(xyz, op, *this);
        } else {
            BaseT::mTree->root().modifyValueAndActiveStateAndCache(xyz, op, *this);
        }
    }

    /// Set the active state of the voxel at the given coordinates but don't change its value.
    void setActiveState(const Coord& xyz, bool on = true)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->setActiveStateAndCache(xyz, on, *this);
        } else {
            BaseT::mTree->root().setActiveStateAndCache(xyz, on, *this);
        }
    }
    /// Mark the voxel at the given coordinates as active but don't change its value.
    void setValueOn(const Coord& xyz) { this->setActiveState(xyz, true); }
    /// Mark the voxel at the given coordinates as inactive but don't change its value.
    void setValueOff(const Coord& xyz) { this->setActiveState(xyz, false); }

    /// Return the cached node of type @a NodeType.  [Mainly for internal use]
    template<typename NodeT>
    NodeT* getNode()
    {
        const NodeT* node = nullptr;
        this->getNode(node);
        return const_cast<NodeT*>(node);
    }

    /// Cache the given node, which should lie along the path from the root node to
    /// the node containing voxel (x, y, z).  [Mainly for internal use]
    template<typename NodeT>
    void insertNode(const Coord& xyz, NodeT& node) { this->insert(xyz, &node); }

    /// If a node of the given type exists in the cache, remove it, so that
    /// isCached(xyz) returns @c false for any voxel (x, y, z) contained in
    /// that node.  [Mainly for internal use]
    template<typename NodeT>
    void eraseNode()
    {
        const NodeT* node = nullptr;
        this->eraseNode(node);
    }

    /// @brief Add the specified leaf to this tree, possibly creating a child branch
    /// in the process.  If the leaf node already exists, replace it.
    void addLeaf(LeafNodeT* leaf)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't add a node to a const tree");
        BaseT::mTree->root().addLeaf(leaf);
    }

    /// @brief Add a tile at the specified tree level that contains voxel (x, y, z),
    /// possibly deleting existing nodes or creating new nodes in the process.
    void addTile(Index level, const Coord& xyz, const ValueType& value, bool state)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't add a tile to a const tree");
        BaseT::mTree->root().addTile(level, xyz, value, state);
    }

    /// @brief @return the leaf node that contains voxel (x, y, z) and
    /// if it doesn't exist, create it, but preserve the values and
    /// active states of all voxels.
    ///
    /// Use this method to preallocate a static tree topology over which to
    /// safely perform multithreaded processing.
    LeafNodeT* touchLeaf(const Coord& xyz)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't get a non-const node from a const tree");
        if (this->isHashed(xyz)) {
            assert(mNode0);
            return const_cast<NodeT0*>(mNode0)->touchLeafAndCache(xyz, *this);
        }
        return BaseT::mTree->root().touchLeafAndCache(xyz, *this);
    }

    /// @brief @return a pointer to the node of the specified type that contains
    /// voxel (x, y, z) and if it doesn't exist, return @c nullptr.
    template<typename NodeT>
    NodeT* probeNode(const Coord& xyz)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't get a non-const node from a const tree");
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if ((std::is_same<NodeT, NodeT0>::value)) {
            if (this->isHashed(xyz)) {
                assert(mNode0);
                return reinterpret_cast<NodeT*>(const_cast<NodeT0*>(mNode0));
            }
            return BaseT::mTree->root().template probeNodeAndCache<NodeT>(xyz, *this);
        }
        return nullptr;
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    LeafNodeT* probeLeaf(const Coord& xyz)
    {
        return this->template probeNode<LeafNodeT>(xyz);
    }

    /// @brief @return a const pointer to the nodeof the specified type that contains
    /// voxel (x, y, z) and if it doesn't exist, return @c nullptr.
    template<typename NodeT>
    const NodeT* probeConstNode(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if ((std::is_same<NodeT, NodeT0>::value)) {
            if (this->isHashed(xyz)) {
                assert(mNode0);
                return reinterpret_cast<const NodeT*>(mNode0);
            }
            return BaseT::mTree->root().template probeConstNodeAndCache<NodeT>(xyz, this->self());
        }
        return nullptr;
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    const LeafNodeT* probeConstLeaf(const Coord& xyz) const
    {
        return this->template probeConstNode<LeafNodeT>(xyz);
    }
    const LeafNodeT* probeLeaf(const Coord& xyz) const { return this->probeConstLeaf(xyz); }

    /// Remove all the cached nodes and invalidate the corresponding hash-keys.
    void clear() override
    {
        mKey0  = Coord::max();
        mNode0 = nullptr;
    }

private:
    // Allow nodes to insert themselves into the cache.
    template<typename> friend class RootNode;
    template<typename, Index> friend class InternalNode;
    template<typename, Index> friend class LeafNode;
    // Allow trees to deregister themselves.
    template<typename> friend class Tree;

    // This private method is merely for convenience.
    inline ValueAccessor1& self() const { return const_cast<ValueAccessor1&>(*this); }

    void getNode(const NodeT0*& node) { node = mNode0; }
    void getNode(const RootNodeT*& node)
    {
        node = (BaseT::mTree ? &BaseT::mTree->root() : nullptr);
    }
    template<typename OtherNodeType> void getNode(const OtherNodeType*& node) { node = nullptr; }
    void eraseNode(const NodeT0*) { mKey0 = Coord::max(); mNode0 = nullptr; }
    template<typename OtherNodeType> void eraseNode(const OtherNodeType*) {}

    /// Private copy method
    inline void copy(const ValueAccessor1& other)
    {
        mKey0  = other.mKey0;
        mNode0 = other.mNode0;
    }

    /// Prevent this accessor from calling Tree::releaseCache() on a tree that
    /// no longer exists.  (Called by mTree when it is destroyed.)
    void release() override
    {
        this->BaseT::release();
        this->clear();
    }
    /// Cache the given node, which should lie along the path from the root node to
    /// the node containing voxel (x, y, z).
    /// @note This operation is not mutex-protected and is intended to be called
    /// only by nodes and only in the context of a getValue() or setValue() call.
    inline void insert(const Coord& xyz, const NodeT0* node)
    {
        assert(node);
        mKey0  = xyz & ~(NodeT0::DIM-1);
        mNode0 = node;
    }

    /// No-op in case a tree traversal attemps to insert a node that
    /// is not cached by the ValueAccessor
    template<typename OtherNodeType> inline void insert(const Coord&, const OtherNodeType*) {}

    inline bool isHashed(const Coord& xyz) const
    {
        return (xyz[0] & ~Coord::ValueType(NodeT0::DIM-1)) == mKey0[0]
            && (xyz[1] & ~Coord::ValueType(NodeT0::DIM-1)) == mKey0[1]
            && (xyz[2] & ~Coord::ValueType(NodeT0::DIM-1)) == mKey0[2];
    }
    mutable Coord mKey0;
    mutable const NodeT0* mNode0;
}; // ValueAccessor1


/// @brief Value accessor with two levels of node caching.
/// @details The node cache levels are specified by L0 and L1
/// with the default values 0 and 1 (defined in the forward declaration)
/// corresponding to a LeafNode and its parent InternalNode.
///
/// @note This class is for experts only and should rarely be used directly.
/// Instead use ValueAccessor with its default template arguments.
template<typename _TreeType, bool IsSafe, Index L0, Index L1>
class ValueAccessor2 : public ValueAccessorBase<_TreeType, IsSafe>
{
public:
    static_assert(_TreeType::DEPTH >= 3, "cache size exceeds tree depth");
    static_assert(L0 < L1, "invalid cache level");
    static_assert(L1 < _TreeType::RootNodeType::LEVEL, "invalid cache level");

    using TreeType = _TreeType;
    using ValueType = typename TreeType::ValueType;
    using RootNodeT = typename TreeType::RootNodeType;
    using LeafNodeT = typename TreeType::LeafNodeType;
    using BaseT = ValueAccessorBase<TreeType, IsSafe>;
    using InvTreeT = typename RootNodeT::NodeChainType;
    using NodeT0 = typename boost::mpl::at<InvTreeT, boost::mpl::int_<L0>>::type;
    using NodeT1 = typename boost::mpl::at<InvTreeT, boost::mpl::int_<L1>>::type;

    /// Constructor from a tree
    ValueAccessor2(TreeType& tree) : BaseT(tree),
                                     mKey0(Coord::max()), mNode0(nullptr),
                                     mKey1(Coord::max()), mNode1(nullptr) {}

    /// Copy constructor
    ValueAccessor2(const ValueAccessor2& other) : BaseT(other) { this->copy(other); }

    /// Return the number of cache levels employed by this ValueAccessor
    static Index numCacheLevels() { return 2; }

    /// Asignment operator
    ValueAccessor2& operator=(const ValueAccessor2& other)
    {
        if (&other != this) {
            this->BaseT::operator=(other);
            this->copy(other);
        }
        return *this;
    }

    /// Virtual destructor
    ~ValueAccessor2() override = default;

    /// Return @c true if any of the nodes along the path to the given
    /// voxel have been cached.
    bool isCached(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        return this->isHashed1(xyz) || this->isHashed0(xyz);
    }

    /// Return the value of the voxel at the given coordinates.
    const ValueType& getValue(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            return mNode0->getValueAndCache(xyz, this->self());
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            return mNode1->getValueAndCache(xyz, this->self());
        }
        return BaseT::mTree->root().getValueAndCache(xyz, this->self());
    }

    /// Return the active state of the voxel at the given coordinates.
    bool isValueOn(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            return mNode0->isValueOnAndCache(xyz, this->self());
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            return mNode1->isValueOnAndCache(xyz, this->self());
        }
        return BaseT::mTree->root().isValueOnAndCache(xyz, this->self());
    }

    /// Return the active state of the voxel as well as its value
    bool probeValue(const Coord& xyz, ValueType& value) const
    {
        assert(BaseT::mTree);
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            return mNode0->probeValueAndCache(xyz, value, this->self());
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            return mNode1->probeValueAndCache(xyz, value, this->self());
        }
        return BaseT::mTree->root().probeValueAndCache(xyz, value, this->self());
    }

    /// Return the tree depth (0 = root) at which the value of voxel (x, y, z) resides,
    /// or -1 if (x, y, z) isn't explicitly represented in the tree (i.e., if it is
    /// implicitly a background voxel).
    int getValueDepth(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            return RootNodeT::LEVEL - mNode0->getValueLevelAndCache(xyz, this->self());
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            return RootNodeT::LEVEL - mNode1->getValueLevelAndCache(xyz, this->self());
        }
        return BaseT::mTree->root().getValueDepthAndCache(xyz, this->self());
    }

    /// Return @c true if the value of voxel (x, y, z) resides at the leaf level
    /// of the tree, i.e., if it is not a tile value.
    bool isVoxel(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            return mNode0->getValueLevelAndCache(xyz, this->self())==0;
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            return mNode1->getValueLevelAndCache(xyz, this->self())==0;
        }
        return BaseT::mTree->root().getValueDepthAndCache(xyz, this->self()) ==
               static_cast<int>(RootNodeT::LEVEL);
    }

    //@{
    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValue(const Coord& xyz, const ValueType& value)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->setValueAndCache(xyz, value, *this);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            const_cast<NodeT1*>(mNode1)->setValueAndCache(xyz, value, *this);
        } else {
            BaseT::mTree->root().setValueAndCache(xyz, value, *this);
        }
    }
    void setValueOn(const Coord& xyz, const ValueType& value) { this->setValue(xyz, value); }
    //@}

    /// Set the value of the voxel at the given coordinate but preserves its active state.
    void setValueOnly(const Coord& xyz, const ValueType& value)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->setValueOnlyAndCache(xyz, value, *this);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            const_cast<NodeT1*>(mNode1)->setValueOnlyAndCache(xyz, value, *this);
        } else {
            BaseT::mTree->root().setValueOnlyAndCache(xyz, value, *this);
        }
    }

    /// Set the value of the voxel at the given coordinates and mark the voxel as inactive.
    void setValueOff(const Coord& xyz, const ValueType& value)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->setValueOffAndCache(xyz, value, *this);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            const_cast<NodeT1*>(mNode1)->setValueOffAndCache(xyz, value, *this);
        } else {
            BaseT::mTree->root().setValueOffAndCache(xyz, value, *this);
        }
    }

    /// @brief Apply a functor to the value of the voxel at the given coordinates
    /// and mark the voxel as active.
    /// @details See Tree::modifyValue() for details.
    template<typename ModifyOp>
    void modifyValue(const Coord& xyz, const ModifyOp& op)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->modifyValueAndCache(xyz, op, *this);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            const_cast<NodeT1*>(mNode1)->modifyValueAndCache(xyz, op, *this);
        } else {
            BaseT::mTree->root().modifyValueAndCache(xyz, op, *this);
        }
    }

    /// @brief Apply a functor to the voxel at the given coordinates.
    /// @details See Tree::modifyValueAndActiveState() for details.
    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->modifyValueAndActiveStateAndCache(xyz, op, *this);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            const_cast<NodeT1*>(mNode1)->modifyValueAndActiveStateAndCache(xyz, op, *this);
        } else {
            BaseT::mTree->root().modifyValueAndActiveStateAndCache(xyz, op, *this);
        }
    }

    /// Set the active state of the voxel at the given coordinates without changing its value.
    void setActiveState(const Coord& xyz, bool on = true)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->setActiveStateAndCache(xyz, on, *this);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            const_cast<NodeT1*>(mNode1)->setActiveStateAndCache(xyz, on, *this);
        } else {
            BaseT::mTree->root().setActiveStateAndCache(xyz, on, *this);
        }
    }
    /// Mark the voxel at the given coordinates as active without changing its value.
    void setValueOn(const Coord& xyz) { this->setActiveState(xyz, true); }
    /// Mark the voxel at the given coordinates as inactive without changing its value.
    void setValueOff(const Coord& xyz) { this->setActiveState(xyz, false); }

    /// Return the cached node of type @a NodeType.  [Mainly for internal use]
    template<typename NodeT>
    NodeT* getNode()
    {
        const NodeT* node = nullptr;
        this->getNode(node);
        return const_cast<NodeT*>(node);
    }

    /// Cache the given node, which should lie along the path from the root node to
    /// the node containing voxel (x, y, z).  [Mainly for internal use]
    template<typename NodeT>
    void insertNode(const Coord& xyz, NodeT& node) { this->insert(xyz, &node); }

    /// If a node of the given type exists in the cache, remove it, so that
    /// isCached(xyz) returns @c false for any voxel (x, y, z) contained in
    /// that node.  [Mainly for internal use]
    template<typename NodeT>
    void eraseNode()
    {
        const NodeT* node = nullptr;
        this->eraseNode(node);
    }

    /// @brief Add the specified leaf to this tree, possibly creating a child branch
    /// in the process.  If the leaf node already exists, replace it.
    void addLeaf(LeafNodeT* leaf)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't add a node to a const tree");
        if (this->isHashed1(leaf->origin())) {
            assert(mNode1);
            return const_cast<NodeT1*>(mNode1)->addLeafAndCache(leaf, *this);
        }
        BaseT::mTree->root().addLeafAndCache(leaf, *this);
    }

    /// @brief Add a tile at the specified tree level that contains voxel (x, y, z),
    /// possibly deleting existing nodes or creating new nodes in the process.
    void addTile(Index level, const Coord& xyz, const ValueType& value, bool state)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't add a tile to a const tree");
        if (this->isHashed1(xyz)) {
            assert(mNode1);
            return const_cast<NodeT1*>(mNode1)->addTileAndCache(level, xyz, value, state, *this);
        }
        BaseT::mTree->root().addTileAndCache(level, xyz, value, state, *this);
    }

    /// @brief @return the leaf node that contains voxel (x, y, z) and
    /// if it doesn't exist, create it, but preserve the values and
    /// active states of all voxels.
    ///
    /// Use this method to preallocate a static tree topology over which to
    /// safely perform multithreaded processing.
    LeafNodeT* touchLeaf(const Coord& xyz)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't get a non-const node from a const tree");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            return const_cast<NodeT0*>(mNode0)->touchLeafAndCache(xyz, *this);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            return const_cast<NodeT1*>(mNode1)->touchLeafAndCache(xyz, *this);
        }
        return BaseT::mTree->root().touchLeafAndCache(xyz, *this);
    }
    /// @brief @return a pointer to the node of the specified type that contains
    /// voxel (x, y, z) and if it doesn't exist, return @c nullptr.
    template<typename NodeT>
    NodeT* probeNode(const Coord& xyz)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't get a non-const node from a const tree");
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if ((std::is_same<NodeT, NodeT0>::value)) {
            if (this->isHashed0(xyz)) {
                assert(mNode0);
                return reinterpret_cast<NodeT*>(const_cast<NodeT0*>(mNode0));
            } else if (this->isHashed1(xyz)) {
                assert(mNode1);
                return const_cast<NodeT1*>(mNode1)->template probeNodeAndCache<NodeT>(xyz, *this);
            }
            return BaseT::mTree->root().template probeNodeAndCache<NodeT>(xyz, *this);
        } else if ((std::is_same<NodeT, NodeT1>::value)) {
            if (this->isHashed1(xyz)) {
                assert(mNode1);
                return reinterpret_cast<NodeT*>(const_cast<NodeT1*>(mNode1));
            }
            return BaseT::mTree->root().template probeNodeAndCache<NodeT>(xyz, *this);
        }
        return nullptr;
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    /// @brief @return a pointer to the leaf node that contains
    /// voxel (x, y, z) and if it doesn't exist, return @c nullptr.
    LeafNodeT* probeLeaf(const Coord& xyz) { return this->template probeNode<LeafNodeT>(xyz); }

    /// @brief @return a const pointer to the node of the specified type that contains
    /// voxel (x, y, z) and if it doesn't exist, return @c nullptr.
    template<typename NodeT>
    const NodeT* probeConstLeaf(const Coord& xyz) const
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if ((std::is_same<NodeT, NodeT0>::value)) {
            if (this->isHashed0(xyz)) {
                assert(mNode0);
                return reinterpret_cast<const NodeT*>(mNode0);
            } else if (this->isHashed1(xyz)) {
                assert(mNode1);
                return mNode1->template probeConstNodeAndCache<NodeT>(xyz, this->self());
            }
            return BaseT::mTree->root().template probeConstNodeAndCache<NodeT>(xyz, this->self());
        } else if ((std::is_same<NodeT, NodeT1>::value)) {
            if (this->isHashed1(xyz)) {
                assert(mNode1);
                return reinterpret_cast<const NodeT*>(mNode1);
            }
            return BaseT::mTree->root().template probeConstNodeAndCache<NodeT>(xyz, this->self());
        }
        return nullptr;
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    /// @brief @return a const pointer to the leaf node that contains
    /// voxel (x, y, z) and if it doesn't exist, return @c nullptr.
    const LeafNodeT* probeConstLeaf(const Coord& xyz) const
    {
        return this->template probeConstNode<LeafNodeT>(xyz);
    }
    const LeafNodeT* probeLeaf(const Coord& xyz) const { return this->probeConstLeaf(xyz); }

    /// @brief @return a const pointer to the node of the specified type that contains
    /// voxel (x, y, z) and if it doesn't exist, return @c nullptr.
    template<typename NodeT>
    const NodeT* probeConstNode(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if ((std::is_same<NodeT, NodeT0>::value)) {
            if (this->isHashed0(xyz)) {
                assert(mNode0);
                return reinterpret_cast<const NodeT*>(mNode0);
            } else if (this->isHashed1(xyz)) {
                assert(mNode1);
                return mNode1->template probeConstNodeAndCache<NodeT>(xyz, this->self());
            }
            return BaseT::mTree->root().template probeConstNodeAndCache<NodeT>(xyz, this->self());
        } else if ((std::is_same<NodeT, NodeT1>::value)) {
            if (this->isHashed1(xyz)) {
                assert(mNode1);
                return reinterpret_cast<const NodeT*>(mNode1);
            }
            return BaseT::mTree->root().template probeConstNodeAndCache<NodeT>(xyz, this->self());
        }
        return nullptr;
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }

    /// Remove all the cached nodes and invalidate the corresponding hash-keys.
    void clear() override
    {
        mKey0  = Coord::max();
        mNode0 = nullptr;
        mKey1  = Coord::max();
        mNode1 = nullptr;
    }

private:
    // Allow nodes to insert themselves into the cache.
    template<typename> friend class RootNode;
    template<typename, Index> friend class InternalNode;
    template<typename, Index> friend class LeafNode;
    // Allow trees to deregister themselves.
    template<typename> friend class Tree;

    // This private method is merely for convenience.
    inline ValueAccessor2& self() const { return const_cast<ValueAccessor2&>(*this); }

    void getNode(const NodeT0*& node) { node = mNode0; }
    void getNode(const NodeT1*& node) { node = mNode1; }
    void getNode(const RootNodeT*& node)
    {
        node = (BaseT::mTree ? &BaseT::mTree->root() : nullptr);
    }
    template<typename OtherNodeType> void getNode(const OtherNodeType*& node) { node = nullptr; }

    void eraseNode(const NodeT0*) { mKey0 = Coord::max(); mNode0 = nullptr; }
    void eraseNode(const NodeT1*) { mKey1 = Coord::max(); mNode1 = nullptr; }
    template<typename OtherNodeType> void eraseNode(const OtherNodeType*) {}

    /// Private copy method
    inline void copy(const ValueAccessor2& other)
    {
        mKey0  = other.mKey0;
        mNode0 = other.mNode0;
        mKey1  = other.mKey1;
        mNode1 = other.mNode1;
    }

    /// Prevent this accessor from calling Tree::releaseCache() on a tree that
    /// no longer exists.  (Called by mTree when it is destroyed.)
    void release() override
    {
        this->BaseT::release();
        this->clear();
    }

    /// Cache the given node, which should lie along the path from the root node to
    /// the node containing voxel (x, y, z).
    /// @note This operation is not mutex-protected and is intended to be called
    /// only by nodes and only in the context of a getValue() or setValue() call.
    inline void insert(const Coord& xyz, const NodeT0* node)
    {
        assert(node);
        mKey0  = xyz & ~(NodeT0::DIM-1);
        mNode0 = node;
    }
    inline void insert(const Coord& xyz, const NodeT1* node)
    {
        assert(node);
        mKey1  = xyz & ~(NodeT1::DIM-1);
        mNode1 = node;
    }
    /// No-op in case a tree traversal attemps to insert a node that
    /// is not cached by the ValueAccessor
    template<typename NodeT> inline void insert(const Coord&, const NodeT*) {}

    inline bool isHashed0(const Coord& xyz) const
    {
        return (xyz[0] & ~Coord::ValueType(NodeT0::DIM-1)) == mKey0[0]
            && (xyz[1] & ~Coord::ValueType(NodeT0::DIM-1)) == mKey0[1]
            && (xyz[2] & ~Coord::ValueType(NodeT0::DIM-1)) == mKey0[2];
    }
    inline bool isHashed1(const Coord& xyz) const
    {
        return (xyz[0] & ~Coord::ValueType(NodeT1::DIM-1)) == mKey1[0]
            && (xyz[1] & ~Coord::ValueType(NodeT1::DIM-1)) == mKey1[1]
            && (xyz[2] & ~Coord::ValueType(NodeT1::DIM-1)) == mKey1[2];
    }
    mutable Coord mKey0;
    mutable const NodeT0* mNode0;
    mutable Coord mKey1;
    mutable const NodeT1* mNode1;
}; // ValueAccessor2


/// @brief Value accessor with three levels of node caching.
/// @details The node cache levels are specified by L0, L1, and L2
/// with the default values 0, 1 and 2 (defined in the forward declaration)
/// corresponding to a LeafNode, its parent InternalNode, and its parent InternalNode.
/// Since the default configuration of all typed trees and grids, e.g.,
/// FloatTree or FloatGrid, has a depth of four, this value accessor is the one
/// used by default.
///
/// @note This class is for experts only and should rarely be used
/// directly. Instead use ValueAccessor with its default template arguments
template<typename _TreeType, bool IsSafe, Index L0, Index L1, Index L2>
class ValueAccessor3 : public ValueAccessorBase<_TreeType, IsSafe>
{
public:
    static_assert(_TreeType::DEPTH >= 4, "cache size exceeds tree depth");
    static_assert(L0 < L1, "invalid cache level");
    static_assert(L1 < L2, "invalid cache level");
    static_assert(L2 < _TreeType::RootNodeType::LEVEL, "invalid cache level");

    using TreeType = _TreeType;
    using ValueType = typename TreeType::ValueType;
    using RootNodeT = typename TreeType::RootNodeType;
    using LeafNodeT = typename TreeType::LeafNodeType;
    using BaseT = ValueAccessorBase<TreeType, IsSafe>;
    using InvTreeT = typename RootNodeT::NodeChainType;
    using NodeT0 = typename boost::mpl::at<InvTreeT, boost::mpl::int_<L0> >::type;
    using NodeT1 = typename boost::mpl::at<InvTreeT, boost::mpl::int_<L1> >::type;
    using NodeT2 = typename boost::mpl::at<InvTreeT, boost::mpl::int_<L2> >::type;

    /// Constructor from a tree
    ValueAccessor3(TreeType& tree) : BaseT(tree),
                                     mKey0(Coord::max()), mNode0(nullptr),
                                     mKey1(Coord::max()), mNode1(nullptr),
                                     mKey2(Coord::max()), mNode2(nullptr) {}

    /// Copy constructor
    ValueAccessor3(const ValueAccessor3& other) : BaseT(other) { this->copy(other); }

    /// Asignment operator
    ValueAccessor3& operator=(const ValueAccessor3& other)
    {
        if (&other != this) {
            this->BaseT::operator=(other);
            this->copy(other);
        }
        return *this;
    }

    /// Return the number of cache levels employed by this ValueAccessor
    static Index numCacheLevels() { return 3; }

    /// Virtual destructor
    ~ValueAccessor3() override = default;

    /// Return @c true if any of the nodes along the path to the given
    /// voxel have been cached.
    bool isCached(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        return this->isHashed2(xyz) || this->isHashed1(xyz) || this->isHashed0(xyz);
    }

    /// Return the value of the voxel at the given coordinates.
    const ValueType& getValue(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            return mNode0->getValueAndCache(xyz, this->self());
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            return mNode1->getValueAndCache(xyz, this->self());
        } else if (this->isHashed2(xyz)) {
            assert(mNode2);
            return mNode2->getValueAndCache(xyz, this->self());
        }
        return BaseT::mTree->root().getValueAndCache(xyz, this->self());
    }

    /// Return the active state of the voxel at the given coordinates.
    bool isValueOn(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            return mNode0->isValueOnAndCache(xyz, this->self());
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            return mNode1->isValueOnAndCache(xyz, this->self());
        } else if (this->isHashed2(xyz)) {
            assert(mNode2);
            return mNode2->isValueOnAndCache(xyz, this->self());
        }
        return BaseT::mTree->root().isValueOnAndCache(xyz, this->self());
    }

    /// Return the active state of the voxel as well as its value
    bool probeValue(const Coord& xyz, ValueType& value) const
    {
        assert(BaseT::mTree);
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            return mNode0->probeValueAndCache(xyz, value, this->self());
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            return mNode1->probeValueAndCache(xyz, value, this->self());
        } else if (this->isHashed2(xyz)) {
            assert(mNode2);
            return mNode2->probeValueAndCache(xyz, value, this->self());
        }
        return BaseT::mTree->root().probeValueAndCache(xyz, value, this->self());
    }

    /// Return the tree depth (0 = root) at which the value of voxel (x, y, z) resides,
    /// or -1 if (x, y, z) isn't explicitly represented in the tree (i.e., if it is
    /// implicitly a background voxel).
    int getValueDepth(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            return RootNodeT::LEVEL - mNode0->getValueLevelAndCache(xyz, this->self());
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            return RootNodeT::LEVEL - mNode1->getValueLevelAndCache(xyz, this->self());
        } else if (this->isHashed2(xyz)) {
            assert(mNode2);
            return RootNodeT::LEVEL - mNode2->getValueLevelAndCache(xyz, this->self());
        }
        return BaseT::mTree->root().getValueDepthAndCache(xyz, this->self());
    }

    /// Return @c true if the value of voxel (x, y, z) resides at the leaf level
    /// of the tree, i.e., if it is not a tile value.
    bool isVoxel(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            return mNode0->getValueLevelAndCache(xyz, this->self())==0;
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            return mNode1->getValueLevelAndCache(xyz, this->self())==0;
        } else if (this->isHashed2(xyz)) {
            assert(mNode2);
            return mNode2->getValueLevelAndCache(xyz, this->self())==0;
        }
        return BaseT::mTree->root().getValueDepthAndCache(xyz, this->self()) ==
               static_cast<int>(RootNodeT::LEVEL);
    }

    //@{
    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValue(const Coord& xyz, const ValueType& value)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->setValueAndCache(xyz, value, *this);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            const_cast<NodeT1*>(mNode1)->setValueAndCache(xyz, value, *this);
        } else if (this->isHashed2(xyz)) {
            assert(mNode2);
            const_cast<NodeT2*>(mNode2)->setValueAndCache(xyz, value, *this);
        } else {
            BaseT::mTree->root().setValueAndCache(xyz, value, *this);
        }
    }
    void setValueOn(const Coord& xyz, const ValueType& value) { this->setValue(xyz, value); }
    //@}

    /// Set the value of the voxel at the given coordinate but preserves its active state.
    void setValueOnly(const Coord& xyz, const ValueType& value)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->setValueOnlyAndCache(xyz, value, *this);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            const_cast<NodeT1*>(mNode1)->setValueOnlyAndCache(xyz, value, *this);
        } else if (this->isHashed2(xyz)) {
            assert(mNode2);
            const_cast<NodeT2*>(mNode2)->setValueOnlyAndCache(xyz, value, *this);
        } else {
            BaseT::mTree->root().setValueOnlyAndCache(xyz, value, *this);
        }
    }

    /// Set the value of the voxel at the given coordinates and mark the voxel as inactive.
    void setValueOff(const Coord& xyz, const ValueType& value)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->setValueOffAndCache(xyz, value, *this);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            const_cast<NodeT1*>(mNode1)->setValueOffAndCache(xyz, value, *this);
        } else if (this->isHashed2(xyz)) {
            assert(mNode2);
            const_cast<NodeT2*>(mNode2)->setValueOffAndCache(xyz, value, *this);
        } else {
            BaseT::mTree->root().setValueOffAndCache(xyz, value, *this);
        }
    }

    /// @brief Apply a functor to the value of the voxel at the given coordinates
    /// and mark the voxel as active.
    /// @details See Tree::modifyValue() for details.
    template<typename ModifyOp>
    void modifyValue(const Coord& xyz, const ModifyOp& op)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->modifyValueAndCache(xyz, op, *this);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            const_cast<NodeT1*>(mNode1)->modifyValueAndCache(xyz, op, *this);
        } else if (this->isHashed2(xyz)) {
            assert(mNode2);
            const_cast<NodeT2*>(mNode2)->modifyValueAndCache(xyz, op, *this);
        } else {
            BaseT::mTree->root().modifyValueAndCache(xyz, op, *this);
        }
    }

    /// @brief Apply a functor to the voxel at the given coordinates.
    /// @details See Tree::modifyValueAndActiveState() for details.
    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->modifyValueAndActiveStateAndCache(xyz, op, *this);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            const_cast<NodeT1*>(mNode1)->modifyValueAndActiveStateAndCache(xyz, op, *this);
        } else if (this->isHashed2(xyz)) {
            assert(mNode2);
            const_cast<NodeT2*>(mNode2)->modifyValueAndActiveStateAndCache(xyz, op, *this);
        } else {
            BaseT::mTree->root().modifyValueAndActiveStateAndCache(xyz, op, *this);
        }
    }

    /// Set the active state of the voxel at the given coordinates without changing its value.
    void setActiveState(const Coord& xyz, bool on = true)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            const_cast<NodeT0*>(mNode0)->setActiveStateAndCache(xyz, on, *this);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            const_cast<NodeT1*>(mNode1)->setActiveStateAndCache(xyz, on, *this);
        } else if (this->isHashed2(xyz)) {
            assert(mNode2);
            const_cast<NodeT2*>(mNode2)->setActiveStateAndCache(xyz, on, *this);
        } else {
            BaseT::mTree->root().setActiveStateAndCache(xyz, on, *this);
        }
    }
    /// Mark the voxel at the given coordinates as active without changing its value.
    void setValueOn(const Coord& xyz) { this->setActiveState(xyz, true); }
    /// Mark the voxel at the given coordinates as inactive without changing its value.
    void setValueOff(const Coord& xyz) { this->setActiveState(xyz, false); }

    /// Return the cached node of type @a NodeType.  [Mainly for internal use]
    template<typename NodeT>
    NodeT* getNode()
    {
        const NodeT* node = nullptr;
        this->getNode(node);
        return const_cast<NodeT*>(node);
    }

    /// Cache the given node, which should lie along the path from the root node to
    /// the node containing voxel (x, y, z).  [Mainly for internal use]
    template<typename NodeT>
    void insertNode(const Coord& xyz, NodeT& node) { this->insert(xyz, &node); }

    /// If a node of the given type exists in the cache, remove it, so that
    /// isCached(xyz) returns @c false for any voxel (x, y, z) contained in
    /// that node.  [Mainly for internal use]
    template<typename NodeT>
    void eraseNode()
    {
        const NodeT* node = nullptr;
        this->eraseNode(node);
    }

    /// @brief Add the specified leaf to this tree, possibly creating a child branch
    /// in the process.  If the leaf node already exists, replace it.
    void addLeaf(LeafNodeT* leaf)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't add a node to a const tree");
        if (this->isHashed1(leaf->origin())) {
            assert(mNode1);
            return const_cast<NodeT1*>(mNode1)->addLeafAndCache(leaf, *this);
        } else if (this->isHashed2(leaf->origin())) {
            assert(mNode2);
            return const_cast<NodeT2*>(mNode2)->addLeafAndCache(leaf, *this);
        }
        BaseT::mTree->root().addLeafAndCache(leaf, *this);
    }

    /// @brief Add a tile at the specified tree level that contains voxel (x, y, z),
    /// possibly deleting existing nodes or creating new nodes in the process.
    void addTile(Index level, const Coord& xyz, const ValueType& value, bool state)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't add a tile to a const tree");
        if (this->isHashed1(xyz)) {
            assert(mNode1);
            return const_cast<NodeT1*>(mNode1)->addTileAndCache(level, xyz, value, state, *this);
        } if (this->isHashed2(xyz)) {
            assert(mNode2);
            return const_cast<NodeT2*>(mNode2)->addTileAndCache(level, xyz, value, state, *this);
        }
        BaseT::mTree->root().addTileAndCache(level, xyz, value, state, *this);
    }

    /// @brief @return the leaf node that contains voxel (x, y, z) and
    /// if it doesn't exist, create it, but preserve the values and
    /// active states of all voxels.
    ///
    /// Use this method to preallocate a static tree topology over which to
    /// safely perform multithreaded processing.
    LeafNodeT* touchLeaf(const Coord& xyz)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't get a non-const node from a const tree");
        if (this->isHashed0(xyz)) {
            assert(mNode0);
            return const_cast<NodeT0*>(mNode0);
        } else if (this->isHashed1(xyz)) {
            assert(mNode1);
            return const_cast<NodeT1*>(mNode1)->touchLeafAndCache(xyz, *this);
        } else if (this->isHashed2(xyz)) {
            assert(mNode2);
            return const_cast<NodeT2*>(mNode2)->touchLeafAndCache(xyz, *this);
        }
        return BaseT::mTree->root().touchLeafAndCache(xyz, *this);
    }
    /// @brief @return a pointer to the node of the specified type that contains
    /// voxel (x, y, z) and if it doesn't exist, return @c nullptr.
    template<typename NodeT>
    NodeT* probeNode(const Coord& xyz)
    {
        assert(BaseT::mTree);
        static_assert(!BaseT::IsConstTree, "can't get a non-const node from a const tree");
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if ((std::is_same<NodeT, NodeT0>::value)) {
            if (this->isHashed0(xyz)) {
                assert(mNode0);
                return reinterpret_cast<NodeT*>(const_cast<NodeT0*>(mNode0));
            } else if (this->isHashed1(xyz)) {
                assert(mNode1);
                return const_cast<NodeT1*>(mNode1)->template probeNodeAndCache<NodeT>(xyz, *this);
            } else if (this->isHashed2(xyz)) {
                assert(mNode2);
                return const_cast<NodeT2*>(mNode2)->template probeNodeAndCache<NodeT>(xyz, *this);
            }
            return BaseT::mTree->root().template probeNodeAndCache<NodeT>(xyz, *this);
        } else if ((std::is_same<NodeT, NodeT1>::value)) {
            if (this->isHashed1(xyz)) {
                assert(mNode1);
                return reinterpret_cast<NodeT*>(const_cast<NodeT1*>(mNode1));
            } else if (this->isHashed2(xyz)) {
                assert(mNode2);
                return const_cast<NodeT2*>(mNode2)->template probeNodeAndCache<NodeT>(xyz, *this);
            }
            return BaseT::mTree->root().template probeNodeAndCache<NodeT>(xyz, *this);
        } else if ((std::is_same<NodeT, NodeT2>::value)) {
            if (this->isHashed2(xyz)) {
                assert(mNode2);
                return reinterpret_cast<NodeT*>(const_cast<NodeT2*>(mNode2));
            }
            return BaseT::mTree->root().template probeNodeAndCache<NodeT>(xyz, *this);
        }
        return nullptr;
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    /// @brief @return a pointer to the leaf node that contains
    /// voxel (x, y, z) and if it doesn't exist, return @c nullptr.
    LeafNodeT* probeLeaf(const Coord& xyz) { return this->template probeNode<LeafNodeT>(xyz); }

    /// @brief @return a const pointer to the node of the specified type that contains
    /// voxel (x, y, z) and if it doesn't exist, return @c nullptr.
    template<typename NodeT>
    const NodeT* probeConstNode(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if ((std::is_same<NodeT, NodeT0>::value)) {
            if (this->isHashed0(xyz)) {
                assert(mNode0);
                return reinterpret_cast<const NodeT*>(mNode0);
            } else if (this->isHashed1(xyz)) {
                assert(mNode1);
                return mNode1->template probeConstNodeAndCache<NodeT>(xyz, this->self());
            } else if (this->isHashed2(xyz)) {
                assert(mNode2);
                return mNode2->template probeConstNodeAndCache<NodeT>(xyz, this->self());
            }
            return BaseT::mTree->root().template probeConstNodeAndCache<NodeT>(xyz, this->self());
        } else if ((std::is_same<NodeT, NodeT1>::value)) {
            if (this->isHashed1(xyz)) {
                assert(mNode1);
                return reinterpret_cast<const NodeT*>(mNode1);
            } else if (this->isHashed2(xyz)) {
                assert(mNode2);
                return mNode2->template probeConstNodeAndCache<NodeT>(xyz, this->self());
            }
            return BaseT::mTree->root().template probeConstNodeAndCache<NodeT>(xyz, this->self());
        } else if ((std::is_same<NodeT, NodeT2>::value)) {
            if (this->isHashed2(xyz)) {
                assert(mNode2);
                return reinterpret_cast<const NodeT*>(mNode2);
            }
            return BaseT::mTree->root().template probeConstNodeAndCache<NodeT>(xyz, this->self());
        }
        return nullptr;
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    /// @brief @return a const pointer to the leaf node that contains
    /// voxel (x, y, z) and if it doesn't exist, return @c nullptr.
    const LeafNodeT* probeConstLeaf(const Coord& xyz) const
    {
        return this->template probeConstNode<LeafNodeT>(xyz);
    }
    const LeafNodeT* probeLeaf(const Coord& xyz) const { return this->probeConstLeaf(xyz); }

    /// Remove all the cached nodes and invalidate the corresponding hash-keys.
    void clear() override
    {
        mKey0  = Coord::max();
        mNode0 = nullptr;
        mKey1  = Coord::max();
        mNode1 = nullptr;
        mKey2  = Coord::max();
        mNode2 = nullptr;
    }

private:
    // Allow nodes to insert themselves into the cache.
    template<typename> friend class RootNode;
    template<typename, Index> friend class InternalNode;
    template<typename, Index> friend class LeafNode;
    // Allow trees to deregister themselves.
    template<typename> friend class Tree;

    // This private method is merely for convenience.
    inline ValueAccessor3& self() const { return const_cast<ValueAccessor3&>(*this); }

    /// Private copy method
    inline void copy(const ValueAccessor3& other)
    {
        mKey0  = other.mKey0;
        mNode0 = other.mNode0;
        mKey1  = other.mKey1;
        mNode1 = other.mNode1;
        mKey2  = other.mKey2;
        mNode2 = other.mNode2;
    }

    /// Prevent this accessor from calling Tree::releaseCache() on a tree that
    /// no longer exists.  (Called by mTree when it is destroyed.)
    void release() override
    {
        this->BaseT::release();
        this->clear();
    }
    void getNode(const NodeT0*& node) { node = mNode0; }
    void getNode(const NodeT1*& node) { node = mNode1; }
    void getNode(const NodeT2*& node) { node = mNode2; }
    void getNode(const RootNodeT*& node)
    {
        node = (BaseT::mTree ? &BaseT::mTree->root() : nullptr);
    }
    template<typename OtherNodeType> void getNode(const OtherNodeType*& node) { node = nullptr; }

    void eraseNode(const NodeT0*) { mKey0 = Coord::max(); mNode0 = nullptr; }
    void eraseNode(const NodeT1*) { mKey1 = Coord::max(); mNode1 = nullptr; }
    void eraseNode(const NodeT2*) { mKey2 = Coord::max(); mNode2 = nullptr; }
    template<typename OtherNodeType> void eraseNode(const OtherNodeType*) {}

    /// Cache the given node, which should lie along the path from the root node to
    /// the node containing voxel (x, y, z).
    /// @note This operation is not mutex-protected and is intended to be called
    /// only by nodes and only in the context of a getValue() or setValue() call.
    inline void insert(const Coord& xyz, const NodeT0* node)
    {
        assert(node);
        mKey0  = xyz & ~(NodeT0::DIM-1);
        mNode0 = node;
    }
    inline void insert(const Coord& xyz, const NodeT1* node)
    {
        assert(node);
        mKey1  = xyz & ~(NodeT1::DIM-1);
        mNode1 = node;
    }
    inline void insert(const Coord& xyz, const NodeT2* node)
    {
        assert(node);
        mKey2  = xyz & ~(NodeT2::DIM-1);
        mNode2 = node;
    }
    /// No-op in case a tree traversal attemps to insert a node that
    /// is not cached by the ValueAccessor
    template<typename OtherNodeType>
    inline void insert(const Coord&, const OtherNodeType*)
    {
    }
    inline bool isHashed0(const Coord& xyz) const
    {
        return (xyz[0] & ~Coord::ValueType(NodeT0::DIM-1)) == mKey0[0]
            && (xyz[1] & ~Coord::ValueType(NodeT0::DIM-1)) == mKey0[1]
            && (xyz[2] & ~Coord::ValueType(NodeT0::DIM-1)) == mKey0[2];
    }
    inline bool isHashed1(const Coord& xyz) const
    {
        return (xyz[0] & ~Coord::ValueType(NodeT1::DIM-1)) == mKey1[0]
            && (xyz[1] & ~Coord::ValueType(NodeT1::DIM-1)) == mKey1[1]
            && (xyz[2] & ~Coord::ValueType(NodeT1::DIM-1)) == mKey1[2];
    }
    inline bool isHashed2(const Coord& xyz) const
    {
        return (xyz[0] & ~Coord::ValueType(NodeT2::DIM-1)) == mKey2[0]
            && (xyz[1] & ~Coord::ValueType(NodeT2::DIM-1)) == mKey2[1]
            && (xyz[2] & ~Coord::ValueType(NodeT2::DIM-1)) == mKey2[2];
    }
    mutable Coord mKey0;
    mutable const NodeT0* mNode0;
    mutable Coord mKey1;
    mutable const NodeT1* mNode1;
    mutable Coord mKey2;
    mutable const NodeT2* mNode2;
}; // ValueAccessor3

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_VALUEACCESSOR_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
