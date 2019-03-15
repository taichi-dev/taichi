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
/// @file tree/Iterator.h
///
/// @author Peter Cucka and Ken Museth

#ifndef OPENVDB_TREE_ITERATOR_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_ITERATOR_HAS_BEEN_INCLUDED

#include <sstream>
#include <type_traits>
#include <openvdb/util/NodeMasks.h>
#include <openvdb/Exceptions.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

/// @brief Base class for iterators over internal and leaf nodes
///
/// This class is typically not instantiated directly, since it doesn't provide methods
/// to dereference the iterator.  Those methods (@vdblink::tree::SparseIteratorBase::operator*()
/// operator*()@endlink, @vdblink::tree::SparseIteratorBase::setValue() setValue()@endlink, etc.)
/// are implemented in the @vdblink::tree::SparseIteratorBase sparse@endlink and
/// @vdblink::tree::DenseIteratorBase dense@endlink iterator subclasses.
template<typename MaskIterT, typename NodeT>
class IteratorBase
{
public:
    IteratorBase(): mParentNode(nullptr) {}
    IteratorBase(const MaskIterT& iter, NodeT* parent): mParentNode(parent), mMaskIter(iter) {}
    IteratorBase(const IteratorBase&) = default;
    IteratorBase& operator=(const IteratorBase&) = default;

    bool operator==(const IteratorBase& other) const
    {
        return (mParentNode == other.mParentNode) && (mMaskIter == other.mMaskIter);
    }
    bool operator!=(const IteratorBase& other) const
    {
        return !(*this == other);
    }

    /// Return a pointer to the node (if any) over which this iterator is iterating.
    NodeT* getParentNode() const { return mParentNode; }
    /// @brief Return a reference to the node over which this iterator is iterating.
    /// @throw ValueError if there is no parent node.
    NodeT& parent() const
    {
        if (!mParentNode) OPENVDB_THROW(ValueError, "iterator references a null node");
        return *mParentNode;
    }

    /// Return this iterator's position as an index into the parent node's table.
    Index offset() const { return mMaskIter.offset(); }

    /// Identical to offset
    Index pos() const { return mMaskIter.offset(); }

    /// Return @c true if this iterator is not yet exhausted.
    bool test() const { return mMaskIter.test(); }
    /// Return @c true if this iterator is not yet exhausted.
    operator bool() const { return this->test(); }

    /// Advance to the next item in the parent node's table.
    bool next() { return mMaskIter.next(); }
    /// Advance to the next item in the parent node's table.
    void increment() { mMaskIter.increment(); }
    /// Advance to the next item in the parent node's table.
    IteratorBase& operator++() { this->increment(); return *this; }
    /// Advance @a n items in the parent node's table.
    void increment(Index n) { mMaskIter.increment(n); }

    /// @brief Return @c true if this iterator is pointing to an active value.
    /// Return @c false if it is pointing to either an inactive value or a child node.
    bool isValueOn() const { return parent().isValueMaskOn(this->pos()); }
    /// @brief If this iterator is pointing to a value, set the value's active state.
    /// Otherwise, do nothing.
    void setValueOn(bool on = true) const { parent().setValueMask(this->pos(), on); }
    /// @brief If this iterator is pointing to a value, mark the value as inactive.
    /// @details If this iterator is pointing to a child node, then the current item
    /// in the parent node's table is required to be inactive.  In that case,
    /// this method has no effect.
    void setValueOff() const { parent().mValueMask.setOff(this->pos()); }

    /// Return the coordinates of the item to which this iterator is pointing.
    Coord getCoord() const { return parent().offsetToGlobalCoord(this->pos()); }
    /// Return in @a xyz the coordinates of the item to which this iterator is pointing.
    void getCoord(Coord& xyz) const { xyz = this->getCoord(); }

private:
    /// @note This parent node pointer is mutable, because setValueOn() and
    /// setValueOff(), though const, need to call non-const methods on the parent.
    /// There is a distinction between a const iterator (e.g., const ValueOnIter),
    /// which is an iterator that can't be incremented, and an iterator over
    /// a const node (e.g., ValueOnCIter), which might be const or non-const itself
    /// but can't call non-const methods like setValue() on the node.
    mutable NodeT* mParentNode;
    MaskIterT mMaskIter;
}; // class IteratorBase


////////////////////////////////////////


/// @brief Base class for sparse iterators over internal and leaf nodes
template<
    typename MaskIterT, // mask iterator type (OnIterator, OffIterator, etc.)
    typename IterT,     // SparseIteratorBase subclass (the "Curiously Recurring Template Pattern")
    typename NodeT,     // type of node over which to iterate
    typename ItemT>     // type of value to which this iterator points
struct SparseIteratorBase: public IteratorBase<MaskIterT, NodeT>
{
    using NodeType = NodeT;
    using ValueType = ItemT;
    using NonConstNodeType = typename std::remove_const<NodeT>::type;
    using NonConstValueType = typename std::remove_const<ItemT>::type;
    static const bool IsSparseIterator = true, IsDenseIterator = false;

    SparseIteratorBase() {}
    SparseIteratorBase(const MaskIterT& iter, NodeT* parent):
        IteratorBase<MaskIterT, NodeT>(iter, parent) {}

    /// @brief Return the item at the given index in the parent node's table.
    /// @note All subclasses must implement this accessor.
    ItemT& getItem(Index) const;
    /// @brief Set the value of the item at the given index in the parent node's table.
    /// @note All non-const iterator subclasses must implement this accessor.
    void setItem(Index, const ItemT&) const;

    /// Return a reference to the item to which this iterator is pointing.
    ItemT& operator*() const { return this->getValue(); }
    /// Return a pointer to the item to which this iterator is pointing.
    ItemT* operator->() const { return &(this->operator*()); }

    /// Return the item to which this iterator is pointing.
    ItemT& getValue() const
    {
        return static_cast<const IterT*>(this)->getItem(this->pos()); // static polymorphism
    }
    /// @brief Set the value of the item to which this iterator is pointing.
    /// (Not valid for const iterators.)
    void setValue(const ItemT& value) const
    {
        static_assert(!std::is_const<NodeT>::value, "setValue() not allowed for const iterators");
        static_cast<const IterT*>(this)->setItem(this->pos(), value); // static polymorphism
    }
    /// @brief Apply a functor to the item to which this iterator is pointing.
    /// (Not valid for const iterators.)
    /// @param op  a functor of the form <tt>void op(ValueType&) const</tt> that modifies
    ///            its argument in place
    /// @see Tree::modifyValue()
    template<typename ModifyOp>
    void modifyValue(const ModifyOp& op) const
    {
        static_assert(!std::is_const<NodeT>::value,
            "modifyValue() not allowed for const iterators");
        static_cast<const IterT*>(this)->modifyItem(this->pos(), op); // static polymorphism
    }
}; // class SparseIteratorBase


////////////////////////////////////////


/// @brief Base class for dense iterators over internal and leaf nodes
/// @note Dense iterators have no @c %operator*() or @c %operator->(),
/// because their return type would have to vary depending on whether
/// the iterator is pointing to a value or a child node.
template<
    typename MaskIterT,  // mask iterator type (typically a DenseIterator)
    typename IterT,      // DenseIteratorBase subclass (the "Curiously Recurring Template Pattern")
    typename NodeT,      // type of node over which to iterate
    typename SetItemT,   // type of set value (ChildNodeType, for non-leaf nodes)
    typename UnsetItemT> // type of unset value (ValueType, usually)
struct DenseIteratorBase: public IteratorBase<MaskIterT, NodeT>
{
    using NodeType = NodeT;
    using ValueType = UnsetItemT;
    using ChildNodeType = SetItemT;
    using NonConstNodeType = typename std::remove_const<NodeT>::type;
    using NonConstValueType = typename std::remove_const<UnsetItemT>::type;
    using NonConstChildNodeType = typename std::remove_const<SetItemT>::type;
    static const bool IsSparseIterator = false, IsDenseIterator = true;

    DenseIteratorBase() {}
    DenseIteratorBase(const MaskIterT& iter, NodeT* parent):
        IteratorBase<MaskIterT, NodeT>(iter, parent) {}

    /// @brief Return @c true if the item at the given index in the parent node's table
    /// is a set value and return either the set value in @a child or the unset value
    /// in @a value.
    /// @note All subclasses must implement this accessor.
    bool getItem(Index, SetItemT*& child, NonConstValueType& value) const;
    /// @brief Set the value of the item at the given index in the parent node's table.
    /// @note All non-const iterator subclasses must implement this accessor.
    void setItem(Index, SetItemT*) const;
    /// @brief "Unset" the value of the item at the given index in the parent node's table.
    /// @note All non-const iterator subclasses must implement this accessor.
    void unsetItem(Index, const UnsetItemT&) const;

    /// Return @c true if this iterator is pointing to a child node.
    bool isChildNode() const { return this->parent().isChildMaskOn(this->pos()); }

    /// @brief If this iterator is pointing to a child node, return a pointer to the node.
    /// Otherwise, return nullptr and, in @a value, the value to which this iterator is pointing.
    SetItemT* probeChild(NonConstValueType& value) const
    {
        SetItemT* child = nullptr;
        static_cast<const IterT*>(this)->getItem(this->pos(), child, value); // static polymorphism
        return child;
    }
    /// @brief If this iterator is pointing to a child node, return @c true and return
    /// a pointer to the child node in @a child.  Otherwise, return @c false and return
    /// the value to which this iterator is pointing in @a value.
    bool probeChild(SetItemT*& child, NonConstValueType& value) const
    {
        child = probeChild(value);
        return (child != nullptr);
    }

    /// @brief Return @c true if this iterator is pointing to a value and return
    /// the value in @a value.  Otherwise, return @c false.
    bool probeValue(NonConstValueType& value) const
    {
        SetItemT* child = nullptr;
        const bool isChild = static_cast<const IterT*>(this)-> // static polymorphism
            getItem(this->pos(), child, value);
        return !isChild;
    }

    /// @brief Replace with the given child node the item in the parent node's table
    /// to which this iterator is pointing.
    void setChild(SetItemT* child) const
    {
        static_cast<const IterT*>(this)->setItem(this->pos(), child); // static polymorphism
    }

    /// @brief Replace with the given value the item in the parent node's table
    /// to which this iterator is pointing.
    void setValue(const UnsetItemT& value) const
    {
        static_cast<const IterT*>(this)->unsetItem(this->pos(), value); // static polymorphism
    }
}; // struct DenseIteratorBase

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_ITERATOR_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
