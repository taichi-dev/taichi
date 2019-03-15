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

/// @file NodeUnion.h
///
/// @details NodeUnion is a templated helper class that controls access to either
/// the child node pointer or the value for a particular element of a root
/// or internal node. For space efficiency, the child pointer and the value
/// are unioned when possible, since the two are never in use simultaneously.

#ifndef OPENVDB_TREE_NODEUNION_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_NODEUNION_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Types.h>
#include <cstring> // for std::memcpy()
#include <type_traits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

#if OPENVDB_ABI_VERSION_NUMBER >= 4

// Forward declaration of traits class
template<typename T> struct CopyTraits;

// Default implementation that stores the child pointer and the value separately
// (i.e., not in a union)
// This implementation is not used for POD, math::Vec or math::Coord value types.
template<typename ValueT, typename ChildT, typename Enable = void>
class NodeUnion
{
private:
    ChildT* mChild;
    ValueT  mValue;

public:
    NodeUnion(): mChild(nullptr), mValue() {}

    ChildT* getChild() const { return mChild; }
    void setChild(ChildT* child) { mChild = child; }

    const ValueT& getValue() const { return mValue; }
    ValueT& getValue() { return mValue; }
    void setValue(const ValueT& val) { mValue = val; }
};


// Template specialization for values of POD types (int, float, pointer, etc.)
template<typename ValueT, typename ChildT>
class NodeUnion<ValueT, ChildT, typename std::enable_if<std::is_pod<ValueT>::value>::type>
{
private:
    union { ChildT* mChild; ValueT mValue; };

public:
    NodeUnion(): mChild(nullptr) {}

    ChildT* getChild() const { return mChild; }
    void setChild(ChildT* child) { mChild = child; }

    const ValueT& getValue() const { return mValue; }
    ValueT& getValue() { return mValue; }
    void setValue(const ValueT& val) { mValue = val; }
};


// Template specialization for values of types such as math::Vec3f and math::Coord
// for which CopyTraits<T>::IsCopyable is true
template<typename ValueT, typename ChildT>
class NodeUnion<ValueT, ChildT, typename std::enable_if<CopyTraits<ValueT>::IsCopyable>::type>
{
private:
    union { ChildT* mChild; ValueT mValue; };

public:
    NodeUnion(): mChild(nullptr) {}
    NodeUnion(const NodeUnion& other): mChild(nullptr)
        { std::memcpy(this, &other, sizeof(*this)); }
    NodeUnion& operator=(const NodeUnion& rhs)
        { std::memcpy(this, &rhs, sizeof(*this)); return *this; }

    ChildT* getChild() const { return mChild; }
    void setChild(ChildT* child) { mChild = child; }

    const ValueT& getValue() const { return mValue; }
    ValueT& getValue() { return mValue; }
    void setValue(const ValueT& val) { mValue = val; }
};


/// @details A type T is copyable if
/// # T stores member values by value (vs. by pointer or reference)
///   and T's true byte size is given by sizeof(T).
/// # T has a trivial destructor
/// # T has a default constructor
/// # T has an assignment operator
template<typename T> struct CopyTraits { static const bool IsCopyable = false; };
template<typename T> struct CopyTraits<math::Vec2<T>> { static const bool IsCopyable = true; };
template<typename T> struct CopyTraits<math::Vec3<T>> { static const bool IsCopyable = true; };
template<typename T> struct CopyTraits<math::Vec4<T>> { static const bool IsCopyable = true; };
template<> struct CopyTraits<math::Coord> { static const bool IsCopyable = true; };


////////////////////////////////////////


#else // OPENVDB_ABI_VERSION_NUMBER <= 3

// Prior to OpenVDB 4 and the introduction of C++11, values of non-POD types
// were heap-allocated and stored by pointer due to C++98 restrictions on unions.

// Internal implementation of a union of a child node pointer and a value
template<bool ValueIsClass, class ValueT, class ChildT> class NodeUnionImpl;


// Partial specialization for values of non-class types
// (int, float, pointer, etc.) that stores elements by value
template<typename ValueT, typename ChildT>
class NodeUnionImpl</*ValueIsClass=*/false, ValueT, ChildT>
{
private:
    union { ChildT* child; ValueT value; } mUnion;

public:
    NodeUnionImpl() { mUnion.child = nullptr; }

    ChildT* getChild() const { return mUnion.child; }
    void setChild(ChildT* child) { mUnion.child = child; }

    const ValueT& getValue() const { return mUnion.value; }
    ValueT& getValue() { return mUnion.value; }
    void setValue(const ValueT& val) { mUnion.value = val; }
};


// Partial specialization for values of class types (std::string,
// math::Vec, etc.) that stores elements by pointer
template<typename ValueT, typename ChildT>
class NodeUnionImpl</*ValueIsClass=*/true, ValueT, ChildT>
{
private:
    union { ChildT* child; ValueT* value; } mUnion;
    bool mHasChild;

public:
    NodeUnionImpl() : mHasChild(true) { this->setChild(nullptr); }
    NodeUnionImpl(const NodeUnionImpl& other) : mHasChild(true)
    {
        if (other.mHasChild) {
            this->setChild(other.getChild());
        } else {
            this->setValue(other.getValue());
        }
    }
    NodeUnionImpl& operator=(const NodeUnionImpl& other)
    {
        if (other.mHasChild) {
            this->setChild(other.getChild());
        } else {
            this->setValue(other.getValue());
        }
        return *this;
    }
    ~NodeUnionImpl() { this->setChild(nullptr); }

    ChildT* getChild() const { return mHasChild ? mUnion.child : nullptr; }
    void setChild(ChildT* child)
    {
        if (!mHasChild) delete mUnion.value;
        mUnion.child = child;
        mHasChild = true;
    }

    const ValueT& getValue() const { return *mUnion.value; }
    ValueT& getValue() { return *mUnion.value; }
    void setValue(const ValueT& val)
    {
        if (!mHasChild) delete mUnion.value;
        mUnion.value = new ValueT(val);
        mHasChild = false;
    }
};


template<typename ValueT, typename ChildT>
struct NodeUnion: public NodeUnionImpl<std::is_class<ValueT>::value, ValueT, ChildT>
{
    NodeUnion() {}
};

#endif

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_NODEUNION_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
