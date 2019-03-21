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

/// @file points/AttributeGroup.h
///
/// @author Dan Bailey
///
/// @brief  Attribute Group access and filtering for iteration.

#ifndef OPENVDB_POINTS_ATTRIBUTE_GROUP_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_ATTRIBUTE_GROUP_HAS_BEEN_INCLUDED

#include "AttributeArray.h"
#include "AttributeSet.h"
#include <memory>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


using GroupType = uint8_t;


////////////////////////////////////////


struct GroupCodec
{
    using StorageType   = GroupType;
    using ValueType     = GroupType;

    template <typename T>
    struct Storage { using Type = StorageType; };

    static void decode(const StorageType&, ValueType&);
    static void encode(const ValueType&, StorageType&);
    static const char* name() { return "grp"; }
};


using GroupAttributeArray = TypedAttributeArray<GroupType, GroupCodec>;


////////////////////////////////////////


inline void
GroupCodec::decode(const StorageType& data, ValueType& val)
{
    val = data;
}


inline void
GroupCodec::encode(const ValueType& val, StorageType& data)
{
    data = val;
}


////////////////////////////////////////


inline bool isGroup(const AttributeArray& array)
{
    return array.isType<GroupAttributeArray>();
}


////////////////////////////////////////


class OPENVDB_API GroupHandle
{
public:
    using Ptr = std::shared_ptr<GroupHandle>;

    // Dummy class that distinguishes an offset from a bitmask on construction
    struct BitMask { };

    using GroupIndex = std::pair<Index, uint8_t>;

    GroupHandle(const GroupAttributeArray& array, const GroupType& offset);
    GroupHandle(const GroupAttributeArray& array, const GroupType& bitMask, BitMask);

    Index size() const { return mArray.size(); }
    bool isUniform() const { return mArray.isUniform(); }

    bool get(Index n) const;
    bool getUnsafe(Index n) const;

protected:
    const GroupAttributeArray& mArray;
    const GroupType mBitMask;
}; // class GroupHandle


////////////////////////////////////////


class OPENVDB_API GroupWriteHandle : public GroupHandle
{
public:
    using Ptr = std::shared_ptr<GroupWriteHandle>;

    GroupWriteHandle(GroupAttributeArray& array, const GroupType& offset);

    void set(Index n, bool on);

    /// @brief Set membership for the whole array and attempt to collapse
    ///
    /// @param on True or false for inclusion in group
    ///
    /// @note This method guarantees that all attributes will have group membership
    /// changed according to the input bool, however compaction will not be performed
    /// if other groups that share the same underlying array are non-uniform.
    /// The return value indicates if the group array ends up being uniform.
    bool collapse(bool on);

    /// Compact the existing array to become uniform if all values are identical
    bool compact();

}; // class GroupWriteHandle


////////////////////////////////////////


/// Index filtering on group membership
class GroupFilter
{
public:
    GroupFilter(const Name& name, const AttributeSet& attributeSet)
        : mIndex(attributeSet.groupIndex(name)) { }

    explicit GroupFilter(const AttributeSet::Descriptor::GroupIndex& index)
        : mIndex(index) { }

    inline bool initialized() const { return bool(mHandle); }

    static index::State state() { return index::PARTIAL; }
    template <typename LeafT>
    static index::State state(const LeafT&) { return index::PARTIAL; }

    template <typename LeafT>
    void reset(const LeafT& leaf) {
        mHandle.reset(new GroupHandle(leaf.groupHandle(mIndex)));
    }

    template <typename IterT>
    bool valid(const IterT& iter) const {
        assert(mHandle);
        return mHandle->getUnsafe(*iter);
    }

private:
    const AttributeSet::Descriptor::GroupIndex mIndex;
    GroupHandle::Ptr mHandle;
}; // class GroupFilter


////////////////////////////////////////


} // namespace points

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_POINTS_ATTRIBUTE_GROUP_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
