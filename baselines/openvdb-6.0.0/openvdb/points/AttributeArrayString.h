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

/// @file points/AttributeArrayString.h
///
/// @author Dan Bailey
///
/// @brief  Attribute array storage for string data using Descriptor Metadata.

#ifndef OPENVDB_POINTS_ATTRIBUTE_ARRAY_STRING_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_ATTRIBUTE_ARRAY_STRING_HAS_BEEN_INCLUDED

#include "AttributeArray.h"
#include <memory>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


////////////////////////////////////////


using StringIndexType = uint32_t;


namespace attribute_traits
{
    template <bool Truncate> struct StringTypeTrait { using Type = StringIndexType; };
    template<> struct StringTypeTrait</*Truncate=*/true> { using Type = uint16_t; };
}


template <bool Truncate>
struct StringCodec
{
    using ValueType = StringIndexType;

    template <typename T>
    struct Storage { using Type = typename attribute_traits::StringTypeTrait<Truncate>::Type; };

    template<typename StorageType> static void decode(const StorageType&, ValueType&);
    template<typename StorageType> static void encode(const ValueType&, StorageType&);
    static const char* name() { return Truncate ? "str_trnc" : "str"; }
};


using StringAttributeArray = TypedAttributeArray<StringIndexType, StringCodec<false>>;


////////////////////////////////////////


class OPENVDB_API StringMetaInserter
{
public:
    StringMetaInserter(MetaMap& metadata);

    /// Insert the string into the metadata
    void insert(const Name& name);

    /// Reset the cache from the metadata
    void resetCache();

private:
    MetaMap& mMetadata;
    std::vector<Index> mIndices;
    std::vector<Name> mValues;
}; // StringMetaInserter


////////////////////////////////////////


template <bool Truncate>
template<typename StorageType>
inline void
StringCodec<Truncate>::decode(const StorageType& data, ValueType& val)
{
    val = static_cast<ValueType>(data);
}


template <bool Truncate>
template<typename StorageType>
inline void
StringCodec<Truncate>::encode(const ValueType& val, StorageType& data)
{
    data = static_cast<ValueType>(val);
}


////////////////////////////////////////


inline bool isString(const AttributeArray& array)
{
    return array.isType<StringAttributeArray>();
}


////////////////////////////////////////


class OPENVDB_API StringAttributeHandle
{
public:
    using Ptr = std::shared_ptr<StringAttributeHandle>;//SharedPtr<StringAttributeHandle>;

    static Ptr create(const AttributeArray& array, const MetaMap& metadata, const bool preserveCompression = true);

    StringAttributeHandle(  const AttributeArray& array,
                            const MetaMap& metadata,
                            const bool preserveCompression = true);

    Index size() const { return mHandle.size(); }
    bool isUniform() const { return mHandle.isUniform(); }

    Name get(Index n, Index m = 0) const;
    void get(Name& name, Index n, Index m = 0) const;

protected:
    AttributeHandle<StringIndexType, StringCodec<false>>    mHandle;
    const MetaMap&                                          mMetadata;
}; // class StringAttributeHandle


////////////////////////////////////////


class OPENVDB_API StringAttributeWriteHandle : public StringAttributeHandle
{
public:
    using Ptr = std::shared_ptr<StringAttributeWriteHandle>;//SharedPtr<StringAttributeWriteHandle>;

    static Ptr create(AttributeArray& array, const MetaMap& metadata, const bool expand = true);

    StringAttributeWriteHandle( AttributeArray& array,
                                const MetaMap& metadata,
                                const bool expand = true);

    /// @brief  If this array is uniform, replace it with an array of length size().
    /// @param  fill if true, assign the uniform value to each element of the array.
    void expand(bool fill = true);

    /// @brief Set membership for the whole array and attempt to collapse
    void collapse();
    /// @brief Set membership for the whole array and attempt to collapse
    /// @param name Name of the String
    void collapse(const Name& name);

    /// Compact the existing array to become uniform if all values are identical
    bool compact();

    /// @brief Fill the existing array with the given value.
    /// @note Identical to collapse() except a non-uniform array will not become uniform.
    void fill(const Name& name);

    /// Set the value of the index to @a name
    void set(Index n, const Name& name);
    void set(Index n, Index m, const Name& name);

    /// Reset the value cache from the metadata
    void resetCache();

private:
    /// Retrieve the index of this string value from the cache
    /// @note throws if name does not exist in cache
    Index getIndex(const Name& name);

    using ValueMap = std::map<std::string, Index>;

    ValueMap                                                    mCache;
    AttributeWriteHandle<StringIndexType, StringCodec<false>>   mWriteHandle;
}; // class StringAttributeWriteHandle


////////////////////////////////////////


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_ATTRIBUTE_ARRAY_STRING_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )

