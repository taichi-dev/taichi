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

#ifndef OPENVDB_METADATA_METAMAP_HAS_BEEN_INCLUDED
#define OPENVDB_METADATA_METAMAP_HAS_BEEN_INCLUDED

#include "Metadata.h"
#include "Types.h"
#include "Exceptions.h"
#include <iosfwd>
#include <map>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

/// Container that maps names (strings) to values of arbitrary types
class OPENVDB_API MetaMap
{
public:
    using Ptr = SharedPtr<MetaMap>;
    using ConstPtr = SharedPtr<const MetaMap>;

    using MetadataMap = std::map<Name, Metadata::Ptr>;
    using MetaIterator = MetadataMap::iterator;
    using ConstMetaIterator = MetadataMap::const_iterator;
        ///< @todo this should really iterate over a map of Metadata::ConstPtrs

    MetaMap() {}
    MetaMap(const MetaMap& other);
    virtual ~MetaMap() {}

    /// Return a copy of this map whose fields are shared with this map.
    MetaMap::Ptr copyMeta() const;
    /// Return a deep copy of this map that shares no data with this map.
    MetaMap::Ptr deepCopyMeta() const;

    /// Assign a deep copy of another map to this map.
    MetaMap& operator=(const MetaMap&);

    /// Unserialize metadata from the given stream.
    void readMeta(std::istream&);
    /// Serialize metadata to the given stream.
    void writeMeta(std::ostream&) const;

    /// @brief Insert a new metadata field or overwrite the value of an existing field.
    /// @details If a field with the given name doesn't already exist, add a new field.
    /// Otherwise, if the new value's type is the same as the existing field's value type,
    /// overwrite the existing value with new value.
    /// @throw TypeError if a field with the given name already exists, but its value type
    /// is not the same as the new value's
    /// @throw ValueError if the given field name is empty.
    void insertMeta(const Name&, const Metadata& value);
    /// @brief Deep copy all of the metadata fields from the given map into this map.
    /// @throw TypeError if any field in the given map has the same name as
    /// but a different value type than one of this map's fields.
    void insertMeta(const MetaMap&);

    /// Remove the given metadata field if it exists.
    void removeMeta(const Name&);

    //@{
    /// @brief Return a pointer to the metadata with the given name.
    /// If no such field exists, return a null pointer.
    Metadata::Ptr operator[](const Name&);
    Metadata::ConstPtr operator[](const Name&) const;
    //@}

    //@{
    /// @brief Return a pointer to a TypedMetadata object of type @c T and with the given name.
    /// If no such field exists or if there is a type mismatch, return a null pointer.
    template<typename T> typename T::Ptr getMetadata(const Name&);
    template<typename T> typename T::ConstPtr getMetadata(const Name&) const;
    //@}

    /// @brief Return a reference to the value of type @c T stored in the given metadata field.
    /// @throw LookupError if no field with the given name exists.
    /// @throw TypeError if the given field is not of type @c T.
    template<typename T> T& metaValue(const Name&);
    template<typename T> const T& metaValue(const Name&) const;

    // Functions for iterating over the metadata
    MetaIterator beginMeta() { return mMeta.begin(); }
    MetaIterator endMeta() { return mMeta.end(); }
    ConstMetaIterator beginMeta() const { return mMeta.begin(); }
    ConstMetaIterator endMeta() const { return mMeta.end(); }

    void clearMetadata() { mMeta.clear(); }

    size_t metaCount() const { return mMeta.size(); }

    /// Return a string describing this metadata map.  Prefix each line with @a indent.
    std::string str(const std::string& indent = "") const;

    /// Return @c true if the given map is equivalent to this map.
    bool operator==(const MetaMap& other) const;
    /// Return @c true if the given map is different from this map.
    bool operator!=(const MetaMap& other) const { return !(*this == other); }

private:
    /// @brief Return a pointer to TypedMetadata with the given template parameter.
    /// @throw LookupError if no field with the given name is found.
    /// @throw TypeError if the given field is not of type T.
    template<typename T>
    typename TypedMetadata<T>::Ptr getValidTypedMetadata(const Name&) const;

    MetadataMap mMeta;
};

/// Write a MetaMap to an output stream
std::ostream& operator<<(std::ostream&, const MetaMap&);


////////////////////////////////////////


inline Metadata::Ptr
MetaMap::operator[](const Name& name)
{
    MetaIterator iter = mMeta.find(name);
    return (iter == mMeta.end() ? Metadata::Ptr() : iter->second);
}

inline Metadata::ConstPtr
MetaMap::operator[](const Name &name) const
{
    ConstMetaIterator iter = mMeta.find(name);
    return (iter == mMeta.end() ? Metadata::Ptr() : iter->second);
}


////////////////////////////////////////


template<typename T>
inline typename T::Ptr
MetaMap::getMetadata(const Name &name)
{
    ConstMetaIterator iter = mMeta.find(name);
    if (iter == mMeta.end()) return typename T::Ptr{};

    // To ensure that we get valid conversion if the metadata pointers cross dso
    // boundaries, we have to check the qualified typename and then do a static
    // cast. This is slower than doing a dynamic_pointer_cast, but is safer when
    // pointers cross dso boundaries.
    if (iter->second->typeName() == T::staticTypeName()) {
        return StaticPtrCast<T, Metadata>(iter->second);
    } // else
    return typename T::Ptr{};
}

template<typename T>
inline typename T::ConstPtr
MetaMap::getMetadata(const Name &name) const
{
    ConstMetaIterator iter = mMeta.find(name);
    if (iter == mMeta.end()) return typename T::ConstPtr{};

    // To ensure that we get valid conversion if the metadata pointers cross dso
    // boundaries, we have to check the qualified typename and then do a static
    // cast. This is slower than doing a dynamic_pointer_cast, but is safer when
    // pointers cross dso boundaries.
    if (iter->second->typeName() == T::staticTypeName()) {
        return StaticPtrCast<const T, const Metadata>(iter->second);
    } // else
    return typename T::ConstPtr{};
}


////////////////////////////////////////


template<typename T>
inline typename TypedMetadata<T>::Ptr
MetaMap::getValidTypedMetadata(const Name &name) const
{
    ConstMetaIterator iter = mMeta.find(name);
    if (iter == mMeta.end()) OPENVDB_THROW(LookupError, "Cannot find metadata " << name);

    // To ensure that we get valid conversion if the metadata pointers cross dso
    // boundaries, we have to check the qualified typename and then do a static
    // cast. This is slower than doing a dynamic_pointer_cast, but is safer when
    // pointers cross dso boundaries.
    typename TypedMetadata<T>::Ptr m;
    if (iter->second->typeName() == TypedMetadata<T>::staticTypeName()) {
        m = StaticPtrCast<TypedMetadata<T>, Metadata>(iter->second);
    }
    if (!m) OPENVDB_THROW(TypeError, "Invalid type for metadata " << name);
    return m;
}


////////////////////////////////////////


template<typename T>
inline T&
MetaMap::metaValue(const Name &name)
{
    typename TypedMetadata<T>::Ptr m = getValidTypedMetadata<T>(name);
    return m->value();
}


template<typename T>
inline const T&
MetaMap::metaValue(const Name &name) const
{
    typename TypedMetadata<T>::Ptr m = getValidTypedMetadata<T>(name);
    return m->value();
}

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_METADATA_METAMAP_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
