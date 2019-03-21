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

#ifndef OPENVDB_METADATA_HAS_BEEN_INCLUDED
#define OPENVDB_METADATA_HAS_BEEN_INCLUDED

#include "version.h"
#include "Exceptions.h"
#include "Types.h"
#include "math/Math.h" // for math::isZero()
#include "util/Name.h"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

/// @brief Base class for storing metadata information in a grid.
class OPENVDB_API Metadata
{
public:
    using Ptr = SharedPtr<Metadata>;
    using ConstPtr = SharedPtr<const Metadata>;

    Metadata() {}
    virtual ~Metadata() {}

    // Disallow copying of instances of this class.
    Metadata(const Metadata&) = delete;
    Metadata& operator=(const Metadata&) = delete;

    /// Return the type name of the metadata.
    virtual Name typeName() const = 0;

    /// Return a copy of the metadata.
    virtual Metadata::Ptr copy() const = 0;

    /// Copy the given metadata into this metadata.
    virtual void copy(const Metadata& other) = 0;

    /// Return a textual representation of this metadata.
    virtual std::string str() const = 0;

    /// Return the boolean representation of this metadata (empty strings
    /// and zeroVals evaluate to false; most other values evaluate to true).
    virtual bool asBool() const = 0;

    /// Return @c true if the given metadata is equivalent to this metadata.
    bool operator==(const Metadata& other) const;
    /// Return @c true if the given metadata is different from this metadata.
    bool operator!=(const Metadata& other) const { return !(*this == other); }

    /// Return the size of this metadata in bytes.
    virtual Index32 size() const = 0;

    /// Unserialize this metadata from a stream.
    void read(std::istream&);
    /// Serialize this metadata to a stream.
    void write(std::ostream&) const;

    /// Create new metadata of the given type.
    static Metadata::Ptr createMetadata(const Name& typeName);

    /// Return @c true if the given type is known by the metadata type registry.
    static bool isRegisteredType(const Name& typeName);

    /// Clear out the metadata registry.
    static void clearRegistry();

    /// Register the given metadata type along with a factory function.
    static void registerType(const Name& typeName, Metadata::Ptr (*createMetadata)());
    static void unregisterType(const Name& typeName);

protected:
    /// Read the size of the metadata from a stream.
    static Index32 readSize(std::istream&);
    /// Write the size of the metadata to a stream.
    void writeSize(std::ostream&) const;

    /// Read the metadata from a stream.
    virtual void readValue(std::istream&, Index32 numBytes) = 0;
    /// Write the metadata to a stream.
    virtual void writeValue(std::ostream&) const = 0;
};


#if OPENVDB_ABI_VERSION_NUMBER >= 5

/// @brief Subclass to hold raw data of an unregistered type
class OPENVDB_API UnknownMetadata: public Metadata
{
public:
    using ByteVec = std::vector<uint8_t>;

    explicit UnknownMetadata(const Name& typ = "<unknown>"): mTypeName(typ) {}

    Name typeName() const override { return mTypeName; }
    Metadata::Ptr copy() const override;
    void copy(const Metadata&) override;
    std::string str() const override { return (mBytes.empty() ? "" : "<binary data>"); }
    bool asBool() const override { return !mBytes.empty(); }
    Index32 size() const override { return static_cast<Index32>(mBytes.size()); }

    void setValue(const ByteVec& bytes) { mBytes = bytes; }
    const ByteVec& value() const { return mBytes; }

protected:
    void readValue(std::istream&, Index32 numBytes) override;
    void writeValue(std::ostream&) const override;

private:
    Name mTypeName;
    ByteVec mBytes;
};

#else // if OPENVDB_ABI_VERSION_NUMBER < 5

/// @brief Subclass to read (and ignore) data of an unregistered type
class OPENVDB_API UnknownMetadata: public Metadata
{
public:
    UnknownMetadata() {}
    Name typeName() const override { return "<unknown>"; }
    Metadata::Ptr copy() const override { OPENVDB_THROW(TypeError, "Metadata has unknown type"); }
    void copy(const Metadata&) override {OPENVDB_THROW(TypeError, "Destination has unknown type");}
    std::string str() const override { return "<unknown>"; }
    bool asBool() const override { return false; }
    Index32 size() const override { return 0; }

protected:
    void readValue(std::istream&, Index32 numBytes) override;
    void writeValue(std::ostream&) const override;
};

#endif


/// @brief Templated metadata class to hold specific types.
template<typename T>
class TypedMetadata: public Metadata
{
public:
    using Ptr = SharedPtr<TypedMetadata<T>>;
    using ConstPtr = SharedPtr<const TypedMetadata<T>>;

    TypedMetadata();
    TypedMetadata(const T& value);
    TypedMetadata(const TypedMetadata<T>& other);
    ~TypedMetadata() override;

    Name typeName() const override;
    Metadata::Ptr copy() const override;
    void copy(const Metadata& other) override;
    std::string str() const override;
    bool asBool() const override;
    Index32 size() const override { return static_cast<Index32>(sizeof(T)); }

    /// Set this metadata's value.
    void setValue(const T&);
    /// Return this metadata's value.
    T& value();
    const T& value() const;

    // Static specialized function for the type name. This function must be
    // template specialized for each type T.
    static Name staticTypeName() { return typeNameAsString<T>(); }

    /// Create new metadata of this type.
    static Metadata::Ptr createMetadata();

    static void registerType();
    static void unregisterType();
    static bool isRegisteredType();

protected:
    void readValue(std::istream&, Index32 numBytes) override;
    void writeValue(std::ostream&) const override;

private:
    T mValue;
};

/// Write a Metadata to an output stream
std::ostream& operator<<(std::ostream& ostr, const Metadata& metadata);


////////////////////////////////////////


inline void
Metadata::writeSize(std::ostream& os) const
{
    const Index32 n = this->size();
    os.write(reinterpret_cast<const char*>(&n), sizeof(Index32));
}


inline Index32
Metadata::readSize(std::istream& is)
{
    Index32 n = 0;
    is.read(reinterpret_cast<char*>(&n), sizeof(Index32));
    return n;
}


inline void
Metadata::read(std::istream& is)
{
    const Index32 numBytes = this->readSize(is);
    this->readValue(is, numBytes);
}


inline void
Metadata::write(std::ostream& os) const
{
    this->writeSize(os);
    this->writeValue(os);
}


////////////////////////////////////////


template <typename T>
inline
TypedMetadata<T>::TypedMetadata() : mValue(T())
{
}

template <typename T>
inline
TypedMetadata<T>::TypedMetadata(const T &value) : mValue(value)
{
}

template <typename T>
inline
TypedMetadata<T>::TypedMetadata(const TypedMetadata<T> &other) :
    Metadata(),
    mValue(other.mValue)
{
}

template <typename T>
inline
TypedMetadata<T>::~TypedMetadata()
{
}

template <typename T>
inline Name
TypedMetadata<T>::typeName() const
{
    return TypedMetadata<T>::staticTypeName();
}

template <typename T>
inline void
TypedMetadata<T>::setValue(const T& val)
{
    mValue = val;
}

template <typename T>
inline T&
TypedMetadata<T>::value()
{
    return mValue;
}

template <typename T>
inline const T&
TypedMetadata<T>::value() const
{
    return mValue;
}

template <typename T>
inline Metadata::Ptr
TypedMetadata<T>::copy() const
{
    Metadata::Ptr metadata(new TypedMetadata<T>());
    metadata->copy(*this);
    return metadata;
}

template <typename T>
inline void
TypedMetadata<T>::copy(const Metadata &other)
{
    const TypedMetadata<T>* t = dynamic_cast<const TypedMetadata<T>*>(&other);
    if (t == nullptr) OPENVDB_THROW(TypeError, "Incompatible type during copy");
    mValue = t->mValue;
}


template<typename T>
inline void
TypedMetadata<T>::readValue(std::istream& is, Index32 /*numBytes*/)
{
    //assert(this->size() == numBytes);
    is.read(reinterpret_cast<char*>(&mValue), this->size());
}

template<typename T>
inline void
TypedMetadata<T>::writeValue(std::ostream& os) const
{
    os.write(reinterpret_cast<const char*>(&mValue), this->size());
}

template <typename T>
inline std::string
TypedMetadata<T>::str() const
{
    std::ostringstream ostr;
    ostr << mValue;
    return ostr.str();
}

template<typename T>
inline bool
TypedMetadata<T>::asBool() const
{
    return !math::isZero(mValue);
}

template <typename T>
inline Metadata::Ptr
TypedMetadata<T>::createMetadata()
{
    Metadata::Ptr ret(new TypedMetadata<T>());
    return ret;
}

template <typename T>
inline void
TypedMetadata<T>::registerType()
{
    Metadata::registerType(TypedMetadata<T>::staticTypeName(),
                           TypedMetadata<T>::createMetadata);
}

template <typename T>
inline void
TypedMetadata<T>::unregisterType()
{
    Metadata::unregisterType(TypedMetadata<T>::staticTypeName());
}

template <typename T>
inline bool
TypedMetadata<T>::isRegisteredType()
{
    return Metadata::isRegisteredType(TypedMetadata<T>::staticTypeName());
}


template<>
inline std::string
TypedMetadata<bool>::str() const
{
    return (mValue ? "true" : "false");
}


inline std::ostream&
operator<<(std::ostream& ostr, const Metadata& metadata)
{
    ostr << metadata.str();
    return ostr;
}


using BoolMetadata   = TypedMetadata<bool>;
using DoubleMetadata = TypedMetadata<double>;
using FloatMetadata  = TypedMetadata<float>;
using Int32Metadata  = TypedMetadata<int32_t>;
using Int64Metadata  = TypedMetadata<int64_t>;
using StringMetadata = TypedMetadata<std::string>;
using Vec2DMetadata  = TypedMetadata<Vec2d>;
using Vec2IMetadata  = TypedMetadata<Vec2i>;
using Vec2SMetadata  = TypedMetadata<Vec2s>;
using Vec3DMetadata  = TypedMetadata<Vec3d>;
using Vec3IMetadata  = TypedMetadata<Vec3i>;
using Vec3SMetadata  = TypedMetadata<Vec3s>;
using Mat4SMetadata  = TypedMetadata<Mat4s>;
using Mat4DMetadata  = TypedMetadata<Mat4d>;


////////////////////////////////////////


template<>
inline Index32
StringMetadata::size() const
{
    return static_cast<Index32>(mValue.size());
}


template<>
inline std::string
StringMetadata::str() const
{
    return mValue;
}


template<>
inline void
StringMetadata::readValue(std::istream& is, Index32 size)
{
    mValue.resize(size, '\0');
    is.read(&mValue[0], size);
}

template<>
inline void
StringMetadata::writeValue(std::ostream& os) const
{
    os.write(reinterpret_cast<const char*>(&mValue[0]), this->size());
}

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_METADATA_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
