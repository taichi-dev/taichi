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

#ifndef OPENVDB_TREE_LEAFBUFFER_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_LEAFBUFFER_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/io/Compression.h> // for io::readCompressedValues(), etc
#include <openvdb/util/NodeMasks.h>
#include <tbb/atomic.h>
#include <tbb/spin_mutex.h>
#include <algorithm> // for std::swap
#include <cstddef> // for offsetof()
#include <iostream>
#include <memory>
#include <type_traits>


class TestLeaf;

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

namespace internal {

/// @internal For delayed loading to be threadsafe, LeafBuffer::mOutOfCore must be
/// memory-fenced when it is set in LeafBuffer::doLoad(), otherwise that operation
/// could be reordered ahead of others in doLoad(), with the possible result that
/// other threads could see the buffer as in-core before it has been fully loaded.
/// Making mOutOfCore a TBB atomic solves the problem, since TBB atomics are release-fenced
/// by default (unlike STL atomics, which are not even guaranteed to be lock-free).
/// However, TBB atomics have stricter alignment requirements than their underlying value_types,
/// so a LeafBuffer with an atomic mOutOfCore is potentially ABI-incompatible with
/// its non-atomic counterpart.
/// This helper class conditionally declares mOutOfCore as an atomic only if doing so
/// doesn't break ABI compatibility.
template<typename T>
struct LeafBufferFlags
{
#if OPENVDB_ABI_VERSION_NUMBER >= 5
    /// The type of LeafBuffer::mOutOfCore
    using type = tbb::atomic<Index32>;
    static constexpr bool IsAtomic = true;
#else // OPENVDB_ABI_VERSION_NUMBER < 5
    // These structs need to have the same data members as LeafBuffer.
    struct Atomic { union { T* data; void* ptr; }; tbb::atomic<Index32> i; tbb::spin_mutex mutex; };
    struct NonAtomic { union { T* data; void* ptr; }; Index32 i; tbb::spin_mutex mutex; };

#ifndef __INTEL_COMPILER
    /// @c true if LeafBuffer::mOutOfCore is atomic, @c false otherwise
    static constexpr bool IsAtomic = ((sizeof(Atomic) == sizeof(NonAtomic))
         && (offsetof(Atomic, i) == offsetof(NonAtomic, i)));
#else
    // We can't use offsetof() with ICC, because it requires the arguments
    // to be POD types.  (C++11 requires only that they be standard layout types,
    // which Atomic and NonAtomic are.)
    static constexpr bool IsAtomic = (sizeof(Atomic) == sizeof(NonAtomic));
#endif
    /// The size of a LeafBuffer when LeafBuffer::mOutOfCore is atomic
    static constexpr size_t size = sizeof(Atomic);
    /// The type of LeafBuffer::mOutOfCore
    using type = typename std::conditional<IsAtomic, tbb::atomic<Index32>, Index32>::type;
#endif
};

} // namespace internal


/// @brief Array of fixed size 2<SUP>3<I>Log2Dim</I></SUP> that stores
/// the voxel values of a LeafNode
template<typename T, Index Log2Dim>
class LeafBuffer
{
public:
    using ValueType = T;
    using StorageType = ValueType;
    using NodeMaskType = util::NodeMask<Log2Dim>;
    static const Index SIZE = 1 << 3 * Log2Dim;

#if OPENVDB_ABI_VERSION_NUMBER >= 3
    struct FileInfo
    {
        FileInfo(): bufpos(0) , maskpos(0) {}
        std::streamoff bufpos;
        std::streamoff maskpos;
        io::MappedFile::Ptr mapping;
        SharedPtr<io::StreamMetadata> meta;
    };
#endif

#if OPENVDB_ABI_VERSION_NUMBER <= 2
    /// Default constructor
    LeafBuffer(): mData(new ValueType[SIZE]) {}
    /// Construct a buffer populated with the specified value.
    explicit LeafBuffer(const ValueType& val): mData(new ValueType[SIZE]) { this->fill(val); }
    /// Copy constructor
    LeafBuffer(const LeafBuffer& other): mData(new ValueType[SIZE]) { *this = other; }
    /// Destructor
    ~LeafBuffer() { delete[] mData; }

    /// Return @c true if this buffer's values have not yet been read from disk.
    bool isOutOfCore() const { return false; }
    /// Return @c true if memory for this buffer has not yet been allocated.
    bool empty() const { return (mData == nullptr); }
#else
    /// Default constructor
    inline LeafBuffer(): mData(new ValueType[SIZE]), mOutOfCore(0) {}
    /// Construct a buffer populated with the specified value.
    explicit inline LeafBuffer(const ValueType&);
    /// Copy constructor
    inline LeafBuffer(const LeafBuffer&);
    /// Construct a buffer but don't allocate memory for the full array of values.
    LeafBuffer(PartialCreate, const ValueType&): mData(nullptr), mOutOfCore(0) {}
    /// Destructor
    inline ~LeafBuffer();

    /// Return @c true if this buffer's values have not yet been read from disk.
    bool isOutOfCore() const { return bool(mOutOfCore); }
    /// Return @c true if memory for this buffer has not yet been allocated.
    bool empty() const { return !mData || this->isOutOfCore(); }
#endif
    /// Allocate memory for this buffer if it has not already been allocated.
    bool allocate() { if (mData == nullptr) mData = new ValueType[SIZE]; return true; }

    /// Populate this buffer with a constant value.
    inline void fill(const ValueType&);

    /// Return a const reference to the i'th element of this buffer.
    const ValueType& getValue(Index i) const { return this->at(i); }
    /// Return a const reference to the i'th element of this buffer.
    const ValueType& operator[](Index i) const { return this->at(i); }
    /// Set the i'th value of this buffer to the specified value.
    inline void setValue(Index i, const ValueType&);

    /// Copy the other buffer's values into this buffer.
    inline LeafBuffer& operator=(const LeafBuffer&);

    /// @brief Return @c true if the contents of the other buffer
    /// exactly equal the contents of this buffer.
    inline bool operator==(const LeafBuffer&) const;
    /// @brief Return @c true if the contents of the other buffer
    /// are not exactly equal to the contents of this buffer.
    inline bool operator!=(const LeafBuffer& other) const { return !(other == *this); }

    /// Exchange this buffer's values with the other buffer's values.
    inline void swap(LeafBuffer&);

    /// Return the memory footprint of this buffer in bytes.
    inline Index memUsage() const;
    /// Return the number of values contained in this buffer.
    static Index size() { return SIZE; }

    /// @brief Return a const pointer to the array of voxel values.
    /// @details This method guarantees that the buffer is allocated and loaded.
    /// @warning This method should only be used by experts seeking low-level optimizations.
    const ValueType* data() const;
    /// @brief Return a pointer to the array of voxel values.
    /// @details This method guarantees that the buffer is allocated and loaded.
    /// @warning This method should only be used by experts seeking low-level optimizations.
    ValueType* data();

private:
    /// If this buffer is empty, return zero, otherwise return the value at index @ i.
    inline const ValueType& at(Index i) const;

    /// @brief Return a non-const reference to the value at index @a i.
    /// @details This method is private since it makes assumptions about the
    /// buffer's memory layout.  LeafBuffers associated with custom leaf node types
    /// (e.g., a bool buffer implemented as a bitmask) might not be able to
    /// return non-const references to their values.
    ValueType& operator[](Index i) { return const_cast<ValueType&>(this->at(i)); }

    bool deallocate();

#if OPENVDB_ABI_VERSION_NUMBER <= 2
    void setOutOfCore(bool) {}
    void loadValues() const {}
    void doLoad() const {}
    bool detachFromFile() { return false; }
#else
    inline void setOutOfCore(bool b) { mOutOfCore = b; }
    // To facilitate inlining in the common case in which the buffer is in-core,
    // the loading logic is split into a separate function, doLoad().
    inline void loadValues() const { if (this->isOutOfCore()) this->doLoad(); }
    inline void doLoad() const;
    inline bool detachFromFile();
#endif


#if OPENVDB_ABI_VERSION_NUMBER <= 2
    ValueType* mData;
#else
    using FlagsType = typename internal::LeafBufferFlags<ValueType>::type;

    union {
        ValueType* mData;
        FileInfo*  mFileInfo;
    };
    FlagsType mOutOfCore; // interpreted as bool; extra bits reserved for future use
    tbb::spin_mutex mMutex; // 1 byte
    //int8_t mReserved[3]; // padding for alignment

    static const ValueType sZero;
#endif

    friend class ::TestLeaf;
    // Allow the parent LeafNode to access this buffer's data pointer.
    template<typename, Index> friend class LeafNode;
}; // class LeafBuffer


////////////////////////////////////////


#if OPENVDB_ABI_VERSION_NUMBER >= 3

template<typename T, Index Log2Dim>
const T LeafBuffer<T, Log2Dim>::sZero = zeroVal<T>();


template<typename T, Index Log2Dim>
inline
LeafBuffer<T, Log2Dim>::LeafBuffer(const ValueType& val)
    : mData(new ValueType[SIZE])
    , mOutOfCore(0)
{
    this->fill(val);
}


template<typename T, Index Log2Dim>
inline
LeafBuffer<T, Log2Dim>::~LeafBuffer()
{
    if (this->isOutOfCore()) {
        this->detachFromFile();
    } else {
        this->deallocate();
    }
}


template<typename T, Index Log2Dim>
inline
LeafBuffer<T, Log2Dim>::LeafBuffer(const LeafBuffer& other)
    : mData(nullptr)
    , mOutOfCore(other.mOutOfCore)
{
    if (other.isOutOfCore()) {
        mFileInfo = new FileInfo(*other.mFileInfo);
    } else if (other.mData != nullptr) {
        this->allocate();
        ValueType* target = mData;
        const ValueType* source = other.mData;
        Index n = SIZE;
        while (n--) *target++ = *source++;
    }
}

#endif // OPENVDB_ABI_VERSION_NUMBER >= 3


template<typename T, Index Log2Dim>
inline void
LeafBuffer<T, Log2Dim>::setValue(Index i, const ValueType& val)
{
    assert(i < SIZE);
#if OPENVDB_ABI_VERSION_NUMBER <= 2
    mData[i] = val;
#else
    this->loadValues();
    if (mData) mData[i] = val;
#endif
}


template<typename T, Index Log2Dim>
inline LeafBuffer<T, Log2Dim>&
LeafBuffer<T, Log2Dim>::operator=(const LeafBuffer& other)
{
    if (&other != this) {
#if OPENVDB_ABI_VERSION_NUMBER <= 2
        if (other.mData != nullptr) {
            this->allocate();
            ValueType* target = mData;
            const ValueType* source = other.mData;
            Index n = SIZE;
            while (n--) *target++ = *source++;
        }
#else // OPENVDB_ABI_VERSION_NUMBER >= 3
        if (this->isOutOfCore()) {
            this->detachFromFile();
        } else {
            if (other.isOutOfCore()) this->deallocate();
        }
        if (other.isOutOfCore()) {
            mOutOfCore = other.mOutOfCore;
            mFileInfo = new FileInfo(*other.mFileInfo);
        } else if (other.mData != nullptr) {
            this->allocate();
            ValueType* target = mData;
            const ValueType* source = other.mData;
            Index n = SIZE;
            while (n--) *target++ = *source++;
        }
#endif
    }
    return *this;
}


template<typename T, Index Log2Dim>
inline void
LeafBuffer<T, Log2Dim>::fill(const ValueType& val)
{
    this->detachFromFile();
    if (mData != nullptr) {
        ValueType* target = mData;
        Index n = SIZE;
        while (n--) *target++ = val;
    }
}


template<typename T, Index Log2Dim>
inline bool
LeafBuffer<T, Log2Dim>::operator==(const LeafBuffer& other) const
{
    this->loadValues();
    other.loadValues();
    const ValueType *target = mData, *source = other.mData;
    if (!target && !source) return true;
    if (!target || !source) return false;
    Index n = SIZE;
    while (n && math::isExactlyEqual(*target++, *source++)) --n;
    return n == 0;
}


template<typename T, Index Log2Dim>
inline void
LeafBuffer<T, Log2Dim>::swap(LeafBuffer& other)
{
    std::swap(mData, other.mData);
#if OPENVDB_ABI_VERSION_NUMBER >= 3
    std::swap(mOutOfCore, other.mOutOfCore);
#endif
}


template<typename T, Index Log2Dim>
inline Index
LeafBuffer<T, Log2Dim>::memUsage() const
{
    size_t n = sizeof(*this);
#if OPENVDB_ABI_VERSION_NUMBER <= 2
    if (mData) n += SIZE * sizeof(ValueType);
#else
    if (this->isOutOfCore()) n += sizeof(FileInfo);
    else if (mData) n += SIZE * sizeof(ValueType);
#endif
    return static_cast<Index>(n);
}


template<typename T, Index Log2Dim>
inline const typename LeafBuffer<T, Log2Dim>::ValueType*
LeafBuffer<T, Log2Dim>::data() const
{
#if OPENVDB_ABI_VERSION_NUMBER >= 3
    this->loadValues();
    if (mData == nullptr) {
        LeafBuffer* self = const_cast<LeafBuffer*>(this);
        // This lock will be contended at most once.
        tbb::spin_mutex::scoped_lock lock(self->mMutex);
        if (mData == nullptr) self->mData = new ValueType[SIZE];
    }
#endif
    return mData;
}

template<typename T, Index Log2Dim>
inline typename LeafBuffer<T, Log2Dim>::ValueType*
LeafBuffer<T, Log2Dim>::data()
{
#if OPENVDB_ABI_VERSION_NUMBER >= 3
    this->loadValues();
    if (mData == nullptr) {
        // This lock will be contended at most once.
        tbb::spin_mutex::scoped_lock lock(mMutex);
        if (mData == nullptr) mData = new ValueType[SIZE];
    }
#endif
    return mData;
}


template<typename T, Index Log2Dim>
inline const typename LeafBuffer<T, Log2Dim>::ValueType&
LeafBuffer<T, Log2Dim>::at(Index i) const
{
    assert(i < SIZE);
#if OPENVDB_ABI_VERSION_NUMBER <= 2
    return mData[i];
#else
    this->loadValues();
    // We can't use the ternary operator here, otherwise Visual C++ returns
    // a reference to a temporary.
    if (mData) return mData[i]; else return sZero;
#endif
}


template<typename T, Index Log2Dim>
inline bool
LeafBuffer<T, Log2Dim>::deallocate()
{
    if (mData != nullptr && !this->isOutOfCore()) {
        delete[] mData;
        mData = nullptr;
        return true;
    }
    return false;
}


#if OPENVDB_ABI_VERSION_NUMBER >= 3

template<typename T, Index Log2Dim>
inline void
LeafBuffer<T, Log2Dim>::doLoad() const
{
    if (!this->isOutOfCore()) return;

    LeafBuffer<T, Log2Dim>* self = const_cast<LeafBuffer<T, Log2Dim>*>(this);

    // This lock will be contended at most once, after which this buffer
    // will no longer be out-of-core.
    tbb::spin_mutex::scoped_lock lock(self->mMutex);
    if (!this->isOutOfCore()) return;

    std::unique_ptr<FileInfo> info(self->mFileInfo);
    assert(info.get() != nullptr);
    assert(info->mapping.get() != nullptr);
    assert(info->meta.get() != nullptr);

    /// @todo For now, we have to clear the mData pointer in order for allocate() to take effect.
    self->mData = nullptr;
    self->allocate();

    SharedPtr<std::streambuf> buf = info->mapping->createBuffer();
    std::istream is(buf.get());

    io::setStreamMetadataPtr(is, info->meta, /*transfer=*/true);

    NodeMaskType mask;
    is.seekg(info->maskpos);
    mask.load(is);

    is.seekg(info->bufpos);
    io::readCompressedValues(is, self->mData, SIZE, mask, io::getHalfFloat(is));

    self->setOutOfCore(false);
}


template<typename T, Index Log2Dim>
inline bool
LeafBuffer<T, Log2Dim>::detachFromFile()
{
    if (this->isOutOfCore()) {
        delete mFileInfo;
        mFileInfo = nullptr;
        this->setOutOfCore(false);
        return true;
    }
    return false;
}

#endif // OPENVDB_ABI_VERSION_NUMBER >= 3


////////////////////////////////////////


// Partial specialization for bool ValueType
template<Index Log2Dim>
class LeafBuffer<bool, Log2Dim>
{
public:
    using NodeMaskType = util::NodeMask<Log2Dim>;
    using WordType = typename NodeMaskType::Word;
    using ValueType = bool;
    using StorageType = WordType;

    static const Index WORD_COUNT = NodeMaskType::WORD_COUNT;
    static const Index SIZE = 1 << 3 * Log2Dim;

    // These static declarations must be on separate lines to avoid VC9 compiler errors.
    static const bool sOn;
    static const bool sOff;

    LeafBuffer() {}
    LeafBuffer(bool on): mData(on) {}
    LeafBuffer(const NodeMaskType& other): mData(other) {}
    LeafBuffer(const LeafBuffer& other): mData(other.mData) {}
    ~LeafBuffer() {}
    void fill(bool val) { mData.set(val); }
    LeafBuffer& operator=(const LeafBuffer& b) { if (&b != this) { mData=b.mData; } return *this; }

    const bool& getValue(Index i) const
    {
        assert(i < SIZE);
        // We can't use the ternary operator here, otherwise Visual C++ returns
        // a reference to a temporary.
        if (mData.isOn(i)) return sOn; else return sOff;
    }
    const bool& operator[](Index i) const { return this->getValue(i); }

    bool operator==(const LeafBuffer& other) const { return mData == other.mData; }
    bool operator!=(const LeafBuffer& other) const { return mData != other.mData; }

    void setValue(Index i, bool val) { assert(i < SIZE); mData.set(i, val); }

    void swap(LeafBuffer& other) { if (&other != this) std::swap(mData, other.mData); }

    Index memUsage() const { return sizeof(*this); }
    static Index size() { return SIZE; }

    /// @brief Return a pointer to the C-style array of words encoding the bits.
    /// @warning This method should only be used by experts seeking low-level optimizations.
    WordType* data() { return &(mData.template getWord<WordType>(0)); }
    /// @brief Return a const pointer to the C-style array of words encoding the bits.
    /// @warning This method should only be used by experts seeking low-level optimizations.
    const WordType* data() const { return const_cast<LeafBuffer*>(this)->data(); }

private:
    // Allow the parent LeafNode to access this buffer's data.
    template<typename, Index> friend class LeafNode;

    NodeMaskType mData;
}; // class LeafBuffer


/// @internal For consistency with other nodes and with iterators, methods like
/// LeafNode::getValue() return a reference to a value.  Since it's not possible
/// to return a reference to a bit in a node mask, we return a reference to one
/// of the following static values instead.
template<Index Log2Dim> const bool LeafBuffer<bool, Log2Dim>::sOn = true;
template<Index Log2Dim> const bool LeafBuffer<bool, Log2Dim>::sOff = false;

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_LEAFBUFFER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
