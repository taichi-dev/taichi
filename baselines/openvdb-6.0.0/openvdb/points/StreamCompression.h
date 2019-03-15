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

/// @file points/StreamCompression.h
///
/// @author Dan Bailey
///
/// @brief Convenience wrappers to using Blosc and reading and writing of Paged data.
///
/// Blosc is most effective with large (> ~256KB) blocks of data. Writing the entire
/// data block contiguously would provide the most optimal compression, however would
/// limit the ability to use delayed-loading as the whole block would be required to
/// be loaded from disk at once. To balance these two competing factors, Paging is used
/// to write out blocks of data that are a reasonable size for Blosc. These Pages are
/// loaded lazily, tracking the input stream pointers and creating Handles that reference
/// portions of the buffer. When the Page buffer is accessed, the data will be read from
/// the stream.

#ifndef OPENVDB_TOOLS_STREAM_COMPRESSION_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_STREAM_COMPRESSION_HAS_BEEN_INCLUDED

#include <openvdb/io/io.h>
#include <tbb/spin_mutex.h>
#include <memory>
#include <string>


class TestStreamCompression;

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace compression {


// This is the minimum number of bytes below which Blosc compression is not used to
// avoid unecessary computation, as Blosc offers minimal compression until this limit
static const int BLOSC_MINIMUM_BYTES = 48;

// This is the minimum number of bytes below which the array is padded with zeros up
// to this number of bytes to allow Blosc to perform compression with small arrays
static const int BLOSC_PAD_BYTES = 128;


/// @brief Returns true if compression is available
OPENVDB_API bool bloscCanCompress();

/// @brief Retrieves the uncompressed size of buffer when uncompressed
///
/// @param buffer the compressed buffer
OPENVDB_API size_t bloscUncompressedSize(const char* buffer);

/// @brief Compress into the supplied buffer.
///
/// @param compressedBuffer     the buffer to compress
/// @param compressedBytes      number of compressed bytes
/// @param bufferBytes          the number of bytes in compressedBuffer available to be filled
/// @param uncompressedBuffer   the uncompressed buffer to compress
/// @param uncompressedBytes    number of uncompressed bytes
OPENVDB_API void bloscCompress(char* compressedBuffer, size_t& compressedBytes,
    const size_t bufferBytes, const char* uncompressedBuffer, const size_t uncompressedBytes);

/// @brief Compress and return the heap-allocated compressed buffer.
///
/// @param buffer               the buffer to compress
/// @param uncompressedBytes    number of uncompressed bytes
/// @param compressedBytes      number of compressed bytes (written to this variable)
/// @param resize               the compressed buffer will be exactly resized to remove the
///                             portion used for Blosc overhead, for efficiency this can be
///                             skipped if it is known that the resulting buffer is temporary
OPENVDB_API std::unique_ptr<char[]> bloscCompress(const char* buffer,
    const size_t uncompressedBytes, size_t& compressedBytes, const bool resize = true);

/// @brief Convenience wrapper to retrieve the compressed size of buffer when compressed
///
/// @param buffer the uncompressed buffer
/// @param uncompressedBytes number of uncompressed bytes
OPENVDB_API size_t bloscCompressedSize(const char* buffer, const size_t uncompressedBytes);

/// @brief Decompress into the supplied buffer. Will throw if decompression fails or
///        uncompressed buffer has insufficient space in which to decompress.
///
/// @param uncompressedBuffer the uncompressed buffer to decompress into
/// @param expectedBytes the number of bytes expected once the buffer is decompressed
/// @param bufferBytes the number of bytes in uncompressedBuffer available to be filled
/// @param compressedBuffer the compressed buffer to decompress
OPENVDB_API void bloscDecompress(char* uncompressedBuffer, const size_t expectedBytes,
    const size_t bufferBytes, const char* compressedBuffer);

/// @brief Decompress and return the the heap-allocated uncompressed buffer.
///
/// @param buffer the buffer to decompress
/// @param expectedBytes the number of bytes expected once the buffer is decompressed
/// @param resize               the compressed buffer will be exactly resized to remove the
///                             portion used for Blosc overhead, for efficiency this can be
///                             skipped if it is known that the resulting buffer is temporary
OPENVDB_API std::unique_ptr<char[]> bloscDecompress(const char* buffer,
    const size_t expectedBytes, const bool resize = true);


////////////////////////////////////////


// 1MB = 1048576 Bytes
static const int PageSize = 1024 * 1024;


/// @brief Stores a variable-size, compressed, delayed-load Page of data
/// that is loaded into memory when accessed. Access to the Page is
/// thread-safe as loading and decompressing the data is protected by a mutex.
class OPENVDB_API Page
{
private:
    struct Info
    {
        io::MappedFile::Ptr mappedFile;
        SharedPtr<io::StreamMetadata> meta;
        std::streamoff filepos;
        long compressedBytes;
        long uncompressedBytes;
    }; // Info

public:
    using Ptr = std::shared_ptr<Page>;

    Page() = default;

    /// @brief load the Page into memory
    void load() const;

    /// @brief Uncompressed bytes of the Paged data, available
    /// when the header has been read.
    long uncompressedBytes() const;

    /// @brief Retrieves a data pointer at the specific @param index
    /// @note Will force a Page load when called.
    const char* buffer(const int index) const;

    /// @brief Read the Page header
    void readHeader(std::istream&);

    /// @brief Read the Page buffers. If @a delayed is true, stream
    /// pointers will be stored to load the data lazily.
    void readBuffers(std::istream&, bool delayed);

    /// @brief Test if the data is out-of-core
    bool isOutOfCore() const;

private:
    /// @brief Convenience method to store a copy of the supplied buffer
    void copy(const std::unique_ptr<char[]>& temp, int pageSize);

    /// @brief Decompress and store the supplied data
    void decompress(const std::unique_ptr<char[]>& temp);

    /// @brief Thread-safe loading of the data
    void doLoad() const;

    std::unique_ptr<Info> mInfo = std::unique_ptr<Info>(new Info);
    std::unique_ptr<char[]> mData;
    tbb::spin_mutex mMutex;
}; // class Page


/// @brief A PageHandle holds a unique ptr to a Page and a specific stream
/// pointer to a point within the decompressed Page buffer
class OPENVDB_API PageHandle
{
public:
#if OPENVDB_ABI_VERSION_NUMBER >= 6
    using Ptr = std::unique_ptr<PageHandle>;
#else
    using Ptr = std::shared_ptr<PageHandle>;
#endif

    /// @brief Create the page handle
    /// @param page a shared ptr to the page that stores the buffer
    /// @param index start position of the buffer to be read
    /// @param size total size of the buffer to be read in bytes
    PageHandle(const Page::Ptr& page, const int index, const int size);

    /// @brief Retrieve a reference to the stored page
    Page& page();

    /// @brief Return the size of the buffer
    int size() const { return mSize; }

    /// @brief Read and return the buffer, loading and decompressing
    /// the Page if necessary.
    std::unique_ptr<char[]> read();

    /// @brief Return a copy of this PageHandle
    Ptr copy() { return Ptr(new PageHandle(mPage, mIndex, mSize)); }

protected:
    friend class ::TestStreamCompression;

private:
    Page::Ptr mPage;
    int mIndex = -1;
    int mSize = 0;
}; // class PageHandle


/// @brief A Paging wrapper to std::istream that is responsible for reading
/// from a given input stream and creating Page objects and PageHandles that
/// reference those pages for delayed reading.
class OPENVDB_API PagedInputStream
{
public:
    using Ptr = std::shared_ptr<PagedInputStream>;

    PagedInputStream() = default;

    explicit PagedInputStream(std::istream& is);

    /// @brief Size-only mode tags the stream as only reading size data.
    void setSizeOnly(bool sizeOnly) { mSizeOnly = sizeOnly; }
    bool sizeOnly() const { return mSizeOnly; }

    // @brief Set and get the input stream
    std::istream& getInputStream() { assert(mIs); return *mIs; }
    void setInputStream(std::istream& is) { mIs = &is; }

    /// @brief Creates a PageHandle to access the next @param n bytes of the Page.
    PageHandle::Ptr createHandle(std::streamsize n);

    /// @brief Takes a @a pageHandle and updates the referenced page with the
    /// current stream pointer position and if @a delayed is false performs
    /// an immediate read of the data.
    void read(PageHandle::Ptr& pageHandle, std::streamsize n, bool delayed = true);

private:
    int mByteIndex = 0;
    int mUncompressedBytes = 0;
    std::istream* mIs = nullptr;
    Page::Ptr mPage;
    bool mSizeOnly = false;
}; // class PagedInputStream


/// @brief A Paging wrapper to std::ostream that is responsible for writing
/// from a given output stream at intervals set by the PageSize. As Pages are
/// variable in size, they are flushed to disk as soon as sufficiently large.
class OPENVDB_API PagedOutputStream
{
public:
    using Ptr = std::shared_ptr<PagedOutputStream>;

    PagedOutputStream();

    explicit PagedOutputStream(std::ostream& os);

    /// @brief Size-only mode tags the stream as only writing size data.
    void setSizeOnly(bool sizeOnly) { mSizeOnly = sizeOnly; }
    bool sizeOnly() const { return mSizeOnly; }

    /// @brief Set and get the output stream
    std::ostream& getOutputStream() { assert(mOs); return *mOs; }
    void setOutputStream(std::ostream& os) { mOs = &os; }

    /// @brief Writes the given @param str buffer of size @param n
    PagedOutputStream& write(const char* str, std::streamsize n);

    /// @brief Manually flushes the current page to disk if non-zero
    void flush();

private:
    /// @brief Compress the @param buffer of @param size bytes and write
    /// out to the stream.
    void compressAndWrite(const char* buffer, size_t size);

    /// @brief Resize the internal page buffer to @param size bytes
    void resize(size_t size);

    std::unique_ptr<char[]> mData = std::unique_ptr<char[]>(new char[PageSize]);
    std::unique_ptr<char[]> mCompressedData = nullptr;
    size_t mCapacity = PageSize;
    int mBytes = 0;
    std::ostream* mOs = nullptr;
    bool mSizeOnly = false;
}; // class PagedOutputStream


} // namespace compression
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_STREAM_COMPRESSION_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
