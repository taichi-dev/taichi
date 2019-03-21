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

/// @file points/StreamCompression.cc

#include "StreamCompression.h"
#include <openvdb/util/logging.h>
#include <map>
#ifdef OPENVDB_USE_BLOSC
#include <blosc.h>
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace compression {


#ifdef OPENVDB_USE_BLOSC


bool
bloscCanCompress()
{
    return true;
}


size_t
bloscUncompressedSize(const char* buffer)
{
    size_t bytes, _1, _2;
    blosc_cbuffer_sizes(buffer, &bytes, &_1, &_2);
    return bytes;
}


void
bloscCompress(char* compressedBuffer, size_t& compressedBytes, const size_t bufferBytes,
    const char* uncompressedBuffer, const size_t uncompressedBytes)
{
    if (bufferBytes > BLOSC_MAX_BUFFERSIZE) {
        OPENVDB_LOG_DEBUG("Blosc compress failed due to exceeding maximum buffer size.");
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }
    if (bufferBytes < uncompressedBytes + BLOSC_MAX_OVERHEAD) {
        OPENVDB_LOG_DEBUG("Blosc compress failed due to insufficient space in compressed buffer.");
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }

    if (uncompressedBytes <= BLOSC_MINIMUM_BYTES) {
        // no Blosc compression performed below this limit
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }

    if (uncompressedBytes < BLOSC_PAD_BYTES && bufferBytes < BLOSC_PAD_BYTES + BLOSC_MAX_OVERHEAD) {
        OPENVDB_LOG_DEBUG(
            "Blosc compress failed due to insufficient space in compressed buffer for padding.");
        compressedBytes = 0;
        compressedBuffer = nullptr;
        return;
    }

    size_t inputBytes = uncompressedBytes;

    const char* buffer = uncompressedBuffer;

    std::unique_ptr<char[]> paddedBuffer;
    if (uncompressedBytes < BLOSC_PAD_BYTES) {
        // input array padded with zeros below this limit to improve compression
        paddedBuffer.reset(new char[BLOSC_PAD_BYTES]);
        std::memcpy(paddedBuffer.get(), buffer, uncompressedBytes);
        for (int i = static_cast<int>(uncompressedBytes); i < BLOSC_PAD_BYTES; i++) {
            paddedBuffer.get()[i] = 0;
        }
        buffer = paddedBuffer.get();
        inputBytes = BLOSC_PAD_BYTES;
    }

    int _compressedBytes = blosc_compress_ctx(
        /*clevel=*/9, // 0 (no compression) to 9 (maximum compression)
        /*doshuffle=*/true,
        /*typesize=*/sizeof(float), // hard-coded to 4-bytes for better compression
        /*srcsize=*/inputBytes,
        /*src=*/buffer,
        /*dest=*/compressedBuffer,
        /*destsize=*/bufferBytes,
        BLOSC_LZ4_COMPNAME,
        /*blocksize=*/inputBytes,
        /*numthreads=*/1);

    if (_compressedBytes <= 0) {
        std::ostringstream ostr;
        ostr << "Blosc failed to compress " << uncompressedBytes << " byte"
            << (uncompressedBytes == 1 ? "" : "s");
        if (_compressedBytes < 0) ostr << " (internal error " << _compressedBytes << ")";
        OPENVDB_LOG_DEBUG(ostr.str());
        compressedBytes = 0;
        return;
    }

    compressedBytes = _compressedBytes;

    // fail if compression does not result in a smaller buffer
    if (compressedBytes >= uncompressedBytes) {
        compressedBytes = 0;
    }
}


std::unique_ptr<char[]>
bloscCompress(const char* buffer, const size_t uncompressedBytes, size_t& compressedBytes,
    const bool resize)
{
    size_t tempBytes = uncompressedBytes;
    // increase temporary buffer for padding if necessary
    if (tempBytes >= BLOSC_MINIMUM_BYTES && tempBytes < BLOSC_PAD_BYTES) {
        tempBytes += BLOSC_PAD_BYTES;
    }
    // increase by Blosc max overhead
    tempBytes += BLOSC_MAX_OVERHEAD;
    const bool outOfRange = tempBytes > BLOSC_MAX_BUFFERSIZE;
    std::unique_ptr<char[]> outBuffer(outOfRange ? new char[1] : new char[tempBytes]);

    bloscCompress(outBuffer.get(), compressedBytes, tempBytes, buffer, uncompressedBytes);

    if (compressedBytes == 0) {
        return nullptr;
    }

    // buffer size is larger due to Blosc overhead so resize
    // (resize can be skipped if the buffer is only temporary)

    if (resize) {
        std::unique_ptr<char[]> newBuffer(new char[compressedBytes]);
        std::memcpy(newBuffer.get(), outBuffer.get(), compressedBytes);
        outBuffer.reset(newBuffer.release());
    }

    return outBuffer;
}


size_t
bloscCompressedSize( const char* buffer, const size_t uncompressedBytes)
{
    size_t compressedBytes;
    bloscCompress(buffer, uncompressedBytes, compressedBytes, /*resize=*/false);
    return compressedBytes;
}


void
bloscDecompress(char* uncompressedBuffer, const size_t expectedBytes,
    const size_t bufferBytes, const char* compressedBuffer)
{
    size_t uncompressedBytes = bloscUncompressedSize(compressedBuffer);

    if (bufferBytes > BLOSC_MAX_BUFFERSIZE) {
        OPENVDB_THROW(RuntimeError,
            "Blosc decompress failed due to exceeding maximum buffer size.");
    }
    if (bufferBytes < uncompressedBytes + BLOSC_MAX_OVERHEAD) {
        OPENVDB_THROW(RuntimeError,
            "Blosc decompress failed due to insufficient space in uncompressed buffer.");
    }

    uncompressedBytes = blosc_decompress_ctx(   /*src=*/compressedBuffer,
                                                /*dest=*/uncompressedBuffer,
                                                bufferBytes,
                                                /*numthreads=*/1);

    if (uncompressedBytes < 1) {
        OPENVDB_THROW(RuntimeError, "Blosc decompress returned error code " << uncompressedBytes);
    }

    if (uncompressedBytes == BLOSC_PAD_BYTES && expectedBytes <= BLOSC_PAD_BYTES) {
        // padded array to improve compression
    }
    else if (uncompressedBytes != expectedBytes) {
        OPENVDB_THROW(RuntimeError, "Expected to decompress " << expectedBytes
            << " byte" << (expectedBytes == 1 ? "" : "s") << ", got "
            << uncompressedBytes << " byte" << (uncompressedBytes == 1 ? "" : "s"));
    }
}


std::unique_ptr<char[]>
bloscDecompress(const char* buffer, const size_t expectedBytes, const bool resize)
{
    size_t uncompressedBytes = bloscUncompressedSize(buffer);
    size_t tempBytes = uncompressedBytes + BLOSC_MAX_OVERHEAD;
    const bool outOfRange = tempBytes > BLOSC_MAX_BUFFERSIZE;
    if (outOfRange)     tempBytes = 1;
    std::unique_ptr<char[]> outBuffer(new char[tempBytes]);

    bloscDecompress(outBuffer.get(), expectedBytes, tempBytes, buffer);

    // buffer size is larger due to Blosc overhead so resize
    // (resize can be skipped if the buffer is only temporary)

    if (resize) {
        std::unique_ptr<char[]> newBuffer(new char[expectedBytes]);
        std::memcpy(newBuffer.get(), outBuffer.get(), expectedBytes);
        outBuffer.reset(newBuffer.release());
    }

    return outBuffer;
}


#else


bool
bloscCanCompress()
{
    OPENVDB_LOG_DEBUG("Can't compress array data without the blosc library.");
    return false;
}


size_t
bloscUncompressedSize(const char*)
{
    OPENVDB_THROW(RuntimeError, "Can't extract compressed data without the blosc library.");
}


void
bloscCompress(char*, size_t& compressedBytes, const size_t, const char*, const size_t)
{
    OPENVDB_LOG_DEBUG("Can't compress array data without the blosc library.");
    compressedBytes = 0;
}


std::unique_ptr<char[]>
bloscCompress(const char*, const size_t, size_t& compressedBytes, const bool)
{
    OPENVDB_LOG_DEBUG("Can't compress array data without the blosc library.");
    compressedBytes = 0;
    return nullptr;
}


size_t
bloscCompressedSize(const char*, const size_t)
{
    OPENVDB_LOG_DEBUG("Can't compress array data without the blosc library.");
    return 0;
}


void
bloscDecompress(char*, const size_t, const size_t, const char*)
{
    OPENVDB_THROW(RuntimeError, "Can't extract compressed data without the blosc library.");
}


std::unique_ptr<char[]>
bloscDecompress(const char*, const size_t, const bool)
{
    OPENVDB_THROW(RuntimeError, "Can't extract compressed data without the blosc library.");
}


#endif // OPENVDB_USE_BLOSC


////////////////////////////////////////


void
Page::load() const
{
    this->doLoad();
}


long
Page::uncompressedBytes() const
{
    assert(mInfo);
    return mInfo->uncompressedBytes;
}


const char*
Page::buffer(const int index) const
{
    if (this->isOutOfCore())   this->load();

    return mData.get() + index;
}


void
Page::readHeader(std::istream& is)
{
    assert(mInfo);

    // read the (compressed) size of the page
    int compressedSize;
    is.read(reinterpret_cast<char*>(&compressedSize), sizeof(int));

    int uncompressedSize;
    // if uncompressed, read the (compressed) size of the page
    if (compressedSize > 0)     is.read(reinterpret_cast<char*>(&uncompressedSize), sizeof(int));
    else                        uncompressedSize = -compressedSize;

    assert(compressedSize != 0);
    assert(uncompressedSize != 0);

    mInfo->compressedBytes = compressedSize;
    mInfo->uncompressedBytes = uncompressedSize;
}


void
Page::readBuffers(std::istream&is, bool delayed)
{
    assert(mInfo);

    bool isCompressed = mInfo->compressedBytes > 0;

    io::MappedFile::Ptr mappedFile = io::getMappedFilePtr(is);

    if (delayed && mappedFile) {
        SharedPtr<io::StreamMetadata> meta = io::getStreamMetadataPtr(is);
        assert(meta);

        std::streamoff filepos = is.tellg();

        // seek over the page
        is.seekg((isCompressed ? mInfo->compressedBytes : -mInfo->compressedBytes),
            std::ios_base::cur);

        mInfo->mappedFile = mappedFile;
        mInfo->meta = meta;
        mInfo->filepos = filepos;

        assert(mInfo->mappedFile);
    }
    else {
        std::unique_ptr<char[]> buffer(new char[
            (isCompressed ? mInfo->compressedBytes : -mInfo->compressedBytes)]);
        is.read(buffer.get(), (isCompressed ? mInfo->compressedBytes : -mInfo->compressedBytes));

        if (mInfo->compressedBytes > 0) {
            this->decompress(buffer);
        } else {
            this->copy(buffer, -static_cast<int>(mInfo->compressedBytes));
        }
        mInfo.reset();
    }
}


bool
Page::isOutOfCore() const
{
    return bool(mInfo);
}


void
Page::copy(const std::unique_ptr<char[]>& temp, int pageSize)
{
    mData.reset(new char[pageSize]);
    std::memcpy(mData.get(), temp.get(), pageSize);
}


void
Page::decompress(const std::unique_ptr<char[]>& temp)
{
    size_t uncompressedBytes = bloscUncompressedSize(temp.get());
    size_t tempBytes = uncompressedBytes;
#ifdef OPENVDB_USE_BLOSC
    tempBytes += uncompressedBytes;
#endif
    mData.reset(new char[tempBytes]);

    bloscDecompress(mData.get(), uncompressedBytes, tempBytes, temp.get());
}


void
Page::doLoad() const
{
    if (!this->isOutOfCore())  return;

    Page* self = const_cast<Page*>(this);

    // This lock will be contended at most once, after which this buffer
    // will no longer be out-of-core.
    tbb::spin_mutex::scoped_lock lock(self->mMutex);
    if (!this->isOutOfCore()) return;

    assert(self->mInfo);

    int compressedBytes = static_cast<int>(self->mInfo->compressedBytes);
    bool compressed = compressedBytes > 0;
    if (!compressed) compressedBytes = -compressedBytes;

    assert(compressedBytes);

    std::unique_ptr<char[]> temp(new char[compressedBytes]);

    assert(self->mInfo->mappedFile);
    SharedPtr<std::streambuf> buf = self->mInfo->mappedFile->createBuffer();
    assert(buf);

    std::istream is(buf.get());
    io::setStreamMetadataPtr(is, self->mInfo->meta, /*transfer=*/true);
    is.seekg(self->mInfo->filepos);

    is.read(temp.get(), compressedBytes);

    if (compressed)     self->decompress(temp);
    else                self->copy(temp, compressedBytes);

    self->mInfo.reset();
}


////////////////////////////////////////


PageHandle::PageHandle( const Page::Ptr& page, const int index, const int size)
    : mPage(page)
    , mIndex(index)
    , mSize(size)
{
}


Page&
PageHandle::page()
{
    assert(mPage);
    return *mPage;
}


std::unique_ptr<char[]>
PageHandle::read()
{
    assert(mIndex >= 0);
    assert(mSize > 0);
    std::unique_ptr<char[]> buffer(new char[mSize]);
    std::memcpy(buffer.get(), mPage->buffer(mIndex), mSize);
    return buffer;
}


////////////////////////////////////////


PagedInputStream::PagedInputStream(std::istream& is)
    : mIs(&is)
{
}


PageHandle::Ptr
PagedInputStream::createHandle(std::streamsize n)
{
    assert(mByteIndex <= mUncompressedBytes);

    if (mByteIndex == mUncompressedBytes) {

        mPage = std::make_shared<Page>();
        mPage->readHeader(*mIs);
        mUncompressedBytes = static_cast<int>(mPage->uncompressedBytes());
        mByteIndex = 0;
    }

#if OPENVDB_ABI_VERSION_NUMBER >= 6
    // TODO: C++14 introduces std::make_unique
    PageHandle::Ptr pageHandle(new PageHandle(mPage, mByteIndex, n));
#else
    PageHandle::Ptr pageHandle = std::make_shared<PageHandle>(mPage, mByteIndex, n);
#endif

    mByteIndex += int(n);

    return pageHandle;
}


void
PagedInputStream::read(PageHandle::Ptr& pageHandle, std::streamsize n, bool delayed)
{
    assert(mByteIndex <= mUncompressedBytes);

    Page& page = pageHandle->page();

    if (mByteIndex == mUncompressedBytes) {
        mUncompressedBytes = static_cast<int>(page.uncompressedBytes());
        page.readBuffers(*mIs, delayed);
        mByteIndex = 0;
    }

    mByteIndex += int(n);
}


////////////////////////////////////////


PagedOutputStream::PagedOutputStream()
{
#ifdef OPENVDB_USE_BLOSC
    mCompressedData.reset(new char[PageSize + BLOSC_MAX_OVERHEAD]);
#endif
}


PagedOutputStream::PagedOutputStream(std::ostream& os)
    : mOs(&os)
{
#ifdef OPENVDB_USE_BLOSC
    mCompressedData.reset(new char[PageSize + BLOSC_MAX_OVERHEAD]);
#endif
}


PagedOutputStream&
PagedOutputStream::write(const char* str, std::streamsize n)
{
    if (n > PageSize) {
        this->flush();
        // write out the block as if a whole page
        this->compressAndWrite(str, size_t(n));
    }
    else {
        // if the size of this block will overflow the page, flush to disk
        if ((int(n) + mBytes) > PageSize) {
            this->flush();
        }

        // store and increment the data in the current page
        std::memcpy(mData.get() + mBytes, str, n);
        mBytes += int(n);
    }

    return *this;
}


void
PagedOutputStream::flush()
{
    this->compressAndWrite(mData.get(), mBytes);
    mBytes = 0;
}


void
PagedOutputStream::compressAndWrite(const char* buffer, size_t size)
{
    if (size == 0)  return;

    assert(size < std::numeric_limits<int>::max());

    this->resize(size);

    size_t compressedBytes(0);
    if (mSizeOnly) {
#ifdef OPENVDB_USE_BLOSC
        compressedBytes = bloscCompressedSize(buffer, size);
#endif
    }
    else {
#ifdef OPENVDB_USE_BLOSC
        bloscCompress(mCompressedData.get(), compressedBytes, mCapacity + BLOSC_MAX_OVERHEAD, buffer, size);
#endif
    }

    if (compressedBytes == 0) {
        int uncompressedBytes = -static_cast<int>(size);
        if (mSizeOnly) {
            mOs->write(reinterpret_cast<const char*>(&uncompressedBytes), sizeof(int));
        }
        else {
            mOs->write(buffer, size);
        }
    }
    else {
        if (mSizeOnly) {
            mOs->write(reinterpret_cast<const char*>(&compressedBytes), sizeof(int));
            mOs->write(reinterpret_cast<const char*>(&size), sizeof(int));
        }
        else {
#ifdef OPENVDB_USE_BLOSC
            mOs->write(mCompressedData.get(), compressedBytes);
#else
            OPENVDB_THROW(RuntimeError, "Cannot write out compressed data without Blosc.");
#endif
        }
    }
}


void
PagedOutputStream::resize(size_t size)
{
    // grow the capacity if not sufficient space
    size_t requiredSize = size;
    if (size < BLOSC_PAD_BYTES && size >= BLOSC_MINIMUM_BYTES) {
        requiredSize = BLOSC_PAD_BYTES;
    }
    if (requiredSize > mCapacity) {
        mCapacity = requiredSize;
        mData.reset(new char[mCapacity]);
#ifdef OPENVDB_USE_BLOSC
        mCompressedData.reset(new char[mCapacity + BLOSC_MAX_OVERHEAD]);
#endif
    }
}

} // namespace compression
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
