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

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/points/StreamCompression.h>

#include <openvdb/io/Compression.h> // io::COMPRESS_BLOSC

#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-macros"
#endif
// Boost.Interprocess uses a header-only portion of Boost.DateTime
#define BOOST_DATE_TIME_NO_LIB
#ifdef __clang__
#pragma GCC diagnostic pop
#endif
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/system/error_code.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/version.hpp> // for BOOST_VERSION

#include <tbb/atomic.h>

#ifdef _MSC_VER
#include <boost/interprocess/detail/os_file_functions.hpp> // open_existing_file(), close_file()
// boost::interprocess::detail was renamed to boost::interprocess::ipcdetail in Boost 1.48.
// Ensure that both namespaces exist.
namespace boost { namespace interprocess { namespace detail {} namespace ipcdetail {} } }
#include <windows.h>
#else
#include <sys/types.h> // for struct stat
#include <sys/stat.h> // for stat()
#include <unistd.h> // for unlink()
#endif

#include <fstream>
#include <numeric> // for std::iota()

#ifdef OPENVDB_USE_BLOSC
#include <blosc.h>
// A Blosc optimization introduced in 1.11.0 uses a slightly smaller block size for
// HCR codecs (LZ4, ZLIB, ZSTD), which otherwise fails a few regression test cases
#if BLOSC_VERSION_MAJOR > 0 && BLOSC_VERSION_MINOR > 10
#define BLOSC_HCR_BLOCKSIZE_OPTIMIZATION
#endif
#endif

/// @brief io::MappedFile has a private constructor, so this unit tests uses a matching proxy
class ProxyMappedFile
{
public:
    explicit ProxyMappedFile(const std::string& filename)
        : mImpl(new Impl(filename)) { }

private:
    class Impl
    {
    public:
        Impl(const std::string& filename)
            : mMap(filename.c_str(), boost::interprocess::read_only)
            , mRegion(mMap, boost::interprocess::read_only)
        {
            mLastWriteTime = 0;
            const char* regionFilename = mMap.get_name();
#ifdef _MSC_VER
            using namespace boost::interprocess::detail;
            using namespace boost::interprocess::ipcdetail;
            using openvdb::Index64;

            if (void* fh = open_existing_file(regionFilename, boost::interprocess::read_only)) {
                FILETIME mtime;
                if (GetFileTime(fh, nullptr, nullptr, &mtime)) {
                    mLastWriteTime = (Index64(mtime.dwHighDateTime) << 32) | mtime.dwLowDateTime;
                }
                close_file(fh);
            }
#else
            struct stat info;
            if (0 == ::stat(regionFilename, &info)) {
                mLastWriteTime = openvdb::Index64(info.st_mtime);
            }
#endif
        }

        using Notifier = std::function<void(std::string /*filename*/)>;
        boost::interprocess::file_mapping mMap;
        boost::interprocess::mapped_region mRegion;
        bool mAutoDelete = false;
        Notifier mNotifier;
        mutable tbb::atomic<openvdb::Index64> mLastWriteTime;
    }; // class Impl
    std::unique_ptr<Impl> mImpl;
}; // class ProxyMappedFile

using namespace openvdb;
using namespace openvdb::compression;

class TestStreamCompression: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestStreamCompression);
    CPPUNIT_TEST(testBlosc);
    CPPUNIT_TEST(testPagedStreams);

    CPPUNIT_TEST_SUITE_END();

    void testBlosc();
    void testPagedStreams();
}; // class TestStreamCompression

CPPUNIT_TEST_SUITE_REGISTRATION(TestStreamCompression);


////////////////////////////////////////


void
TestStreamCompression::testBlosc()
{
    // ensure that the library and unit tests are both built with or without Blosc enabled
#ifdef OPENVDB_USE_BLOSC
    CPPUNIT_ASSERT(bloscCanCompress());
#else
    CPPUNIT_ASSERT(!bloscCanCompress());
#endif

    const int count = 256;

    { // valid buffer
        // compress

        std::unique_ptr<int[]> uncompressedBuffer(new int[count]);

        for (int i = 0; i < count; i++) {
            uncompressedBuffer.get()[i] = i / 2;
        }

        size_t uncompressedBytes = count * sizeof(int);
        size_t compressedBytes;

        size_t testCompressedBytes = bloscCompressedSize(
            reinterpret_cast<char*>(uncompressedBuffer.get()), uncompressedBytes);

        std::unique_ptr<char[]> compressedBuffer = bloscCompress(
            reinterpret_cast<char*>(uncompressedBuffer.get()), uncompressedBytes, compressedBytes);

#ifdef OPENVDB_USE_BLOSC
        CPPUNIT_ASSERT(compressedBytes < uncompressedBytes);
        CPPUNIT_ASSERT(compressedBuffer);
        CPPUNIT_ASSERT_EQUAL(testCompressedBytes, compressedBytes);

        // uncompressedSize

        CPPUNIT_ASSERT_EQUAL(uncompressedBytes, bloscUncompressedSize(compressedBuffer.get()));

        // decompress

        std::unique_ptr<char[]> newUncompressedBuffer =
            bloscDecompress(compressedBuffer.get(), uncompressedBytes);

        // incorrect number of expected bytes
        CPPUNIT_ASSERT_THROW(newUncompressedBuffer =
            bloscDecompress(compressedBuffer.get(), 1), openvdb::RuntimeError);

        CPPUNIT_ASSERT(newUncompressedBuffer);
#else
        CPPUNIT_ASSERT(!compressedBuffer);
        CPPUNIT_ASSERT_EQUAL(testCompressedBytes, size_t(0));

        // uncompressedSize

        CPPUNIT_ASSERT_THROW(bloscUncompressedSize(compressedBuffer.get()), openvdb::RuntimeError);

        // decompress

        std::unique_ptr<char[]> newUncompressedBuffer;
        CPPUNIT_ASSERT_THROW(
            newUncompressedBuffer = bloscDecompress(compressedBuffer.get(), uncompressedBytes),
            openvdb::RuntimeError);

        CPPUNIT_ASSERT(!newUncompressedBuffer);
#endif
    }

    { // one value (below minimum bytes)
        std::unique_ptr<int[]> uncompressedBuffer(new int[1]);
        uncompressedBuffer.get()[0] = 10;

        size_t compressedBytes;

        std::unique_ptr<char[]> compressedBuffer = bloscCompress(
            reinterpret_cast<char*>(uncompressedBuffer.get()), sizeof(int), compressedBytes);

        CPPUNIT_ASSERT(!compressedBuffer);
        CPPUNIT_ASSERT_EQUAL(compressedBytes, size_t(0));
    }

    { // padded buffer
        std::unique_ptr<char[]> largeBuffer(new char[2048]);

        for (int paddedCount = 1; paddedCount < 256; paddedCount++) {

            std::unique_ptr<char[]> newTest(new char[paddedCount]);
            for (int i = 0; i < paddedCount; i++)  newTest.get()[i] = char(0);

#ifdef OPENVDB_USE_BLOSC
            size_t compressedBytes;
            std::unique_ptr<char[]> compressedBuffer = bloscCompress(
                newTest.get(), paddedCount, compressedBytes);

            // compress into a large buffer to check for any padding issues
            size_t compressedSizeBytes;
            bloscCompress(largeBuffer.get(), compressedSizeBytes, size_t(2048),
                newTest.get(), paddedCount);

            // regardless of compression, these numbers should always match
            CPPUNIT_ASSERT_EQUAL(compressedSizeBytes, compressedBytes);

            // no compression performed due to buffer being too small
            if (paddedCount <= BLOSC_MINIMUM_BYTES) {
                CPPUNIT_ASSERT(!compressedBuffer);
            }
            else {
                CPPUNIT_ASSERT(compressedBuffer);
                CPPUNIT_ASSERT(compressedBytes > 0);
                CPPUNIT_ASSERT(int(compressedBytes) < paddedCount);

                std::unique_ptr<char[]> uncompressedBuffer = bloscDecompress(
                    compressedBuffer.get(), paddedCount);

                CPPUNIT_ASSERT(uncompressedBuffer);

                for (int i = 0; i < paddedCount; i++) {
                    CPPUNIT_ASSERT_EQUAL((uncompressedBuffer.get())[i], newTest[i]);
                }
            }
#endif
        }
    }

    { // invalid buffer (out of range)

        // compress

        std::vector<int> smallBuffer;
        smallBuffer.reserve(count);

        for (int i = 0; i < count; i++)     smallBuffer[i] = i;

        size_t invalidBytes = INT_MAX - 1;

        size_t testCompressedBytes = bloscCompressedSize(
            reinterpret_cast<char*>(&smallBuffer[0]), invalidBytes);

        CPPUNIT_ASSERT_EQUAL(testCompressedBytes, size_t(0));

        std::unique_ptr<char[]> buffer = bloscCompress(
            reinterpret_cast<char*>(&smallBuffer[0]), invalidBytes, testCompressedBytes);

        CPPUNIT_ASSERT(!buffer);
        CPPUNIT_ASSERT_EQUAL(testCompressedBytes, size_t(0));

        // decompress

#ifdef OPENVDB_USE_BLOSC
        std::unique_ptr<char[]> compressedBuffer = bloscCompress(
            reinterpret_cast<char*>(&smallBuffer[0]), count * sizeof(int), testCompressedBytes);

        CPPUNIT_ASSERT_THROW(buffer = bloscDecompress(
            reinterpret_cast<char*>(compressedBuffer.get()), invalidBytes - 16),
            openvdb::RuntimeError);

        CPPUNIT_ASSERT(!buffer);

        CPPUNIT_ASSERT_THROW(bloscDecompress(
            reinterpret_cast<char*>(compressedBuffer.get()), count * sizeof(int) + 1),
            openvdb::RuntimeError);
#endif
    }

    { // uncompressible buffer
        const int uncompressedCount = 32;

        std::vector<int> values;
        values.reserve(uncompressedCount); // 128 bytes

        for (int i = 0; i < uncompressedCount; i++)     values.push_back(i*10000);

        std::random_shuffle(values.begin(), values.end());

        std::unique_ptr<int[]> uncompressedBuffer(new int[values.size()]);

        for (size_t i = 0; i < values.size(); i++)     uncompressedBuffer.get()[i] = values[i];

        size_t uncompressedBytes = values.size() * sizeof(int);
        size_t compressedBytes;

        std::unique_ptr<char[]> compressedBuffer = bloscCompress(
            reinterpret_cast<char*>(uncompressedBuffer.get()), uncompressedBytes, compressedBytes);

        CPPUNIT_ASSERT(!compressedBuffer);
        CPPUNIT_ASSERT_EQUAL(compressedBytes, size_t(0));
    }
}


void
TestStreamCompression::testPagedStreams()
{
    { // one small value
        std::ostringstream ostr(std::ios_base::binary);
        PagedOutputStream ostream(ostr);

        int foo = 5;
        ostream.write(reinterpret_cast<const char*>(&foo), sizeof(int));
        CPPUNIT_ASSERT_EQUAL(ostr.tellp(), std::streampos(0));

        ostream.flush();
        CPPUNIT_ASSERT_EQUAL(ostr.tellp(), std::streampos(sizeof(int)));
    }

    { // small values up to page threshold
        std::ostringstream ostr(std::ios_base::binary);
        PagedOutputStream ostream(ostr);

        for (int i = 0; i < PageSize; i++) {
            uint8_t oneByte = 255;
            ostream.write(reinterpret_cast<const char*>(&oneByte), sizeof(uint8_t));
        }
        CPPUNIT_ASSERT_EQUAL(ostr.tellp(), std::streampos(0));

        std::vector<uint8_t> values;
        values.assign(PageSize, uint8_t(255));
        size_t compressedSize = compression::bloscCompressedSize(
            reinterpret_cast<const char*>(&values[0]), PageSize);

        uint8_t oneMoreByte(255);
        ostream.write(reinterpret_cast<const char*>(&oneMoreByte), sizeof(char));

        if (compressedSize == 0) {
            CPPUNIT_ASSERT_EQUAL(ostr.tellp(), std::streampos(PageSize));
        }
        else {
            CPPUNIT_ASSERT_EQUAL(ostr.tellp(), std::streampos(compressedSize));
        }
    }

    { // one large block at exactly page threshold
        std::ostringstream ostr(std::ios_base::binary);
        PagedOutputStream ostream(ostr);

        std::vector<uint8_t> values;
        values.assign(PageSize, uint8_t(255));
        ostream.write(reinterpret_cast<const char*>(&values[0]), values.size());

        CPPUNIT_ASSERT_EQUAL(ostr.tellp(), std::streampos(0));
    }

    { // two large blocks at page threshold + 1 byte
        std::ostringstream ostr(std::ios_base::binary);
        PagedOutputStream ostream(ostr);

        std::vector<uint8_t> values;
        values.assign(PageSize + 1, uint8_t(255));
        ostream.write(reinterpret_cast<const char*>(&values[0]), values.size());

        size_t compressedSize = compression::bloscCompressedSize(
            reinterpret_cast<const char*>(&values[0]), values.size());

#ifndef OPENVDB_USE_BLOSC
        compressedSize = values.size();
#endif

        CPPUNIT_ASSERT_EQUAL(ostr.tellp(), std::streampos(compressedSize));

        ostream.write(reinterpret_cast<const char*>(&values[0]), values.size());

        CPPUNIT_ASSERT_EQUAL(ostr.tellp(), std::streampos(compressedSize * 2));

        uint8_t oneMoreByte(255);
        ostream.write(reinterpret_cast<const char*>(&oneMoreByte), sizeof(uint8_t));

        ostream.flush();

        CPPUNIT_ASSERT_EQUAL(ostr.tellp(), std::streampos(compressedSize * 2 + 1));
    }

    { // one full page
        std::stringstream ss(std::ios_base::out | std::ios_base::in | std::ios_base::binary);

        // write

        PagedOutputStream ostreamSizeOnly(ss);
        ostreamSizeOnly.setSizeOnly(true);

        CPPUNIT_ASSERT_EQUAL(ss.tellp(), std::streampos(0));

        std::vector<uint8_t> values;
        values.resize(PageSize);
        std::iota(values.begin(), values.end(), 0); // ascending integer values
        ostreamSizeOnly.write(reinterpret_cast<const char*>(&values[0]), values.size());
        ostreamSizeOnly.flush();

#ifdef OPENVDB_USE_BLOSC
        // two integers - compressed size and uncompressed size
        CPPUNIT_ASSERT_EQUAL(ss.tellp(), std::streampos(sizeof(int)*2));
#else
        // one integer - uncompressed size
        CPPUNIT_ASSERT_EQUAL(ss.tellp(), std::streampos(sizeof(int)));
#endif

        PagedOutputStream ostream(ss);
        ostream.write(reinterpret_cast<const char*>(&values[0]), values.size());
        ostream.flush();

#ifdef OPENVDB_USE_BLOSC
#ifdef BLOSC_HCR_BLOCKSIZE_OPTIMIZATION
        CPPUNIT_ASSERT_EQUAL(ss.tellp(), std::streampos(4422));
#else
        CPPUNIT_ASSERT_EQUAL(ss.tellp(), std::streampos(4452));
#endif
#else
        CPPUNIT_ASSERT_EQUAL(ss.tellp(), std::streampos(PageSize+sizeof(int)));
#endif

        // read

        CPPUNIT_ASSERT_EQUAL(ss.tellg(), std::streampos(0));

        PagedInputStream istream(ss);
        istream.setSizeOnly(true);

        PageHandle::Ptr handle = istream.createHandle(values.size());

#ifdef OPENVDB_USE_BLOSC
        // two integers - compressed size and uncompressed size
        CPPUNIT_ASSERT_EQUAL(ss.tellg(), std::streampos(sizeof(int)*2));
#else
        // one integer - uncompressed size
        CPPUNIT_ASSERT_EQUAL(ss.tellg(), std::streampos(sizeof(int)));
#endif

        istream.read(handle, values.size(), false);

#ifdef OPENVDB_USE_BLOSC
#ifdef BLOSC_HCR_BLOCKSIZE_OPTIMIZATION
        CPPUNIT_ASSERT_EQUAL(ss.tellg(), std::streampos(4422));
#else
        CPPUNIT_ASSERT_EQUAL(ss.tellg(), std::streampos(4452));
#endif
#else
        CPPUNIT_ASSERT_EQUAL(ss.tellg(), std::streampos(PageSize+sizeof(int)));
#endif

        std::unique_ptr<uint8_t[]> newValues(reinterpret_cast<uint8_t*>(handle->read().release()));

        CPPUNIT_ASSERT(newValues);

        for (size_t i = 0; i < values.size(); i++) {
            CPPUNIT_ASSERT_EQUAL(values[i], newValues.get()[i]);
        }
    }

    std::string tempDir;
    if (const char* dir = std::getenv("TMPDIR")) tempDir = dir;
#ifdef _MSC_VER
    if (tempDir.empty()) {
        char tempDirBuffer[MAX_PATH+1];
        int tempDirLen = GetTempPath(MAX_PATH+1, tempDirBuffer);
        CPPUNIT_ASSERT(tempDirLen > 0 && tempDirLen <= MAX_PATH);
        tempDir = tempDirBuffer;
    }
#else
    if (tempDir.empty()) tempDir = P_tmpdir;
#endif

    {
        std::string filename = tempDir + "/openvdb_page1";
        io::StreamMetadata::Ptr streamMetadata(new io::StreamMetadata);

        { // ascending values up to 10 million written in blocks of PageSize/3
            std::ofstream fileout(filename.c_str(), std::ios_base::binary);

            io::setStreamMetadataPtr(fileout, streamMetadata);
            io::setDataCompression(fileout, openvdb::io::COMPRESS_BLOSC);

            std::vector<uint8_t> values;
            values.resize(10*1000*1000);
            std::iota(values.begin(), values.end(), 0); // ascending integer values

            // write page sizes

            PagedOutputStream ostreamSizeOnly(fileout);
            ostreamSizeOnly.setSizeOnly(true);

            CPPUNIT_ASSERT_EQUAL(fileout.tellp(), std::streampos(0));

            int increment = PageSize/3;

            for (size_t i = 0; i < values.size(); i += increment) {
                if (size_t(i+increment) > values.size()) {
                    ostreamSizeOnly.write(
                        reinterpret_cast<const char*>(&values[0]+i), values.size() - i);
                }
                else {
                    ostreamSizeOnly.write(reinterpret_cast<const char*>(&values[0]+i), increment);
                }
            }
            ostreamSizeOnly.flush();

#ifdef OPENVDB_USE_BLOSC
            int pages = static_cast<int>(fileout.tellp() / (sizeof(int)*2));
#else
            int pages = static_cast<int>(fileout.tellp() / (sizeof(int)));
#endif

            CPPUNIT_ASSERT_EQUAL(pages, 10);

            // write

            PagedOutputStream ostream(fileout);

            for (size_t i = 0; i < values.size(); i += increment) {
                if (size_t(i+increment) > values.size()) {
                    ostream.write(reinterpret_cast<const char*>(&values[0]+i), values.size() - i);
                }
                else {
                    ostream.write(reinterpret_cast<const char*>(&values[0]+i), increment);
                }
            }

            ostream.flush();

#ifdef OPENVDB_USE_BLOSC
#ifdef BLOSC_HCR_BLOCKSIZE_OPTIMIZATION
            CPPUNIT_ASSERT_EQUAL(fileout.tellp(), std::streampos(42424));
#else
            CPPUNIT_ASSERT_EQUAL(fileout.tellp(), std::streampos(42724));
#endif
#else
            CPPUNIT_ASSERT_EQUAL(fileout.tellp(), std::streampos(values.size()+sizeof(int)*pages));
#endif

            // abuse File being a friend of MappedFile to get around the private constructor
            ProxyMappedFile* proxy = new ProxyMappedFile(filename);
            SharedPtr<io::MappedFile> mappedFile(reinterpret_cast<io::MappedFile*>(proxy));

            // read

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            CPPUNIT_ASSERT_EQUAL(filein.tellg(), std::streampos(0));

            PagedInputStream istreamSizeOnly(filein);
            istreamSizeOnly.setSizeOnly(true);

            std::vector<PageHandle::Ptr> handles;

            for (size_t i = 0; i < values.size(); i += increment) {
                if (size_t(i+increment) > values.size()) {
                    handles.push_back(istreamSizeOnly.createHandle(values.size() - i));
                }
                else {
                    handles.push_back(istreamSizeOnly.createHandle(increment));
                }
            }

#ifdef OPENVDB_USE_BLOSC
            // two integers - compressed size and uncompressed size
            CPPUNIT_ASSERT_EQUAL(filein.tellg(), std::streampos(pages*sizeof(int)*2));
#else
            // one integer - uncompressed size
            CPPUNIT_ASSERT_EQUAL(filein.tellg(), std::streampos(pages*sizeof(int)));
#endif

            PagedInputStream istream(filein);

            int pageHandle = 0;

            for (size_t i = 0; i < values.size(); i += increment) {
                if (size_t(i+increment) > values.size()) {
                    istream.read(handles[pageHandle++], values.size() - i);
                }
                else {
                    istream.read(handles[pageHandle++], increment);
                }
            }

            // first three handles live in the same page

            Page& page0 = handles[0]->page();
            Page& page1 = handles[1]->page();
            Page& page2 = handles[2]->page();
            Page& page3 = handles[3]->page();

            CPPUNIT_ASSERT(page0.isOutOfCore());
            CPPUNIT_ASSERT(page1.isOutOfCore());
            CPPUNIT_ASSERT(page2.isOutOfCore());
            CPPUNIT_ASSERT(page3.isOutOfCore());

            handles[0]->read();

            // store the Page shared_ptr

            Page::Ptr page = handles[0]->mPage;

            // verify use count is four (one plus three handles)

            CPPUNIT_ASSERT_EQUAL(page.use_count(), long(4));

            // on reading from the first handle, all pages referenced
            // in the first three handles are in-core

            CPPUNIT_ASSERT(!page0.isOutOfCore());
            CPPUNIT_ASSERT(!page1.isOutOfCore());
            CPPUNIT_ASSERT(!page2.isOutOfCore());
            CPPUNIT_ASSERT(page3.isOutOfCore());

            handles[1]->read();

            CPPUNIT_ASSERT(handles[0]->mPage);

            handles[2]->read();

            handles.erase(handles.begin());
            handles.erase(handles.begin());
            handles.erase(handles.begin());

            // after all three handles have been read,
            // page should have just one use count (itself)

            CPPUNIT_ASSERT_EQUAL(page.use_count(), long(1));
        }
        std::remove(filename.c_str());
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
