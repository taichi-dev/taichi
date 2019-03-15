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

#ifndef OPENVDB_IO_IO_HAS_BEEN_INCLUDED
#define OPENVDB_IO_IO_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include <openvdb/Types.h> // for SharedPtr
#include <openvdb/version.h>
#include <boost/any.hpp>
#include <functional>
#include <iosfwd> // for std::ios_base
#include <map>
#include <memory>
#include <string>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

class MetaMap;

namespace io {

/// @brief Container for metadata describing how to unserialize grids from and/or
/// serialize grids to a stream (which file format, compression scheme, etc. to use)
/// @details This class is mainly for internal use.
class OPENVDB_API StreamMetadata
{
public:
    using Ptr = SharedPtr<StreamMetadata>;
    using ConstPtr = SharedPtr<const StreamMetadata>;

    StreamMetadata();
    StreamMetadata(const StreamMetadata&);
    explicit StreamMetadata(std::ios_base&);
    ~StreamMetadata();

    StreamMetadata& operator=(const StreamMetadata&);

    /// @brief Transfer metadata items directly to the given stream.
    /// @todo Deprecate direct transfer; use StreamMetadata structs everywhere.
    void transferTo(std::ios_base&) const;

    uint32_t fileVersion() const;
    void setFileVersion(uint32_t);

    VersionId libraryVersion() const;
    void setLibraryVersion(VersionId);

    uint32_t compression() const;
    void setCompression(uint32_t);

    uint32_t gridClass() const;
    void setGridClass(uint32_t);

    const void* backgroundPtr() const;
    void setBackgroundPtr(const void*);

    bool halfFloat() const;
    void setHalfFloat(bool);

    bool writeGridStats() const;
    void setWriteGridStats(bool);

    bool seekable() const;
    void setSeekable(bool);

    bool countingPasses() const;
    void setCountingPasses(bool);

    uint32_t pass() const;
    void setPass(uint32_t);

    //@{
    /// @brief Return a (reference to a) copy of the metadata of the grid
    /// currently being read or written.
    /// @details Some grid metadata might duplicate information returned by
    /// gridClass(), backgroundPtr() and other accessors, but those values
    /// are not guaranteed to be kept in sync.
    MetaMap& gridMetadata();
    const MetaMap& gridMetadata() const;
    //@}

    using AuxDataMap = std::map<std::string, boost::any>;
    //@{
    /// @brief Return a map that can be populated with arbitrary user data.
    AuxDataMap& auxData();
    const AuxDataMap& auxData() const;
    //@}

    /// Return a string describing this stream metadata.
    std::string str() const;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
}; // class StreamMetadata


/// Write a description of the given metadata to an output stream.
std::ostream& operator<<(std::ostream&, const StreamMetadata&);

std::ostream& operator<<(std::ostream&, const StreamMetadata::AuxDataMap&);


////////////////////////////////////////


/// @brief Leaf nodes that require multi-pass I/O must inherit from this struct.
/// @sa Grid::hasMultiPassIO()
struct MultiPass {};


////////////////////////////////////////


class File;

/// @brief Handle to control the lifetime of a memory-mapped .vdb file
class OPENVDB_API MappedFile
{
public:
    using Ptr = SharedPtr<MappedFile>;

    ~MappedFile();
    MappedFile(const MappedFile&) = delete; // not copyable
    MappedFile& operator=(const MappedFile&) = delete;

    /// Return the filename of the mapped file.
    std::string filename() const;

    /// @brief Return a new stream buffer for the mapped file.
    /// @details Typical usage is
    /// @code
    /// openvdb::io::MappedFile::Ptr mappedFile = ...;
    /// auto buf = mappedFile->createBuffer();
    /// std::istream istrm{buf.get()};
    /// // Read from istrm...
    /// @endcode
    /// The buffer must persist as long as the stream is open.
    SharedPtr<std::streambuf> createBuffer() const;

    using Notifier = std::function<void(std::string /*filename*/)>;
    /// @brief Register a function that will be called with this file's name
    /// when the file is unmapped.
    void setNotifier(const Notifier&);
    /// Deregister the notifier.
    void clearNotifier();

private:
    friend class File;

    explicit MappedFile(const std::string& filename, bool autoDelete = false);

    class Impl;
    std::unique_ptr<Impl> mImpl;
}; // class MappedFile


////////////////////////////////////////


/// Return a string (possibly empty) describing the given system error code.
std::string getErrorString(int errorNum);


/// Return a string (possibly empty) describing the most recent system error.
std::string getErrorString();


////////////////////////////////////////


/// @brief Return the file format version number associated with the given input stream.
/// @sa File::setFormatVersion()
OPENVDB_API uint32_t getFormatVersion(std::ios_base&);

/// @brief Return the (major, minor) library version number associated with the given input stream.
/// @sa File::setLibraryVersion()
OPENVDB_API VersionId getLibraryVersion(std::ios_base&);

/// @brief Return a string of the form "<major>.<minor>/<format>", giving the library
/// and file format version numbers associated with the given input stream.
OPENVDB_API std::string getVersion(std::ios_base&);

/// Associate the current file format and library version numbers with the given input stream.
OPENVDB_API void setCurrentVersion(std::istream&);

/// @brief Associate specific file format and library version numbers with the given stream.
/// @details This is typically called immediately after reading a header that contains
/// the version numbers.  Data read subsequently can then be interpreted appropriately.
OPENVDB_API void setVersion(std::ios_base&, const VersionId& libraryVersion, uint32_t fileVersion);

/// @brief Return a bitwise OR of compression option flags (COMPRESS_ZIP,
/// COMPRESS_ACTIVE_MASK, etc.) specifying whether and how input data is compressed
/// or output data should be compressed.
OPENVDB_API uint32_t getDataCompression(std::ios_base&);
/// @brief Associate with the given stream a bitwise OR of compression option flags
/// (COMPRESS_ZIP, COMPRESS_ACTIVE_MASK, etc.) specifying whether and how input data
/// is compressed or output data should be compressed.
OPENVDB_API void setDataCompression(std::ios_base&, uint32_t compressionFlags);

/// @brief Return the class (GRID_LEVEL_SET, GRID_UNKNOWN, etc.) of the grid
/// currently being read from or written to the given stream.
OPENVDB_API uint32_t getGridClass(std::ios_base&);
/// @brief Associate with the given stream the class (GRID_LEVEL_SET, GRID_UNKNOWN, etc.)
/// of the grid currently being read or written.
OPENVDB_API void setGridClass(std::ios_base&, uint32_t);

/// @brief Return true if floating-point values should be quantized to 16 bits when writing
/// to the given stream or promoted back from 16-bit to full precision when reading from it.
OPENVDB_API bool getHalfFloat(std::ios_base&);
/// @brief Specify whether floating-point values should be quantized to 16 bits when writing
/// to the given stream or promoted back from 16-bit to full precision when reading from it.
OPENVDB_API void setHalfFloat(std::ios_base&, bool);

/// @brief Return a pointer to the background value of the grid
/// currently being read from or written to the given stream.
OPENVDB_API const void* getGridBackgroundValuePtr(std::ios_base&);
/// @brief Specify (a pointer to) the background value of the grid
/// currently being read from or written to the given stream.
/// @note The pointer must remain valid until the entire grid has been read or written.
OPENVDB_API void setGridBackgroundValuePtr(std::ios_base&, const void* background);

/// @brief Return @c true if grid statistics (active voxel count and bounding box, etc.)
/// should be computed and stored as grid metadata when writing to the given stream.
OPENVDB_API bool getWriteGridStatsMetadata(std::ios_base&);
/// @brief Specify whether to compute grid statistics (active voxel count and bounding box, etc.)
/// and store them as grid metadata when writing to the given stream.
OPENVDB_API void setWriteGridStatsMetadata(std::ios_base&, bool writeGridStats);

/// @brief Return a shared pointer to the memory-mapped file with which the given stream
/// is associated, or a null pointer if the stream is not associated with a memory-mapped file.
OPENVDB_API SharedPtr<MappedFile> getMappedFilePtr(std::ios_base&);
/// @brief Associate the given stream with (a shared pointer to) a memory-mapped file.
/// @note The shared pointer object (not just the io::MappedFile object to which it points)
/// must remain valid until the file is closed.
OPENVDB_API void setMappedFilePtr(std::ios_base&, SharedPtr<MappedFile>&);

/// @brief Return a shared pointer to an object that stores metadata (file format,
/// compression scheme, etc.) for use when reading from or writing to the given stream.
OPENVDB_API SharedPtr<StreamMetadata> getStreamMetadataPtr(std::ios_base&);
/// @brief Associate the given stream with (a shared pointer to) an object that stores
/// metadata (file format, compression scheme, etc.) for use when reading from
/// or writing to the stream.
/// @details If @a transfer is true, copy metadata from the object directly to the stream
/// (for backward compatibility with older versions of the library).
/// @note The shared pointer object (not just the io::StreamMetadata object to which it points)
/// must remain valid until the file is closed.
OPENVDB_API void setStreamMetadataPtr(std::ios_base&,
    SharedPtr<StreamMetadata>&, bool transfer = true);
/// @brief Dissociate the given stream from its metadata object (if it has one)
/// and return a shared pointer to the object.
OPENVDB_API SharedPtr<StreamMetadata> clearStreamMetadataPtr(std::ios_base&);

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_IO_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
