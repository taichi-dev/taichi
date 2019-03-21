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

#ifndef OPENVDB_IO_ARCHIVE_HAS_BEEN_INCLUDED
#define OPENVDB_IO_ARCHIVE_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include "Compression.h" // for COMPRESS_ZIP, etc.
#include <openvdb/Grid.h>
#include <openvdb/MetaMap.h>
#include <openvdb/Platform.h>
#include <openvdb/version.h> // for VersionId
#include <boost/uuid/uuid.hpp>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>


class TestFile;

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

class GridDescriptor;


/// Grid serializer/unserializer
class OPENVDB_API Archive
{
public:
    using Ptr = SharedPtr<Archive>;
    using ConstPtr = SharedPtr<const Archive>;

    static const uint32_t DEFAULT_COMPRESSION_FLAGS;

    Archive();
    Archive(const Archive&) = default;
    Archive& operator=(const Archive&) = default;
    virtual ~Archive();

    /// @brief Return a copy of this archive.
    virtual Ptr copy() const;

    /// @brief Return the UUID that was most recently written (or read,
    /// if no UUID has been written yet).
    std::string getUniqueTag() const;
    /// @brief Return @c true if the given UUID matches this archive's UUID.
    bool isIdentical(const std::string& uuidStr) const;

    /// @brief Return the file format version number of the input stream.
    uint32_t fileVersion() const { return mFileVersion; }
    /// @brief Return the (major, minor) version number of the library that was
    /// used to write the input stream.
    VersionId libraryVersion() const { return mLibraryVersion; }
    /// @brief Return a string of the form "<major>.<minor>/<format>", giving the
    /// library and file format version numbers associated with the input stream.
    std::string version() const;

    /// @brief Return @c true if trees shared by multiple grids are written out
    /// only once, @c false if they are written out once per grid.
    bool isInstancingEnabled() const { return mEnableInstancing; }
    /// @brief Specify whether trees shared by multiple grids should be
    /// written out only once (@c true) or once per grid (@c false).
    /// @note Instancing is enabled by default.
    void setInstancingEnabled(bool b) { mEnableInstancing = b; }

    /// Return @c true if the OpenVDB library includes support for the Blosc compressor.
    static bool hasBloscCompression();

    /// Return a bit mask specifying compression options for the data stream.
    uint32_t compression() const { return mCompression; }
    /// @brief Specify whether and how the data stream should be compressed.
    /// @param c bitwise OR (e.g., COMPRESS_ZIP | COMPRESS_ACTIVE_MASK) of
    ///     compression option flags (see Compression.h for the available flags)
    /// @note Not all combinations of compression options are supported.
    void setCompression(uint32_t c) { mCompression = c; }

    /// @brief Return @c true if grid statistics (active voxel count and
    /// bounding box, etc.) are computed and written as grid metadata.
    bool isGridStatsMetadataEnabled() const { return mEnableGridStats; }
    /// @brief Specify whether grid statistics (active voxel count and
    /// bounding box, etc.) should be computed and written as grid metadata.
    void setGridStatsMetadataEnabled(bool b) { mEnableGridStats = b; }

    /// @brief Write the grids in the given container to this archive's output stream.
    virtual void write(const GridCPtrVec&, const MetaMap& = MetaMap()) const {}

    /// @brief Return @c true if delayed loading is enabled.
    /// @details If enabled, delayed loading can be disabled for individual files,
    /// but not vice-versa.
    /// @note Define the environment variable @c OPENVDB_DISABLE_DELAYED_LOAD
    /// to disable delayed loading unconditionally.
    static bool isDelayedLoadingEnabled();

protected:
    /// @brief Return @c true if the input stream contains grid offsets
    /// that allow for random access or partial reading.
    bool inputHasGridOffsets() const { return mInputHasGridOffsets; }
    void setInputHasGridOffsets(bool b) { mInputHasGridOffsets = b; }

    /// @brief Tag the given input stream with the input file format version number.
    ///
    /// The tag can be retrieved with getFormatVersion().
    /// @sa getFormatVersion()
    void setFormatVersion(std::istream&);

    /// @brief Tag the given input stream with the version number of
    /// the library with which the input stream was created.
    ///
    /// The tag can be retrieved with getLibraryVersion().
    /// @sa getLibraryVersion()
    void setLibraryVersion(std::istream&);

    /// @brief Tag the given input stream with flags indicating whether
    /// the input stream contains compressed data and how it is compressed.
    void setDataCompression(std::istream&);

    /// @brief Tag an output stream with flags specifying only those
    /// compression options that are applicable to the given grid.
    void setGridCompression(std::ostream&, const GridBase&) const;
    /// @brief Read in the compression flags for a grid and
    /// tag the given input stream with those flags.
    static void readGridCompression(std::istream&);

    /// Read in and return the number of grids on the input stream.
    static int32_t readGridCount(std::istream&);

    /// Populate the given grid from the input stream.
    static void readGrid(GridBase::Ptr, const GridDescriptor&, std::istream&);
#if OPENVDB_ABI_VERSION_NUMBER >= 3
    /// @brief Populate the given grid from the input stream, but only where it
    /// intersects the given world-space bounding box.
    static void readGrid(GridBase::Ptr, const GridDescriptor&, std::istream&, const BBoxd&);
    /// @brief Populate the given grid from the input stream, but only where it
    /// intersects the given index-space bounding box.
    static void readGrid(GridBase::Ptr, const GridDescriptor&, std::istream&, const CoordBBox&);
#endif

    using NamedGridMap = std::map<Name /*uniqueName*/, GridBase::Ptr>;

    /// @brief If the grid represented by the given grid descriptor
    /// is an instance, connect it with its instance parent.
    void connectInstance(const GridDescriptor&, const NamedGridMap&) const;

    /// Write the given grid descriptor and grid to an output stream
    /// and update the GridDescriptor offsets.
    /// @param seekable  if true, the output stream supports seek operations
    void writeGrid(GridDescriptor&, GridBase::ConstPtr, std::ostream&, bool seekable) const;
    /// Write the given grid descriptor and grid metadata to an output stream
    /// and update the GridDescriptor offsets, but don't write the grid's tree,
    /// since it is shared with another grid.
    /// @param seekable  if true, the output stream supports seek operations
    void writeGridInstance(GridDescriptor&, GridBase::ConstPtr,
        std::ostream&, bool seekable) const;

    /// @brief Read the magic number, version numbers, UUID, etc. from the given input stream.
    /// @return @c true if the input UUID differs from the previously-read UUID.
    bool readHeader(std::istream&);
    /// @brief Write the magic number, version numbers, UUID, etc. to the given output stream.
    /// @param seekable  if true, the output stream supports seek operations
    /// @todo This method should not be const since it actually redefines the UUID!
    void writeHeader(std::ostream&, bool seekable) const;

    //@{
    /// Write the given grids to an output stream.
    void write(std::ostream&, const GridPtrVec&, bool seekable, const MetaMap& = MetaMap()) const;
    void write(std::ostream&, const GridCPtrVec&, bool seekable, const MetaMap& = MetaMap()) const;
    //@}

private:
    friend class ::TestFile;

    /// The version of the file that was read
    uint32_t mFileVersion;
    /// The version of the library that was used to create the file that was read
    VersionId mLibraryVersion;
    /// 16-byte (128-bit) UUID
    mutable boost::uuids::uuid mUuid;// needs to be mutable since writeHeader is const!
    /// Flag indicating whether the input stream contains grid offsets
    /// and therefore supports partial reading
    bool mInputHasGridOffsets;
    /// Flag indicating whether a tree shared by multiple grids should be
    /// written out only once (true) or once per grid (false)
    bool mEnableInstancing;
    /// Flags indicating whether and how the data stream is compressed
    uint32_t mCompression;
    /// Flag indicating whether grid statistics metadata should be written
    bool mEnableGridStats;
}; // class Archive

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_ARCHIVE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
