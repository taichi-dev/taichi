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

/// @file io/File.cc

#include "File.h"

#include "TempFile.h"
#include <openvdb/Exceptions.h>
#include <openvdb/util/logging.h>
#include <cstdint>
#include <boost/iostreams/copy.hpp>
#ifndef _MSC_VER
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <cassert>
#include <cstdlib> // for getenv(), strtoul()
#include <cstring> // for strerror_r()
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

// Implementation details of the File class
struct File::Impl
{
    enum { DEFAULT_COPY_MAX_BYTES = 500000000 }; // 500 MB

    struct NoBBox {};

    // Common implementation of the various File::readGrid() overloads,
    // with and without bounding box clipping
    template<typename BoxType>
    static GridBase::Ptr readGrid(const File& file, const GridDescriptor& gd, const BoxType& bbox)
    {
        // This method should not be called for files that don't contain grid offsets.
        assert(file.inputHasGridOffsets());

        GridBase::Ptr grid = file.createGrid(gd);
        gd.seekToGrid(file.inputStream());
        unarchive(file, grid, gd, bbox);
        return grid;
    }

    static void unarchive(const File& file, GridBase::Ptr& grid,
        const GridDescriptor& gd, NoBBox)
    {
        file.Archive::readGrid(grid, gd, file.inputStream());
    }

#if OPENVDB_ABI_VERSION_NUMBER >= 3
    static void unarchive(const File& file, GridBase::Ptr& grid,
        const GridDescriptor& gd, const CoordBBox& indexBBox)
    {
        file.Archive::readGrid(grid, gd, file.inputStream(), indexBBox);
    }

    static void unarchive(const File& file, GridBase::Ptr& grid,
        const GridDescriptor& gd, const BBoxd& worldBBox)
    {
        file.Archive::readGrid(grid, gd, file.inputStream(), worldBBox);
    }
#endif

    static Index64 getDefaultCopyMaxBytes()
    {
        Index64 result = DEFAULT_COPY_MAX_BYTES;
        if (const char* s = std::getenv("OPENVDB_DELAYED_LOAD_COPY_MAX_BYTES")) {
            char* endptr = nullptr;
            result = std::strtoul(s, &endptr, /*base=*/10);
        }
        return result;
    }

    std::string mFilename;
    // The file-level metadata
    MetaMap::Ptr mMeta;
    // The memory-mapped file
    MappedFile::Ptr mFileMapping;
    // The buffer for the input stream, if it is a memory-mapped file
    SharedPtr<std::streambuf> mStreamBuf;
    // The file stream that is open for reading
    std::unique_ptr<std::istream> mInStream;
    // File-level stream metadata (file format, compression, etc.)
    StreamMetadata::Ptr mStreamMetadata;
    // Flag indicating if we have read in the global information (header,
    // metadata, and grid descriptors) for this VDB file
    bool mIsOpen;
    // File size limit for copying during delayed loading
    Index64 mCopyMaxBytes;
    // Grid descriptors for all grids stored in the file, indexed by grid name
    NameMap mGridDescriptors;
    // All grids, indexed by unique name (used only when mHasGridOffsets is false)
    Archive::NamedGridMap mNamedGrids;
    // All grids stored in the file (used only when mHasGridOffsets is false)
    GridPtrVecPtr mGrids;
}; // class File::Impl


////////////////////////////////////////


File::File(const std::string& filename): mImpl(new Impl)
{
    mImpl->mFilename = filename;
    mImpl->mIsOpen = false;
    mImpl->mCopyMaxBytes = Impl::getDefaultCopyMaxBytes();
    setInputHasGridOffsets(true);
}


File::~File()
{
}


File::File(const File& other)
    : Archive(other)
    , mImpl(new Impl)
{
    *this = other;
}


File&
File::operator=(const File& other)
{
    if (&other != this) {
        Archive::operator=(other);
        const Impl& otherImpl = *other.mImpl;
        mImpl->mFilename = otherImpl.mFilename;
        mImpl->mMeta = otherImpl.mMeta;
        mImpl->mIsOpen = false; // don't want two file objects reading from the same stream
        mImpl->mCopyMaxBytes = otherImpl.mCopyMaxBytes;
        mImpl->mGridDescriptors = otherImpl.mGridDescriptors;
        mImpl->mNamedGrids = otherImpl.mNamedGrids;
        mImpl->mGrids = otherImpl.mGrids;
    }
    return *this;
}


SharedPtr<Archive>
File::copy() const
{
    return SharedPtr<Archive>{new File{*this}};
}


////////////////////////////////////////


const std::string&
File::filename() const
{
    return mImpl->mFilename;
}


MetaMap::Ptr
File::fileMetadata()
{
    return mImpl->mMeta;
}

MetaMap::ConstPtr
File::fileMetadata() const
{
    return mImpl->mMeta;
}


const File::NameMap&
File::gridDescriptors() const
{
    return mImpl->mGridDescriptors;
}

File::NameMap&
File::gridDescriptors()
{
    return mImpl->mGridDescriptors;
}


std::istream&
File::inputStream() const
{
    if (!mImpl->mInStream) {
        OPENVDB_THROW(IoError, filename() << " is not open for reading");
    }
    return *mImpl->mInStream;
}


////////////////////////////////////////


Index64
File::getSize() const
{
    /// @internal boost::filesystem::file_size() would be a more portable alternative,
    /// but as of 9/2014, Houdini ships without the Boost.Filesystem library,
    /// which makes it much less convenient to use that library.

    Index64 result = std::numeric_limits<Index64>::max();

    std::string mesg = "could not get size of file " + filename();

#ifdef _MSC_VER
    // Get the file size by seeking to the end of the file.
    std::ifstream fstrm(filename());
    if (fstrm) {
        fstrm.seekg(0, fstrm.end);
        result = static_cast<Index64>(fstrm.tellg());
    } else {
        OPENVDB_THROW(IoError, mesg);
    }
#else
    // Get the file size using the stat() system call.
    struct stat info;
    if (0 != ::stat(filename().c_str(), &info)) {
        std::string s = getErrorString();
        if (!s.empty()) mesg += " (" + s + ")";
        OPENVDB_THROW(IoError, mesg);
    }
    if (!S_ISREG(info.st_mode)) {
        mesg += " (not a regular file)";
        OPENVDB_THROW(IoError, mesg);
    }
    result = static_cast<Index64>(info.st_size);
#endif

    return result;
}


Index64
File::copyMaxBytes() const
{
    return mImpl->mCopyMaxBytes;
}


void
File::setCopyMaxBytes(Index64 bytes)
{
    mImpl->mCopyMaxBytes = bytes;
}


////////////////////////////////////////


bool
File::isOpen() const
{
    return mImpl->mIsOpen;
}


bool
File::open(bool delayLoad, const MappedFile::Notifier& notifier)
{
    if (isOpen()) {
        OPENVDB_THROW(IoError, filename() << " is already open");
    }
    mImpl->mInStream.reset();

    // Open the file.
    std::unique_ptr<std::istream> newStream;
    SharedPtr<std::streambuf> newStreamBuf;
    MappedFile::Ptr newFileMapping;
    if (!delayLoad || !Archive::isDelayedLoadingEnabled()) {
        newStream.reset(new std::ifstream(
            filename().c_str(), std::ios_base::in | std::ios_base::binary));
    } else {
        bool isTempFile = false;
        std::string fname = filename();
        if (getSize() < copyMaxBytes()) {
            // If the file is not too large, make a temporary private copy of it
            // and open the copy instead.  The original file can then be modified
            // or removed without affecting delayed load.
            try {
                TempFile tempFile;
                std::ifstream fstrm(filename().c_str(),
                    std::ios_base::in | std::ios_base::binary);
                boost::iostreams::copy(fstrm, tempFile);
                fname = tempFile.filename();
                isTempFile = true;
            } catch (std::exception& e) {
                std::string mesg;
                if (e.what()) mesg = std::string(" (") + e.what() + ")";
                OPENVDB_LOG_WARN("failed to create a temporary copy of " << filename()
                    << " for delayed loading" << mesg
                    << "; will read directly from " << filename() << " instead");
            }
        }

        // While the file is open, its mapping, stream buffer and stream
        // must all be maintained.  Once the file is closed, the buffer and
        // the stream can be discarded, but the mapping needs to persist
        // if any grids were lazily loaded.
        try {
            newFileMapping.reset(new MappedFile(fname, /*autoDelete=*/isTempFile));
            newStreamBuf = newFileMapping->createBuffer();
            newStream.reset(new std::istream(newStreamBuf.get()));
        } catch (std::exception& e) {
            std::ostringstream ostr;
            ostr << "could not open file " << filename();
            if (e.what() != nullptr) ostr << " (" << e.what() << ")";
            OPENVDB_THROW(IoError, ostr.str());
        }
    }

    if (newStream->fail()) {
        OPENVDB_THROW(IoError, "could not open file " << filename());
    }

    // Read in the file header.
    bool newFile = false;
    try {
        newFile = Archive::readHeader(*newStream);
    } catch (IoError& e) {
        if (e.what() && std::string("not a VDB file") == e.what()) {
            // Rethrow, adding the filename.
            OPENVDB_THROW(IoError, filename() << " is not a VDB file");
        }
        throw;
    }

    mImpl->mFileMapping = newFileMapping;
    if (mImpl->mFileMapping) mImpl->mFileMapping->setNotifier(notifier);
    mImpl->mStreamBuf = newStreamBuf;
    mImpl->mInStream.swap(newStream);

    // Tag the input stream with the file format and library version numbers
    // and other metadata.
    mImpl->mStreamMetadata.reset(new StreamMetadata);
    mImpl->mStreamMetadata->setSeekable(true);
    io::setStreamMetadataPtr(inputStream(), mImpl->mStreamMetadata, /*transfer=*/false);
    Archive::setFormatVersion(inputStream());
    Archive::setLibraryVersion(inputStream());
    Archive::setDataCompression(inputStream());
    io::setMappedFilePtr(inputStream(), mImpl->mFileMapping);

    // Read in the VDB metadata.
    mImpl->mMeta = MetaMap::Ptr(new MetaMap);
    mImpl->mMeta->readMeta(inputStream());

    if (!inputHasGridOffsets()) {
        OPENVDB_LOG_DEBUG_RUNTIME("file " << filename() << " does not support partial reading");

        mImpl->mGrids.reset(new GridPtrVec);
        mImpl->mNamedGrids.clear();

        // Stream in the entire contents of the file and append all grids to mGrids.
        const int32_t gridCount = readGridCount(inputStream());
        for (int32_t i = 0; i < gridCount; ++i) {
            GridDescriptor gd;
            gd.read(inputStream());

            GridBase::Ptr grid = createGrid(gd);
            Archive::readGrid(grid, gd, inputStream());

            gridDescriptors().insert(std::make_pair(gd.gridName(), gd));
            mImpl->mGrids->push_back(grid);
            mImpl->mNamedGrids[gd.uniqueName()] = grid;
        }
        // Connect instances (grids that share trees with other grids).
        for (NameMapCIter it = gridDescriptors().begin(); it != gridDescriptors().end(); ++it) {
            Archive::connectInstance(it->second, mImpl->mNamedGrids);
        }
    } else {
        // Read in just the grid descriptors.
        readGridDescriptors(inputStream());
    }

    mImpl->mIsOpen = true;
    return newFile; // true if file is not identical to opened file
}


void
File::close()
{
    // Reset all data.
    mImpl->mMeta.reset();
    mImpl->mGridDescriptors.clear();
    mImpl->mGrids.reset();
    mImpl->mNamedGrids.clear();
    mImpl->mInStream.reset();
    mImpl->mStreamBuf.reset();
    mImpl->mStreamMetadata.reset();
    mImpl->mFileMapping.reset();

    mImpl->mIsOpen = false;
    setInputHasGridOffsets(true);
}


////////////////////////////////////////


bool
File::hasGrid(const Name& name) const
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, filename() << " is not open for reading");
    }
    return (findDescriptor(name) != gridDescriptors().end());
}


MetaMap::Ptr
File::getMetadata() const
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, filename() << " is not open for reading");
    }
    // Return a deep copy of the file-level metadata, which was read
    // when the file was opened.
    return MetaMap::Ptr(new MetaMap(*mImpl->mMeta));
}


GridPtrVecPtr
File::getGrids() const
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, filename() << " is not open for reading");
    }

    GridPtrVecPtr ret;
    if (!inputHasGridOffsets()) {
        // If the input file doesn't have grid offsets, then all of the grids
        // have already been streamed in and stored in mGrids.
        ret = mImpl->mGrids;
    } else {
        ret.reset(new GridPtrVec);

        Archive::NamedGridMap namedGrids;

        // Read all grids represented by the GridDescriptors.
        for (NameMapCIter i = gridDescriptors().begin(), e = gridDescriptors().end(); i != e; ++i) {
            const GridDescriptor& gd = i->second;
            GridBase::Ptr grid = readGrid(gd);
            ret->push_back(grid);
            namedGrids[gd.uniqueName()] = grid;
        }

        // Connect instances (grids that share trees with other grids).
        for (NameMapCIter i = gridDescriptors().begin(), e = gridDescriptors().end(); i != e; ++i) {
            Archive::connectInstance(i->second, namedGrids);
        }
    }
    return ret;
}


GridBase::Ptr
File::retrieveCachedGrid(const Name& name) const
{
    // If the file has grid offsets, grids are read on demand
    // and not cached in mNamedGrids.
    if (inputHasGridOffsets()) return GridBase::Ptr();

    // If the file does not have grid offsets, mNamedGrids should already
    // contain the entire contents of the file.

    // Search by unique name.
    Archive::NamedGridMap::const_iterator it =
        mImpl->mNamedGrids.find(GridDescriptor::stringAsUniqueName(name));
    // If not found, search by grid name.
    if (it == mImpl->mNamedGrids.end()) it = mImpl->mNamedGrids.find(name);
    if (it == mImpl->mNamedGrids.end()) {
        OPENVDB_THROW(KeyError, filename() << " has no grid named \"" << name << "\"");
    }
    return it->second;
}


////////////////////////////////////////


GridPtrVecPtr
File::readAllGridMetadata()
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, filename() << " is not open for reading");
    }

    GridPtrVecPtr ret(new GridPtrVec);

    if (!inputHasGridOffsets()) {
        // If the input file doesn't have grid offsets, then all of the grids
        // have already been streamed in and stored in mGrids.
        for (size_t i = 0, N = mImpl->mGrids->size(); i < N; ++i) {
            // Return copies of the grids, but with empty trees.
#if OPENVDB_ABI_VERSION_NUMBER <= 3
            ret->push_back((*mImpl->mGrids)[i]->copyGrid(/*treePolicy=*/CP_NEW));
#else
            ret->push_back((*mImpl->mGrids)[i]->copyGridWithNewTree());
#endif
        }
    } else {
        // Read just the metadata and transforms for all grids.
        for (NameMapCIter i = gridDescriptors().begin(), e = gridDescriptors().end(); i != e; ++i) {
            const GridDescriptor& gd = i->second;
            GridBase::ConstPtr grid = readGridPartial(gd, /*readTopology=*/false);
            // Return copies of the grids, but with empty trees.
            // (As of 0.98.0, at least, it would suffice to just const cast
            // the grid pointers returned by readGridPartial(), but shallow
            // copying the grids helps to ensure future compatibility.)
#if OPENVDB_ABI_VERSION_NUMBER <= 3
            ret->push_back(grid->copyGrid(/*treePolicy=*/CP_NEW));
#else
            ret->push_back(grid->copyGridWithNewTree());
#endif
        }
    }
    return ret;
}


GridBase::Ptr
File::readGridMetadata(const Name& name)
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, filename() << " is not open for reading.");
    }

    GridBase::ConstPtr ret;
    if (!inputHasGridOffsets()) {
        // Retrieve the grid from mGrids, which should already contain
        // the entire contents of the file.
        ret = readGrid(name);
    } else {
        NameMapCIter it = findDescriptor(name);
        if (it == gridDescriptors().end()) {
            OPENVDB_THROW(KeyError, filename() << " has no grid named \"" << name << "\"");
        }

        // Seek to and read in the grid from the file.
        const GridDescriptor& gd = it->second;
        ret = readGridPartial(gd, /*readTopology=*/false);
    }
#if OPENVDB_ABI_VERSION_NUMBER <= 3
    return ret->copyGrid(/*treePolicy=*/CP_NEW);
#else
    return ret->copyGridWithNewTree();
#endif
}


////////////////////////////////////////


GridBase::Ptr
File::readGrid(const Name& name)
{
    return readGridByName(name, BBoxd());
}


#if OPENVDB_ABI_VERSION_NUMBER >= 3
GridBase::Ptr
File::readGrid(const Name& name, const BBoxd& bbox)
{
    return readGridByName(name, bbox);
}
#endif


#if OPENVDB_ABI_VERSION_NUMBER <= 2
GridBase::Ptr
File::readGridByName(const Name& name, const BBoxd&)
#else
GridBase::Ptr
File::readGridByName(const Name& name, const BBoxd& bbox)
#endif
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, filename() << " is not open for reading.");
    }

#if OPENVDB_ABI_VERSION_NUMBER >= 3
    const bool clip = bbox.isSorted();
#endif

    // If a grid with the given name was already read and cached
    // (along with the entire contents of the file, because the file
    // doesn't support random access), retrieve and return it.
    GridBase::Ptr grid = retrieveCachedGrid(name);
    if (grid) {
#if OPENVDB_ABI_VERSION_NUMBER >= 3
        if (clip) {
            grid = grid->deepCopyGrid();
            grid->clipGrid(bbox);
        }
#endif
        return grid;
    }

    NameMapCIter it = findDescriptor(name);
    if (it == gridDescriptors().end()) {
        OPENVDB_THROW(KeyError, filename() << " has no grid named \"" << name << "\"");
    }

    // Seek to and read in the grid from the file.
    const GridDescriptor& gd = it->second;
#if OPENVDB_ABI_VERSION_NUMBER <= 2
    grid = readGrid(gd);
#else
    grid = (clip ? readGrid(gd, bbox) : readGrid(gd));
#endif

    if (gd.isInstance()) {
        /// @todo Refactor to share code with Archive::connectInstance()?
        NameMapCIter parentIt =
            findDescriptor(GridDescriptor::nameAsString(gd.instanceParentName()));
        if (parentIt == gridDescriptors().end()) {
            OPENVDB_THROW(KeyError, "missing instance parent \""
                << GridDescriptor::nameAsString(gd.instanceParentName())
                << "\" for grid " << GridDescriptor::nameAsString(gd.uniqueName())
                << " in file " << filename());
        }

        GridBase::Ptr parent;
#if OPENVDB_ABI_VERSION_NUMBER <= 2
        parent = readGrid(parentIt->second);
#else
        if (clip) {
            const CoordBBox indexBBox = grid->constTransform().worldToIndexNodeCentered(bbox);
            parent = readGrid(parentIt->second, indexBBox);
        } else {
            parent = readGrid(parentIt->second);
        }
#endif
        if (parent) grid->setTree(parent->baseTreePtr());
    }
    return grid;
}


////////////////////////////////////////


void
File::writeGrids(const GridCPtrVec& grids, const MetaMap& meta) const
{
    if (isOpen()) {
        OPENVDB_THROW(IoError,
            filename() << " cannot be written because it is open for reading");
    }

    // Create a file stream and write it out.
    std::ofstream file;
    file.open(filename().c_str(),
        std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

    if (file.fail()) {
        OPENVDB_THROW(IoError, "could not open " << filename() << " for writing");
    }

    // Write out the vdb.
    Archive::write(file, grids, /*seekable=*/true, meta);

    file.close();
}


////////////////////////////////////////


void
File::readGridDescriptors(std::istream& is)
{
    // This method should not be called for files that don't contain grid offsets.
    assert(inputHasGridOffsets());

    gridDescriptors().clear();

    for (int32_t i = 0, N = readGridCount(is); i < N; ++i) {
        // Read the grid descriptor.
        GridDescriptor gd;
        gd.read(is);

        // Add the descriptor to the dictionary.
        gridDescriptors().insert(std::make_pair(gd.gridName(), gd));

        // Skip forward to the next descriptor.
        gd.seekToEnd(is);
    }
}


////////////////////////////////////////


File::NameMapCIter
File::findDescriptor(const Name& name) const
{
    const Name uniqueName = GridDescriptor::stringAsUniqueName(name);

    // Find all descriptors with the given grid name.
    std::pair<NameMapCIter, NameMapCIter> range = gridDescriptors().equal_range(name);

    if (range.first == range.second) {
        // If no descriptors were found with the given grid name, the name might have
        // a suffix ("name[N]").  In that case, remove the "[N]" suffix and search again.
        range = gridDescriptors().equal_range(GridDescriptor::stripSuffix(uniqueName));
    }

    const size_t count = size_t(std::distance(range.first, range.second));
    if (count > 1 && name == uniqueName) {
        OPENVDB_LOG_WARN(filename() << " has more than one grid named \"" << name << "\"");
    }

    NameMapCIter ret = gridDescriptors().end();

    if (count > 0) {
        if (name == uniqueName) {
            // If the given grid name is unique or if no "[N]" index was given,
            // use the first matching descriptor.
            ret = range.first;
        } else {
            // If the given grid name has a "[N]" index, find the descriptor
            // with a matching unique name.
            for (NameMapCIter it = range.first; it != range.second; ++it) {
                const Name candidateName = it->second.uniqueName();
                if (candidateName == uniqueName || candidateName == name) {
                    ret = it;
                    break;
                }
            }
        }
    }
    return ret;
}


////////////////////////////////////////


GridBase::Ptr
File::createGrid(const GridDescriptor& gd) const
{
    // Create the grid.
    if (!GridBase::isRegistered(gd.gridType())) {
        OPENVDB_THROW(KeyError, "Cannot read grid "
            << GridDescriptor::nameAsString(gd.uniqueName())
            << " from " << filename() << ": grid type "
            << gd.gridType() << " is not registered");
    }

    GridBase::Ptr grid = GridBase::createGrid(gd.gridType());
    if (grid) grid->setSaveFloatAsHalf(gd.saveFloatAsHalf());

    return grid;
}


GridBase::ConstPtr
File::readGridPartial(const GridDescriptor& gd, bool readTopology) const
{
    // This method should not be called for files that don't contain grid offsets.
    assert(inputHasGridOffsets());

    GridBase::Ptr grid = createGrid(gd);

    // Seek to grid.
    gd.seekToGrid(inputStream());

    // Read the grid partially.
    readGridPartial(grid, inputStream(), gd.isInstance(), readTopology);

    // Promote to a const grid.
    GridBase::ConstPtr constGrid = grid;

    return constGrid;
}


GridBase::Ptr
File::readGrid(const GridDescriptor& gd) const
{
    return Impl::readGrid(*this, gd, Impl::NoBBox());
}


#if OPENVDB_ABI_VERSION_NUMBER >= 3
GridBase::Ptr
File::readGrid(const GridDescriptor& gd, const BBoxd& bbox) const
{
    return Impl::readGrid(*this, gd, bbox);
}


GridBase::Ptr
File::readGrid(const GridDescriptor& gd, const CoordBBox& bbox) const
{
    return Impl::readGrid(*this, gd, bbox);
}
#endif


void
File::readGridPartial(GridBase::Ptr grid, std::istream& is,
    bool isInstance, bool readTopology) const
{
    // This method should not be called for files that don't contain grid offsets.
    assert(inputHasGridOffsets());

    // This code needs to stay in sync with io::Archive::readGrid(), in terms of
    // the order of operations.
    readGridCompression(is);
    grid->readMeta(is);
    if (getFormatVersion(is) >= OPENVDB_FILE_VERSION_GRID_INSTANCING) {
        grid->readTransform(is);
        if (!isInstance && readTopology) {
            grid->readTopology(is);
        }
    } else {
        if (readTopology) {
            grid->readTopology(is);
            grid->readTransform(is);
        }
    }
}


////////////////////////////////////////


File::NameIterator
File::beginName() const
{
    if (!isOpen()) {
        OPENVDB_THROW(IoError, filename() << " is not open for reading");
    }
    return File::NameIterator(gridDescriptors().begin());
}


File::NameIterator
File::endName() const
{
    return File::NameIterator(gridDescriptors().end());
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
