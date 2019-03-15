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

#include "Stream.h"

#include "File.h" ///< @todo refactor
#include "GridDescriptor.h"
#include "TempFile.h"
#include <openvdb/Exceptions.h>
#include <cstdint>
#include <boost/iostreams/copy.hpp>
#include <cstdio> // for remove()
#include <functional> // for std::bind()
#include <iostream>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

struct Stream::Impl
{
    Impl(): mOutputStream{nullptr} {}
    Impl(const Impl& other) { *this = other; }
    Impl& operator=(const Impl& other)
    {
        if (&other != this) {
            mMeta = other.mMeta; ///< @todo deep copy?
            mGrids = other.mGrids; ///< @todo deep copy?
            mOutputStream = other.mOutputStream;
            mFile.reset();
        }
        return *this;
    }

    MetaMap::Ptr mMeta;
    GridPtrVecPtr mGrids;
    std::ostream* mOutputStream;
    std::unique_ptr<File> mFile;
};


////////////////////////////////////////


namespace {

/// @todo Use MappedFile auto-deletion instead.
void
removeTempFile(const std::string expectedFilename, const std::string& filename)
{
    if (filename == expectedFilename) {
        if (0 != std::remove(filename.c_str())) {
            std::string mesg = getErrorString();
            if (!mesg.empty()) mesg = " (" + mesg + ")";
            OPENVDB_LOG_WARN("failed to remove temporary file " << filename << mesg);
        }
    }
}

}


Stream::Stream(std::istream& is, bool delayLoad): mImpl(new Impl)
{
    if (!is) return;

    if (delayLoad && Archive::isDelayedLoadingEnabled()) {
        // Copy the contents of the stream to a temporary private file
        // and open the file instead.
        std::unique_ptr<TempFile> tempFile;
        try {
            tempFile.reset(new TempFile);
        } catch (std::exception& e) {
            std::string mesg;
            if (e.what()) mesg = std::string(" (") + e.what() + ")";
            OPENVDB_LOG_WARN("failed to create a temporary file for delayed loading" << mesg
                << "; will read directly from the input stream instead");
        }
        if (tempFile) {
            boost::iostreams::copy(is, *tempFile);
            const std::string& filename = tempFile->filename();
            mImpl->mFile.reset(new File(filename));
            mImpl->mFile->setCopyMaxBytes(0); // don't make a copy of the temporary file
            /// @todo Need to pass auto-deletion flag to MappedFile.
            mImpl->mFile->open(delayLoad,
                std::bind(&removeTempFile, filename, std::placeholders::_1));
        }
    }

    if (!mImpl->mFile) {
        readHeader(is);

        // Tag the input stream with the library and file format version numbers
        // and the compression options specified in the header.
        StreamMetadata::Ptr streamMetadata(new StreamMetadata);
        io::setStreamMetadataPtr(is, streamMetadata, /*transfer=*/false);
        io::setVersion(is, libraryVersion(), fileVersion());
        io::setDataCompression(is, compression());

        // Read in the VDB metadata.
        mImpl->mMeta.reset(new MetaMap);
        mImpl->mMeta->readMeta(is);

        // Read in the number of grids.
        const int32_t gridCount = readGridCount(is);

        // Read in all grids and insert them into mGrids.
        mImpl->mGrids.reset(new GridPtrVec);
        std::vector<GridDescriptor> descriptors;
        descriptors.reserve(gridCount);
        Archive::NamedGridMap namedGrids;
        for (int32_t i = 0; i < gridCount; ++i) {
            GridDescriptor gd;
            gd.read(is);
            descriptors.push_back(gd);
            GridBase::Ptr grid = readGrid(gd, is);
            mImpl->mGrids->push_back(grid);
            namedGrids[gd.uniqueName()] = grid;
        }

        // Connect instances (grids that share trees with other grids).
        for (size_t i = 0, N = descriptors.size(); i < N; ++i) {
            Archive::connectInstance(descriptors[i], namedGrids);
        }
    }
}


Stream::Stream(): mImpl(new Impl)
{
}


Stream::Stream(std::ostream& os): mImpl(new Impl)
{
    mImpl->mOutputStream = &os;
}


Stream::~Stream()
{
}


Stream::Stream(const Stream& other): Archive(other), mImpl(new Impl(*other.mImpl))
{
}


Stream&
Stream::operator=(const Stream& other)
{
    if (&other != this) {
        mImpl.reset(new Impl(*other.mImpl));
    }
    return *this;
}


SharedPtr<Archive>
Stream::copy() const
{
    return SharedPtr<Archive>(new Stream(*this));
}


////////////////////////////////////////


GridBase::Ptr
Stream::readGrid(const GridDescriptor& gd, std::istream& is) const
{
    GridBase::Ptr grid;

    if (!GridBase::isRegistered(gd.gridType())) {
        OPENVDB_THROW(TypeError, "can't read grid \""
            << GridDescriptor::nameAsString(gd.uniqueName()) <<
            "\" from input stream because grid type " << gd.gridType() << " is unknown");
    } else {
        grid = GridBase::createGrid(gd.gridType());
        if (grid) grid->setSaveFloatAsHalf(gd.saveFloatAsHalf());

        Archive::readGrid(grid, gd, is);
    }
    return grid;
}


void
Stream::write(const GridCPtrVec& grids, const MetaMap& metadata) const
{
    if (mImpl->mOutputStream == nullptr) {
        OPENVDB_THROW(ValueError, "no output stream was specified");
    }
    this->writeGrids(*mImpl->mOutputStream, grids, metadata);
}


void
Stream::writeGrids(std::ostream& os, const GridCPtrVec& grids, const MetaMap& metadata) const
{
    Archive::write(os, grids, /*seekable=*/false, metadata);
}


////////////////////////////////////////


MetaMap::Ptr
Stream::getMetadata() const
{
    MetaMap::Ptr result;
    if (mImpl->mFile) {
        result = mImpl->mFile->getMetadata();
    } else if (mImpl->mMeta) {
        // Return a deep copy of the file-level metadata
        // that was read when this object was constructed.
        result.reset(new MetaMap(*mImpl->mMeta));
    }
    return result;
}


GridPtrVecPtr
Stream::getGrids()
{
    if (mImpl->mFile) {
        return mImpl->mFile->getGrids();
    }
    return mImpl->mGrids;
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
