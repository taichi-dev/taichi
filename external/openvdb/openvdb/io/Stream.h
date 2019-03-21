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

#ifndef OPENVDB_IO_STREAM_HAS_BEEN_INCLUDED
#define OPENVDB_IO_STREAM_HAS_BEEN_INCLUDED

#include "Archive.h"
#include <iosfwd>
#include <memory>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

class GridDescriptor;


/// Grid archive associated with arbitrary input and output streams (not necessarily files)
class OPENVDB_API Stream: public Archive
{
public:
    /// @brief Read grids from an input stream.
    /// @details If @a delayLoad is true, map the contents of the input stream
    /// into memory and enable delayed loading of grids.
    /// @note Define the environment variable @c OPENVDB_DISABLE_DELAYED_LOAD
    /// to disable delayed loading unconditionally.
    explicit Stream(std::istream&, bool delayLoad = true);

    /// Construct an archive for stream output.
    Stream();
    /// Construct an archive for output to the given stream.
    explicit Stream(std::ostream&);

    Stream(const Stream&);
    Stream& operator=(const Stream&);

    ~Stream() override;

    /// @brief Return a copy of this archive.
    Archive::Ptr copy() const override;

    /// Return the file-level metadata in a newly created MetaMap.
    MetaMap::Ptr getMetadata() const;

    /// Return pointers to the grids that were read from the input stream.
    GridPtrVecPtr getGrids();

    /// @brief Write the grids in the given container to this archive's output stream.
    /// @throw ValueError if this archive was constructed without specifying an output stream.
    void write(const GridCPtrVec&, const MetaMap& = MetaMap()) const override;

    /// @brief Write the grids in the given container to this archive's output stream.
    /// @throw ValueError if this archive was constructed without specifying an output stream.
    template<typename GridPtrContainerT>
    void write(const GridPtrContainerT&, const MetaMap& = MetaMap()) const;

private:
    /// Create a new grid of the type specified by the given descriptor,
    /// then populate the grid from the given input stream.
    /// @return the newly created grid.
    GridBase::Ptr readGrid(const GridDescriptor&, std::istream&) const;

    void writeGrids(std::ostream&, const GridCPtrVec&, const MetaMap&) const;


    struct Impl;
    std::unique_ptr<Impl> mImpl;
};


////////////////////////////////////////


template<typename GridPtrContainerT>
inline void
Stream::write(const GridPtrContainerT& container, const MetaMap& metadata) const
{
    GridCPtrVec grids;
    std::copy(container.begin(), container.end(), std::back_inserter(grids));
    this->write(grids, metadata);
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_STREAM_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
