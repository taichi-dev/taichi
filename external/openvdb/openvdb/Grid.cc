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

#include "Grid.h"

#include <openvdb/Metadata.h>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <tbb/mutex.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

/// @note For Houdini compatibility, boolean-valued metadata names
/// should begin with "is_".
const char
    * const GridBase::META_GRID_CLASS = "class",
    * const GridBase::META_GRID_CREATOR = "creator",
    * const GridBase::META_GRID_NAME = "name",
    * const GridBase::META_SAVE_HALF_FLOAT = "is_saved_as_half_float",
    * const GridBase::META_IS_LOCAL_SPACE = "is_local_space",
    * const GridBase::META_VECTOR_TYPE = "vector_type",
    * const GridBase::META_FILE_BBOX_MIN = "file_bbox_min",
    * const GridBase::META_FILE_BBOX_MAX = "file_bbox_max",
    * const GridBase::META_FILE_COMPRESSION = "file_compression",
    * const GridBase::META_FILE_MEM_BYTES = "file_mem_bytes",
    * const GridBase::META_FILE_VOXEL_COUNT = "file_voxel_count";


////////////////////////////////////////


namespace {

using GridFactoryMap = std::map<Name, GridBase::GridFactory>;
using GridFactoryMapCIter = GridFactoryMap::const_iterator;

using Mutex = tbb::mutex;
using Lock = Mutex::scoped_lock;

struct LockedGridRegistry {
    LockedGridRegistry() {}
    Mutex mMutex;
    GridFactoryMap mMap;
};

// Declare this at file scope to ensure thread-safe initialization.
Mutex sInitGridRegistryMutex;


// Global function for accessing the registry
LockedGridRegistry*
getGridRegistry()
{
    Lock lock(sInitGridRegistryMutex);

    static LockedGridRegistry* registry = nullptr;

    if (registry == nullptr) {

#ifdef __ICC
// Disable ICC "assignment to statically allocated variable" warning.
// This assignment is mutex-protected and therefore thread-safe.
__pragma(warning(disable:1711))
#endif

        registry = new LockedGridRegistry();

#ifdef __ICC
__pragma(warning(default:1711))
#endif

    }

    return registry;
}

} // unnamed namespace


bool
GridBase::isRegistered(const Name& name)
{
    LockedGridRegistry* registry = getGridRegistry();
    Lock lock(registry->mMutex);

    return (registry->mMap.find(name) != registry->mMap.end());
}


void
GridBase::registerGrid(const Name& name, GridFactory factory)
{
    LockedGridRegistry* registry = getGridRegistry();
    Lock lock(registry->mMutex);

    if (registry->mMap.find(name) != registry->mMap.end()) {
        OPENVDB_THROW(KeyError, "Grid type " << name << " is already registered");
    }

    registry->mMap[name] = factory;
}


void
GridBase::unregisterGrid(const Name& name)
{
    LockedGridRegistry* registry = getGridRegistry();
    Lock lock(registry->mMutex);

    registry->mMap.erase(name);
}


GridBase::Ptr
GridBase::createGrid(const Name& name)
{
    LockedGridRegistry* registry = getGridRegistry();
    Lock lock(registry->mMutex);

    GridFactoryMapCIter iter = registry->mMap.find(name);

    if (iter == registry->mMap.end()) {
        OPENVDB_THROW(LookupError, "Cannot create grid of unregistered type " << name);
    }

    return (iter->second)();
}


void
GridBase::clearRegistry()
{
    LockedGridRegistry* registry = getGridRegistry();
    Lock lock(registry->mMutex);

    registry->mMap.clear();
}


////////////////////////////////////////


GridClass
GridBase::stringToGridClass(const std::string& s)
{
    GridClass ret = GRID_UNKNOWN;
    std::string str = s;
    boost::trim(str);
    boost::to_lower(str);
    if (str == gridClassToString(GRID_LEVEL_SET)) {
        ret = GRID_LEVEL_SET;
    } else if (str == gridClassToString(GRID_FOG_VOLUME)) {
        ret = GRID_FOG_VOLUME;
    } else if (str == gridClassToString(GRID_STAGGERED)) {
        ret = GRID_STAGGERED;
    }
    return ret;
}


std::string
GridBase::gridClassToString(GridClass cls)
{
    std::string ret;
    switch (cls) {
        case GRID_UNKNOWN: ret = "unknown"; break;
        case GRID_LEVEL_SET: ret = "level set"; break;
        case GRID_FOG_VOLUME: ret = "fog volume"; break;
        case GRID_STAGGERED: ret = "staggered"; break;
    }
    return ret;
}

std::string
GridBase::gridClassToMenuName(GridClass cls)
{
    std::string ret;
    switch (cls) {
        case GRID_UNKNOWN: ret = "Other"; break;
        case GRID_LEVEL_SET: ret = "Level Set"; break;
        case GRID_FOG_VOLUME: ret = "Fog Volume"; break;
        case GRID_STAGGERED: ret = "Staggered Vector Field"; break;
    }
    return ret;
}



GridClass
GridBase::getGridClass() const
{
    GridClass cls = GRID_UNKNOWN;
    if (StringMetadata::ConstPtr s = this->getMetadata<StringMetadata>(META_GRID_CLASS)) {
        cls = stringToGridClass(s->value());
    }
    return cls;
}


void
GridBase::setGridClass(GridClass cls)
{
    this->insertMeta(META_GRID_CLASS, StringMetadata(gridClassToString(cls)));
}


void
GridBase::clearGridClass()
{
    this->removeMeta(META_GRID_CLASS);
}


////////////////////////////////////////


VecType
GridBase::stringToVecType(const std::string& s)
{
    VecType ret = VEC_INVARIANT;
    std::string str = s;
    boost::trim(str);
    boost::to_lower(str);
    if (str == vecTypeToString(VEC_COVARIANT)) {
        ret = VEC_COVARIANT;
    } else if (str == vecTypeToString(VEC_COVARIANT_NORMALIZE)) {
        ret = VEC_COVARIANT_NORMALIZE;
    } else if (str == vecTypeToString(VEC_CONTRAVARIANT_RELATIVE)) {
        ret = VEC_CONTRAVARIANT_RELATIVE;
    } else if (str == vecTypeToString(VEC_CONTRAVARIANT_ABSOLUTE)) {
        ret = VEC_CONTRAVARIANT_ABSOLUTE;
    }
    return ret;
}


std::string
GridBase::vecTypeToString(VecType typ)
{
    std::string ret;
    switch (typ) {
        case VEC_INVARIANT: ret = "invariant"; break;
        case VEC_COVARIANT: ret = "covariant"; break;
        case VEC_COVARIANT_NORMALIZE: ret = "covariant normalize"; break;
        case VEC_CONTRAVARIANT_RELATIVE: ret = "contravariant relative"; break;
        case VEC_CONTRAVARIANT_ABSOLUTE: ret = "contravariant absolute"; break;
    }
    return ret;
}


std::string
GridBase::vecTypeExamples(VecType typ)
{
    std::string ret;
    switch (typ) {
        case VEC_INVARIANT: ret = "Tuple/Color/UVW"; break;
        case VEC_COVARIANT: ret = "Gradient/Normal"; break;
        case VEC_COVARIANT_NORMALIZE: ret = "Unit Normal"; break;
        case VEC_CONTRAVARIANT_RELATIVE: ret = "Displacement/Velocity/Acceleration"; break;
        case VEC_CONTRAVARIANT_ABSOLUTE: ret = "Position"; break;
    }
    return ret;
}


std::string
GridBase::vecTypeDescription(VecType typ)
{
    std::string ret;
    switch (typ) {
        case VEC_INVARIANT:
            ret = "Does not transform";
            break;
        case VEC_COVARIANT:
            ret = "Apply the inverse-transpose transform matrix but ignore translation";
            break;
        case VEC_COVARIANT_NORMALIZE:
            ret = "Apply the inverse-transpose transform matrix but ignore translation"
                " and renormalize vectors";
            break;
        case VEC_CONTRAVARIANT_RELATIVE:
            ret = "Apply the forward transform matrix but ignore translation";
            break;
        case VEC_CONTRAVARIANT_ABSOLUTE:
            ret = "Apply the forward transform matrix, including translation";
            break;
    }
    return ret;
}


VecType
GridBase::getVectorType() const
{
    VecType typ = VEC_INVARIANT;
    if (StringMetadata::ConstPtr s = this->getMetadata<StringMetadata>(META_VECTOR_TYPE)) {
        typ = stringToVecType(s->value());
    }
    return typ;
}


void
GridBase::setVectorType(VecType typ)
{
    this->insertMeta(META_VECTOR_TYPE, StringMetadata(vecTypeToString(typ)));
}


void
GridBase::clearVectorType()
{
    this->removeMeta(META_VECTOR_TYPE);
}


////////////////////////////////////////


std::string
GridBase::getName() const
{
    if (Metadata::ConstPtr meta = (*this)[META_GRID_NAME]) return meta->str();
    return "";
}


void
GridBase::setName(const std::string& name)
{
    this->removeMeta(META_GRID_NAME);
    this->insertMeta(META_GRID_NAME, StringMetadata(name));
}


////////////////////////////////////////


std::string
GridBase::getCreator() const
{
    if (Metadata::ConstPtr meta = (*this)[META_GRID_CREATOR]) return meta->str();
    return "";
}


void
GridBase::setCreator(const std::string& creator)
{
    this->removeMeta(META_GRID_CREATOR);
    this->insertMeta(META_GRID_CREATOR, StringMetadata(creator));
}


////////////////////////////////////////


bool
GridBase::saveFloatAsHalf() const
{
    if (Metadata::ConstPtr meta = (*this)[META_SAVE_HALF_FLOAT]) {
        return meta->asBool();
    }
    return false;
}


void
GridBase::setSaveFloatAsHalf(bool saveAsHalf)
{
    this->removeMeta(META_SAVE_HALF_FLOAT);
    this->insertMeta(META_SAVE_HALF_FLOAT, BoolMetadata(saveAsHalf));
}


////////////////////////////////////////


bool
GridBase::isInWorldSpace() const
{
    bool local = false;
    if (Metadata::ConstPtr meta = (*this)[META_IS_LOCAL_SPACE]) {
        local = meta->asBool();
    }
    return !local;
}


void
GridBase::setIsInWorldSpace(bool world)
{
    this->removeMeta(META_IS_LOCAL_SPACE);
    this->insertMeta(META_IS_LOCAL_SPACE, BoolMetadata(!world));
}


////////////////////////////////////////


void
GridBase::addStatsMetadata()
{
    const CoordBBox bbox = this->evalActiveVoxelBoundingBox();
    this->removeMeta(META_FILE_BBOX_MIN);
    this->removeMeta(META_FILE_BBOX_MAX);
    this->removeMeta(META_FILE_MEM_BYTES);
    this->removeMeta(META_FILE_VOXEL_COUNT);
    this->insertMeta(META_FILE_BBOX_MIN,    Vec3IMetadata(bbox.min().asVec3i()));
    this->insertMeta(META_FILE_BBOX_MAX,    Vec3IMetadata(bbox.max().asVec3i()));
    this->insertMeta(META_FILE_MEM_BYTES,   Int64Metadata(this->memUsage()));
    this->insertMeta(META_FILE_VOXEL_COUNT, Int64Metadata(this->activeVoxelCount()));
}


MetaMap::Ptr
GridBase::getStatsMetadata() const
{
    const char* const fields[] = {
        META_FILE_BBOX_MIN,
        META_FILE_BBOX_MAX,
        META_FILE_MEM_BYTES,
        META_FILE_VOXEL_COUNT,
        nullptr
    };

    /// @todo Check that the fields are of the correct type?
    MetaMap::Ptr ret(new MetaMap);
    for (int i = 0; fields[i] != nullptr; ++i) {
        if (Metadata::ConstPtr m = (*this)[fields[i]]) {
            ret->insertMeta(fields[i], *m);
        }
    }
    return ret;
}


////////////////////////////////////////


#if OPENVDB_ABI_VERSION_NUMBER >= 3
void
GridBase::clipGrid(const BBoxd& worldBBox)
{
    const CoordBBox indexBBox =
        this->constTransform().worldToIndexNodeCentered(worldBBox);
    this->clip(indexBBox);
}
#endif

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
