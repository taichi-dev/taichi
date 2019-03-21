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

#include "Metadata.h"

#include <tbb/mutex.h>
#include <algorithm> // for std::min()
#include <map>
#include <sstream>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

using Mutex = tbb::mutex;
using Lock = Mutex::scoped_lock;

using createMetadata = Metadata::Ptr (*)();
using MetadataFactoryMap = std::map<Name, createMetadata>;
using MetadataFactoryMapCIter = MetadataFactoryMap::const_iterator;

struct LockedMetadataTypeRegistry {
    LockedMetadataTypeRegistry() {}
    ~LockedMetadataTypeRegistry() {}
    Mutex mMutex;
    MetadataFactoryMap mMap;
};

// Declare this at file scope to ensure thread-safe initialization
static Mutex theInitMetadataTypeRegistryMutex;

// Global function for accessing the regsitry
static LockedMetadataTypeRegistry*
getMetadataTypeRegistry()
{
    Lock lock(theInitMetadataTypeRegistryMutex);

    static LockedMetadataTypeRegistry *registry = nullptr;

    if (registry == nullptr) {
#if defined(__ICC)
__pragma(warning(disable:1711)) // disable ICC "assignment to static variable" warnings
#endif
        registry = new LockedMetadataTypeRegistry();
#if defined(__ICC)
__pragma(warning(default:1711))
#endif
    }

    return registry;
}

bool
Metadata::isRegisteredType(const Name &typeName)
{
    LockedMetadataTypeRegistry *registry = getMetadataTypeRegistry();
    Lock lock(registry->mMutex);

    return (registry->mMap.find(typeName) != registry->mMap.end());
}

void
Metadata::registerType(const Name &typeName, Metadata::Ptr (*createMetadata)())
{
    LockedMetadataTypeRegistry *registry = getMetadataTypeRegistry();
    Lock lock(registry->mMutex);

    if (registry->mMap.find(typeName) != registry->mMap.end()) {
        OPENVDB_THROW(KeyError,
            "Cannot register " << typeName << ". Type is already registered");
    }

    registry->mMap[typeName] = createMetadata;
}

void
Metadata::unregisterType(const Name &typeName)
{
    LockedMetadataTypeRegistry *registry = getMetadataTypeRegistry();
    Lock lock(registry->mMutex);

    registry->mMap.erase(typeName);
}

Metadata::Ptr
Metadata::createMetadata(const Name &typeName)
{
    LockedMetadataTypeRegistry *registry = getMetadataTypeRegistry();
    Lock lock(registry->mMutex);

    MetadataFactoryMapCIter iter = registry->mMap.find(typeName);

    if (iter == registry->mMap.end()) {
        OPENVDB_THROW(LookupError,
            "Cannot create metadata for unregistered type " << typeName);
    }

    return (iter->second)();
}

void
Metadata::clearRegistry()
{
    LockedMetadataTypeRegistry *registry = getMetadataTypeRegistry();
    Lock lock(registry->mMutex);

    registry->mMap.clear();
}


////////////////////////////////////////


bool
Metadata::operator==(const Metadata& other) const
{
    if (other.size() != this->size()) return false;
    if (other.typeName() != this->typeName()) return false;

    std::ostringstream
        bytes(std::ios_base::binary),
        otherBytes(std::ios_base::binary);
    try {
        this->writeValue(bytes);
        other.writeValue(otherBytes);
        return (bytes.str() == otherBytes.str());
    } catch (Exception&) {}
    return false;
}


////////////////////////////////////////


#if OPENVDB_ABI_VERSION_NUMBER >= 5

Metadata::Ptr
UnknownMetadata::copy() const
{
    Metadata::Ptr metadata{new UnknownMetadata{mTypeName}};
    static_cast<UnknownMetadata*>(metadata.get())->setValue(mBytes);
    return metadata;
}


void
UnknownMetadata::copy(const Metadata& other)
{
    std::ostringstream ostr(std::ios_base::binary);
    other.write(ostr);
    std::istringstream istr(ostr.str(), std::ios_base::binary);
    const auto numBytes = readSize(istr);
    readValue(istr, numBytes);
}


void
UnknownMetadata::readValue(std::istream& is, Index32 numBytes)
{
    mBytes.clear();
    if (numBytes > 0) {
        ByteVec buffer(numBytes);
        is.read(reinterpret_cast<char*>(&buffer[0]), numBytes);
        mBytes.swap(buffer);
    }
}


void
UnknownMetadata::writeValue(std::ostream& os) const
{
    if (!mBytes.empty()) {
        os.write(reinterpret_cast<const char*>(&mBytes[0]), mBytes.size());
    }
}

#else // if OPENVDB_ABI_VERSION_NUMBER < 5

void
UnknownMetadata::readValue(std::istream& is, Index32 numBytes)
{
    // Read and discard the metadata (without seeking, because
    // the stream might not be seekable).
    const size_t BUFFER_SIZE = 1024;
    std::vector<char> buffer(BUFFER_SIZE);
    for (Index32 bytesRemaining = numBytes; bytesRemaining > 0; ) {
        const Index32 bytesToSkip = std::min<Index32>(bytesRemaining, BUFFER_SIZE);
        is.read(&buffer[0], bytesToSkip);
        bytesRemaining -= bytesToSkip;
    }
}


void
UnknownMetadata::writeValue(std::ostream&) const
{
    OPENVDB_THROW(TypeError, "Metadata has unknown type");
}

#endif

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
