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

/// @file version.h
/// @brief Library and file format version numbers
///
/// @details
/// When the library is built with the latest ABI, its namespace has the form
/// <B>openvdb::vX_Y</B>, where @e X and @e Y are the major and minor version numbers.
///
/// The library can be built using an older ABI by changing the value of the
/// @b OPENVDB_ABI_VERSION_NUMBER macro (e.g., via <TT>-DOPENVDB_ABI_VERSION_NUMBER=<I>N</I></TT>).
/// In that case, the namespace has the form <B>openvdb::vX_YabiN</B>,
/// where N is the ABI version number.
/// The ABI version must be set consistently when building code that depends on OpenVDB.
///
/// The ABI version number defaults to the library major version number,
/// which gets incremented whenever changes are made to the ABI of the
/// Grid class or related classes (Tree, Transform, Metadata, etc.).
/// Setting the ABI version number to an earlier library version number
/// disables grid ABI changes made since that library version.
/// The OpenVDB 1.x ABI is no longer supported, and support for other old ABIs
/// might also eventually be dropped.
///
/// The library minor version number gets incremented whenever a change is made
/// to any aspect of the public API (not just the grid API) that necessitates
/// changes to client code.  Changes to APIs in private or internal namespaces
/// do not trigger a minor version number increment; such APIs should not be used
/// in client code.
///
/// A patch version number increment indicates a change&mdash;usually a new feature
/// or a bug fix&mdash;that does not necessitate changes to client code but rather
/// only recompilation of that code (because the library namespace incorporates
/// the version number).
///
/// The file format version number gets incremented when it becomes possible
/// to write files that cannot safely be read with older versions of the library.
/// Not all files written in a newer format are incompatible with older libraries, however.
/// And in general, files containing grids of unknown type can be read safely,
/// although the unknown grids will not be accessible.

#ifndef OPENVDB_VERSION_HAS_BEEN_INCLUDED
#define OPENVDB_VERSION_HAS_BEEN_INCLUDED

#include "Platform.h"

/// @name Utilities
/// @{
/// @cond OPENVDB_VERSION_INTERNAL
#define OPENVDB_PREPROC_STRINGIFY_(x) #x
/// @endcond
/// @brief Return @a x as a string literal.  If @a x is a macro,
/// return its value as a string literal.
/// @hideinitializer
#define OPENVDB_PREPROC_STRINGIFY(x) OPENVDB_PREPROC_STRINGIFY_(x)

/// @cond OPENVDB_VERSION_INTERNAL
#define OPENVDB_PREPROC_CONCAT_(x, y) x ## y
/// @endcond
/// @brief Form a new token by concatenating two existing tokens.
/// If either token is a macro, concatenate its value.
/// @hideinitializer
#define OPENVDB_PREPROC_CONCAT(x, y) OPENVDB_PREPROC_CONCAT_(x, y)
/// @}


// Library major, minor and patch version numbers
#define OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER 6
#define OPENVDB_LIBRARY_MINOR_VERSION_NUMBER 0
#define OPENVDB_LIBRARY_PATCH_VERSION_NUMBER 0

// If OPENVDB_ABI_VERSION_NUMBER is already defined (e.g., via -DOPENVDB_ABI_VERSION_NUMBER=N)
// use that ABI version.  Otherwise, use this library version's default ABI.
#ifdef OPENVDB_ABI_VERSION_NUMBER
    #if OPENVDB_ABI_VERSION_NUMBER > OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER
        #error expected OPENVDB_ABI_VERSION_NUMBER <= OPENVDB_LIBRARY_MAJOR VERSION_NUMBER
    #endif
#else
    // Older versions of the library used the macros OPENVDB_2_ABI_COMPATIBLE
    // and OPENVDB_3_ABI_COMPATIBLE.  For now, continue to support them.
    #if defined OPENVDB_2_ABI_COMPATIBLE ///< @todo deprecated
        #define OPENVDB_ABI_VERSION_NUMBER 2
    #elif defined OPENVDB_3_ABI_COMPATIBLE ///< @todo deprecated
        #define OPENVDB_ABI_VERSION_NUMBER 3
    #else
        #define OPENVDB_ABI_VERSION_NUMBER OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER
    #endif
#endif

#if OPENVDB_ABI_VERSION_NUMBER == OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER
    /// @brief The version namespace name for this library version
    /// @hideinitializer
    ///
    /// When the ABI version number matches the library major version number,
    /// symbols are named as in the following examples:
    /// - @b openvdb::vX_Y::Vec3i
    /// - @b openvdb::vX_Y::io::File
    /// - @b openvdb::vX_Y::tree::Tree
    ///
    /// where X and Y are the major and minor version numbers.
    ///
    /// When the ABI version number does not match the library major version number,
    /// symbol names include the ABI version:
    /// - @b openvdb::vX_YabiN::Vec3i
    /// - @b openvdb::vX_YabiN::io::File
    /// - @b openvdb::vX_YabiN::tree::Tree
    ///
    /// where X, Y and N are the major, minor and ABI version numbers, respectively.
    #define OPENVDB_VERSION_NAME                                            \
        OPENVDB_PREPROC_CONCAT(v,                                           \
        OPENVDB_PREPROC_CONCAT(OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER,        \
        OPENVDB_PREPROC_CONCAT(_, OPENVDB_LIBRARY_MINOR_VERSION_NUMBER)))
#else
    // This duplication of code is necessary to avoid issues with recursive macro expansion.
    #define OPENVDB_VERSION_NAME                                            \
        OPENVDB_PREPROC_CONCAT(v,                                           \
        OPENVDB_PREPROC_CONCAT(OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER,        \
        OPENVDB_PREPROC_CONCAT(_,                                           \
        OPENVDB_PREPROC_CONCAT(OPENVDB_LIBRARY_MINOR_VERSION_NUMBER,        \
        OPENVDB_PREPROC_CONCAT(abi, OPENVDB_ABI_VERSION_NUMBER)))))
#endif

/// @brief Library version number string of the form "<major>.<minor>.<patch>"
/// @details This is a macro rather than a static constant because we typically
/// want the compile-time version number, not the runtime version number
/// (although the two are usually the same).
/// @hideinitializer
#define OPENVDB_LIBRARY_VERSION_STRING \
    OPENVDB_PREPROC_STRINGIFY(OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER) "." \
    OPENVDB_PREPROC_STRINGIFY(OPENVDB_LIBRARY_MINOR_VERSION_NUMBER) "." \
    OPENVDB_PREPROC_STRINGIFY(OPENVDB_LIBRARY_PATCH_VERSION_NUMBER)

/// @brief Library version number string of the form "<major>.<minor>.<patch>abi<abi>"
/// @details This is a macro rather than a static constant because we typically
/// want the compile-time version number, not the runtime version number
/// (although the two are usually the same).
/// @hideinitializer
#define OPENVDB_LIBRARY_ABI_VERSION_STRING \
    OPENVDB_LIBRARY_VERSION_STRING "abi" OPENVDB_PREPROC_STRINGIFY(OPENVDB_ABI_VERSION_NUMBER)

/// Library version number as a packed integer ("%02x%02x%04x", major, minor, patch)
#define OPENVDB_LIBRARY_VERSION_NUMBER \
    ((OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER << 24) | \
    ((OPENVDB_LIBRARY_MINOR_VERSION_NUMBER & 0xFF) << 16) | \
    (OPENVDB_LIBRARY_PATCH_VERSION_NUMBER & 0xFFFF))


/// By default, the @b OPENVDB_REQUIRE_VERSION_NAME macro is undefined, and
/// symbols from the version namespace are promoted to the top-level namespace
/// so that, for example, @b openvdb::v6_0::io::File can be referred to
/// simply as @b openvdb::io::File.
///
/// When @b OPENVDB_REQUIRE_VERSION_NAME is defined, symbols must be
/// fully namespace-qualified.
/// @hideinitializer
#ifdef OPENVDB_REQUIRE_VERSION_NAME
#define OPENVDB_USE_VERSION_NAMESPACE
#else
// The empty namespace clause below ensures that OPENVDB_VERSION_NAME
// is recognized as a namespace name.
#define OPENVDB_USE_VERSION_NAMESPACE \
    namespace OPENVDB_VERSION_NAME {} \
    using namespace OPENVDB_VERSION_NAME;
#endif


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

/// @brief The magic number is stored in the first four bytes of every VDB file.
/// @details This can be used to quickly test whether we have a valid file or not.
const int32_t OPENVDB_MAGIC = 0x56444220;

// Library major, minor and patch version numbers
const uint32_t
    OPENVDB_LIBRARY_MAJOR_VERSION = OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER,
    OPENVDB_LIBRARY_MINOR_VERSION = OPENVDB_LIBRARY_MINOR_VERSION_NUMBER,
    OPENVDB_LIBRARY_PATCH_VERSION = OPENVDB_LIBRARY_PATCH_VERSION_NUMBER;
/// Library version number as a packed integer ("%02x%02x%04x", major, minor, patch)
const uint32_t OPENVDB_LIBRARY_VERSION = OPENVDB_LIBRARY_VERSION_NUMBER;
// ABI version number
const uint32_t OPENVDB_ABI_VERSION = OPENVDB_ABI_VERSION_NUMBER;

/// @brief The current version number of the VDB file format
/// @details  This can be used to enable various backwards compatibility switches
/// or to reject files that cannot be read.
const uint32_t OPENVDB_FILE_VERSION = 224;

/// Notable file format version numbers
enum {
    OPENVDB_FILE_VERSION_ROOTNODE_MAP = 213,
    OPENVDB_FILE_VERSION_INTERNALNODE_COMPRESSION = 214,
    OPENVDB_FILE_VERSION_SIMPLIFIED_GRID_TYPENAME = 215,
    OPENVDB_FILE_VERSION_GRID_INSTANCING = 216,
    OPENVDB_FILE_VERSION_BOOL_LEAF_OPTIMIZATION = 217,
    OPENVDB_FILE_VERSION_BOOST_UUID = 218,
    OPENVDB_FILE_VERSION_NO_GRIDMAP = 219,
    OPENVDB_FILE_VERSION_NEW_TRANSFORM = 219,
    OPENVDB_FILE_VERSION_SELECTIVE_COMPRESSION = 220,
    OPENVDB_FILE_VERSION_FLOAT_FRUSTUM_BBOX = 221,
    OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION = 222,
    OPENVDB_FILE_VERSION_BLOSC_COMPRESSION = 223,
    OPENVDB_FILE_VERSION_POINT_INDEX_GRID = 223,
    OPENVDB_FILE_VERSION_MULTIPASS_IO = 224
};


/// Return a library version number string of the form "<major>.<minor>.<patch>".
inline constexpr const char* getLibraryVersionString() { return OPENVDB_LIBRARY_VERSION_STRING; }
/// Return a library version number string of the form "<major>.<minor>.<patch>abi<abi>".
inline constexpr const char* getLibraryAbiVersionString() {
    return OPENVDB_LIBRARY_ABI_VERSION_STRING;
}


struct VersionId {
    uint32_t first, second;
    VersionId(): first(0), second(0) {}
    VersionId(uint32_t major, uint32_t minor): first(major), second(minor) {}
};

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_VERSION_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
