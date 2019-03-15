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

/// @author Ken Museth
///
/// @file LevelSetPlatonic.h
///
/// @brief Generate a narrow-band level sets of the five platonic solids.
///
/// @note By definition a level set has a fixed narrow band width
/// (the half width is defined by LEVEL_SET_HALF_WIDTH in Types.h),
/// whereas an SDF can have a variable narrow band width.

#ifndef OPENVDB_TOOLS_LEVELSETPLATONIC_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETPLATONIC_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/util/NullInterrupter.h>
#include <type_traits>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a platonic solid.
///
/// @param faceCount    number of faces of the platonic solid, i.e. 4, 6, 8, 12 or 20
/// @param scale        scale of the platonic solid in world units
/// @param center       center of the platonic solid in world units
/// @param voxelSize    voxel size in world units
/// @param halfWidth    half the width of the narrow band, in voxel units
/// @param interrupt    a pointer adhering to the util::NullInterrupter interface
///
/// @details Faces: TETRAHEDRON=4, CUBE=6, OCTAHEDRON=8, DODECAHEDRON=12, ICOSAHEDRON=20
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template<typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetPlatonic(
    int faceCount, // 4, 6, 8, 12 or 20
    float scale = 1.0f,
    const Vec3f& center = Vec3f(0.0f),
    float voxelSize = 0.1f,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupt = nullptr);

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a platonic solid.
///
/// @param faceCount    number of faces of the platonic solid, i.e. 4, 6, 8, 12 or 20
/// @param scale        scale of the platonic solid in world units
/// @param center       center of the platonic solid in world units
/// @param voxelSize    voxel size in world units
/// @param halfWidth    half the width of the narrow band, in voxel units
///
/// @details Faces: TETRAHEDRON=4, CUBE=6, OCTAHEDRON=8, DODECAHEDRON=12, ICOSAHEDRON=20
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template<typename GridType>
typename GridType::Ptr
createLevelSetPlatonic(
    int faceCount,// 4, 6, 8, 12 or 20
    float scale = 1.0f,
    const Vec3f& center = Vec3f(0.0f),
    float voxelSize = 0.1f,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH))
{
    util::NullInterrupter tmp;
    return createLevelSetPlatonic<GridType>(faceCount, scale, center, voxelSize, halfWidth, &tmp);
}

////////////////////////////////////////////////////////////////////////////////

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a tetrahedron.
///
/// @param scale        scale of the platonic solid in world units
/// @param center       center of the platonic solid in world units
/// @param voxelSize    voxel size in world units
/// @param halfWidth    half the width of the narrow band, in voxel units
/// @param interrupt    a pointer adhering to the util::NullInterrupter interface
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template<typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetTetrahedron(
    float scale = 1.0f,
    const Vec3f& center = Vec3f(0.0f),
    float voxelSize = 0.1f,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupt =  nullptr)
{
    return createLevelSetPlatonic<GridType, InterruptT>(
        4, scale, center, voxelSize, halfWidth, interrupt);
}

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a tetrahedron.
///
/// @param scale        scale of the platonic solid in world units
/// @param center       center of the platonic solid in world units
/// @param voxelSize    voxel size in world units
/// @param halfWidth    half the width of the narrow band, in voxel units
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template<typename GridType>
typename GridType::Ptr
createLevelSetTetrahedron(
    float scale = 1.0f,
    const Vec3f& center = Vec3f(0.0f),
    float voxelSize = 0.1f,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH))
{
    util::NullInterrupter tmp;
    return createLevelSetPlatonic<GridType>(4, scale, center, voxelSize, halfWidth, &tmp);
}

////////////////////////////////////////////////////////////////////////////////

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a cube.
///
/// @param scale        scale of the platonic solid in world units
/// @param center       center of the platonic solid in world units
/// @param voxelSize    voxel size in world units
/// @param halfWidth    half the width of the narrow band, in voxel units
/// @param interrupt    a pointer adhering to the util::NullInterrupter interface
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template<typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetCube(
    float scale = 1.0f,
    const Vec3f& center = Vec3f(0.0f),
    float voxelSize = 0.1f,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupt =  nullptr)
{
    return createLevelSetPlatonic<GridType>(6, scale, center, voxelSize, halfWidth, interrupt);
}

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a cube.
///
/// @param scale        scale of the platonic solid in world units
/// @param center       center of the platonic solid in world units
/// @param voxelSize    voxel size in world units
/// @param halfWidth    half the width of the narrow band, in voxel units
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template<typename GridType>
typename GridType::Ptr
createLevelSetCube(
    float scale = 1.0f,
    const Vec3f& center = Vec3f(0.0f),
    float voxelSize = 0.1f,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH))
{
    util::NullInterrupter tmp;
    return createLevelSetPlatonic<GridType>(6, scale, center, voxelSize, halfWidth, &tmp);
}

////////////////////////////////////////////////////////////////////////////////

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of an octahedron.
///
/// @param scale        scale of the platonic solid in world units
/// @param center       center of the platonic solid in world units
/// @param voxelSize    voxel size in world units
/// @param halfWidth    half the width of the narrow band, in voxel units
/// @param interrupt    a pointer adhering to the util::NullInterrupter interface
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template<typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetOctahedron(
    float scale = 1.0f,
    const Vec3f& center = Vec3f(0.0f),
    float voxelSize = 0.1f,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupt = nullptr)
{
    return createLevelSetPlatonic<GridType>(8, scale, center, voxelSize, halfWidth, interrupt);
}

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of an octahedron.
///
/// @param scale        scale of the platonic solid in world units
/// @param center       center of the platonic solid in world units
/// @param voxelSize    voxel size in world units
/// @param halfWidth    half the width of the narrow band, in voxel units
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template<typename GridType>
typename GridType::Ptr
createLevelSetOctahedron(
    float scale = 1.0f,
    const Vec3f& center = Vec3f(0.0f),
    float voxelSize = 0.1f,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH))
{
    util::NullInterrupter tmp;
    return createLevelSetPlatonic<GridType>(8, scale, center, voxelSize, halfWidth, &tmp);
}

////////////////////////////////////////////////////////////////////////////////

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a dodecahedron.
///
/// @param scale        scale of the platonic solid in world units
/// @param center       center of the platonic solid in world units
/// @param voxelSize    voxel size in world units
/// @param halfWidth    half the width of the narrow band, in voxel units
/// @param interrupt    a pointer adhering to the util::NullInterrupter interface
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template<typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetDodecahedron(
    float scale = 1.0f,
    const Vec3f& center = Vec3f(0.0f),
    float voxelSize = 0.1f,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupt = nullptr)
{
    return createLevelSetPlatonic<GridType>(12, scale, center, voxelSize, halfWidth, interrupt);
}

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a dodecahedron.
///
/// @param scale        scale of the platonic solid in world units
/// @param center       center of the platonic solid in world units
/// @param voxelSize    voxel size in world units
/// @param halfWidth    half the width of the narrow band, in voxel units
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template<typename GridType>
typename GridType::Ptr
createLevelSetDodecahedron(
    float scale = 1.0f,
    const Vec3f& center = Vec3f(0.0f),
    float voxelSize = 0.1f,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH))
{
    util::NullInterrupter tmp;
    return createLevelSetPlatonic<GridType>(12, scale, center, voxelSize, halfWidth, &tmp);
}

////////////////////////////////////////////////////////////////////////////////

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of an icosahedron.
///
/// @param scale        scale of the platonic solid in world units
/// @param center       center of the platonic solid in world units
/// @param voxelSize    voxel size in world units
/// @param halfWidth    half the width of the narrow band, in voxel units
/// @param interrupt    a pointer adhering to the util::NullInterrupter interface
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template<typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetIcosahedron(
    float scale = 1.0f,
    const Vec3f& center = Vec3f(0.0f),
    float voxelSize = 0.1f,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupt = nullptr)
{
    return createLevelSetPlatonic<GridType>(20, scale, center, voxelSize, halfWidth, interrupt);
}

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of an icosahedron.
///
/// @param scale        scale of the platonic solid in world units
/// @param center       center of the platonic solid in world units
/// @param voxelSize    voxel size in world units
/// @param halfWidth    half the width of the narrow band, in voxel units
///
/// @note @c GridType::ValueType must be a floating-point scalar.
template<typename GridType>
typename GridType::Ptr
createLevelSetIcosahedron(
    float scale = 1.0f,
    const Vec3f& center = Vec3f(0.0f),
    float voxelSize = 0.1f,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH))
{
    util::NullInterrupter tmp;
    return createLevelSetPlatonic<GridType>(20, scale, center, voxelSize, halfWidth, &tmp);
}

////////////////////////////////////////////////////////////////////////////////

template<typename GridType, typename InterruptT>
typename GridType::Ptr
createLevelSetPlatonic(int faceCount,float scale, const Vec3f& center,
    float voxelSize, float halfWidth, InterruptT *interrupt)
{
    // GridType::ValueType is required to be a floating-point scalar.
    static_assert(std::is_floating_point<typename GridType::ValueType>::value,
        "level set grids must have scalar, floating-point value types");

    const math::Transform::Ptr xform = math::Transform::createLinearTransform( voxelSize );

    std::vector<Vec3f> vtx;
    std::vector<Vec3I> tri;
    std::vector<Vec4I> qua;

    if (faceCount == 4) {// Tetrahedron

        vtx.push_back( Vec3f( 0.0f,          1.0f,         0.0f) );
        vtx.push_back( Vec3f(-0.942810297f, -0.333329707f, 0.0f) );
        vtx.push_back( Vec3f( 0.471405149f, -0.333329707f, 0.816497624f) );
        vtx.push_back( Vec3f( 0.471405149f, -0.333329707f, -0.816497624f) );

        tri.push_back( Vec3I(0, 2, 3) );
        tri.push_back( Vec3I(0, 3, 1) );
        tri.push_back( Vec3I(0, 1, 2) );
        tri.push_back( Vec3I(1, 3, 2) );

    } else if (faceCount == 6) {// Cube

        vtx.push_back( Vec3f(-0.5f, -0.5f, -0.5f) );
        vtx.push_back( Vec3f( 0.5f, -0.5f, -0.5f) );
        vtx.push_back( Vec3f( 0.5f, -0.5f,  0.5f) );
        vtx.push_back( Vec3f(-0.5f, -0.5f,  0.5f) );
        vtx.push_back( Vec3f(-0.5f,  0.5f, -0.5f) );
        vtx.push_back( Vec3f( 0.5f,  0.5f, -0.5f) );
        vtx.push_back( Vec3f( 0.5f,  0.5f,  0.5f) );
        vtx.push_back( Vec3f(-0.5f,  0.5f,  0.5f) );

        qua.push_back( Vec4I(1, 0, 4, 5) );
        qua.push_back( Vec4I(2, 1, 5, 6) );
        qua.push_back( Vec4I(3, 2, 6, 7) );
        qua.push_back( Vec4I(0, 3, 7, 4) );
        qua.push_back( Vec4I(2, 3, 0, 1) );
        qua.push_back( Vec4I(5, 4, 7, 6) );

    } else if (faceCount == 8) {// Octahedron

        vtx.push_back( Vec3f( 0.0f, 0.0f, -1.0f) );
        vtx.push_back( Vec3f( 1.0f, 0.0f,  0.0f) );
        vtx.push_back( Vec3f( 0.0f, 0.0f,  1.0f) );
        vtx.push_back( Vec3f(-1.0f, 0.0f,  0.0f) );
        vtx.push_back( Vec3f( 0.0f,-1.0f,  0.0f) );
        vtx.push_back( Vec3f( 0.0f, 1.0f,  0.0f) );

        tri.push_back( Vec3I(0, 4, 3) );
        tri.push_back( Vec3I(0, 1, 4) );
        tri.push_back( Vec3I(1, 2, 4) );
        tri.push_back( Vec3I(2, 3, 4) );
        tri.push_back( Vec3I(0, 3, 5) );
        tri.push_back( Vec3I(0, 5, 1) );
        tri.push_back( Vec3I(1, 5, 2) );
        tri.push_back( Vec3I(2, 5, 3) );

    } else if (faceCount == 12) {// Dodecahedron

        vtx.push_back( Vec3f( 0.354437858f,  0.487842113f, -0.789344311f) );
        vtx.push_back( Vec3f( 0.573492587f, -0.186338872f, -0.78934437f) );
        vtx.push_back( Vec3f( 0.0f,         -0.603005826f, -0.78934443f) );
        vtx.push_back( Vec3f(-0.573492587f, -0.186338872f, -0.78934437f) );
        vtx.push_back( Vec3f(-0.354437858f,  0.487842113f, -0.789344311f) );
        vtx.push_back( Vec3f(-0.573492587f,  0.789345026f, -0.186338797f) );
        vtx.push_back( Vec3f(-0.927930415f, -0.301502913f, -0.186338872f) );
        vtx.push_back( Vec3f( 0.0f,         -0.975683928f, -0.186338902f) );
        vtx.push_back( Vec3f( 0.927930415f, -0.301502913f, -0.186338872f) );
        vtx.push_back( Vec3f( 0.573492587f,  0.789345026f, -0.186338797f) );
        vtx.push_back( Vec3f( 0.0f,          0.975683868f,  0.186338902f) );
        vtx.push_back( Vec3f(-0.927930415f,  0.301502913f,  0.186338872f) );
        vtx.push_back( Vec3f(-0.573492587f, -0.789345026f,  0.186338797f) );
        vtx.push_back( Vec3f( 0.573492587f, -0.789345026f,  0.186338797f) );
        vtx.push_back( Vec3f( 0.927930415f,  0.301502913f,  0.186338872f) );
        vtx.push_back( Vec3f( 0.0f,          0.603005826f,  0.78934443f) );
        vtx.push_back( Vec3f( 0.573492587f,  0.186338872f,  0.78934437f) );
        vtx.push_back( Vec3f( 0.354437858f, -0.487842113f,  0.789344311f) );
        vtx.push_back( Vec3f(-0.354437858f, -0.487842113f,  0.789344311f) );
        vtx.push_back( Vec3f(-0.573492587f,  0.186338872f,  0.78934437f) );

        qua.push_back( Vec4I(0, 1, 2, 3) );
        tri.push_back( Vec3I(0, 3, 4) );
        qua.push_back( Vec4I(0, 4, 5, 10) );
        tri.push_back( Vec3I(0, 10, 9) );
        qua.push_back( Vec4I(0, 9, 14, 8) );
        tri.push_back( Vec3I(0, 8, 1) );
        qua.push_back( Vec4I(1, 8, 13, 7) );
        tri.push_back( Vec3I(1, 7, 2) );
        qua.push_back( Vec4I(2, 7, 12, 6) );
        tri.push_back( Vec3I(2, 6, 3) );
        qua.push_back( Vec4I(3, 6, 11, 5) );
        tri.push_back( Vec3I(3, 5, 4) );
        qua.push_back( Vec4I(5, 11, 19, 15) );
        tri.push_back( Vec3I(5, 15, 10) );
        qua.push_back( Vec4I(6, 12, 18, 19) );
        tri.push_back( Vec3I(6, 19, 11) );
        qua.push_back( Vec4I(7, 13, 17, 18) );
        tri.push_back( Vec3I(7, 18, 12) );
        qua.push_back( Vec4I(8, 14, 16, 17) );
        tri.push_back( Vec3I(8, 17, 13) );
        qua.push_back( Vec4I(9, 10, 15, 16) );
        tri.push_back( Vec3I(9, 16, 14) );
        qua.push_back( Vec4I(15, 19, 18, 17) );
        tri.push_back( Vec3I(15, 17, 16) );

    } else if (faceCount == 20) {// Icosahedron

        vtx.push_back( Vec3f(0.0f, 0.0f, -1.0f) );
        vtx.push_back( Vec3f(0.0f, 0.894427359f, -0.447213143f) );
        vtx.push_back( Vec3f(0.850650847f, 0.276393682f, -0.447213203f) );
        vtx.push_back( Vec3f(0.525731206f, -0.723606944f, -0.447213262f) );
        vtx.push_back( Vec3f(-0.525731206f, -0.723606944f, -0.447213262f) );
        vtx.push_back( Vec3f(-0.850650847f, 0.276393682f, -0.447213203f) );
        vtx.push_back( Vec3f(-0.525731206f, 0.723606944f, 0.447213262f) );
        vtx.push_back( Vec3f(-0.850650847f, -0.276393682f, 0.447213203f) );
        vtx.push_back( Vec3f(0.0f, -0.894427359f, 0.447213143f) );
        vtx.push_back( Vec3f(0.850650847f, -0.276393682f, 0.447213203f) );
        vtx.push_back( Vec3f(0.525731206f, 0.723606944f, 0.447213262f) );
        vtx.push_back( Vec3f(0.0f, 0.0f, 1.0f) );

        tri.push_back( Vec3I( 2,  0,  1) );
        tri.push_back( Vec3I( 3,  0,  2) );
        tri.push_back( Vec3I( 4,  0,  3) );
        tri.push_back( Vec3I( 5,  0,  4) );
        tri.push_back( Vec3I( 1,  0,  5) );
        tri.push_back( Vec3I( 6,  1,  5) );
        tri.push_back( Vec3I( 7,  5,  4) );
        tri.push_back( Vec3I( 8,  4,  3) );
        tri.push_back( Vec3I( 9,  3,  2) );
        tri.push_back( Vec3I(10,  2,  1) );
        tri.push_back( Vec3I(10,  1,  6) );
        tri.push_back( Vec3I( 6,  5,  7) );
        tri.push_back( Vec3I( 7,  4,  8) );
        tri.push_back( Vec3I( 8,  3,  9) );
        tri.push_back( Vec3I( 9,  2, 10) );
        tri.push_back( Vec3I( 6, 11, 10) );
        tri.push_back( Vec3I(10, 11,  9) );
        tri.push_back( Vec3I( 9, 11,  8) );
        tri.push_back( Vec3I( 8, 11,  7) );
        tri.push_back( Vec3I( 7, 11,  6) );

    } else {
        OPENVDB_THROW(RuntimeError, "Invalid face count");
    }

    // Apply scale and translation to all the vertices
    for ( size_t i = 0; i<vtx.size(); ++i ) vtx[i] = scale * vtx[i] + center;

    typename GridType::Ptr grid;

    if (interrupt == nullptr) {
        util::NullInterrupter tmp;
        grid = meshToLevelSet<GridType>(tmp, *xform, vtx, tri, qua, halfWidth);
    } else {
        grid = meshToLevelSet<GridType>(*interrupt, *xform, vtx, tri, qua, halfWidth);
    }

    return grid;
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETPLATONIC_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
