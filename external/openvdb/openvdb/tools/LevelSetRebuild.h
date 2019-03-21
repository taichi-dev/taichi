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

#ifndef OPENVDB_TOOLS_LEVELSETREBUILD_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETREBUILD_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include <openvdb/Exceptions.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/util/Util.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <type_traits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief Return a new grid of type @c GridType that contains a narrow-band level set
/// representation of an isosurface of a given grid.
///
/// @param grid       a scalar, floating-point grid with one or more disjoint,
///                   closed isosurfaces at the given @a isovalue
/// @param isovalue   the isovalue that defines the implicit surface (defaults to zero,
///                   which is typical if the input grid is already a level set or a SDF).
/// @param halfWidth  half the width of the narrow band, in voxel units
///                   (defaults to 3 voxels, which is required for some level set operations)
/// @param xform      optional transform for the output grid
///                   (if not provided, the transform of the input @a grid will be matched)
///
/// @throw TypeError if @a grid is not scalar or not floating-point
///
/// @note If the input grid contains overlapping isosurfaces, interior edges will be lost.
template<class GridType>
inline typename GridType::Ptr
levelSetRebuild(const GridType& grid, float isovalue = 0,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH), const math::Transform* xform = nullptr);


/// @brief Return a new grid of type @c GridType that contains a narrow-band level set
/// representation of an isosurface of a given grid.
///
/// @param grid         a scalar, floating-point grid with one or more disjoint,
///                     closed isosurfaces at the given @a isovalue
/// @param isovalue     the isovalue that defines the implicit surface
/// @param exBandWidth  the exterior narrow-band width in voxel units
/// @param inBandWidth  the interior narrow-band width in voxel units
/// @param xform        optional transform for the output grid
///                     (if not provided, the transform of the input @a grid will be matched)
///
/// @throw TypeError if @a grid is not scalar or not floating-point
///
/// @note If the input grid contains overlapping isosurfaces, interior edges will be lost.
template<class GridType>
inline typename GridType::Ptr
levelSetRebuild(const GridType& grid, float isovalue, float exBandWidth, float inBandWidth,
    const math::Transform* xform = nullptr);


/// @brief Return a new grid of type @c GridType that contains a narrow-band level set
/// representation of an isosurface of a given grid.
///
/// @param grid         a scalar, floating-point grid with one or more disjoint,
///                     closed isosurfaces at the given @a isovalue
/// @param isovalue     the isovalue that defines the implicit surface
/// @param exBandWidth  the exterior narrow-band width in voxel units
/// @param inBandWidth  the interior narrow-band width in voxel units
/// @param xform        optional transform for the output grid
///                     (if not provided, the transform of the input @a grid will be matched)
/// @param interrupter  optional interrupter object
///
/// @throw TypeError if @a grid is not scalar or not floating-point
///
/// @note If the input grid contains overlapping isosurfaces, interior edges will be lost.
template<class GridType, typename InterruptT>
inline typename GridType::Ptr
levelSetRebuild(const GridType& grid, float isovalue, float exBandWidth, float inBandWidth,
    const math::Transform* xform = nullptr, InterruptT* interrupter = nullptr);


////////////////////////////////////////


// Internal utility objects and implementation details

namespace internal {

class PointListTransform
{
public:
    PointListTransform(const PointList& pointsIn, std::vector<Vec3s>& pointsOut,
        const math::Transform& xform)
        : mPointsIn(pointsIn)
        , mPointsOut(&pointsOut)
        , mXform(xform)
    {
    }

    void runParallel()
    {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mPointsOut->size()), *this);
    }

    void runSerial()
    {
        (*this)(tbb::blocked_range<size_t>(0, mPointsOut->size()));
    }

    inline void operator()(const tbb::blocked_range<size_t>& range) const
    {
        for (size_t n = range.begin(); n < range.end(); ++n) {
            (*mPointsOut)[n] = Vec3s(mXform.worldToIndex(mPointsIn[n]));
        }
    }

private:
    const PointList& mPointsIn;
    std::vector<Vec3s> * const mPointsOut;
    const math::Transform& mXform;
};


class PrimCpy
{
public:
    PrimCpy(const PolygonPoolList& primsIn, const std::vector<size_t>& indexList,
        std::vector<Vec4I>& primsOut)
        : mPrimsIn(primsIn)
        , mIndexList(indexList)
        , mPrimsOut(&primsOut)
    {
    }

    void runParallel()
    {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mIndexList.size()), *this);
    }

    void runSerial()
    {
        (*this)(tbb::blocked_range<size_t>(0, mIndexList.size()));
    }

    inline void operator()(const tbb::blocked_range<size_t>& range) const
    {
        openvdb::Vec4I quad;
        quad[3] = openvdb::util::INVALID_IDX;
        std::vector<Vec4I>& primsOut = *mPrimsOut;

        for (size_t n = range.begin(); n < range.end(); ++n) {
            size_t index = mIndexList[n];
            PolygonPool& polygons = mPrimsIn[n];

            // Copy quads
            for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {
                primsOut[index++] = polygons.quad(i);
            }
            polygons.clearQuads();

            // Copy triangles (adaptive mesh)
            for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {
                const openvdb::Vec3I& triangle = polygons.triangle(i);
                quad[0] = triangle[0];
                quad[1] = triangle[1];
                quad[2] = triangle[2];
                primsOut[index++] = quad;
            }

            polygons.clearTriangles();
        }
    }

private:
    const PolygonPoolList& mPrimsIn;
    const std::vector<size_t>& mIndexList;
    std::vector<Vec4I> * const mPrimsOut;
};

} // namespace internal


////////////////////////////////////////


//{
/// @cond OPENVDB_LEVEL_SET_REBUILD_INTERNAL

/// The normal entry points for level set rebuild are the levelSetRebuild() functions.
/// doLevelSetRebuild() is mainly for internal use, but when the isovalue and half band
/// widths are given in ValueType units (for example, if they are queried from
/// a grid), it might be more convenient to call this function directly.
///
/// @internal This overload is enabled only for grids with a scalar, floating-point ValueType.
template<class GridType, typename InterruptT>
inline typename std::enable_if<
    std::is_floating_point<typename GridType::ValueType>::value, typename GridType::Ptr>::type
doLevelSetRebuild(const GridType& grid, typename GridType::ValueType iso,
    typename GridType::ValueType exWidth, typename GridType::ValueType inWidth,
    const math::Transform* xform, InterruptT* interrupter)
{
    const float
        isovalue = float(iso),
        exBandWidth = float(exWidth),
        inBandWidth = float(inWidth);

    tools::VolumeToMesh mesher(isovalue);
    mesher(grid);

    math::Transform::Ptr transform = (xform != nullptr) ? xform->copy() : grid.transform().copy();

    std::vector<Vec3s> points(mesher.pointListSize());

    { // Copy and transform (required for MeshToVolume) points to grid space.
        internal::PointListTransform ptnXForm(mesher.pointList(), points, *transform);
        ptnXForm.runParallel();
        mesher.pointList().reset(nullptr);
    }

    std::vector<Vec4I> primitives;

    { // Copy primitives.
        PolygonPoolList& polygonPoolList = mesher.polygonPoolList();

        size_t numPrimitives = 0;
        std::vector<size_t> indexlist(mesher.polygonPoolListSize());

        for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
            const openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
            indexlist[n] = numPrimitives;
            numPrimitives += polygons.numQuads();
            numPrimitives += polygons.numTriangles();
        }

        primitives.resize(numPrimitives);
        internal::PrimCpy primCpy(polygonPoolList, indexlist, primitives);
        primCpy.runParallel();
    }

    QuadAndTriangleDataAdapter<Vec3s, Vec4I> mesh(points, primitives);

    if (interrupter) {
        return meshToVolume<GridType>(*interrupter, mesh, *transform, exBandWidth, inBandWidth,
            DISABLE_RENORMALIZATION, nullptr);
    }

    return meshToVolume<GridType>(mesh, *transform, exBandWidth, inBandWidth,
        DISABLE_RENORMALIZATION, nullptr);
}


/// @internal This overload is enabled only for grids that do not have a scalar,
/// floating-point ValueType.
template<class GridType, typename InterruptT>
inline typename std::enable_if<
    !std::is_floating_point<typename GridType::ValueType>::value, typename GridType::Ptr>::type
doLevelSetRebuild(const GridType&, typename GridType::ValueType /*isovalue*/,
    typename GridType::ValueType /*exWidth*/, typename GridType::ValueType /*inWidth*/,
    const math::Transform*, InterruptT*)
{
    OPENVDB_THROW(TypeError,
        "level set rebuild is supported only for scalar, floating-point grids");
}

/// @endcond
//}


////////////////////////////////////////


template<class GridType, typename InterruptT>
inline typename GridType::Ptr
levelSetRebuild(const GridType& grid, float iso, float exWidth, float inWidth,
    const math::Transform* xform, InterruptT* interrupter)
{
    using ValueT = typename GridType::ValueType;
    ValueT
        isovalue(zeroVal<ValueT>() + ValueT(iso)),
        exBandWidth(zeroVal<ValueT>() + ValueT(exWidth)),
        inBandWidth(zeroVal<ValueT>() + ValueT(inWidth));

    return doLevelSetRebuild(grid, isovalue, exBandWidth, inBandWidth, xform, interrupter);
}


template<class GridType>
inline typename GridType::Ptr
levelSetRebuild(const GridType& grid, float iso, float exWidth, float inWidth,
    const math::Transform* xform)
{
    using ValueT = typename GridType::ValueType;
    ValueT
        isovalue(zeroVal<ValueT>() + ValueT(iso)),
        exBandWidth(zeroVal<ValueT>() + ValueT(exWidth)),
        inBandWidth(zeroVal<ValueT>() + ValueT(inWidth));

    return doLevelSetRebuild<GridType, util::NullInterrupter>(
        grid, isovalue, exBandWidth, inBandWidth, xform, nullptr);
}


template<class GridType>
inline typename GridType::Ptr
levelSetRebuild(const GridType& grid, float iso, float halfVal, const math::Transform* xform)
{
    using ValueT = typename GridType::ValueType;
    ValueT
        isovalue(zeroVal<ValueT>() + ValueT(iso)),
        halfWidth(zeroVal<ValueT>() + ValueT(halfVal));

    return doLevelSetRebuild<GridType, util::NullInterrupter>(
        grid, isovalue, halfWidth, halfWidth, xform, nullptr);
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETREBUILD_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
