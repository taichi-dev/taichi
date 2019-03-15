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

/// @file Interpolation.h
///
/// Sampler classes such as PointSampler and BoxSampler that are intended for use
/// with tools::GridTransformer should operate in voxel space and must adhere to
/// the interface described in the example below:
/// @code
/// struct MySampler
/// {
///     // Return a short name that can be used to identify this sampler
///     // in error messages and elsewhere.
///     const char* name() { return "mysampler"; }
///
///     // Return the radius of the sampling kernel in voxels, not including
///     // the center voxel.  This is the number of voxels of padding that
///     // are added to all sides of a volume as a result of resampling.
///     int radius() { return 2; }
///
///     // Return true if scaling by a factor smaller than 0.5 (along any axis)
///     // should be handled via a mipmapping-like scheme of successive halvings
///     // of a grid's resolution, until the remaining scale factor is
///     // greater than or equal to 1/2.  Set this to false only when high-quality
///     // scaling is not required.
///     bool mipmap() { return true; }
///
///     // Specify if sampling at a location that is collocated with a grid point
///     // is guaranteed to return the exact value at that grid point.
///     // For most sampling kernels, this should be false.
///     bool consistent() { return false; }
///
///     // Sample the tree at the given coordinates and return the result in val.
///     // Return true if the sampled value is active.
///     template<class TreeT>
///     bool sample(const TreeT& tree, const Vec3R& coord, typename TreeT::ValueType& val);
/// };
/// @endcode

#ifndef OPENVDB_TOOLS_INTERPOLATION_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_INTERPOLATION_HAS_BEEN_INCLUDED

#include <openvdb/version.h> // for OPENVDB_VERSION_NAME
#include <openvdb/Platform.h> // for round()
#include <openvdb/math/Math.h>// for SmoothUnitStep
#include <openvdb/math/Transform.h> // for Transform
#include <openvdb/Grid.h>
#include <openvdb/tree/ValueAccessor.h>
#include <cmath>
#include <type_traits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Provises a unified interface for sampling, i.e. interpolation.
/// @details Order = 0: closest point
///          Order = 1: tri-linear
///          Order = 2: tri-quadratic
///          Staggered: Set to true for MAC grids
template <size_t Order, bool Staggered = false>
struct Sampler
{
    static_assert(Order < 3, "Samplers of order higher than 2 are not supported");
    static const char* name();
    static int radius();
    static bool mipmap();
    static bool consistent();
    static bool staggered();
    static size_t order();

    /// @brief Sample @a inTree at the floating-point index coordinate @a inCoord
    /// and store the result in @a result.
    ///
    /// @return @c true if the sampled value is active.
    template<class TreeT>
    static bool sample(const TreeT& inTree, const Vec3R& inCoord,
                       typename TreeT::ValueType& result);

    /// @brief Sample @a inTree at the floating-point index coordinate @a inCoord.
    ///
    /// @return the reconstructed value
    template<class TreeT>
    static typename TreeT::ValueType sample(const TreeT& inTree, const Vec3R& inCoord);
};

//////////////////////////////////////// Non-Staggered Samplers

// The following samplers operate in voxel space.
// When the samplers are applied to grids holding vector or other non-scalar data,
// the data is assumed to be collocated.  For example, using the BoxSampler on a grid
// with ValueType Vec3f assumes that all three elements in a vector can be assigned
// the same physical location. Consider using the GridSampler below instead.

struct PointSampler
{
    static const char* name() { return "point"; }
    static int radius() { return 0; }
    static bool mipmap() { return false; }
    static bool consistent() { return true; }
    static bool staggered() { return false; }
    static size_t order() { return 0; }

    /// @brief Sample @a inTree at the nearest neighbor to @a inCoord
    /// and store the result in @a result.
    /// @return @c true if the sampled value is active.
    template<class TreeT>
    static bool sample(const TreeT& inTree, const Vec3R& inCoord,
                       typename TreeT::ValueType& result);

    /// @brief Sample @a inTree at the nearest neighbor to @a inCoord
    /// @return the reconstructed value
    template<class TreeT>
    static typename TreeT::ValueType sample(const TreeT& inTree, const Vec3R& inCoord);
};


struct BoxSampler
{
    static const char* name() { return "box"; }
    static int radius() { return 1; }
    static bool mipmap() { return true; }
    static bool consistent() { return true; }
    static bool staggered() { return false; }
    static size_t order() { return 1; }

    /// @brief Trilinearly reconstruct @a inTree at @a inCoord
    /// and store the result in @a result.
    /// @return @c true if any one of the sampled values is active.
    template<class TreeT>
    static bool sample(const TreeT& inTree, const Vec3R& inCoord,
                       typename TreeT::ValueType& result);

    /// @brief Trilinearly reconstruct @a inTree at @a inCoord.
    /// @return the reconstructed value
    template<class TreeT>
    static typename TreeT::ValueType sample(const TreeT& inTree, const Vec3R& inCoord);

    /// @brief Import all eight values from @a inTree to support
    /// tri-linear interpolation.
    template<class ValueT, class TreeT, size_t N>
    static inline void getValues(ValueT (&data)[N][N][N], const TreeT& inTree, Coord ijk);

    /// @brief Import all eight values from @a inTree to support
    /// tri-linear interpolation.
    /// @return @c true if any of the eight values are active
    template<class ValueT, class TreeT, size_t N>
    static inline bool probeValues(ValueT (&data)[N][N][N], const TreeT& inTree, Coord ijk);

    /// @brief Find the minimum and maximum values of the eight cell
    /// values in @ data.
    template<class ValueT, size_t N>
    static inline void extrema(ValueT (&data)[N][N][N], ValueT& vMin, ValueT& vMax);

    /// @return the tri-linear interpolation with the unit cell coordinates @a uvw
    template<class ValueT, size_t N>
    static inline ValueT trilinearInterpolation(ValueT (&data)[N][N][N], const Vec3R& uvw);
};


struct QuadraticSampler
{
    static const char* name() { return "quadratic"; }
    static int radius() { return 1; }
    static bool mipmap() { return true; }
    static bool consistent() { return false; }
    static bool staggered() { return false; }
    static size_t order() { return 2; }

    /// @brief Triquadratically reconstruct @a inTree at @a inCoord
    /// and store the result in @a result.
    /// @return @c true if any one of the sampled values is active.
    template<class TreeT>
    static bool sample(const TreeT& inTree, const Vec3R& inCoord,
                       typename TreeT::ValueType& result);

    /// @brief Triquadratically reconstruct @a inTree at to @a inCoord.
    /// @return the reconstructed value
    template<class TreeT>
    static typename TreeT::ValueType sample(const TreeT& inTree, const Vec3R& inCoord);

    template<class ValueT, size_t N>
    static inline ValueT triquadraticInterpolation(ValueT (&data)[N][N][N], const Vec3R& uvw);
};


//////////////////////////////////////// Staggered Samplers


// The following samplers operate in voxel space and are designed for Vec3
// staggered grid data (e.g., fluid simulations using the Marker-and-Cell approach
// associate elements of the velocity vector with different physical locations:
// the faces of a cube).

struct StaggeredPointSampler
{
    static const char* name() { return "point"; }
    static int radius() { return 0; }
    static bool mipmap() { return false; }
    static bool consistent() { return false; }
    static bool staggered() { return true; }
    static size_t order() { return 0; }

    /// @brief Sample @a inTree at the nearest neighbor to @a inCoord
    /// and store the result in @a result.
    /// @return true if the sampled value is active.
    template<class TreeT>
    static bool sample(const TreeT& inTree, const Vec3R& inCoord,
                       typename TreeT::ValueType& result);

    /// @brief Sample @a inTree at the nearest neighbor to @a inCoord
    /// @return the reconstructed value
    template<class TreeT>
    static typename TreeT::ValueType sample(const TreeT& inTree, const Vec3R& inCoord);
};


struct StaggeredBoxSampler
{
    static const char* name() { return "box"; }
    static int radius() { return 1; }
    static bool mipmap() { return true; }
    static bool consistent() { return false; }
    static bool staggered() { return true; }
    static size_t order() { return 1; }

    /// @brief Trilinearly reconstruct @a inTree at @a inCoord
    /// and store the result in @a result.
    /// @return true if any one of the sampled value is active.
    template<class TreeT>
    static bool sample(const TreeT& inTree, const Vec3R& inCoord,
                       typename TreeT::ValueType& result);

    /// @brief Trilinearly reconstruct @a inTree at @a inCoord.
    /// @return the reconstructed value
    template<class TreeT>
    static typename TreeT::ValueType sample(const TreeT& inTree, const Vec3R& inCoord);
};


struct StaggeredQuadraticSampler
{
    static const char* name() { return "quadratic"; }
    static int radius() { return 1; }
    static bool mipmap() { return true; }
    static bool consistent() { return false; }
    static bool staggered() { return true; }
    static size_t order() { return 2; }

    /// @brief Triquadratically reconstruct @a inTree at @a inCoord
    /// and store the result in @a result.
    /// @return true if any one of the sampled values is active.
    template<class TreeT>
    static bool sample(const TreeT& inTree, const Vec3R& inCoord,
                       typename TreeT::ValueType& result);

    /// @brief Triquadratically reconstruct @a inTree at to @a inCoord.
    /// @return the reconstructed value
    template<class TreeT>
    static typename TreeT::ValueType sample(const TreeT& inTree, const Vec3R& inCoord);
};


//////////////////////////////////////// GridSampler


/// @brief Class that provides the interface for continuous sampling
/// of values in a tree.
///
/// @details Since trees support only discrete voxel sampling, TreeSampler
/// must be used to sample arbitrary continuous points in (world or
/// index) space.
///
/// @warning This implementation of the GridSampler stores a pointer
/// to a Tree for value access. While this is thread-safe it is
/// uncached and hence slow compared to using a
/// ValueAccessor. Consequently it is normally advisable to use the
/// template specialization below that employs a
/// ValueAccessor. However, care must be taken when dealing with
/// multi-threading (see warning below).
template<typename GridOrTreeType, typename SamplerType>
class GridSampler
{
public:
    using Ptr = SharedPtr<GridSampler>;
    using ValueType = typename GridOrTreeType::ValueType;
    using GridType = typename TreeAdapter<GridOrTreeType>::GridType;
    using TreeType = typename TreeAdapter<GridOrTreeType>::TreeType;
    using AccessorType = typename TreeAdapter<GridOrTreeType>::AccessorType;

     /// @param grid  a grid to be sampled
    explicit GridSampler(const GridType& grid)
        : mTree(&(grid.tree())), mTransform(&(grid.transform())) {}

    /// @param tree  a tree to be sampled, or a ValueAccessor for the tree
    /// @param transform is used when sampling world space locations.
    GridSampler(const TreeType& tree, const math::Transform& transform)
        : mTree(&tree), mTransform(&transform) {}

    const math::Transform& transform() const { return *mTransform; }

    /// @brief Sample a point in index space in the grid.
    /// @param x Fractional x-coordinate of point in index-coordinates of grid
    /// @param y Fractional y-coordinate of point in index-coordinates of grid
    /// @param z Fractional z-coordinate of point in index-coordinates of grid
    template<typename RealType>
    ValueType sampleVoxel(const RealType& x, const RealType& y, const RealType& z) const
    {
        return this->isSample(Vec3d(x,y,z));
    }

    /// @brief Sample value in integer index space
    /// @param i Integer x-coordinate in index space
    /// @param j Integer y-coordinate in index space
    /// @param k Integer x-coordinate in index space
    ValueType sampleVoxel(typename Coord::ValueType i,
                          typename Coord::ValueType j,
                          typename Coord::ValueType k) const
    {
        return this->isSample(Coord(i,j,k));
    }

    /// @brief Sample value in integer index space
    /// @param ijk the location in index space
    ValueType isSample(const Coord& ijk) const { return mTree->getValue(ijk); }

    /// @brief Sample in fractional index space
    /// @param ispoint the location in index space
    ValueType isSample(const Vec3d& ispoint) const
    {
        ValueType result = zeroVal<ValueType>();
        SamplerType::sample(*mTree, ispoint, result);
        return result;
    }

    /// @brief Sample in world space
    /// @param wspoint the location in world space
    ValueType wsSample(const Vec3d& wspoint) const
    {
        ValueType result = zeroVal<ValueType>();
        SamplerType::sample(*mTree, mTransform->worldToIndex(wspoint), result);
        return result;
    }

private:
    const TreeType*        mTree;
    const math::Transform* mTransform;
}; // class GridSampler


/// @brief Specialization of GridSampler for construction from a ValueAccessor type
///
/// @note This version should normally be favored over the one above
/// that takes a Grid or Tree. The reason is this version uses a
/// ValueAccessor that performs fast (cached) access where the
/// tree-based flavor performs slower (uncached) access.
///
/// @warning Since this version stores a pointer to an (externally
/// allocated) value accessor it is not threadsafe. Hence each thread
/// should have its own instance of a GridSampler constructed from a
/// local ValueAccessor. Alternatively the Grid/Tree-based GridSampler
/// is threadsafe, but also slower.
template<typename TreeT, typename SamplerType>
class GridSampler<tree::ValueAccessor<TreeT>, SamplerType>
{
public:
    using Ptr = SharedPtr<GridSampler>;
    using ValueType = typename TreeT::ValueType;
    using TreeType = TreeT;
    using GridType = Grid<TreeType>;
    using AccessorType = typename tree::ValueAccessor<TreeT>;

    /// @param acc  a ValueAccessor to be sampled
    /// @param transform is used when sampling world space locations.
    GridSampler(const AccessorType& acc,
                const math::Transform& transform)
        : mAccessor(&acc), mTransform(&transform) {}

     const math::Transform& transform() const { return *mTransform; }

    /// @brief Sample a point in index space in the grid.
    /// @param x Fractional x-coordinate of point in index-coordinates of grid
    /// @param y Fractional y-coordinate of point in index-coordinates of grid
    /// @param z Fractional z-coordinate of point in index-coordinates of grid
    template<typename RealType>
    ValueType sampleVoxel(const RealType& x, const RealType& y, const RealType& z) const
    {
        return this->isSample(Vec3d(x,y,z));
    }

    /// @brief Sample value in integer index space
    /// @param i Integer x-coordinate in index space
    /// @param j Integer y-coordinate in index space
    /// @param k Integer x-coordinate in index space
    ValueType sampleVoxel(typename Coord::ValueType i,
                          typename Coord::ValueType j,
                          typename Coord::ValueType k) const
    {
        return this->isSample(Coord(i,j,k));
    }

    /// @brief Sample value in integer index space
    /// @param ijk the location in index space
    ValueType isSample(const Coord& ijk) const { return mAccessor->getValue(ijk); }

    /// @brief Sample in fractional index space
    /// @param ispoint the location in index space
    ValueType isSample(const Vec3d& ispoint) const
    {
        ValueType result = zeroVal<ValueType>();
        SamplerType::sample(*mAccessor, ispoint, result);
        return result;
    }

    /// @brief Sample in world space
    /// @param wspoint the location in world space
    ValueType wsSample(const Vec3d& wspoint) const
    {
        ValueType result = zeroVal<ValueType>();
        SamplerType::sample(*mAccessor, mTransform->worldToIndex(wspoint), result);
        return result;
    }

private:
    const AccessorType*    mAccessor;//not thread-safe!
    const math::Transform* mTransform;
};//Specialization of GridSampler


//////////////////////////////////////// DualGridSampler


/// @brief This is a simple convenience class that allows for sampling
/// from a source grid into the index space of a target grid. At
/// construction the source and target grids are checked for alignment
/// which potentially renders interpolation unnecessary. Else
/// interpolation is performed according to the templated Sampler
/// type.
///
/// @warning For performance reasons the check for alignment of the
/// two grids is only performed at construction time!
template<typename GridOrTreeT,
         typename SamplerT>
class DualGridSampler
{
public:
    using ValueType = typename GridOrTreeT::ValueType;
    using GridType = typename TreeAdapter<GridOrTreeT>::GridType;
    using TreeType = typename TreeAdapter<GridOrTreeT>::TreeType;
    using AccessorType = typename TreeAdapter<GridType>::AccessorType;

    /// @brief Grid and transform constructor.
    /// @param sourceGrid Source grid.
    /// @param targetXform Transform of the target grid.
    DualGridSampler(const GridType& sourceGrid,
                    const math::Transform& targetXform)
        : mSourceTree(&(sourceGrid.tree()))
        , mSourceXform(&(sourceGrid.transform()))
        , mTargetXform(&targetXform)
        , mAligned(targetXform == *mSourceXform)
    {
    }
    /// @brief Tree and transform constructor.
    /// @param sourceTree Source tree.
    /// @param sourceXform Transform of the source grid.
    /// @param targetXform Transform of the target grid.
    DualGridSampler(const TreeType& sourceTree,
                    const math::Transform& sourceXform,
                    const math::Transform& targetXform)
        : mSourceTree(&sourceTree)
        , mSourceXform(&sourceXform)
        , mTargetXform(&targetXform)
        , mAligned(targetXform == sourceXform)
    {
    }
    /// @brief Return the value of the source grid at the index
    /// coordinates, ijk, relative to the target grid (or its tranform).
    inline ValueType operator()(const Coord& ijk) const
    {
        if (mAligned) return mSourceTree->getValue(ijk);
        const Vec3R world = mTargetXform->indexToWorld(ijk);
        return SamplerT::sample(*mSourceTree, mSourceXform->worldToIndex(world));
    }
    /// @brief Return true if the two grids are aligned.
    inline bool isAligned() const { return mAligned; }
private:
    const TreeType*        mSourceTree;
    const math::Transform* mSourceXform;
    const math::Transform* mTargetXform;
    const bool             mAligned;
};// DualGridSampler

/// @brief Specialization of DualGridSampler for construction from a ValueAccessor type.
template<typename TreeT,
         typename SamplerT>
class DualGridSampler<tree::ValueAccessor<TreeT>, SamplerT>
{
    public:
    using ValueType = typename TreeT::ValueType;
    using TreeType = TreeT;
    using GridType = Grid<TreeType>;
    using AccessorType = typename tree::ValueAccessor<TreeT>;

    /// @brief ValueAccessor and transform constructor.
    /// @param sourceAccessor ValueAccessor into the source grid.
    /// @param sourceXform Transform for the source grid.
    /// @param targetXform Transform for the target grid.
    DualGridSampler(const AccessorType& sourceAccessor,
                    const math::Transform& sourceXform,
                    const math::Transform& targetXform)
        : mSourceAcc(&sourceAccessor)
        , mSourceXform(&sourceXform)
        , mTargetXform(&targetXform)
        , mAligned(targetXform == sourceXform)
    {
    }
    /// @brief Return the value of the source grid at the index
    /// coordinates, ijk, relative to the target grid.
    inline ValueType operator()(const Coord& ijk) const
    {
        if (mAligned) return mSourceAcc->getValue(ijk);
        const Vec3R world = mTargetXform->indexToWorld(ijk);
        return SamplerT::sample(*mSourceAcc, mSourceXform->worldToIndex(world));
    }
    /// @brief Return true if the two grids are aligned.
    inline bool isAligned() const { return mAligned; }
private:
    const AccessorType*    mSourceAcc;
    const math::Transform* mSourceXform;
    const math::Transform* mTargetXform;
    const bool             mAligned;
};//Specialization of DualGridSampler

//////////////////////////////////////// AlphaMask


// Class to derive the normalized alpha mask
template <typename GridT,
          typename MaskT,
          typename SamplerT = tools::BoxSampler,
          typename FloatT = float>
class AlphaMask
{
public:
    static_assert(std::is_floating_point<FloatT>::value,
        "AlphaMask requires a floating-point value type");
    using GridType = GridT;
    using MaskType = MaskT;
    using SamlerType = SamplerT;
    using FloatType = FloatT;

    AlphaMask(const GridT& grid, const MaskT& mask, FloatT min, FloatT max, bool invert)
        : mAcc(mask.tree())
        , mSampler(mAcc, mask.transform() , grid.transform())
        , mMin(min)
        , mInvNorm(1/(max-min))
        , mInvert(invert)
    {
        assert(min < max);
    }

    inline bool operator()(const Coord& xyz, FloatT& a, FloatT& b) const
    {
        a = math::SmoothUnitStep( (mSampler(xyz) - mMin) * mInvNorm );//smooth mapping to 0->1
        b = 1 - a;
        if (mInvert) std::swap(a,b);
        return a>0;
    }

protected:
    using AccT = typename MaskType::ConstAccessor;
    AccT mAcc;
    tools::DualGridSampler<AccT, SamplerT> mSampler;
    const FloatT mMin, mInvNorm;
    const bool mInvert;
};// AlphaMask

////////////////////////////////////////

namespace local_util {

inline Vec3i
floorVec3(const Vec3R& v)
{
    return Vec3i(int(std::floor(v(0))), int(std::floor(v(1))), int(std::floor(v(2))));
}


inline Vec3i
ceilVec3(const Vec3R& v)
{
    return Vec3i(int(std::ceil(v(0))), int(std::ceil(v(1))), int(std::ceil(v(2))));
}


inline Vec3i
roundVec3(const Vec3R& v)
{
    return Vec3i(int(::round(v(0))), int(::round(v(1))), int(::round(v(2))));
}

} // namespace local_util


//////////////////////////////////////// PointSampler


template<class TreeT>
inline bool
PointSampler::sample(const TreeT& inTree, const Vec3R& inCoord,
                     typename TreeT::ValueType& result)
{
    return inTree.probeValue(Coord(local_util::roundVec3(inCoord)), result);
}

template<class TreeT>
inline typename TreeT::ValueType
PointSampler::sample(const TreeT& inTree, const Vec3R& inCoord)
{
    return inTree.getValue(Coord(local_util::roundVec3(inCoord)));
}


//////////////////////////////////////// BoxSampler

template<class ValueT, class TreeT, size_t N>
inline void
BoxSampler::getValues(ValueT (&data)[N][N][N], const TreeT& inTree, Coord ijk)
{
    data[0][0][0] = inTree.getValue(ijk); // i, j, k

    ijk[2] += 1;
    data[0][0][1] = inTree.getValue(ijk); // i, j, k + 1

    ijk[1] += 1;
    data[0][1][1] = inTree.getValue(ijk); // i, j+1, k + 1

    ijk[2] -= 1;
    data[0][1][0] = inTree.getValue(ijk); // i, j+1, k

    ijk[0] += 1;
    ijk[1] -= 1;
    data[1][0][0] = inTree.getValue(ijk); // i+1, j, k

    ijk[2] += 1;
    data[1][0][1] = inTree.getValue(ijk); // i+1, j, k + 1

    ijk[1] += 1;
    data[1][1][1] = inTree.getValue(ijk); // i+1, j+1, k + 1

    ijk[2] -= 1;
    data[1][1][0] = inTree.getValue(ijk); // i+1, j+1, k
}

template<class ValueT, class TreeT, size_t N>
inline bool
BoxSampler::probeValues(ValueT (&data)[N][N][N], const TreeT& inTree, Coord ijk)
{
    bool hasActiveValues = false;
    hasActiveValues |= inTree.probeValue(ijk, data[0][0][0]); // i, j, k

    ijk[2] += 1;
    hasActiveValues |= inTree.probeValue(ijk, data[0][0][1]); // i, j, k + 1

    ijk[1] += 1;
    hasActiveValues |= inTree.probeValue(ijk, data[0][1][1]); // i, j+1, k + 1

    ijk[2] -= 1;
    hasActiveValues |= inTree.probeValue(ijk, data[0][1][0]); // i, j+1, k

    ijk[0] += 1;
    ijk[1] -= 1;
    hasActiveValues |= inTree.probeValue(ijk, data[1][0][0]); // i+1, j, k

    ijk[2] += 1;
    hasActiveValues |= inTree.probeValue(ijk, data[1][0][1]); // i+1, j, k + 1

    ijk[1] += 1;
    hasActiveValues |= inTree.probeValue(ijk, data[1][1][1]); // i+1, j+1, k + 1

    ijk[2] -= 1;
    hasActiveValues |= inTree.probeValue(ijk, data[1][1][0]); // i+1, j+1, k

    return hasActiveValues;
}

template<class ValueT, size_t N>
inline void
BoxSampler::extrema(ValueT (&data)[N][N][N], ValueT& vMin, ValueT &vMax)
{
    vMin = vMax = data[0][0][0];
    vMin = math::Min(vMin, data[0][0][1]);
    vMax = math::Max(vMax, data[0][0][1]);
    vMin = math::Min(vMin, data[0][1][0]);
    vMax = math::Max(vMax, data[0][1][0]);
    vMin = math::Min(vMin, data[0][1][1]);
    vMax = math::Max(vMax, data[0][1][1]);
    vMin = math::Min(vMin, data[1][0][0]);
    vMax = math::Max(vMax, data[1][0][0]);
    vMin = math::Min(vMin, data[1][0][1]);
    vMax = math::Max(vMax, data[1][0][1]);
    vMin = math::Min(vMin, data[1][1][0]);
    vMax = math::Max(vMax, data[1][1][0]);
    vMin = math::Min(vMin, data[1][1][1]);
    vMax = math::Max(vMax, data[1][1][1]);
}


template<class ValueT, size_t N>
inline ValueT
BoxSampler::trilinearInterpolation(ValueT (&data)[N][N][N], const Vec3R& uvw)
{
    // Trilinear interpolation:
    // The eight surrounding latice values are used to construct the result. \n
    // result(x,y,z) =
    //     v000 (1-x)(1-y)(1-z) + v001 (1-x)(1-y)z + v010 (1-x)y(1-z) + v011 (1-x)yz
    //   + v100 x(1-y)(1-z)     + v101 x(1-y)z     + v110 xy(1-z)     + v111 xyz

    ValueT resultA, resultB;

    resultA = data[0][0][0] + ValueT((data[0][0][1] - data[0][0][0]) * uvw[2]);
    resultB = data[0][1][0] + ValueT((data[0][1][1] - data[0][1][0]) * uvw[2]);
    ValueT result1 = resultA + ValueT((resultB-resultA) * uvw[1]);

    resultA = data[1][0][0] + ValueT((data[1][0][1] - data[1][0][0]) * uvw[2]);
    resultB = data[1][1][0] + ValueT((data[1][1][1] - data[1][1][0]) * uvw[2]);
    ValueT result2 = resultA + ValueT((resultB - resultA) * uvw[1]);

    return result1 + ValueT(uvw[0] * (result2 - result1));
}


template<class TreeT>
inline bool
BoxSampler::sample(const TreeT& inTree, const Vec3R& inCoord,
                   typename TreeT::ValueType& result)
{
    using ValueT = typename TreeT::ValueType;

    const Vec3i inIdx = local_util::floorVec3(inCoord);
    const Vec3R uvw = inCoord - inIdx;

    // Retrieve the values of the eight voxels surrounding the
    // fractional source coordinates.
    ValueT data[2][2][2];

    const bool hasActiveValues = BoxSampler::probeValues(data, inTree, Coord(inIdx));

    result = BoxSampler::trilinearInterpolation(data, uvw);

    return hasActiveValues;
}


template<class TreeT>
inline typename TreeT::ValueType
BoxSampler::sample(const TreeT& inTree, const Vec3R& inCoord)
{
    using ValueT = typename TreeT::ValueType;

    const Vec3i inIdx = local_util::floorVec3(inCoord);
    const Vec3R uvw = inCoord - inIdx;

    // Retrieve the values of the eight voxels surrounding the
    // fractional source coordinates.
    ValueT data[2][2][2];

    BoxSampler::getValues(data, inTree, Coord(inIdx));

    return BoxSampler::trilinearInterpolation(data, uvw);
}


//////////////////////////////////////// QuadraticSampler

template<class ValueT, size_t N>
inline ValueT
QuadraticSampler::triquadraticInterpolation(ValueT (&data)[N][N][N], const Vec3R& uvw)
{
    /// @todo For vector types, interpolate over each component independently.
    ValueT vx[3];
    for (int dx = 0; dx < 3; ++dx) {
        ValueT vy[3];
        for (int dy = 0; dy < 3; ++dy) {
            // Fit a parabola to three contiguous samples in z
            // (at z=-1, z=0 and z=1), then evaluate the parabola at z',
            // where z' is the fractional part of inCoord.z, i.e.,
            // inCoord.z - inIdx.z.  The coefficients come from solving
            //
            // | (-1)^2  -1   1 || a |   | v0 |
            // |    0     0   1 || b | = | v1 |
            // |   1^2    1   1 || c |   | v2 |
            //
            // for a, b and c.
            const ValueT* vz = &data[dx][dy][0];
            const ValueT
                az = static_cast<ValueT>(0.5 * (vz[0] + vz[2]) - vz[1]),
                bz = static_cast<ValueT>(0.5 * (vz[2] - vz[0])),
                cz = static_cast<ValueT>(vz[1]);
            vy[dy] = static_cast<ValueT>(uvw.z() * (uvw.z() * az + bz) + cz);
        }//loop over y
        // Fit a parabola to three interpolated samples in y, then
        // evaluate the parabola at y', where y' is the fractional
        // part of inCoord.y.
        const ValueT
            ay = static_cast<ValueT>(0.5 * (vy[0] + vy[2]) - vy[1]),
            by = static_cast<ValueT>(0.5 * (vy[2] - vy[0])),
            cy = static_cast<ValueT>(vy[1]);
        vx[dx] = static_cast<ValueT>(uvw.y() * (uvw.y() * ay + by) + cy);
    }//loop over x
    // Fit a parabola to three interpolated samples in x, then
    // evaluate the parabola at the fractional part of inCoord.x.
    const ValueT
        ax = static_cast<ValueT>(0.5 * (vx[0] + vx[2]) - vx[1]),
        bx = static_cast<ValueT>(0.5 * (vx[2] - vx[0])),
        cx = static_cast<ValueT>(vx[1]);
    return static_cast<ValueT>(uvw.x() * (uvw.x() * ax + bx) + cx);
}

template<class TreeT>
inline bool
QuadraticSampler::sample(const TreeT& inTree, const Vec3R& inCoord,
    typename TreeT::ValueType& result)
{
    using ValueT = typename TreeT::ValueType;

    const Vec3i inIdx = local_util::floorVec3(inCoord), inLoIdx = inIdx - Vec3i(1, 1, 1);
    const Vec3R uvw = inCoord - inIdx;

    // Retrieve the values of the 27 voxels surrounding the
    // fractional source coordinates.
    bool active = false;
    ValueT data[3][3][3];
    for (int dx = 0, ix = inLoIdx.x(); dx < 3; ++dx, ++ix) {
        for (int dy = 0, iy = inLoIdx.y(); dy < 3; ++dy, ++iy) {
            for (int dz = 0, iz = inLoIdx.z(); dz < 3; ++dz, ++iz) {
                if (inTree.probeValue(Coord(ix, iy, iz), data[dx][dy][dz])) active = true;
            }
        }
    }

    result = QuadraticSampler::triquadraticInterpolation(data, uvw);

    return active;
}

template<class TreeT>
inline typename TreeT::ValueType
QuadraticSampler::sample(const TreeT& inTree, const Vec3R& inCoord)
{
    using ValueT = typename TreeT::ValueType;

    const Vec3i inIdx = local_util::floorVec3(inCoord), inLoIdx = inIdx - Vec3i(1, 1, 1);
    const Vec3R uvw = inCoord - inIdx;

    // Retrieve the values of the 27 voxels surrounding the
    // fractional source coordinates.
    ValueT data[3][3][3];
    for (int dx = 0, ix = inLoIdx.x(); dx < 3; ++dx, ++ix) {
        for (int dy = 0, iy = inLoIdx.y(); dy < 3; ++dy, ++iy) {
            for (int dz = 0, iz = inLoIdx.z(); dz < 3; ++dz, ++iz) {
                data[dx][dy][dz] = inTree.getValue(Coord(ix, iy, iz));
            }
        }
    }

    return QuadraticSampler::triquadraticInterpolation(data, uvw);
}


//////////////////////////////////////// StaggeredPointSampler


template<class TreeT>
inline bool
StaggeredPointSampler::sample(const TreeT& inTree, const Vec3R& inCoord,
                              typename TreeT::ValueType& result)
{
    using ValueType = typename TreeT::ValueType;

    ValueType tempX, tempY, tempZ;
    bool active = false;

    active = PointSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.5, 0, 0), tempX) || active;
    active = PointSampler::sample<TreeT>(inTree, inCoord + Vec3R(0, 0.5, 0), tempY) || active;
    active = PointSampler::sample<TreeT>(inTree, inCoord + Vec3R(0, 0, 0.5), tempZ) || active;

    result.x() = tempX.x();
    result.y() = tempY.y();
    result.z() = tempZ.z();

    return active;
}

template<class TreeT>
inline typename TreeT::ValueType
StaggeredPointSampler::sample(const TreeT& inTree, const Vec3R& inCoord)
{
    using ValueT = typename TreeT::ValueType;

    const ValueT tempX = PointSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.5, 0.0, 0.0));
    const ValueT tempY = PointSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.0, 0.5, 0.0));
    const ValueT tempZ = PointSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.0, 0.0, 0.5));

    return ValueT(tempX.x(), tempY.y(), tempZ.z());
}


//////////////////////////////////////// StaggeredBoxSampler


template<class TreeT>
inline bool
StaggeredBoxSampler::sample(const TreeT& inTree, const Vec3R& inCoord,
                            typename TreeT::ValueType& result)
{
    using ValueType = typename TreeT::ValueType;

    ValueType tempX, tempY, tempZ;
    tempX = tempY = tempZ = zeroVal<ValueType>();
    bool active = false;

    active = BoxSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.5, 0, 0), tempX) || active;
    active = BoxSampler::sample<TreeT>(inTree, inCoord + Vec3R(0, 0.5, 0), tempY) || active;
    active = BoxSampler::sample<TreeT>(inTree, inCoord + Vec3R(0, 0, 0.5), tempZ) || active;

    result.x() = tempX.x();
    result.y() = tempY.y();
    result.z() = tempZ.z();

    return active;
}

template<class TreeT>
inline typename TreeT::ValueType
StaggeredBoxSampler::sample(const TreeT& inTree, const Vec3R& inCoord)
{
    using ValueT = typename TreeT::ValueType;

    const ValueT tempX = BoxSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.5, 0.0, 0.0));
    const ValueT tempY = BoxSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.0, 0.5, 0.0));
    const ValueT tempZ = BoxSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.0, 0.0, 0.5));

    return ValueT(tempX.x(), tempY.y(), tempZ.z());
}


//////////////////////////////////////// StaggeredQuadraticSampler


template<class TreeT>
inline bool
StaggeredQuadraticSampler::sample(const TreeT& inTree, const Vec3R& inCoord,
    typename TreeT::ValueType& result)
{
    using ValueType = typename TreeT::ValueType;

    ValueType tempX, tempY, tempZ;
    bool active = false;

    active = QuadraticSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.5, 0, 0), tempX) || active;
    active = QuadraticSampler::sample<TreeT>(inTree, inCoord + Vec3R(0, 0.5, 0), tempY) || active;
    active = QuadraticSampler::sample<TreeT>(inTree, inCoord + Vec3R(0, 0, 0.5), tempZ) || active;

    result.x() = tempX.x();
    result.y() = tempY.y();
    result.z() = tempZ.z();

    return active;
}

template<class TreeT>
inline typename TreeT::ValueType
StaggeredQuadraticSampler::sample(const TreeT& inTree, const Vec3R& inCoord)
{
    using ValueT = typename TreeT::ValueType;

    const ValueT tempX = QuadraticSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.5, 0.0, 0.0));
    const ValueT tempY = QuadraticSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.0, 0.5, 0.0));
    const ValueT tempZ = QuadraticSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.0, 0.0, 0.5));

    return ValueT(tempX.x(), tempY.y(), tempZ.z());
}

//////////////////////////////////////// Sampler

template <>
struct Sampler<0, false> : public PointSampler {};

template <>
struct Sampler<1, false> : public BoxSampler {};

template <>
struct Sampler<2, false> : public QuadraticSampler {};

template <>
struct Sampler<0, true> : public StaggeredPointSampler {};

template <>
struct Sampler<1, true> : public StaggeredBoxSampler {};

template <>
struct Sampler<2, true> : public StaggeredQuadraticSampler {};

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_INTERPOLATION_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
