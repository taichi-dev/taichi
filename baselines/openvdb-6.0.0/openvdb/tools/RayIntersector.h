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
///
/// @file RayIntersector.h
///
/// @author Ken Museth
///
/// @brief Accelerated intersection of a ray with a narrow-band level
/// set or a generic (e.g. density) volume. This will of course be
/// useful for respectively surface and volume rendering.
///
/// @details This file defines two main classes,
/// LevelSetRayIntersector and VolumeRayIntersector, as well as the
/// three support classes LevelSetHDDA, VolumeHDDA and LinearSearchImpl.
/// The LevelSetRayIntersector is templated on the LinearSearchImpl class
/// and calls instances of the LevelSetHDDA class. The reason to split
/// level set ray intersection into three classes is twofold. First
/// LevelSetRayIntersector defines the public API for client code and
/// LinearSearchImpl defines the actual algorithm used for the
/// ray level-set intersection. In other words this design will allow
/// for the public API to be fixed while the intersection algorithm
/// can change without resolving to (slow) virtual methods. Second,
/// LevelSetHDDA, which implements a hierarchical Differential Digital
/// Analyzer, relies on partial template specialization, so it has to
/// be a standalone class (as opposed to a member class of
/// LevelSetRayIntersector). The VolumeRayIntersector is conceptually
/// much simpler than the LevelSetRayIntersector, and hence it only
/// depends on VolumeHDDA that implements the hierarchical
/// Differential Digital Analyzer.


#ifndef OPENVDB_TOOLS_RAYINTERSECTOR_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_RAYINTERSECTOR_HAS_BEEN_INCLUDED

#include <openvdb/math/DDA.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Ray.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include "Morphology.h"
#include <iostream>
#include <type_traits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

// Helper class that implements the actual search of the zero-crossing
// of the level set along the direction of a ray. This particular
// implementation uses iterative linear search.
template<typename GridT, int Iterations = 0, typename RealT = double>
class LinearSearchImpl;


///////////////////////////////////// LevelSetRayIntersector /////////////////////////////////////


/// @brief This class provides the public API for intersecting a ray
/// with a narrow-band level set.
///
/// @details It wraps a SearchImplT with a simple public API and
/// performs the actual hierarchical tree node and voxel traversal.
///
/// @warning Use the (default) copy-constructor to make sure each
/// computational thread has their own instance of this class. This is
/// important since the SearchImplT contains a ValueAccessor that is
/// not thread-safe. However copying is very efficient.
///
/// @see tools/RayTracer.h for examples of intended usage.
///
/// @todo Add TrilinearSearchImpl, as an alternative to LinearSearchImpl,
/// that performs analytical 3D trilinear intersection tests, i.e., solves
/// cubic equations. This is slower but also more accurate than the 1D
/// linear interpolation in LinearSearchImpl.
template<typename GridT,
         typename SearchImplT = LinearSearchImpl<GridT>,
         int NodeLevel = GridT::TreeType::RootNodeType::ChildNodeType::LEVEL,
         typename RayT = math::Ray<Real> >
class LevelSetRayIntersector
{
public:
    using GridType = GridT;
    using RayType = RayT;
    using RealType = typename RayT::RealType;
    using Vec3Type = typename RayT::Vec3T;
    using ValueT = typename GridT::ValueType;
    using TreeT = typename GridT::TreeType;

    static_assert(NodeLevel >= -1 && NodeLevel < int(TreeT::DEPTH)-1, "NodeLevel out of range");
    static_assert(std::is_floating_point<ValueT>::value,
        "level set grids must have scalar, floating-point value types");

    /// @brief Constructor
    /// @param grid level set grid to intersect rays against.
    /// @param isoValue optional iso-value for the ray-intersection.
    LevelSetRayIntersector(const GridT& grid, const ValueT& isoValue = zeroVal<ValueT>())
        : mTester(grid, isoValue)
    {
        if (!grid.hasUniformVoxels() ) {
            OPENVDB_THROW(RuntimeError,
                          "LevelSetRayIntersector only supports uniform voxels!");
        }
        if (grid.getGridClass() != GRID_LEVEL_SET) {
            OPENVDB_THROW(RuntimeError,
                          "LevelSetRayIntersector only supports level sets!"
                          "\nUse Grid::setGridClass(openvdb::GRID_LEVEL_SET)");
        }
    }

    /// @brief Return the iso-value used for ray-intersections
    const ValueT& getIsoValue() const { return mTester.getIsoValue(); }

    /// @brief Return @c true if the index-space ray intersects the level set.
    /// @param iRay ray represented in index space.
    bool intersectsIS(const RayType& iRay) const
    {
        if (!mTester.setIndexRay(iRay)) return false;//missed bbox
        return math::LevelSetHDDA<TreeT, NodeLevel>::test(mTester);
    }

    /// @brief Return @c true if the index-space ray intersects the level set
    /// @param iRay  ray represented in index space.
    /// @param iTime if an intersection was found it is assigned the time of the
    ///              intersection along the index ray.
    bool intersectsIS(const RayType& iRay, RealType &iTime) const
    {
        if (!mTester.setIndexRay(iRay)) return false;//missed bbox
        iTime = mTester.getIndexTime();
        return math::LevelSetHDDA<TreeT, NodeLevel>::test(mTester);
    }

    /// @brief Return @c true if the index-space ray intersects the level set.
    /// @param iRay ray represented in index space.
    /// @param xyz  if an intersection was found it is assigned the
    ///             intersection point in index space, otherwise it is unchanged.
    bool intersectsIS(const RayType& iRay, Vec3Type& xyz) const
    {
        if (!mTester.setIndexRay(iRay)) return false;//missed bbox
        if (!math::LevelSetHDDA<TreeT, NodeLevel>::test(mTester)) return false;//missed level set
        mTester.getIndexPos(xyz);
        return true;
    }

    /// @brief Return @c true if the index-space ray intersects the level set.
    /// @param iRay  ray represented in index space.
    /// @param xyz   if an intersection was found it is assigned the
    ///              intersection point in index space, otherwise it is unchanged.
    /// @param iTime if an intersection was found it is assigned the time of the
    ///              intersection along the index ray.
    bool intersectsIS(const RayType& iRay, Vec3Type& xyz, RealType &iTime) const
    {
        if (!mTester.setIndexRay(iRay)) return false;//missed bbox
        if (!math::LevelSetHDDA<TreeT, NodeLevel>::test(mTester)) return false;//missed level set
        mTester.getIndexPos(xyz);
        iTime = mTester.getIndexTime();
        return true;
    }

    /// @brief Return @c true if the world-space ray intersects the level set.
    /// @param wRay   ray represented in world space.
    bool intersectsWS(const RayType& wRay) const
    {
        if (!mTester.setWorldRay(wRay)) return false;//missed bbox
        return math::LevelSetHDDA<TreeT, NodeLevel>::test(mTester);
    }

    /// @brief Return @c true if the world-space ray intersects the level set.
    /// @param wRay   ray represented in world space.
    /// @param wTime  if an intersection was found it is assigned the time of the
    ///               intersection along the world ray.
    bool intersectsWS(const RayType& wRay, RealType &wTime) const
    {
        if (!mTester.setWorldRay(wRay)) return false;//missed bbox
        wTime = mTester.getWorldTime();
        return math::LevelSetHDDA<TreeT, NodeLevel>::test(mTester);
    }

    /// @brief Return @c true if the world-space ray intersects the level set.
    /// @param wRay   ray represented in world space.
    /// @param world  if an intersection was found it is assigned the
    ///               intersection point in world space, otherwise it is unchanged
    bool intersectsWS(const RayType& wRay, Vec3Type& world) const
    {
        if (!mTester.setWorldRay(wRay)) return false;//missed bbox
        if (!math::LevelSetHDDA<TreeT, NodeLevel>::test(mTester)) return false;//missed level set
        mTester.getWorldPos(world);
        return true;
    }

    /// @brief Return @c true if the world-space ray intersects the level set.
    /// @param wRay   ray represented in world space.
    /// @param world  if an intersection was found it is assigned the
    ///               intersection point in world space, otherwise it is unchanged.
    /// @param wTime  if an intersection was found it is assigned the time of the
    ///               intersection along the world ray.
    bool intersectsWS(const RayType& wRay, Vec3Type& world, RealType &wTime) const
    {
        if (!mTester.setWorldRay(wRay)) return false;//missed bbox
        if (!math::LevelSetHDDA<TreeT, NodeLevel>::test(mTester)) return false;//missed level set
        mTester.getWorldPos(world);
        wTime = mTester.getWorldTime();
        return true;
    }

    /// @brief Return @c true if the world-space ray intersects the level set.
    /// @param wRay   ray represented in world space.
    /// @param world  if an intersection was found it is assigned the
    ///               intersection point in world space, otherwise it is unchanged.
    /// @param normal if an intersection was found it is assigned the normal
    ///               of the level set surface in world space, otherwise it is unchanged.
    bool intersectsWS(const RayType& wRay, Vec3Type& world, Vec3Type& normal) const
    {
        if (!mTester.setWorldRay(wRay)) return false;//missed bbox
        if (!math::LevelSetHDDA<TreeT, NodeLevel>::test(mTester)) return false;//missed level set
        mTester.getWorldPosAndNml(world, normal);
        return true;
    }

    /// @brief Return @c true if the world-space ray intersects the level set.
    /// @param wRay   ray represented in world space.
    /// @param world  if an intersection was found it is assigned the
    ///               intersection point in world space, otherwise it is unchanged.
    /// @param normal if an intersection was found it is assigned the normal
    ///               of the level set surface in world space, otherwise it is unchanged.
    /// @param wTime  if an intersection was found it is assigned the time of the
    ///               intersection along the world ray.
    bool intersectsWS(const RayType& wRay, Vec3Type& world, Vec3Type& normal, RealType &wTime) const
    {
        if (!mTester.setWorldRay(wRay)) return false;//missed bbox
        if (!math::LevelSetHDDA<TreeT, NodeLevel>::test(mTester)) return false;//missed level set
        mTester.getWorldPosAndNml(world, normal);
        wTime = mTester.getWorldTime();
        return true;
    }

private:

    mutable SearchImplT mTester;

};// LevelSetRayIntersector


////////////////////////////////////// VolumeRayIntersector //////////////////////////////////////


/// @brief This class provides the public API for intersecting a ray
/// with a generic (e.g. density) volume.
/// @details Internally it performs the actual hierarchical tree node traversal.
/// @warning Use the (default) copy-constructor to make sure each
/// computational thread has their own instance of this class. This is
/// important since it contains a ValueAccessor that is
/// not thread-safe and a CoordBBox of the active voxels that should
/// not be re-computed for each thread. However copying is very efficient.
/// @par Example:
/// @code
/// // Create an instance for the master thread
/// VolumeRayIntersector inter(grid);
/// // For each additional thread use the copy constructor. This
/// // amortizes the overhead of computing the bbox of the active voxels!
/// VolumeRayIntersector inter2(inter);
/// // Before each ray-traversal set the index ray.
/// iter.setIndexRay(ray);
/// // or world ray
/// iter.setWorldRay(ray);
/// // Now you can begin the ray-marching using consecutive calls to VolumeRayIntersector::march
/// double t0=0, t1=0;// note the entry and exit times are with respect to the INDEX ray
/// while ( inter.march(t0, t1) ) {
///   // perform line-integration between t0 and t1
/// }}
/// @endcode
template<typename GridT,
         int NodeLevel = GridT::TreeType::RootNodeType::ChildNodeType::LEVEL,
         typename RayT = math::Ray<Real> >
class VolumeRayIntersector
{
public:
    using GridType = GridT;
    using RayType = RayT;
    using RealType = typename RayT::RealType;
    using RootType = typename GridT::TreeType::RootNodeType;
    using TreeT = tree::Tree<typename RootType::template ValueConverter<bool>::Type>;

    static_assert(NodeLevel >= 0 && NodeLevel < int(TreeT::DEPTH)-1, "NodeLevel out of range");

    /// @brief Grid constructor
    /// @param grid Generic grid to intersect rays against.
    /// @param dilationCount The number of voxel dilations performed
    /// on (a boolean copy of) the input grid. This allows the
    /// intersector to account for the size of interpolation kernels
    /// in client code.
    /// @throw RuntimeError if the voxels of the grid are not uniform
    /// or the grid is empty.
    VolumeRayIntersector(const GridT& grid, int dilationCount = 0)
        : mIsMaster(true)
        , mTree(new TreeT(grid.tree(), false, TopologyCopy()))
        , mGrid(&grid)
        , mAccessor(*mTree)
    {
        if (!grid.hasUniformVoxels() ) {
            OPENVDB_THROW(RuntimeError,
                          "VolumeRayIntersector only supports uniform voxels!");
        }
        if ( grid.empty() ) {
            OPENVDB_THROW(RuntimeError, "LinearSearchImpl does not supports empty grids");
        }

        // Dilate active voxels to better account for the size of interpolation kernels
        tools::dilateVoxels(*mTree, dilationCount);

        mTree->root().evalActiveBoundingBox(mBBox, /*visit individual voxels*/false);

        mBBox.max().offset(1);//padding so the bbox of a node becomes (origin,origin + node_dim)
    }

    /// @brief Grid and BBox constructor
    /// @param grid Generic grid to intersect rays against.
    /// @param bbox The axis-aligned bounding-box in the index space of the grid.
    /// @warning It is assumed that bbox = (min, min + dim) where min denotes
    /// to the smallest grid coordinates and dim are the integer length of the bbox.
    /// @throw RuntimeError if the voxels of the grid are not uniform
    /// or the grid is empty.
    VolumeRayIntersector(const GridT& grid, const math::CoordBBox& bbox)
        : mIsMaster(true)
        , mTree(new TreeT(grid.tree(), false, TopologyCopy()))
        , mGrid(&grid)
        , mAccessor(*mTree)
        , mBBox(bbox)
    {
        if (!grid.hasUniformVoxels() ) {
            OPENVDB_THROW(RuntimeError,
                          "VolumeRayIntersector only supports uniform voxels!");
        }
        if ( grid.empty() ) {
            OPENVDB_THROW(RuntimeError, "LinearSearchImpl does not supports empty grids");
        }
    }

    /// @brief Shallow copy constructor
    /// @warning This copy constructor creates shallow copies of data
    /// members of the instance passed as the argument. For
    /// performance reasons we are not using shared pointers (their
    /// mutex-lock impairs multi-threading).
    VolumeRayIntersector(const VolumeRayIntersector& other)
        : mIsMaster(false)
        , mTree(other.mTree)//shallow copy
        , mGrid(other.mGrid)//shallow copy
        , mAccessor(*mTree)//initialize new (vs deep copy)
        , mRay(other.mRay)//deep copy
        , mTmax(other.mTmax)//deep copy
        , mBBox(other.mBBox)//deep copy
    {
    }

    /// @brief Destructor
    ~VolumeRayIntersector() { if (mIsMaster) delete mTree; }

    /// @brief Return @c false if the index ray misses the bbox of the grid.
    /// @param iRay Ray represented in index space.
    /// @warning Call this method (or setWorldRay) before the ray
    /// traversal starts and use the return value to decide if further
    /// marching is required.
    inline bool setIndexRay(const RayT& iRay)
    {
        mRay = iRay;
        const bool hit = mRay.clip(mBBox);
        if (hit) mTmax = mRay.t1();
        return hit;
    }

    /// @brief Return @c false if the world ray misses the bbox of the grid.
    /// @param wRay Ray represented in world space.
    /// @warning Call this method (or setIndexRay) before the ray
    /// traversal starts and use the return value to decide if further
    /// marching is required.
    /// @details Since hit times are computed with respect to the ray
    /// represented in index space of the current grid, it is
    /// recommended that either the client code uses getIndexPos to
    /// compute index position from hit times or alternatively keeps
    /// an instance of the index ray and instead uses setIndexRay to
    /// initialize the ray.
    inline bool setWorldRay(const RayT& wRay)
    {
        return this->setIndexRay(wRay.worldToIndex(*mGrid));
    }

    inline typename RayT::TimeSpan march()
    {
        const typename RayT::TimeSpan t = mHDDA.march(mRay, mAccessor);
        if (t.t1>0) mRay.setTimes(t.t1 + math::Delta<RealType>::value(), mTmax);
        return t;
    }

    /// @brief Return @c true if the ray intersects active values,
    /// i.e. either active voxels or tiles. Only when a hit is
    /// detected are t0 and t1 updated with the corresponding entry
    /// and exit times along the INDEX ray!
    /// @note Note that t0 and t1 are only resolved at the node level
    /// (e.g. a LeafNode with active voxels) as opposed to the individual
    /// active voxels.
    /// @param t0 If the return value > 0 this is the time of the
    /// first hit of an active tile or leaf.
    /// @param t1 If the return value > t0 this is the time of the
    /// first hit (> t0) of an inactive tile or exit point of the
    /// BBOX for the leaf nodes.
    /// @warning t0 and t1 are computed with respect to the ray represented in
    /// index space of the current grid, not world space!
    inline bool march(RealType& t0, RealType& t1)
    {
        const typename RayT::TimeSpan t = this->march();
        t.get(t0, t1);
        return t.valid();
    }

    /// @brief Generates a list of hits along the ray.
    ///
    /// @param list List of hits represented as time spans.
    ///
    /// @note ListType is a list of RayType::TimeSpan and is required to
    /// have the two methods: clear() and push_back(). Thus, it could
    /// be std::vector<typename RayType::TimeSpan> or
    /// std::deque<typename RayType::TimeSpan>.
    template <typename ListType>
    inline void hits(ListType& list)
    {
        mHDDA.hits(mRay, mAccessor, list);
    }

    /// @brief Return the floating-point index position along the
    /// current index ray at the specified time.
    inline Vec3R getIndexPos(RealType time) const { return mRay(time); }

    /// @brief Return the floating-point world position along the
    /// current index ray at the specified time.
    inline Vec3R getWorldPos(RealType time) const { return mGrid->indexToWorld(mRay(time)); }

    inline RealType getWorldTime(RealType time) const
    {
        return time*mGrid->transform().baseMap()->applyJacobian(mRay.dir()).length();
    }

    /// @brief Return a const reference to the input grid.
    const GridT& grid() const { return *mGrid; }

    /// @brief Return a const reference to the (potentially dilated)
    /// bool tree used to accelerate the ray marching.
    const TreeT& tree() const { return *mTree; }

    /// @brief Return a const reference to the BBOX of the grid
    const math::CoordBBox& bbox() const { return mBBox; }

    /// @brief Print bbox, statistics, memory usage and other information.
    /// @param os            a stream to which to write textual information
    /// @param verboseLevel  1: print bbox only; 2: include boolean tree
    ///                      statistics; 3: include memory usage
    void print(std::ostream& os = std::cout, int verboseLevel = 1)
    {
        if (verboseLevel>0) {
            os << "BBox: " << mBBox << std::endl;
            if (verboseLevel==2) {
                mTree->print(os, 1);
            } else if (verboseLevel>2) {
                mTree->print(os, 2);
            }
        }
    }

private:
    using AccessorT = typename tree::ValueAccessor<const TreeT,/*IsSafe=*/false>;

    const bool      mIsMaster;
    TreeT*          mTree;
    const GridT*    mGrid;
    AccessorT       mAccessor;
    RayT            mRay;
    RealType        mTmax;
    math::CoordBBox mBBox;
    math::VolumeHDDA<TreeT, RayType, NodeLevel> mHDDA;

};// VolumeRayIntersector


//////////////////////////////////////// LinearSearchImpl ////////////////////////////////////////


/// @brief Implements linear iterative search for an iso-value of
/// the level set along the direction of the ray.
///
/// @note Since this class is used internally in
/// LevelSetRayIntersector (define above) and LevelSetHDDA (defined below)
/// client code should never interact directly with its API. This also
/// explains why we are not concerned with the fact that several of
/// its methods are unsafe to call unless roots were already detected.
///
/// @details It is approximate due to the limited number of iterations
/// which can can be defined with a template parameter. However the default value
/// has proven surprisingly accurate and fast. In fact more iterations
/// are not guaranteed to give significantly better results.
///
/// @warning Since the root-searching algorithm is approximate
/// (first-order) it is possible to miss intersections if the
/// iso-value is too close to the inside or outside of the narrow
/// band (typically a distance less than a voxel unit).
///
/// @warning Since this class internally stores a ValueAccessor it is NOT thread-safe,
/// so make sure to give each thread its own instance.  This of course also means that
/// the cost of allocating an instance should (if possible) be amortized over
/// as many ray intersections as possible.
template<typename GridT, int Iterations, typename RealT>
class LinearSearchImpl
{
public:
    using RayT = math::Ray<RealT>;
    using VecT = math::Vec3<RealT>;
    using ValueT = typename GridT::ValueType;
    using AccessorT = typename GridT::ConstAccessor;
    using StencilT = math::BoxStencil<GridT>;

    /// @brief Constructor from a grid.
    /// @throw RunTimeError if the grid is empty.
    /// @throw ValueError if the isoValue is not inside the narrow-band.
    LinearSearchImpl(const GridT& grid, const ValueT& isoValue = zeroVal<ValueT>())
        : mStencil(grid),
          mIsoValue(isoValue),
          mMinValue(isoValue - ValueT(2 * grid.voxelSize()[0])),
          mMaxValue(isoValue + ValueT(2 * grid.voxelSize()[0]))
      {
          if ( grid.empty() ) {
              OPENVDB_THROW(RuntimeError, "LinearSearchImpl does not supports empty grids");
          }
          if (mIsoValue<= -grid.background() ||
              mIsoValue>=  grid.background() ){
              OPENVDB_THROW(ValueError, "The iso-value must be inside the narrow-band!");
          }
          grid.tree().root().evalActiveBoundingBox(mBBox, /*visit individual voxels*/false);
      }

    /// @brief Return the iso-value used for ray-intersections
    const ValueT& getIsoValue() const { return mIsoValue; }

    /// @brief Return @c false if the ray misses the bbox of the grid.
    /// @param iRay Ray represented in index space.
    /// @warning Call this method before the ray traversal starts.
    inline bool setIndexRay(const RayT& iRay)
    {
        mRay = iRay;
        return mRay.clip(mBBox);//did it hit the bbox
    }

    /// @brief Return @c false if the ray misses the bbox of the grid.
    /// @param wRay Ray represented in world space.
    /// @warning Call this method before the ray traversal starts.
    inline bool setWorldRay(const RayT& wRay)
    {
        mRay = wRay.worldToIndex(mStencil.grid());
        return mRay.clip(mBBox);//did it hit the bbox
    }

    /// @brief Get the intersection point in index space.
    /// @param xyz The position in index space of the intersection.
    inline void getIndexPos(VecT& xyz) const { xyz = mRay(mTime); }

    /// @brief Get the intersection point in world space.
    /// @param xyz The position in world space of the intersection.
    inline void getWorldPos(VecT& xyz) const { xyz = mStencil.grid().indexToWorld(mRay(mTime)); }

    /// @brief Get the intersection point and normal in world space
    /// @param xyz The position in world space of the intersection.
    /// @param nml The surface normal in world space of the intersection.
    inline void getWorldPosAndNml(VecT& xyz, VecT& nml)
    {
        this->getIndexPos(xyz);
        mStencil.moveTo(xyz);
        nml = mStencil.gradient(xyz);
        nml.normalize();
        xyz = mStencil.grid().indexToWorld(xyz);
    }

    /// @brief Return the time of intersection along the index ray.
    inline RealT getIndexTime() const { return mTime; }

    /// @brief Return the time of intersection along the world ray.
    inline RealT getWorldTime() const
    {
        return mTime*mStencil.grid().transform().baseMap()->applyJacobian(mRay.dir()).length();
    }

private:

    /// @brief Initiate the local voxel intersection test.
    /// @warning Make sure to call this method before the local voxel intersection test.
    inline void init(RealT t0)
    {
        mT[0] = t0;
        mV[0] = static_cast<ValueT>(this->interpValue(t0));
    }

    inline void setRange(RealT t0, RealT t1) { mRay.setTimes(t0, t1); }

    /// @brief Return a const reference to the ray.
    inline const RayT& ray() const { return mRay; }

    /// @brief Return true if a node of the specified type exists at ijk.
    template <typename NodeT>
    inline bool hasNode(const Coord& ijk)
    {
        return mStencil.accessor().template probeConstNode<NodeT>(ijk) != nullptr;
    }

    /// @brief Return @c true if an intersection is detected.
    /// @param ijk Grid coordinate of the node origin or voxel being tested.
    /// @param time Time along the index ray being tested.
    /// @warning Only if an intersection is detected is it safe to
    /// call getIndexPos, getWorldPos and getWorldPosAndNml!
    inline bool operator()(const Coord& ijk, RealT time)
    {
        ValueT V;
        if (mStencil.accessor().probeValue(ijk, V) &&//within narrow band
            V>mMinValue && V<mMaxValue) {// and close to iso-value?
            mT[1] = time;
            mV[1] = static_cast<ValueT>(this->interpValue(time));
            if (math::ZeroCrossing(mV[0], mV[1])) {
                mTime = this->interpTime();
                OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
                for (int n=0; Iterations>0 && n<Iterations; ++n) {//resolved at compile-time
                    V = static_cast<ValueT>(this->interpValue(mTime));
                    const int m = math::ZeroCrossing(mV[0], V) ? 1 : 0;
                    mV[m] = V;
                    mT[m] = mTime;
                    mTime = this->interpTime();
                }
                OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
                return true;
            }
            mT[0] = mT[1];
            mV[0] = mV[1];
        }
        return false;
    }

    inline RealT interpTime()
    {
        assert( math::isApproxLarger(mT[1], mT[0], RealT(1e-6) ) );
        return mT[0]+(mT[1]-mT[0])*mV[0]/(mV[0]-mV[1]);
    }

    inline RealT interpValue(RealT time)
    {
        const VecT pos = mRay(time);
        mStencil.moveTo(pos);
        return mStencil.interpolation(pos) - mIsoValue;
    }

    template<typename, int> friend struct math::LevelSetHDDA;

    RayT            mRay;
    StencilT        mStencil;
    RealT           mTime;//time of intersection
    ValueT          mV[2];
    RealT           mT[2];
    const ValueT    mIsoValue, mMinValue, mMaxValue;
    math::CoordBBox mBBox;
};// LinearSearchImpl

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_RAYINTERSECTOR_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
