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
/// @file tools/ParticlesToLevelSet.h
///
/// @brief Rasterize particles with position, radius and velocity
/// into either a boolean mask grid or a narrow-band level set grid.
///
/// @details Optionally, arbitrary attributes on the particles can be transferred,
/// resulting in additional output grids with the same topology as the main grid.
///
/// @note Particle to level set conversion is intended to be combined with
/// some kind of surface postprocessing, using
/// @vdblink::tools::LevelSetFilter LevelSetFilter@endlink, for example.
/// Without such postprocessing the generated surface is typically too noisy and blobby.
/// However, it serves as a great and fast starting point for subsequent
/// level set surface processing and convolution.
///
/// @details For particle access, any class with the following interface may be used
/// (see the unit test or the From Particles Houdini SOP for practical examples):
/// @code
/// struct ParticleList
/// {
///     // Return the total number of particles in the list.
///     // Always required!
///     size_t size() const;
///
///     // Get the world-space position of the nth particle.
///     // Required by rasterizeSpheres().
///     void getPos(size_t n, Vec3R& xyz) const;
///
///     // Get the world-space position and radius of the nth particle.
///     // Required by rasterizeSpheres().
///     void getPosRad(size_t n, Vec3R& xyz, Real& radius) const;
///
///     // Get the world-space position, radius and velocity of the nth particle.
///     // Required by rasterizeTrails().
///     void getPosRadVel(size_t n, Vec3R& xyz, Real& radius, Vec3R& velocity) const;
///
///     // Get the value of the nth particle's user-defined attribute (of type @c AttributeType).
///     // Required only if attribute transfer is enabled in ParticlesToLevelSet.
///     void getAtt(size_t n, AttributeType& att) const;
/// };
/// @endcode
///
/// Some functions accept an interrupter argument.  This refers to any class
/// with the following interface:
/// @code
/// struct Interrupter
/// {
///     void start(const char* name = nullptr) // called when computations begin
///     void end()                             // called when computations end
///     bool wasInterrupted(int percent=-1)    // return true to abort computation
/// };
/// @endcode
///
/// The default interrupter is @vdblink::util::NullInterrupter NullInterrupter@endlink,
/// for which all calls are no-ops that incur no computational overhead.

#ifndef OPENVDB_TOOLS_PARTICLES_TO_LEVELSET_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_PARTICLES_TO_LEVELSET_HAS_BEEN_INCLUDED

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/util/logging.h>
#include <openvdb/util/NullInterrupter.h>
#include "Composite.h" // for csgUnion()
#include "PointPartitioner.h"
#include "Prune.h"
#include "SignedFloodFill.h"
#include <functional>
#include <iostream>
#include <type_traits>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Populate a scalar, floating-point grid with CSG-unioned level set spheres
/// described by the given particle positions and radii.
/// @details For more control over the output, including attribute transfer,
/// use the ParticlesToLevelSet class directly.
template<typename GridT, typename ParticleListT, typename InterrupterT = util::NullInterrupter>
inline void particlesToSdf(const ParticleListT&, GridT&, InterrupterT* = nullptr);

/// @brief Populate a scalar, floating-point grid with fixed-size, CSG-unioned
/// level set spheres described by the given particle positions and the specified radius.
/// @details For more control over the output, including attribute transfer,
/// use the ParticlesToLevelSet class directly.
template<typename GridT, typename ParticleListT, typename InterrupterT = util::NullInterrupter>
inline void particlesToSdf(const ParticleListT&, GridT&, Real radius, InterrupterT* = nullptr);

/// @brief Populate a scalar, floating-point grid with CSG-unioned trails
/// of level set spheres with decreasing radius, where the starting position and radius
/// and the direction of each trail is given by particle attributes.
/// @details For more control over the output, including attribute transfer,
/// use the ParticlesToLevelSet class directly.
/// @note The @a delta parameter controls the distance between spheres in a trail.
/// Be careful not to use too small a value.
template<typename GridT, typename ParticleListT, typename InterrupterT = util::NullInterrupter>
inline void particleTrailsToSdf(const ParticleListT&, GridT&, Real delta=1, InterrupterT* =nullptr);

/// @brief Activate a boolean grid wherever it intersects the spheres
/// described by the given particle positions and radii.
/// @details For more control over the output, including attribute transfer,
/// use the ParticlesToLevelSet class directly.
template<typename GridT, typename ParticleListT, typename InterrupterT = util::NullInterrupter>
inline void particlesToMask(const ParticleListT&, GridT&, InterrupterT* = nullptr);

/// @brief Activate a boolean grid wherever it intersects the fixed-size spheres
/// described by the given particle positions and the specified radius.
/// @details For more control over the output, including attribute transfer,
/// use the ParticlesToLevelSet class directly.
template<typename GridT, typename ParticleListT, typename InterrupterT = util::NullInterrupter>
inline void particlesToMask(const ParticleListT&, GridT&, Real radius, InterrupterT* = nullptr);

/// @brief Activate a boolean grid wherever it intersects trails of spheres
/// with decreasing radius, where the starting position and radius and the direction
/// of each trail is given by particle attributes.
/// @details For more control over the output, including attribute transfer,
/// use the ParticlesToLevelSet class directly.
/// @note The @a delta parameter controls the distance between spheres in a trail.
/// Be careful not to use too small a value.
template<typename GridT, typename ParticleListT, typename InterrupterT = util::NullInterrupter>
inline void particleTrailsToMask(const ParticleListT&, GridT&,Real delta=1,InterrupterT* =nullptr);


////////////////////////////////////////


namespace p2ls_internal {
// This is a simple type that combines a distance value and a particle
// attribute. It's required for attribute transfer which is performed
// in the ParticlesToLevelSet::Raster member class defined below.
/// @private
template<typename VisibleT, typename BlindT> class BlindData;
}


template<typename SdfGridT,
         typename AttributeT = void,
         typename InterrupterT = util::NullInterrupter>
class ParticlesToLevelSet
{
public:
    using DisableT = typename std::is_void<AttributeT>::type;
    using InterrupterType = InterrupterT;

    using SdfGridType = SdfGridT;
    using SdfType = typename SdfGridT::ValueType;

    using AttType = typename std::conditional<DisableT::value, size_t, AttributeT>::type;
    using AttGridType = typename SdfGridT::template ValueConverter<AttType>::Type;

    static const bool OutputIsMask = std::is_same<SdfType, bool>::value;

    /// @brief Constructor using an existing boolean or narrow-band level set grid
    ///
    /// @param grid       grid into which particles are rasterized
    /// @param interrupt  callback to interrupt a long-running process
    ///
    /// @details If the input grid is already populated with signed distances,
    /// particles are unioned onto the existing level set surface.
    ///
    /// @details The width in voxel units of the generated narrow band level set
    /// is given by 2&times;<I>background</I>/<I>dx</I>, where @a background
    /// is the background value stored in the grid and @a dx is the voxel size
    /// derived from the transform associated with the grid.
    /// Also note that &minus;<I>background</I> corresponds to the constant value
    /// inside the generated narrow-band level set.
    ///
    /// @note If attribute transfer is enabled, i.e., if @c AttributeT is not @c void,
    /// attributes are generated only for voxels that overlap with particles,
    /// not for any other preexisting voxels (for which no attributes exist!).
    explicit ParticlesToLevelSet(SdfGridT& grid, InterrupterT* interrupt = nullptr);

    ~ParticlesToLevelSet() { delete mBlindGrid; }

    /// @brief This method syncs up the level set and attribute grids
    /// and therefore needs to be called before any of those grids are
    /// used and after the last call to any of the rasterizer methods.
    /// @details It has no effect or overhead if attribute transfer is disabled
    /// (i.e., if @c AttributeT is @c void) and @a prune is @c false.
    ///
    /// @note Avoid calling this method more than once, and call it only after
    /// all the particles have been rasterized.
    void finalize(bool prune = false);

    /// @brief Return a pointer to the grid containing the optional user-defined attribute.
    /// @warning If attribute transfer is disabled (i.e., if @c AttributeT is @c void)
    /// or if @link finalize() finalize@endlink is not called, the pointer will be null.
    typename AttGridType::Ptr attributeGrid() { return mAttGrid; }

    /// @brief Return the size of a voxel in world units.
    Real getVoxelSize() const { return mDx; }

    /// @brief Return the half-width of the narrow band in voxel units.
    Real getHalfWidth() const { return mHalfWidth; }

    /// @brief Return the smallest radius allowed in voxel units.
    Real getRmin() const { return mRmin; }
    /// @brief Set the smallest radius allowed in voxel units.
    void setRmin(Real Rmin) { mRmin = math::Max(Real(0),Rmin); }

    /// @brief Return the largest radius allowed in voxel units.
    Real getRmax() const { return mRmax; }
    /// @brief Set the largest radius allowed in voxel units.
    void setRmax(Real Rmax) { mRmax = math::Max(mRmin,Rmax); }

    /// @brief Return @c true if any particles were ignored due to their size.
    bool ignoredParticles() const { return mMinCount>0 || mMaxCount>0; }
    /// @brief Return the number of particles that were ignored because they were
    /// smaller than the minimum radius.
    size_t getMinCount() const { return mMinCount; }
    /// @brief Return the number of particles that were ignored because they were
    /// larger than the maximum radius.
    size_t getMaxCount() const { return mMaxCount; }

    /// @brief Return the grain size used for threading
    int getGrainSize() const { return mGrainSize; }
    /// @brief Set the grain size used for threading.
    /// @note A grain size of zero or less disables threading.
    void setGrainSize(int grainSize) { mGrainSize = grainSize; }

    /// @brief Rasterize each particle as a sphere with the particle's position and radius.
    /// @details For level set output, all spheres are CSG-unioned.
    template<typename ParticleListT>
    void rasterizeSpheres(const ParticleListT& pa);

    /// @brief Rasterize each particle as a sphere with the particle's position
    /// and a fixed radius.
    /// @details For level set output, all spheres are CSG-unioned.
    ///
    /// @param pa      particles with positions
    /// @param radius  fixed sphere radius in world units.
    template<typename ParticleListT>
    void rasterizeSpheres(const ParticleListT& pa, Real radius);

    /// @brief Rasterize each particle as a trail comprising the CSG union
    /// of spheres of decreasing radius.
    ///
    /// @param pa     particles with position, radius and velocity.
    /// @param delta  controls the distance between sphere instances
    ///
    /// @warning Be careful not to use too small values for @a delta,
    /// since this can lead to excessive computation per trail (which the
    /// interrupter can't stop).
    ///
    /// @note The direction of a trail is opposite to that of the velocity vector,
    /// and its length is given by the magnitude of the velocity.
    /// The radius at the head of the trail is given by the radius of the particle,
    /// and the radius at the tail is @a Rmin voxel units, which has
    /// a default value of 1.5 corresponding to the Nyquist frequency!
    template<typename ParticleListT>
    void rasterizeTrails(const ParticleListT& pa, Real delta=1.0);

private:
    using BlindType = p2ls_internal::BlindData<SdfType, AttType>;
    using BlindGridType = typename SdfGridT::template ValueConverter<BlindType>::Type;

    /// Class with multi-threaded implementation of particle rasterization
    template<typename ParticleListT, typename GridT> struct Raster;

    SdfGridType*   mSdfGrid;
    typename AttGridType::Ptr   mAttGrid;
    BlindGridType* mBlindGrid;
    InterrupterT*  mInterrupter;
    Real           mDx, mHalfWidth;
    Real           mRmin, mRmax; // ignore particles outside this range of radii in voxel
    size_t         mMinCount, mMaxCount; // counters for ignored particles
    int            mGrainSize;
}; // class ParticlesToLevelSet


template<typename SdfGridT, typename AttributeT, typename InterrupterT>
inline ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT>::
ParticlesToLevelSet(SdfGridT& grid, InterrupterT* interrupter) :
    mSdfGrid(&grid),
    mBlindGrid(nullptr),
    mInterrupter(interrupter),
    mDx(grid.voxelSize()[0]),
    mHalfWidth(grid.background()/mDx),
    mRmin(1.5),// corresponds to the Nyquist grid sampling frequency
    mRmax(100.0),// corresponds to a huge particle (probably too large!)
    mMinCount(0),
    mMaxCount(0),
    mGrainSize(1)
{
    if (!mSdfGrid->hasUniformVoxels()) {
        OPENVDB_THROW(RuntimeError, "ParticlesToLevelSet only supports uniform voxels!");
    }
    if (!DisableT::value) {
        mBlindGrid = new BlindGridType(BlindType(grid.background()));
        mBlindGrid->setTransform(mSdfGrid->transform().copy());
    }
}

template<typename SdfGridT, typename AttributeT, typename InterrupterT>
template<typename ParticleListT>
inline void ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT>::
rasterizeSpheres(const ParticleListT& pa)
{
    if (DisableT::value) {
        Raster<ParticleListT, SdfGridT> r(*this, mSdfGrid, pa);
        r.rasterizeSpheres();
    } else {
        Raster<ParticleListT, BlindGridType> r(*this, mBlindGrid, pa);
        r.rasterizeSpheres();
    }
}

template<typename SdfGridT, typename AttributeT, typename InterrupterT>
template<typename ParticleListT>
inline void ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT>::
rasterizeSpheres(const ParticleListT& pa, Real radius)
{
    if (DisableT::value) {
        Raster<ParticleListT, SdfGridT> r(*this, mSdfGrid, pa);
        r.rasterizeSpheres(radius/mDx);
    } else {
        Raster<ParticleListT, BlindGridType> r(*this, mBlindGrid, pa);
        r.rasterizeSpheres(radius/mDx);
    }
}

template<typename SdfGridT, typename AttributeT, typename InterrupterT>
template<typename ParticleListT>
inline void ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT>::
rasterizeTrails(const ParticleListT& pa, Real delta)
{
    if (DisableT::value) {
        Raster<ParticleListT, SdfGridT> r(*this, mSdfGrid, pa);
        r.rasterizeTrails(delta);
    } else {
        Raster<ParticleListT, BlindGridType> r(*this, mBlindGrid, pa);
        r.rasterizeTrails(delta);
    }
}


template<typename SdfGridT, typename AttributeT, typename InterrupterT>
inline void
ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT>::finalize(bool prune)
{
    OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN

    if (!mBlindGrid) {
        if (prune) {
            if (OutputIsMask) {
                tools::prune(mSdfGrid->tree());
            } else {
                tools::pruneLevelSet(mSdfGrid->tree());
            }
        }
        return;
    }

    if (prune) tools::prune(mBlindGrid->tree());

    using AttTreeT = typename AttGridType::TreeType;
    using AttLeafT = typename AttTreeT::LeafNodeType;
    using BlindTreeT = typename BlindGridType::TreeType;
    using BlindLeafIterT = typename BlindTreeT::LeafCIter;
    using BlindLeafT = typename BlindTreeT::LeafNodeType;
    using SdfTreeT = typename SdfGridType::TreeType;
    using SdfLeafT = typename SdfTreeT::LeafNodeType;

    // Use topology copy constructors since output grids have the same topology as mBlindDataGrid
    const BlindTreeT& blindTree = mBlindGrid->tree();

    // Create the output attribute grid.
    typename AttTreeT::Ptr attTree(new AttTreeT(
        blindTree, blindTree.background().blind(), openvdb::TopologyCopy()));
    // Note this overwrites any existing attribute grids!
    mAttGrid = typename AttGridType::Ptr(new AttGridType(attTree));
    mAttGrid->setTransform(mBlindGrid->transform().copy());

    typename SdfTreeT::Ptr sdfTree; // the output mask or level set tree

    // Extract the attribute grid and the mask or level set grid from mBlindDataGrid.
    if (OutputIsMask) {
        sdfTree.reset(new SdfTreeT(blindTree,
            /*off=*/SdfType(0), /*on=*/SdfType(1), TopologyCopy()));

        // Copy leaf voxels in parallel.
        tree::LeafManager<AttTreeT> leafNodes(*attTree);
        leafNodes.foreach([&](AttLeafT& attLeaf, size_t /*leafIndex*/) {
            if (const auto* blindLeaf = blindTree.probeConstLeaf(attLeaf.origin())) {
                for (auto iter = attLeaf.beginValueOn(); iter; ++iter) {
                    const auto pos = iter.pos();
                    attLeaf.setValueOnly(pos, blindLeaf->getValue(pos).blind());
                }
            }
        });
        // Copy tiles serially.
        const auto blindAcc = mBlindGrid->getConstAccessor();
        auto iter = attTree->beginValueOn();
        iter.setMaxDepth(AttTreeT::ValueOnIter::LEAF_DEPTH - 1);
        for ( ; iter; ++iter) {
            iter.modifyValue([&](AttType& v) { v = blindAcc.getValue(iter.getCoord()).blind(); });
        }
    } else {
        // Here we exploit the fact that by design level sets have no active tiles.
        // Only leaf voxels can be active.
        sdfTree.reset(new SdfTreeT(blindTree, blindTree.background().visible(), TopologyCopy()));
        for (BlindLeafIterT n = blindTree.cbeginLeaf(); n; ++n) {
            const BlindLeafT& leaf = *n;
            const openvdb::Coord xyz = leaf.origin();
            // Get leafnodes that were allocated during topology construction!
            SdfLeafT* sdfLeaf = sdfTree->probeLeaf(xyz);
            AttLeafT* attLeaf = attTree->probeLeaf(xyz);
            // Use linear offset (vs coordinate) access for better performance!
            typename BlindLeafT::ValueOnCIter m=leaf.cbeginValueOn();
            if (!m) {//no active values in leaf node so copy everything
                for (openvdb::Index k = 0; k!=BlindLeafT::SIZE; ++k) {
                    const BlindType& v = leaf.getValue(k);
                    sdfLeaf->setValueOnly(k, v.visible());
                    attLeaf->setValueOnly(k, v.blind());
                }
            } else {//only copy active values (using flood fill for the inactive values)
                for(; m; ++m) {
                    const openvdb::Index k = m.pos();
                    const BlindType& v = *m;
                    sdfLeaf->setValueOnly(k, v.visible());
                    attLeaf->setValueOnly(k, v.blind());
                }
            }
        }
        tools::signedFloodFill(*sdfTree);//required since we only transferred active voxels!
    }

    if (mSdfGrid->empty()) {
        mSdfGrid->setTree(sdfTree);
    } else {
        if (OutputIsMask) {
            mSdfGrid->tree().topologyUnion(*sdfTree);
            tools::prune(mSdfGrid->tree());
        } else {
            tools::csgUnion(mSdfGrid->tree(), *sdfTree, /*prune=*/true);
        }
    }

    OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
}


///////////////////////////////////////////////////////////


template<typename SdfGridT, typename AttributeT, typename InterrupterT>
template<typename ParticleListT, typename GridT>
struct ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT>::Raster
{
    using DisableT = typename std::is_void<AttributeT>::type;
    using ParticlesToLevelSetT = ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT>;
    using SdfT = typename ParticlesToLevelSetT::SdfType; // type of signed distance values
    using AttT = typename ParticlesToLevelSetT::AttType; // type of particle attribute
    using ValueT = typename GridT::ValueType;
    using AccessorT = typename GridT::Accessor;
    using TreeT = typename GridT::TreeType;
    using LeafNodeT = typename TreeT::LeafNodeType;
    using PointPartitionerT = PointPartitioner<Index32, LeafNodeT::LOG2DIM>;

    static const bool
        OutputIsMask = std::is_same<SdfT, bool>::value,
        DoAttrXfer = !DisableT::value;

    /// @brief Main constructor
    Raster(ParticlesToLevelSetT& parent, GridT* grid, const ParticleListT& particles)
        : mParent(parent)
        , mParticles(particles)
        , mGrid(grid)
        , mMap(*(mGrid->transform().baseMap()))
        , mMinCount(0)
        , mMaxCount(0)
        , mIsCopy(false)
    {
        mPointPartitioner = new PointPartitionerT;
        mPointPartitioner->construct(particles, mGrid->transform());
    }

    /// @brief Copy constructor called by tbb threads
    Raster(Raster& other, tbb::split)
        : mParent(other.mParent)
        , mParticles(other.mParticles)
        , mGrid(new GridT(*other.mGrid, openvdb::ShallowCopy()))
        , mMap(other.mMap)
        , mMinCount(0)
        , mMaxCount(0)
        , mTask(other.mTask)
        , mIsCopy(true)
        , mPointPartitioner(other.mPointPartitioner)
    {
        mGrid->newTree();
    }

    virtual ~Raster()
    {
        // Copy-constructed Rasters own temporary grids that have to be deleted,
        // while the original has ownership of the bucket array.
        if (mIsCopy) {
            delete mGrid;
        } else {
            delete mPointPartitioner;
        }
    }

    void rasterizeSpheres()
    {
        mMinCount = mMaxCount = 0;
        if (mParent.mInterrupter) {
            mParent.mInterrupter->start("Rasterizing particles to level set using spheres");
        }
        mTask = std::bind(&Raster::rasterSpheres, std::placeholders::_1, std::placeholders::_2);
        this->cook();
        if (mParent.mInterrupter) mParent.mInterrupter->end();
    }

    void rasterizeSpheres(Real radius)
    {
        mMinCount = radius < mParent.mRmin ? mParticles.size() : 0;
        mMaxCount = radius > mParent.mRmax ? mParticles.size() : 0;
        if (mMinCount>0 || mMaxCount>0) {//skipping all particles!
            mParent.mMinCount = mMinCount;
            mParent.mMaxCount = mMaxCount;
        } else {
            if (mParent.mInterrupter) {
                mParent.mInterrupter->start(
                    "Rasterizing particles to level set using const spheres");
            }
            mTask = std::bind(&Raster::rasterFixedSpheres,
                std::placeholders::_1, std::placeholders::_2, radius);
            this->cook();
            if (mParent.mInterrupter) mParent.mInterrupter->end();
        }
    }

    void rasterizeTrails(Real delta=1.0)
    {
        mMinCount = mMaxCount = 0;
        if (mParent.mInterrupter) {
            mParent.mInterrupter->start("Rasterizing particles to level set using trails");
        }
        mTask = std::bind(&Raster::rasterTrails,
            std::placeholders::_1, std::placeholders::_2, delta);
        this->cook();
        if (mParent.mInterrupter) mParent.mInterrupter->end();
    }

    /// @brief Kick off the optionally multithreaded computation.
    void operator()(const tbb::blocked_range<size_t>& r)
    {
        assert(mTask);
        mTask(this, r);
        mParent.mMinCount = mMinCount;
        mParent.mMaxCount = mMaxCount;
    }

    /// @brief Required by tbb::parallel_reduce
    void join(Raster& other)
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (OutputIsMask) {
            if (DoAttrXfer) {
                tools::compMax(*mGrid, *other.mGrid);
            } else {
                mGrid->topologyUnion(*other.mGrid);
            }
        } else {
            tools::csgUnion(*mGrid, *other.mGrid, /*prune=*/true);
        }
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
        mMinCount += other.mMinCount;
        mMaxCount += other.mMaxCount;
    }

private:
    /// Disallow assignment since some of the members are references
    Raster& operator=(const Raster&) { return *this; }

    /// @return true if the particle is too small or too large
    bool ignoreParticle(Real R)
    {
        if (R < mParent.mRmin) {// below the cutoff radius
            ++mMinCount;
            return true;
        }
        if (R > mParent.mRmax) {// above the cutoff radius
            ++mMaxCount;
            return true;
        }
        return false;
    }

    /// @brief Threaded rasterization of particles as spheres with variable radius
    /// @param r  range of indices into the list of particles
    void rasterSpheres(const tbb::blocked_range<size_t>& r)
    {
        AccessorT acc = mGrid->getAccessor(); // local accessor
        bool run = true;
        const Real invDx = 1 / mParent.mDx;
        AttT att;
        Vec3R pos;
        Real rad;

        // Loop over buckets
        for (size_t n = r.begin(), N = r.end(); n < N; ++n) {
            // Loop over particles in bucket n.
            typename PointPartitionerT::IndexIterator iter = mPointPartitioner->indices(n);
            for ( ; run && iter; ++iter) {
                const Index32& id = *iter;
                mParticles.getPosRad(id, pos, rad);
                const Real R = invDx * rad;// in voxel units
                if (this->ignoreParticle(R)) continue;
                const Vec3R P = mMap.applyInverseMap(pos);
                this->getAtt<DisableT>(id, att);
                run = this->makeSphere(P, R, att, acc);
            }//end loop over particles
        }//end loop over buckets
    }

    /// @brief Threaded rasterization of particles as spheres with a fixed radius
    /// @param r  range of indices into the list of particles
    /// @param R  radius of fixed-size spheres
    void rasterFixedSpheres(const tbb::blocked_range<size_t>& r, Real R)
    {
        AccessorT acc = mGrid->getAccessor(); // local accessor
        AttT att;
        Vec3R pos;

        // Loop over buckets
        for (size_t n = r.begin(), N = r.end(); n < N; ++n) {
            // Loop over particles in bucket n.
            for (auto iter = mPointPartitioner->indices(n); iter; ++iter) {
                const Index32& id = *iter;
                this->getAtt<DisableT>(id, att);
                mParticles.getPos(id, pos);
                const Vec3R P = mMap.applyInverseMap(pos);
                this->makeSphere(P, R, att, acc);
            }
        }
    }

    /// @brief Threaded rasterization of particles as spheres with velocity trails
    /// @param r      range of indices into the list of particles
    /// @param delta  inter-sphere spacing
    void rasterTrails(const tbb::blocked_range<size_t>& r, Real delta)
    {
        AccessorT acc = mGrid->getAccessor(); // local accessor
        bool run = true;
        AttT att;
        Vec3R pos, vel;
        Real rad;
        const Vec3R origin = mMap.applyInverseMap(Vec3R(0,0,0));
        const Real Rmin = mParent.mRmin, invDx = 1 / mParent.mDx;

        // Loop over buckets
        for (size_t n = r.begin(), N = r.end(); n < N; ++n) {
            // Loop over particles in bucket n.
            typename PointPartitionerT::IndexIterator iter = mPointPartitioner->indices(n);
            for ( ; run && iter; ++iter) {
                const Index32& id = *iter;
                mParticles.getPosRadVel(id, pos, rad, vel);
                const Real R0 = invDx * rad;
                if (this->ignoreParticle(R0)) continue;
                this->getAtt<DisableT>(id, att);
                const Vec3R P0 = mMap.applyInverseMap(pos);
                const Vec3R V  = mMap.applyInverseMap(vel) - origin; // exclude translation
                const Real speed = V.length(), invSpeed = 1.0 / speed;
                const Vec3R Nrml = -V * invSpeed; // inverse normalized direction
                Vec3R P = P0; // local position of instance
                Real R = R0, d = 0; // local radius and length of trail
                for (size_t m = 0; run && d <= speed ; ++m) {
                    run = this->makeSphere(P, R, att, acc);
                    P += 0.5 * delta * R * Nrml; // adaptive offset along inverse velocity direction
                    d = (P - P0).length(); // current length of trail
                    R = R0 - (R0 - Rmin) * d * invSpeed; // R = R0 -> mRmin(e.g. 1.5)
                }//end loop over sphere instances
            }//end loop over particles
        }//end loop over buckets
    }

    void cook()
    {
        // parallelize over the point buckets
        const Index32 bucketCount = Index32(mPointPartitioner->size());

        if (mParent.mGrainSize>0) {
            tbb::parallel_reduce(
              tbb::blocked_range<size_t>(0, bucketCount, mParent.mGrainSize), *this);
        } else {
            (*this)(tbb::blocked_range<size_t>(0, bucketCount));
        }
    }

    /// @brief Rasterize sphere at position P and radius R.
    /// @return @c false if rasterization was interrupted
    ///
    /// @param P coordinates of the particle position in voxel units
    /// @param R radius of particle in voxel units
    /// @param att
    /// @param acc grid accessor with a private copy of the grid
    bool makeSphere(const Vec3R& P, Real R, const AttT& att, AccessorT& acc)
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (OutputIsMask) {
            return makeSphereMask(P, R, att, acc);
        } else {
            return makeNarrowBandSphere(P, R, att, acc);
        }
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }

    /// @brief Rasterize sphere at position P and radius R into
    /// a narrow-band level set with half-width, mHalfWidth.
    /// @return @c false if rasterization was interrupted
    ///
    /// @param P    coordinates of the particle position in voxel units
    /// @param R    radius of particle in voxel units
    /// @param att  an optional user-defined attribute value to be associated with voxels
    /// @param acc  grid accessor with a private copy of the grid
    ///
    /// @note For best performance all computations are performed in voxel space,
    /// with the important exception of the final level set value that is converted
    /// to world units (the grid stores the closest Euclidean signed distances
    /// measured in world units).  Also note we use the convention of positive distances
    /// outside the surface and negative distances inside the surface.
    bool makeNarrowBandSphere(const Vec3R& P, Real R, const AttT& att, AccessorT& acc)
    {
        const Real
            dx = mParent.mDx,
            w = mParent.mHalfWidth,
            max = R + w, // maximum distance in voxel units
            max2 = math::Pow2(max), // square of maximum distance in voxel units
            min2 = math::Pow2(math::Max(Real(0), R - w)); // square of minimum distance
        // Bounding box of the sphere
        const Coord
            lo(math::Floor(P[0]-max),math::Floor(P[1]-max),math::Floor(P[2]-max)),
            hi(math::Ceil( P[0]+max),math::Ceil( P[1]+max),math::Ceil( P[2]+max));
        const ValueT inside = -mGrid->background();

        ValueT v;
        size_t count = 0;
        for (Coord c = lo; c.x() <= hi.x(); ++c.x()) {
            //only check interrupter every 32'th scan in x
            if (!(count++ & ((1<<5)-1)) && util::wasInterrupted(mParent.mInterrupter)) {
                tbb::task::self().cancel_group_execution();
                return false;
            }
            const Real x2 = math::Pow2(c.x() - P[0]);
            for (c.y() = lo.y(); c.y() <= hi.y(); ++c.y()) {
                const Real x2y2 = x2 + math::Pow2(c.y() - P[1]);
                for (c.z() = lo.z(); c.z() <= hi.z(); ++c.z()) {
                    const Real x2y2z2 = x2y2 + math::Pow2(c.z()-P[2]); // squared dist from c to P
                    if (x2y2z2 >= max2 || (!acc.probeValue(c, v) && (v < ValueT(0))))
                        continue;//outside narrow band of the particle or inside existing level set
                    if (x2y2z2 <= min2) {//inside narrow band of the particle.
                        acc.setValueOff(c, inside);
                        continue;
                    }
                    // convert signed distance from voxel units to world units
                    //const ValueT d=dx*(math::Sqrt(x2y2z2) - R);
                    const ValueT d = Merge(static_cast<SdfT>(dx*(math::Sqrt(x2y2z2)-R)), att);
                    if (d < v) acc.setValue(c, d);//CSG union
                }//end loop over z
            }//end loop over y
        }//end loop over x
        return true;
    }

    /// @brief Rasterize a sphere of radius @a r at position @a p into a boolean mask grid.
    /// @return @c false if rasterization was interrupted
    bool makeSphereMask(const Vec3R& p, Real r, const AttT& att, AccessorT& acc)
    {
        const Real
            rSquared = r * r, // sphere radius squared, in voxel units
            inW = r / math::Sqrt(6.0); // half the width in voxel units of an inscribed cube
        const Coord
            // Bounding box of the sphere
            outLo(math::Floor(p[0] - r), math::Floor(p[1] - r), math::Floor(p[2] - r)),
            outHi(math::Ceil(p[0] + r),  math::Ceil(p[1] + r),  math::Ceil(p[2] + r)),
            // Bounds of the inscribed cube
            inLo(math::Ceil(p[0] - inW), math::Ceil(p[1] - inW), math::Ceil(p[2] - inW)),
            inHi(math::Floor(p[0] + inW),  math::Floor(p[1] + inW),  math::Floor(p[2] + inW));
        // Bounding boxes of regions comprising out - in
        /// @todo These could be divided further into sparsely- and densely-filled subregions.
        const std::vector<CoordBBox> padding{
            CoordBBox(outLo.x(),  outLo.y(),  outLo.z(),  inLo.x()-1, outHi.y(),  outHi.z()),
            CoordBBox(inHi.x()+1, outLo.y(),  outLo.z(),  outHi.x(),  outHi.y(),  outHi.z()),
            CoordBBox(outLo.x(),  outLo.y(),  outLo.z(),  outHi.x(),  inLo.y()-1, outHi.z()),
            CoordBBox(outLo.x(),  inHi.y()+1, outLo.z(),  outHi.x(),  outHi.y(),  outHi.z()),
            CoordBBox(outLo.x(),  outLo.y(),  outLo.z(),  outHi.x(),  outHi.y(),  inLo.z()-1),
            CoordBBox(outLo.x(),  outLo.y(),  inHi.z()+1, outHi.x(),  outHi.y(),  outHi.z()),
        };
        const ValueT onValue = Merge(SdfT(1), att);

        // Sparsely fill the inscribed cube.
        /// @todo Use sparse fill only if 2r > leaf width?
        acc.tree().sparseFill(CoordBBox(inLo, inHi), onValue);

        // Densely fill the remaining regions.
        for (const auto& bbox: padding) {
            if (util::wasInterrupted(mParent.mInterrupter)) {
                tbb::task::self().cancel_group_execution();
                return false;
            }
            const Coord &bmin = bbox.min(), &bmax = bbox.max();
            Coord c;
            Real cx, cy, cz;
            for (c = bmin, cx = c.x(); c.x() <= bmax.x(); ++c.x(), cx += 1) {
                const Real x2 = math::Pow2(cx - p[0]);
                for (c.y() = bmin.y(), cy = c.y(); c.y() <= bmax.y(); ++c.y(), cy += 1) {
                    const Real x2y2 = x2 + math::Pow2(cy - p[1]);
                    for (c.z() = bmin.z(), cz = c.z(); c.z() <= bmax.z(); ++c.z(), cz += 1) {
                        const Real x2y2z2 = x2y2 + math::Pow2(cz - p[2]);
                        if (x2y2z2 < rSquared) {
                            acc.setValue(c, onValue);
                        }
                    }
                }
            }
        }
        return true;
    }

    using FuncType = typename std::function<void (Raster*, const tbb::blocked_range<size_t>&)>;

    template<typename DisableType>
    typename std::enable_if<DisableType::value>::type
    getAtt(size_t, AttT&) const {}

    template<typename DisableType>
    typename std::enable_if<!DisableType::value>::type
    getAtt(size_t n, AttT& a) const { mParticles.getAtt(n, a); }

    template<typename T>
    typename std::enable_if<std::is_same<T, ValueT>::value, ValueT>::type
    Merge(T s, const AttT&) const { return s; }

    template<typename T>
    typename std::enable_if<!std::is_same<T, ValueT>::value, ValueT>::type
    Merge(T s, const AttT& a) const { return ValueT(s,a); }

    ParticlesToLevelSetT& mParent;
    const ParticleListT&  mParticles;//list of particles
    GridT*                mGrid;
    const math::MapBase&  mMap;
    size_t                mMinCount, mMaxCount;//counters for ignored particles!
    FuncType              mTask;
    const bool            mIsCopy;
    PointPartitionerT*    mPointPartitioner;
}; // struct ParticlesToLevelSet::Raster


///////////////////// YOU CAN SAFELY IGNORE THIS SECTION /////////////////////


namespace p2ls_internal {

// This is a simple type that combines a distance value and a particle
// attribute. It's required for attribute transfer which is defined in the
// Raster class above.
/// @private
template<typename VisibleT, typename BlindT>
class BlindData
{
public:
    using type = VisibleT;
    using VisibleType = VisibleT;
    using BlindType = BlindT;

    BlindData() {}
    explicit BlindData(VisibleT v) : mVisible(v), mBlind(zeroVal<BlindType>()) {}
    BlindData(VisibleT v, BlindT b) : mVisible(v), mBlind(b) {}
    BlindData(const BlindData&) = default;
    BlindData& operator=(const BlindData&) = default;
    const VisibleT& visible() const { return mVisible; }
    const BlindT&   blind()   const { return mBlind; }
    OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN
    bool operator==(const BlindData& rhs)     const { return mVisible == rhs.mVisible; }
    OPENVDB_NO_FP_EQUALITY_WARNING_END
    bool operator< (const BlindData& rhs)     const { return mVisible <  rhs.mVisible; }
    bool operator> (const BlindData& rhs)     const { return mVisible >  rhs.mVisible; }
    BlindData operator+(const BlindData& rhs) const { return BlindData(mVisible + rhs.mVisible); }
    BlindData operator+(const VisibleT&  rhs) const { return BlindData(mVisible + rhs); }
    BlindData operator-(const BlindData& rhs) const { return BlindData(mVisible - rhs.mVisible); }
    BlindData operator-() const { return BlindData(-mVisible, mBlind); }

protected:
    VisibleT mVisible;
    BlindT   mBlind;
};

/// @private
// Required by several of the tree nodes
template<typename VisibleT, typename BlindT>
inline std::ostream& operator<<(std::ostream& ostr, const BlindData<VisibleT, BlindT>& rhs)
{
    ostr << rhs.visible();
    return ostr;
}

/// @private
// Required by math::Abs
template<typename VisibleT, typename BlindT>
inline BlindData<VisibleT, BlindT> Abs(const BlindData<VisibleT, BlindT>& x)
{
    return BlindData<VisibleT, BlindT>(math::Abs(x.visible()), x.blind());
}

} // namespace p2ls_internal


//////////////////////////////////////////////////////////////////////////////


// The following are convenience functions for common use cases.

template<typename GridT, typename ParticleListT, typename InterrupterT>
inline void
particlesToSdf(const ParticleListT& plist, GridT& grid, InterrupterT* interrupt)
{
    static_assert(std::is_floating_point<typename GridT::ValueType>::value,
        "particlesToSdf requires an SDF grid with floating-point values");

    if (grid.getGridClass() != GRID_LEVEL_SET) {
        OPENVDB_LOG_WARN("particlesToSdf requires a level set grid;"
            " try Grid::setGridClass(openvdb::GRID_LEVEL_SET)");
    }

    ParticlesToLevelSet<GridT> p2ls(grid, interrupt);
    p2ls.rasterizeSpheres(plist);
    tools::pruneLevelSet(grid.tree());
}

template<typename GridT, typename ParticleListT, typename InterrupterT>
inline void
particlesToSdf(const ParticleListT& plist, GridT& grid, Real radius, InterrupterT* interrupt)
{
    static_assert(std::is_floating_point<typename GridT::ValueType>::value,
        "particlesToSdf requires an SDF grid with floating-point values");

    if (grid.getGridClass() != GRID_LEVEL_SET) {
        OPENVDB_LOG_WARN("particlesToSdf requires a level set grid;"
            " try Grid::setGridClass(openvdb::GRID_LEVEL_SET)");
    }

    ParticlesToLevelSet<GridT> p2ls(grid, interrupt);
    p2ls.rasterizeSpheres(plist, radius);
    tools::pruneLevelSet(grid.tree());
}

template<typename GridT, typename ParticleListT, typename InterrupterT>
inline void
particleTrailsToSdf(const ParticleListT& plist, GridT& grid, Real delta, InterrupterT* interrupt)
{
    static_assert(std::is_floating_point<typename GridT::ValueType>::value,
        "particleTrailsToSdf requires an SDF grid with floating-point values");

    if (grid.getGridClass() != GRID_LEVEL_SET) {
        OPENVDB_LOG_WARN("particlesToSdf requires a level set grid;"
            " try Grid::setGridClass(openvdb::GRID_LEVEL_SET)");
    }

    ParticlesToLevelSet<GridT> p2ls(grid, interrupt);
    p2ls.rasterizeTrails(plist, delta);
    tools::pruneLevelSet(grid.tree());
}

template<typename GridT, typename ParticleListT, typename InterrupterT>
inline void
particlesToMask(const ParticleListT& plist, GridT& grid, InterrupterT* interrupt)
{
    static_assert(std::is_same<bool, typename GridT::ValueType>::value,
        "particlesToMask requires a boolean-valued grid");
    ParticlesToLevelSet<GridT> p2ls(grid, interrupt);
    p2ls.rasterizeSpheres(plist);
    tools::prune(grid.tree());
}

template<typename GridT, typename ParticleListT, typename InterrupterT>
inline void
particlesToMask(const ParticleListT& plist, GridT& grid, Real radius, InterrupterT* interrupt)
{
    static_assert(std::is_same<bool, typename GridT::ValueType>::value,
        "particlesToMask requires a boolean-valued grid");
    ParticlesToLevelSet<GridT> p2ls(grid, interrupt);
    p2ls.rasterizeSpheres(plist, radius);
    tools::prune(grid.tree());
}

template<typename GridT, typename ParticleListT, typename InterrupterT>
inline void
particleTrailsToMask(const ParticleListT& plist, GridT& grid, Real delta, InterrupterT* interrupt)
{
    static_assert(std::is_same<bool, typename GridT::ValueType>::value,
        "particleTrailsToMask requires a boolean-valued grid");
    ParticlesToLevelSet<GridT> p2ls(grid, interrupt);
    p2ls.rasterizeTrails(plist, delta);
    tools::prune(grid.tree());
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_PARTICLES_TO_LEVELSET_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
