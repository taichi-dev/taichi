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
/// @file tools/PointsToMask.h
///
/// @brief This tool produces a grid where every voxel that contains a
/// point is active. It employes thread-local storage for best performance.
///
/// The @c PointListT template argument below refers to any class
/// with the following interface (see unittest/TestPointsToMask.cc
/// and SOP_OpenVDB_From_Particles.cc for practical examples):
/// @code
///
/// class PointList {
///   ...
/// public:
///
///   // Return the total number of particles in list.
///   size_t size() const;
///
///   // Get the world space position of the nth particle.
///   void getPos(size_t n, Vec3R& xyz) const;
/// };
/// @endcode
///
/// @note See unittest/TestPointsToMask.cc for an example.
///
/// The @c InterruptT template argument below refers to any class
/// with the following interface:
/// @code
/// class Interrupter {
///   ...
/// public:
///   void start(const char* name = nullptr) // called when computations begin
///   void end()                             // called when computations end
///   bool wasInterrupted(int percent = -1)  // return true to break computation
/// };
/// @endcode
///
/// @note If no template argument is provided for this InterruptT
/// the util::NullInterrupter is used which implies that all
/// interrupter calls are no-ops (i.e. incurs no computational overhead).

#ifndef OPENVDB_TOOLS_POINTSTOMASK_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINTSTOMASK_HAS_BEEN_INCLUDED

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <openvdb/openvdb.h> // for MaskGrid
#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/util/NullInterrupter.h>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

// Forward declaration of main class
template<typename GridT = MaskGrid, typename InterrupterT = util::NullInterrupter>
class PointsToMask;

/// @brief Makes every voxel of the  @c grid active if it contains a point.
///
/// @param points     points that active the voxels of @c grid
/// @param grid       on out its voxels with points are active
template<typename PointListT, typename GridT>
inline void
maskPoints(const PointListT& points, GridT& grid)
{
    PointsToMask<GridT, util::NullInterrupter> tmp(grid, nullptr);
    tmp.addPoints(points);
}

/// @brief Return a MaskGrid where each binary voxel value
/// is on if the voxel contains one (or more) points (i.e.
/// the 3D position of a point is closer to this voxel than
/// any other voxels).
///
/// @param points     points that active the voxels in the returned grid.
/// @param xform      transform from world space to voxels in grid space.
template<typename PointListT>
inline MaskGrid::Ptr
createPointMask(const PointListT& points, const math::Transform& xform)
{
    MaskGrid::Ptr grid = createGrid<MaskGrid>( false );
    grid->setTransform( xform.copy() );
    maskPoints( points, *grid );
    return grid;
}

////////////////////////////////////////

/// @brief Makes every voxel of a grid active if it contains a point.
template<typename GridT, typename InterrupterT>
class PointsToMask
{
public:
    using ValueT = typename GridT::ValueType;

    /// @brief Constructor from a grid and optional interrupter
    ///
    /// @param grid        Grid whoes voxels will have their state activated by points.
    /// @param interrupter Optional interrupter to prematurely terminate execution.
    explicit PointsToMask(GridT& grid, InterrupterT* interrupter = nullptr)
        : mGrid(&grid)
        , mInterrupter(interrupter)
    {
    }

    /// @brief Activates the state of any voxel in the input grid that contains a point.
    ///
    /// @param points    List of points that active the voxels in the input grid.
    /// @param grainSize Set the grain-size used for multi-threading. A value of 0
    ///                  disables multi-threading!
    template<typename PointListT>
    void addPoints(const PointListT& points, size_t grainSize = 1024)
    {
        if (mInterrupter) mInterrupter->start("PointsToMask: adding points");
        if (grainSize > 0) {
#if OPENVDB_ABI_VERSION_NUMBER <= 3
            typename GridT::Ptr examplar = mGrid->copy(CP_NEW);
#else
            typename GridT::Ptr examplar = mGrid->copyWithNewTree();
#endif
            PoolType pool( *examplar );//thread local storage pool of grids
            AddPoints<PointListT> tmp(points, pool, grainSize, *this );
            if ( this->interrupt() ) return;
            ReducePool reducePool(pool, mGrid, size_t(0));
        } else {
            const math::Transform& xform = mGrid->transform();
            typename GridT::Accessor acc = mGrid->getAccessor();
            Vec3R wPos;
            for (size_t i = 0, n = points.size(); i < n; ++i) {
                if ( this->interrupt() ) break;
                points.getPos(i, wPos);
                acc.setValueOn( xform.worldToIndexCellCentered( wPos ) );
            }
        }
        if (mInterrupter) mInterrupter->end();
    }

private:
    // Disallow copy construction and copy by assignment!
    PointsToMask(const PointsToMask&);// not implemented
    PointsToMask& operator=(const PointsToMask&);// not implemented

    bool interrupt() const
    {
        if (mInterrupter && util::wasInterrupted(mInterrupter)) {
            tbb::task::self().cancel_group_execution();
            return true;
        }
        return false;
    }

    // Private struct that implements concurrent thread-local
    // insersion of points into a grid
    using PoolType = tbb::enumerable_thread_specific<GridT>;
    template<typename PointListT> struct AddPoints;

    // Private class that implements concurrent reduction of a thread-local pool
    struct ReducePool;

    GridT*        mGrid;
    InterrupterT* mInterrupter;
};// PointsToMask

// Private member class that implements concurrent thread-local
// insersion of points into a grid
template<typename GridT, typename InterrupterT>
template<typename PointListT>
struct PointsToMask<GridT, InterrupterT>::AddPoints
{
    AddPoints(const PointListT& points,
              PoolType& pool,
              size_t grainSize,
              const PointsToMask& parent)
        : mPoints(&points)
        , mParent(&parent)
        , mPool(&pool)
    {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mPoints->size(), grainSize), *this);
    }
    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        if (mParent->interrupt()) return;
        GridT& grid = mPool->local();
        const math::Transform& xform = grid.transform();
        typename GridT::Accessor acc = grid.getAccessor();
        Vec3R wPos;
        for (size_t i=range.begin(), n=range.end(); i!=n; ++i) {
            mPoints->getPos(i, wPos);
            acc.setValueOn( xform.worldToIndexCellCentered( wPos ) );
        }
    }
    const PointListT*    mPoints;
    const PointsToMask* mParent;
    PoolType*            mPool;

};// end of private member class AddPoints

// Private member class that implements concurrent reduction of a thread-local pool
template<typename GridT, typename InterrupterT>
struct PointsToMask<GridT, InterrupterT>::ReducePool
{
    using VecT = std::vector<GridT*>;
    using IterT = typename VecT::iterator;
    using RangeT = tbb::blocked_range<IterT>;

    ReducePool(PoolType& pool, GridT* grid, size_t grainSize = 1)
        : mOwnsGrid(false)
        , mGrid(grid)
    {
        if ( grainSize == 0 ) {
            using IterT = typename PoolType::const_iterator;
            for (IterT i=pool.begin(); i!=pool.end(); ++i) mGrid->topologyUnion( *i );
        } else {
            VecT grids( pool.size() );
            typename PoolType::iterator i = pool.begin();
            for (size_t j=0; j != pool.size(); ++i, ++j) grids[j] = &(*i);
            tbb::parallel_reduce( RangeT( grids.begin(), grids.end(), grainSize ), *this );
        }
    }

    ReducePool(const ReducePool&, tbb::split)
        : mOwnsGrid(true)
        , mGrid(new GridT())
    {
    }

    ~ReducePool() { if (mOwnsGrid) delete mGrid; }

    void operator()(const RangeT& r)
    {
        for (IterT i=r.begin(); i!=r.end(); ++i) mGrid->topologyUnion( *(*i) );
    }

    void join(ReducePool& other) { mGrid->topologyUnion(*other.mGrid); }

    const bool mOwnsGrid;
    GridT*     mGrid;
};// end of private member class ReducePool

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_POINTSTOMASK_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
