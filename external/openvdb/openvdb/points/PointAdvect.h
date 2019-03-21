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

/// @author Dan Bailey
///
/// @file points/PointAdvect.h
///
/// @brief Ability to advect VDB Points through a velocity field.

#ifndef OPENVDB_POINTS_POINT_ADVECT_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_ADVECT_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/VelocityFields.h>

#include <openvdb/points/AttributeGroup.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointMove.h>

#include <memory>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


/// @brief Advect points in a PointDataGrid through a velocity grid
/// @param points               the PointDataGrid containing the points to be advected.
/// @param velocity             a velocity grid to be sampled.
/// @param integrationOrder     the integration scheme to use (1 is forward euler, 4 is runge-kutta 4th)
/// @param dt                   delta time.
/// @param timeSteps            number of advection steps to perform.
/// @param advectFilter         an optional advection index filter (moves a subset of the points)
/// @param filter               an optional index filter (deletes a subset of the points)
/// @param cached               caches velocity interpolation for faster performance, disable to use
///                             less memory (default is on).
template <typename PointDataGridT, typename VelGridT,
    typename AdvectFilterT = NullFilter, typename FilterT = NullFilter>
inline void advectPoints(PointDataGridT& points, const VelGridT& velocity,
                         const Index integrationOrder, const double dt, const Index timeSteps,
                         const AdvectFilterT& advectFilter = NullFilter(),
                         const FilterT& filter = NullFilter(),
                         const bool cached = true);


////////////////////////////////////////


namespace point_advect_internal {

enum IntegrationOrder {
    INTEGRATION_ORDER_FWD_EULER = 1,
    INTEGRATION_ORDER_RK_2ND,
    INTEGRATION_ORDER_RK_3RD,
    INTEGRATION_ORDER_RK_4TH
};

template <typename VelGridT, Index IntegrationOrder, bool Staggered, typename FilterT>
class AdvectionDeformer
{
public:
    using IntegratorT = openvdb::tools::VelocityIntegrator<VelGridT, Staggered>;

    AdvectionDeformer(const VelGridT& velocityGrid, const double timeStep, const int steps,
                      const FilterT& filter)
        : mIntegrator(velocityGrid)
        , mTimeStep(timeStep)
        , mSteps(steps)
        , mFilter(filter) { }

    template <typename LeafT>
    void reset(const LeafT& leaf, size_t /*idx*/)
    {
        mFilter.reset(leaf);
    }

    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT& iter) const
    {
        if (mFilter.valid(iter)) {
            for (int n = 0; n < mSteps; ++n) {
                mIntegrator.template rungeKutta<IntegrationOrder, openvdb::Vec3d>(
                    static_cast<typename IntegratorT::ElementType>(mTimeStep), position);
            }
        }
    }

private:
    IntegratorT mIntegrator;
    double mTimeStep;
    const int mSteps;
    FilterT mFilter;
}; // class AdvectionDeformer


template <typename PointDataGridT, typename VelGridT, typename AdvectFilterT, typename FilterT>
struct AdvectionOp
{
    using CachedDeformerT = CachedDeformer<double>;

    AdvectionOp(PointDataGridT& points, const VelGridT& velocity,
                const Index integrationOrder, const double timeStep, const Index steps,
                const AdvectFilterT& advectFilter,
                const FilterT& filter)
        : mPoints(points)
        , mVelocity(velocity)
        , mIntegrationOrder(integrationOrder)
        , mTimeStep(timeStep)
        , mSteps(steps)
        , mAdvectFilter(advectFilter)
        , mFilter(filter) { }

    void cache()
    {
        mCachedDeformer.reset(new CachedDeformerT(mCache));
        (*this)(true);
    }

    void advect()
    {
        (*this)(false);
    }

private:
    template <int IntegrationOrder, bool Staggered>
    void resolveIntegrationOrder(bool buildCache)
    {
        const auto leaf = mPoints.constTree().cbeginLeaf();
        if (!leaf)  return;

        // move points according to the pre-computed cache
        if (!buildCache && mCachedDeformer) {
            movePoints(mPoints, *mCachedDeformer, mFilter);
            return;
        }

        NullFilter nullFilter;

        if (buildCache) {
            // disable group filtering from the advection deformer and perform group filtering
            // in the cache deformer instead, this restricts the cache to just containing
            // positions from points which are both deforming *and* are not being deleted
            AdvectionDeformer<VelGridT, IntegrationOrder, Staggered, NullFilter> deformer(
                mVelocity, mTimeStep, mSteps, nullFilter);
            if (mFilter.state() == index::ALL && mAdvectFilter.state() == index::ALL) {
                mCachedDeformer->evaluate(mPoints, deformer, nullFilter);
            } else {
                BinaryFilter<AdvectFilterT, FilterT, /*And=*/true> binaryFilter(
                    mAdvectFilter, mFilter);
                mCachedDeformer->evaluate(mPoints, deformer, binaryFilter);
            }
        }
        else {
            // revert to NullFilter if all points are being evaluated
            if (mAdvectFilter.state() == index::ALL) {
                AdvectionDeformer<VelGridT, IntegrationOrder, Staggered, NullFilter> deformer(
                    mVelocity, mTimeStep, mSteps, nullFilter);
                movePoints(mPoints, deformer, mFilter);
            }
            else {
                AdvectionDeformer<VelGridT, IntegrationOrder, Staggered, AdvectFilterT> deformer(
                    mVelocity, mTimeStep, mSteps, mAdvectFilter);
                movePoints(mPoints, deformer, mFilter);
            }
        }
    }

    template <bool Staggered>
    void resolveStaggered(bool buildCache)
    {
        if (mIntegrationOrder == INTEGRATION_ORDER_FWD_EULER) {
            resolveIntegrationOrder<1, Staggered>(buildCache);
        } else if (mIntegrationOrder == INTEGRATION_ORDER_RK_2ND) {
            resolveIntegrationOrder<2, Staggered>(buildCache);
        } else if (mIntegrationOrder == INTEGRATION_ORDER_RK_3RD) {
            resolveIntegrationOrder<3, Staggered>(buildCache);
        } else if (mIntegrationOrder == INTEGRATION_ORDER_RK_4TH) {
            resolveIntegrationOrder<4, Staggered>(buildCache);
        }
    }

    void operator()(bool buildCache)
    {
        // early-exit if no leafs
        if (mPoints.constTree().leafCount() == 0)            return;

        if (mVelocity.getGridClass() == openvdb::GRID_STAGGERED) {
            resolveStaggered<true>(buildCache);
        } else {
            resolveStaggered<false>(buildCache);
        }
    }

    PointDataGridT& mPoints;
    const VelGridT& mVelocity;
    const Index mIntegrationOrder;
    const double mTimeStep;
    const Index mSteps;
    const AdvectFilterT& mAdvectFilter;
    const FilterT& mFilter;
    CachedDeformerT::Cache mCache;
    std::unique_ptr<CachedDeformerT> mCachedDeformer;
}; // struct AdvectionOp

} // namespace point_advect_internal


////////////////////////////////////////


template <typename PointDataGridT, typename VelGridT, typename AdvectFilterT, typename FilterT>
inline void advectPoints(PointDataGridT& points, const VelGridT& velocity,
                         const Index integrationOrder, const double timeStep, const Index steps,
                         const AdvectFilterT& advectFilter,
                         const FilterT& filter,
                         const bool cached)
{
    using namespace point_advect_internal;

    if (steps == 0)     return;

    if (integrationOrder > 4) {
        throw ValueError{"Unknown integration order for advecting points."};
    }

    AdvectionOp<PointDataGridT, VelGridT, AdvectFilterT, FilterT> op(
        points, velocity, integrationOrder, timeStep, steps,
        advectFilter, filter);

    // if caching is enabled, sample the velocity field using a CachedDeformer to store the
    // intermediate positions before moving the points, this uses more memory but typically
    // results in faster overall performance
    if (cached)     op.cache();

    // advect the points
    op.advect();
}

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_ADVECT_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
