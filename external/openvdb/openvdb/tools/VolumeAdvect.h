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
//
///////////////////////////////////////////////////////////////////////////
//
/// @author Ken Museth
///
/// @file tools/VolumeAdvect.h
///
/// @brief Sparse hyperbolic advection of volumes, e.g. a density or
///        velocity (vs a level set interface).

#ifndef OPENVDB_TOOLS_VOLUME_ADVECT_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VOLUME_ADVECT_HAS_BEEN_INCLUDED

#include <tbb/parallel_for.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>
#include <openvdb/util/NullInterrupter.h>
#include "Interpolation.h"// for Sampler
#include "VelocityFields.h" // for VelocityIntegrator
#include "Morphology.h"//for dilateActiveValues and dilateVoxels
#include "Prune.h"// for prune
#include "Statistics.h" // for extrema
#include <functional>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

namespace Scheme {
    /// @brief Numerical advections schemes.
    enum SemiLagrangian { SEMI, MID, RK3, RK4, MAC, BFECC };
    /// @brief Flux-limiters employed to stabalize the second-order
    /// advection schemes MacCormack and BFECC.
    enum Limiter { NO_LIMITER, CLAMP, REVERT };
}

/// @brief Performs advections of an arbitrary type of volume in a
///        static velocity field. The advections are performed by means
///        of various derivatives of Semi-Lagrangian integration, i.e.
///        backwards tracking along the hyperbolic characteristics
///        followed by interpolation.
///
/// @note  Optionally a limiter can be combined with the higher-order
///        integration schemes MacCormack and BFECC. There are two
///        types of limiters (CLAMP and REVERT) that supress
///        non-physical oscillations by means of either claminging or
///        reverting to a first-order schemes when the function is not
///        bounded by the cell values used for tri-linear interpolation.
///
/// @verbatim The supported integrations schemes:
///
///    ================================================================
///    |  Lable | Accuracy |  Integration Scheme   |  Interpolations  |
///    |        |Time/Space|                       |  velocity/volume |
///    ================================================================
///    |  SEMI  |   1/1    | Semi-Lagrangian       |        1/1       |
///    |  MID   |   2/1    | Mid-Point             |        2/1       |
///    |  RK3   |   3/1    | 3rd Order Runge-Kutta |        3/1       |
///    |  RK4   |   4/1    | 4th Order Runge-Kutta |        4/1       |
///    |  MAC   |   2/2    | MacCormack            |        2/2       |
///    |  BFECC |   2/2    | BFECC                 |        3/2       |
///    ================================================================
/// @endverbatim

template<typename VelocityGridT = Vec3fGrid,
         bool StaggeredVelocity = false,
         typename InterrupterType = util::NullInterrupter>
class VolumeAdvection
{
public:

    /// @brief Constructor
    ///
    /// @param velGrid     Velocity grid responsible for the (passive) advection.
    /// @param interrupter Optional interrupter used to prematurely end computations.
    ///
    /// @note The velocity field is assumed to be constant for the duration of the
    ///       advection.
    VolumeAdvection(const VelocityGridT& velGrid, InterrupterType* interrupter = nullptr)
        : mVelGrid(velGrid)
        , mInterrupter(interrupter)
        , mIntegrator( Scheme::SEMI )
        , mLimiter( Scheme::CLAMP )
        , mGrainSize( 128 )
        , mSubSteps( 1 )
    {
        math::Extrema e = extrema(velGrid.cbeginValueAll(), /*threading*/true);
        e.add(velGrid.background().length());
        mMaxVelocity = e.max();
    }

    virtual ~VolumeAdvection()
    {
    }

    /// @brief Return the spatial order of accuracy of the advection scheme
    ///
    /// @note This is the optimal order in smooth regions. In
    /// non-smooth regions the flux-limiter will drop the order of
    /// accuracy to add numerical dissipation.
    int spatialOrder() const { return (mIntegrator == Scheme::MAC ||
                                       mIntegrator == Scheme::BFECC) ? 2 : 1; }

    /// @brief Return the temporal order of accuracy of the advection scheme
    ///
    /// @note This is the optimal order in smooth regions. In
    /// non-smooth regions the flux-limiter will drop the order of
    /// accuracy to add numerical dissipation.
    int temporalOrder() const {
        switch (mIntegrator) {
        case Scheme::SEMI: return 1;
        case Scheme::MID:  return 2;
        case Scheme::RK3:  return 3;
        case Scheme::RK4:  return 4;
        case Scheme::BFECC:return 2;
        case Scheme::MAC:  return 2;
        }
        return 0;//should never reach this point
    }

    /// @brief Set the integrator (see details in the table above)
    void setIntegrator(Scheme::SemiLagrangian integrator) { mIntegrator = integrator; }

    /// @brief Return the integrator (see details in the table above)
    Scheme::SemiLagrangian getIntegrator() const { return mIntegrator; }

    /// @brief Set the limiter (see details above)
    void setLimiter(Scheme::Limiter limiter) { mLimiter = limiter; }

    /// @brief Retrun the limiter (see details above)
    Scheme::Limiter getLimiter() const { return mLimiter; }

    /// @brief Return @c true if a limiter will be applied based on
    /// the current settings.
    bool isLimiterOn() const { return this->spatialOrder()>1 &&
                                      mLimiter != Scheme::NO_LIMITER; }

    /// @return the grain-size used for multi-threading
    /// @note A grainsize of 0 implies serial execution
    size_t getGrainSize() const { return mGrainSize; }

    /// @brief Set the grain-size used for multi-threading
    /// @note A grainsize of 0 disables multi-threading
    /// @warning A small grainsize can degrade performance,
    ///          both in terms of time and memory footprint!
    void setGrainSize(size_t grainsize) { mGrainSize = grainsize; }

    /// @return the number of sub-steps per integration (always larger
    /// than or equal to 1).
    int getSubSteps() const { return mSubSteps; }

    /// @brief Set the number of sub-steps per integration.
    /// @note The only reason to increase the sub-step above its
    ///       default value of one is to reduce the memory footprint
    ///       due to significant dilation. Values smaller than 1 will
    ///       be clamped to 1!
    void setSubSteps(int substeps) { mSubSteps = math::Max(1, substeps); }

    /// @brief Return the maximum magnitude of the velocity in the
    /// advection velocity field defined during construction.
    double getMaxVelocity() const { return mMaxVelocity; }

    /// @return Returns the maximum distance in voxel units of @a inGrid
    /// that a particle can travel in the time-step @a dt when advected
    /// in the velocity field defined during construction.
    ///
    /// @details This method is useful when dilating sparse volume
    /// grids to pad boundary regions. Excessive dilation can be
    /// computationally expensive so use this method to prevent
    /// or warn against run-away computation.
    ///
    /// @throw RuntimeError if @a inGrid does not have uniform voxels.
    template<typename VolumeGridT>
    int getMaxDistance(const VolumeGridT& inGrid, double dt) const
    {
        if (!inGrid.hasUniformVoxels()) {
            OPENVDB_THROW(RuntimeError, "Volume grid does not have uniform voxels!");
        }
        const double d = mMaxVelocity*math::Abs(dt)/inGrid.voxelSize()[0];
        return static_cast<int>( math::RoundUp(d) );
    }

    /// @return Returns a new grid that is the result of passive advection
    ///         of all the active values the input grid by @a timeStep.
    ///
    /// @param inGrid   The input grid to be advected (unmodified)
    /// @param timeStep Time-step of the Runge-Kutta integrator.
    ///
    /// @details This method will advect all of the active values in
    ///          the input @a inGrid. To achieve this a
    ///          deep-copy is dilated to account for the material
    ///          transport. This dilation step can be slow for large
    ///          time steps @a dt or a velocity field with large magnitudes.
    ///
    /// @warning If the VolumeSamplerT is of higher order than one
    ///          (i.e. tri-linear interpolation) instabilities are
    ///          known to occure. To suppress those monotonicity
    ///          constrains or flux-limiters need to be applies.
    ///
    /// @throw RuntimeError if @a inGrid does not have uniform voxels.
    template<typename VolumeGridT,
             typename VolumeSamplerT>//only C++11 allows for a default argument
    typename VolumeGridT::Ptr advect(const VolumeGridT& inGrid, double timeStep)
    {
        typename VolumeGridT::Ptr outGrid = inGrid.deepCopy();
        const double dt = timeStep/mSubSteps;
        const int n = this->getMaxDistance(inGrid, dt);
        dilateActiveValues( outGrid->tree(), n, NN_FACE, EXPAND_TILES);
        this->template cook<VolumeGridT, VolumeSamplerT>(*outGrid, inGrid, dt);
        for (int step = 1; step < mSubSteps; ++step) {
            typename VolumeGridT::Ptr tmpGrid = outGrid->deepCopy();
            dilateActiveValues( tmpGrid->tree(), n, NN_FACE, EXPAND_TILES);
            this->template cook<VolumeGridT, VolumeSamplerT>(*tmpGrid, *outGrid, dt);
            outGrid.swap( tmpGrid );
        }

        return outGrid;
    }

    /// @return Returns a new grid that is the result of
    ///         passive advection of the active values in @a inGrid
    ///         that intersect the active values in @c mask. The time
    ///         of the output grid is incremented by @a timeStep.
    ///
    /// @param inGrid   The input grid to be advected (unmodified).
    /// @param mask     The mask of active values defining the active voxels
    ///                 in @c inGrid on which to perform advection. Only
    ///                 if a value is active in both grids will it be modified.
    /// @param timeStep Time-step for a single Runge-Kutta integration step.
    ///
    ///
    /// @details This method will advect all of the active values in
    ///          the input @a inGrid that intersects with the
    ///          active values in @a mask. To achieve this a
    ///          deep-copy is dilated to account for the material
    ///          transport and finally cropped to the intersection
    ///          with @a mask. The dilation step can be slow for large
    ///          time steps @a dt or fast moving velocity fields.
    ///
    /// @warning If the VolumeSamplerT is of higher order the one
    ///          (i.e. tri-linear interpolation) instabilities are
    ///          known to occure. To suppress those monotonicity
    ///          constrains or flux-limiters need to be applies.
    ///
    /// @throw RuntimeError if @a inGrid is not aligned with @a mask
    ///        or if its voxels are not uniform.
    template<typename VolumeGridT,
             typename MaskGridT,
             typename VolumeSamplerT>//only C++11 allows for a default argument
    typename VolumeGridT::Ptr advect(const VolumeGridT& inGrid, const MaskGridT& mask, double timeStep)
    {
        if (inGrid.transform() != mask.transform()) {
            OPENVDB_THROW(RuntimeError, "Volume grid and mask grid are misaligned! Consider "
                          "resampling either of the two grids into the index space of the other.");
        }
        typename VolumeGridT::Ptr outGrid = inGrid.deepCopy();
        const double dt = timeStep/mSubSteps;
        const int n = this->getMaxDistance(inGrid, dt);
        dilateActiveValues( outGrid->tree(), n, NN_FACE, EXPAND_TILES);
        outGrid->topologyIntersection( mask );
        pruneInactive( outGrid->tree(), mGrainSize>0, mGrainSize );
        this->template cook<VolumeGridT, VolumeSamplerT>(*outGrid, inGrid, dt);
        outGrid->topologyUnion( inGrid );

        for (int step = 1; step < mSubSteps; ++step) {
            typename VolumeGridT::Ptr tmpGrid = outGrid->deepCopy();
            dilateActiveValues( tmpGrid->tree(), n, NN_FACE, EXPAND_TILES);
            tmpGrid->topologyIntersection( mask );
            pruneInactive( tmpGrid->tree(), mGrainSize>0, mGrainSize );
            this->template cook<VolumeGridT, VolumeSamplerT>(*tmpGrid, *outGrid, dt);
            tmpGrid->topologyUnion( inGrid );
            outGrid.swap( tmpGrid );
        }
        return outGrid;
    }

private:
    // disallow copy construction and copy by assignment!
    VolumeAdvection(const VolumeAdvection&);// not implemented
    VolumeAdvection& operator=(const VolumeAdvection&);// not implemented

    void start(const char* str) const
    {
        if (mInterrupter) mInterrupter->start(str);
    }
    void stop() const
    {
        if (mInterrupter) mInterrupter->end();
    }
    bool interrupt() const
    {
        if (mInterrupter && util::wasInterrupted(mInterrupter)) {
            tbb::task::self().cancel_group_execution();
            return true;
        }
        return false;
    }

    template<typename VolumeGridT, typename VolumeSamplerT>
    void cook(VolumeGridT& outGrid, const VolumeGridT& inGrid, double dt)
    {
        switch (mIntegrator) {
        case Scheme::SEMI: {
            Advect<VolumeGridT, 1, VolumeSamplerT> adv(inGrid, *this);
            adv.cook(outGrid, dt);
            break;
        }
        case Scheme::MID: {
            Advect<VolumeGridT, 2, VolumeSamplerT> adv(inGrid, *this);
            adv.cook(outGrid, dt);
            break;
        }
        case Scheme::RK3: {
            Advect<VolumeGridT, 3, VolumeSamplerT> adv(inGrid, *this);
            adv.cook(outGrid, dt);
            break;
        }
        case Scheme::RK4: {
            Advect<VolumeGridT, 4, VolumeSamplerT> adv(inGrid, *this);
            adv.cook(outGrid, dt);
            break;
        }
        case Scheme::BFECC: {
            Advect<VolumeGridT, 1, VolumeSamplerT> adv(inGrid, *this);
            adv.cook(outGrid, dt);
            break;
        }
        case Scheme::MAC: {
            Advect<VolumeGridT, 1, VolumeSamplerT> adv(inGrid, *this);
            adv.cook(outGrid, dt);
            break;
        }
        default:
            OPENVDB_THROW(ValueError, "Spatial difference scheme not supported!");
        }
        pruneInactive(outGrid.tree(), mGrainSize>0, mGrainSize);
    }

    // Private class that implements the multi-threaded advection
    template<typename VolumeGridT, size_t OrderRK, typename SamplerT> struct Advect;

    // Private member data of VolumeAdvection
    const VelocityGridT&   mVelGrid;
    double                 mMaxVelocity;
    InterrupterType*       mInterrupter;
    Scheme::SemiLagrangian mIntegrator;
    Scheme::Limiter        mLimiter;
    size_t                 mGrainSize;
    int                    mSubSteps;
};//end of VolumeAdvection class

// Private class that implements the multi-threaded advection
template<typename VelocityGridT, bool StaggeredVelocity, typename InterrupterType>
template<typename VolumeGridT, size_t OrderRK, typename SamplerT>
struct VolumeAdvection<VelocityGridT, StaggeredVelocity, InterrupterType>::Advect
{
    using TreeT = typename VolumeGridT::TreeType;
    using AccT = typename VolumeGridT::ConstAccessor;
    using ValueT = typename TreeT::ValueType;
    using LeafManagerT = typename tree::LeafManager<TreeT>;
    using LeafNodeT = typename LeafManagerT::LeafNodeType;
    using LeafRangeT = typename LeafManagerT::LeafRange;
    using VelocityIntegratorT = VelocityIntegrator<VelocityGridT, StaggeredVelocity>;
    using RealT = typename VelocityIntegratorT::ElementType;
    using VoxelIterT = typename TreeT::LeafNodeType::ValueOnIter;

    Advect(const VolumeGridT& inGrid, const VolumeAdvection& parent)
        : mTask(0)
        , mInGrid(&inGrid)
        , mVelocityInt(parent.mVelGrid)
        , mParent(&parent)
    {
    }
    inline void cook(const LeafRangeT& range)
    {
        if (mParent->mGrainSize > 0) {
            tbb::parallel_for(range, *this);
        } else {
            (*this)(range);
        }
    }
    void operator()(const LeafRangeT& range) const
    {
        assert(mTask);
        mTask(const_cast<Advect*>(this), range);
    }
    void cook(VolumeGridT& outGrid, double time_step)
    {
        namespace ph = std::placeholders;

        mParent->start("Advecting volume");
        LeafManagerT manager(outGrid.tree(), mParent->spatialOrder()==2 ? 1 : 0);
        const LeafRangeT range = manager.leafRange(mParent->mGrainSize);
        const RealT dt = static_cast<RealT>(-time_step);//method of characteristics backtracks
        if (mParent->mIntegrator == Scheme::MAC) {
            mTask = std::bind(&Advect::rk, ph::_1, ph::_2, dt, 0, mInGrid);//out[0]=forward
            this->cook(range);
            mTask = std::bind(&Advect::rk, ph::_1, ph::_2,-dt, 1, &outGrid);//out[1]=backward
            this->cook(range);
            mTask = std::bind(&Advect::mac, ph::_1, ph::_2);//out[0] = out[0] + (in[0] - out[1])/2
            this->cook(range);
        } else if (mParent->mIntegrator == Scheme::BFECC) {
            mTask = std::bind(&Advect::rk, ph::_1, ph::_2, dt, 0, mInGrid);//out[0]=forward
            this->cook(range);
            mTask = std::bind(&Advect::rk, ph::_1, ph::_2,-dt, 1, &outGrid);//out[1]=backward
            this->cook(range);
            mTask = std::bind(&Advect::bfecc, ph::_1, ph::_2);//out[0] = (3*in[0] - out[1])/2
            this->cook(range);
            mTask = std::bind(&Advect::rk, ph::_1, ph::_2, dt, 1, &outGrid);//out[1]=forward
            this->cook(range);
            manager.swapLeafBuffer(1);// out[0] = out[1]
        } else {// SEMI, MID, RK3 and RK4
            mTask = std::bind(&Advect::rk, ph::_1, ph::_2,  dt, 0, mInGrid);//forward
            this->cook(range);
        }

        if (mParent->spatialOrder()==2) manager.removeAuxBuffers();

        mTask = std::bind(&Advect::limiter, ph::_1, ph::_2, dt);// out[0] = limiter( out[0] )
        this->cook(range);

        mParent->stop();
    }
    // Last step of the MacCormack scheme: out[0] = out[0] + (in[0] - out[1])/2
    void mac(const LeafRangeT& range) const
    {
        if (mParent->interrupt()) return;
        assert( mParent->mIntegrator == Scheme::MAC );
        AccT acc = mInGrid->getAccessor();
        for (typename LeafRangeT::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
            ValueT* out0 = leafIter.buffer( 0 ).data();// forward
            const ValueT* out1 = leafIter.buffer( 1 ).data();// backward
            const LeafNodeT* leaf = acc.probeConstLeaf( leafIter->origin() );
            if (leaf != nullptr) {
                const ValueT* in0 = leaf->buffer().data();
                for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                    const Index i = voxelIter.pos();
                    out0[i] += RealT(0.5) * ( in0[i] - out1[i] );
                }
            } else {
                for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                    const Index i = voxelIter.pos();
                    out0[i] += RealT(0.5) * ( acc.getValue(voxelIter.getCoord()) - out1[i] );
                }//loop over active voxels
            }
        }//loop over leaf nodes
    }
    // Intermediate step in the BFECC scheme: out[0] = (3*in[0] - out[1])/2
    void bfecc(const LeafRangeT& range) const
    {
        if (mParent->interrupt()) return;
        assert( mParent->mIntegrator == Scheme::BFECC );
        AccT acc = mInGrid->getAccessor();
        for (typename LeafRangeT::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
            ValueT* out0 = leafIter.buffer( 0 ).data();// forward
            const ValueT* out1 = leafIter.buffer( 1 ).data();// backward
            const LeafNodeT* leaf = acc.probeConstLeaf(leafIter->origin());
            if (leaf != nullptr) {
                const ValueT* in0 = leaf->buffer().data();
                for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                    const Index i = voxelIter.pos();
                    out0[i] = RealT(0.5)*( RealT(3)*in0[i] - out1[i] );
                }//loop over active voxels
            } else {
                for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                    const Index i = voxelIter.pos();
                    out0[i] = RealT(0.5)*( RealT(3)*acc.getValue(voxelIter.getCoord()) - out1[i] );
                }//loop over active voxels
            }
        }//loop over leaf nodes
    }
    // Semi-Lagrangian integration with Runge-Kutta of various orders (1->4)
    void rk(const LeafRangeT& range, RealT dt, size_t n, const VolumeGridT* grid) const
    {
        if (mParent->interrupt()) return;
        const math::Transform& xform = mInGrid->transform();
        AccT acc = grid->getAccessor();
        for (typename LeafRangeT::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
            ValueT* phi = leafIter.buffer( n ).data();
            for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                ValueT& value = phi[voxelIter.pos()];
                Vec3d wPos = xform.indexToWorld(voxelIter.getCoord());
                mVelocityInt.template rungeKutta<OrderRK, Vec3d>(dt, wPos);
                value = SamplerT::sample(acc, xform.worldToIndex(wPos));
            }//loop over active voxels
        }//loop over leaf nodes
    }
    void limiter(const LeafRangeT& range, RealT dt) const
    {
        if (mParent->interrupt()) return;
        const bool doLimiter = mParent->isLimiterOn();
        const bool doClamp = mParent->mLimiter == Scheme::CLAMP;
        ValueT data[2][2][2], vMin, vMax;
        const math::Transform& xform = mInGrid->transform();
        AccT acc = mInGrid->getAccessor();
        const ValueT backg = mInGrid->background();
        for (typename LeafRangeT::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
            ValueT* phi = leafIter.buffer( 0 ).data();
            for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                ValueT& value = phi[voxelIter.pos()];

                if ( doLimiter ) {
                    assert(OrderRK == 1);
                    Vec3d wPos = xform.indexToWorld(voxelIter.getCoord());
                    mVelocityInt.template rungeKutta<1, Vec3d>(dt, wPos);// Explicit Euler
                    Vec3d iPos = xform.worldToIndex(wPos);
                    Coord ijk  = Coord::floor( iPos );
                    BoxSampler::getValues(data, acc, ijk);
                    BoxSampler::extrema(data, vMin, vMax);
                    if ( doClamp ) {
                        value = math::Clamp( value, vMin, vMax);
                    } else if (value < vMin || value > vMax ) {
                        iPos -= Vec3R(ijk[0], ijk[1], ijk[2]);//unit coordinates
                        value = BoxSampler::trilinearInterpolation( data, iPos );
                    }
                }

                if (math::isApproxEqual(value, backg, math::Delta<ValueT>::value())) {
                    value = backg;
                    leafIter->setValueOff( voxelIter.pos() );
                }
            }//loop over active voxels
        }//loop over leaf nodes
    }
    // Public member data of the private Advect class

    typename std::function<void (Advect*, const LeafRangeT&)> mTask;
    const VolumeGridT*        mInGrid;
    const VelocityIntegratorT mVelocityInt;// lightweight!
    const VolumeAdvection*    mParent;
};// end of private member class Advect

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_VOLUME_ADVECT_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
