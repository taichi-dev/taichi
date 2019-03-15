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
/// @file tools/LevelSetMorph.h
///
/// @brief Shape morphology of level sets. Morphing from a source
/// narrow-band level sets to a target narrow-band level set.

#ifndef OPENVDB_TOOLS_LEVEL_SET_MORPH_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_MORPH_HAS_BEEN_INCLUDED

#include "LevelSetTracker.h"
#include "Interpolation.h" // for BoxSampler, etc.
#include <openvdb/math/FiniteDifference.h>
#include <functional>
#include <limits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Shape morphology of level sets. Morphing from a source
/// narrow-band level sets to a target narrow-band level set.
///
/// @details
/// The @c InterruptType template argument below refers to any class
/// with the following interface:
/// @code
/// class Interrupter {
///   ...
/// public:
///   void start(const char* name = nullptr) // called when computations begin
///   void end()                             // called when computations end
///   bool wasInterrupted(int percent=-1)    // return true to break computation
/// };
/// @endcode
///
/// @note If no template argument is provided for this InterruptType,
/// the util::NullInterrupter is used, which implies that all interrupter
/// calls are no-ops (i.e., they incur no computational overhead).
template<typename GridT,
         typename InterruptT = util::NullInterrupter>
class LevelSetMorphing
{
public:
    using GridType = GridT;
    using TreeType = typename GridT::TreeType;
    using TrackerT = LevelSetTracker<GridT, InterruptT>;
    using LeafRange = typename TrackerT::LeafRange;
    using LeafType = typename TrackerT::LeafType;
    using BufferType = typename TrackerT::BufferType;
    using ValueType = typename TrackerT::ValueType;

    /// Main constructor
    LevelSetMorphing(GridT& sourceGrid, const GridT& targetGrid, InterruptT* interrupt = nullptr)
        : mTracker(sourceGrid, interrupt)
        , mTarget(&targetGrid)
        , mMask(nullptr)
        , mSpatialScheme(math::HJWENO5_BIAS)
        , mTemporalScheme(math::TVD_RK2)
        , mMinMask(0)
        , mDeltaMask(1)
        , mInvertMask(false)
    {
    }

    virtual ~LevelSetMorphing() {}

    /// Redefine the target level set
    void setTarget(const GridT& targetGrid) { mTarget = &targetGrid; }

    /// Define the alpha mask
    void setAlphaMask(const GridT& maskGrid) { mMask = &maskGrid; }

    /// Return the spatial finite-difference scheme
    math::BiasedGradientScheme getSpatialScheme() const { return mSpatialScheme; }
    /// Set the spatial finite-difference scheme
    void setSpatialScheme(math::BiasedGradientScheme scheme) { mSpatialScheme = scheme; }

    /// Return the temporal integration scheme
    math::TemporalIntegrationScheme getTemporalScheme() const { return mTemporalScheme; }
    /// Set the temporal integration scheme
    void setTemporalScheme(math::TemporalIntegrationScheme scheme) { mTemporalScheme = scheme; }

    /// Return the spatial finite-difference scheme
    math::BiasedGradientScheme getTrackerSpatialScheme() const
    {
        return mTracker.getSpatialScheme();
    }
    /// Set the spatial finite-difference scheme
    void setTrackerSpatialScheme(math::BiasedGradientScheme scheme)
    {
        mTracker.setSpatialScheme(scheme);
    }
    /// Return the temporal integration scheme
    math::TemporalIntegrationScheme getTrackerTemporalScheme() const
    {
        return mTracker.getTemporalScheme();
    }
    /// Set the temporal integration scheme
    void setTrackerTemporalScheme(math::TemporalIntegrationScheme scheme)
    {
        mTracker.setTemporalScheme(scheme);
    }
    /// Return the number of normalizations performed per track or normalize call.
    int  getNormCount() const { return mTracker.getNormCount(); }
    /// Set the number of normalizations performed per track or normalize call.
    void setNormCount(int n) { mTracker.setNormCount(n); }

    /// Return the grain size used for multithreading
    int  getGrainSize() const { return mTracker.getGrainSize(); }
    /// @brief Set the grain size used for multithreading.
    /// @note A grain size of 0 or less disables multithreading!
    void setGrainSize(int grainsize) { mTracker.setGrainSize(grainsize); }

    /// @brief Return the minimum value of the mask to be used for the
    /// derivation of a smooth alpha value.
    ValueType minMask() const { return mMinMask; }

    /// @brief Return the maximum value of the mask to be used for the
    /// derivation of a smooth alpha value.
    ValueType maxMask() const { return mDeltaMask + mMinMask; }

    /// @brief Define the range for the (optional) scalar mask.
    /// @param min Minimum value of the range.
    /// @param max Maximum value of the range.
    /// @details Mask values outside the range maps to alpha values of
    /// respectfully zero and one, and values inside the range maps
    /// smoothly to 0->1 (unless of course the mask is inverted).
    /// @throw ValueError if @a min is not smaller than @a max.
    void setMaskRange(ValueType min, ValueType max)
    {
        if (!(min < max)) OPENVDB_THROW(ValueError, "Invalid mask range (expects min < max)");
        mMinMask   = min;
        mDeltaMask = max-min;
    }

    /// @brief Return true if the mask is inverted, i.e. min->max in the
    /// original mask maps to 1->0 in the inverted alpha mask.
    bool isMaskInverted() const { return mInvertMask; }
    /// @brief Invert the optional mask, i.e. min->max in the original
    /// mask maps to 1->0 in the inverted alpha mask.
    void invertMask(bool invert=true) { mInvertMask = invert; }

    /// @brief Advect the level set from its current time, @a time0, to its
    /// final time, @a time1. If @a time0 > @a time1, perform backward advection.
    ///
    /// @return the number of CFL iterations used to advect from @a time0 to @a time1
    size_t advect(ValueType time0, ValueType time1);

private:

    // disallow copy construction and copy by assignment!
    LevelSetMorphing(const LevelSetMorphing&);// not implemented
    LevelSetMorphing& operator=(const LevelSetMorphing&);// not implemented

    template<math::BiasedGradientScheme SpatialScheme>
    size_t advect1(ValueType time0, ValueType time1);

    template<math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme>
    size_t advect2(ValueType time0, ValueType time1);

    template<math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme,
             typename MapType>
    size_t advect3(ValueType time0, ValueType time1);

    TrackerT                        mTracker;
    const GridT                    *mTarget, *mMask;
    math::BiasedGradientScheme      mSpatialScheme;
    math::TemporalIntegrationScheme mTemporalScheme;
    ValueType                       mMinMask, mDeltaMask;
    bool                            mInvertMask;

    // This templated private class implements all the level set magic.
    template<typename MapT, math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme>
    struct Morph
    {
        /// Main constructor
        Morph(LevelSetMorphing<GridT, InterruptT>& parent);
        /// Shallow copy constructor called by tbb::parallel_for() threads
        Morph(const Morph& other);
        /// Shallow copy constructor called by tbb::parallel_reduce() threads
        Morph(Morph& other, tbb::split);
        /// destructor
        virtual ~Morph() {}
        /// Advect the level set from its current time, time0, to its final time, time1.
        /// @return number of CFL iterations
        size_t advect(ValueType time0, ValueType time1);
        /// Used internally by tbb::parallel_for()
        void operator()(const LeafRange& r) const
        {
            if (mTask) mTask(const_cast<Morph*>(this), r);
            else OPENVDB_THROW(ValueError, "task is undefined - don\'t call this method directly");
        }
        /// Used internally by tbb::parallel_reduce()
        void operator()(const LeafRange& r)
        {
            if (mTask) mTask(this, r);
            else OPENVDB_THROW(ValueError, "task is undefined - don\'t call this method directly");
        }
        /// This is only called by tbb::parallel_reduce() threads
        void join(const Morph& other) { mMaxAbsS = math::Max(mMaxAbsS, other.mMaxAbsS); }

        /// Enum to define the type of multithreading
        enum ThreadingMode { PARALLEL_FOR, PARALLEL_REDUCE }; // for internal use
        // method calling tbb
        void cook(ThreadingMode mode, size_t swapBuffer = 0);

        /// Sample field and return the CFT time step
        typename GridT::ValueType sampleSpeed(ValueType time0, ValueType time1, Index speedBuffer);
        void sampleXformedSpeed(const LeafRange& r, Index speedBuffer);
        void sampleAlignedSpeed(const LeafRange& r, Index speedBuffer);

        // Convex combination of Phi and a forward Euler advection steps:
        // Phi(result) = alpha * Phi(phi) + (1-alpha) * (Phi(0) - dt * Speed(speed)*|Grad[Phi(0)]|);
        template <int Nominator, int Denominator>
        void euler(const LeafRange&, ValueType, Index, Index, Index);
        inline void euler01(const LeafRange& r, ValueType t, Index s) {this->euler<0,1>(r,t,0,1,s);}
        inline void euler12(const LeafRange& r, ValueType t) {this->euler<1,2>(r, t, 1, 1, 2);}
        inline void euler34(const LeafRange& r, ValueType t) {this->euler<3,4>(r, t, 1, 2, 3);}
        inline void euler13(const LeafRange& r, ValueType t) {this->euler<1,3>(r, t, 1, 2, 3);}

        using FuncType = typename std::function<void (Morph*, const LeafRange&)>;
        LevelSetMorphing* mParent;
        ValueType         mMinAbsS, mMaxAbsS;
        const MapT*       mMap;
        FuncType          mTask;
    }; // end of private Morph struct

};//end of LevelSetMorphing

template<typename GridT, typename InterruptT>
inline size_t
LevelSetMorphing<GridT, InterruptT>::advect(ValueType time0, ValueType time1)
{
    switch (mSpatialScheme) {
    case math::FIRST_BIAS:
        return this->advect1<math::FIRST_BIAS  >(time0, time1);
    //case math::SECOND_BIAS:
    //return this->advect1<math::SECOND_BIAS >(time0, time1);
    //case math::THIRD_BIAS:
    //return this->advect1<math::THIRD_BIAS  >(time0, time1);
    //case math::WENO5_BIAS:
    //return this->advect1<math::WENO5_BIAS  >(time0, time1);
    case math::HJWENO5_BIAS:
        return this->advect1<math::HJWENO5_BIAS>(time0, time1);
    case math::SECOND_BIAS:
    case math::THIRD_BIAS:
    case math::WENO5_BIAS:
    case math::UNKNOWN_BIAS:
    default:
        OPENVDB_THROW(ValueError, "Spatial difference scheme not supported!");
    }
    return 0;
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme>
inline size_t
LevelSetMorphing<GridT, InterruptT>::advect1(ValueType time0, ValueType time1)
{
    switch (mTemporalScheme) {
    case math::TVD_RK1:
        return this->advect2<SpatialScheme, math::TVD_RK1>(time0, time1);
    case math::TVD_RK2:
        return this->advect2<SpatialScheme, math::TVD_RK2>(time0, time1);
    case math::TVD_RK3:
        return this->advect2<SpatialScheme, math::TVD_RK3>(time0, time1);
    case math::UNKNOWN_TIS:
    default:
        OPENVDB_THROW(ValueError, "Temporal integration scheme not supported!");
    }
    return 0;
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
inline size_t
LevelSetMorphing<GridT, InterruptT>::advect2(ValueType time0, ValueType time1)
{
    const math::Transform& trans = mTracker.grid().transform();
    if (trans.mapType() == math::UniformScaleMap::mapType()) {
        return this->advect3<SpatialScheme, TemporalScheme, math::UniformScaleMap>(time0, time1);
    } else if (trans.mapType() == math::UniformScaleTranslateMap::mapType()) {
        return this->advect3<SpatialScheme, TemporalScheme, math::UniformScaleTranslateMap>(
            time0, time1);
    } else if (trans.mapType() == math::UnitaryMap::mapType()) {
        return this->advect3<SpatialScheme, TemporalScheme, math::UnitaryMap    >(time0, time1);
    } else if (trans.mapType() == math::TranslationMap::mapType()) {
        return this->advect3<SpatialScheme, TemporalScheme, math::TranslationMap>(time0, time1);
    } else {
        OPENVDB_THROW(ValueError, "MapType not supported!");
    }
    return 0;
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MapT>
inline size_t
LevelSetMorphing<GridT, InterruptT>::advect3(ValueType time0, ValueType time1)
{
    Morph<MapT, SpatialScheme, TemporalScheme> tmp(*this);
    return tmp.advect(time0, time1);
}


///////////////////////////////////////////////////////////////////////

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline
LevelSetMorphing<GridT, InterruptT>::
Morph<MapT, SpatialScheme, TemporalScheme>::
Morph(LevelSetMorphing<GridT, InterruptT>& parent)
    : mParent(&parent)
    , mMinAbsS(ValueType(1e-6))
    , mMap(parent.mTracker.grid().transform().template constMap<MapT>().get())
    , mTask(0)
{
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline
LevelSetMorphing<GridT, InterruptT>::
Morph<MapT, SpatialScheme, TemporalScheme>::
Morph(const Morph& other)
    : mParent(other.mParent)
    , mMinAbsS(other.mMinAbsS)
    , mMaxAbsS(other.mMaxAbsS)
    , mMap(other.mMap)
    , mTask(other.mTask)
{
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline
LevelSetMorphing<GridT, InterruptT>::
Morph<MapT, SpatialScheme, TemporalScheme>::
Morph(Morph& other, tbb::split)
    : mParent(other.mParent)
    , mMinAbsS(other.mMinAbsS)
    , mMaxAbsS(other.mMaxAbsS)
    , mMap(other.mMap)
    , mTask(other.mTask)
{
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline size_t
LevelSetMorphing<GridT, InterruptT>::
Morph<MapT, SpatialScheme, TemporalScheme>::
advect(ValueType time0, ValueType time1)
{
    namespace ph = std::placeholders;

    // Make sure we have enough temporal auxiliary buffers for the time
    // integration AS WELL AS an extra buffer with the speed function!
    static const Index auxBuffers = 1 + (TemporalScheme == math::TVD_RK3 ? 2 : 1);
    size_t countCFL = 0;
    while (time0 < time1 && mParent->mTracker.checkInterrupter()) {
        mParent->mTracker.leafs().rebuildAuxBuffers(auxBuffers);

        const ValueType dt = this->sampleSpeed(time0, time1, auxBuffers);
        if ( math::isZero(dt) ) break;//V is essentially zero so terminate

        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN //switch is resolved at compile-time
        switch(TemporalScheme) {
        case math::TVD_RK1:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * Speed(2) * |Grad[Phi(0)]|
            mTask = std::bind(&Morph::euler01, ph::_1, ph::_2, dt, /*speed*/2);

            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(PARALLEL_FOR, 1);
            break;
        case math::TVD_RK2:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * Speed(2) * |Grad[Phi(0)]|
            mTask = std::bind(&Morph::euler01, ph::_1, ph::_2, dt, /*speed*/2);

            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(PARALLEL_FOR, 1);

            // Convex combine explict Euler step: t2 = t0 + dt
            // Phi_t2(1) = 1/2 * Phi_t0(1) + 1/2 * (Phi_t1(0) - dt * Speed(2) * |Grad[Phi(0)]|)
            mTask = std::bind(&Morph::euler12, ph::_1, ph::_2, dt);

            // Cook and swap buffer 0 and 1 such that Phi_t2(0) and Phi_t1(1)
            this->cook(PARALLEL_FOR, 1);
            break;
        case math::TVD_RK3:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * Speed(3) * |Grad[Phi(0)]|
            mTask = std::bind(&Morph::euler01, ph::_1, ph::_2, dt, /*speed*/3);

            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(PARALLEL_FOR, 1);

            // Convex combine explict Euler step: t2 = t0 + dt/2
            // Phi_t2(2) = 3/4 * Phi_t0(1) + 1/4 * (Phi_t1(0) - dt * Speed(3) * |Grad[Phi(0)]|)
            mTask = std::bind(&Morph::euler34, ph::_1, ph::_2, dt);

            // Cook and swap buffer 0 and 2 such that Phi_t2(0) and Phi_t1(2)
            this->cook(PARALLEL_FOR, 2);

            // Convex combine explict Euler step: t3 = t0 + dt
            // Phi_t3(2) = 1/3 * Phi_t0(1) + 2/3 * (Phi_t2(0) - dt * Speed(3) * |Grad[Phi(0)]|)
            mTask = std::bind(&Morph::euler13, ph::_1, ph::_2, dt);

            // Cook and swap buffer 0 and 2 such that Phi_t3(0) and Phi_t2(2)
            this->cook(PARALLEL_FOR, 2);
            break;
        case math::UNKNOWN_TIS:
        default:
            OPENVDB_THROW(ValueError, "Temporal integration scheme not supported!");
        }//end of compile-time resolved switch
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END

        time0 += dt;
        ++countCFL;
        mParent->mTracker.leafs().removeAuxBuffers();

        // Track the narrow band
        mParent->mTracker.track();
    }//end wile-loop over time

    return countCFL;//number of CLF propagation steps
}

template<typename GridT, typename InterruptT>
template<typename MapT, math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
inline typename GridT::ValueType
LevelSetMorphing<GridT, InterruptT>::
Morph<MapT, SpatialScheme, TemporalScheme>::
sampleSpeed(ValueType time0, ValueType time1, Index speedBuffer)
{
    namespace ph = std::placeholders;

    mMaxAbsS = mMinAbsS;
    const size_t leafCount = mParent->mTracker.leafs().leafCount();
    if (leafCount==0 || time0 >= time1) return ValueType(0);

    const math::Transform& xform  = mParent->mTracker.grid().transform();
    if (mParent->mTarget->transform() == xform &&
        (mParent->mMask == nullptr || mParent->mMask->transform() == xform)) {
        mTask = std::bind(&Morph::sampleAlignedSpeed, ph::_1, ph::_2, speedBuffer);
    } else {
        mTask = std::bind(&Morph::sampleXformedSpeed, ph::_1, ph::_2, speedBuffer);
    }
    this->cook(PARALLEL_REDUCE);
    if (math::isApproxEqual(mMinAbsS, mMaxAbsS)) return ValueType(0);//speed is essentially zero
    static const ValueType CFL = (TemporalScheme == math::TVD_RK1 ? ValueType(0.3) :
                                  TemporalScheme == math::TVD_RK2 ? ValueType(0.9) :
                                  ValueType(1.0))/math::Sqrt(ValueType(3.0));
    const ValueType dt = math::Abs(time1 - time0), dx = mParent->mTracker.voxelSize();
    return math::Min(dt, ValueType(CFL*dx/mMaxAbsS));
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetMorphing<GridT, InterruptT>::
Morph<MapT, SpatialScheme, TemporalScheme>::
sampleXformedSpeed(const LeafRange& range, Index speedBuffer)
{
    using VoxelIterT = typename LeafType::ValueOnCIter;
    using SamplerT = tools::GridSampler<typename GridT::ConstAccessor, tools::BoxSampler>;

    const MapT& map = *mMap;
    mParent->mTracker.checkInterrupter();

    typename GridT::ConstAccessor targetAcc = mParent->mTarget->getAccessor();
    SamplerT target(targetAcc, mParent->mTarget->transform());
    if (mParent->mMask == nullptr) {
        for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
            ValueType* speed = leafIter.buffer(speedBuffer).data();
            bool isZero = true;
            for (VoxelIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
                ValueType& s = speed[voxelIter.pos()];
                s -= target.wsSample(map.applyMap(voxelIter.getCoord().asVec3d()));
                if (!math::isApproxZero(s)) isZero = false;
                mMaxAbsS = math::Max(mMaxAbsS, math::Abs(s));
            }
            if (isZero) speed[0] = std::numeric_limits<ValueType>::max();//tag first voxel
        }
    } else {
        const ValueType min = mParent->mMinMask, invNorm = 1.0f/(mParent->mDeltaMask);
        const bool invMask = mParent->isMaskInverted();
        typename GridT::ConstAccessor maskAcc = mParent->mMask->getAccessor();
        SamplerT mask(maskAcc,  mParent->mMask->transform());
        for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
            ValueType* speed = leafIter.buffer(speedBuffer).data();
            bool isZero = true;
            for (VoxelIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
                const Vec3R xyz = map.applyMap(voxelIter.getCoord().asVec3d());//world space
                const ValueType a = math::SmoothUnitStep((mask.wsSample(xyz)-min)*invNorm);
                ValueType& s = speed[voxelIter.pos()];
                s -= target.wsSample(xyz);
                s *= invMask ? 1 - a : a;
                if (!math::isApproxZero(s)) isZero = false;
                mMaxAbsS = math::Max(mMaxAbsS, math::Abs(s));
            }
            if (isZero) speed[0] = std::numeric_limits<ValueType>::max();//tag first voxel
        }
    }
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetMorphing<GridT, InterruptT>::
Morph<MapT, SpatialScheme, TemporalScheme>::
sampleAlignedSpeed(const LeafRange& range, Index speedBuffer)
{
    using VoxelIterT = typename LeafType::ValueOnCIter;

    mParent->mTracker.checkInterrupter();

    typename GridT::ConstAccessor target = mParent->mTarget->getAccessor();

    if (mParent->mMask == nullptr) {
        for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
            ValueType* speed = leafIter.buffer(speedBuffer).data();
            bool isZero = true;
            for (VoxelIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
                ValueType& s = speed[voxelIter.pos()];
                s -= target.getValue(voxelIter.getCoord());
                if (!math::isApproxZero(s)) isZero = false;
                mMaxAbsS = math::Max(mMaxAbsS, math::Abs(s));
            }
            if (isZero) speed[0] = std::numeric_limits<ValueType>::max();//tag first voxel
        }
    } else {
        const ValueType min = mParent->mMinMask, invNorm = 1.0f/(mParent->mDeltaMask);
        const bool invMask = mParent->isMaskInverted();
        typename GridT::ConstAccessor mask = mParent->mMask->getAccessor();
        for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
            ValueType* speed = leafIter.buffer(speedBuffer).data();
            bool isZero = true;
            for (VoxelIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
                const Coord ijk = voxelIter.getCoord();//index space
                const ValueType a = math::SmoothUnitStep((mask.getValue(ijk)-min)*invNorm);
                ValueType& s = speed[voxelIter.pos()];
                s -= target.getValue(ijk);
                s *= invMask ? 1 - a : a;
                if (!math::isApproxZero(s)) isZero = false;
                mMaxAbsS = math::Max(mMaxAbsS, math::Abs(s));
            }
            if (isZero) speed[0] = std::numeric_limits<ValueType>::max();//tag first voxel
        }
    }
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetMorphing<GridT, InterruptT>::
Morph<MapT, SpatialScheme, TemporalScheme>::
cook(ThreadingMode mode, size_t swapBuffer)
{
    mParent->mTracker.startInterrupter("Morphing level set");

    const int grainSize   = mParent->mTracker.getGrainSize();
    const LeafRange range = mParent->mTracker.leafs().leafRange(grainSize);

    if (mParent->mTracker.getGrainSize()==0) {
        (*this)(range);
    } else if (mode == PARALLEL_FOR) {
        tbb::parallel_for(range, *this);
    } else if (mode == PARALLEL_REDUCE) {
        tbb::parallel_reduce(range, *this);
    } else {
        OPENVDB_THROW(ValueError, "expected threading mode " << int(PARALLEL_FOR)
            << " or " << int(PARALLEL_REDUCE) << ", got " << int(mode));
    }

    mParent->mTracker.leafs().swapLeafBuffer(swapBuffer, grainSize == 0);

    mParent->mTracker.endInterrupter();
}

template<typename GridT, typename InterruptT>
template<typename MapT, math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
template <int Nominator, int Denominator>
inline void
LevelSetMorphing<GridT,InterruptT>::
Morph<MapT, SpatialScheme, TemporalScheme>::
euler(const LeafRange& range, ValueType dt,
      Index phiBuffer, Index resultBuffer, Index speedBuffer)
{
    using SchemeT = math::BIAS_SCHEME<SpatialScheme>;
    using StencilT = typename SchemeT::template ISStencil<GridType>::StencilType;
    using VoxelIterT = typename LeafType::ValueOnCIter;
    using NumGrad = math::GradientNormSqrd<MapT, SpatialScheme>;

    static const ValueType Alpha = ValueType(Nominator)/ValueType(Denominator);
    static const ValueType Beta  = ValueType(1) - Alpha;

    mParent->mTracker.checkInterrupter();
    const MapT& map = *mMap;
    StencilT stencil(mParent->mTracker.grid());

    for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
        const ValueType* speed = leafIter.buffer(speedBuffer).data();
        if (math::isExactlyEqual(speed[0], std::numeric_limits<ValueType>::max())) continue;
        const ValueType* phi = leafIter.buffer(phiBuffer).data();
        ValueType* result = leafIter.buffer(resultBuffer).data();
        for (VoxelIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
            const Index n = voxelIter.pos();
            if (math::isApproxZero(speed[n])) continue;
            stencil.moveTo(voxelIter);
            const ValueType v = stencil.getValue() - dt * speed[n] * NumGrad::result(map, stencil);
            result[n] = Nominator ? Alpha * phi[n] + Beta * v : v;
        }//loop over active voxels in the leaf of the mask
    }//loop over leafs of the level set
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_MORPH_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
