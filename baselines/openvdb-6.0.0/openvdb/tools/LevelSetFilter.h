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
/// @author Ken Museth
///
/// @file tools/LevelSetFilter.h
///
/// @brief Performs various types of level set deformations with
/// interface tracking. These unrestricted deformations include
/// surface smoothing (e.g., Laplacian flow), filtering (e.g., mean
/// value) and morphological operations (e.g., morphological opening).
/// All these operations can optionally be masked with another grid that
/// acts as an alpha-mask.

#ifndef OPENVDB_TOOLS_LEVELSETFILTER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETFILTER_HAS_BEEN_INCLUDED

#include "LevelSetTracker.h"
#include "Interpolation.h"
#include <algorithm> // for std::max()
#include <functional>
#include <type_traits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Filtering (e.g. diffusion) of narrow-band level sets. An
/// optional scalar field can be used to produce a (smooth) alpha mask
/// for the filtering.
///
/// @note This class performs proper interface tracking which allows
/// for unrestricted surface deformations
template<typename GridT,
         typename MaskT = typename GridT::template ValueConverter<float>::Type,
         typename InterruptT = util::NullInterrupter>
class LevelSetFilter : public LevelSetTracker<GridT, InterruptT>
{
public:
    using BaseType = LevelSetTracker<GridT, InterruptT>;
    using GridType = GridT;
    using MaskType = MaskT;
    using TreeType = typename GridType::TreeType;
    using ValueType = typename TreeType::ValueType;
    using AlphaType = typename MaskType::ValueType;
    static_assert(std::is_floating_point<AlphaType>::value,
        "LevelSetFilter requires a mask grid with floating-point values");

    /// @brief Main constructor from a grid
    /// @param grid The level set to be filtered.
    /// @param interrupt Optional interrupter.
    LevelSetFilter(GridType& grid, InterruptT* interrupt = nullptr)
        : BaseType(grid, interrupt)
        , mMinMask(0)
        , mMaxMask(1)
        , mInvertMask(false)
    {
    }
    /// @brief Default destructor
    ~LevelSetFilter() override {}

    /// @brief Return the minimum value of the mask to be used for the
    /// derivation of a smooth alpha value.
    AlphaType minMask() const { return mMinMask; }
    /// @brief Return the maximum value of the mask to be used for the
    /// derivation of a smooth alpha value.
    AlphaType maxMask() const { return mMaxMask; }
    /// @brief Define the range for the (optional) scalar mask.
    /// @param min Minimum value of the range.
    /// @param max Maximum value of the range.
    /// @details Mask values outside the range maps to alpha values of
    /// respectfully zero and one, and values inside the range maps
    /// smoothly to 0->1 (unless of course the mask is inverted).
    /// @throw ValueError if @a min is not smaller than @a max.
    void setMaskRange(AlphaType min, AlphaType max)
    {
        if (!(min < max)) OPENVDB_THROW(ValueError, "Invalid mask range (expects min < max)");
        mMinMask = min;
        mMaxMask = max;
    }

    /// @brief Return true if the mask is inverted, i.e. min->max in the
    /// original mask maps to 1->0 in the inverted alpha mask.
    bool isMaskInverted() const { return mInvertMask; }
    /// @brief Invert the optional mask, i.e. min->max in the original
    /// mask maps to 1->0 in the inverted alpha mask.
    void invertMask(bool invert=true) { mInvertMask = invert; }

    /// @brief One iteration of mean-curvature flow of the level set.
    /// @param mask Optional alpha mask.
    void meanCurvature(const MaskType* mask = nullptr)
    {
        Filter f(this, mask); f.meanCurvature();
    }

    /// @brief One iteration of Laplacian flow of the level set.
    /// @param mask Optional alpha mask.
    void laplacian(const MaskType* mask = nullptr)
    {
        Filter f(this, mask); f.laplacian();
    }

    /// @brief One iteration of a fast separable Gaussian filter.
    /// @param width Width of the Gaussian kernel in voxel units.
    /// @param mask Optional alpha mask.
    ///
    /// @note This is approximated as 4 iterations of a separable mean filter
    /// which typically leads an approximation that's better than 95%!
    void gaussian(int width = 1, const MaskType* mask = nullptr)
    {
        Filter f(this, mask); f.gaussian(width);
    }

    /// @brief Offset the level set by the specified (world) distance.
    /// @param offset Value of the offset.
    /// @param mask Optional alpha mask.
    void offset(ValueType offset, const MaskType* mask = nullptr)
    {
        Filter f(this, mask); f.offset(offset);
    }

    /// @brief One iteration of median-value flow of the level set.
    /// @param width Width of the median-value kernel in voxel units.
    /// @param mask Optional alpha mask.
    ///
    /// @warning This filter is not separable and is hence relatively
    /// slow!
    void median(int width = 1, const MaskType* mask = nullptr)
    {
        Filter f(this, mask); f.median(width);
    }

    /// @brief One iteration of mean-value flow of the level set.
    /// @param width Width of the mean-value kernel in voxel units.
    /// @param mask Optional alpha mask.
    ///
    /// @note This filter is separable so it's fast!
    void mean(int width = 1, const MaskType* mask = nullptr)
    {
        Filter f(this, mask); f.mean(width);
    }

private:
    // disallow copy construction and copy by assignment!
    LevelSetFilter(const LevelSetFilter&);// not implemented
    LevelSetFilter& operator=(const LevelSetFilter&);// not implemented

    // Private struct that implements all the filtering.
    struct Filter
    {
        using LeafT = typename TreeType::LeafNodeType;
        using VoxelIterT = typename LeafT::ValueOnIter;
        using VoxelCIterT = typename LeafT::ValueOnCIter;
        using BufferT = typename tree::LeafManager<TreeType>::BufferType;
        using LeafRange = typename tree::LeafManager<TreeType>::LeafRange;
        using LeafIterT = typename LeafRange::Iterator;
        using AlphaMaskT = tools::AlphaMask<GridT, MaskT>;

        Filter(LevelSetFilter* parent, const MaskType* mask) : mParent(parent), mMask(mask) {}
        Filter(const Filter&) = default;
        virtual ~Filter() {}

        void box(int width);
        void median(int width);
        void mean(int width);
        void gaussian(int width);
        void laplacian();
        void meanCurvature();
        void offset(ValueType value);
        void operator()(const LeafRange& r) const
        {
            if (mTask) mTask(const_cast<Filter*>(this), r);
            else OPENVDB_THROW(ValueError, "task is undefined - don\'t call this method directly");
        }
        void cook(bool swap)
        {
            const int n = mParent->getGrainSize();
            if (n>0) {
                tbb::parallel_for(mParent->leafs().leafRange(n), *this);
            } else {
                (*this)(mParent->leafs().leafRange());
            }
            if (swap) mParent->leafs().swapLeafBuffer(1, n==0);
        }

        template <size_t Axis>
        struct Avg {
            Avg(const GridT& grid, Int32 w) :
                acc(grid.tree()), width(w), frac(1/ValueType(2*w+1)) {}
            inline ValueType operator()(Coord xyz)
            {
                ValueType sum = zeroVal<ValueType>();
                Int32& i = xyz[Axis], j = i + width;
                for (i -= width; i <= j; ++i) sum += acc.getValue(xyz);
                return sum*frac;
            }
            typename GridT::ConstAccessor acc;
            const Int32 width;
            const ValueType frac;
        };

        template<typename AvgT>
        void boxImpl(const LeafRange& r, Int32 w);

        void boxXImpl(const LeafRange& r, Int32 w) { this->boxImpl<Avg<0> >(r,w); }
        void boxZImpl(const LeafRange& r, Int32 w) { this->boxImpl<Avg<1> >(r,w); }
        void boxYImpl(const LeafRange& r, Int32 w) { this->boxImpl<Avg<2> >(r,w); }

        void medianImpl(const LeafRange&, int);
        void meanCurvatureImpl(const LeafRange&);
        void laplacianImpl(const LeafRange&);
        void offsetImpl(const LeafRange&, ValueType);

        LevelSetFilter* mParent;
        const MaskType* mMask;
        typename std::function<void (Filter*, const LeafRange&)> mTask;
    }; // end of private Filter struct

    AlphaType mMinMask, mMaxMask;
    bool      mInvertMask;

}; // end of LevelSetFilter class


////////////////////////////////////////

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::Filter::median(int width)
{
    mParent->startInterrupter("Median-value flow of level set");

    mParent->leafs().rebuildAuxBuffers(1, mParent->getGrainSize()==0);

    mTask = std::bind(&Filter::medianImpl,
        std::placeholders::_1, std::placeholders::_2, std::max(1, width));
    this->cook(true);

    mParent->track();

    mParent->endInterrupter();
}

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::Filter::mean(int width)
{
    mParent->startInterrupter("Mean-value flow of level set");

    this->box(width);

    mParent->endInterrupter();
}

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::Filter::gaussian(int width)
{
    mParent->startInterrupter("Gaussian flow of level set");

    for (int n=0; n<4; ++n) this->box(width);

    mParent->endInterrupter();
}

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::Filter::box(int width)
{
    mParent->leafs().rebuildAuxBuffers(1, mParent->getGrainSize()==0);

    width = std::max(1, width);

    mTask = std::bind(&Filter::boxXImpl, std::placeholders::_1, std::placeholders::_2, width);
    this->cook(true);

    mTask = std::bind(&Filter::boxYImpl, std::placeholders::_1, std::placeholders::_2, width);
    this->cook(true);

    mTask = std::bind(&Filter::boxZImpl, std::placeholders::_1, std::placeholders::_2, width);
    this->cook(true);

    mParent->track();
}

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::Filter::meanCurvature()
{
    mParent->startInterrupter("Mean-curvature flow of level set");

    mParent->leafs().rebuildAuxBuffers(1, mParent->getGrainSize()==0);

    mTask = std::bind(&Filter::meanCurvatureImpl, std::placeholders::_1, std::placeholders::_2);
    this->cook(true);

    mParent->track();

    mParent->endInterrupter();
}

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::Filter::laplacian()
{
    mParent->startInterrupter("Laplacian flow of level set");

    mParent->leafs().rebuildAuxBuffers(1, mParent->getGrainSize()==0);

    mTask = std::bind(&Filter::laplacianImpl, std::placeholders::_1, std::placeholders::_2);
    this->cook(true);

    mParent->track();

    mParent->endInterrupter();
}

template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::Filter::offset(ValueType value)
{
    mParent->startInterrupter("Offsetting level set");

    mParent->leafs().removeAuxBuffers();// no auxiliary buffers required

    const ValueType CFL = ValueType(0.5) * mParent->voxelSize(), offset = openvdb::math::Abs(value);
    ValueType dist = 0.0;
    while (offset-dist > ValueType(0.001)*CFL && mParent->checkInterrupter()) {
        const ValueType delta = openvdb::math::Min(offset-dist, CFL);
        dist += delta;

        mTask = std::bind(&Filter::offsetImpl,
            std::placeholders::_1, std::placeholders::_2, copysign(delta, value));
        this->cook(false);

        mParent->track();
    }

    mParent->endInterrupter();
}


///////////////////////// PRIVATE METHODS //////////////////////

/// Performs parabolic mean-curvature diffusion
template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::Filter::meanCurvatureImpl(const LeafRange& range)
{
    mParent->checkInterrupter();
    //const float CFL = 0.9f, dt = CFL * mDx * mDx / 6.0f;
    const ValueType dx = mParent->voxelSize(), dt = math::Pow2(dx) / ValueType(3.0);
    math::CurvatureStencil<GridType> stencil(mParent->grid(), dx);
    if (mMask) {
        typename AlphaMaskT::FloatType a, b;
        AlphaMaskT alpha(mParent->grid(), *mMask, mParent->minMask(),
                         mParent->maxMask(), mParent->isMaskInverted());
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            ValueType* buffer = leafIter.buffer(1).data();
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                if (alpha(iter.getCoord(), a, b)) {
                    stencil.moveTo(iter);
                    const ValueType phi0 = *iter, phi1 = phi0 + dt*stencil.meanCurvatureNormGrad();
                    buffer[iter.pos()] = b * phi0 + a * phi1;
                }
            }
        }
    } else {
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            ValueType* buffer = leafIter.buffer(1).data();
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                stencil.moveTo(iter);
                buffer[iter.pos()] = *iter + dt*stencil.meanCurvatureNormGrad();
            }
        }
    }
}

/// Performs Laplacian diffusion. Note if the grids contains a true
/// signed distance field (e.g. a solution to the Eikonal equation)
/// Laplacian diffusions (e.g. geometric heat equation) is actually
/// identical to mean curvature diffusion, yet less computationally
/// expensive! In other words if you're performing renormalization
/// anyway (e.g. rebuilding the narrow-band) you should consider
/// performing Laplacian diffusion over mean curvature flow!
template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::Filter::laplacianImpl(const LeafRange& range)
{
    mParent->checkInterrupter();
    //const float CFL = 0.9f, half_dt = CFL * mDx * mDx / 12.0f;
    const ValueType dx = mParent->voxelSize(), dt = math::Pow2(dx) / ValueType(6.0);
    math::GradStencil<GridType> stencil(mParent->grid(), dx);
    if (mMask) {
        typename AlphaMaskT::FloatType a, b;
        AlphaMaskT alpha(mParent->grid(), *mMask, mParent->minMask(),
                         mParent->maxMask(), mParent->isMaskInverted());
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            ValueType* buffer = leafIter.buffer(1).data();
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                if (alpha(iter.getCoord(), a, b)) {
                    stencil.moveTo(iter);
                    const ValueType phi0 = *iter, phi1 = phi0 + dt*stencil.laplacian();
                    buffer[iter.pos()] = b * phi0 + a * phi1;
                }
            }
        }
    } else {
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            ValueType* buffer = leafIter.buffer(1).data();
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                stencil.moveTo(iter);
                buffer[iter.pos()] = *iter + dt*stencil.laplacian();
            }
        }
    }
}

/// Offsets the values by a constant
template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::Filter::offsetImpl(
    const LeafRange& range, ValueType offset)
{
    mParent->checkInterrupter();
    if (mMask) {
        typename AlphaMaskT::FloatType a, b;
        AlphaMaskT alpha(mParent->grid(), *mMask, mParent->minMask(),
                         mParent->maxMask(), mParent->isMaskInverted());
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            for (VoxelIterT iter = leafIter->beginValueOn(); iter; ++iter) {
                if (alpha(iter.getCoord(), a, b)) iter.setValue(*iter + a*offset);
            }
        }
    } else {
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            for (VoxelIterT iter = leafIter->beginValueOn(); iter; ++iter) {
                iter.setValue(*iter + offset);
            }
        }
    }
}

/// Performs simple but slow median-value diffusion
template<typename GridT, typename MaskT, typename InterruptT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::Filter::medianImpl(const LeafRange& range, int width)
{
    mParent->checkInterrupter();
    typename math::DenseStencil<GridType> stencil(mParent->grid(), width);//creates local cache!
    if (mMask) {
        typename AlphaMaskT::FloatType a, b;
        AlphaMaskT alpha(mParent->grid(), *mMask, mParent->minMask(),
                         mParent->maxMask(), mParent->isMaskInverted());
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            ValueType* buffer = leafIter.buffer(1).data();
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                if (alpha(iter.getCoord(), a, b)) {
                    stencil.moveTo(iter);
                    buffer[iter.pos()] = b * (*iter) + a * stencil.median();
                }
            }
        }
    } else {
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            ValueType* buffer = leafIter.buffer(1).data();
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                stencil.moveTo(iter);
                buffer[iter.pos()] = stencil.median();
            }
        }
    }
}

/// One dimensional convolution of a separable box filter
template<typename GridT, typename MaskT, typename InterruptT>
template <typename AvgT>
inline void
LevelSetFilter<GridT, MaskT, InterruptT>::Filter::boxImpl(const LeafRange& range, Int32 w)
{
    mParent->checkInterrupter();
    AvgT avg(mParent->grid(), w);
    if (mMask) {
        typename AlphaMaskT::FloatType a, b;
        AlphaMaskT alpha(mParent->grid(), *mMask, mParent->minMask(),
                         mParent->maxMask(), mParent->isMaskInverted());
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            ValueType* buffer = leafIter.buffer(1).data();
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                const Coord xyz = iter.getCoord();
                if (alpha(xyz, a, b)) buffer[iter.pos()] = b * (*iter)+ a * avg(xyz);
            }
        }
    } else {
        for (LeafIterT leafIter=range.begin(); leafIter; ++leafIter) {
            ValueType* buffer = leafIter.buffer(1).data();
            for (VoxelCIterT iter = leafIter->cbeginValueOn(); iter; ++iter) {
                buffer[iter.pos()] = avg(iter.getCoord());
            }
        }
    }
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETFILTER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
