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
/// @file Diagnostics.h
///
/// @author Ken Museth
///
/// @brief Various diagnostic tools to identify potential issues with
///        for example narrow-band level sets or fog volumes
///
#ifndef OPENVDB_TOOLS_DIAGNOSTICS_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_DIAGNOSTICS_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Vec3.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/math/Operators.h>
#include <openvdb/tree/LeafManager.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <cmath> // for std::isnan(), std::isfinite()
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

////////////////////////////////////////////////////////////////////////////////

/// @brief Perform checks on a grid to see if it is a valid symmetric,
/// narrow-band level set.
///
/// @param grid      Grid to be checked
/// @param number    Number of the checks to be performed (see below)
/// @return string with a message indicating the nature of the
/// issue. If no issue is detected the return string is empty.
///
/// @details @a number refers to the following ordered list of
/// checks - always starting from the top.
/// Fast checks
/// 1: value type is floating point
/// 2: has level set class type
/// 3: has uniform scale
/// 4: background value is positive and n*dx
///
/// Slower checks
/// 5: no active tiles
/// 6: all the values are finite, i.e not NaN or infinite
/// 7: active values in range between +-background
/// 8: abs of inactive values = background, i.e. assuming a symmetric
/// narrow band!
///
/// Relatively slow check (however multithreaded)
/// 9: norm gradient is close to one, i.e. satisfied the Eikonal equation.
template<class GridType>
std::string
checkLevelSet(const GridType& grid, size_t number=9);

////////////////////////////////////////////////////////////////////////////////

/// @brief Perform checks on a grid to see if it is a valid fog volume.
///
/// @param grid      Grid to be checked
/// @param number    Number of the checks to be performed (see below)
/// @return string with a message indicating the nature of the
/// issue. If no issue is detected the return string is empty.
///
/// @details @a number refers to the following ordered list of
/// checks - always starting from the top.
/// Fast checks
/// 1: value type is floating point
/// 2: has FOG volume class type
/// 3: background value is zero
///
/// Slower checks
/// 4: all the values are finite, i.e not NaN or infinite
/// 5: inactive values are zero
/// 6: active values are in the range [0,1]
template<class GridType>
std::string
checkFogVolume(const GridType& grid, size_t number=6);

////////////////////////////////////////////////////////////////////////////////

/// @brief  Threaded method to find unique inactive values.
///
/// @param grid         A VDB volume.
/// @param values       List of unique inactive values, returned by this method.
/// @param numValues    Number of values to look for.
/// @return @c false if the @a grid has more than @a numValues inactive values.
template<class GridType>
bool
uniqueInactiveValues(const GridType& grid,
    std::vector<typename GridType::ValueType>& values, size_t numValues);


////////////////////////////////////////////////////////////////////////////////

/// @brief Checks NaN values
template<typename GridT, typename TreeIterT = typename GridT::ValueOnCIter>
struct CheckNan
{
    using ElementType = typename VecTraits<typename GridT::ValueType>::ElementType;
    using TileIterT = TreeIterT;
    using VoxelIterT = typename tree::IterTraits<
        typename TreeIterT::NodeT, typename TreeIterT::ValueIterT>::template
            NodeConverter<typename GridT::TreeType::LeafNodeType>::Type;

    /// @brief Default constructor
    CheckNan() {}

    /// Return true if the scalar value is NaN
    inline bool operator()(const ElementType& v) const { return std::isnan(v); }

    /// @brief This allows for vector values to be checked component-wise
    template<typename T>
    inline typename std::enable_if<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const
    {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;//should unroll
        return false;
    }

    /// @brief Return true if the tile at the iterator location is NaN
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the voxel at the iterator location is NaN
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const { return "NaN"; }

};// CheckNan

////////////////////////////////////////////////////////////////////////////////

/// @brief Checks for infinite values, e.g. 1/0 or -1/0
template <typename GridT,
          typename TreeIterT = typename GridT::ValueOnCIter>
struct CheckInf
{
    using ElementType = typename VecTraits<typename GridT::ValueType>::ElementType;
    using TileIterT = TreeIterT;
    using VoxelIterT = typename tree::IterTraits<typename TreeIterT::NodeT,
        typename TreeIterT::ValueIterT> ::template NodeConverter<
            typename GridT::TreeType::LeafNodeType>::Type;

    /// @brief Default constructor
    CheckInf() {}

    /// Return true if the value is infinite
    inline bool operator()(const ElementType& v) const { return std::isinf(v); }

    /// Return true if any of the vector components are infinite.
    template<typename T>
    inline typename std::enable_if<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const
    {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;
        return false;
    }

    /// @brief Return true if the tile at the iterator location is infinite
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the tile at the iterator location is infinite
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const { return "infinite"; }
};// CheckInf

////////////////////////////////////////////////////////////////////////////////

/// @brief Checks for both NaN and inf values, i.e. any value that is not finite.
template <typename GridT,
          typename TreeIterT = typename GridT::ValueOnCIter>
struct CheckFinite
{
    using ElementType = typename VecTraits<typename GridT::ValueType>::ElementType;
    using TileIterT = TreeIterT;
    using VoxelIterT = typename tree::IterTraits<typename TreeIterT::NodeT,
        typename TreeIterT::ValueIterT> ::template NodeConverter<
            typename GridT::TreeType::LeafNodeType>::Type;

    /// @brief Default constructor
    CheckFinite() {}

    /// Return true if the value is NOT finite, i.e. it's NaN or infinite
    inline bool operator()(const ElementType& v) const { return !std::isfinite(v); }

    /// Return true if any of the vector components are NaN or infinite.
    template<typename T>
    inline typename std::enable_if<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;
        return false;
    }

    /// @brief Return true if the tile at the iterator location is NaN or infinite.
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the tile at the iterator location is NaN or infinite.
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const { return "not finite"; }
};// CheckFinite

////////////////////////////////////////////////////////////////////////////////

/// @brief Check that the magnitude of a value, a, is close to a fixed
/// magnitude, b, given a fixed tolerance c. That is | |a| - |b| | <= c
template <typename GridT,
          typename TreeIterT = typename GridT::ValueOffCIter>
struct CheckMagnitude
{
    using ElementType = typename VecTraits<typename GridT::ValueType>::ElementType;
    using TileIterT = TreeIterT;
    using VoxelIterT = typename tree::IterTraits<typename TreeIterT::NodeT,
        typename TreeIterT::ValueIterT> ::template NodeConverter<
            typename GridT::TreeType::LeafNodeType>::Type;

    /// @brief Default constructor
    CheckMagnitude(const ElementType& a,
                   const ElementType& t = math::Tolerance<ElementType>::value())
        : absVal(math::Abs(a)), tolVal(math::Abs(t))
    {
    }

    /// Return true if the magnitude of the value is not approximately
    /// equal to totVal.
    inline bool operator()(const ElementType& v) const
    {
        return math::Abs(math::Abs(v) - absVal) > tolVal;
    }

    /// Return true if any of the vector components are infinite.
    template<typename T>
    inline typename std::enable_if<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const
    {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;
        return false;
    }

    /// @brief Return true if the tile at the iterator location is infinite
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the tile at the iterator location is infinite
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const
    {
        std::ostringstream ss;
        ss << "not equal to +/-"<<absVal<<" with a tolerance of "<<tolVal;
        return ss.str();
    }

    const ElementType absVal, tolVal;
};// CheckMagnitude

////////////////////////////////////////////////////////////////////////////////

/// @brief Checks a value against a range
template <typename GridT,
          bool MinInclusive = true,//is min part of the range?
          bool MaxInclusive = true,//is max part of the range?
          typename TreeIterT = typename GridT::ValueOnCIter>
struct CheckRange
{
    using ElementType = typename VecTraits<typename GridT::ValueType>::ElementType;
    using TileIterT = TreeIterT;
    using VoxelIterT = typename tree::IterTraits<typename TreeIterT::NodeT,
        typename TreeIterT::ValueIterT> ::template NodeConverter<
            typename GridT::TreeType::LeafNodeType>::Type;

    // @brief Constructor taking a range to be tested against.
    CheckRange(const ElementType& _min, const ElementType& _max) : minVal(_min), maxVal(_max)
    {
        if (minVal > maxVal) {
            OPENVDB_THROW(ValueError, "CheckRange: Invalid range (min > max)");
        }
    }

    /// Return true if the value is smaller than min or larger than max.
    inline bool operator()(const ElementType& v) const
    {
        return (MinInclusive ? v<minVal : v<=minVal) ||
               (MaxInclusive ? v>maxVal : v>=maxVal);
    }

    /// Return true if any of the vector components are out of range.
    template<typename T>
    inline typename std::enable_if<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;
        return false;
    }

    /// @brief Return true if the voxel at the iterator location is out of range.
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the tile at the iterator location is out of range.
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const
    {
        std::ostringstream ss;
        ss << "outside the value range " << (MinInclusive ? "[" : "]")
           << minVal << "," << maxVal    << (MaxInclusive ? "]" : "[");
        return ss.str();
    }

    const ElementType minVal, maxVal;
};// CheckRange

////////////////////////////////////////////////////////////////////////////////

/// @brief Checks a value against a minimum
template <typename GridT,
          typename TreeIterT = typename GridT::ValueOnCIter>
struct CheckMin
{
    using ElementType = typename VecTraits<typename GridT::ValueType>::ElementType;
    using TileIterT = TreeIterT;
    using VoxelIterT = typename tree::IterTraits<typename TreeIterT::NodeT,
        typename TreeIterT::ValueIterT> ::template NodeConverter<
            typename GridT::TreeType::LeafNodeType>::Type;

    // @brief Constructor taking a minimum to be tested against.
    CheckMin(const ElementType& _min) : minVal(_min) {}

    /// Return true if the value is smaller than min.
    inline bool operator()(const ElementType& v) const { return v<minVal; }

    /// Return true if any of the vector components are smaller than min.
    template<typename T>
    inline typename std::enable_if<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;
        return false;
    }

    /// @brief Return true if the voxel at the iterator location is smaller than min.
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the tile at the iterator location is smaller than min.
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const
    {
        std::ostringstream ss;
        ss << "smaller than "<<minVal;
        return ss.str();
    }

    const ElementType minVal;
};// CheckMin

////////////////////////////////////////////////////////////////////////////////

/// @brief Checks a value against a maximum
template <typename GridT,
          typename TreeIterT = typename GridT::ValueOnCIter>
struct CheckMax
{
    using ElementType = typename VecTraits<typename GridT::ValueType>::ElementType;
    using TileIterT = TreeIterT;
    using VoxelIterT = typename tree::IterTraits<typename TreeIterT::NodeT,
        typename TreeIterT::ValueIterT> ::template NodeConverter<
            typename GridT::TreeType::LeafNodeType>::Type;

    /// @brief Constructor taking a maximum to be tested against.
    CheckMax(const ElementType& _max) : maxVal(_max) {}

    /// Return true if the value is larger than max.
    inline bool operator()(const ElementType& v) const { return v>maxVal; }

    /// Return true if any of the vector components are larger than max.
    template<typename T>
    inline typename std::enable_if<VecTraits<T>::IsVec, bool>::type
    operator()(const T& v) const {
        for (int i=0; i<VecTraits<T>::Size; ++i) if ((*this)(v[i])) return true;
        return false;
    }

    /// @brief Return true if the tile at the iterator location is larger than max.
    bool operator()(const TreeIterT  &iter) const { return (*this)(*iter); }

    /// @brief Return true if the voxel at the iterator location is larger than max.
    bool operator()(const VoxelIterT &iter) const { return (*this)(*iter); }

    /// @brief Return a string describing a failed check.
    std::string str() const
    {
        std::ostringstream ss;
        ss << "larger than "<<maxVal;
        return ss.str();
    }

    const ElementType maxVal;
};// CheckMax

////////////////////////////////////////////////////////////////////////////////

/// @brief Checks the norm of the gradient against a range, i.e.,
/// |&nabla;&Phi;| &isin; [min, max]
///
/// @note Internally the test is performed as
/// |&nabla;&Phi;|&sup2; &isin; [min&sup2;, max&sup2;] for optimization reasons.
template<typename GridT,
         typename TreeIterT = typename GridT::ValueOnCIter,
         math::BiasedGradientScheme GradScheme = math::FIRST_BIAS>//math::WENO5_BIAS>
struct CheckNormGrad
{
    using ValueType = typename GridT::ValueType;
    static_assert(std::is_floating_point<ValueType>::value,
        "openvdb::tools::CheckNormGrad requires a scalar, floating-point grid");
    using TileIterT = TreeIterT;
    using VoxelIterT = typename tree::IterTraits<typename TreeIterT::NodeT,
        typename TreeIterT::ValueIterT> ::template NodeConverter<
            typename GridT::TreeType::LeafNodeType>::Type;
    using AccT = typename GridT::ConstAccessor;

    /// @brief Constructor taking a grid and a range to be tested against.
    CheckNormGrad(const GridT&  grid, const ValueType& _min, const ValueType& _max)
        : acc(grid.getConstAccessor())
        , invdx2(ValueType(1.0/math::Pow2(grid.voxelSize()[0])))
        , minVal2(_min*_min)
        , maxVal2(_max*_max)
    {
        if ( !grid.hasUniformVoxels() ) {
            OPENVDB_THROW(ValueError, "CheckNormGrad: The transform must have uniform scale");
        }
        if (_min > _max) {
            OPENVDB_THROW(ValueError, "CheckNormGrad: Invalid range (min > max)");
        }
    }

    CheckNormGrad(const CheckNormGrad& other)
        : acc(other.acc.tree())
        , invdx2(other.invdx2)
        , minVal2(other.minVal2)
        , maxVal2(other.maxVal2)
    {
    }

    /// Return true if the value is smaller than min or larger than max.
    inline bool operator()(const ValueType& v) const { return v<minVal2 || v>maxVal2; }

    /// @brief Return true if zero is outside the range.
    /// @note We assume that the norm of the gradient of a tile is always zero.
    inline bool operator()(const TreeIterT&) const { return (*this)(ValueType(0)); }

    /// @brief Return true if the norm of the gradient at a voxel
    /// location of the iterator is out of range.
    inline bool operator()(const VoxelIterT &iter) const
    {
        const Coord ijk = iter.getCoord();
        return (*this)(invdx2 * math::ISGradientNormSqrd<GradScheme>::result(acc, ijk));
    }

    /// @brief Return a string describing a failed check.
    std::string str() const
    {
        std::ostringstream ss;
        ss << "outside the range of NormGrad ["<<math::Sqrt(minVal2)<<","<<math::Sqrt(maxVal2)<<"]";
        return ss.str();
    }

    AccT acc;
    const ValueType invdx2, minVal2, maxVal2;
};// CheckNormGrad

////////////////////////////////////////////////////////////////////////////////

/// @brief Checks the norm of the gradient at zero-crossing voxels against a range
/// @details CheckEikonal differs from CheckNormGrad in that it only
/// checks the norm of the gradient at voxel locations where the
/// FD-stencil crosses the zero isosurface!
template<typename GridT,
         typename TreeIterT = typename GridT::ValueOnCIter,
         typename StencilT  = math::WenoStencil<GridT> >//math::GradStencil<GridT>
struct CheckEikonal
{
    using ValueType = typename GridT::ValueType;
    static_assert(std::is_floating_point<ValueType>::value,
        "openvdb::tools::CheckEikonal requires a scalar, floating-point grid");
    using TileIterT = TreeIterT;
    using VoxelIterT = typename tree::IterTraits<typename TreeIterT::NodeT,
        typename TreeIterT::ValueIterT> ::template NodeConverter<
            typename GridT::TreeType::LeafNodeType>::Type;

    /// @brief Constructor taking a grid and a range to be tested against.
    CheckEikonal(const GridT&  grid, const ValueType& _min, const ValueType& _max)
        : stencil(grid), minVal(_min), maxVal(_max)
    {
        if ( !grid.hasUniformVoxels() ) {
            OPENVDB_THROW(ValueError, "CheckEikonal: The transform must have uniform scale");
        }
        if (minVal > maxVal) {
            OPENVDB_THROW(ValueError, "CheckEikonal: Invalid range (min > max)");
        }
    }

    CheckEikonal(const CheckEikonal& other)
        : stencil(other.stencil.grid()), minVal(other.minVal), maxVal(other.maxVal)
    {
    }

    /// Return true if the value is smaller than min or larger than max.
    inline bool operator()(const ValueType& v) const { return v<minVal || v>maxVal; }

    /// @brief Return true if zero is outside the range.
    /// @note We assume that the norm of the gradient of a tile is always zero.
    inline bool operator()(const TreeIterT&) const { return (*this)(ValueType(0)); }

    /// @brief Return true if the norm of the gradient at a
    /// zero-crossing voxel location of the iterator is out of range.
    inline bool operator()(const VoxelIterT &iter) const
    {
        stencil.moveTo(iter);
        if (!stencil.zeroCrossing()) return false;
        return (*this)(stencil.normSqGrad());
    }

    /// @brief Return a string describing a failed check.
    std::string str() const
    {
        std::ostringstream ss;
        ss << "outside the range of NormGrad ["<<minVal<<","<<maxVal<<"]";
        return ss.str();
    }

    mutable StencilT stencil;
    const ValueType minVal, maxVal;
};// CheckEikonal

////////////////////////////////////////////////////////////////////////////////

/// @brief Checks the divergence against a range
template<typename GridT,
         typename TreeIterT = typename GridT::ValueOnCIter,
         math::DScheme DiffScheme = math::CD_2ND>
struct CheckDivergence
{
    using ValueType = typename GridT::ValueType;
    using ElementType = typename VecTraits<ValueType>::ElementType;
    static_assert(std::is_floating_point<ElementType>::value,
        "openvdb::tools::CheckDivergence requires a floating-point vector grid");
    using TileIterT = TreeIterT;
    using VoxelIterT = typename tree::IterTraits<typename TreeIterT::NodeT,
        typename TreeIterT::ValueIterT>::template NodeConverter<
            typename GridT::TreeType::LeafNodeType>::Type;
    using AccT = typename GridT::ConstAccessor;

    /// @brief Constructor taking a grid and a range to be tested against.
    CheckDivergence(const GridT&  grid,
                    const ValueType& _min,
                    const ValueType& _max)
        : acc(grid.getConstAccessor())
        , invdx(ValueType(1.0/grid.voxelSize()[0]))
        , minVal(_min)
        , maxVal(_max)
    {
        if ( !grid.hasUniformVoxels() ) {
            OPENVDB_THROW(ValueError, "CheckDivergence: The transform must have uniform scale");
        }
        if (minVal > maxVal) {
            OPENVDB_THROW(ValueError, "CheckDivergence: Invalid range (min > max)");
        }
    }
    /// Return true if the value is smaller than min or larger than max.
    inline bool operator()(const ElementType& v) const { return v<minVal || v>maxVal; }

    /// @brief Return true if zero is outside the range.
    /// @note We assume that the divergence of a tile is always zero.
    inline bool operator()(const TreeIterT&) const { return (*this)(ElementType(0)); }

    /// @brief Return true if the divergence at a voxel location of
    /// the iterator is out of range.
    inline bool operator()(const VoxelIterT &iter) const
    {
        const Coord ijk = iter.getCoord();
        return (*this)(invdx * math::ISDivergence<DiffScheme>::result(acc, ijk));
    }

    /// @brief Return a string describing a failed check.
    std::string str() const
    {
        std::ostringstream ss;
        ss << "outside the range of divergence ["<<minVal<<","<<maxVal<<"]";
        return ss.str();
    }

    AccT acc;
    const ValueType invdx, minVal, maxVal;
};// CheckDivergence

////////////////////////////////////////////////////////////////////////////////

/// @brief Performs multithreaded diagnostics of a grid
/// @note More documentation will be added soon!
template <typename GridT>
class Diagnose
{
public:
    using MaskType = typename GridT::template ValueConverter<bool>::Type;

    Diagnose(const GridT& grid) : mGrid(&grid), mMask(new MaskType()), mCount(0)
    {
        mMask->setTransform(grid.transformPtr()->copy());
    }

    template <typename CheckT>
    std::string check(const CheckT& check,
                      bool updateMask = false,
                      bool checkVoxels = true,
                      bool checkTiles = true,
                      bool checkBackground = true)
    {
        typename MaskType::TreeType* mask = updateMask ? &(mMask->tree()) : nullptr;
        CheckValues<CheckT> cc(mask, mGrid, check);
        std::ostringstream ss;
        if (checkBackground) ss << cc.checkBackground();
        if (checkTiles)      ss << cc.checkTiles();
        if (checkVoxels)     ss << cc.checkVoxels();
        mCount += cc.mCount;
        return ss.str();
    }

    //@{
    /// @brief Return a boolean mask of all the values
    /// (i.e. tiles and/or voxels) that have failed one or
    /// more checks.
    typename MaskType::ConstPtr mask() const { return mMask; }
    typename MaskType::Ptr mask() { return mMask; }
    //@}

    /// @brief Return the number of values (i.e. background, tiles or
    /// voxels) that have failed one or more checks.
    Index64 valueCount() const { return mMask->activeVoxelCount(); }

    /// @brief Return total number of failed checks
    /// @note If only one check was performed and the mask was updated
    /// failureCount equals valueCount.
    Index64 failureCount() const { return mCount; }

    /// @brief Return a const reference to the grid
    const GridT& grid() const { return *mGrid; }

    /// @brief Clear the mask and error counter
    void clear() { mMask = new MaskType(); mCount = 0; }

private:
    // disallow copy construction and copy by assignment!
    Diagnose(const Diagnose&);// not implemented
    Diagnose& operator=(const Diagnose&);// not implemented

    const GridT*           mGrid;
    typename MaskType::Ptr mMask;
    Index64                mCount;

    /// @brief Private class that performs the multithreaded checks
    template <typename CheckT>
    struct CheckValues
    {
        using MaskT = typename MaskType::TreeType;
        using LeafT = typename GridT::TreeType::LeafNodeType;
        using LeafManagerT = typename tree::LeafManager<const typename GridT::TreeType>;
        const bool      mOwnsMask;
        MaskT*          mMask;
        const GridT*    mGrid;
        const CheckT    mCheck;
        Index64         mCount;

        CheckValues(MaskT* mask, const GridT* grid, const CheckT& check)
            : mOwnsMask(false)
            , mMask(mask)
            , mGrid(grid)
            , mCheck(check)
            , mCount(0)
        {
        }
        CheckValues(CheckValues& other, tbb::split)
            : mOwnsMask(true)
            , mMask(other.mMask ? new MaskT() : nullptr)
            , mGrid(other.mGrid)
            , mCheck(other.mCheck)
            , mCount(0)
        {
        }
        ~CheckValues() { if (mOwnsMask) delete mMask; }

        std::string checkBackground()
        {
            std::ostringstream ss;
            if (mCheck(mGrid->background())) {
                ++mCount;
                ss << "Background is " + mCheck.str() << std::endl;
            }
            return ss.str();
        }

        std::string checkTiles()
        {
            std::ostringstream ss;
            const Index64 n = mCount;
            typename CheckT::TileIterT i(mGrid->tree());
            for (i.setMaxDepth(GridT::TreeType::RootNodeType::LEVEL - 1); i; ++i) {
                if (mCheck(i)) {
                    ++mCount;
                    if (mMask) mMask->fill(i.getBoundingBox(), true, true);
                }
            }
            if (const Index64 m = mCount - n) {
                ss << m << " tile" << (m==1 ? " is " : "s are ") + mCheck.str() << std::endl;
            }
            return ss.str();
        }

        std::string checkVoxels()
        {
            std::ostringstream ss;
            LeafManagerT leafs(mGrid->tree());
            const Index64 n = mCount;
            tbb::parallel_reduce(leafs.leafRange(), *this);
            if (const Index64 m = mCount - n) {
                ss << m << " voxel" << (m==1 ? " is " : "s are ") + mCheck.str() << std::endl;
            }
            return ss.str();
        }

        void operator()(const typename LeafManagerT::LeafRange& r)
        {
            using VoxelIterT = typename CheckT::VoxelIterT;
            if (mMask) {
                for (typename LeafManagerT::LeafRange::Iterator i=r.begin(); i; ++i) {
                    typename MaskT::LeafNodeType* maskLeaf = nullptr;
                    for (VoxelIterT j = tree::IterTraits<LeafT, VoxelIterT>::begin(*i); j; ++j) {
                        if (mCheck(j)) {
                            ++mCount;
                            if (maskLeaf == nullptr) maskLeaf = mMask->touchLeaf(j.getCoord());
                            maskLeaf->setValueOn(j.pos(), true);
                        }
                    }
                }
            } else {
                for (typename LeafManagerT::LeafRange::Iterator i=r.begin(); i; ++i) {
                    for (VoxelIterT j = tree::IterTraits<LeafT, VoxelIterT>::begin(*i); j; ++j) {
                        if (mCheck(j)) ++mCount;
                    }
                }
            }
        }
        void join(const CheckValues& other)
        {
            if (mMask) mMask->merge(*(other.mMask), openvdb::MERGE_ACTIVE_STATES_AND_NODES);
            mCount += other.mCount;
        }
    };//End of private class CheckValues

};// End of public class Diagnose


////////////////////////////////////////////////////////////////////////////////

/// @brief Class that performs various types of checks on narrow-band level sets.
///
/// @note The most common usage is to simply call CheckLevelSet::check()
template<class GridType>
class CheckLevelSet
{
public:
    using ValueType = typename GridType::ValueType;
    using MaskType = typename GridType::template ValueConverter<bool>::Type;

    CheckLevelSet(const GridType& grid) : mDiagnose(grid) {}

    //@{
    /// @brief Return a boolean mask of all the values
    /// (i.e. tiles and/or voxels) that have failed one or
    /// more checks.
    typename MaskType::ConstPtr mask() const { return mDiagnose.mask(); }
    typename MaskType::Ptr mask() { return mDiagnose.mask(); }
    //@}

    /// @brief Return the number of values (i.e. background, tiles or
    /// voxels) that have failed one or more checks.
    Index64 valueCount() const { return mDiagnose.valueCount(); }

    /// @brief Return total number of failed checks
    /// @note If only one check was performed and the mask was updated
    /// failureCount equals valueCount.
    Index64 failureCount() const { return mDiagnose.failureCount(); }

    /// @brief Return a const reference to the grid
    const GridType& grid() const { return mDiagnose.grid(); }

    /// @brief Clear the mask and error counter
    void clear() { mDiagnose.clear(); }

    /// @brief Return a nonempty message if the grid's value type is a floating point.
    ///
    /// @note No run-time overhead
    static std::string checkValueType()
    {
        static const bool test = std::is_floating_point<ValueType>::value;
        return test ? "" : "Value type is not floating point\n";
    }

    /// @brief Return message if the grid's class is a level set.
    ///
    /// @note Small run-time overhead
    std::string checkClassType() const
    {
        const bool test = mDiagnose.grid().getGridClass() == GRID_LEVEL_SET;
        return test ? "" : "Class type is not \"GRID_LEVEL_SET\"\n";
    }

    /// @brief Return a nonempty message if the grid's transform does not have uniform scaling.
    ///
    /// @note Small run-time overhead
    std::string checkTransform() const
    {
        return mDiagnose.grid().hasUniformVoxels() ? "" : "Does not have uniform voxels\n";
    }

    /// @brief Return a nonempty message if the background value is larger than or
    /// equal to the halfWidth*voxelSize.
    ///
    /// @note Small run-time overhead
    std::string checkBackground(Real halfWidth = LEVEL_SET_HALF_WIDTH) const
    {
        const Real w = mDiagnose.grid().background() / mDiagnose.grid().voxelSize()[0];
        if (w < halfWidth) {
            std::ostringstream ss;
            ss << "The background value ("<< mDiagnose.grid().background()<<") is less than "
               << halfWidth << " voxel units\n";
            return ss.str();
        }
        return "";
    }

    /// @brief Return a nonempty message if the grid has no active tile values.
    ///
    /// @note Medium run-time overhead
    std::string checkTiles() const
    {
        const bool test = mDiagnose.grid().tree().hasActiveTiles();
        return test ? "Has active tile values\n" : "";
    }

    /// @brief Return a nonempty message if any of the values are not finite. i.e. NaN or inf.
    ///
    /// @note Medium run-time overhead
    std::string checkFinite(bool updateMask = false)
    {
        CheckFinite<GridType,typename GridType::ValueAllCIter> c;
        return mDiagnose.check(c, updateMask, /*voxel*/true, /*tiles*/true, /*background*/true);
    }

    /// @brief Return a nonempty message if the active voxel values are out-of-range.
    ///
    /// @note Medium run-time overhead
    std::string checkRange(bool updateMask = false)
    {
        const ValueType& background = mDiagnose.grid().background();
        CheckRange<GridType> c(-background, background);
        return mDiagnose.check(c, updateMask, /*voxel*/true, /*tiles*/false, /*background*/false);
    }

    /// @brief Return a nonempty message if the the inactive values do not have a
    /// magnitude equal to the background value.
    ///
    /// @note Medium run-time overhead
    std::string checkInactiveValues(bool updateMask = false)
    {
        const ValueType& background = mDiagnose.grid().background();
        CheckMagnitude<GridType, typename GridType::ValueOffCIter> c(background);
        return mDiagnose.check(c, updateMask, /*voxel*/true, /*tiles*/true, /*background*/false);
    }

    /// @brief Return a nonempty message if the norm of the gradient of the
    /// active voxels is out of the range minV to maxV.
    ///
    /// @note Significant run-time overhead
    std::string checkEikonal(bool updateMask = false, ValueType minV = 0.5, ValueType maxV = 1.5)
    {
        CheckEikonal<GridType> c(mDiagnose.grid(), minV, maxV);
        return mDiagnose.check(c, updateMask, /*voxel*/true, /*tiles*/false, /*background*/false);
    }

    /// @brief Return a nonempty message if an error or issue is detected. Only
    /// runs tests with a number lower than or equal to n, where:
    ///
    /// Fast checks
    /// 1: value type is floating point
    /// 2: has level set class type
    /// 3: has uniform scale
    /// 4: background value is positive and n*dx
    ///
    /// Slower checks
    /// 5: no active tiles
    /// 6: all the values are finite, i.e not NaN or infinite
    /// 7: active values in range between +-background
    /// 8: abs of inactive values = background, i.e. assuming a symmetric narrow band!
    ///
    /// Relatively slow check (however multi-threaded)
    /// 9: norm of gradient at zero-crossings is one, i.e. satisfied the Eikonal equation.
    std::string check(size_t n=9, bool updateMask = false)
    {
        std::string str = this->checkValueType();
        if (str.empty() && n>1) str = this->checkClassType();
        if (str.empty() && n>2) str = this->checkTransform();
        if (str.empty() && n>3) str = this->checkBackground();
        if (str.empty() && n>4) str = this->checkTiles();
        if (str.empty() && n>5) str = this->checkFinite(updateMask);
        if (str.empty() && n>6) str = this->checkRange(updateMask);
        if (str.empty() && n>7) str = this->checkInactiveValues(updateMask);
        if (str.empty() && n>8) str = this->checkEikonal(updateMask);
        return str;
    }

private:
    // disallow copy construction and copy by assignment!
    CheckLevelSet(const CheckLevelSet&);// not implemented
    CheckLevelSet& operator=(const CheckLevelSet&);// not implemented

    // Member data
    Diagnose<GridType> mDiagnose;
};// CheckLevelSet

template<class GridType>
std::string
checkLevelSet(const GridType& grid, size_t n)
{
    CheckLevelSet<GridType> c(grid);
    return c.check(n, false);
}

////////////////////////////////////////////////////////////////////////////////

/// @brief Class that performs various types of checks on fog volumes.
///
/// @note The most common usage is to simply call CheckFogVolume::check()
template<class GridType>
class CheckFogVolume
{
public:
    using ValueType = typename GridType::ValueType;
    using MaskType = typename GridType::template ValueConverter<bool>::Type;

    CheckFogVolume(const GridType& grid) : mDiagnose(grid) {}

    //@{
    /// @brief Return a boolean mask of all the values
    /// (i.e. tiles and/or voxels) that have failed one or
    /// more checks.
    typename MaskType::ConstPtr mask() const { return mDiagnose.mask(); }
    typename MaskType::Ptr mask() { return mDiagnose.mask(); }
    //@}

    /// @brief Return the number of values (i.e. background, tiles or
    /// voxels) that have failed one or more checks.
    Index64 valueCount() const { return mDiagnose.valueCount(); }

    /// @brief Return total number of failed checks
    /// @note If only one check was performed and the mask was updated
    /// failureCount equals valueCount.
    Index64 failureCount() const { return mDiagnose.failureCount(); }

    /// @brief Return a const reference to the grid
    const GridType& grid() const { return mDiagnose.grid(); }

    /// @brief Clear the mask and error counter
    void clear() { mDiagnose.clear(); }

    /// @brief Return a nonempty message if the grid's value type is a floating point.
    ///
    /// @note No run-time overhead
    static std::string checkValueType()
    {
        static const bool test = std::is_floating_point<ValueType>::value;
        return test ? "" : "Value type is not floating point";
    }

    /// @brief Return a nonempty message if the grid's class is a level set.
    ///
    /// @note Small run-time overhead
    std::string checkClassType() const
    {
        const bool test = mDiagnose.grid().getGridClass() == GRID_FOG_VOLUME;
        return test ? "" : "Class type is not \"GRID_LEVEL_SET\"";
    }

    /// @brief Return a nonempty message if the background value is not zero.
    ///
    /// @note Small run-time overhead
    std::string checkBackground() const
    {
        if (!math::isApproxZero(mDiagnose.grid().background())) {
            std::ostringstream ss;
            ss << "The background value ("<< mDiagnose.grid().background()<<") is not zero";
            return ss.str();
        }
        return "";
    }

    /// @brief Return a nonempty message if any of the values are not finite. i.e. NaN or inf.
    ///
    /// @note Medium run-time overhead
    std::string checkFinite(bool updateMask = false)
    {
        CheckFinite<GridType,typename GridType::ValueAllCIter> c;
        return mDiagnose.check(c, updateMask, /*voxel*/true, /*tiles*/true, /*background*/true);
    }

    /// @brief Return a nonempty message if any of the inactive values are not zero.
    ///
    /// @note Medium run-time overhead
    std::string checkInactiveValues(bool updateMask = false)
    {
        CheckMagnitude<GridType, typename GridType::ValueOffCIter> c(0);
        return mDiagnose.check(c, updateMask, /*voxel*/true, /*tiles*/true, /*background*/true);
    }

    /// @brief Return a nonempty message if the active voxel values
    /// are out-of-range, i.e. not in the range [0,1].
    ///
    /// @note Medium run-time overhead
    std::string checkRange(bool updateMask = false)
    {
        CheckRange<GridType> c(0, 1);
        return mDiagnose.check(c, updateMask, /*voxel*/true, /*tiles*/true, /*background*/false);
    }

    /// @brief Return a nonempty message if an error or issue is detected. Only
    /// runs tests with a number lower than or equal to n, where:
    ///
    /// Fast checks
    /// 1: value type is floating point
    /// 2: has FOG volume class type
    /// 3: background value is zero
    ///
    /// Slower checks
    /// 4: all the values are finite, i.e not NaN or infinite
    /// 5: inactive values are zero
    /// 6: active values are in the range [0,1]
    std::string check(size_t n=6, bool updateMask = false)
    {
        std::string str = this->checkValueType();
        if (str.empty() && n>1) str = this->checkClassType();
        if (str.empty() && n>2) str = this->checkBackground();
        if (str.empty() && n>3) str = this->checkFinite(updateMask);
        if (str.empty() && n>4) str = this->checkInactiveValues(updateMask);
        if (str.empty() && n>5) str = this->checkRange(updateMask);
        return str;
    }

private:
    // disallow copy construction and copy by assignment!
    CheckFogVolume(const CheckFogVolume&);// not implemented
    CheckFogVolume& operator=(const CheckFogVolume&);// not implemented

    // Member data
    Diagnose<GridType> mDiagnose;
};// CheckFogVolume

template<class GridType>
std::string
checkFogVolume(const GridType& grid, size_t n)
{
    CheckFogVolume<GridType> c(grid);
    return c.check(n, false);
}


////////////////////////////////////////////////////////////////////////////////

// Internal utility objects and implementation details


namespace diagnostics_internal {


template<typename TreeType>
class InactiveVoxelValues
{
public:
    using LeafArray = tree::LeafManager<TreeType>;
    using ValueType = typename TreeType::ValueType;
    using SetType = std::set<ValueType>;

    InactiveVoxelValues(LeafArray&, size_t numValues);

    void runParallel();
    void runSerial();

    void getInactiveValues(SetType&) const;

    inline InactiveVoxelValues(const InactiveVoxelValues<TreeType>&, tbb::split);
    inline void operator()(const tbb::blocked_range<size_t>&);
    inline void join(const InactiveVoxelValues<TreeType>&);

private:
    LeafArray& mLeafArray;
    SetType mInactiveValues;
    size_t mNumValues;
};// InactiveVoxelValues

template<typename TreeType>
InactiveVoxelValues<TreeType>::InactiveVoxelValues(LeafArray& leafs, size_t numValues)
    : mLeafArray(leafs)
    , mInactiveValues()
    , mNumValues(numValues)
{
}

template <typename TreeType>
inline
InactiveVoxelValues<TreeType>::InactiveVoxelValues(
    const InactiveVoxelValues<TreeType>& rhs, tbb::split)
    : mLeafArray(rhs.mLeafArray)
    , mInactiveValues()
    , mNumValues(rhs.mNumValues)
{
}

template<typename TreeType>
void
InactiveVoxelValues<TreeType>::runParallel()
{
    tbb::parallel_reduce(mLeafArray.getRange(), *this);
}


template<typename TreeType>
void
InactiveVoxelValues<TreeType>::runSerial()
{
    (*this)(mLeafArray.getRange());
}


template<typename TreeType>
inline void
InactiveVoxelValues<TreeType>::operator()(const tbb::blocked_range<size_t>& range)
{
    typename TreeType::LeafNodeType::ValueOffCIter iter;

    for (size_t n = range.begin(); n < range.end() && !tbb::task::self().is_cancelled(); ++n) {
        for (iter = mLeafArray.leaf(n).cbeginValueOff(); iter; ++iter) {
            mInactiveValues.insert(iter.getValue());
        }

        if (mInactiveValues.size() > mNumValues) {
            tbb::task::self().cancel_group_execution();
        }
    }
}

template<typename TreeType>
inline void
InactiveVoxelValues<TreeType>::join(const InactiveVoxelValues<TreeType>& rhs)
{
    mInactiveValues.insert(rhs.mInactiveValues.begin(), rhs.mInactiveValues.end());
}

template<typename TreeType>
inline void
InactiveVoxelValues<TreeType>::getInactiveValues(SetType& values) const
{
    values.insert(mInactiveValues.begin(), mInactiveValues.end());
}


////////////////////////////////////////


template<typename TreeType>
class InactiveTileValues
{
public:
    using IterRange = tree::IteratorRange<typename TreeType::ValueOffCIter>;
    using ValueType = typename TreeType::ValueType;
    using SetType = std::set<ValueType>;

    InactiveTileValues(size_t numValues);

    void runParallel(IterRange&);
    void runSerial(IterRange&);

    void getInactiveValues(SetType&) const;

    inline InactiveTileValues(const InactiveTileValues<TreeType>&, tbb::split);
    inline void operator()(IterRange&);
    inline void join(const InactiveTileValues<TreeType>&);

private:
    SetType mInactiveValues;
    size_t mNumValues;
};


template<typename TreeType>
InactiveTileValues<TreeType>::InactiveTileValues(size_t numValues)
    : mInactiveValues()
    , mNumValues(numValues)
{
}

template <typename TreeType>
inline
InactiveTileValues<TreeType>::InactiveTileValues(
    const InactiveTileValues<TreeType>& rhs, tbb::split)
    : mInactiveValues()
    , mNumValues(rhs.mNumValues)
{
}

template<typename TreeType>
void
InactiveTileValues<TreeType>::runParallel(IterRange& range)
{
    tbb::parallel_reduce(range, *this);
}


template<typename TreeType>
void
InactiveTileValues<TreeType>::runSerial(IterRange& range)
{
    (*this)(range);
}


template<typename TreeType>
inline void
InactiveTileValues<TreeType>::operator()(IterRange& range)
{
    for (; range && !tbb::task::self().is_cancelled(); ++range) {
        typename TreeType::ValueOffCIter iter = range.iterator();
        for (; iter; ++iter) {
            mInactiveValues.insert(iter.getValue());
        }

        if (mInactiveValues.size() > mNumValues) {
            tbb::task::self().cancel_group_execution();
        }
    }
}

template<typename TreeType>
inline void
InactiveTileValues<TreeType>::join(const InactiveTileValues<TreeType>& rhs)
{
    mInactiveValues.insert(rhs.mInactiveValues.begin(), rhs.mInactiveValues.end());
}

template<typename TreeType>
inline void
InactiveTileValues<TreeType>::getInactiveValues(SetType& values) const
{
    values.insert(mInactiveValues.begin(), mInactiveValues.end());
}

} // namespace diagnostics_internal


////////////////////////////////////////


template<class GridType>
bool
uniqueInactiveValues(const GridType& grid,
    std::vector<typename GridType::ValueType>& values, size_t numValues)
{
    using TreeType = typename GridType::TreeType;
    using ValueType = typename GridType::ValueType;
    using SetType = std::set<ValueType>;

    SetType uniqueValues;

    { // Check inactive voxels
        TreeType& tree = const_cast<TreeType&>(grid.tree());
        tree::LeafManager<TreeType> leafs(tree);
        diagnostics_internal::InactiveVoxelValues<TreeType> voxelOp(leafs, numValues);
        voxelOp.runParallel();
        voxelOp.getInactiveValues(uniqueValues);
    }

    // Check inactive tiles
    if (uniqueValues.size() <= numValues) {
        typename TreeType::ValueOffCIter iter(grid.tree());
        iter.setMaxDepth(TreeType::ValueAllIter::LEAF_DEPTH - 1);
        diagnostics_internal::InactiveTileValues<TreeType> tileOp(numValues);

        tree::IteratorRange<typename TreeType::ValueOffCIter> range(iter);
        tileOp.runParallel(range);

        tileOp.getInactiveValues(uniqueValues);
    }

    values.clear();
    values.reserve(uniqueValues.size());

    typename SetType::iterator it = uniqueValues.begin();
    for ( ; it != uniqueValues.end(); ++it) {
        values.push_back(*it);
    }

    return values.size() <= numValues;
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_DIAGNOSTICS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
