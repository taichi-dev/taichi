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
/// @file tools/PointScatter.h
///
/// @brief We offer three different algorithms (each in its own class)
///        for scattering of points in active voxels:
///
/// 1) UniformPointScatter. Has two modes: Either randomly distributes
///    a fixed number of points into the active voxels, or the user can
///    specify a fixed probability of having a points per unit of volume.
///
/// 2) DenseUniformPointScatter. Randomly distributes points into active
///    voxels using a fixed number of points per voxel.
///
/// 3) NonIniformPointScatter. Define the local probability of having
///    a point in a voxel as the product of a global density and the
///    value of the voxel itself.

#ifndef OPENVDB_TOOLS_POINT_SCATTER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_SCATTER_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h>
#include <openvdb/util/NullInterrupter.h>
#include <tbb/parallel_sort.h>
#include <tbb/parallel_for.h>
#include <iostream>
#include <memory>
#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// Forward declaration of base class
template<typename PointAccessorType,
         typename RandomGenerator,
         typename InterruptType = util::NullInterrupter>
class BasePointScatter;

/// @brief The two point scatters UniformPointScatter and
/// NonUniformPointScatter depend on the following two classes:
///
/// The @c PointAccessorType template argument below refers to any class
/// with the following interface:
/// @code
/// class PointAccessor {
///   ...
/// public:
///   void add(const openvdb::Vec3R &pos);// appends point with world positions pos
/// };
/// @endcode
///
///
/// The @c InterruptType template argument below refers to any class
/// with the following interface:
/// @code
/// class Interrupter {
///   ...
/// public:
///   void start(const char* name = nullptr) // called when computations begin
///   void end()                             // called when computations end
///   bool wasInterrupted(int percent=-1)    // return true to break computation
///};
/// @endcode
///
/// @note If no template argument is provided for this InterruptType
/// the util::NullInterrupter is used which implies that all
/// interrupter calls are no-ops (i.e. incurs no computational overhead).


/// @brief Uniformly scatters points in the active voxels.
/// The point count is either explicitly defined or implicitly
/// through the specification of a global density (=points-per-volume)
///
/// @note This uniform scattering technique assumes that the number of
/// points is generally smaller than the number of active voxels
/// (including virtual active voxels in active tiles).
template<typename PointAccessorType,
         typename RandomGenerator,
         typename InterruptType = util::NullInterrupter>
class UniformPointScatter : public BasePointScatter<PointAccessorType,
                                                    RandomGenerator,
                                                    InterruptType>
{
public:
    using BaseT = BasePointScatter<PointAccessorType, RandomGenerator, InterruptType>;

    UniformPointScatter(PointAccessorType& points,
                        Index64 pointCount,
                        RandomGenerator& randGen,
                        double spread = 1.0,
                        InterruptType* interrupt = nullptr)
        : BaseT(points, randGen, spread, interrupt)
        , mTargetPointCount(pointCount)
        , mPointsPerVolume(0.0f)
    {
    }
    UniformPointScatter(PointAccessorType& points,
                        float pointsPerVolume,
                        RandomGenerator& randGen,
                        double spread = 1.0,
                        InterruptType* interrupt = nullptr)
        : BaseT(points, randGen, spread, interrupt)
        , mTargetPointCount(0)
        , mPointsPerVolume(pointsPerVolume)
    {
    }

    /// This is the main functor method implementing the actual scattering of points.
    template<typename GridT>
    bool operator()(const GridT& grid)
    {
        mVoxelCount = grid.activeVoxelCount();
        if (mVoxelCount == 0) return false;

        const auto voxelVolume = grid.transform().voxelVolume();
        if (mPointsPerVolume > 0) {
            BaseT::start("Uniform scattering with fixed point density");
            mTargetPointCount = Index64(mPointsPerVolume * voxelVolume * double(mVoxelCount));
        } else if (mTargetPointCount > 0) {
            BaseT::start("Uniform scattering with fixed point count");
            mPointsPerVolume = mTargetPointCount / float(voxelVolume * mVoxelCount);
        } else {
            return false;
        }

        std::unique_ptr<Index64[]> idList{new Index64[mTargetPointCount]};
        math::RandInt<Index64, RandomGenerator> rand(BaseT::mRand01.engine(), 0, mVoxelCount-1);
        for (Index64 i=0; i<mTargetPointCount; ++i) idList[i] = rand();
        tbb::parallel_sort(idList.get(), idList.get() + mTargetPointCount);

        CoordBBox bbox;
        const Vec3R offset(0.5, 0.5, 0.5);
        typename GridT::ValueOnCIter valueIter = grid.cbeginValueOn();

        for (Index64 i=0, n=valueIter.getVoxelCount() ; i != mTargetPointCount; ++i) {
            if (BaseT::interrupt()) return false;
            const Index64 voxelId = idList[i];
            while ( n <= voxelId ) {
                ++valueIter;
                n += valueIter.getVoxelCount();
            }
            if (valueIter.isVoxelValue()) {// a majority is expected to be voxels
                BaseT::addPoint(grid, valueIter.getCoord() - offset);
            } else {// tiles contain multiple (virtual) voxels
                valueIter.getBoundingBox(bbox);
                BaseT::addPoint(grid, bbox.min() - offset, bbox.extents());
            }
        }//loop over all the active voxels and tiles
        //}

        BaseT::end();
        return true;
    }

    // The following methods should only be called after the
    // the operator() method was called
    void print(const std::string &name, std::ostream& os = std::cout) const
    {
        os << "Uniformly scattered " << mPointCount << " points into " << mVoxelCount
           << " active voxels in \"" << name << "\" corresponding to "
           << mPointsPerVolume << " points per volume." << std::endl;
    }

    float   getPointsPerVolume()  const { return mPointsPerVolume; }
    Index64 getTargetPointCount() const { return mTargetPointCount; }

private:

    using BaseT::mPointCount;
    using BaseT::mVoxelCount;
    Index64 mTargetPointCount;
    float mPointsPerVolume;

}; // class UniformPointScatter

/// @brief Scatters a fixed (and integer) number of points in all
/// active voxels and tiles.
template<typename PointAccessorType,
         typename RandomGenerator,
         typename InterruptType = util::NullInterrupter>
class DenseUniformPointScatter : public BasePointScatter<PointAccessorType,
                                                         RandomGenerator,
                                                         InterruptType>
{
public:
    using BaseT = BasePointScatter<PointAccessorType, RandomGenerator, InterruptType>;

    DenseUniformPointScatter(PointAccessorType& points,
                             float pointsPerVoxel,
                             RandomGenerator& randGen,
                             double spread = 1.0,
                             InterruptType* interrupt = nullptr)
        : BaseT(points, randGen, spread, interrupt)
        , mPointsPerVoxel(pointsPerVoxel)
    {
    }

    /// This is the main functor method implementing the actual scattering of points.
    template<typename GridT>
    bool operator()(const GridT& grid)
    {
        using ValueIter = typename GridT::ValueOnCIter;
        if (mPointsPerVoxel < 1.0e-6) return false;
        mVoxelCount = grid.activeVoxelCount();
        if (mVoxelCount == 0) return false;
        BaseT::start("Dense uniform scattering with fixed point count");
        CoordBBox bbox;
        const Vec3R offset(0.5, 0.5, 0.5);

        const int ppv = math::Floor(mPointsPerVoxel);
        const double delta = mPointsPerVoxel - ppv;
        const bool fractional = !math::isApproxZero(delta, 1.0e-6);

        for (ValueIter iter = grid.cbeginValueOn(); iter; ++iter) {
            if (BaseT::interrupt()) return false;
            if (iter.isVoxelValue()) {// a majority is expected to be voxels
                const Vec3R dmin = iter.getCoord() - offset;
                for (int n = 0; n != ppv; ++n) BaseT::addPoint(grid, dmin);
                if (fractional && BaseT::getRand01() < delta) BaseT::addPoint(grid, dmin);
            } else {// tiles contain multiple (virtual) voxels
                iter.getBoundingBox(bbox);
                const Coord size(bbox.extents());
                const Vec3R dmin = bbox.min() - offset;
                const double d = mPointsPerVoxel * iter.getVoxelCount();
                const int m = math::Floor(d);
                for (int n = 0; n != m; ++n)  BaseT::addPoint(grid, dmin, size);
                if (BaseT::getRand01() < d - m) BaseT::addPoint(grid, dmin, size);
            }
        }//loop over all the active voxels and tiles
        //}
        BaseT::end();
        return true;
    }

    // The following methods should only be called after the
    // the operator() method was called
    void print(const std::string &name, std::ostream& os = std::cout) const
    {
        os << "Dense uniformly scattered " << mPointCount << " points into " << mVoxelCount
           << " active voxels in \"" << name << "\" corresponding to "
           << mPointsPerVoxel << " points per voxel." << std::endl;
    }

    float getPointsPerVoxel() const { return mPointsPerVoxel; }

private:
    using BaseT::mPointCount;
    using BaseT::mVoxelCount;
    float mPointsPerVoxel;
}; // class DenseUniformPointScatter

/// @brief Non-uniform scatters of point in the active voxels.
/// The local point count is implicitly defined as a product of
/// of a global density (called pointsPerVolume) and the local voxel
/// (or tile) value.
///
/// @note This scattering technique can be significantly slower
/// than a uniform scattering since its computational complexity
/// is proportional to the active voxel (and tile) count.
template<typename PointAccessorType,
         typename RandomGenerator,
         typename InterruptType = util::NullInterrupter>
class NonUniformPointScatter : public BasePointScatter<PointAccessorType,
                                                       RandomGenerator,
                                                       InterruptType>
{
public:
    using BaseT = BasePointScatter<PointAccessorType, RandomGenerator, InterruptType>;

    NonUniformPointScatter(PointAccessorType& points,
                           float pointsPerVolume,
                           RandomGenerator& randGen,
                           double spread = 1.0,
                           InterruptType* interrupt = nullptr)
        : BaseT(points, randGen, spread, interrupt)
        , mPointsPerVolume(pointsPerVolume)//note this is merely a
                                           //multiplier for the local point density
    {
    }

    /// This is the main functor method implementing the actual scattering of points.
    template<typename GridT>
    bool operator()(const GridT& grid)
    {
        if (mPointsPerVolume <= 0.0f) return false;
        mVoxelCount = grid.activeVoxelCount();
        if (mVoxelCount == 0) return false;
        BaseT::start("Non-uniform scattering with local point density");
        const Vec3d dim = grid.voxelSize();
        const double volumePerVoxel = dim[0]*dim[1]*dim[2],
                     pointsPerVoxel = mPointsPerVolume * volumePerVoxel;
        CoordBBox bbox;
        const Vec3R offset(0.5, 0.5, 0.5);
        for (typename GridT::ValueOnCIter iter = grid.cbeginValueOn(); iter; ++iter) {
            if (BaseT::interrupt()) return false;
            const double d = (*iter) * pointsPerVoxel * iter.getVoxelCount();
            const int n = int(d);
            if (iter.isVoxelValue()) { // a majority is expected to be voxels
                const Vec3R dmin =iter.getCoord() - offset;
                for (int i = 0; i < n; ++i) BaseT::addPoint(grid, dmin);
                if (BaseT::getRand01() < (d - n)) BaseT::addPoint(grid, dmin);
            } else { // tiles contain multiple (virtual) voxels
                iter.getBoundingBox(bbox);
                const Coord size(bbox.extents());
                const Vec3R dmin = bbox.min() - offset;
                for (int i = 0; i < n; ++i) BaseT::addPoint(grid, dmin, size);
                if (BaseT::getRand01() < (d - n)) BaseT::addPoint(grid, dmin, size);
            }
        }//loop over all the active voxels and tiles
        BaseT::end();
        return true;
    }

    // The following methods should only be called after the
    // the operator() method was called
    void print(const std::string &name, std::ostream& os = std::cout) const
    {
        os << "Non-uniformly scattered " << mPointCount << " points into " << mVoxelCount
           << " active voxels in \"" << name << "\"." << std::endl;
    }

    float getPointPerVolume() const { return mPointsPerVolume; }

private:
    using BaseT::mPointCount;
    using BaseT::mVoxelCount;
    float mPointsPerVolume;

}; // class NonUniformPointScatter

/// Base class of all the point scattering classes defined above
template<typename PointAccessorType,
         typename RandomGenerator,
         typename InterruptType>
class BasePointScatter
{
public:

    Index64 getPointCount() const { return mPointCount; }
    Index64 getVoxelCount() const { return mVoxelCount; }

protected:

    PointAccessorType&        mPoints;
    InterruptType*            mInterrupter;
    Index64                   mPointCount;
    Index64                   mVoxelCount;
    Index64                   mInterruptCount;
    const double              mSpread;
    math::Rand01<double, RandomGenerator> mRand01;

    /// This is a base class so the constructor is protected
    BasePointScatter(PointAccessorType& points,
                     RandomGenerator& randGen,
                     double spread,
                     InterruptType* interrupt = nullptr)
        : mPoints(points)
        , mInterrupter(interrupt)
        , mPointCount(0)
        , mVoxelCount(0)
        , mInterruptCount(0)
        , mSpread(math::Clamp01(spread))
        , mRand01(randGen)
    {
    }

    inline void start(const char* name)
    {
        if (mInterrupter) mInterrupter->start(name);
    }

    inline void end()
    {
        if (mInterrupter) mInterrupter->end();
    }

    inline bool interrupt()
    {
        //only check interrupter for every 32'th call
        return !(mInterruptCount++ & ((1<<5)-1)) && util::wasInterrupted(mInterrupter);
    }

    /// @brief Return a random floating point number between zero and one
    inline double getRand01() { return mRand01(); }

    /// @brief Return a random floating point number between 0.5 -+ mSpread/2
    inline double getRand() { return 0.5 + mSpread * (mRand01() - 0.5); }

    template <typename GridT>
    inline void addPoint(const GridT &grid, const Vec3R &dmin)
    {
        const Vec3R pos(dmin[0] + this->getRand(),
                        dmin[1] + this->getRand(),
                        dmin[2] + this->getRand());
        mPoints.add(grid.indexToWorld(pos));
        ++mPointCount;
    }

    template <typename GridT>
    inline void addPoint(const GridT &grid, const Vec3R &dmin, const Coord &size)
    {
        const Vec3R pos(dmin[0] + size[0]*this->getRand(),
                        dmin[1] + size[1]*this->getRand(),
                        dmin[2] + size[2]*this->getRand());
        mPoints.add(grid.indexToWorld(pos));
        ++mPointCount;
    }
};// class BasePointScatter

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_POINT_SCATTER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
