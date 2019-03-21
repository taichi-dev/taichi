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

/// @file     ParticleAtlas.h
///
/// @brief    Space-partitioning acceleration structure for particles, points with
///           radius. Partitions particle indices into voxels to accelerate range
///           and nearest neighbor searches.
///
/// @note     This acceleration structure only stores integer offsets into an external particle
///           data structure that conforms to the ParticleArray interface. 
///
/// @details  Constructs and maintains a sequence of @c PointIndexGrids each of progressively
///           lower resolution. Particles are uniquely assigned to a particular resolution
///           level based on their radius. This strategy has proven efficient for accelerating
///           spatial queries on particle data sets with varying radii.
///
/// @details  The data structure automatically detects and adapts to particle data sets with
///           uniform radii. The construction is simplified and spatial queries pre-cache the
///           uniform particle radius to avoid redundant access calls to the
///           ParticleArray::getRadius method.
///
/// @author   Mihai Alden

#ifndef OPENVDB_TOOLS_PARTICLE_ATLAS_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_PARTICLE_ATLAS_HAS_BEEN_INCLUDED

#include "PointIndexGrid.h"

#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tree/LeafNode.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <algorithm> // for std::min(), std::max()
#include <cmath> // for std::sqrt()
#include <deque>
#include <limits>
#include <memory>
#include <utility> // for std::pair
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////


/// @brief  Partition particles and performs range and nearest-neighbor searches.
///
/// @interface ParticleArray
/// Expected interface for the ParticleArray container:
/// @code
/// template<typename VectorType>
/// struct ParticleArray
/// {
///     // The type used to represent world-space positions
///     using PosType = VectorType;
///     using ScalarType = typename PosType::value_type;
///
///     // Return the number of particles in the array
///     size_t size() const;
///
///     // Return the world-space position for the nth particle.
///     void getPos(size_t n, PosType& xyz) const;
///
///     // Return the world-space radius for the nth particle.
///     void getRadius(size_t n, ScalarType& radius) const;
/// };
/// @endcode
///
/// @details    Constructs a collection of @c PointIndexGrids of different resolutions
///             to accelerate spatial searches for particles with varying radius.
template<typename PointIndexGridType = PointIndexGrid>
struct ParticleAtlas
{
    using Ptr = SharedPtr<ParticleAtlas>;
    using ConstPtr = SharedPtr<const ParticleAtlas>;

    using PointIndexGridPtr = typename PointIndexGridType::Ptr;
    using IndexType = typename PointIndexGridType::ValueType;

    struct Iterator;

    //////////

    ParticleAtlas() : mIndexGridArray(), mMinRadiusArray(), mMaxRadiusArray() {}

    /// @brief Partitions particle indices
    ///
    /// @param particles        container conforming to the ParticleArray interface
    /// @param minVoxelSize     minimum voxel size limit
    /// @param maxLevels        maximum number of resolution levels
    template<typename ParticleArrayType>
    void construct(const ParticleArrayType& particles, double minVoxelSize, size_t maxLevels = 50);

    /// @brief Create a new @c ParticleAtlas from the given @a particles.
    ///
    /// @param particles        container conforming to the ParticleArray interface
    /// @param minVoxelSize     minimum voxel size limit
    /// @param maxLevels        maximum number of resolution levels
    template<typename ParticleArrayType>
    static Ptr create(const ParticleArrayType& particles,
        double minVoxelSize, size_t maxLevels = 50);

    /// @brief Returns the number of resolution levels.
    size_t levels() const { return mIndexGridArray.size(); }
    /// @brief true if the container size is 0, false otherwise.
    bool empty() const { return mIndexGridArray.empty(); }

    /// @brief Returns minimum particle radius for level @a n.
    double minRadius(size_t n) const { return mMinRadiusArray[n]; }
    /// @brief Returns maximum particle radius for level @a n.
    double maxRadius(size_t n) const { return mMaxRadiusArray[n]; }

    /// @brief Returns the @c PointIndexGrid that represents the given level @a n.
    PointIndexGridType& pointIndexGrid(size_t n) { return *mIndexGridArray[n]; }
    /// @brief Returns the @c PointIndexGrid that represents the given level @a n.
    const PointIndexGridType& pointIndexGrid(size_t n) const { return *mIndexGridArray[n]; }

private:
    // Disallow copying
    ParticleAtlas(const ParticleAtlas&);
    ParticleAtlas& operator=(const ParticleAtlas&);

    std::vector<PointIndexGridPtr>  mIndexGridArray;
    std::vector<double> mMinRadiusArray, mMaxRadiusArray;
}; // struct ParticleAtlas


using ParticleIndexAtlas = ParticleAtlas<PointIndexGrid>;


////////////////////////////////////////


/// @brief Provides accelerated range and nearest-neighbor searches for
///        particles that are partitioned using the ParticleAtlas.
///
/// @note  Prefer to construct the iterator object once and reuse it
///        for subsequent queries.
template<typename PointIndexGridType>
struct ParticleAtlas<PointIndexGridType>::Iterator
{
    using TreeType = typename PointIndexGridType::TreeType;
    using ConstAccessor = tree::ValueAccessor<const TreeType>;
    using ConstAccessorPtr = std::unique_ptr<ConstAccessor>;

    /////

    /// @brief Construct an iterator from the given @a atlas.
    explicit Iterator(const ParticleAtlas& atlas);

    /// @brief Clear the iterator and update it with the result of the given
    ///        world-space radial query.
    /// @param center    world-space center
    /// @param radius    world-space search radius
    /// @param particles container conforming to the ParticleArray interface
    template<typename ParticleArrayType>
    void worldSpaceSearchAndUpdate(const Vec3d& center, double radius,
        const ParticleArrayType& particles);

    /// @brief Clear the iterator and update it with the result of the given
    ///        world-space radial query.
    /// @param bbox      world-space bounding box
    /// @param particles container conforming to the ParticleArray interface
    template<typename ParticleArrayType>
    void worldSpaceSearchAndUpdate(const BBoxd& bbox, const ParticleArrayType& particles);

    /// @brief Returns the total number of resolution levels.
    size_t levels() const { return mAtlas->levels(); }

    /// @brief Clear the iterator and update it with all particles that reside
    ///        at the given resolution @a level.
    void updateFromLevel(size_t level);

    /// Reset the iterator to point to the first item.
    void reset();

    /// Return a const reference to the item to which this iterator is pointing.
    const IndexType& operator*() const { return *mRange.first; }

    /// @{
    /// @brief  Return @c true if this iterator is not yet exhausted.
    bool test() const { return mRange.first < mRange.second || mIter != mRangeList.end(); }
    operator bool() const { return this->test(); }
    /// @}

    /// Advance iterator to next item.
    void increment();

    /// Advance iterator to next item.
    void operator++() { this->increment(); }

    /// @brief Advance iterator to next item.
    /// @return @c true if this iterator is not yet exhausted.
    bool next();

    /// Return the number of point indices in the iterator range.
    size_t size() const;

    /// Return @c true if both iterators point to the same element.
    bool operator==(const Iterator& p) const { return mRange.first == p.mRange.first; }
    bool operator!=(const Iterator& p) const { return !this->operator==(p); }

private:
    Iterator(const Iterator& rhs);
    Iterator& operator=(const Iterator& rhs);

    void clear();

    using Range = std::pair<const IndexType*, const IndexType*>;
    using RangeDeque = std::deque<Range>;
    using RangeDequeCIter = typename RangeDeque::const_iterator;
    using IndexArray = std::unique_ptr<IndexType[]>;

    ParticleAtlas const * const mAtlas;
    std::unique_ptr<ConstAccessorPtr[]> mAccessorList;

    // Primary index collection
    Range           mRange;
    RangeDeque      mRangeList;
    RangeDequeCIter mIter;
    // Secondary index collection
    IndexArray      mIndexArray;
    size_t          mIndexArraySize, mAccessorListSize;
}; // struct ParticleAtlas::Iterator


////////////////////////////////////////

// Internal operators and implementation details


namespace particle_atlas_internal {


template<typename ParticleArrayT>
struct ComputeExtremas
{
    using PosType = typename ParticleArrayT::PosType;
    using ScalarType = typename PosType::value_type;

    ComputeExtremas(const ParticleArrayT& particles)
        : particleArray(&particles)
        , minRadius(std::numeric_limits<ScalarType>::max())
        , maxRadius(-std::numeric_limits<ScalarType>::max())
    {
    }

    ComputeExtremas(ComputeExtremas& rhs, tbb::split)
        : particleArray(rhs.particleArray)
        , minRadius(std::numeric_limits<ScalarType>::max())
        , maxRadius(-std::numeric_limits<ScalarType>::max())
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) {

        ScalarType radius, tmpMin = minRadius, tmpMax = maxRadius;

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            particleArray->getRadius(n, radius);
            tmpMin = std::min(radius, tmpMin);
            tmpMax = std::max(radius, tmpMax);
        }

        minRadius = std::min(minRadius, tmpMin);
        maxRadius = std::max(maxRadius, tmpMax);
    }

    void join(const ComputeExtremas& rhs) {
        minRadius = std::min(minRadius, rhs.minRadius);
        maxRadius = std::max(maxRadius, rhs.maxRadius);
    }

    ParticleArrayT const * const particleArray;
    ScalarType minRadius, maxRadius;
}; // struct ComputeExtremas


template<typename ParticleArrayT, typename PointIndex>
struct SplittableParticleArray
{
    using Ptr = SharedPtr<SplittableParticleArray>;
    using ConstPtr = SharedPtr<const SplittableParticleArray>;
    using ParticleArray = ParticleArrayT;

    using PosType = typename ParticleArray::PosType;
    using ScalarType = typename PosType::value_type;

    SplittableParticleArray(const ParticleArrayT& particles)
        : mIndexMap(), mParticleArray(&particles), mSize(particles.size())
    {
        updateExtremas();
    }

    SplittableParticleArray(const ParticleArrayT& particles, double minR, double maxR)
        : mIndexMap(), mParticleArray(&particles), mSize(particles.size())
    {
        mMinRadius = ScalarType(minR);
        mMaxRadius = ScalarType(maxR);
    }

    const ParticleArrayT& particleArray() const { return *mParticleArray; }

    size_t size() const { return mSize; }

    void getPos(size_t n, PosType& xyz) const
        { return mParticleArray->getPos(getGlobalIndex(n), xyz); }
    void getRadius(size_t n, ScalarType& radius) const
        { return mParticleArray->getRadius(getGlobalIndex(n), radius); }

    ScalarType minRadius() const { return mMinRadius; }
    ScalarType maxRadius() const { return mMaxRadius; }

    size_t getGlobalIndex(size_t n) const { return mIndexMap ? size_t(mIndexMap[n]) : n; }

    /// Move all particle indices that have a radius larger or equal to @a maxRadiusLimit
    /// into a separate container.
    Ptr split(ScalarType maxRadiusLimit) {

        if (mMaxRadius < maxRadiusLimit) return Ptr();

        std::unique_ptr<bool[]> mask{new bool[mSize]};

        tbb::parallel_for(tbb::blocked_range<size_t>(0, mSize),
            MaskParticles(*this, mask, maxRadiusLimit));

        Ptr output(new SplittableParticleArray(*this, mask));
        if (output->size() == 0) return Ptr();

        size_t newSize = 0;
        for (size_t n = 0, N = mSize; n < N; ++n) {
            newSize += size_t(!mask[n]);
        }

        std::unique_ptr<PointIndex[]> newIndexMap{new PointIndex[newSize]};

        setIndexMap(newIndexMap, mask, false);

        mSize = newSize;
        mIndexMap.swap(newIndexMap);
        updateExtremas();

        return output;
    }


private:
    // Disallow copying
    SplittableParticleArray(const SplittableParticleArray&);
    SplittableParticleArray& operator=(const SplittableParticleArray&);

    // Masked copy constructor
    SplittableParticleArray(const SplittableParticleArray& other,
        const std::unique_ptr<bool[]>& mask)
        : mIndexMap(), mParticleArray(&other.particleArray()), mSize(0)
    {
        for (size_t n = 0, N = other.size(); n < N; ++n) {
            mSize += size_t(mask[n]);
        }

        if (mSize != 0) {
            mIndexMap.reset(new PointIndex[mSize]);
            other.setIndexMap(mIndexMap, mask, true);
        }

        updateExtremas();
    }

    struct MaskParticles {
        MaskParticles(const SplittableParticleArray& particles,
            const std::unique_ptr<bool[]>& mask, ScalarType radius)
            : particleArray(&particles)
            , particleMask(mask.get())
            , radiusLimit(radius)
        {
        }

        void operator()(const tbb::blocked_range<size_t>& range) const {
            const ScalarType maxRadius = radiusLimit;
            ScalarType radius;
            for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
                particleArray->getRadius(n, radius);
                particleMask[n] = !(radius < maxRadius);
            }
        }

        SplittableParticleArray const * const particleArray;
        bool                          * const particleMask;
        ScalarType                      const radiusLimit;
    }; // struct MaskParticles

    inline void updateExtremas() {
        ComputeExtremas<SplittableParticleArray> op(*this);
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, mSize), op);
        mMinRadius = op.minRadius;
        mMaxRadius = op.maxRadius;
    }

    void setIndexMap(std::unique_ptr<PointIndex[]>& newIndexMap,
        const std::unique_ptr<bool[]>& mask, bool maskValue) const
    {
        if (mIndexMap.get() != nullptr) {
                const PointIndex* indices = mIndexMap.get();
            for (size_t idx = 0, n = 0, N = mSize; n < N; ++n) {
                if (mask[n] == maskValue) newIndexMap[idx++] = indices[n];
            }
        } else {
            for (size_t idx = 0, n = 0, N = mSize; n < N; ++n) {
                if (mask[n] == maskValue) {
                    newIndexMap[idx++] = PointIndex(static_cast<typename PointIndex::IntType>(n));
                }
            }
        }
    }


    //////////

    std::unique_ptr<PointIndex[]> mIndexMap;
    ParticleArrayT const * const    mParticleArray;
    size_t                          mSize;
    ScalarType                      mMinRadius, mMaxRadius;
}; // struct SplittableParticleArray


template<typename ParticleArrayType, typename PointIndexLeafNodeType>
struct RemapIndices {

    RemapIndices(const ParticleArrayType& particles, std::vector<PointIndexLeafNodeType*>& nodes)
        : mParticles(&particles)
        , mNodes(nodes.empty() ? nullptr : &nodes.front())
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        using PointIndexType = typename PointIndexLeafNodeType::ValueType;
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            PointIndexLeafNodeType& node = *mNodes[n];
            const size_t numIndices = node.indices().size();

            if (numIndices > 0) {
                PointIndexType* begin = &node.indices().front();
                const PointIndexType* end = begin + numIndices;

                while (begin < end) {
                    *begin = PointIndexType(static_cast<typename PointIndexType::IntType>(
                        mParticles->getGlobalIndex(*begin)));
                    ++begin;
                }
            }
        }
    }

    ParticleArrayType         const * const mParticles;
    PointIndexLeafNodeType  * const * const mNodes;
}; // struct RemapIndices


template<typename ParticleArrayType, typename IndexT>
struct RadialRangeFilter
{
    using PosType = typename ParticleArrayType::PosType;
    using ScalarType = typename PosType::value_type;

    using Range = std::pair<const IndexT*, const IndexT*>;
    using RangeDeque = std::deque<Range>;
    using IndexDeque = std::deque<IndexT>;

    RadialRangeFilter(RangeDeque& ranges, IndexDeque& indices, const PosType& xyz,
        ScalarType radius, const ParticleArrayType& particles, bool hasUniformRadius = false)
        : mRanges(ranges)
        , mIndices(indices)
        , mCenter(xyz)
        , mRadius(radius)
        , mParticles(&particles)
        , mHasUniformRadius(hasUniformRadius)
    {
        if (mHasUniformRadius) {
            ScalarType uniformRadius;
            mParticles->getRadius(0, uniformRadius);
            mRadius = mRadius + uniformRadius;
            mRadius *= mRadius;
        }
    }

    template <typename LeafNodeType>
    void filterLeafNode(const LeafNodeType& leaf)
    {
        const size_t numIndices = leaf.indices().size();
        if (numIndices > 0) {
            const IndexT* begin = &leaf.indices().front();
            filterVoxel(leaf.origin(), begin, begin + numIndices);
        }
    }

    void filterVoxel(const Coord&, const IndexT* begin, const IndexT* end)
    {
        PosType pos;

        if (mHasUniformRadius) {

            const ScalarType searchRadiusSqr = mRadius;

            while (begin < end) {
                mParticles->getPos(size_t(*begin), pos);
                const ScalarType distSqr = (mCenter - pos).lengthSqr();
                if (distSqr < searchRadiusSqr) {
                    mIndices.push_back(*begin);
                }
                ++begin;
            }
        } else {
            while (begin < end) {
                const size_t idx = size_t(*begin);
                mParticles->getPos(idx, pos);

                ScalarType radius;
                mParticles->getRadius(idx, radius);

                ScalarType searchRadiusSqr = mRadius + radius;
                searchRadiusSqr *= searchRadiusSqr;

                const ScalarType distSqr = (mCenter - pos).lengthSqr();

                if (distSqr < searchRadiusSqr) {
                    mIndices.push_back(*begin);
                }

                ++begin;
            }
        }
    }

private:
    RadialRangeFilter(const RadialRangeFilter&);
    RadialRangeFilter& operator=(const RadialRangeFilter&);

    RangeDeque&                     mRanges;
    IndexDeque&                     mIndices;
    PosType                   const mCenter;
    ScalarType                      mRadius;
    ParticleArrayType const * const mParticles;
    bool                      const mHasUniformRadius;
}; // struct RadialRangeFilter


template<typename ParticleArrayType, typename IndexT>
struct BBoxFilter
{
    using PosType = typename ParticleArrayType::PosType;
    using ScalarType = typename PosType::value_type;

    using Range = std::pair<const IndexT*, const IndexT*>;
    using RangeDeque = std::deque<Range>;
    using IndexDeque = std::deque<IndexT>;

    BBoxFilter(RangeDeque& ranges, IndexDeque& indices,
        const BBoxd& bbox, const ParticleArrayType& particles, bool hasUniformRadius = false)
        : mRanges(ranges)
        , mIndices(indices)
        , mBBox(PosType(bbox.min()), PosType(bbox.max()))
        , mCenter(mBBox.getCenter())
        , mParticles(&particles)
        , mHasUniformRadius(hasUniformRadius)
        , mUniformRadiusSqr(ScalarType(0.0))
    {
        if (mHasUniformRadius) {
            mParticles->getRadius(0, mUniformRadiusSqr);
            mUniformRadiusSqr *= mUniformRadiusSqr;
        }
    }

    template <typename LeafNodeType>
    void filterLeafNode(const LeafNodeType& leaf)
    {
        const size_t numIndices = leaf.indices().size();
        if (numIndices > 0) {
            const IndexT* begin = &leaf.indices().front();
            filterVoxel(leaf.origin(), begin, begin + numIndices);
        }
    }

    void filterVoxel(const Coord&, const IndexT* begin, const IndexT* end)
    {
        PosType pos;

        if (mHasUniformRadius) {
            const ScalarType radiusSqr = mUniformRadiusSqr;

            while (begin < end) {

                mParticles->getPos(size_t(*begin), pos);
                if (mBBox.isInside(pos)) {
                    mIndices.push_back(*begin++);
                    continue;
                }

                const ScalarType distSqr = pointToBBoxDistSqr(pos);
                if (!(distSqr > radiusSqr)) {
                    mIndices.push_back(*begin);
                }

                ++begin;
            }

        } else {
            while (begin < end) {

                const size_t idx = size_t(*begin);
                mParticles->getPos(idx, pos);
                if (mBBox.isInside(pos)) {
                    mIndices.push_back(*begin++);
                    continue;
                }

                ScalarType radius;
                mParticles->getRadius(idx, radius);
                const ScalarType distSqr = pointToBBoxDistSqr(pos);
                if (!(distSqr > (radius * radius))) {
                    mIndices.push_back(*begin);
                }

                ++begin;
            }
        }
    }

private:
    BBoxFilter(const BBoxFilter&);
    BBoxFilter& operator=(const BBoxFilter&);

    ScalarType pointToBBoxDistSqr(const PosType& pos) const
    {
        ScalarType distSqr = ScalarType(0.0);

        for (int i = 0; i < 3; ++i) {

            const ScalarType a = pos[i];

            ScalarType b = mBBox.min()[i];
            if (a < b) {
                ScalarType delta = b - a;
                distSqr += delta * delta;
            }

            b = mBBox.max()[i];
            if (a > b) {
                ScalarType delta = a - b;
                distSqr += delta * delta;
            }
        }

        return distSqr;
    }

    RangeDeque&                     mRanges;
    IndexDeque&                     mIndices;
    math::BBox<PosType>       const mBBox;
    PosType                   const mCenter;
    ParticleArrayType const * const mParticles;
    bool                      const mHasUniformRadius;
    ScalarType                      mUniformRadiusSqr;
}; // struct BBoxFilter


} // namespace particle_atlas_internal


////////////////////////////////////////


template<typename PointIndexGridType>
template<typename ParticleArrayType>
inline void
ParticleAtlas<PointIndexGridType>::construct(
    const ParticleArrayType& particles, double minVoxelSize, size_t maxLevels)
{
    using SplittableParticleArray =
        typename particle_atlas_internal::SplittableParticleArray<ParticleArrayType, IndexType>;
    using SplittableParticleArrayPtr = typename SplittableParticleArray::Ptr;
    using ScalarType = typename ParticleArrayType::ScalarType;

    /////

    particle_atlas_internal::ComputeExtremas<ParticleArrayType> extremas(particles);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, particles.size()), extremas);
    const double firstMin = extremas.minRadius;
    const double firstMax = extremas.maxRadius;
    const double firstVoxelSize = std::max(minVoxelSize, firstMin);

    if (!(firstMax < (firstVoxelSize * double(2.0))) && maxLevels > 1) {

        std::vector<SplittableParticleArrayPtr> levels;
        levels.push_back(SplittableParticleArrayPtr(
                new SplittableParticleArray(particles, firstMin, firstMax)));

        std::vector<double> voxelSizeArray;
        voxelSizeArray.push_back(firstVoxelSize);

        for (size_t n = 0; n < maxLevels; ++n) {

            const double maxParticleRadius = double(levels.back()->maxRadius());
            const double particleRadiusLimit = voxelSizeArray.back() * double(2.0);
            if (maxParticleRadius < particleRadiusLimit) break;

            SplittableParticleArrayPtr newLevel =
                levels.back()->split(ScalarType(particleRadiusLimit));
            if (!newLevel) break;

            levels.push_back(newLevel);
            voxelSizeArray.push_back(double(newLevel->minRadius()));
        }

        size_t numPoints = 0;

        using PointIndexTreeType = typename PointIndexGridType::TreeType;
        using PointIndexLeafNodeType = typename PointIndexTreeType::LeafNodeType;

        std::vector<PointIndexLeafNodeType*> nodes;

        for (size_t n = 0, N = levels.size(); n < N; ++n) {

            const SplittableParticleArray& particleArray = *levels[n];

            numPoints += particleArray.size();

            mMinRadiusArray.push_back(double(particleArray.minRadius()));
            mMaxRadiusArray.push_back(double(particleArray.maxRadius()));

            PointIndexGridPtr grid =
                createPointIndexGrid<PointIndexGridType>(particleArray, voxelSizeArray[n]);

            nodes.clear();
            grid->tree().getNodes(nodes);

            tbb::parallel_for(tbb::blocked_range<size_t>(0, nodes.size()),
                particle_atlas_internal::RemapIndices<SplittableParticleArray,
                    PointIndexLeafNodeType>(particleArray, nodes));

            mIndexGridArray.push_back(grid);
        }

    } else {
        mMinRadiusArray.push_back(firstMin);
        mMaxRadiusArray.push_back(firstMax);
        mIndexGridArray.push_back(
            createPointIndexGrid<PointIndexGridType>(particles, firstVoxelSize));
    }
}


template<typename PointIndexGridType>
template<typename ParticleArrayType>
inline typename ParticleAtlas<PointIndexGridType>::Ptr
ParticleAtlas<PointIndexGridType>::create(
    const ParticleArrayType& particles, double minVoxelSize, size_t maxLevels)
{
    Ptr ret(new ParticleAtlas());
    ret->construct(particles, minVoxelSize, maxLevels);
    return ret;
}


////////////////////////////////////////

// ParticleAtlas::Iterator implementation

template<typename PointIndexGridType>
inline
ParticleAtlas<PointIndexGridType>::Iterator::Iterator(const ParticleAtlas& atlas)
    : mAtlas(&atlas)
    , mAccessorList()
    , mRange(static_cast<IndexType*>(nullptr), static_cast<IndexType*>(nullptr))
    , mRangeList()
    , mIter(mRangeList.begin())
    , mIndexArray()
    , mIndexArraySize(0)
    , mAccessorListSize(atlas.levels())
{
    if (mAccessorListSize > 0) {
        mAccessorList.reset(new ConstAccessorPtr[mAccessorListSize]);
        for (size_t n = 0, N = mAccessorListSize; n < N; ++n) {
            mAccessorList[n].reset(new ConstAccessor(atlas.pointIndexGrid(n).tree()));
        }
    }
}


template<typename PointIndexGridType>
inline void
ParticleAtlas<PointIndexGridType>::Iterator::reset()
{
    mIter = mRangeList.begin();
    if (!mRangeList.empty()) {
        mRange = mRangeList.front();
    } else if (mIndexArray) {
        mRange.first = mIndexArray.get();
        mRange.second = mRange.first + mIndexArraySize;
    } else {
        mRange.first = static_cast<IndexType*>(nullptr);
        mRange.second = static_cast<IndexType*>(nullptr);
    }
}


template<typename PointIndexGridType>
inline void
ParticleAtlas<PointIndexGridType>::Iterator::increment()
{
    ++mRange.first;
    if (mRange.first >= mRange.second && mIter != mRangeList.end()) {
        ++mIter;
        if (mIter != mRangeList.end()) {
            mRange = *mIter;
        } else if (mIndexArray) {
            mRange.first = mIndexArray.get();
            mRange.second = mRange.first + mIndexArraySize;
        }
    }
}


template<typename PointIndexGridType>
inline bool
ParticleAtlas<PointIndexGridType>::Iterator::next()
{
    if (!this->test()) return false;
    this->increment();
    return this->test();
}


template<typename PointIndexGridType>
inline size_t
ParticleAtlas<PointIndexGridType>::Iterator::size() const
{
    size_t count = 0;
    typename RangeDeque::const_iterator it =
        mRangeList.begin(), end = mRangeList.end();

    for ( ; it != end; ++it) {
        count += it->second - it->first;
    }

    return count + mIndexArraySize;
}


template<typename PointIndexGridType>
inline void
ParticleAtlas<PointIndexGridType>::Iterator::clear()
{
    mRange.first = static_cast<IndexType*>(nullptr);
    mRange.second = static_cast<IndexType*>(nullptr);
    mRangeList.clear();
    mIter = mRangeList.end();
    mIndexArray.reset();
    mIndexArraySize = 0;
}


template<typename PointIndexGridType>
inline void
ParticleAtlas<PointIndexGridType>::Iterator::updateFromLevel(size_t level)
{
    using TreeType = typename PointIndexGridType::TreeType;
    using LeafNodeType = typename TreeType::LeafNodeType;

    this->clear();

    if (mAccessorListSize > 0) {
        const size_t levelIdx = std::min(mAccessorListSize - 1, level);

        const TreeType& tree = mAtlas->pointIndexGrid(levelIdx).tree();


        std::vector<const LeafNodeType*> nodes;
        tree.getNodes(nodes);

        for (size_t n = 0, N = nodes.size(); n < N; ++n) {

            const LeafNodeType& node = *nodes[n];
            const size_t numIndices = node.indices().size();

            if (numIndices > 0) {
                const IndexType* begin = &node.indices().front();
                mRangeList.push_back(Range(begin, (begin + numIndices)));
            }
        }
    }

    this->reset();
}


template<typename PointIndexGridType>
template<typename ParticleArrayType>
inline void
ParticleAtlas<PointIndexGridType>::Iterator::worldSpaceSearchAndUpdate(
    const Vec3d& center, double radius, const ParticleArrayType& particles)
{
    using PosType = typename ParticleArrayType::PosType;
    using ScalarType = typename ParticleArrayType::ScalarType;

    /////

    this->clear();

    std::deque<IndexType> filteredIndices;
    std::vector<CoordBBox> searchRegions;

    const double iRadius = radius * double(1.0 / std::sqrt(3.0));

    const Vec3d ibMin(center[0] - iRadius, center[1] - iRadius, center[2] - iRadius);
    const Vec3d ibMax(center[0] + iRadius, center[1] + iRadius, center[2] + iRadius);

    const Vec3d bMin(center[0] - radius, center[1] - radius, center[2] - radius);
    const Vec3d bMax(center[0] + radius, center[1] + radius, center[2] + radius);

    const PosType pos = PosType(center);
    const ScalarType dist = ScalarType(radius);

    for (size_t n = 0, N = mAccessorListSize; n < N; ++n) {

        const double maxRadius = mAtlas->maxRadius(n);
        const bool uniformRadius = math::isApproxEqual(mAtlas->minRadius(n), maxRadius);

        const openvdb::math::Transform& xform = mAtlas->pointIndexGrid(n).transform();

        ConstAccessor& acc = *mAccessorList[n];

        openvdb::CoordBBox inscribedRegion(
            xform.worldToIndexCellCentered(ibMin),
            xform.worldToIndexCellCentered(ibMax));

        inscribedRegion.expand(-1); // erode by one voxel

        // collect indices that don't need to be tested
        point_index_grid_internal::pointIndexSearch(mRangeList, acc, inscribedRegion);

        searchRegions.clear();

        const openvdb::CoordBBox region(
            xform.worldToIndexCellCentered(bMin - maxRadius),
            xform.worldToIndexCellCentered(bMax + maxRadius));

        inscribedRegion.expand(1);
        point_index_grid_internal::constructExclusiveRegions(
            searchRegions, region, inscribedRegion);

        using FilterType = particle_atlas_internal::RadialRangeFilter<ParticleArrayType, IndexType>;
        FilterType filter(mRangeList, filteredIndices, pos, dist, particles, uniformRadius);

        for (size_t i = 0, I = searchRegions.size(); i < I; ++i) {
            point_index_grid_internal::filteredPointIndexSearch(filter, acc, searchRegions[i]);
        }
    }

    point_index_grid_internal::dequeToArray(filteredIndices, mIndexArray, mIndexArraySize);

    this->reset();
}


template<typename PointIndexGridType>
template<typename ParticleArrayType>
inline void
ParticleAtlas<PointIndexGridType>::Iterator::worldSpaceSearchAndUpdate(
    const BBoxd& bbox, const ParticleArrayType& particles)
{
    this->clear();

    std::deque<IndexType> filteredIndices;
    std::vector<CoordBBox> searchRegions;

    for (size_t n = 0, N = mAccessorListSize; n < N; ++n) {

        const double maxRadius = mAtlas->maxRadius(n);
        const bool uniformRadius = math::isApproxEqual(mAtlas->minRadius(n), maxRadius);
        const openvdb::math::Transform& xform = mAtlas->pointIndexGrid(n).transform();

        ConstAccessor& acc = *mAccessorList[n];

        openvdb::CoordBBox inscribedRegion(
            xform.worldToIndexCellCentered(bbox.min()),
            xform.worldToIndexCellCentered(bbox.max()));

        inscribedRegion.expand(-1); // erode by one voxel

        // collect indices that don't need to be tested
        point_index_grid_internal::pointIndexSearch(mRangeList, acc, inscribedRegion);

        searchRegions.clear();

        const openvdb::CoordBBox region(
            xform.worldToIndexCellCentered(bbox.min() - maxRadius),
            xform.worldToIndexCellCentered(bbox.max() + maxRadius));

        inscribedRegion.expand(1);
        point_index_grid_internal::constructExclusiveRegions(
            searchRegions, region, inscribedRegion);

        using FilterType = particle_atlas_internal::BBoxFilter<ParticleArrayType, IndexType>;
        FilterType filter(mRangeList, filteredIndices, bbox, particles, uniformRadius);

        for (size_t i = 0, I = searchRegions.size(); i < I; ++i) {
            point_index_grid_internal::filteredPointIndexSearch(filter, acc, searchRegions[i]);
        }
    }

    point_index_grid_internal::dequeToArray(filteredIndices, mIndexArray, mIndexArraySize);

    this->reset();
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_PARTICLE_ATLAS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
