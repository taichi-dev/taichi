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

/// @file points/IndexFilter.h
///
/// @author Dan Bailey
///
/// @brief  Index filters primarily designed to be used with a FilterIndexIter.
///
/// Filters must adhere to the interface described in the example below:
/// @code
/// struct MyFilter
/// {
///     // Return true when the filter has been initialized for first use
///     bool initialized() { return true; }
///
///     // Return index::ALL if all points are valid, index::NONE if no points are valid
///     // and index::PARTIAL if some points are valid
///     index::State state() { return index::PARTIAL; }
///
///     // Return index::ALL if all points in this leaf are valid, index::NONE if no points
///     // in this leaf are valid and index::PARTIAL if some points in this leaf are valid
///     template <typename LeafT>
///     index::State state(const LeafT&) { return index::PARTIAL; }
///
///     // Resets the filter to refer to the specified leaf, all subsequent valid() calls
///     // will be relative to this leaf until reset() is called with a different leaf.
///     // Although a required method, many filters will provide an empty implementation if
///     // there is no leaf-specific logic needed.
///     template <typename LeafT> void reset(const LeafT&) { }
///
///     // Returns true if the filter is valid for the supplied iterator
///     template <typename IterT> bool valid(const IterT&) { return true; }
/// };
/// @endcode

#ifndef OPENVDB_POINTS_INDEX_FILTER_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_INDEX_FILTER_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Types.h>

#include <openvdb/math/Transform.h>
#include <openvdb/tools/Interpolation.h>

#include "IndexIterator.h"
#include "AttributeArray.h"
#include "AttributeGroup.h"
#include "AttributeSet.h"

#include <boost/ptr_container/ptr_vector.hpp>

#include <random> // std::mt19937
#include <numeric> // std::iota


class TestIndexFilter;

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


////////////////////////////////////////



namespace index_filter_internal {


// generate a random subset of n indices from the range [0:m]
template <typename RandGenT, typename IntType>
std::vector<IntType>
generateRandomSubset(const unsigned int seed, const IntType n, const IntType m)
{
    if (n <= 0)     return std::vector<IntType>();

    // fill vector with ascending indices
    std::vector<IntType> values(m);
    std::iota(values.begin(), values.end(), 0);
    if (n >= m) return values;

    // shuffle indices using random generator

    RandGenT randGen(seed);
    std::shuffle(values.begin(), values.end(), randGen);

    // resize the container to n elements
    values.resize(n);

    // sort the subset of the indices vector that will be used
    std::sort(values.begin(), values.end());

    return values;
}


} // namespace index_filter_internal


/// Index filtering on active / inactive state of host voxel
template <bool On>
class ValueMaskFilter
{
public:
    static bool initialized() { return true; }
    static index::State state() { return index::PARTIAL; }
    template <typename LeafT>
    static index::State state(const LeafT& leaf)
    {
        if (leaf.isDense())         return On ? index::ALL : index::NONE;
        else if (leaf.isEmpty())    return On ? index::NONE : index::ALL;
        return index::PARTIAL;
    }

    template <typename LeafT>
    void reset(const LeafT&) { }

    template <typename IterT>
    bool valid(const IterT& iter) const
    {
        const bool valueOn = iter.isValueOn();
        return On ? valueOn : !valueOn;
    }
};


using ActiveFilter = ValueMaskFilter<true>;
using InactiveFilter = ValueMaskFilter<false>;


/// Index filtering on multiple group membership for inclusion and exclusion
///
/// @note include filters are applied first, then exclude filters
class MultiGroupFilter
{
public:
    using NameVector    = std::vector<Name>;
    using IndexVector   = std::vector<AttributeSet::Descriptor::GroupIndex>;
    using HandleVector  = std::vector<GroupHandle>;

private:
    static IndexVector namesToIndices(const AttributeSet& attributeSet, const NameVector& names) {
        IndexVector indices;
        for (const auto& name : names) {
            try {
                indices.emplace_back(attributeSet.groupIndex(name));
            } catch (LookupError&) {
                // silently drop group names that don't exist
            }
        }
        return indices;
    }

public:
    MultiGroupFilter(   const NameVector& include,
                        const NameVector& exclude,
                        const AttributeSet& attributeSet)
        : mInclude(MultiGroupFilter::namesToIndices(attributeSet, include))
        , mExclude(MultiGroupFilter::namesToIndices(attributeSet, exclude)) { }

    MultiGroupFilter(   const IndexVector& include,
                        const IndexVector& exclude)
        : mInclude(include)
        , mExclude(exclude) { }

    MultiGroupFilter(   const MultiGroupFilter& filter)
        : mInclude(filter.mInclude)
        , mExclude(filter.mExclude)
        , mIncludeHandles(filter.mIncludeHandles)
        , mExcludeHandles(filter.mExcludeHandles)
        , mInitialized(filter.mInitialized) { }

    inline bool initialized() const { return mInitialized; }

    inline index::State state() const
    {
        return (mInclude.empty() && mExclude.empty()) ? index::ALL : index::PARTIAL;
    }

    template <typename LeafT>
    static index::State state(const LeafT&) { return index::PARTIAL; }

    template <typename LeafT>
    void reset(const LeafT& leaf) {
        mIncludeHandles.clear();
        mExcludeHandles.clear();
        for (const auto& i : mInclude) {
            mIncludeHandles.emplace_back(leaf.groupHandle(i));
        }
        for (const auto& i : mExclude) {
            mExcludeHandles.emplace_back(leaf.groupHandle(i));
        }
        mInitialized = true;
    }

    template <typename IterT>
    bool valid(const IterT& iter) const {
        assert(mInitialized);
        // accept no include filters as valid
        bool includeValid = mIncludeHandles.empty();
        for (const GroupHandle& handle : mIncludeHandles) {
            if (handle.getUnsafe(*iter)) {
                includeValid = true;
                break;
            }
        }
        if (!includeValid)          return false;
        for (const GroupHandle& handle : mExcludeHandles) {
            if (handle.getUnsafe(*iter))  return false;
        }
        return true;
    }

private:
    IndexVector mInclude;
    IndexVector mExclude;
    HandleVector mIncludeHandles;
    HandleVector mExcludeHandles;
    bool mInitialized = false;
}; // class MultiGroupFilter


// Random index filtering per leaf
template <typename PointDataTreeT, typename RandGenT>
class RandomLeafFilter
{
public:
    using SeedCountPair = std::pair<Index, Index>;
    using LeafMap       = std::map<openvdb::Coord, SeedCountPair>;

    RandomLeafFilter(   const PointDataTreeT& tree,
                        const Index64 targetPoints,
                        const unsigned int seed = 0) {
        Index64 currentPoints = 0;
        for (auto iter = tree.cbeginLeaf(); iter; ++iter) {
            currentPoints += iter->pointCount();
        }

        const float factor = targetPoints > currentPoints ? 1.0f : float(targetPoints) / float(currentPoints);

        std::mt19937 generator(seed);
        std::uniform_int_distribution<unsigned int> dist(0, std::numeric_limits<unsigned int>::max() - 1);

        Index32 leafCounter = 0;
        float totalPointsFloat = 0.0f;
        int totalPoints = 0;
        for (auto iter = tree.cbeginLeaf(); iter; ++iter) {
            // for the last leaf - use the remaining points to reach the target points
            if (leafCounter + 1 == tree.leafCount()) {
                const int leafPoints = static_cast<int>(targetPoints) - totalPoints;
                mLeafMap[iter->origin()] = SeedCountPair(dist(generator), leafPoints);
                break;
            }
            totalPointsFloat += factor * iter->pointCount();
            const auto leafPoints = static_cast<int>(math::Floor(totalPointsFloat));
            totalPointsFloat -= static_cast<float>(leafPoints);
            totalPoints += leafPoints;

            mLeafMap[iter->origin()] = SeedCountPair(dist(generator), leafPoints);

            leafCounter++;
        }
    }

    inline bool initialized() const { return mNextIndex == -1; }

    static index::State state() { return index::PARTIAL; }
    template <typename LeafT>
    static index::State state(const LeafT&) { return index::PARTIAL; }

    template <typename LeafT>
    void reset(const LeafT& leaf) {
        using index_filter_internal::generateRandomSubset;

        auto it = mLeafMap.find(leaf.origin());
        if (it == mLeafMap.end()) {
            OPENVDB_THROW(openvdb::KeyError,
                "Cannot find leaf origin in map for random filter - " << leaf.origin());
        }

        const SeedCountPair& value = it->second;
        const unsigned int seed = static_cast<unsigned int>(value.first);
        const auto total = static_cast<Index>(leaf.pointCount());
        mCount = std::min(value.second, total);

        mIndices = generateRandomSubset<RandGenT, int>(seed, mCount, total);

        mSubsetOffset = -1;
        mNextIndex = -1;
    }

    inline void next() const {
        mSubsetOffset++;
        mNextIndex =    mSubsetOffset >= mCount ?
                        std::numeric_limits<int>::max() :
                        mIndices[mSubsetOffset];
    }

    template <typename IterT>
    bool valid(const IterT& iter) const {
        const int index = *iter;
        while (mNextIndex < index)     this->next();
        return mNextIndex == index;
    }

protected:
    friend class ::TestIndexFilter;

private:
    LeafMap mLeafMap;
    std::vector<int> mIndices;
    int mCount = 0;
    mutable int mSubsetOffset = -1;
    mutable int mNextIndex = -1;
}; // class RandomLeafFilter


// Hash attribute value for deterministic, but approximate filtering
template <typename RandGenT, typename IntType>
class AttributeHashFilter
{
public:
    using Handle = AttributeHandle<IntType>;

    AttributeHashFilter(const size_t index,
                        const double percentage,
                        const unsigned int seed = 0)
        : mIndex(index)
        , mFactor(percentage / 100.0)
        , mSeed(seed) { }

    AttributeHashFilter(const AttributeHashFilter& filter)
        : mIndex(filter.mIndex)
        , mFactor(filter.mFactor)
        , mSeed(filter.mSeed)
    {
        if (filter.mIdHandle)   mIdHandle.reset(new Handle(*filter.mIdHandle));
    }

    inline bool initialized() const { return bool(mIdHandle); }

    static index::State state() { return index::PARTIAL; }
    template <typename LeafT>
    static index::State state(const LeafT&) { return index::PARTIAL; }

    template <typename LeafT>
    void reset(const LeafT& leaf) {
        assert(leaf.hasAttribute(mIndex));
        mIdHandle.reset(new Handle(leaf.constAttributeArray(mIndex)));
    }

    template <typename IterT>
    bool valid(const IterT& iter) const {
        assert(mIdHandle);
        const IntType id = mIdHandle->get(*iter);
        const unsigned int seed = mSeed + static_cast<unsigned int>(id);
        RandGenT generator(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(generator) < mFactor;
    }

private:
    const size_t mIndex;
    const double mFactor;
    const unsigned int mSeed;
    typename Handle::UniquePtr mIdHandle;
}; // class AttributeHashFilter


template <typename LevelSetGridT>
class LevelSetFilter
{
public:
    using ValueT = typename LevelSetGridT::ValueType;
    using Handle = AttributeHandle<openvdb::Vec3f>;

    LevelSetFilter( const LevelSetGridT& grid,
                    const math::Transform& transform,
                    const ValueT min,
                    const ValueT max)
        : mAccessor(grid.getConstAccessor())
        , mLevelSetTransform(grid.transform())
        , mTransform(transform)
        , mMin(min)
        , mMax(max) { }

    LevelSetFilter(const LevelSetFilter& filter)
        : mAccessor(filter.mAccessor)
        , mLevelSetTransform(filter.mLevelSetTransform)
        , mTransform(filter.mTransform)
        , mMin(filter.mMin)
        , mMax(filter.mMax)
    {
        if (filter.mPositionHandle)    mPositionHandle.reset(new Handle(*filter.mPositionHandle));
    }

    inline bool initialized() const { return bool(mPositionHandle); }

    static index::State state() { return index::PARTIAL; }
    template <typename LeafT>
    static index::State state(const LeafT&) { return index::PARTIAL; }

    template <typename LeafT>
    void reset(const LeafT& leaf) {
        mPositionHandle.reset(new Handle(leaf.constAttributeArray("P")));
    }

    template <typename IterT>
    bool valid(const IterT& iter) const {
        assert(mPositionHandle);
        assert(iter);

        const openvdb::Coord ijk = iter.getCoord();
        const openvdb::Vec3f voxelIndexSpace = ijk.asVec3d();

        // Retrieve point position in voxel space
        const openvdb::Vec3f& pointVoxelSpace = mPositionHandle->get(*iter);

        // Compute point position in index space
        const openvdb::Vec3f pointWorldSpace = mTransform.indexToWorld(pointVoxelSpace + voxelIndexSpace);
        const openvdb::Vec3f pointIndexSpace = mLevelSetTransform.worldToIndex(pointWorldSpace);

        // Perform level-set sampling
        const typename LevelSetGridT::ValueType value = tools::BoxSampler::sample(mAccessor, pointIndexSpace);

        // if min is greater than max, we invert so that values are valid outside of the range (not inside)
        const bool invert = mMin > mMax;

        return invert ? (value < mMax || value > mMin) : (value < mMax && value > mMin);
    }

private:
    // not a reference to ensure const-accessor is unique per-thread
    const typename LevelSetGridT::ConstAccessor mAccessor;
    const math::Transform& mLevelSetTransform;
    const math::Transform& mTransform;
    const ValueT mMin;
    const ValueT mMax;
    Handle::UniquePtr mPositionHandle;
}; // class LevelSetFilter


// BBox index filtering
class BBoxFilter
{
public:
    using Handle = AttributeHandle<openvdb::Vec3f>;

    BBoxFilter(const openvdb::math::Transform& transform,
             const openvdb::BBoxd& bboxWS)
            : mTransform(transform)
            , mBbox(transform.worldToIndex(bboxWS)) { }

    BBoxFilter(const BBoxFilter& filter)
        : mTransform(filter.mTransform)
        , mBbox(filter.mBbox)
    {
        if (filter.mPositionHandle)     mPositionHandle.reset(new Handle(*filter.mPositionHandle));
    }

    inline bool initialized() const { return bool(mPositionHandle); }

    inline index::State state() const
    {
        return mBbox.empty() ? index::NONE : index::PARTIAL;
    }
    template <typename LeafT>
    static index::State state(const LeafT&) { return index::PARTIAL; }

    template <typename LeafT>
    void reset(const LeafT& leaf) {
        mPositionHandle.reset(new Handle(leaf.constAttributeArray("P")));
    }

    template <typename IterT>
    bool valid(const IterT& iter) const {
        assert(mPositionHandle);

        const openvdb::Coord ijk = iter.getCoord();
        const openvdb::Vec3f voxelIndexSpace = ijk.asVec3d();

        // Retrieve point position in voxel space
        const openvdb::Vec3f& pointVoxelSpace = mPositionHandle->get(*iter);

        // Compute point position in index space
        const openvdb::Vec3f pointIndexSpace = pointVoxelSpace + voxelIndexSpace;

        return mBbox.isInside(pointIndexSpace);
    }

private:
    const openvdb::math::Transform& mTransform;
    const openvdb::BBoxd mBbox;
    Handle::UniquePtr mPositionHandle;
}; // class BBoxFilter


// Index filtering based on evaluating both sub-filters
template <typename T1, typename T2, bool And = true>
class BinaryFilter
{
public:
    BinaryFilter(   const T1& filter1,
                    const T2& filter2)
        : mFilter1(filter1)
        , mFilter2(filter2) { }

    inline bool initialized() const { return mFilter1.initialized() && mFilter2.initialized(); }

    inline index::State state() const
    {
        return this->computeState(mFilter1.state(), mFilter2.state());
    }
    template <typename LeafT>
    inline index::State state(const LeafT& leaf) const
    {
        return this->computeState(mFilter1.state(leaf), mFilter2.state(leaf));
    }

    template <typename LeafT>
    void reset(const LeafT& leaf) {
        mFilter1.reset(leaf);
        mFilter2.reset(leaf);
    }

    template <typename IterT>
    bool valid(const IterT& iter) const {
        if (And)      return mFilter1.valid(iter) && mFilter2.valid(iter);
        return mFilter1.valid(iter) || mFilter2.valid(iter);
    }

private:
    inline index::State computeState(   index::State state1,
                                        index::State state2) const
    {
        if (And) {
            if (state1 == index::NONE || state2 == index::NONE)       return index::NONE;
            else if (state1 == index::ALL && state2 == index::ALL)    return index::ALL;
        } else {
            if (state1 == index::NONE && state2 == index::NONE)       return index::NONE;
            else if (state1 == index::ALL && state2 == index::ALL)    return index::ALL;
        }
        return index::PARTIAL;
    }

    T1 mFilter1;
    T2 mFilter2;
}; // class BinaryFilter


////////////////////////////////////////


template<typename T>
struct FilterTraits {
    static const bool RequiresCoord = false;
};
template<>
struct FilterTraits<BBoxFilter> {
    static const bool RequiresCoord = true;
};
template <typename T>
struct FilterTraits<LevelSetFilter<T>> {
    static const bool RequiresCoord = true;
};
template <typename T0, typename T1, bool And>
struct FilterTraits<BinaryFilter<T0, T1, And>> {
    static const bool RequiresCoord =   FilterTraits<T0>::RequiresCoord ||
                                        FilterTraits<T1>::RequiresCoord;
};


////////////////////////////////////////


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_INDEX_FILTER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
