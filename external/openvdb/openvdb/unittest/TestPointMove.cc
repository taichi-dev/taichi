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

#include <cppunit/extensions/HelperMacros.h>
#include "util.h"
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointMove.h>
#include <openvdb/points/PointScatter.h>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <tbb/atomic.h>
#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace openvdb;
using namespace openvdb::points;

class TestPointMove: public CppUnit::TestCase
{
public:

    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointMove);
    CPPUNIT_TEST(testCachedDeformer);
    CPPUNIT_TEST(testMoveLocal);
    CPPUNIT_TEST(testMoveGlobal);
    CPPUNIT_TEST(testCustomDeformer);
    CPPUNIT_TEST(testPointData);
    CPPUNIT_TEST(testPointOrder);
    CPPUNIT_TEST_SUITE_END();

    void testCachedDeformer();
    void testMoveLocal();
    void testMoveGlobal();
    void testCustomDeformer();
    void testPointData();
    void testPointOrder();
}; // class TestPointMove

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointMove);


////////////////////////////////////////


namespace {

struct OffsetDeformer
{
    OffsetDeformer(const Vec3d& _offset)
        : offset(_offset){ }

    template <typename LeafT>
    void reset(const LeafT&, size_t /*idx*/) { }

    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT&) const
    {
        position += offset;
    }

    Vec3d offset;
}; // struct OffsetDeformer

template <typename FilterT>
struct OffsetFilteredDeformer
{
    OffsetFilteredDeformer(const Vec3d& _offset,
                    const FilterT& _filter)
        : offset(_offset)
        , filter(_filter) { }

    template <typename LeafT>
    void reset(const LeafT& leaf, size_t /*idx*/)
    {
        filter.template reset<LeafT>(leaf);
    }

    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT& iter) const
    {
        if (!filter.template valid<IndexIterT>(iter))    return;
        position += offset;
    }

    //void finalize() const { }

    Vec3d offset;
    FilterT filter;
}; // struct OffsetFilteredDeformer


PointDataGrid::Ptr
positionsToGrid(const std::vector<Vec3s>& positions, const float voxelSize = 1.0)
{
    const PointAttributeVector<Vec3s> pointList(positions);

    openvdb::math::Transform::Ptr transform(
        openvdb::math::Transform::createLinearTransform(voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                                                          pointList, *transform);

    // assign point 3 to new group "test" if more than 3 points

    if (positions.size() > 3) {
        appendGroup(points->tree(), "test");

        std::vector<short> groups(positions.size(), 0);
        groups[2] = 1;

        setGroup(points->tree(), pointIndexGrid->tree(), groups, "test");
    }

    return points;
}


std::vector<Vec3s>
gridToPositions(const PointDataGrid::Ptr& points, bool sort = true)
{
    std::vector<Vec3s> positions;

    for (auto leaf = points->tree().beginLeaf(); leaf; ++leaf) {

        const openvdb::points::AttributeArray& positionArray =
            leaf->constAttributeArray("P");
        openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(positionArray);

        for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
            openvdb::Vec3f voxelPosition = positionHandle.get(*iter);
            openvdb::Vec3d xyz = iter.getCoord().asVec3d();
            openvdb::Vec3f worldPosition = points->transform().indexToWorld(voxelPosition + xyz);

            positions.push_back(worldPosition);
        }
    }

    if (sort)   std::sort(positions.begin(), positions.end());
    return positions;
}


std::vector<Vec3s>
applyOffset(const std::vector<Vec3s>& positions, const Vec3s& offset)
{
    std::vector<Vec3s> newPositions;

    for (const auto& it : positions) {
        newPositions.emplace_back(it + offset);
    }

    std::sort(newPositions.begin(), newPositions.end());
    return newPositions;
}


template<typename T>
inline void
ASSERT_APPROX_EQUAL(const std::vector<T>& a, const std::vector<T>& b,
                    const Index lineNumber, const double /*tolerance*/ = 1e-6)
{
    std::stringstream ss;
    ss << "Assertion Line Number: " << lineNumber;

    CPPUNIT_ASSERT_EQUAL_MESSAGE(ss.str(), a.size(), b.size());

    for (int i = 0; i < a.size(); i++) {
        CPPUNIT_ASSERT_MESSAGE(ss.str(), math::isApproxEqual(a[i], b[i]));
    }
}


template<typename T>
inline void
ASSERT_APPROX_EQUAL(const std::vector<math::Vec3<T>>& a, const std::vector<math::Vec3<T>>& b,
                    const Index lineNumber, const double /*tolerance*/ = 1e-6)
{
    std::stringstream ss;
    ss << "Assertion Line Number: " << lineNumber;

    CPPUNIT_ASSERT_EQUAL_MESSAGE(ss.str(), a.size(), b.size());

    for (size_t i = 0; i < a.size(); i++) {
        CPPUNIT_ASSERT_MESSAGE(ss.str(), math::isApproxEqual(a[i], b[i]));
    }
}

struct NullObject { };

// A dummy iterator that can be used to match LeafIter and IndexIter interfaces
struct DummyIter
{
    DummyIter(Index _index): index(_index) { }
    //Index pos() const { return index; }
    Index operator*() const { return index; }
    Index index;
};

struct OddIndexFilter
{
    static bool initialized() { return true; }
    static index::State state() { return index::PARTIAL; }
    template <typename LeafT>
    static index::State state(const LeafT&) { return index::PARTIAL; }

    template <typename LeafT>
    void reset(const LeafT&) { }
    template <typename IterT>
    bool valid(const IterT& iter) const {
        return ((*iter) % 2) == 1;
    }
};

} // namespace


void
TestPointMove::testCachedDeformer()
{
    NullObject nullObject;

    // create an empty cache and CachedDeformer
    CachedDeformer<double>::Cache cache;
    CPPUNIT_ASSERT(cache.leafs.empty());

    CachedDeformer<double> cachedDeformer(cache);

    // check initialization is as expected
    CPPUNIT_ASSERT(cachedDeformer.mLocalLeafVec.empty());
    CPPUNIT_ASSERT(cachedDeformer.mLeafVec == nullptr);
    CPPUNIT_ASSERT(cachedDeformer.mLeafMap == nullptr);

    // throw when resetting cachedDeformer with an empty cache
    CPPUNIT_ASSERT_THROW(cachedDeformer.reset(nullObject, size_t(0)), openvdb::IndexError);

    // manually create one leaf in the cache
    cache.leafs.resize(1);
    auto& leaf = cache.leafs[0];
    CPPUNIT_ASSERT(leaf.vecData.empty());
    CPPUNIT_ASSERT(leaf.mapData.empty());
    CPPUNIT_ASSERT_EQUAL(Index(0), leaf.totalSize);

    // reset should no longer throw and leaf vec pointer should now be non-null
    CPPUNIT_ASSERT_NO_THROW(cachedDeformer.reset(nullObject, size_t(0)));
    CPPUNIT_ASSERT(cachedDeformer.mLocalLeafVec.empty());
    CPPUNIT_ASSERT(cachedDeformer.mLeafMap == nullptr);
    CPPUNIT_ASSERT(cachedDeformer.mLeafVec != nullptr);
    CPPUNIT_ASSERT(cachedDeformer.mLeafVec->empty());

    // nothing stored in the cache so position is unchanged
    DummyIter indexIter(0);
    Vec3d position(0,0,0);
    Vec3d newPosition(position);
    cachedDeformer.apply(newPosition, indexIter);
    CPPUNIT_ASSERT(math::isApproxEqual(position, newPosition));

    // insert a new value into the leaf vector and verify tbe position is deformed
    Vec3d deformedPosition(5,10,15);
    leaf.vecData.push_back(deformedPosition);
    cachedDeformer.apply(newPosition, indexIter);
    CPPUNIT_ASSERT(math::isApproxEqual(deformedPosition, newPosition));

    // insert a new value into the leaf map and verify the position is deformed as before
    Vec3d newDeformedPosition(2,3,4);
    leaf.mapData.insert({0, newDeformedPosition});
    newPosition.setZero();
    cachedDeformer.apply(newPosition, indexIter);
    CPPUNIT_ASSERT(math::isApproxEqual(deformedPosition, newPosition));

    // now reset the cached deformer and verify the value is updated
    // (map has precedence over vector)
    cachedDeformer.reset(nullObject, size_t(0));
    CPPUNIT_ASSERT(cachedDeformer.mLocalLeafVec.empty());
    CPPUNIT_ASSERT(cachedDeformer.mLeafMap != nullptr);
    CPPUNIT_ASSERT(cachedDeformer.mLeafVec == nullptr);
    newPosition.setZero();
    cachedDeformer.apply(newPosition, indexIter);
    CPPUNIT_ASSERT(math::isApproxEqual(newDeformedPosition, newPosition));

    // test map -> vector expansion
    leaf.mapData.clear();
    for (int i = 0; i < 16; i++) {
        leaf.mapData.insert({i, Vec3d(0, i, 0)});
    }

    cachedDeformer.reset(nullObject, size_t(0));

    // 16 values (or less) so local vector is not populated
    CPPUNIT_ASSERT(cachedDeformer.mLocalLeafVec.empty());
    CPPUNIT_ASSERT(cachedDeformer.mLeafMap != nullptr);
    CPPUNIT_ASSERT(cachedDeformer.mLeafVec == nullptr);

    // check value access
    for (int i = 0; i < 16; i++) {
        DummyIter indexIterI(i);
        cachedDeformer.apply(newPosition, indexIterI);
        CPPUNIT_ASSERT(math::isApproxEqual(Vec3d(0, i, 0), newPosition));
    }

    leaf.mapData.insert({16, Vec3d(0, 16, 0)});

    // ValueError thrown because totalSize has not been set correctly
    CPPUNIT_ASSERT_THROW(cachedDeformer.reset(nullObject, size_t(0)), openvdb::ValueError);

    // use very large total size to prevent local expansion
    leaf.totalSize = 17 * 256 + 1;

    CPPUNIT_ASSERT_NO_THROW(cachedDeformer.reset(nullObject, size_t(0)));

    CPPUNIT_ASSERT(cachedDeformer.mLocalLeafVec.empty());
    CPPUNIT_ASSERT(cachedDeformer.mLeafMap != nullptr);
    CPPUNIT_ASSERT(cachedDeformer.mLeafVec == nullptr);

    // use total size that represents a sequential dataset
    leaf.totalSize = Index(leaf.mapData.size());

    CPPUNIT_ASSERT_NO_THROW(cachedDeformer.reset(nullObject, size_t(0)));

    // greater than 16 values so local vector is populated
    CPPUNIT_ASSERT_EQUAL(leaf.mapData.size(), cachedDeformer.mLocalLeafVec.size());
    CPPUNIT_ASSERT(cachedDeformer.mLeafMap == nullptr);
    CPPUNIT_ASSERT(cachedDeformer.mLeafVec != nullptr);

    // check value access
    for (int i = 0; i < 17; i++) {
        DummyIter indexIterI(i);
        cachedDeformer.apply(newPosition, indexIterI);
        CPPUNIT_ASSERT(math::isApproxEqual(Vec3d(0, i, 0), newPosition));
    }

    // four points, some same leaf, some different
    const float voxelSize = 1.0f;
    std::vector<Vec3s> positions =  {
                                        {5, 2, 3},
                                        {2, 4, 1},
                                        {50, 5, 1},
                                        {3, 20, 1},
                                    };

    PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

    // evaluate with null deformer and no filter

    NullDeformer nullDeformer;
    NullFilter nullFilter;
    cachedDeformer.evaluate(*points, nullDeformer, nullFilter);

    CPPUNIT_ASSERT_EQUAL(size_t(points->tree().leafCount()), cache.leafs.size());

    int leafIndex = 0;
    for (auto leafIter = points->tree().cbeginLeaf(); leafIter; ++leafIter) {
        for (auto iter = leafIter->beginIndexOn(); iter; ++iter) {
            AttributeHandle<Vec3f> handle(leafIter->constAttributeArray("P"));
            Vec3f pos(handle.get(*iter) + iter.getCoord().asVec3s());
            Vec3f cachePosition = cache.leafs[leafIndex].vecData[*iter];
            CPPUNIT_ASSERT(math::isApproxEqual(pos, cachePosition));
        }
        leafIndex++;
    }

    // evaluate with Offset deformer and no filter

    Vec3d yOffset(1,2,3);
    OffsetDeformer yOffsetDeformer(yOffset);

    cachedDeformer.evaluate(*points, yOffsetDeformer, nullFilter);

    CPPUNIT_ASSERT_EQUAL(size_t(points->tree().leafCount()), cache.leafs.size());

    leafIndex = 0;
    for (auto leafIter = points->tree().cbeginLeaf(); leafIter; ++leafIter) {
        for (auto iter = leafIter->beginIndexOn(); iter; ++iter) {
            AttributeHandle<Vec3f> handle(leafIter->constAttributeArray("P"));
            Vec3f pos(handle.get(*iter) + iter.getCoord().asVec3s() + yOffset);
            Vec3f cachePosition = cache.leafs[leafIndex].vecData[*iter];
            CPPUNIT_ASSERT(math::isApproxEqual(pos, cachePosition));
        }
        leafIndex++;
    }

    // evaluate with Offset deformer and OddIndex filter

    OddIndexFilter oddFilter;
    cachedDeformer.evaluate(*points, yOffsetDeformer, oddFilter);

    CPPUNIT_ASSERT_EQUAL(size_t(points->tree().leafCount()), cache.leafs.size());

    leafIndex = 0;
    for (auto leafIter = points->tree().cbeginLeaf(); leafIter; ++leafIter) {
        for (auto iter = leafIter->beginIndexOn(); iter; ++iter) {
            AttributeHandle<Vec3f> handle(leafIter->constAttributeArray("P"));
            Vec3f pos(handle.get(*iter) + iter.getCoord().asVec3s() + yOffset);
            Vec3f cachePosition = cache.leafs[leafIndex].vecData[*iter];
            CPPUNIT_ASSERT(math::isApproxEqual(pos, cachePosition));
        }
        leafIndex++;
    }
}


void
TestPointMove::testMoveLocal()
{
    // This test is for points that only move locally, meaning that
    // they remain in the leaf from which they originated

    { // single point, y offset, same voxel
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {10, 10, 10},
                                                };

        std::vector<Vec3s> desiredPositions = applyOffset(positions, offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // two points, y offset, same voxel
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {10, 10, 10},
                                                    {10, 10.1f, 10},
                                                };

        std::vector<Vec3s> desiredPositions = applyOffset(positions, offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // two points, y offset, different voxels
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {10, 10, 10},
                                                    {10, 11, 10},
                                                };

        std::vector<Vec3s> desiredPositions = applyOffset(positions, offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, y offset, same voxel, only third point is kept
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {10, 10, 10},
                                                    {10, 10.1f, 10},
                                                    {10, 10.2f, 10},
                                                    {10, 10.3f, 10},
                                                };

        std::vector<Vec3s> desiredPositions;
        desiredPositions.emplace_back(positions[2]+offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        movePoints(*points, deformer, filter);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, y offset, different voxels, only third point is kept
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {10, 10, 10},
                                                    {10, 11, 10},
                                                    {10, 12, 10},
                                                    {10, 13, 10},
                                                };

        std::vector<Vec3s> desiredPositions;
        desiredPositions.emplace_back(positions[2]+offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        movePoints(*points, deformer, filter);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, y offset, different voxels, only third point is moved
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);

        std::vector<Vec3s> positions =          {
                                                    {10, 10, 10},
                                                    {10, 11, 10},
                                                    {10, 12, 10},
                                                    {10, 13, 10},
                                                };

        std::vector<Vec3s> desiredPositions(positions);
        desiredPositions[2] = Vec3s(positions[2] + offset);

        std::sort(desiredPositions.begin(), desiredPositions.end());

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        OffsetFilteredDeformer<MultiGroupFilter> deformer(offset, filter);
        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }
}


void
TestPointMove::testMoveGlobal()
{
    { // four points, all different leafs
        const float voxelSize = 0.1f;
        Vec3d offset(0, 10.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {1, 1, 1},
                                                    {1, 5.05f, 1},
                                                    {2, 1, 1},
                                                    {2, 2, 1},
                                                };

        std::vector<Vec3s> desiredPositions = applyOffset(positions, offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, all different leafs, only third point is kept
        const float voxelSize = 0.1f;
        Vec3d offset(0, 10.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {1, 1, 1},
                                                    {1, 5.05f, 1},
                                                    {2, 1, 1},
                                                    {2, 2, 1},
                                                };

        std::vector<Vec3s> desiredPositions;
        desiredPositions.emplace_back(positions[2]+offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        movePoints(*points, deformer, filter);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, all different leafs, third point is deleted
        const float voxelSize = 0.1f;
        Vec3d offset(0, 10.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {1, 1, 1},
                                                    {1, 5.05f, 1},
                                                    {2, 1, 1},
                                                    {2, 2, 1},
                                                };

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<Vec3s> desiredPositions;
        desiredPositions.emplace_back(positions[0]+offset);
        desiredPositions.emplace_back(positions[1]+offset);
        desiredPositions.emplace_back(positions[3]+offset);

        std::vector<std::string> includeGroups;
        std::vector<std::string> excludeGroups{"test"};

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        movePoints(*points, deformer, filter);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // six points, some same leaf, some different
        const float voxelSize = 1.0f;
        Vec3d offset(0, 0.1, 0);
        OffsetDeformer deformer(offset);

        std::vector<Vec3s> positions =          {
                                                    {1,     1,     1},
                                                    {1.01f, 1.01f, 1.01f},
                                                    {1,     5.05f, 1},
                                                    {2,     1,     1},
                                                    {2.01f, 1.01f, 1.01f},
                                                    {2,     2,     1},
                                                };

        std::vector<Vec3s> desiredPositions = applyOffset(positions, offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, all different leafs, only third point is moved
        const float voxelSize = 0.1f;
        Vec3d offset(0, 10.1, 0);

        std::vector<Vec3s> positions =          {
                                                    {1, 1, 1},
                                                    {1, 5.05f, 1},
                                                    {2, 1, 1},
                                                    {2, 2, 1},
                                                };

        std::vector<Vec3s> desiredPositions(positions);
        desiredPositions[2] = Vec3s(positions[2] + offset);

        std::sort(desiredPositions.begin(), desiredPositions.end());

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        OffsetFilteredDeformer<MultiGroupFilter> deformer(offset, filter);
        movePoints(*points, deformer);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }

    { // four points, all different leafs, only third point is kept but not moved
        const float voxelSize = 0.1f;
        Vec3d offset(0, 10.1, 0);

        std::vector<Vec3s> positions =          {
                                                    {1, 1, 1},
                                                    {1, 5.05f, 1},
                                                    {2, 1, 1},
                                                    {2, 2, 1},
                                                };

        std::vector<Vec3s> desiredPositions;
        desiredPositions.emplace_back(positions[2]);

        std::sort(desiredPositions.begin(), desiredPositions.end());

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        // these groups mark which points are kept

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;

        // these groups mark which points are moved

        std::vector<std::string> moveIncludeGroups;
        std::vector<std::string> moveExcludeGroups{"test"};

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter moveFilter(moveIncludeGroups, moveExcludeGroups, leaf->attributeSet());
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        OffsetFilteredDeformer<MultiGroupFilter> deformer(offset, moveFilter);
        movePoints(*points, deformer, filter);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
    }
}


namespace {

// Custom Deformer with reset and apply counters
struct CustomDeformer
{
    using LeafT = PointDataGrid::TreeType::LeafNodeType;

    CustomDeformer(const openvdb::Vec3d& offset,
                   tbb::atomic<int>& resetCalls,
                   tbb::atomic<int>& applyCalls)
        : mOffset(offset)
        , mResetCalls(resetCalls)
        , mApplyCalls(applyCalls) { }

    template <typename LeafT>
    void reset(const LeafT& /*leaf*/, size_t /*idx*/)
    {
        mResetCalls++;
    }

    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT&) const
    {
        // ensure reset has been called at least once
        if (mResetCalls > 0) {
            position += mOffset;
        }
        mApplyCalls++;
    }

    const openvdb::Vec3d mOffset;
    tbb::atomic<int>& mResetCalls;
    tbb::atomic<int>& mApplyCalls;
}; // struct CustomDeformer

// Custom Deformer that always returns the position supplied in the constructor
struct StaticDeformer
{
    StaticDeformer(const openvdb::Vec3d& position)
        : mPosition(position) { }

    template <typename LeafT>
    void reset(const LeafT& /*leaf*/, size_t /*idx*/) { }

    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT&) const
    {
        position = mPosition;
    }

    const openvdb::Vec3d mPosition;
}; // struct StaticDeformer

} // namespace

void
TestPointMove::testCustomDeformer()
{
    { // four points, some same leaf, some different, custom deformer
        const float voxelSize = 1.0f;
        Vec3d offset(4.5,3.2,1.85);

        std::vector<Vec3s> positions =          {
                                                    {5, 2, 3},
                                                    {2, 4, 1},
                                                    {50, 5, 1},
                                                    {3, 20, 1},
                                                };

        std::vector<Vec3s> desiredPositions = applyOffset(positions, offset);

        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);
        PointDataGrid::Ptr cachedPoints = points->deepCopy();

        const int leafCount = points->tree().leafCount();
        const int pointCount = int(positions.size());

        tbb::atomic<int> resetCalls = 0;
        tbb::atomic<int> applyCalls = 0;

        // this deformer applies an offset and tracks the number of calls

        CustomDeformer deformer(offset, resetCalls, applyCalls);

        movePoints(*points, deformer);

        CPPUNIT_ASSERT(2*leafCount == resetCalls);
        CPPUNIT_ASSERT(2*pointCount == applyCalls);

        std::vector<Vec3s> actualPositions = gridToPositions(points);

        ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);

        // use CachedDeformer

        resetCalls = 0;
        applyCalls = 0;

        CachedDeformer<double>::Cache cache;
        CachedDeformer<double> cachedDeformer(cache);
        NullFilter filter;
        cachedDeformer.evaluate(*cachedPoints, deformer, filter);

        movePoints(*cachedPoints, cachedDeformer);

        CPPUNIT_ASSERT(leafCount == resetCalls);
        CPPUNIT_ASSERT(pointCount == applyCalls);

        std::vector<Vec3s> cachedPositions = gridToPositions(cachedPoints);

        ASSERT_APPROX_EQUAL(desiredPositions, cachedPositions, __LINE__);
    }

    {
        { // four points, some same leaf, some different, static deformer
            const float voxelSize = 1.0f;
            Vec3d newPosition(15.2,18.3,-100.9);

            std::vector<Vec3s> positions =          {
                                                        {5, 2, 3},
                                                        {2, 4, 1},
                                                        {50, 5, 1},
                                                        {3, 20, 1},
                                                    };

            std::vector<Vec3s> desiredPositions(positions.size(), newPosition);

            PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

            StaticDeformer deformer(newPosition);

            movePoints(*points, deformer);

            std::vector<Vec3s> actualPositions = gridToPositions(points);

            ASSERT_APPROX_EQUAL(desiredPositions, actualPositions, __LINE__);
        }
    }
}


namespace {

// Custom deformer that stores a map of current positions to new positions
struct AssignDeformer
{
    AssignDeformer(const std::map<Vec3d, Vec3d>& _values)
        : values(_values) { }

    template <typename LeafT>
    void reset(const LeafT&, size_t /*idx*/) { }

    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT&) const
    {
        position = values.at(position);
    }

    std::map<Vec3d, Vec3d> values;
}; // struct AssignDeformer

}


void
TestPointMove::testPointData()
{
    // four points, some same leaf, some different
    // spatial order is (1, 0, 3, 2)

    const float voxelSize = 1.0f;
    std::vector<Vec3s> positions =  {
                                    {5, 2, 3},
                                    {2, 4, 1},
                                    {50, 5, 1},
                                    {3, 20, 1},
                                };

    // simple reversing deformer

    std::map<Vec3d, Vec3d> remap;
    remap.insert({positions[0], positions[3]});
    remap.insert({positions[1], positions[2]});
    remap.insert({positions[2], positions[1]});
    remap.insert({positions[3], positions[0]});

    AssignDeformer deformer(remap);

    { // reversing point positions results in the same iteration order due to spatial organisation
        PointDataGrid::Ptr points = positionsToGrid(positions, voxelSize);

        std::vector<Vec3s> initialPositions = gridToPositions(points, /*sort=*/false);

        std::vector<std::string> includeGroups;
        std::vector<std::string> excludeGroups;

        movePoints(*points, deformer);

        std::vector<Vec3s> finalPositions1 = gridToPositions(points, /*sort=*/false);

        ASSERT_APPROX_EQUAL(initialPositions, finalPositions1, __LINE__);

        // now we delete the third point while sorting, using the test group

        excludeGroups.push_back("test");

        auto leaf = points->tree().cbeginLeaf();
        MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
        movePoints(*points, deformer, filter);

        std::vector<Vec3s> desiredPositions;
        desiredPositions.emplace_back(positions[0]);
        desiredPositions.emplace_back(positions[1]);
        desiredPositions.emplace_back(positions[3]);

        std::vector<Vec3s> finalPositions2 = gridToPositions(points, /*sort=*/false);

        std::sort(desiredPositions.begin(), desiredPositions.end());
        std::sort(finalPositions2.begin(), finalPositions2.end());

        ASSERT_APPROX_EQUAL(desiredPositions, finalPositions2, __LINE__);
    }

    { // additional point data - integer "id", float "pscale", "odd" and "even" groups

        std::vector<int> id;
        id.push_back(0);
        id.push_back(1);
        id.push_back(2);
        id.push_back(3);

        std::vector<float> radius;
        radius.push_back(0.1f);
        radius.push_back(0.15f);
        radius.push_back(0.2f);
        radius.push_back(0.5f);

        // manually construct point data grid instead of using positionsToGrid()

        const PointAttributeVector<Vec3s> pointList(positions);

        openvdb::math::Transform::Ptr transform(
            openvdb::math::Transform::createLinearTransform(voxelSize));

        tools::PointIndexGrid::Ptr pointIndexGrid =
            tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

        PointDataGrid::Ptr points =
                createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                                                              pointList, *transform);
        auto idAttributeType =
            openvdb::points::TypedAttributeArray<int>::attributeType();
        openvdb::points::appendAttribute(points->tree(), "id", idAttributeType);

        // create a wrapper around the id vector
        openvdb::points::PointAttributeVector<int> idWrapper(id);

        openvdb::points::populateAttribute<openvdb::points::PointDataTree,
            openvdb::tools::PointIndexTree, openvdb::points::PointAttributeVector<int>>(
                points->tree(), pointIndexGrid->tree(), "id", idWrapper);

        // use fixed-point codec for radius
        // note that this attribute type is not registered by default so needs to be
        // explicitly registered.
        using Codec = openvdb::points::FixedPointCodec</*1-byte=*/false,
                openvdb::points::UnitRange>;
        openvdb::points::TypedAttributeArray<float, Codec>::registerType();
        auto radiusAttributeType =
            openvdb::points::TypedAttributeArray<float, Codec>::attributeType();
        openvdb::points::appendAttribute(points->tree(), "pscale", radiusAttributeType);

        // create a wrapper around the radius vector
        openvdb::points::PointAttributeVector<float> radiusWrapper(radius);

        openvdb::points::populateAttribute<openvdb::points::PointDataTree,
            openvdb::tools::PointIndexTree, openvdb::points::PointAttributeVector<float>>(
                points->tree(), pointIndexGrid->tree(), "pscale", radiusWrapper);

        appendGroup(points->tree(), "odd");
        appendGroup(points->tree(), "even");
        appendGroup(points->tree(), "nonzero");

        std::vector<short> oddGroups(positions.size(), 0);
        oddGroups[1] = 1;
        oddGroups[3] = 1;
        std::vector<short> evenGroups(positions.size(), 0);
        evenGroups[0] = 1;
        evenGroups[2] = 1;
        std::vector<short> nonZeroGroups(positions.size(), 1);
        nonZeroGroups[0] = 0;

        setGroup(points->tree(), pointIndexGrid->tree(), evenGroups, "even");
        setGroup(points->tree(), pointIndexGrid->tree(), oddGroups, "odd");
        setGroup(points->tree(), pointIndexGrid->tree(), nonZeroGroups, "nonzero");

        movePoints(*points, deformer);

        // extract data

        std::vector<int> id2;
        std::vector<float> radius2;
        std::vector<short> oddGroups2;
        std::vector<short> evenGroups2;
        std::vector<short> nonZeroGroups2;

        for (auto leaf = points->tree().cbeginLeaf(); leaf; ++leaf) {

            AttributeHandle<int> idHandle(leaf->constAttributeArray("id"));
            AttributeHandle<float> pscaleHandle(leaf->constAttributeArray("pscale"));
            GroupHandle oddHandle(leaf->groupHandle("odd"));
            GroupHandle evenHandle(leaf->groupHandle("even"));
            GroupHandle nonZeroHandle(leaf->groupHandle("nonzero"));

            for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                id2.push_back(idHandle.get(*iter));
                radius2.push_back(pscaleHandle.get(*iter));
                oddGroups2.push_back(oddHandle.get(*iter) ? 1 : 0);
                evenGroups2.push_back(evenHandle.get(*iter) ? 1 : 0);
                nonZeroGroups2.push_back(nonZeroHandle.get(*iter) ? 1 : 0);
            }
        }

        // new reversed order is (2, 3, 0, 1)

        CPPUNIT_ASSERT_EQUAL(2, id2[0]);
        CPPUNIT_ASSERT_EQUAL(3, id2[1]);
        CPPUNIT_ASSERT_EQUAL(0, id2[2]);
        CPPUNIT_ASSERT_EQUAL(1, id2[3]);

        CPPUNIT_ASSERT(math::isApproxEqual(radius[0], radius2[2], 1e-3f));
        CPPUNIT_ASSERT(math::isApproxEqual(radius[1], radius2[3], 1e-3f));
        CPPUNIT_ASSERT(math::isApproxEqual(radius[2], radius2[0], 1e-3f));
        CPPUNIT_ASSERT(math::isApproxEqual(radius[3], radius2[1], 1e-3f));

        CPPUNIT_ASSERT_EQUAL(short(0), oddGroups2[0]);
        CPPUNIT_ASSERT_EQUAL(short(1), oddGroups2[1]);
        CPPUNIT_ASSERT_EQUAL(short(0), oddGroups2[2]);
        CPPUNIT_ASSERT_EQUAL(short(1), oddGroups2[3]);

        CPPUNIT_ASSERT_EQUAL(short(1), evenGroups2[0]);
        CPPUNIT_ASSERT_EQUAL(short(0), evenGroups2[1]);
        CPPUNIT_ASSERT_EQUAL(short(1), evenGroups2[2]);
        CPPUNIT_ASSERT_EQUAL(short(0), evenGroups2[3]);

        CPPUNIT_ASSERT_EQUAL(short(1), nonZeroGroups2[0]);
        CPPUNIT_ASSERT_EQUAL(short(1), nonZeroGroups2[1]);
        CPPUNIT_ASSERT_EQUAL(short(0), nonZeroGroups2[2]);
        CPPUNIT_ASSERT_EQUAL(short(1), nonZeroGroups2[3]);
    }
}


void
TestPointMove::testPointOrder()
{
    struct Local
    {
        using GridT = points::PointDataGrid;

        static void populate(std::vector<Vec3s>& positions, const GridT& points,
            const math::Transform& transform, bool /*threaded*/)
        {
            auto newPoints1 = points.deepCopy();

            points::NullDeformer nullDeformer;
            points::NullFilter nullFilter;
            points::movePoints(*newPoints1, transform, nullDeformer, nullFilter);

            size_t totalPoints = points::pointCount(newPoints1->tree());

            positions.reserve(totalPoints);

            for (auto leaf = newPoints1->tree().cbeginLeaf(); leaf; ++leaf) {
                AttributeHandle<Vec3f> handle(leaf->constAttributeArray("P"));
                for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                    positions.push_back(handle.get(*iter));
                }
            }
        }
    };

    auto sourceTransform = math::Transform::createLinearTransform(/*voxelSize=*/0.1);
    auto targetTransform = math::Transform::createLinearTransform(/*voxelSize=*/1.0);

    auto mask = MaskGrid::create();
    mask->setTransform(sourceTransform);
    mask->denseFill(CoordBBox(Coord(-20,-20,-20), Coord(20,20,20)), true);

    auto points = points::denseUniformPointScatter(*mask, /*pointsPerVoxel=*/8);

    // three copies of the points, two multi-threaded and one single-threaded
    std::vector<Vec3s> positions1;
    std::vector<Vec3s> positions2;
    std::vector<Vec3s> positions3;

    Local::populate(positions1, *points, *targetTransform, true);
    Local::populate(positions2, *points, *targetTransform, true);
    Local::populate(positions3, *points, *targetTransform, false);

    // verify all sequences are identical to confirm that points are ordered deterministically

    ASSERT_APPROX_EQUAL(positions1, positions2, __LINE__);
    ASSERT_APPROX_EQUAL(positions1, positions3, __LINE__);
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
