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

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/openvdb.h>

#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointConversion.h>

#include <cmath>
#include <cstdio> // for std::remove()
#include <cstdlib> // for std::getenv()
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace openvdb;
using namespace openvdb::points;

class TestPointCount: public CppUnit::TestCase
{
public:

    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointCount);
    CPPUNIT_TEST(testCount);
    CPPUNIT_TEST(testGroup);
    CPPUNIT_TEST(testOffsets);
    CPPUNIT_TEST(testCountGrid);
    CPPUNIT_TEST_SUITE_END();

    void testCount();
    void testGroup();
    void testOffsets();
    void testCountGrid();

}; // class TestPointCount

using LeafType  = PointDataTree::LeafNodeType;
using ValueType = LeafType::ValueType;


struct NotZeroFilter
{
    NotZeroFilter() = default;
    static bool initialized() { return true; }
    template <typename LeafT>
    void reset(const LeafT&) { }
    template <typename IterT>
    bool valid(const IterT& iter) const {
        return *iter != 0;
    }
};

void
TestPointCount::testCount()
{
    // create a tree and check there are no points

    PointDataGrid::Ptr grid = createGrid<PointDataGrid>();
    PointDataTree& tree = grid->tree();

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(0));

    // add a new leaf to a tree and re-test

    LeafType* leafPtr = tree.touchLeaf(openvdb::Coord(0, 0, 0));
    LeafType& leaf(*leafPtr);

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(0));

    // now manually set some offsets

    leaf.setOffsetOn(0, 4);
    leaf.setOffsetOn(1, 7);

    ValueVoxelCIter voxelIter = leaf.beginValueVoxel(openvdb::Coord(0, 0, 0));

    IndexIter<ValueVoxelCIter, NullFilter> testIter(voxelIter, NullFilter());

    leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0));

    CPPUNIT_ASSERT_EQUAL(int(*leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0))), 0);
    CPPUNIT_ASSERT_EQUAL(int(leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0)).end()), 4);

    CPPUNIT_ASSERT_EQUAL(int(*leaf.beginIndexVoxel(openvdb::Coord(0, 0, 1))), 4);
    CPPUNIT_ASSERT_EQUAL(int(leaf.beginIndexVoxel(openvdb::Coord(0, 0, 1)).end()), 7);

    // test filtered, index voxel iterator

    CPPUNIT_ASSERT_EQUAL(int(*leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0), NotZeroFilter())), 1);
    CPPUNIT_ASSERT_EQUAL(int(leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0), NotZeroFilter()).end()), 4);

    {
        LeafType::IndexVoxelIter iter = leaf.beginIndexVoxel(openvdb::Coord(0, 0, 0));

        CPPUNIT_ASSERT_EQUAL(int(*iter), 0);
        CPPUNIT_ASSERT_EQUAL(int(iter.end()), 4);

        LeafType::IndexVoxelIter iter2 = leaf.beginIndexVoxel(openvdb::Coord(0, 0, 1));

        CPPUNIT_ASSERT_EQUAL(int(*iter2), 4);
        CPPUNIT_ASSERT_EQUAL(int(iter2.end()), 7);

        CPPUNIT_ASSERT_EQUAL(iterCount(iter2), Index64(7 - 4));

        // check pointCount ignores active/inactive state

        leaf.setValueOff(1);

        LeafType::IndexVoxelIter iter3 = leaf.beginIndexVoxel(openvdb::Coord(0, 0, 1));

        CPPUNIT_ASSERT_EQUAL(iterCount(iter3), Index64(7 - 4));

        leaf.setValueOn(1);
    }

    // one point per voxel

    for (unsigned int i = 0; i < LeafType::SIZE; i++) {
        leaf.setOffsetOn(i, i);
    }

    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(leaf.onPointCount(), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(leaf.offPointCount(), Index64(0));

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(pointCount(tree, ActiveFilter()), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(pointCount(tree, InactiveFilter()), Index64(0));

    // manually de-activate two voxels

    leaf.setValueOff(100);
    leaf.setValueOff(101);

    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(leaf.onPointCount(), Index64(LeafType::SIZE - 3));
    CPPUNIT_ASSERT_EQUAL(leaf.offPointCount(), Index64(2));

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(pointCount(tree, ActiveFilter()), Index64(LeafType::SIZE - 3));
    CPPUNIT_ASSERT_EQUAL(pointCount(tree, InactiveFilter()), Index64(2));

    // one point per every other voxel and de-activate empty voxels

    unsigned sum = 0;

    for (unsigned int i = 0; i < LeafType::SIZE; i++) {
        leaf.setOffsetOn(i, sum);
        if (i % 2 == 0)     sum++;
    }

    leaf.updateValueMask();

    CPPUNIT_ASSERT_EQUAL(leaf.pointCount(), Index64(LeafType::SIZE / 2));
    CPPUNIT_ASSERT_EQUAL(leaf.onPointCount(), Index64(LeafType::SIZE / 2));
    CPPUNIT_ASSERT_EQUAL(leaf.offPointCount(), Index64(0));

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(LeafType::SIZE / 2));
    CPPUNIT_ASSERT_EQUAL(pointCount(tree, ActiveFilter()), Index64(LeafType::SIZE / 2));
    CPPUNIT_ASSERT_EQUAL(pointCount(tree, InactiveFilter()), Index64(0));

    // add a new non-empty leaf and check totalPointCount is correct

    LeafType* leaf2Ptr = tree.touchLeaf(openvdb::Coord(0, 0, 8));
    LeafType& leaf2(*leaf2Ptr);

    // on adding, tree now obtains ownership and is reponsible for deletion

    for (unsigned int i = 0; i < LeafType::SIZE; i++) {
        leaf2.setOffsetOn(i, i);
    }

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(LeafType::SIZE / 2 + LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(pointCount(tree, ActiveFilter()), Index64(LeafType::SIZE / 2 + LeafType::SIZE - 1));
    CPPUNIT_ASSERT_EQUAL(pointCount(tree, InactiveFilter()), Index64(0));
}


void
TestPointCount::testGroup()
{
    using namespace openvdb::math;

    using Descriptor = AttributeSet::Descriptor;

    // four points in the same leaf

    std::vector<Vec3s> positions{{1, 1, 1}, {1, 2, 1}, {2, 1, 1}, {2, 2, 1}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // setup temp directory

    std::string tempDir;
    if (const char* dir = std::getenv("TMPDIR")) tempDir = dir;
#ifdef _MSC_VER
    if (tempDir.empty()) {
        char tempDirBuffer[MAX_PATH+1];
        int tempDirLen = GetTempPath(MAX_PATH+1, tempDirBuffer);
        CPPUNIT_ASSERT(tempDirLen > 0 && tempDirLen <= MAX_PATH);
        tempDir = tempDirBuffer;
    }
#else
    if (tempDir.empty()) tempDir = P_tmpdir;
#endif

    std::string filename;

    // check one leaf
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(1));

    // retrieve first and last leaf attribute sets

    PointDataTree::LeafIter leafIter = tree.beginLeaf();
    const AttributeSet& firstAttributeSet = leafIter->attributeSet();

    // ensure zero groups
    CPPUNIT_ASSERT_EQUAL(firstAttributeSet.descriptor().groupMap().size(), size_t(0));

    {// add an empty group
        appendGroup(tree, "test");

        CPPUNIT_ASSERT_EQUAL(firstAttributeSet.descriptor().groupMap().size(), size_t(1));

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(4));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, ActiveFilter()), Index64(4));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, InactiveFilter()), Index64(0));
        CPPUNIT_ASSERT_EQUAL(leafIter->pointCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(leafIter->onPointCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(leafIter->offPointCount(), Index64(0));

        // no points found when filtered by the empty group

        CPPUNIT_ASSERT_EQUAL(pointCount(tree, GroupFilter("test", firstAttributeSet)), Index64(0));
        CPPUNIT_ASSERT_EQUAL(leafIter->groupPointCount("test"), Index64(0));
    }

    { // assign two points to the group, test offsets and point counts
        const Descriptor::GroupIndex index = firstAttributeSet.groupIndex("test");

        CPPUNIT_ASSERT(index.first != AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(index.first < firstAttributeSet.size());

        AttributeArray& array = leafIter->attributeArray(index.first);

        CPPUNIT_ASSERT(isGroup(array));

        GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

        groupArray.set(0, GroupType(1) << index.second);
        groupArray.set(3, GroupType(1) << index.second);

        // only two out of four points should be found when group filtered

        GroupFilter firstGroupFilter("test", firstAttributeSet);

        CPPUNIT_ASSERT_EQUAL(pointCount(tree, GroupFilter("test", firstAttributeSet)), Index64(2));
        CPPUNIT_ASSERT_EQUAL(leafIter->groupPointCount("test"), Index64(2));

        {
            CPPUNIT_ASSERT_EQUAL(pointCount(tree, BinaryFilter<GroupFilter, ActiveFilter>(
                firstGroupFilter, ActiveFilter())), Index64(2));
            CPPUNIT_ASSERT_EQUAL(pointCount(tree, BinaryFilter<GroupFilter, InactiveFilter>(
                firstGroupFilter, InactiveFilter())), Index64(0));
        }

        CPPUNIT_ASSERT_NO_THROW(leafIter->validateOffsets());

        // manually modify offsets so one of the points is marked as inactive

        std::vector<ValueType> offsets, modifiedOffsets;
        offsets.resize(PointDataTree::LeafNodeType::SIZE);
        modifiedOffsets.resize(PointDataTree::LeafNodeType::SIZE);

        for (Index n = 0; n < PointDataTree::LeafNodeType::NUM_VALUES; n++) {
            const unsigned offset = leafIter->getValue(n);
            offsets[n] = offset;
            modifiedOffsets[n] = offset > 0 ? offset - 1 : offset;
        }

        leafIter->setOffsets(modifiedOffsets);

        // confirm that validation fails

        CPPUNIT_ASSERT_THROW(leafIter->validateOffsets(), openvdb::ValueError);

        // replace offsets with original offsets but leave value mask

        leafIter->setOffsets(offsets, /*updateValueMask=*/ false);

        // confirm that validation now succeeds

        CPPUNIT_ASSERT_NO_THROW(leafIter->validateOffsets());

        // ensure active / inactive point counts are correct

        CPPUNIT_ASSERT_EQUAL(pointCount(tree, GroupFilter("test", firstAttributeSet)), Index64(2));
        CPPUNIT_ASSERT_EQUAL(leafIter->groupPointCount("test"), Index64(2));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, BinaryFilter<GroupFilter, ActiveFilter>(
            firstGroupFilter, ActiveFilter())), Index64(1));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, BinaryFilter<GroupFilter, InactiveFilter>(
            firstGroupFilter, InactiveFilter())), Index64(1));

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(4));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, ActiveFilter()), Index64(3));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, InactiveFilter()), Index64(1));

        // write out grid to a temp file
        {
            filename = tempDir + "/openvdb_test_point_load";

            io::File fileOut(filename);

            GridCPtrVec grids{grid};

            fileOut.write(grids);
        }

        // test point count of a delay-loaded grid
        {
            io::File fileIn(filename);
            fileIn.open();

            GridPtrVecPtr grids = fileIn.getGrids();

            fileIn.close();

            CPPUNIT_ASSERT_EQUAL(grids->size(), size_t(1));

            PointDataGrid::Ptr inputGrid = GridBase::grid<PointDataGrid>((*grids)[0]);

            CPPUNIT_ASSERT(inputGrid);

            PointDataTree& inputTree = inputGrid->tree();
            const auto& attributeSet = inputTree.cbeginLeaf()->attributeSet();

            GroupFilter groupFilter("test", attributeSet);

            bool inCoreOnly = true;

#if OPENVDB_ABI_VERSION_NUMBER >= 3
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, NullFilter(), inCoreOnly), Index64(0));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, ActiveFilter(), inCoreOnly), Index64(0));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, InactiveFilter(), inCoreOnly), Index64(0));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, groupFilter, inCoreOnly), Index64(0));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, BinaryFilter<GroupFilter, ActiveFilter>(
                groupFilter, ActiveFilter()), inCoreOnly), Index64(0));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, BinaryFilter<GroupFilter, InactiveFilter>(
                groupFilter, InactiveFilter()), inCoreOnly), Index64(0));
#else
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, NullFilter(), inCoreOnly), Index64(4));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, ActiveFilter(), inCoreOnly), Index64(3));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, InactiveFilter(), inCoreOnly), Index64(1));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, groupFilter, inCoreOnly), Index64(2));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, BinaryFilter<GroupFilter, ActiveFilter>(
                groupFilter, ActiveFilter()), inCoreOnly), Index64(1));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, BinaryFilter<GroupFilter, InactiveFilter>(
                groupFilter, InactiveFilter()), inCoreOnly), Index64(1));
#endif

            inCoreOnly = false;

            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, NullFilter(), inCoreOnly), Index64(4));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, ActiveFilter(), inCoreOnly), Index64(3));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, InactiveFilter(), inCoreOnly), Index64(1));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, groupFilter, inCoreOnly), Index64(2));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, BinaryFilter<GroupFilter, ActiveFilter>(
                groupFilter, ActiveFilter()), inCoreOnly), Index64(1));
            CPPUNIT_ASSERT_EQUAL(pointCount(inputTree, BinaryFilter<GroupFilter, InactiveFilter>(
                groupFilter, InactiveFilter()), inCoreOnly), Index64(1));
        }

        // update the value mask and confirm point counts once again

        leafIter->updateValueMask();

        CPPUNIT_ASSERT_NO_THROW(leafIter->validateOffsets());

        auto& attributeSet = tree.cbeginLeaf()->attributeSet();

        CPPUNIT_ASSERT_EQUAL(pointCount(tree, GroupFilter("test", attributeSet)), Index64(2));
        CPPUNIT_ASSERT_EQUAL(leafIter->groupPointCount("test"), Index64(2));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, BinaryFilter<GroupFilter, ActiveFilter>(
            firstGroupFilter, ActiveFilter())), Index64(2));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, BinaryFilter<GroupFilter, InactiveFilter>(
            firstGroupFilter, InactiveFilter())), Index64(0));

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(4));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, ActiveFilter()), Index64(4));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, InactiveFilter()), Index64(0));
    }

    // create a tree with multiple leaves

    positions.emplace_back(20, 1, 1);
    positions.emplace_back(1, 20, 1);
    positions.emplace_back(1, 1, 20);

    grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree2 = grid->tree();

    CPPUNIT_ASSERT_EQUAL(tree2.leafCount(), Index32(4));

    leafIter = tree2.beginLeaf();

    appendGroup(tree2, "test");

    { // assign two points to the group
        const auto& attributeSet = leafIter->attributeSet();
        const Descriptor::GroupIndex index = attributeSet.groupIndex("test");

        CPPUNIT_ASSERT(index.first != AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(index.first < attributeSet.size());

        AttributeArray& array = leafIter->attributeArray(index.first);

        CPPUNIT_ASSERT(isGroup(array));

        GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

        groupArray.set(0, GroupType(1) << index.second);
        groupArray.set(3, GroupType(1) << index.second);

        CPPUNIT_ASSERT_EQUAL(pointCount(tree2, GroupFilter("test", attributeSet)), Index64(2));
        CPPUNIT_ASSERT_EQUAL(leafIter->groupPointCount("test"), Index64(2));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree2), Index64(7));
    }

    ++leafIter;

    CPPUNIT_ASSERT(leafIter);

    { // assign another point to the group in a different leaf
        const auto& attributeSet = leafIter->attributeSet();
        const Descriptor::GroupIndex index = attributeSet.groupIndex("test");

        CPPUNIT_ASSERT(index.first != AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(index.first < leafIter->attributeSet().size());

        AttributeArray& array = leafIter->attributeArray(index.first);

        CPPUNIT_ASSERT(isGroup(array));

        GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

        groupArray.set(0, GroupType(1) << index.second);

        CPPUNIT_ASSERT_EQUAL(pointCount(tree2, GroupFilter("test", attributeSet)), Index64(3));
        CPPUNIT_ASSERT_EQUAL(leafIter->groupPointCount("test"), Index64(1));
        CPPUNIT_ASSERT_EQUAL(pointCount(tree2), Index64(7));
    }
}


void
TestPointCount::testOffsets()
{
    using namespace openvdb::math;

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    // five points across four leafs

    std::vector<Vec3s> positions{{1, 1, 1}, {1, 101, 1}, {2, 101, 1}, {101, 1, 1}, {101, 101, 1}};

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    { // all point offsets
        std::vector<Index64> offsets;
        Index64 total = pointOffsets(offsets, tree);

        CPPUNIT_ASSERT_EQUAL(offsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(offsets[0], Index64(1));
        CPPUNIT_ASSERT_EQUAL(offsets[1], Index64(3));
        CPPUNIT_ASSERT_EQUAL(offsets[2], Index64(4));
        CPPUNIT_ASSERT_EQUAL(offsets[3], Index64(5));
        CPPUNIT_ASSERT_EQUAL(total, Index64(5));
    }

    { // all point offsets when using a non-existant exclude group

        std::vector<Index64> offsets;

        std::vector<Name> includeGroups;
        std::vector<Name> excludeGroups{"empty"};

        MultiGroupFilter filter(includeGroups, excludeGroups, tree.cbeginLeaf()->attributeSet());
        Index64 total = pointOffsets(offsets, tree, filter);

        CPPUNIT_ASSERT_EQUAL(offsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(offsets[0], Index64(1));
        CPPUNIT_ASSERT_EQUAL(offsets[1], Index64(3));
        CPPUNIT_ASSERT_EQUAL(offsets[2], Index64(4));
        CPPUNIT_ASSERT_EQUAL(offsets[3], Index64(5));
        CPPUNIT_ASSERT_EQUAL(total, Index64(5));
    }

    appendGroup(tree, "test");

    // add one point to the group from the leaf that contains two points

    PointDataTree::LeafIter iter = ++tree.beginLeaf();
    GroupWriteHandle groupHandle = iter->groupWriteHandle("test");
    groupHandle.set(0, true);

    { // include this group
        std::vector<Index64> offsets;

        std::vector<Name> includeGroups{"test"};
        std::vector<Name> excludeGroups;

        MultiGroupFilter filter(includeGroups, excludeGroups, tree.cbeginLeaf()->attributeSet());
        Index64 total = pointOffsets(offsets, tree, filter);

        CPPUNIT_ASSERT_EQUAL(offsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(offsets[0], Index64(0));
        CPPUNIT_ASSERT_EQUAL(offsets[1], Index64(1));
        CPPUNIT_ASSERT_EQUAL(offsets[2], Index64(1));
        CPPUNIT_ASSERT_EQUAL(offsets[3], Index64(1));
        CPPUNIT_ASSERT_EQUAL(total, Index64(1));
    }

    { // exclude this group
        std::vector<Index64> offsets;

        std::vector<Name> includeGroups;
        std::vector<Name> excludeGroups{"test"};

        MultiGroupFilter filter(includeGroups, excludeGroups, tree.cbeginLeaf()->attributeSet());
        Index64 total = pointOffsets(offsets, tree, filter);

        CPPUNIT_ASSERT_EQUAL(offsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(offsets[0], Index64(1));
        CPPUNIT_ASSERT_EQUAL(offsets[1], Index64(2));
        CPPUNIT_ASSERT_EQUAL(offsets[2], Index64(3));
        CPPUNIT_ASSERT_EQUAL(offsets[3], Index64(4));
        CPPUNIT_ASSERT_EQUAL(total, Index64(4));
    }

    // setup temp directory

    std::string tempDir;
    if (const char* dir = std::getenv("TMPDIR")) tempDir = dir;
#ifdef _MSC_VER
    if (tempDir.empty()) {
        char tempDirBuffer[MAX_PATH+1];
        int tempDirLen = GetTempPath(MAX_PATH+1, tempDirBuffer);
        CPPUNIT_ASSERT(tempDirLen > 0 && tempDirLen <= MAX_PATH);
        tempDir = tempDirBuffer;
    }
#else
    if (tempDir.empty()) tempDir = P_tmpdir;
#endif

    std::string filename;

    // write out grid to a temp file
    {
        filename = tempDir + "/openvdb_test_point_load";

        io::File fileOut(filename);

        GridCPtrVec grids{grid};

        fileOut.write(grids);
    }

    // test point offsets for a delay-loaded grid
    {
        io::File fileIn(filename);
        fileIn.open();

        GridPtrVecPtr grids = fileIn.getGrids();

        fileIn.close();

        CPPUNIT_ASSERT_EQUAL(grids->size(), size_t(1));

        PointDataGrid::Ptr inputGrid = GridBase::grid<PointDataGrid>((*grids)[0]);

        CPPUNIT_ASSERT(inputGrid);

        PointDataTree& inputTree = inputGrid->tree();

        std::vector<Index64> offsets;
        std::vector<Name> includeGroups;
        std::vector<Name> excludeGroups;

        MultiGroupFilter filter(includeGroups, excludeGroups, inputTree.cbeginLeaf()->attributeSet());
        Index64 total = pointOffsets(offsets, inputTree, filter, /*inCoreOnly=*/true);

#if OPENVDB_ABI_VERSION_NUMBER >= 3
        CPPUNIT_ASSERT_EQUAL(offsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(offsets[0], Index64(0));
        CPPUNIT_ASSERT_EQUAL(offsets[1], Index64(0));
        CPPUNIT_ASSERT_EQUAL(offsets[2], Index64(0));
        CPPUNIT_ASSERT_EQUAL(offsets[3], Index64(0));
        CPPUNIT_ASSERT_EQUAL(total, Index64(0));
#else
        CPPUNIT_ASSERT_EQUAL(offsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(offsets[0], Index64(1));
        CPPUNIT_ASSERT_EQUAL(offsets[1], Index64(3));
        CPPUNIT_ASSERT_EQUAL(offsets[2], Index64(4));
        CPPUNIT_ASSERT_EQUAL(offsets[3], Index64(5));
        CPPUNIT_ASSERT_EQUAL(total, Index64(5));
#endif

        offsets.clear();

        total = pointOffsets(offsets, inputTree, filter, /*inCoreOnly=*/false);

        CPPUNIT_ASSERT_EQUAL(offsets.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(offsets[0], Index64(1));
        CPPUNIT_ASSERT_EQUAL(offsets[1], Index64(3));
        CPPUNIT_ASSERT_EQUAL(offsets[2], Index64(4));
        CPPUNIT_ASSERT_EQUAL(offsets[3], Index64(5));
        CPPUNIT_ASSERT_EQUAL(total, Index64(5));
    }
    std::remove(filename.c_str());
}


namespace {

// sum all voxel values
template<typename GridT>
inline Index64
voxelSum(const GridT& grid)
{
    Index64 total = 0;
    for (auto iter = grid.cbeginValueOn(); iter; ++iter) {
        total += static_cast<Index64>(*iter);
    }
    return total;
}

// Generate random points by uniformly distributing points on a unit-sphere.
inline void
genPoints(std::vector<Vec3R>& positions, const int numPoints, const double scale)
{
    // init
    math::Random01 randNumber(0);
    const int n = int(std::sqrt(double(numPoints)));
    const double xScale = (2.0 * M_PI) / double(n);
    const double yScale = M_PI / double(n);

    double x, y, theta, phi;
    Vec3R pos;

    positions.reserve(n*n);

    // loop over a [0 to n) x [0 to n) grid.
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {

            // jitter, move to random pos. inside the current cell
            x = double(a) + randNumber();
            y = double(b) + randNumber();

            // remap to a lat/long map
            theta = y * yScale; // [0 to PI]
            phi   = x * xScale; // [0 to 2PI]

            // convert to cartesian coordinates on a unit sphere.
            // spherical coordinate triplet (r=1, theta, phi)
            pos[0] = static_cast<float>(std::sin(theta)*std::cos(phi)*scale);
            pos[1] = static_cast<float>(std::sin(theta)*std::sin(phi)*scale);
            pos[2] = static_cast<float>(std::cos(theta)*scale);

            positions.push_back(pos);
        }
    }
}

} // namespace


void
TestPointCount::testCountGrid()
{
    using namespace openvdb::math;

    { // five points
        std::vector<Vec3s> positions{   {1, 1, 1},
                                        {1, 101, 1},
                                        {2, 101, 1},
                                        {101, 1, 1},
                                        {101, 101, 1}};

        { // in five voxels

            math::Transform::Ptr transform(math::Transform::createLinearTransform(1.0f));
            PointDataGrid::Ptr points = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);

            // generate a count grid with the same transform

            Int32Grid::Ptr count = pointCountGrid(*points);

            CPPUNIT_ASSERT_EQUAL(count->activeVoxelCount(), points->activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(count->evalActiveVoxelBoundingBox(), points->evalActiveVoxelBoundingBox());
            CPPUNIT_ASSERT_EQUAL(voxelSum(*count), pointCount(points->tree()));
        }

        { // in four voxels

            math::Transform::Ptr transform(math::Transform::createLinearTransform(10.0f));
            PointDataGrid::Ptr points = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);

            // generate a count grid with the same transform

            Int32Grid::Ptr count = pointCountGrid(*points);

            CPPUNIT_ASSERT_EQUAL(count->activeVoxelCount(), points->activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(count->evalActiveVoxelBoundingBox(), points->evalActiveVoxelBoundingBox());
            CPPUNIT_ASSERT_EQUAL(voxelSum(*count), pointCount(points->tree()));
        }

        { // in one voxel

            math::Transform::Ptr transform(math::Transform::createLinearTransform(1000.0f));
            PointDataGrid::Ptr points = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);

            // generate a count grid with the same transform

            Int32Grid::Ptr count = pointCountGrid(*points);

            CPPUNIT_ASSERT_EQUAL(count->activeVoxelCount(), points->activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(count->evalActiveVoxelBoundingBox(), points->evalActiveVoxelBoundingBox());
            CPPUNIT_ASSERT_EQUAL(voxelSum(*count), pointCount(points->tree()));
        }

        { // in four voxels, Int64 grid

            math::Transform::Ptr transform(math::Transform::createLinearTransform(10.0f));
            PointDataGrid::Ptr points = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);

            // generate a count grid with the same transform

            Int64Grid::Ptr count = pointCountGrid<PointDataGrid, Int64Grid>(*points);

            CPPUNIT_ASSERT_EQUAL(count->activeVoxelCount(), points->activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(count->evalActiveVoxelBoundingBox(), points->evalActiveVoxelBoundingBox());
            CPPUNIT_ASSERT_EQUAL(voxelSum(*count), pointCount(points->tree()));
        }

        { // in four voxels, float grid

            math::Transform::Ptr transform(math::Transform::createLinearTransform(10.0f));
            PointDataGrid::Ptr points = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);

            // generate a count grid with the same transform

            FloatGrid::Ptr count = pointCountGrid<PointDataGrid, FloatGrid>(*points);

            CPPUNIT_ASSERT_EQUAL(count->activeVoxelCount(), points->activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(count->evalActiveVoxelBoundingBox(), points->evalActiveVoxelBoundingBox());
            CPPUNIT_ASSERT_EQUAL(voxelSum(*count), pointCount(points->tree()));
        }

        { // in four voxels

            math::Transform::Ptr transform(math::Transform::createLinearTransform(10.0f));
            const PointAttributeVector<Vec3s> pointList(positions);
            tools::PointIndexGrid::Ptr pointIndexGrid =
                tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

            PointDataGrid::Ptr points =
                    createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                                                                  pointList, *transform);

            auto& tree = points->tree();

            // assign point 3 to new group "test"

            appendGroup(tree, "test");

            std::vector<short> groups{0,0,1,0,0};

            setGroup(tree, pointIndexGrid->tree(), groups, "test");

            std::vector<std::string> includeGroups{"test"};
            std::vector<std::string> excludeGroups;

            // generate a count grid with the same transform

            MultiGroupFilter filter(includeGroups, excludeGroups,
                tree.cbeginLeaf()->attributeSet());
            Int32Grid::Ptr count = pointCountGrid(*points, filter);

            CPPUNIT_ASSERT_EQUAL(count->activeVoxelCount(), Index64(1));
            CPPUNIT_ASSERT_EQUAL(voxelSum(*count), Index64(1));

            MultiGroupFilter filter2(excludeGroups, includeGroups,
                tree.cbeginLeaf()->attributeSet());
            count = pointCountGrid(*points, filter2);

            CPPUNIT_ASSERT_EQUAL(count->activeVoxelCount(), Index64(4));
            CPPUNIT_ASSERT_EQUAL(voxelSum(*count), Index64(4));
        }
    }

    { // 40,000 points on a unit sphere
        std::vector<Vec3R> positions;
        const size_t total = 40000;
        genPoints(positions, total, /*scale=*/100.0);
        CPPUNIT_ASSERT_EQUAL(positions.size(), total);

        math::Transform::Ptr transform1(math::Transform::createLinearTransform(1.0f));
        math::Transform::Ptr transform5(math::Transform::createLinearTransform(5.0f));

        PointDataGrid::Ptr points1 =
            createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform1);
        PointDataGrid::Ptr points5 =
            createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform5);

        CPPUNIT_ASSERT(points1->activeVoxelCount() != points5->activeVoxelCount());
        CPPUNIT_ASSERT(points1->evalActiveVoxelBoundingBox() != points5->evalActiveVoxelBoundingBox());
        CPPUNIT_ASSERT_EQUAL(pointCount(points1->tree()), pointCount(points5->tree()));

        { // generate count grids with the same transform

            Int32Grid::Ptr count1 = pointCountGrid(*points1);

            CPPUNIT_ASSERT_EQUAL(count1->activeVoxelCount(), points1->activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(count1->evalActiveVoxelBoundingBox(), points1->evalActiveVoxelBoundingBox());
            CPPUNIT_ASSERT_EQUAL(voxelSum(*count1), pointCount(points1->tree()));

            Int32Grid::Ptr count5 = pointCountGrid(*points5);

            CPPUNIT_ASSERT_EQUAL(count5->activeVoxelCount(), points5->activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(count5->evalActiveVoxelBoundingBox(), points5->evalActiveVoxelBoundingBox());
            CPPUNIT_ASSERT_EQUAL(voxelSum(*count5), pointCount(points5->tree()));
        }

        { // generate count grids with differing transforms

            Int32Grid::Ptr count1 = pointCountGrid(*points5, *transform1);

            CPPUNIT_ASSERT_EQUAL(count1->activeVoxelCount(), points1->activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(count1->evalActiveVoxelBoundingBox(), points1->evalActiveVoxelBoundingBox());
            CPPUNIT_ASSERT_EQUAL(voxelSum(*count1), pointCount(points5->tree()));

            Int32Grid::Ptr count5 = pointCountGrid(*points1, *transform5);

            CPPUNIT_ASSERT_EQUAL(count5->activeVoxelCount(), points5->activeVoxelCount());
            CPPUNIT_ASSERT_EQUAL(count5->evalActiveVoxelBoundingBox(), points5->evalActiveVoxelBoundingBox());
            CPPUNIT_ASSERT_EQUAL(voxelSum(*count5), pointCount(points1->tree()));
        }
    }
}


CPPUNIT_TEST_SUITE_REGISTRATION(TestPointCount);

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
