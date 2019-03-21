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
#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointConversion.h>

#include <cstdio> // for std::remove()
#include <cstdlib> // for std::getenv()
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace openvdb;
using namespace openvdb::points;

class TestPointGroup: public CppUnit::TestCase
{
public:
    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointGroup);
    CPPUNIT_TEST(testDescriptor);
    CPPUNIT_TEST(testAppendDrop);
    CPPUNIT_TEST(testCompact);
    CPPUNIT_TEST(testSet);
    CPPUNIT_TEST(testFilter);

    CPPUNIT_TEST_SUITE_END();

    void testDescriptor();
    void testAppendDrop();
    void testCompact();
    void testSet();
    void testFilter();
}; // class TestPointGroup

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointGroup);

////////////////////////////////////////


class FirstFilter
{
public:
    static bool initialized() { return true; }

    static index::State state() { return index::PARTIAL; }
    template <typename LeafT>
    static index::State state(const LeafT&) { return index::PARTIAL; }

    template <typename LeafT> void reset(const LeafT&) { }

    template <typename IterT> bool valid(const IterT& iter) const
    {
        return *iter == 0;
    }
}; // class FirstFilter


////////////////////////////////////////


namespace {

    bool testStringVector(std::vector<Name>& input)
    {
        return input.size() == 0;
    }

    bool testStringVector(std::vector<Name>& input, const Name& name1)
    {
        if (input.size() != 1)  return false;
        if (input[0] != name1)  return false;
        return true;
    }

    bool testStringVector(std::vector<Name>& input, const Name& name1, const Name& name2)
    {
        if (input.size() != 2)  return false;
        if (input[0] != name1)  return false;
        if (input[1] != name2)  return false;
        return true;
    }

} // namespace


void
TestPointGroup::testDescriptor()
{
    // test missing groups deletion

    { // no groups, empty Descriptor
        std::vector<std::string> groups;
        AttributeSet::Descriptor descriptor;
        deleteMissingPointGroups(groups, descriptor);
        CPPUNIT_ASSERT(testStringVector(groups));
    }

    { // one group, empty Descriptor
        std::vector<std::string> groups{"group1"};
        AttributeSet::Descriptor descriptor;
        deleteMissingPointGroups(groups, descriptor);
        CPPUNIT_ASSERT(testStringVector(groups));
    }

    { // one group, Descriptor with same group
        std::vector<std::string> groups{"group1"};
        AttributeSet::Descriptor descriptor;
        descriptor.setGroup("group1", 0);
        deleteMissingPointGroups(groups, descriptor);
        CPPUNIT_ASSERT(testStringVector(groups, "group1"));
    }

    { // one group, Descriptor with different group
        std::vector<std::string> groups{"group1"};
        AttributeSet::Descriptor descriptor;
        descriptor.setGroup("group2", 0);
        deleteMissingPointGroups(groups, descriptor);
        CPPUNIT_ASSERT(testStringVector(groups));
    }

    { // three groups, Descriptor with three groups, one different
        std::vector<std::string> groups{"group1", "group3", "group4"};
        AttributeSet::Descriptor descriptor;
        descriptor.setGroup("group1", 0);
        descriptor.setGroup("group2", 0);
        descriptor.setGroup("group4", 0);
        deleteMissingPointGroups(groups, descriptor);
        CPPUNIT_ASSERT(testStringVector(groups, "group1", "group4"));
    }
}


////////////////////////////////////////


void
TestPointGroup::testAppendDrop()
{
    std::vector<Vec3s> positions{{1, 1, 1}, {1, 10, 1}, {10, 1, 1}, {10, 10, 1}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(4));

    // retrieve first and last leaf attribute sets

    PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();
    const AttributeSet& attributeSet = leafIter->attributeSet();

    ++leafIter;
    ++leafIter;
    ++leafIter;

    const AttributeSet& attributeSet4 = leafIter->attributeSet();

    { // throw on append or drop an empty group
        CPPUNIT_ASSERT_THROW(appendGroup(tree, ""), openvdb::KeyError);
        CPPUNIT_ASSERT_THROW(dropGroup(tree, ""), openvdb::KeyError);
    }

    { // append a group
        appendGroup(tree, "test");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(1));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test"));
    }

    { // append a group with non-unique name (repeat the append)
        appendGroup(tree, "test");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(1));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test"));
    }

    { // append multiple groups
        std::vector<Name> names{"test2", "test3"};

        appendGroups(tree, names);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(3));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test2"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test2"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test3"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test3"));
    }

    { // append to a copy
        PointDataTree tree2(tree);

        appendGroup(tree2, "copy1");

        CPPUNIT_ASSERT(!attributeSet.descriptor().hasGroup("copy1"));
        CPPUNIT_ASSERT(tree2.beginLeaf()->attributeSet().descriptor().hasGroup("copy1"));
    }

    { // drop a group
        dropGroup(tree, "test2");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(2));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test3"));
        CPPUNIT_ASSERT(attributeSet4.descriptor().hasGroup("test3"));
    }

    { // drop multiple groups
        std::vector<Name> names{"test", "test3"};

        dropGroups(tree, names);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(0));
    }

    { // drop a copy
        appendGroup(tree, "copy2");

        PointDataTree tree2(tree);

        dropGroup(tree2, "copy2");

        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("copy2"));
        CPPUNIT_ASSERT(!tree2.beginLeaf()->attributeSet().descriptor().hasGroup("copy2"));

        dropGroup(tree, "copy2");
    }

    { // set group membership
        appendGroup(tree, "test");

        setGroup(tree, "test", true);

        GroupFilter filter("test", tree.cbeginLeaf()->attributeSet());
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, filter), Index64(4));

        setGroup(tree, "test", false);

        CPPUNIT_ASSERT_EQUAL(pointCount(tree, filter), Index64(0));

        dropGroup(tree, "test");
    }

    { // drop all groups
        appendGroup(tree, "test");
        appendGroup(tree, "test2");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(2));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(1));

        dropGroups(tree);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(0));
    }

    { // check that newly added groups have empty group membership

        // recreate the grid with 3 points in one leaf

        positions = {{1, 1, 1}, {1, 2, 1}, {2, 1, 1}};
        grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
        PointDataTree& newTree = grid->tree();

        appendGroup(newTree, "test");

        // test that a completely new group (with a new group attribute)
        // has empty membership

        CPPUNIT_ASSERT(newTree.cbeginLeaf());
        GroupFilter filter("test", newTree.cbeginLeaf()->attributeSet());
        CPPUNIT_ASSERT_EQUAL(pointCount(newTree, filter), Index64(0));

        // check that membership in a group that was not created with a
        // new attribute array is still empty.
        // we will append a second group, set its membership, then
        // drop it and append a new group with the same name again

        appendGroup(newTree, "test2");

        PointDataTree::LeafIter leafIter2 = newTree.beginLeaf();
        CPPUNIT_ASSERT(leafIter2);

        GroupWriteHandle test2Handle = leafIter2->groupWriteHandle("test2");

        test2Handle.set(0, true);
        test2Handle.set(2, true);

        GroupFilter filter2("test2", newTree.cbeginLeaf()->attributeSet());
        CPPUNIT_ASSERT_EQUAL(pointCount(newTree, filter2), Index64(2));

        // drop and re-add group

        dropGroup(newTree, "test2");
        appendGroup(newTree, "test2");

        // check that group is fully cleared and does not have previously existing data

        CPPUNIT_ASSERT_EQUAL(pointCount(newTree, filter2), Index64(0));
    }

}


void
TestPointGroup::testCompact()
{
    std::vector<Vec3s> positions{{1, 1, 1}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(1));

    // retrieve first and last leaf attribute sets

    PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();
    const AttributeSet& attributeSet = leafIter->attributeSet();

    std::stringstream ss;

    { // append nine groups
        for (int i = 0; i < 8; i++) {
            ss.str("");
            ss << "test" << i;
            appendGroup(tree, ss.str());
        }

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(8));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(1));

        appendGroup(tree, "test8");

        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test0"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test7"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test8"));

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(9));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(2));
    }

    { // drop first attribute then compact
        dropGroup(tree, "test5", /*compact=*/false);

        CPPUNIT_ASSERT(!attributeSet.descriptor().hasGroup("test5"));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(8));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(2));

        compactGroups(tree);

        CPPUNIT_ASSERT(!attributeSet.descriptor().hasGroup("test5"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test7"));
        CPPUNIT_ASSERT(attributeSet.descriptor().hasGroup("test8"));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(8));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(1));
    }

    { // append seventeen groups, drop most of them, then compact
        for (int i = 0; i < 17; i++) {
            ss.str("");
            ss << "test" << i;
            appendGroup(tree, ss.str());
        }

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(17));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(3));

        // delete all but 0, 5, 9, 15

        for (int i = 0; i < 17; i++) {
            if (i == 0 || i == 5 || i == 9 || i == 15)  continue;
            ss.str("");
            ss << "test" << i;
            dropGroup(tree, ss.str(), /*compact=*/false);
        }

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(3));

        // make a copy

        PointDataTree tree2(tree);

        // compact - should now occupy one attribute

        compactGroups(tree);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().groupMap().size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().count(GroupAttributeArray::attributeType()), size_t(1));

        // check descriptor has been deep copied

        CPPUNIT_ASSERT_EQUAL(tree2.cbeginLeaf()->attributeSet().descriptor().groupMap().size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(tree2.cbeginLeaf()->attributeSet().descriptor().count(GroupAttributeArray::attributeType()), size_t(3));
    }
}


void
TestPointGroup::testSet()
{
    // four points in the same leaf

    std::vector<Vec3s> positions =  {
                                        {1, 1, 1},
                                        {1, 2, 1},
                                        {2, 1, 1},
                                        {2, 2, 1},
                                        {100, 100, 100},
                                        {100, 101, 100}
                                    };

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    const PointAttributeVector<Vec3s> pointList(positions);

    openvdb::tools::PointIndexGrid::Ptr pointIndexGrid =
        openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);
    PointDataTree& tree = grid->tree();

    appendGroup(tree, "test");

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(6));
    GroupFilter filter("test", tree.cbeginLeaf()->attributeSet());
    CPPUNIT_ASSERT_EQUAL(pointCount(tree, filter), Index64(0));

    std::vector<short> membership{1, 0, 1, 1, 0, 1};

    // copy tree for descriptor sharing test

    PointDataTree tree2(tree);

    setGroup(tree, pointIndexGrid->tree(), membership, "test");

    // check that descriptor remains shared

    appendGroup(tree2, "copy1");

    CPPUNIT_ASSERT(!tree.cbeginLeaf()->attributeSet().descriptor().hasGroup("copy1"));

    dropGroup(tree2, "copy1");

    CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(6));
    GroupFilter filter2("test", tree.cbeginLeaf()->attributeSet());
    CPPUNIT_ASSERT_EQUAL(pointCount(tree, filter2), Index64(4));

    { // IO
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

        // read test groups
        {
            io::File fileIn(filename);
            fileIn.open();

            GridPtrVecPtr grids = fileIn.getGrids();

            fileIn.close();

            CPPUNIT_ASSERT_EQUAL(grids->size(), size_t(1));

            PointDataGrid::Ptr inputGrid = GridBase::grid<PointDataGrid>((*grids)[0]);
            PointDataTree& treex = inputGrid->tree();

            CPPUNIT_ASSERT(treex.cbeginLeaf());

            const PointDataGrid::TreeType::LeafNodeType& leaf = *treex.cbeginLeaf();

            const AttributeSet::Descriptor& descriptor = leaf.attributeSet().descriptor();

            CPPUNIT_ASSERT(descriptor.hasGroup("test"));
            CPPUNIT_ASSERT_EQUAL(descriptor.groupMap().size(), size_t(1));

            CPPUNIT_ASSERT_EQUAL(pointCount(treex), Index64(6));
            GroupFilter filter3("test", leaf.attributeSet());
            CPPUNIT_ASSERT_EQUAL(pointCount(treex, filter3), Index64(4));
        }
        std::remove(filename.c_str());
    }
}


void
TestPointGroup::testFilter()
{
    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));
    PointDataGrid::Ptr grid;

    { // four points in the same leaf
        std::vector<Vec3s> positions =  {
                                            {1, 1, 1},
                                            {1, 2, 1},
                                            {2, 1, 1},
                                            {2, 2, 1},
                                            {100, 100, 100},
                                            {100, 101, 100}
                                        };

        const PointAttributeVector<Vec3s> pointList(positions);

        openvdb::tools::PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(pointList, *transform);

        grid = createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);
    }

    PointDataTree& tree = grid->tree();

    { // first point filter
        appendGroup(tree, "first");

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(6));
        GroupFilter filter("first", tree.cbeginLeaf()->attributeSet());
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, filter), Index64(0));

        FirstFilter filter2;

        setGroupByFilter<PointDataTree, FirstFilter>(tree, "first", filter2);

        auto iter = tree.cbeginLeaf();

        for ( ; iter; ++iter) {
            CPPUNIT_ASSERT_EQUAL(iter->groupPointCount("first"), Index64(1));
        }

        GroupFilter filter3("first", tree.cbeginLeaf()->attributeSet());
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, filter3), Index64(2));
    }

    const openvdb::BBoxd bbox(openvdb::Vec3d(0, 1.5, 0), openvdb::Vec3d(101, 100.5, 101));

    { // bbox filter
        appendGroup(tree, "bbox");

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(6));
        GroupFilter filter("bbox", tree.cbeginLeaf()->attributeSet());
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, filter), Index64(0));

        BBoxFilter filter2(*transform, bbox);

        setGroupByFilter<PointDataTree, BBoxFilter>(tree, "bbox", filter2);

        GroupFilter filter3("bbox", tree.cbeginLeaf()->attributeSet());
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, filter3), Index64(3));
    }

    { // first point filter and bbox filter (intersection of the above two filters)
        appendGroup(tree, "first_bbox");

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(6));
        GroupFilter filter("first_bbox", tree.cbeginLeaf()->attributeSet());
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, filter), Index64(0));

        using FirstBBoxFilter = BinaryFilter<FirstFilter, BBoxFilter>;

        FirstFilter firstFilter;
        BBoxFilter bboxFilter(*transform, bbox);
        FirstBBoxFilter filter2(firstFilter, bboxFilter);

        setGroupByFilter<PointDataTree, FirstBBoxFilter>(tree, "first_bbox", filter2);

        GroupFilter filter3("first_bbox", tree.cbeginLeaf()->attributeSet());
        CPPUNIT_ASSERT_EQUAL(pointCount(tree, filter3), Index64(1));

        std::vector<Vec3f> positions;

        for (auto iter = tree.cbeginLeaf(); iter; ++iter) {
            GroupFilter filterx("first_bbox", iter->attributeSet());
            auto filterIndexIter = iter->beginIndexOn(filterx);

            auto handle = AttributeHandle<Vec3f>::create(iter->attributeArray("P"));

            for ( ; filterIndexIter; ++filterIndexIter) {
                const openvdb::Coord ijk = filterIndexIter.getCoord();
                positions.push_back(handle->get(*filterIndexIter) + ijk.asVec3d());
            }
        }

        CPPUNIT_ASSERT_EQUAL(positions.size(), size_t(1));
        CPPUNIT_ASSERT_EQUAL(positions[0], Vec3f(100, 100, 100));
    }

    { // add 1000 points in three leafs (positions aren't important)

        std::vector<Vec3s> positions(1000, {1, 1, 1});
        positions.insert(positions.end(), 1000, {1, 1, 9});
        positions.insert(positions.end(), 1000, {9, 9, 9});

        const PointAttributeVector<Vec3s> pointList(positions);

        openvdb::tools::PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(pointList, *transform);

        grid = createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);

        PointDataTree& newTree = grid->tree();

        CPPUNIT_ASSERT_EQUAL(pointCount(newTree), Index64(3000));

        // random - maximum

        appendGroup(newTree, "random_maximum");

        const Index64 target = 1001;

        setGroupByRandomTarget(newTree, "random_maximum", target);

        GroupFilter filter("random_maximum", newTree.cbeginLeaf()->attributeSet());
        CPPUNIT_ASSERT_EQUAL(pointCount(newTree, filter), target);

        // random - percentage

        appendGroup(newTree, "random_percentage");

        setGroupByRandomPercentage(newTree, "random_percentage", 33.333333f);

        GroupFilter filter2("random_percentage", newTree.cbeginLeaf()->attributeSet());
        CPPUNIT_ASSERT_EQUAL(pointCount(newTree, filter2), Index64(1000));
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
