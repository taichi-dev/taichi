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
#include <openvdb/points/PointDelete.h>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace openvdb::points;

class TestPointDelete: public CppUnit::TestCase
{
public:
    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointDelete);
    CPPUNIT_TEST(testDeleteFromGroups);

    CPPUNIT_TEST_SUITE_END();

    void testDeleteFromGroups();
}; // class TestPointDelete

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointDelete);

////////////////////////////////////////

void
TestPointDelete::testDeleteFromGroups()
{
    using openvdb::math::Vec3s;
    using openvdb::tools::PointIndexGrid;
    using openvdb::Index64;

    const float voxelSize(1.0);
    openvdb::math::Transform::Ptr transform(openvdb::math::Transform::createLinearTransform(voxelSize));

    const std::vector<Vec3s> positions6Points =  {
                                                {1, 1, 1},
                                                {1, 2, 1},
                                                {2, 1, 1},
                                                {2, 2, 1},
                                                {100, 100, 100},
                                                {100, 101, 100}
                                           };
    const PointAttributeVector<Vec3s> pointList6Points(positions6Points);

    {
        // delete from a tree with 2 leaves, checking that group membership is updated as
        // expected

        PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList6Points, *transform);

        PointDataGrid::Ptr grid =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList6Points,
                *transform);
        PointDataTree& tree = grid->tree();

        // first test will delete 3 groups, with the third one empty.

        appendGroup(tree, "test1");
        appendGroup(tree, "test2");
        appendGroup(tree, "test3");
        appendGroup(tree, "test4");

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(6));

        std::vector<short> membership1{1, 0, 0, 0, 0, 1};

        setGroup(tree, pointIndexGrid->tree(), membership1, "test1");

        std::vector<short> membership2{0, 0, 1, 1, 0, 1};

        setGroup(tree, pointIndexGrid->tree(), membership2, "test2");

        std::vector<std::string> groupsToDelete{"test1", "test2", "test3"};

        deleteFromGroups(tree, groupsToDelete);

        // 4 points should have been deleted, so only 2 remain
        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(2));

        // check that first three groups are deleted but the last is not

        const PointDataTree::LeafCIter leafIterAfterDeletion = tree.cbeginLeaf();

        AttributeSet attributeSetAfterDeletion = leafIterAfterDeletion->attributeSet();
        AttributeSet::Descriptor& descriptor = attributeSetAfterDeletion.descriptor();

        CPPUNIT_ASSERT(!descriptor.hasGroup("test1"));
        CPPUNIT_ASSERT(!descriptor.hasGroup("test2"));
        CPPUNIT_ASSERT(!descriptor.hasGroup("test3"));
        CPPUNIT_ASSERT(descriptor.hasGroup("test4"));
    }

    {
        // check deletion from a single leaf tree and that attribute values are preserved
        // correctly after deletion

        std::vector<Vec3s> positions4Points = {
                                                {1, 1, 1},
                                                {1, 2, 1},
                                                {2, 1, 1},
                                                {2, 2, 1},
                                              };

        const PointAttributeVector<Vec3s> pointList4Points(positions4Points);

        PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList4Points, *transform);

        PointDataGrid::Ptr grid =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                pointList4Points, *transform);
        PointDataTree& tree = grid->tree();

        appendGroup(tree, "test");
        appendAttribute(tree, "testAttribute", TypedAttributeArray<int32_t>::attributeType());

        CPPUNIT_ASSERT(tree.beginLeaf());

        const PointDataTree::LeafIter leafIter = tree.beginLeaf();

        AttributeWriteHandle<int>
            testAttributeWriteHandle(leafIter->attributeArray("testAttribute"));

        for(int i = 0; i < 4; i++) {
            testAttributeWriteHandle.set(i,i+1);
        }

        std::vector<short> membership{0, 1, 1, 0};
        setGroup(tree, pointIndexGrid->tree(), membership, "test");

        deleteFromGroup(tree, "test");

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(2));

        const PointDataTree::LeafCIter leafIterAfterDeletion = tree.cbeginLeaf();
        const AttributeSet attributeSetAfterDeletion = leafIterAfterDeletion->attributeSet();
        const AttributeSet::Descriptor& descriptor = attributeSetAfterDeletion.descriptor();

        CPPUNIT_ASSERT(descriptor.find("testAttribute") != AttributeSet::INVALID_POS);

        AttributeHandle<int> testAttributeHandle(*attributeSetAfterDeletion.get("testAttribute"));

        CPPUNIT_ASSERT_EQUAL(1, testAttributeHandle.get(0));
        CPPUNIT_ASSERT_EQUAL(4, testAttributeHandle.get(1));
    }

    {
        // test the invert flag using data similar to that used in the first test

        PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList6Points, *transform);
        PointDataGrid::Ptr grid =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList6Points,
                *transform);
        PointDataTree& tree = grid->tree();

        appendGroup(tree, "test1");
        appendGroup(tree, "test2");
        appendGroup(tree, "test3");
        appendGroup(tree, "test4");

        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(6));

        std::vector<short> membership1{1, 0, 1, 1, 0, 1};

        setGroup(tree, pointIndexGrid->tree(), membership1, "test1");

        std::vector<short> membership2{0, 0, 1, 1, 0, 1};

        setGroup(tree, pointIndexGrid->tree(), membership2, "test2");

        std::vector<std::string> groupsToDelete{"test1", "test3"};

        deleteFromGroups(tree, groupsToDelete, /*invert=*/ true);

        const PointDataTree::LeafCIter leafIterAfterDeletion = tree.cbeginLeaf();
        const AttributeSet attributeSetAfterDeletion = leafIterAfterDeletion->attributeSet();
        const AttributeSet::Descriptor& descriptor = attributeSetAfterDeletion.descriptor();

        // no groups should be dropped when invert = true
        CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(descriptor.groupMap().size()),
                             static_cast<size_t>(4));

        // 4 points should remain since test1 and test3 have 4 members between then
        CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(pointCount(tree)),
                             static_cast<size_t>(4));
    }

    {
        // similar to first test, but don't drop groups

        PointIndexGrid::Ptr pointIndexGrid =
            openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList6Points, *transform);

        PointDataGrid::Ptr grid =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList6Points,
                *transform);
        PointDataTree& tree = grid->tree();

        // first test will delete 3 groups, with the third one empty.

        appendGroup(tree, "test1");
        appendGroup(tree, "test2");
        appendGroup(tree, "test3");
        appendGroup(tree, "test4");

        std::vector<short> membership1{1, 0, 0, 0, 0, 1};

        setGroup(tree, pointIndexGrid->tree(), membership1, "test1");

        std::vector<short> membership2{0, 0, 1, 1, 0, 1};

        setGroup(tree, pointIndexGrid->tree(), membership2, "test2");

        std::vector<std::string> groupsToDelete{"test1", "test2", "test3"};

        deleteFromGroups(tree, groupsToDelete, /*invert=*/ false, /*drop=*/ false);

        // 4 points should have been deleted, so only 2 remain
        CPPUNIT_ASSERT_EQUAL(pointCount(tree), Index64(2));

        // check that first three groups are deleted but the last is not

        const PointDataTree::LeafCIter leafIterAfterDeletion = tree.cbeginLeaf();

        AttributeSet attributeSetAfterDeletion = leafIterAfterDeletion->attributeSet();
        AttributeSet::Descriptor& descriptor = attributeSetAfterDeletion.descriptor();

        // all group should still be present

        CPPUNIT_ASSERT(descriptor.hasGroup("test1"));
        CPPUNIT_ASSERT(descriptor.hasGroup("test2"));
        CPPUNIT_ASSERT(descriptor.hasGroup("test3"));
        CPPUNIT_ASSERT(descriptor.hasGroup("test4"));
    }

}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
