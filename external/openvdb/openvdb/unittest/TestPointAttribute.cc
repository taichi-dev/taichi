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
#include <openvdb/points/AttributeArrayString.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointConversion.h>
#include <vector>

using namespace openvdb;
using namespace openvdb::points;

class TestPointAttribute: public CppUnit::TestCase
{
public:
    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointAttribute);
    CPPUNIT_TEST(testAppendDrop);
    CPPUNIT_TEST(testRename);
    CPPUNIT_TEST(testBloscCompress);

    CPPUNIT_TEST_SUITE_END();

    void testAppendDrop();
    void testRename();
    void testBloscCompress();
}; // class TestPointAttribute

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointAttribute);

////////////////////////////////////////


void
TestPointAttribute::testAppendDrop()
{
    using AttributeI = TypedAttributeArray<int>;

    std::vector<Vec3s> positions{{1, 1, 1}, {1, 10, 1}, {10, 1, 1}, {10, 10, 1}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(4));

    // retrieve first and last leaf attribute sets

    auto leafIter = tree.cbeginLeaf();
    const AttributeSet& attributeSet = leafIter->attributeSet();

    ++leafIter;
    ++leafIter;
    ++leafIter;

    const AttributeSet& attributeSet4 = leafIter->attributeSet();

    // check just one attribute exists (position)
    CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(1));

    { // append an attribute, different initial values and collapse
        appendAttribute<int>(tree,  "id");

        CPPUNIT_ASSERT(tree.beginLeaf()->hasAttribute("id"));

        AttributeArray& array = tree.beginLeaf()->attributeArray("id");
        CPPUNIT_ASSERT(array.isUniform());
        CPPUNIT_ASSERT_EQUAL(AttributeI::cast(array).get(0), zeroVal<AttributeI::ValueType>());

        dropAttribute(tree, "id");

        appendAttribute<int>(tree, "id", 10, /*stride*/1);

        CPPUNIT_ASSERT(tree.beginLeaf()->hasAttribute("id"));

        AttributeArray& array2 = tree.beginLeaf()->attributeArray("id");
        CPPUNIT_ASSERT(array2.isUniform());
        CPPUNIT_ASSERT_EQUAL(AttributeI::cast(array2).get(0), AttributeI::ValueType(10));

        array2.expand();
        CPPUNIT_ASSERT(!array2.isUniform());

        collapseAttribute<int>(tree, "id", 50);

        AttributeArray& array3 = tree.beginLeaf()->attributeArray("id");
        CPPUNIT_ASSERT(array3.isUniform());
        CPPUNIT_ASSERT_EQUAL(AttributeI::cast(array3).get(0), AttributeI::ValueType(50));

        dropAttribute(tree, "id");

        appendAttribute<Name>(tree, "name", "test");

        AttributeArray& array4 = tree.beginLeaf()->attributeArray("name");
        CPPUNIT_ASSERT(array4.isUniform());
        StringAttributeHandle handle(array4, attributeSet.descriptor().getMetadata());
        CPPUNIT_ASSERT_EQUAL(handle.get(0), Name("test"));

        dropAttribute(tree, "name");
    }

    { // append a strided attribute
        appendAttribute<int>(tree, "id", 0, /*stride=*/1);

        AttributeArray& array = tree.beginLeaf()->attributeArray("id");
        CPPUNIT_ASSERT_EQUAL(array.stride(), Index(1));

        dropAttribute(tree, "id");

        appendAttribute<int>(tree, "id", 0, /*stride=*/10);

        CPPUNIT_ASSERT(tree.beginLeaf()->hasAttribute("id"));

        AttributeArray& array2 = tree.beginLeaf()->attributeArray("id");
        CPPUNIT_ASSERT_EQUAL(array2.stride(), Index(10));

        dropAttribute(tree, "id");
    }

    { // append an attribute, check descriptors are as expected, default value test
        appendAttribute<int>(tree,  "id",
                                /*uniformValue*/0,
                                /*stride=*/1,
                                /*constantStride=*/true,
                                /*defaultValue*/TypedMetadata<int>(10).copy(),
                                /*hidden=*/false, /*transient=*/false);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(2));
        CPPUNIT_ASSERT(attributeSet.descriptor() == attributeSet4.descriptor());
        CPPUNIT_ASSERT(&attributeSet.descriptor() == &attributeSet4.descriptor());

        CPPUNIT_ASSERT(attributeSet.descriptor().getMetadata()["default:id"]);
    }

    { // append three attributes, check ordering is consistent with insertion
        appendAttribute<float>(tree, "test3");
        appendAttribute<float>(tree, "test1");
        appendAttribute<float>(tree, "test2");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(5));

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("id"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test3"), size_t(2));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test1"), size_t(3));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test2"), size_t(4));
    }

    { // drop an attribute by index, check ordering remains consistent
        std::vector<size_t> indices{2};

        dropAttributes(tree, indices);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(4));

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("id"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test1"), size_t(2));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test2"), size_t(3));
    }

    { // drop attributes by index, check ordering remains consistent
        std::vector<size_t> indices{1, 3};

        dropAttributes(tree, indices);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(2));

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test1"), size_t(1));
    }

    { // drop last non-position attribute
        std::vector<size_t> indices{1};

        dropAttributes(tree, indices);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(1));
    }

    { // attempt (and fail) to drop position
        std::vector<size_t> indices{0};

        CPPUNIT_ASSERT_THROW(dropAttributes(tree, indices), openvdb::KeyError);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(1));
        CPPUNIT_ASSERT(attributeSet.descriptor().find("P") != AttributeSet::INVALID_POS);
    }

    { // add back previous attributes
        appendAttribute<int>(tree, "id");
        appendAttribute<float>(tree, "test3");
        appendAttribute<float>(tree, "test1");
        appendAttribute<float>(tree, "test2");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(5));
    }

    { // attempt (and fail) to drop non-existing attribute
        std::vector<Name> names{"test1000"};

        CPPUNIT_ASSERT_THROW(dropAttributes(tree, names), openvdb::KeyError);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(5));
    }

    { // drop by name
        std::vector<Name> names{"test1", "test2"};

        dropAttributes(tree, names);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(3));
        CPPUNIT_ASSERT(attributeSet.descriptor() == attributeSet4.descriptor());
        CPPUNIT_ASSERT(&attributeSet.descriptor() == &attributeSet4.descriptor());

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("id"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("test3"), size_t(2));
    }

    { // attempt (and fail) to drop position
        std::vector<Name> names{"P"};

        CPPUNIT_ASSERT_THROW(dropAttributes(tree, names), openvdb::KeyError);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(3));
        CPPUNIT_ASSERT(attributeSet.descriptor().find("P") != AttributeSet::INVALID_POS);
    }

    { // drop one attribute by name
        dropAttribute(tree, "test3");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(2));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("id"), size_t(1));
    }

    { // drop one attribute by id
        dropAttribute(tree, 1);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(1));
        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().find("P"), size_t(0));
    }

    { // attempt to add an attribute with a name that already exists
        appendAttribute<float>(tree, "test3");
        CPPUNIT_ASSERT_THROW(appendAttribute<float>(tree, "test3"), openvdb::KeyError);

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(2));
    }

    { // attempt to add an attribute with an unregistered type (Vec2R)
        CPPUNIT_ASSERT_THROW(appendAttribute<Vec2R>(tree, "unregistered"), openvdb::KeyError);
    }

    { // append attributes marked as hidden, transient, group and string
        appendAttribute<float>(tree, "testHidden", 0,
            /*stride=*/1, /*constantStride=*/true, Metadata::Ptr(), true, false);
        appendAttribute<float>(tree, "testTransient", 0,
            /*stride=*/1, /*constantStride=*/true, Metadata::Ptr(), false, true);
        appendAttribute<Name>(tree, "testString", "",
            /*stride=*/1, /*constantStride=*/true, Metadata::Ptr(), false, false);

        const AttributeArray& arrayHidden = leafIter->attributeArray("testHidden");
        const AttributeArray& arrayTransient = leafIter->attributeArray("testTransient");
        const AttributeArray& arrayString = leafIter->attributeArray("testString");

        CPPUNIT_ASSERT(arrayHidden.isHidden());
        CPPUNIT_ASSERT(!arrayTransient.isHidden());

        CPPUNIT_ASSERT(!arrayHidden.isTransient());
        CPPUNIT_ASSERT(arrayTransient.isTransient());
        CPPUNIT_ASSERT(!arrayString.isTransient());

        CPPUNIT_ASSERT(!isGroup(arrayHidden));
        CPPUNIT_ASSERT(!isGroup(arrayTransient));
        CPPUNIT_ASSERT(!isGroup(arrayString));

        CPPUNIT_ASSERT(!isString(arrayHidden));
        CPPUNIT_ASSERT(!isString(arrayTransient));
        CPPUNIT_ASSERT(isString(arrayString));
    }

    { // collapsing non-existing attribute throws exception
        CPPUNIT_ASSERT_THROW(collapseAttribute<int>(tree, "unknown", 0), openvdb::KeyError);
        CPPUNIT_ASSERT_THROW(collapseAttribute<Name>(tree, "unknown", "unknown"), openvdb::KeyError);
    }
}

void
TestPointAttribute::testRename()
{
    std::vector<Vec3s> positions{{1, 1, 1}, {1, 10, 1}, {10, 1, 1}, {10, 10, 1}};

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check one leaf per point
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(4));

    const openvdb::TypedMetadata<float> defaultValue(5.0f);

    appendAttribute<float>(tree, "test1", 0,
        /*stride=*/1, /*constantStride=*/true, defaultValue.copy());
    appendAttribute<int>(tree, "id");
    appendAttribute<float>(tree, "test2");

    // retrieve first and last leaf attribute sets

    auto leafIter = tree.cbeginLeaf();
    const AttributeSet& attributeSet = leafIter->attributeSet();
    ++leafIter;
    const AttributeSet& attributeSet4 = leafIter->attributeSet();

    { // rename one attribute
        renameAttribute(tree, "test1", "test1renamed");

        CPPUNIT_ASSERT_EQUAL(attributeSet.descriptor().size(), size_t(4));
        CPPUNIT_ASSERT(attributeSet.descriptor().find("test1") == AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(attributeSet.descriptor().find("test1renamed") != AttributeSet::INVALID_POS);

        CPPUNIT_ASSERT_EQUAL(attributeSet4.descriptor().size(), size_t(4));
        CPPUNIT_ASSERT(attributeSet4.descriptor().find("test1") == AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(attributeSet4.descriptor().find("test1renamed") != AttributeSet::INVALID_POS);

        renameAttribute(tree, "test1renamed", "test1");
    }

    { // rename non-existing, matching and existing attributes
        CPPUNIT_ASSERT_THROW(renameAttribute(tree, "nonexist", "newname"), openvdb::KeyError);
        CPPUNIT_ASSERT_THROW(renameAttribute(tree, "test1", "test1"), openvdb::KeyError);
        CPPUNIT_ASSERT_THROW(renameAttribute(tree, "test2", "test1"), openvdb::KeyError);
    }

    { // rename multiple attributes
        std::vector<Name> oldNames{"test1", "test2"};
        std::vector<Name> newNames{"test1renamed"};

        CPPUNIT_ASSERT_THROW(renameAttributes(tree, oldNames, newNames), openvdb::ValueError);

        newNames.push_back("test2renamed");
        renameAttributes(tree, oldNames, newNames);

        renameAttribute(tree, "test1renamed", "test1");
        renameAttribute(tree, "test2renamed", "test2");
    }

    { // rename an attribute with a default value
        CPPUNIT_ASSERT(attributeSet.descriptor().hasDefaultValue("test1"));

        renameAttribute(tree, "test1", "test1renamed");

        CPPUNIT_ASSERT(attributeSet.descriptor().hasDefaultValue("test1renamed"));
    }
}

void
TestPointAttribute::testBloscCompress()
{
    std::vector<Vec3s> positions;
    for (float i = 1.f; i < 6.f; i += 0.1f) {
        positions.emplace_back(1, i, 1);
        positions.emplace_back(1, 1, i);
        positions.emplace_back(10, i, 1);
        positions.emplace_back(10, 1, i);
    }

    const float voxelSize(1.0);
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    PointDataGrid::Ptr grid = createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
    PointDataTree& tree = grid->tree();

    // check two leaves
    CPPUNIT_ASSERT_EQUAL(tree.leafCount(), Index32(2));

    // retrieve first and last leaf attribute sets

    auto leafIter = tree.beginLeaf();
    auto leafIter2 = ++tree.beginLeaf();

    { // append an attribute, check descriptors are as expected
        appendAttribute<int>(tree, "compact");
        appendAttribute<int>(tree, "id");
        appendAttribute<int>(tree, "id2");
    }

    using AttributeHandleRWI = AttributeWriteHandle<int>;

    { // set some id values (leaf 1)
        AttributeHandleRWI handleCompact(leafIter->attributeArray("compact"));
        AttributeHandleRWI handleId(leafIter->attributeArray("id"));
        AttributeHandleRWI handleId2(leafIter->attributeArray("id2"));

        const int size = leafIter->attributeArray("id").size();

        CPPUNIT_ASSERT_EQUAL(size, 102);

        for (int i = 0; i < size; i++) {
            handleCompact.set(i, 5);
            handleId.set(i, i);
            handleId2.set(i, i);
        }
    }

    { // set some id values (leaf 2)
        AttributeHandleRWI handleCompact(leafIter2->attributeArray("compact"));
        AttributeHandleRWI handleId(leafIter2->attributeArray("id"));
        AttributeHandleRWI handleId2(leafIter2->attributeArray("id2"));

        const int size = leafIter2->attributeArray("id").size();

        CPPUNIT_ASSERT_EQUAL(size, 102);

        for (int i = 0; i < size; i++) {
            handleCompact.set(i, 10);
            handleId.set(i, i);
            handleId2.set(i, i);
        }
    }

    compactAttributes(tree);

    CPPUNIT_ASSERT(leafIter->attributeArray("compact").isUniform());
    CPPUNIT_ASSERT(leafIter2->attributeArray("compact").isUniform());

// disable deprecated warnings for in-memory compression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

    bloscCompressAttribute(tree, "id");

#ifdef OPENVDB_USE_BLOSC
    CPPUNIT_ASSERT(!leafIter->attributeArray("id").isCompressed());
    CPPUNIT_ASSERT(!leafIter->attributeArray("id2").isCompressed());
    CPPUNIT_ASSERT(!leafIter2->attributeArray("id").isCompressed());
    CPPUNIT_ASSERT(!leafIter2->attributeArray("id2").isCompressed());
#endif

#pragma GCC diagnostic pop
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
