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
#include <openvdb/openvdb.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointSample.h>
#include "util.h"
#include <string>
#include <vector>

using namespace openvdb;

class TestPointSample: public CppUnit::TestCase
{
public:

    void setUp() override { initialize(); }
    void tearDown() override { uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointSample);
    CPPUNIT_TEST(testPointSample);
    CPPUNIT_TEST(testPointSampleWithGroups);

    CPPUNIT_TEST_SUITE_END();

    void testPointSample();
    void testPointSampleWithGroups();

}; // class TestPointSample

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointSample);


namespace
{

/// Utility function to quickly create a very simple grid (with specified value type), set a value
/// at its origin and then create and sample to an attribute
///
template <typename ValueType>
typename points::AttributeHandle<ValueType>::Ptr
testAttribute(points::PointDataGrid& points, const std::string& attributeName,
              const math::Transform::Ptr xform, const ValueType& val)
{
    using TreeT = typename tree::Tree4<ValueType, 5, 4, 3>::Type;
    using GridT = Grid<TreeT>;

    typename GridT::Ptr grid = GridT::create();

    grid->setTransform(xform);
    grid->tree().setValue(Coord(0,0,0), val);

    points::boxSample(points, *grid, attributeName);

    return(points::AttributeHandle<ValueType>::create(
        points.tree().cbeginLeaf()->attributeArray(attributeName)));
}

} // anonymous namespace


void
TestPointSample::testPointSample()
{
    using points::PointDataGrid;
    using points::NullCodec;

    const float voxelSize = 0.1f;
    math::Transform::Ptr transform(math::Transform::createLinearTransform(voxelSize));

    {
        // check that all supported grid types can be sampled.
        // This check will use very basic grids with a point at a cell-centered positions

        // create test point grid with a single point

        std::vector<Vec3f> pointPositions{Vec3f(0.0f, 0.0f, 0.0f)};
        PointDataGrid::Ptr points = points::createPointDataGrid<NullCodec, PointDataGrid, Vec3f>(
            pointPositions, *transform);

        CPPUNIT_ASSERT(points);

        // bool

        points::AttributeHandle<bool>::Ptr boolHandle =
            testAttribute<bool>(*points, "test_bool", transform, true);

        CPPUNIT_ASSERT(boolHandle->get(0));

        // int16

#if (defined _MSC_VER) || (defined __INTEL_COMPILER) || (defined __clang__)
        // GCC warns warns of narrowing conversions from int to int16_t,
        // and GCC 4.8, at least, ignores the -Wconversion suppression pragma.
        // So for now, skip this test if compiling with GCC.
        points::AttributeHandle<int16_t>::Ptr int16Handle =
            testAttribute<int16_t>(*points, "test_int16", transform, int16_t(10));

        CPPUNIT_ASSERT_EQUAL(int16Handle->get(0), int16_t(10));
#endif

        // int32

        points::AttributeHandle<Int32>::Ptr int32Handle =
            testAttribute<Int32>(*points, "test_Int32", transform, Int32(3));

        CPPUNIT_ASSERT_EQUAL(Int32(3), int32Handle->get(0));

        // int64

        points::AttributeHandle<Int64>::Ptr int64Handle =
            testAttribute<Int64>(*points, "test_Int64", transform, Int64(2));

        CPPUNIT_ASSERT_EQUAL(Int64(2), int64Handle->get(0));

        // double

        points::AttributeHandle<double>::Ptr doubleHandle =
            testAttribute<double>(*points, "test_double", transform, 4.0);

        CPPUNIT_ASSERT_EQUAL(4.0, doubleHandle->get(0));

        // Vec3i

        points::AttributeHandle<math::Vec3i>::Ptr vec3iHandle =
            testAttribute<Vec3i>(*points, "test_vec3i", transform, math::Vec3i(9, 8, 7));

        CPPUNIT_ASSERT_EQUAL(vec3iHandle->get(0), math::Vec3i(9, 8, 7));

        // Vec3f

        points::AttributeHandle<Vec3f>::Ptr vec3fHandle =
            testAttribute<Vec3f>(*points, "test_vec3f", transform, Vec3f(111.0f, 222.0f, 333.0f));

        CPPUNIT_ASSERT_EQUAL(vec3fHandle->get(0), Vec3f(111.0f, 222.0f, 333.0f));

        // Vec3d

        points::AttributeHandle<Vec3d>::Ptr vec3dHandle =
            testAttribute<Vec3d>(*points, "test_vec3d", transform, Vec3d(1.0, 2.0, 3.0));

        CPPUNIT_ASSERT(math::isApproxEqual(Vec3d(1.0, 2.0, 3.0), vec3dHandle->get(0)));
    }

    {
        // empty source grid

        std::vector<Vec3f> pointPositions{Vec3f(0.0f, 0.0f, 0.0f)};

        PointDataGrid::Ptr points = points::createPointDataGrid<NullCodec, PointDataGrid, Vec3f>(
            pointPositions, *transform);

        points::appendAttribute<Vec3f>(points->tree(), "test");

        VectorGrid::Ptr testGrid = VectorGrid::create();

        points::boxSample(*points, *testGrid, "test");

        points::AttributeHandle<Vec3f>::Ptr handle =
            points::AttributeHandle<Vec3f>::create(
                points->tree().cbeginLeaf()->attributeArray("test"));

        CPPUNIT_ASSERT(math::isApproxEqual(Vec3f(0.0f, 0.0f, 0.0f), handle->get(0)));
    }

    {
        // empty point grid

        std::vector<Vec3f> pointPositions;
        PointDataGrid::Ptr points = points::createPointDataGrid<NullCodec, PointDataGrid, Vec3f>(
            pointPositions, *transform);

        CPPUNIT_ASSERT(points);

        FloatGrid::Ptr testGrid = FloatGrid::create(1.0);

        points::appendAttribute<float>(points->tree(), "test");

        CPPUNIT_ASSERT_NO_THROW(points::boxSample(*points, *testGrid, "test"));
    }

    {
        // exception if one tries to sample to "P" attribute

        std::vector<Vec3f> pointPositions{Vec3f(0.0f, 0.0f, 0.0f)};

        PointDataGrid::Ptr points = points::createPointDataGrid<NullCodec, PointDataGrid, Vec3f>(
            pointPositions, *transform);

        CPPUNIT_ASSERT(points);

        FloatGrid::Ptr testGrid = FloatGrid::create(1.0);

        CPPUNIT_ASSERT_THROW_MESSAGE("Cannot sample onto the \"P\" attribute",
            points::boxSample(*points, *testGrid, "P"), RuntimeError);

        // name of the grid is used if no attribute is provided

        testGrid->setName("test_grid");

        CPPUNIT_ASSERT(!points->tree().cbeginLeaf()->hasAttribute("test_grid"));

        points::boxSample(*points, *testGrid);

        CPPUNIT_ASSERT(points->tree().cbeginLeaf()->hasAttribute("test_grid"));

        // name fails if the grid is called "P"

        testGrid->setName("P");

        CPPUNIT_ASSERT_THROW_MESSAGE("Cannot sample onto the \"P\" attribute",
            points::boxSample(*points, *testGrid), RuntimeError);
    }

    {
        // test non-cell centered points with scalar data and matching transform
        // use various sampling orders

        std::vector<Vec3f> pointPositions{Vec3f(0.03f, 0.0f, 0.0f), Vec3f(0.11f, 0.03f, 0.0f)};

        PointDataGrid::Ptr points = points::createPointDataGrid<NullCodec, PointDataGrid, Vec3f>(
            pointPositions, *transform);

        CPPUNIT_ASSERT(points);

        FloatGrid::Ptr testGrid = FloatGrid::create();

        testGrid->setTransform(transform);
        testGrid->tree().setValue(Coord(-1,0,0), -1.0f);
        testGrid->tree().setValue(Coord(0,0,0), 1.0f);
        testGrid->tree().setValue(Coord(1,0,0), 2.0f);
        testGrid->tree().setValue(Coord(2,0,0), 4.0f);
        testGrid->tree().setValue(Coord(0,1,0), 3.0f);

        points::appendAttribute<float>(points->tree(), "test");
        points::AttributeHandle<float>::Ptr handle =
            points::AttributeHandle<float>::create(
                points->tree().cbeginLeaf()->attributeArray("test"));

        CPPUNIT_ASSERT(handle.get());

        FloatGrid::ConstAccessor testGridAccessor = testGrid->getConstAccessor();

        // check nearest-neighbour sampling

        points::pointSample(*points, *testGrid, "test");

        float expected = tools::PointSampler::sample(testGridAccessor, Vec3f(0.3f, 0.0f, 0.0f));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, handle->get(0), 1e-6);

        expected = tools::PointSampler::sample(testGridAccessor, Vec3f(1.1f, 0.3f, 0.0f));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, handle->get(1), 1e-6);

        // check tri-linear sampling

        points::boxSample(*points, *testGrid, "test");

        expected = tools::BoxSampler::sample(testGridAccessor, Vec3f(0.3f, 0.0f, 0.0f));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, handle->get(0), 1e-6);

        expected = tools::BoxSampler::sample(testGridAccessor, Vec3f(1.1f, 0.3f, 0.0f));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, handle->get(1), 1e-6);

        // check tri-quadratic sampling

        points::quadraticSample(*points, *testGrid, "test");

        expected = tools::QuadraticSampler::sample(testGridAccessor, Vec3f(0.3f, 0.0f, 0.0f));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, handle->get(0), 1e-6);

        expected = tools::QuadraticSampler::sample(testGridAccessor, Vec3f(1.1f, 0.3f, 0.0f));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, handle->get(1), 1e-6);
    }

    {
        // staggered grid and mismatching transforms

        std::vector<Vec3f> pointPositions{Vec3f(0.03f, 0.0f, 0.0f), Vec3f(0.0f, 0.03f, 0.0f),
            Vec3f(0.0f, 0.0f, 0.03f),};

        PointDataGrid::Ptr points =
            points::createPointDataGrid<points::NullCodec, PointDataGrid, Vec3f>(pointPositions,
                *transform);

        CPPUNIT_ASSERT(points);

        VectorGrid::Ptr testGrid = VectorGrid::create();

        testGrid->setGridClass(GRID_STAGGERED);
        testGrid->tree().setValue(Coord(0,0,0), Vec3f(1.0f, 2.0f, 3.0f));
        testGrid->tree().setValue(Coord(0,1,0), Vec3f(1.5f, 2.5f, 3.5f));
        testGrid->tree().setValue(Coord(0,0,1), Vec3f(2.0f, 3.0f, 4.0));

        points::appendAttribute<Vec3f>(points->tree(), "test");

        points::AttributeHandle<Vec3f>::Ptr handle =
            points::AttributeHandle<Vec3f>::create(
                points->tree().cbeginLeaf()->attributeArray("test"));

        CPPUNIT_ASSERT(handle.get());

        Vec3fGrid::ConstAccessor testGridAccessor = testGrid->getConstAccessor();

        // nearest-neighbour staggered sampling

        points::pointSample(*points, *testGrid, "test");

        Vec3f expected = tools::StaggeredPointSampler::sample(testGridAccessor,
            Vec3f(0.03f, 0.0f, 0.0f));

        CPPUNIT_ASSERT(math::isApproxEqual(expected, handle->get(0)));

        expected = tools::StaggeredPointSampler::sample(testGridAccessor, Vec3f(0.0f, 0.03f, 0.0f));

        CPPUNIT_ASSERT(math::isApproxEqual(expected, handle->get(1)));

        // tri-linear staggered sampling

        points::boxSample(*points, *testGrid, "test");

        expected = tools::StaggeredBoxSampler::sample(testGridAccessor,
            Vec3f(0.03f, 0.0f, 0.0f));

        CPPUNIT_ASSERT(math::isApproxEqual(expected, handle->get(0)));

        expected = tools::StaggeredBoxSampler::sample(testGridAccessor, Vec3f(0.0f, 0.03f, 0.0f));

        CPPUNIT_ASSERT(math::isApproxEqual(expected, handle->get(1)));

        // tri-quadratic staggered sampling

        points::quadraticSample(*points, *testGrid, "test");

        expected = tools::StaggeredQuadraticSampler::sample(testGridAccessor,
          Vec3f(0.03f, 0.0f, 0.0f));

        CPPUNIT_ASSERT(math::isApproxEqual(expected, handle->get(0)));

        expected = tools::StaggeredQuadraticSampler::sample(testGridAccessor,
            Vec3f(0.0f, 0.03f, 0.0f));

        CPPUNIT_ASSERT(math::isApproxEqual(expected, handle->get(1)));
    }

    {
        // value type of grid and attribute type don't match

        std::vector<Vec3f> pointPositions{Vec3f(0.3f, 0.0f, 0.0f)};

        math::Transform::Ptr transform2(math::Transform::createLinearTransform(1.0f));
        PointDataGrid::Ptr points =
            points::createPointDataGrid<NullCodec, PointDataGrid, Vec3f>(pointPositions,
                *transform2);

        CPPUNIT_ASSERT(points);

        FloatGrid::Ptr testFloatGrid = FloatGrid::create();

        testFloatGrid->setTransform(transform2);
        testFloatGrid->tree().setValue(Coord(0,0,0), 1.1f);
        testFloatGrid->tree().setValue(Coord(1,0,0), 2.8f);
        testFloatGrid->tree().setValue(Coord(0,1,0), 3.4f);

        points::appendAttribute<int>(points->tree(), "testint");
        points::boxSample(*points, *testFloatGrid, "testint");
        points::AttributeHandle<int>::Ptr handle = points::AttributeHandle<int>::create(
            points->tree().cbeginLeaf()->attributeArray("testint"));

        CPPUNIT_ASSERT(handle.get());

        FloatGrid::ConstAccessor testFloatGridAccessor = testFloatGrid->getConstAccessor();

        // check against box sampler values

        const float sampledValue = tools::BoxSampler::sample(testFloatGridAccessor,
            Vec3f(0.3f, 0.0f, 0.0f));
        const int expected = static_cast<int>(math::Round(sampledValue));

        CPPUNIT_ASSERT_EQUAL(expected, handle->get(0));

        // check mismatching grid type using vector types

        Vec3fGrid::Ptr testVec3fGrid = Vec3fGrid::create();

        testVec3fGrid->setTransform(transform2);
        testVec3fGrid->tree().setValue(Coord(0,0,0), Vec3f(1.0f, 2.0f, 3.0f));
        testVec3fGrid->tree().setValue(Coord(1,0,0), Vec3f(1.5f, 2.5f, 3.5f));
        testVec3fGrid->tree().setValue(Coord(0,1,0), Vec3f(2.0f, 3.0f, 4.0f));

        points::appendAttribute<Vec3d>(points->tree(), "testvec3d");
        points::boxSample(*points, *testVec3fGrid, "testvec3d");
        points::AttributeHandle<Vec3d>::Ptr handle2 = points::AttributeHandle<Vec3d>::create(
            points->tree().cbeginLeaf()->attributeArray("testvec3d"));

        Vec3fGrid::ConstAccessor testVec3fGridAccessor = testVec3fGrid->getConstAccessor();
        const Vec3d expected2 = static_cast<Vec3d>(tools::BoxSampler::sample(testVec3fGridAccessor,
            Vec3f(0.3f, 0.0f, 0.0f)));

        CPPUNIT_ASSERT(math::isExactlyEqual(expected2, handle2->get(0)));

        // check implicit casting of types for sampling using sampleGrid()

        points::appendAttribute<Vec3d>(points->tree(), "testvec3d2");
        points::sampleGrid(/*linear*/1, *points, *testVec3fGrid, "testvec3d2");
        points::AttributeHandle<Vec3d>::Ptr handle3 = points::AttributeHandle<Vec3d>::create(
            points->tree().cbeginLeaf()->attributeArray("testvec3d2"));

        CPPUNIT_ASSERT(math::isExactlyEqual(expected2, handle3->get(0)));

        // check explicit casting of types for sampling using sampleGrid()

        points::sampleGrid<PointDataGrid, Vec3SGrid, Vec3d>(
            /*linear*/1, *points, *testVec3fGrid, "testvec3d3");
        points::AttributeHandle<Vec3d>::Ptr handle4 = points::AttributeHandle<Vec3d>::create(
            points->tree().cbeginLeaf()->attributeArray("testvec3d3"));

        CPPUNIT_ASSERT(math::isExactlyEqual(expected2, handle4->get(0)));

        // check invalid casting of types

        points::appendAttribute<float>(points->tree(), "testfloat");

        // The following is a substitute for CPPUNIT_ASSERT_THROW_MESSAGE(),
        // which generates a compiler warning when the expected exception type
        // is std::exception.
        try {
            points::boxSample(*points, *testVec3fGrid, "testfloat");
            CPPUNIT_FAIL("expected exception not thrown:"
                " cannot sample a vec3s grid on to a float attribute");
        } catch (std::exception&) {
        } catch (...) {
            CPPUNIT_FAIL("expected std::exception or derived");
        }

        // check invalid existing attribute type (Vec4s attribute)

        points::TypedAttributeArray<Vec4s>::registerType();
        points::appendAttribute<Vec4s>(points->tree(), "testv4f");
        CPPUNIT_ASSERT_THROW(points::boxSample(*points, *testVec3fGrid, "testv4f"), TypeError);
    }

    { // sample a non-standard grid type (a Vec4<float> grid)
        using Vec4STree = tree::Tree4<Vec4s, 5, 4, 3>::Type;
        using Vec4SGrid = Grid<Vec4STree>;
        Vec4SGrid::registerGrid();
        points::TypedAttributeArray<Vec4s>::registerType();

        std::vector<Vec3f> pointPositions{Vec3f(0.3f, 0.0f, 0.0f)};

        math::Transform::Ptr transform2(math::Transform::createLinearTransform(1.0f));
        PointDataGrid::Ptr points =
            points::createPointDataGrid<NullCodec, PointDataGrid, Vec3f>(pointPositions,
                *transform2);

        auto testVec4fGrid = Vec4SGrid::create();
        testVec4fGrid->setTransform(transform2);
        testVec4fGrid->tree().setValue(Coord(0,0,0), Vec4s(1.0f, 2.0f, 3.0f, 4.0f));
        testVec4fGrid->tree().setValue(Coord(1,0,0), Vec4s(1.5f, 2.5f, 3.5f, 4.5f));
        testVec4fGrid->tree().setValue(Coord(0,1,0), Vec4s(2.0f, 3.0f, 4.0f, 5.0f));

        points::boxSample(*points, *testVec4fGrid, "testvec4f");
        points::AttributeHandle<Vec4s>::Ptr handle2 = points::AttributeHandle<Vec4s>::create(
            points->tree().cbeginLeaf()->attributeArray("testvec4f"));

        Vec4SGrid::ConstAccessor testVec4fGridAccessor = testVec4fGrid->getConstAccessor();
        const Vec4s expected2 = static_cast<Vec4s>(tools::BoxSampler::sample(testVec4fGridAccessor,
            Vec3f(0.3f, 0.0f, 0.0f)));

        CPPUNIT_ASSERT(math::isExactlyEqual(expected2, handle2->get(0)));
    }
}

void
TestPointSample::testPointSampleWithGroups()
{
    using points::PointDataGrid;

    std::vector<Vec3f> pointPositions{Vec3f(0.03f, 0.0f, 0.0f), Vec3f(0.0f, 0.03f, 0.0f),
        Vec3f(0.0f, 0.0f, 0.0f)};

    math::Transform::Ptr transform(math::Transform::createLinearTransform(0.1f));
    PointDataGrid::Ptr points = points::createPointDataGrid<points::NullCodec,
            PointDataGrid, Vec3f>(pointPositions, *transform);

    CPPUNIT_ASSERT(points);

    DoubleGrid::Ptr testGrid = DoubleGrid::create();

    testGrid->setTransform(transform);
    testGrid->tree().setValue(Coord(0,0,0), 1.0);
    testGrid->tree().setValue(Coord(1,0,0), 2.0);
    testGrid->tree().setValue(Coord(0,1,0), 3.0);

    points::appendGroup(points->tree(), "group1");

    auto leaf = points->tree().beginLeaf();

    points::GroupWriteHandle group1Handle = leaf->groupWriteHandle("group1");

    group1Handle.set(0, true);
    group1Handle.set(1, false);
    group1Handle.set(2, true);

    points::appendAttribute<double>(points->tree(), "test_include");

    std::vector<std::string> includeGroups({"group1"});
    std::vector<std::string> excludeGroups;
    points::MultiGroupFilter filter1(includeGroups, excludeGroups, leaf->attributeSet());
    points::boxSample(*points, *testGrid, "test_include", filter1);

    points::AttributeHandle<double>::Ptr handle =
        points::AttributeHandle<double>::create(
            points->tree().cbeginLeaf()->attributeArray("test_include"));

    DoubleGrid::ConstAccessor testGridAccessor = testGrid->getConstAccessor();

    double expected = tools::BoxSampler::sample(testGridAccessor, Vec3f(0.3f, 0.0f, 0.0f));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, handle->get(0), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, handle->get(1), 1e-6);

    expected = tools::BoxSampler::sample(testGridAccessor, Vec3f(0.0f, 0.0f, 0.0f));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, handle->get(2), 1e-6);

    points::appendAttribute<double>(points->tree(), "test_exclude");

    // test with group treated as "exclusion" group

    points::MultiGroupFilter filter2(excludeGroups, includeGroups, leaf->attributeSet());
    points::boxSample(*points, *testGrid, "test_exclude", filter2);

    points::AttributeHandle<double>::Ptr handle2 =
        points::AttributeHandle<double>::create(
            points->tree().cbeginLeaf()->attributeArray("test_exclude"));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, handle2->get(0), 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, handle2->get(2), 1e-6);

    expected = tools::BoxSampler::sample(testGridAccessor, Vec3f(0.0f, 0.3f, 0.0f));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, handle2->get(1), 1e-6);
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
