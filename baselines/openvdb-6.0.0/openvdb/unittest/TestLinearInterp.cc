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
#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/math/Stencils.h>

// CPPUNIT_TEST_SUITE() invokes CPPUNIT_TESTNAMER_DECL() to generate a suite name
// from the FixtureType.  But if FixtureType is a templated type, the generated name
// can become long and messy.  This macro overrides the normal naming logic,
// instead invoking FixtureType::testSuiteName(), which should be a static member
// function that returns a std::string containing the suite name for the specific
// template instantiation.
#undef CPPUNIT_TESTNAMER_DECL
#define CPPUNIT_TESTNAMER_DECL( variableName, FixtureType ) \
    CPPUNIT_NS::TestNamer variableName( FixtureType::testSuiteName() )

namespace {
// Absolute tolerance for floating-point equality comparisons
const double TOLERANCE = 1.e-6;
}

template<typename GridType>
class TestLinearInterp: public CppUnit::TestCase
{
public:
    static std::string testSuiteName()
    {
        std::string name = openvdb::typeNameAsString<typename GridType::ValueType>();
        if (!name.empty()) name[0] = static_cast<char>(::toupper(name[0]));
        return "TestLinearInterp" + name;
    }

    CPPUNIT_TEST_SUITE(TestLinearInterp);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST(testTree);
    CPPUNIT_TEST(testAccessor);
    CPPUNIT_TEST(testConstantValues);
    CPPUNIT_TEST(testFillValues);
    CPPUNIT_TEST(testNegativeIndices);
    CPPUNIT_TEST(testStencilsMatch);
    CPPUNIT_TEST_SUITE_END();

    void test();
    void testTree();
    void testAccessor();
    void testConstantValues();
    void testFillValues();
    void testNegativeIndices();
    void testStencilsMatch();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLinearInterp<openvdb::FloatGrid>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestLinearInterp<openvdb::DoubleGrid>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestLinearInterp<openvdb::Vec3SGrid>);


template<typename GridType>
void
TestLinearInterp<GridType>::test()
{
    typename GridType::TreeType TreeType;
    float fillValue = 256.0f;

    GridType grid(fillValue);
    typename GridType::TreeType& tree = grid.tree();

    tree.setValue(openvdb::Coord(10, 10, 10), 1.0);

    tree.setValue(openvdb::Coord(11, 10, 10), 2.0);
    tree.setValue(openvdb::Coord(11, 11, 10), 2.0);
    tree.setValue(openvdb::Coord(10, 11, 10), 2.0);
    tree.setValue(openvdb::Coord( 9, 11, 10), 2.0);
    tree.setValue(openvdb::Coord( 9, 10, 10), 2.0);
    tree.setValue(openvdb::Coord( 9,  9, 10), 2.0);
    tree.setValue(openvdb::Coord(10,  9, 10), 2.0);
    tree.setValue(openvdb::Coord(11,  9, 10), 2.0);

    tree.setValue(openvdb::Coord(10, 10, 11), 3.0);
    tree.setValue(openvdb::Coord(11, 10, 11), 3.0);
    tree.setValue(openvdb::Coord(11, 11, 11), 3.0);
    tree.setValue(openvdb::Coord(10, 11, 11), 3.0);
    tree.setValue(openvdb::Coord( 9, 11, 11), 3.0);
    tree.setValue(openvdb::Coord( 9, 10, 11), 3.0);
    tree.setValue(openvdb::Coord( 9,  9, 11), 3.0);
    tree.setValue(openvdb::Coord(10,  9, 11), 3.0);
    tree.setValue(openvdb::Coord(11,  9, 11), 3.0);

    tree.setValue(openvdb::Coord(10, 10, 9), 4.0);
    tree.setValue(openvdb::Coord(11, 10, 9), 4.0);
    tree.setValue(openvdb::Coord(11, 11, 9), 4.0);
    tree.setValue(openvdb::Coord(10, 11, 9), 4.0);
    tree.setValue(openvdb::Coord( 9, 11, 9), 4.0);
    tree.setValue(openvdb::Coord( 9, 10, 9), 4.0);
    tree.setValue(openvdb::Coord( 9,  9, 9), 4.0);
    tree.setValue(openvdb::Coord(10,  9, 9), 4.0);
    tree.setValue(openvdb::Coord(11,  9, 9), 4.0);

    {//using BoxSampler

        // transform used for worldspace interpolation)
        openvdb::tools::GridSampler<GridType, openvdb::tools::BoxSampler>
            interpolator(grid);
        //openvdb::tools::LinearInterp<GridType> interpolator(*tree);

        typename GridType::ValueType val =
            interpolator.sampleVoxel(10.5, 10.5, 10.5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.375, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.0, 10.0, 10.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, val, TOLERANCE);

        val = interpolator.sampleVoxel(11.0, 10.0, 10.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

        val = interpolator.sampleVoxel(11.0, 11.0, 10.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

        val = interpolator.sampleVoxel(11.0, 11.0, 11.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, val, TOLERANCE);

        val = interpolator.sampleVoxel(9.0, 11.0, 9.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, val, TOLERANCE);

        val = interpolator.sampleVoxel(9.0, 10.0, 9.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.1, 10.0, 10.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.1, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.8, 10.8, 10.8);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.792, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.1, 10.8, 10.5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.41, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.8, 10.1, 10.5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.41, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.5, 10.1, 10.8);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.71, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.5, 10.8, 10.1);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.01, val, TOLERANCE);

    }
    {//using Sampler<1>

        // transform used for worldspace interpolation)
        openvdb::tools::GridSampler<GridType, openvdb::tools::Sampler<1> >
            interpolator(grid);
        //openvdb::tools::LinearInterp<GridType> interpolator(*tree);

        typename GridType::ValueType val =
            interpolator.sampleVoxel(10.5, 10.5, 10.5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.375, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.0, 10.0, 10.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, val, TOLERANCE);

        val = interpolator.sampleVoxel(11.0, 10.0, 10.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

        val = interpolator.sampleVoxel(11.0, 11.0, 10.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

        val = interpolator.sampleVoxel(11.0, 11.0, 11.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, val, TOLERANCE);

        val = interpolator.sampleVoxel(9.0, 11.0, 9.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, val, TOLERANCE);

        val = interpolator.sampleVoxel(9.0, 10.0, 9.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.1, 10.0, 10.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.1, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.8, 10.8, 10.8);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.792, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.1, 10.8, 10.5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.41, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.8, 10.1, 10.5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.41, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.5, 10.1, 10.8);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.71, val, TOLERANCE);

        val = interpolator.sampleVoxel(10.5, 10.8, 10.1);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.01, val, TOLERANCE);
    }
}


template<>
void
TestLinearInterp<openvdb::Vec3SGrid>::test()
{
    using namespace openvdb;

    Vec3s fillValue = Vec3s(256.0f, 256.0f, 256.0f);

    Vec3SGrid grid(fillValue);
    Vec3STree& tree = grid.tree();

    tree.setValue(openvdb::Coord(10, 10, 10), Vec3s(1.0, 1.0, 1.0));

    tree.setValue(openvdb::Coord(11, 10, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(11, 11, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(10, 11, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9, 11, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9, 10, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9,  9, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(10,  9, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(11,  9, 10), Vec3s(2.0, 2.0, 2.0));

    tree.setValue(openvdb::Coord(10, 10, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(11, 10, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(11, 11, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(10, 11, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord( 9, 11, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord( 9, 10, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord( 9,  9, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(10,  9, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(11,  9, 11), Vec3s(3.0, 3.0, 3.0));

    tree.setValue(openvdb::Coord(10, 10, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(11, 10, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(11, 11, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(10, 11, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord( 9, 11, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord( 9, 10, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord( 9,  9, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(10,  9, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(11,  9, 9), Vec3s(4.0, 4.0, 4.0));

    openvdb::tools::GridSampler<Vec3SGrid, openvdb::tools::BoxSampler>
        interpolator(grid);

    //openvdb::tools::LinearInterp<Vec3STree> interpolator(*tree);

    Vec3SGrid::ValueType val = interpolator.sampleVoxel(10.5, 10.5, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.375f)));

    val = interpolator.sampleVoxel(10.0, 10.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(1.f)));

    val = interpolator.sampleVoxel(11.0, 10.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.f)));

    val = interpolator.sampleVoxel(11.0, 11.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.f)));

    val = interpolator.sampleVoxel(11.0, 11.0, 11.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(3.f)));

    val = interpolator.sampleVoxel(9.0, 11.0, 9.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(4.f)));

    val = interpolator.sampleVoxel(9.0, 10.0, 9.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(4.f)));

    val = interpolator.sampleVoxel(10.1, 10.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(1.1f)));

    val = interpolator.sampleVoxel(10.8, 10.8, 10.8);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.792f)));

    val = interpolator.sampleVoxel(10.1, 10.8, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.41f)));

    val = interpolator.sampleVoxel(10.8, 10.1, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.41f)));

    val = interpolator.sampleVoxel(10.5, 10.1, 10.8);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.71f)));

    val = interpolator.sampleVoxel(10.5, 10.8, 10.1);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.01f)));
}

template<typename GridType>
void
TestLinearInterp<GridType>::testTree()
{
    float fillValue = 256.0f;
    typedef typename GridType::TreeType TreeType;
    TreeType tree(fillValue);

    tree.setValue(openvdb::Coord(10, 10, 10), 1.0);

    tree.setValue(openvdb::Coord(11, 10, 10), 2.0);
    tree.setValue(openvdb::Coord(11, 11, 10), 2.0);
    tree.setValue(openvdb::Coord(10, 11, 10), 2.0);
    tree.setValue(openvdb::Coord( 9, 11, 10), 2.0);
    tree.setValue(openvdb::Coord( 9, 10, 10), 2.0);
    tree.setValue(openvdb::Coord( 9,  9, 10), 2.0);
    tree.setValue(openvdb::Coord(10,  9, 10), 2.0);
    tree.setValue(openvdb::Coord(11,  9, 10), 2.0);

    tree.setValue(openvdb::Coord(10, 10, 11), 3.0);
    tree.setValue(openvdb::Coord(11, 10, 11), 3.0);
    tree.setValue(openvdb::Coord(11, 11, 11), 3.0);
    tree.setValue(openvdb::Coord(10, 11, 11), 3.0);
    tree.setValue(openvdb::Coord( 9, 11, 11), 3.0);
    tree.setValue(openvdb::Coord( 9, 10, 11), 3.0);
    tree.setValue(openvdb::Coord( 9,  9, 11), 3.0);
    tree.setValue(openvdb::Coord(10,  9, 11), 3.0);
    tree.setValue(openvdb::Coord(11,  9, 11), 3.0);

    tree.setValue(openvdb::Coord(10, 10, 9), 4.0);
    tree.setValue(openvdb::Coord(11, 10, 9), 4.0);
    tree.setValue(openvdb::Coord(11, 11, 9), 4.0);
    tree.setValue(openvdb::Coord(10, 11, 9), 4.0);
    tree.setValue(openvdb::Coord( 9, 11, 9), 4.0);
    tree.setValue(openvdb::Coord( 9, 10, 9), 4.0);
    tree.setValue(openvdb::Coord( 9,  9, 9), 4.0);
    tree.setValue(openvdb::Coord(10,  9, 9), 4.0);
    tree.setValue(openvdb::Coord(11,  9, 9), 4.0);

    // transform used for worldspace interpolation)
    openvdb::tools::GridSampler<TreeType, openvdb::tools::BoxSampler>
        interpolator(tree, openvdb::math::Transform());

    typename GridType::ValueType val =
        interpolator.sampleVoxel(10.5, 10.5, 10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.375, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.0, 10.0, 10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(11.0, 10.0, 10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(11.0, 11.0, 10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(11.0, 11.0, 11.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(9.0, 11.0, 9.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(9.0, 10.0, 9.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.1, 10.0, 10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.1, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.8, 10.8, 10.8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.792, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.1, 10.8, 10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.41, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.8, 10.1, 10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.41, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.5, 10.1, 10.8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.71, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.5, 10.8, 10.1);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.01, val, TOLERANCE);
}


template<>
void
TestLinearInterp<openvdb::Vec3SGrid>::testTree()
{
    using namespace openvdb;

    Vec3s fillValue = Vec3s(256.0f, 256.0f, 256.0f);

    Vec3STree tree(fillValue);

    tree.setValue(openvdb::Coord(10, 10, 10), Vec3s(1.0, 1.0, 1.0));

    tree.setValue(openvdb::Coord(11, 10, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(11, 11, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(10, 11, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9, 11, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9, 10, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9,  9, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(10,  9, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(11,  9, 10), Vec3s(2.0, 2.0, 2.0));

    tree.setValue(openvdb::Coord(10, 10, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(11, 10, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(11, 11, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(10, 11, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord( 9, 11, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord( 9, 10, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord( 9,  9, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(10,  9, 11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(11,  9, 11), Vec3s(3.0, 3.0, 3.0));

    tree.setValue(openvdb::Coord(10, 10, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(11, 10, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(11, 11, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(10, 11, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord( 9, 11, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord( 9, 10, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord( 9,  9, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(10,  9, 9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(11,  9, 9), Vec3s(4.0, 4.0, 4.0));

    openvdb::tools::GridSampler<Vec3STree, openvdb::tools::BoxSampler>
        interpolator(tree, openvdb::math::Transform());

    //openvdb::tools::LinearInterp<Vec3STree> interpolator(*tree);

    Vec3SGrid::ValueType val = interpolator.sampleVoxel(10.5, 10.5, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.375f)));

    val = interpolator.sampleVoxel(10.0, 10.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(1.f)));

    val = interpolator.sampleVoxel(11.0, 10.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.f)));

    val = interpolator.sampleVoxel(11.0, 11.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.f)));

    val = interpolator.sampleVoxel(11.0, 11.0, 11.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(3.f)));

    val = interpolator.sampleVoxel(9.0, 11.0, 9.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(4.f)));

    val = interpolator.sampleVoxel(9.0, 10.0, 9.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(4.f)));

    val = interpolator.sampleVoxel(10.1, 10.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(1.1f)));

    val = interpolator.sampleVoxel(10.8, 10.8, 10.8);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.792f)));

    val = interpolator.sampleVoxel(10.1, 10.8, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.41f)));

    val = interpolator.sampleVoxel(10.8, 10.1, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.41f)));

    val = interpolator.sampleVoxel(10.5, 10.1, 10.8);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.71f)));

    val = interpolator.sampleVoxel(10.5, 10.8, 10.1);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.01f)));
}

template<typename GridType>
void
TestLinearInterp<GridType>::testAccessor()
{
    float fillValue = 256.0f;

    GridType grid(fillValue);
    typedef typename GridType::Accessor AccessorType;

    AccessorType acc = grid.getAccessor();

    acc.setValue(openvdb::Coord(10, 10, 10), 1.0);

    acc.setValue(openvdb::Coord(11, 10, 10), 2.0);
    acc.setValue(openvdb::Coord(11, 11, 10), 2.0);
    acc.setValue(openvdb::Coord(10, 11, 10), 2.0);
    acc.setValue(openvdb::Coord( 9, 11, 10), 2.0);
    acc.setValue(openvdb::Coord( 9, 10, 10), 2.0);
    acc.setValue(openvdb::Coord( 9,  9, 10), 2.0);
    acc.setValue(openvdb::Coord(10,  9, 10), 2.0);
    acc.setValue(openvdb::Coord(11,  9, 10), 2.0);

    acc.setValue(openvdb::Coord(10, 10, 11), 3.0);
    acc.setValue(openvdb::Coord(11, 10, 11), 3.0);
    acc.setValue(openvdb::Coord(11, 11, 11), 3.0);
    acc.setValue(openvdb::Coord(10, 11, 11), 3.0);
    acc.setValue(openvdb::Coord( 9, 11, 11), 3.0);
    acc.setValue(openvdb::Coord( 9, 10, 11), 3.0);
    acc.setValue(openvdb::Coord( 9,  9, 11), 3.0);
    acc.setValue(openvdb::Coord(10,  9, 11), 3.0);
    acc.setValue(openvdb::Coord(11,  9, 11), 3.0);

    acc.setValue(openvdb::Coord(10, 10, 9), 4.0);
    acc.setValue(openvdb::Coord(11, 10, 9), 4.0);
    acc.setValue(openvdb::Coord(11, 11, 9), 4.0);
    acc.setValue(openvdb::Coord(10, 11, 9), 4.0);
    acc.setValue(openvdb::Coord( 9, 11, 9), 4.0);
    acc.setValue(openvdb::Coord( 9, 10, 9), 4.0);
    acc.setValue(openvdb::Coord( 9,  9, 9), 4.0);
    acc.setValue(openvdb::Coord(10,  9, 9), 4.0);
    acc.setValue(openvdb::Coord(11,  9, 9), 4.0);

    // transform used for worldspace interpolation)
    openvdb::tools::GridSampler<AccessorType, openvdb::tools::BoxSampler>
        interpolator(acc, grid.transform());

    typename GridType::ValueType val =
        interpolator.sampleVoxel(10.5, 10.5, 10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.375, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.0, 10.0, 10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(11.0, 10.0, 10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(11.0, 11.0, 10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(11.0, 11.0, 11.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(9.0, 11.0, 9.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(9.0, 10.0, 9.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.1, 10.0, 10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.1, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.8, 10.8, 10.8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.792, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.1, 10.8, 10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.41, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.8, 10.1, 10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.41, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.5, 10.1, 10.8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.71, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.5, 10.8, 10.1);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.01, val, TOLERANCE);
}


template<>
void
TestLinearInterp<openvdb::Vec3SGrid>::testAccessor()
{
    using namespace openvdb;

    Vec3s fillValue = Vec3s(256.0f, 256.0f, 256.0f);

    Vec3SGrid grid(fillValue);
    typedef Vec3SGrid::Accessor AccessorType;
    AccessorType acc = grid.getAccessor();

    acc.setValue(openvdb::Coord(10, 10, 10), Vec3s(1.0, 1.0, 1.0));

    acc.setValue(openvdb::Coord(11, 10, 10), Vec3s(2.0, 2.0, 2.0));
    acc.setValue(openvdb::Coord(11, 11, 10), Vec3s(2.0, 2.0, 2.0));
    acc.setValue(openvdb::Coord(10, 11, 10), Vec3s(2.0, 2.0, 2.0));
    acc.setValue(openvdb::Coord( 9, 11, 10), Vec3s(2.0, 2.0, 2.0));
    acc.setValue(openvdb::Coord( 9, 10, 10), Vec3s(2.0, 2.0, 2.0));
    acc.setValue(openvdb::Coord( 9,  9, 10), Vec3s(2.0, 2.0, 2.0));
    acc.setValue(openvdb::Coord(10,  9, 10), Vec3s(2.0, 2.0, 2.0));
    acc.setValue(openvdb::Coord(11,  9, 10), Vec3s(2.0, 2.0, 2.0));

    acc.setValue(openvdb::Coord(10, 10, 11), Vec3s(3.0, 3.0, 3.0));
    acc.setValue(openvdb::Coord(11, 10, 11), Vec3s(3.0, 3.0, 3.0));
    acc.setValue(openvdb::Coord(11, 11, 11), Vec3s(3.0, 3.0, 3.0));
    acc.setValue(openvdb::Coord(10, 11, 11), Vec3s(3.0, 3.0, 3.0));
    acc.setValue(openvdb::Coord( 9, 11, 11), Vec3s(3.0, 3.0, 3.0));
    acc.setValue(openvdb::Coord( 9, 10, 11), Vec3s(3.0, 3.0, 3.0));
    acc.setValue(openvdb::Coord( 9,  9, 11), Vec3s(3.0, 3.0, 3.0));
    acc.setValue(openvdb::Coord(10,  9, 11), Vec3s(3.0, 3.0, 3.0));
    acc.setValue(openvdb::Coord(11,  9, 11), Vec3s(3.0, 3.0, 3.0));

    acc.setValue(openvdb::Coord(10, 10, 9), Vec3s(4.0, 4.0, 4.0));
    acc.setValue(openvdb::Coord(11, 10, 9), Vec3s(4.0, 4.0, 4.0));
    acc.setValue(openvdb::Coord(11, 11, 9), Vec3s(4.0, 4.0, 4.0));
    acc.setValue(openvdb::Coord(10, 11, 9), Vec3s(4.0, 4.0, 4.0));
    acc.setValue(openvdb::Coord( 9, 11, 9), Vec3s(4.0, 4.0, 4.0));
    acc.setValue(openvdb::Coord( 9, 10, 9), Vec3s(4.0, 4.0, 4.0));
    acc.setValue(openvdb::Coord( 9,  9, 9), Vec3s(4.0, 4.0, 4.0));
    acc.setValue(openvdb::Coord(10,  9, 9), Vec3s(4.0, 4.0, 4.0));
    acc.setValue(openvdb::Coord(11,  9, 9), Vec3s(4.0, 4.0, 4.0));

    openvdb::tools::GridSampler<AccessorType, openvdb::tools::BoxSampler>
        interpolator(acc, grid.transform());

    Vec3SGrid::ValueType val = interpolator.sampleVoxel(10.5, 10.5, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.375f)));

    val = interpolator.sampleVoxel(10.0, 10.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(1.0f)));

    val = interpolator.sampleVoxel(11.0, 10.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.0f)));

    val = interpolator.sampleVoxel(11.0, 11.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.0f)));

    val = interpolator.sampleVoxel(11.0, 11.0, 11.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(3.0f)));

    val = interpolator.sampleVoxel(9.0, 11.0, 9.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(4.0f)));

    val = interpolator.sampleVoxel(9.0, 10.0, 9.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(4.0f)));

    val = interpolator.sampleVoxel(10.1, 10.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(1.1f)));

    val = interpolator.sampleVoxel(10.8, 10.8, 10.8);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.792f)));

    val = interpolator.sampleVoxel(10.1, 10.8, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.41f)));

    val = interpolator.sampleVoxel(10.8, 10.1, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.41f)));

    val = interpolator.sampleVoxel(10.5, 10.1, 10.8);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.71f)));

    val = interpolator.sampleVoxel(10.5, 10.8, 10.1);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.01f)));
}

template<typename GridType>
void
TestLinearInterp<GridType>::testConstantValues()
{
    typedef typename GridType::TreeType TreeType;
    float fillValue = 256.0f;

    GridType grid(fillValue);
    TreeType& tree = grid.tree();

    // Add values to buffer zero.
    tree.setValue(openvdb::Coord(10, 10, 10), 2.0);

    tree.setValue(openvdb::Coord(11, 10, 10), 2.0);
    tree.setValue(openvdb::Coord(11, 11, 10), 2.0);
    tree.setValue(openvdb::Coord(10, 11, 10), 2.0);
    tree.setValue(openvdb::Coord( 9, 11, 10), 2.0);
    tree.setValue(openvdb::Coord( 9, 10, 10), 2.0);
    tree.setValue(openvdb::Coord( 9,  9, 10), 2.0);
    tree.setValue(openvdb::Coord(10,  9, 10), 2.0);
    tree.setValue(openvdb::Coord(11,  9, 10), 2.0);

    tree.setValue(openvdb::Coord(10, 10, 11), 2.0);
    tree.setValue(openvdb::Coord(11, 10, 11), 2.0);
    tree.setValue(openvdb::Coord(11, 11, 11), 2.0);
    tree.setValue(openvdb::Coord(10, 11, 11), 2.0);
    tree.setValue(openvdb::Coord( 9, 11, 11), 2.0);
    tree.setValue(openvdb::Coord( 9, 10, 11), 2.0);
    tree.setValue(openvdb::Coord( 9,  9, 11), 2.0);
    tree.setValue(openvdb::Coord(10,  9, 11), 2.0);
    tree.setValue(openvdb::Coord(11,  9, 11), 2.0);

    tree.setValue(openvdb::Coord(10, 10, 9), 2.0);
    tree.setValue(openvdb::Coord(11, 10, 9), 2.0);
    tree.setValue(openvdb::Coord(11, 11, 9), 2.0);
    tree.setValue(openvdb::Coord(10, 11, 9), 2.0);
    tree.setValue(openvdb::Coord( 9, 11, 9), 2.0);
    tree.setValue(openvdb::Coord( 9, 10, 9), 2.0);
    tree.setValue(openvdb::Coord( 9,  9, 9), 2.0);
    tree.setValue(openvdb::Coord(10,  9, 9), 2.0);
    tree.setValue(openvdb::Coord(11,  9, 9), 2.0);

    openvdb::tools::GridSampler<TreeType, openvdb::tools::BoxSampler>  interpolator(grid);
    //openvdb::tools::LinearInterp<GridType> interpolator(*tree);

    typename GridType::ValueType val =
        interpolator.sampleVoxel(10.5, 10.5, 10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.0, 10.0, 10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.1, 10.0, 10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.8, 10.8, 10.8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.1, 10.8, 10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.8, 10.1, 10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.5, 10.1, 10.8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.5, 10.8, 10.1);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);
}


template<>
void
TestLinearInterp<openvdb::Vec3SGrid>::testConstantValues()
{
    using namespace openvdb;

    Vec3s fillValue = Vec3s(256.0f, 256.0f, 256.0f);

    Vec3SGrid grid(fillValue);
    Vec3STree& tree = grid.tree();

    // Add values to buffer zero.
    tree.setValue(openvdb::Coord(10, 10, 10), Vec3s(2.0, 2.0, 2.0));

    tree.setValue(openvdb::Coord(11, 10, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(11, 11, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(10, 11, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9, 11, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9, 10, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9,  9, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(10,  9, 10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(11,  9, 10), Vec3s(2.0, 2.0, 2.0));

    tree.setValue(openvdb::Coord(10, 10, 11), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(11, 10, 11), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(11, 11, 11), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(10, 11, 11), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9, 11, 11), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9, 10, 11), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9,  9, 11), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(10,  9, 11), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(11,  9, 11), Vec3s(2.0, 2.0, 2.0));

    tree.setValue(openvdb::Coord(10, 10, 9), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(11, 10, 9), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(11, 11, 9), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(10, 11, 9), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9, 11, 9), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9, 10, 9), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( 9,  9, 9), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(10,  9, 9), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(11,  9, 9), Vec3s(2.0, 2.0, 2.0));

    openvdb::tools::GridSampler<Vec3STree, openvdb::tools::BoxSampler>  interpolator(grid);
    //openvdb::tools::LinearInterp<Vec3STree> interpolator(*tree);

    Vec3SGrid::ValueType val = interpolator.sampleVoxel(10.5, 10.5, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.0, 2.0, 2.0)));

    val = interpolator.sampleVoxel(10.0, 10.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.0, 2.0, 2.0)));

    val = interpolator.sampleVoxel(10.1, 10.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.0, 2.0, 2.0)));

    val = interpolator.sampleVoxel(10.8, 10.8, 10.8);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.0, 2.0, 2.0)));

    val = interpolator.sampleVoxel(10.1, 10.8, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.0, 2.0, 2.0)));

    val = interpolator.sampleVoxel(10.8, 10.1, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.0, 2.0, 2.0)));

    val = interpolator.sampleVoxel(10.5, 10.1, 10.8);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.0, 2.0, 2.0)));

    val = interpolator.sampleVoxel(10.5, 10.8, 10.1);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.0, 2.0, 2.0)));
}


template<typename GridType>
void
TestLinearInterp<GridType>::testFillValues()
{
    //typedef typename GridType::TreeType TreeType;
    float fillValue = 256.0f;

    GridType grid(fillValue);
    //typename GridType::TreeType& tree = grid.tree();

    openvdb::tools::GridSampler<GridType, openvdb::tools::BoxSampler>
        interpolator(grid);
    //openvdb::tools::LinearInterp<GridType> interpolator(*tree);

    typename GridType::ValueType val =
        interpolator.sampleVoxel(10.5, 10.5, 10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(256.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.0, 10.0, 10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(256.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.1, 10.0, 10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(256.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.8, 10.8, 10.8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(256.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.1, 10.8, 10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(256.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.8, 10.1, 10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(256.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.5, 10.1, 10.8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(256.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(10.5, 10.8, 10.1);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(256.0, val, TOLERANCE);
}


template<>
void
TestLinearInterp<openvdb::Vec3SGrid>::testFillValues()
{
    using namespace openvdb;

    Vec3s fillValue = Vec3s(256.0f, 256.0f, 256.0f);

    Vec3SGrid grid(fillValue);
    //Vec3STree& tree = grid.tree();

    openvdb::tools::GridSampler<Vec3SGrid, openvdb::tools::BoxSampler>
        interpolator(grid);
    //openvdb::tools::LinearInterp<Vec3STree> interpolator(*tree);

    Vec3SGrid::ValueType val = interpolator.sampleVoxel(10.5, 10.5, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(256.0, 256.0, 256.0)));

    val = interpolator.sampleVoxel(10.0, 10.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(256.0, 256.0, 256.0)));

    val = interpolator.sampleVoxel(10.1, 10.0, 10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(256.0, 256.0, 256.0)));

    val = interpolator.sampleVoxel(10.8, 10.8, 10.8);
    CPPUNIT_ASSERT(val.eq(Vec3s(256.0, 256.0, 256.0)));

    val = interpolator.sampleVoxel(10.1, 10.8, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(256.0, 256.0, 256.0)));

    val = interpolator.sampleVoxel(10.8, 10.1, 10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(256.0, 256.0, 256.0)));

    val = interpolator.sampleVoxel(10.5, 10.1, 10.8);
    CPPUNIT_ASSERT(val.eq(Vec3s(256.0, 256.0, 256.0)));

    val = interpolator.sampleVoxel(10.5, 10.8, 10.1);
    CPPUNIT_ASSERT(val.eq(Vec3s(256.0, 256.0, 256.0)));
}


template<typename GridType>
void
TestLinearInterp<GridType>::testNegativeIndices()
{
    typedef typename GridType::TreeType TreeType;
    float fillValue = 256.0f;

    GridType grid(fillValue);
    TreeType& tree = grid.tree();

    tree.setValue(openvdb::Coord(-10, -10, -10), 1.0);

    tree.setValue(openvdb::Coord(-11, -10, -10), 2.0);
    tree.setValue(openvdb::Coord(-11, -11, -10), 2.0);
    tree.setValue(openvdb::Coord(-10, -11, -10), 2.0);
    tree.setValue(openvdb::Coord( -9, -11, -10), 2.0);
    tree.setValue(openvdb::Coord( -9, -10, -10), 2.0);
    tree.setValue(openvdb::Coord( -9,  -9, -10), 2.0);
    tree.setValue(openvdb::Coord(-10,  -9, -10), 2.0);
    tree.setValue(openvdb::Coord(-11,  -9, -10), 2.0);

    tree.setValue(openvdb::Coord(-10, -10, -11), 3.0);
    tree.setValue(openvdb::Coord(-11, -10, -11), 3.0);
    tree.setValue(openvdb::Coord(-11, -11, -11), 3.0);
    tree.setValue(openvdb::Coord(-10, -11, -11), 3.0);
    tree.setValue(openvdb::Coord( -9, -11, -11), 3.0);
    tree.setValue(openvdb::Coord( -9, -10, -11), 3.0);
    tree.setValue(openvdb::Coord( -9,  -9, -11), 3.0);
    tree.setValue(openvdb::Coord(-10,  -9, -11), 3.0);
    tree.setValue(openvdb::Coord(-11,  -9, -11), 3.0);

    tree.setValue(openvdb::Coord(-10, -10, -9), 4.0);
    tree.setValue(openvdb::Coord(-11, -10, -9), 4.0);
    tree.setValue(openvdb::Coord(-11, -11, -9), 4.0);
    tree.setValue(openvdb::Coord(-10, -11, -9), 4.0);
    tree.setValue(openvdb::Coord( -9, -11, -9), 4.0);
    tree.setValue(openvdb::Coord( -9, -10, -9), 4.0);
    tree.setValue(openvdb::Coord( -9,  -9, -9), 4.0);
    tree.setValue(openvdb::Coord(-10,  -9, -9), 4.0);
    tree.setValue(openvdb::Coord(-11,  -9, -9), 4.0);

    //openvdb::tools::LinearInterp<GridType> interpolator(*tree);
    openvdb::tools::GridSampler<TreeType, openvdb::tools::BoxSampler>  interpolator(grid);

    typename GridType::ValueType val =
        interpolator.sampleVoxel(-10.5, -10.5, -10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.375, val, TOLERANCE);

    val = interpolator.sampleVoxel(-10.0, -10.0, -10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(-11.0, -10.0, -10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(-11.0, -11.0, -10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(-11.0, -11.0, -11.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(-9.0, -11.0, -9.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(-9.0, -10.0, -9.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, val, TOLERANCE);

    val = interpolator.sampleVoxel(-10.1, -10.0, -10.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.1, val, TOLERANCE);

    val = interpolator.sampleVoxel(-10.8, -10.8, -10.8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.792, val, TOLERANCE);

    val = interpolator.sampleVoxel(-10.1, -10.8, -10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.41, val, TOLERANCE);

    val = interpolator.sampleVoxel(-10.8, -10.1, -10.5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.41, val, TOLERANCE);

    val = interpolator.sampleVoxel(-10.5, -10.1, -10.8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.71, val, TOLERANCE);

    val = interpolator.sampleVoxel(-10.5, -10.8, -10.1);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.01, val, TOLERANCE);
}


template<>
void
TestLinearInterp<openvdb::Vec3SGrid>::testNegativeIndices()
{
    using namespace openvdb;

    Vec3s fillValue = Vec3s(256.0f, 256.0f, 256.0f);

    Vec3SGrid grid(fillValue);
    Vec3STree& tree = grid.tree();

    tree.setValue(openvdb::Coord(-10, -10, -10), Vec3s(1.0, 1.0, 1.0));

    tree.setValue(openvdb::Coord(-11, -10, -10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(-11, -11, -10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(-10, -11, -10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( -9, -11, -10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( -9, -10, -10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord( -9,  -9, -10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(-10,  -9, -10), Vec3s(2.0, 2.0, 2.0));
    tree.setValue(openvdb::Coord(-11,  -9, -10), Vec3s(2.0, 2.0, 2.0));

    tree.setValue(openvdb::Coord(-10, -10, -11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(-11, -10, -11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(-11, -11, -11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(-10, -11, -11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord( -9, -11, -11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord( -9, -10, -11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord( -9,  -9, -11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(-10,  -9, -11), Vec3s(3.0, 3.0, 3.0));
    tree.setValue(openvdb::Coord(-11,  -9, -11), Vec3s(3.0, 3.0, 3.0));

    tree.setValue(openvdb::Coord(-10, -10, -9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(-11, -10, -9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(-11, -11, -9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(-10, -11, -9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord( -9, -11, -9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord( -9, -10, -9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord( -9,  -9, -9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(-10,  -9, -9), Vec3s(4.0, 4.0, 4.0));
    tree.setValue(openvdb::Coord(-11,  -9, -9), Vec3s(4.0, 4.0, 4.0));

    openvdb::tools::GridSampler<Vec3SGrid, openvdb::tools::BoxSampler>  interpolator(grid);
    //openvdb::tools::LinearInterp<Vec3STree> interpolator(*tree);

    Vec3SGrid::ValueType val = interpolator.sampleVoxel(-10.5, -10.5, -10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.375f)));

    val = interpolator.sampleVoxel(-10.0, -10.0, -10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(1.0f)));

    val = interpolator.sampleVoxel(-11.0, -10.0, -10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.0f)));

    val = interpolator.sampleVoxel(-11.0, -11.0, -10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.0f)));

    val = interpolator.sampleVoxel(-11.0, -11.0, -11.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(3.0f)));

    val = interpolator.sampleVoxel(-9.0, -11.0, -9.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(4.0f)));

    val = interpolator.sampleVoxel(-9.0, -10.0, -9.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(4.0f)));

    val = interpolator.sampleVoxel(-10.1, -10.0, -10.0);
    CPPUNIT_ASSERT(val.eq(Vec3s(1.1f)));

    val = interpolator.sampleVoxel(-10.8, -10.8, -10.8);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.792f)));

    val = interpolator.sampleVoxel(-10.1, -10.8, -10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.41f)));

    val = interpolator.sampleVoxel(-10.8, -10.1, -10.5);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.41f)));

    val = interpolator.sampleVoxel(-10.5, -10.1, -10.8);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.71f)));

    val = interpolator.sampleVoxel(-10.5, -10.8, -10.1);
    CPPUNIT_ASSERT(val.eq(Vec3s(2.01f)));
}


template<typename GridType>
void
TestLinearInterp<GridType>::testStencilsMatch()
{
    typedef typename GridType::ValueType ValueType;

    GridType grid;
    typename GridType::TreeType& tree = grid.tree();

    // using mostly recurring numbers

    tree.setValue(openvdb::Coord(0, 0, 0), ValueType(1.0/3.0));
    tree.setValue(openvdb::Coord(0, 1, 0), ValueType(1.0/11.0));
    tree.setValue(openvdb::Coord(0, 0, 1), ValueType(1.0/81.0));
    tree.setValue(openvdb::Coord(1, 0, 0), ValueType(1.0/97.0));
    tree.setValue(openvdb::Coord(1, 1, 0), ValueType(1.0/61.0));
    tree.setValue(openvdb::Coord(0, 1, 1), ValueType(9.0/7.0));
    tree.setValue(openvdb::Coord(1, 0, 1), ValueType(9.0/11.0));
    tree.setValue(openvdb::Coord(1, 1, 1), ValueType(22.0/7.0));

    const openvdb::Vec3f pos(7.0f/12.0f, 1.0f/3.0f, 2.0f/3.0f);

    {//using BoxSampler and BoxStencil

        openvdb::tools::GridSampler<GridType, openvdb::tools::BoxSampler>
            interpolator(grid);

        openvdb::math::BoxStencil<const GridType>
            stencil(grid);

        typename GridType::ValueType val1 = interpolator.sampleVoxel(pos.x(), pos.y(), pos.z());

        stencil.moveTo(pos);
        typename GridType::ValueType val2 = stencil.interpolation(pos);
        CPPUNIT_ASSERT_EQUAL(val1, val2);
    }
}

template<>
void
TestLinearInterp<openvdb::Vec3SGrid>::testStencilsMatch() {}


// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
