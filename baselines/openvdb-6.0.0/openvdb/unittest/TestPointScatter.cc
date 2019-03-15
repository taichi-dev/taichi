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

#include <random>

#include <cppunit/extensions/HelperMacros.h>

#include <openvdb/openvdb.h>

#include <openvdb/points/PointScatter.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointDataGrid.h>

#include <openvdb/math/Math.h>
#include <openvdb/math/Coord.h>

using namespace openvdb;
using namespace openvdb::points;

class TestPointScatter: public CppUnit::TestCase
{
public:

    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointScatter);
    CPPUNIT_TEST(testUniformPointScatter);
    CPPUNIT_TEST(testDenseUniformPointScatter);
    CPPUNIT_TEST(testNonUniformPointScatter);
    CPPUNIT_TEST_SUITE_END();

    void testUniformPointScatter();
    void testDenseUniformPointScatter();
    void testNonUniformPointScatter();

}; // class TestPointScatter


void
TestPointScatter::testUniformPointScatter()
{
    const Index64 total = 50;
    const math::CoordBBox boxBounds(math::Coord(-1), math::Coord(1)); // 27 voxels across 8 leaves

    // Test the free function for all default grid types - 50 points across 27 voxels
    // ensures all voxels receive points

    {
        BoolGrid grid;
        grid.sparseFill(boxBounds, false, /*active*/true);
        auto points = points::uniformPointScatter(grid, total);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(total, pointCount(points->tree()));
    }
    {
        DoubleGrid grid;
        grid.sparseFill(boxBounds, 0.0, /*active*/true);
        auto points = points::uniformPointScatter(grid, total);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(total, pointCount(points->tree()));
    }
    {
        FloatGrid grid;
        grid.sparseFill(boxBounds, 0.0f, /*active*/true);
        auto points = points::uniformPointScatter(grid, total);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(total, pointCount(points->tree()));
    }
    {
        Int32Grid grid;
        grid.sparseFill(boxBounds, 0, /*active*/true);
        auto points = points::uniformPointScatter(grid, total);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(total, pointCount(points->tree()));
    }
    {
        Int64Grid grid;
        grid.sparseFill(boxBounds, 0, /*active*/true);
        auto points = points::uniformPointScatter(grid, total);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(total, pointCount(points->tree()));
    }
    {
        MaskGrid grid;
        grid.sparseFill(boxBounds, /*maskBuffer*/true);
        auto points = points::uniformPointScatter(grid, total);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(total, pointCount(points->tree()));
    }
    {
        StringGrid grid;
        grid.sparseFill(boxBounds, "", /*active*/true);
        auto points = points::uniformPointScatter(grid, total);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(total, pointCount(points->tree()));
    }
    {
        Vec3DGrid grid;
        grid.sparseFill(boxBounds, Vec3d(), /*active*/true);
        auto points = points::uniformPointScatter(grid, total);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(total, pointCount(points->tree()));
    }
    {
        Vec3IGrid grid;
        grid.sparseFill(boxBounds, Vec3i(), /*active*/true);
        auto points = points::uniformPointScatter(grid, total);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(total, pointCount(points->tree()));
    }
    {
        Vec3SGrid grid;
        grid.sparseFill(boxBounds, Vec3f(), /*active*/true);
        auto points = points::uniformPointScatter(grid, total);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(total, pointCount(points->tree()));
    }
    {
        PointDataGrid grid;
        grid.sparseFill(boxBounds, 0, /*active*/true);
        auto points = points::uniformPointScatter(grid, total);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(total, pointCount(points->tree()));
    }

    // Test 0 produces empty grid

    {
        BoolGrid grid;
        grid.sparseFill(boxBounds, false, /*active*/true);
        auto points = points::uniformPointScatter(grid, 0);
        CPPUNIT_ASSERT(points->empty());
    }

    // Test single point scatter and topology

    {
        BoolGrid grid;
        grid.sparseFill(boxBounds, false, /*active*/true);
        auto points = points::uniformPointScatter(grid, 1);
        CPPUNIT_ASSERT_EQUAL(Index32(1), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(1), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(1), pointCount(points->tree()));
    }

    // Test a grid containing tiles scatters correctly

    BoolGrid grid;
    grid.tree().addTile(/*level*/1, math::Coord(0), /*value*/true, /*active*/true);

    const Index32 NUM_VALUES = BoolGrid::TreeType::LeafNodeType::NUM_VALUES;

    CPPUNIT_ASSERT_EQUAL(Index64(NUM_VALUES), grid.activeVoxelCount());

    auto points = points::uniformPointScatter(grid, total);

#ifndef OPENVDB_2_ABI_COMPATIBLE
    CPPUNIT_ASSERT_EQUAL(Index64(0), points->tree().activeTileCount());
#endif
    CPPUNIT_ASSERT_EQUAL(Index32(1), points->tree().leafCount());
    CPPUNIT_ASSERT(Index64(NUM_VALUES) > points->tree().activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(total, pointCount(points->tree()));

    // Explicitly check P attribute

    const auto* attributeSet = &(points->tree().cbeginLeaf()->attributeSet());
    CPPUNIT_ASSERT_EQUAL(size_t(1), attributeSet->size());
    const auto* array = attributeSet->getConst(0);
    CPPUNIT_ASSERT(array);

    using PositionArrayT = TypedAttributeArray<Vec3f, NullCodec>;
    CPPUNIT_ASSERT(array->isType<PositionArrayT>());

    size_t size = array->size();
    CPPUNIT_ASSERT_EQUAL(size_t(total), size);

    AttributeHandle<Vec3f, NullCodec>::Ptr pHandle =
        AttributeHandle<Vec3f, NullCodec>::create(*array);
    for (size_t i = 0; i < size; ++i) {
        const Vec3f P = pHandle->get(Index(i));
        CPPUNIT_ASSERT(P[0] >=-0.5f);
        CPPUNIT_ASSERT(P[0] <= 0.5f);
        CPPUNIT_ASSERT(P[1] >=-0.5f);
        CPPUNIT_ASSERT(P[1] <= 0.5f);
        CPPUNIT_ASSERT(P[2] >=-0.5f);
        CPPUNIT_ASSERT(P[2] <= 0.5f);
    }

    // Test the rng seed

    const Vec3f firstPosition = pHandle->get(0);
    points = points::uniformPointScatter(grid, total, /*seed*/1);

    attributeSet = &(points->tree().cbeginLeaf()->attributeSet());
    CPPUNIT_ASSERT_EQUAL(size_t(1), attributeSet->size());

    array = attributeSet->getConst(0);
    CPPUNIT_ASSERT(array);
    CPPUNIT_ASSERT(array->isType<PositionArrayT>());

    size = array->size();
    CPPUNIT_ASSERT_EQUAL(size_t(total), size);
    pHandle = AttributeHandle<Vec3f, NullCodec>::create(*array);

    const Vec3f secondPosition = pHandle->get(0);
    CPPUNIT_ASSERT(!math::isExactlyEqual(firstPosition[0], secondPosition[0]));
    CPPUNIT_ASSERT(!math::isExactlyEqual(firstPosition[1], secondPosition[1]));
    CPPUNIT_ASSERT(!math::isExactlyEqual(firstPosition[2], secondPosition[2]));

    // Test spread

    points = points::uniformPointScatter(grid, total, /*seed*/1, /*spread*/0.2f);

    attributeSet = &(points->tree().cbeginLeaf()->attributeSet());
    CPPUNIT_ASSERT_EQUAL(size_t(1), attributeSet->size());
    array = attributeSet->getConst(0);
    CPPUNIT_ASSERT(array);
    CPPUNIT_ASSERT(array->isType<PositionArrayT>());

    size = array->size();
    CPPUNIT_ASSERT_EQUAL(size_t(total), size);

    pHandle = AttributeHandle<Vec3f, NullCodec>::create(*array);
    for (size_t i = 0; i < size; ++i) {
        const Vec3f P = pHandle->get(Index(i));
        CPPUNIT_ASSERT(P[0] >=-0.2f);
        CPPUNIT_ASSERT(P[0] <= 0.2f);
        CPPUNIT_ASSERT(P[1] >=-0.2f);
        CPPUNIT_ASSERT(P[1] <= 0.2f);
        CPPUNIT_ASSERT(P[2] >=-0.2f);
        CPPUNIT_ASSERT(P[2] <= 0.2f);
    }

    // Test mt11213b

    using mt11213b = std::mersenne_twister_engine<uint32_t, 32, 351, 175, 19,
        0xccab8ee7, 11, 0xffffffff, 7, 0x31b6ab00, 15, 0xffe50000, 17, 1812433253>;

    points = points::uniformPointScatter<BoolGrid, mt11213b>(grid, total);

    CPPUNIT_ASSERT_EQUAL(Index32(1), points->tree().leafCount());
    CPPUNIT_ASSERT(Index64(NUM_VALUES) > points->tree().activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(total, pointCount(points->tree()));

    // Test no remainder - grid contains one tile, scatter NUM_VALUES points

    points = points::uniformPointScatter(grid, Index64(NUM_VALUES));

    CPPUNIT_ASSERT_EQUAL(Index32(1), points->tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(Index64(NUM_VALUES), points->activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(Index64(NUM_VALUES), pointCount(points->tree()));

    const auto* const leaf = points->tree().probeConstLeaf(math::Coord(0));
    CPPUNIT_ASSERT(leaf);
    CPPUNIT_ASSERT(leaf->isDense());

    const auto* const data = leaf->buffer().data();
    CPPUNIT_ASSERT_EQUAL(Index32(1), Index32(data[1] - data[0]));

    for (size_t i = 1; i < NUM_VALUES; ++i) {
        const Index32 offset = data[i] - data[i - 1];
        CPPUNIT_ASSERT_EQUAL(Index32(1), offset);
    }
}

void
TestPointScatter::testDenseUniformPointScatter()
{
    const Index32 pointsPerVoxel = 8;
    const math::CoordBBox boxBounds(math::Coord(-1), math::Coord(1)); // 27 voxels across 8 leaves

    // Test the free function for all default grid types

    {
        BoolGrid grid;
        grid.sparseFill(boxBounds, false, /*active*/true);
        auto points = points::denseUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        DoubleGrid grid;
        grid.sparseFill(boxBounds, 0.0, /*active*/true);
        auto points = points::denseUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        FloatGrid grid;
        grid.sparseFill(boxBounds, 0.0f, /*active*/true);
        auto points = points::denseUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        Int32Grid grid;
        grid.sparseFill(boxBounds, 0, /*active*/true);
        auto points = points::denseUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        Int64Grid grid;
        grid.sparseFill(boxBounds, 0, /*active*/true);
        auto points = points::denseUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        MaskGrid grid;
        grid.sparseFill(boxBounds, /*maskBuffer*/true);
        auto points = points::denseUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        StringGrid grid;
        grid.sparseFill(boxBounds, "", /*active*/true);
        auto points = points::denseUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        Vec3DGrid grid;
        grid.sparseFill(boxBounds, Vec3d(), /*active*/true);
        auto points = points::denseUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        Vec3IGrid grid;
        grid.sparseFill(boxBounds, Vec3i(), /*active*/true);
        auto points = points::denseUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        Vec3SGrid grid;
        grid.sparseFill(boxBounds, Vec3f(), /*active*/true);
        auto points = points::denseUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        PointDataGrid grid;
        grid.sparseFill(boxBounds, 0, /*active*/true);
        auto points = points::denseUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }

    // Test 0 produces empty grid

    {
        BoolGrid grid;
        grid.sparseFill(boxBounds, false, /*active*/true);
        auto points = points::denseUniformPointScatter(grid, 0.0f);
        CPPUNIT_ASSERT(points->empty());
    }

    // Test topology between 0 - 1

    {
        BoolGrid grid;
        grid.sparseFill(boxBounds, false, /*active*/true);
        auto points = points::denseUniformPointScatter(grid, 0.8f);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        // Note that a value of 22 is precomputed as the number of active
        // voxels/points produced by a value of 0.8
        CPPUNIT_ASSERT_EQUAL(Index64(22), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(22), pointCount(points->tree()));

        // Test below 0 throws

        CPPUNIT_ASSERT_THROW(points::denseUniformPointScatter(grid, -0.1f), openvdb::ValueError);
    }

    // Test a grid containing tiles scatters correctly

    BoolGrid grid;
    grid.tree().addTile(/*level*/1, math::Coord(0), /*value*/true, /*active*/true);
    grid.tree().setValueOn(math::Coord(8,0,0)); // add another leaf

    const Index32 NUM_VALUES = BoolGrid::TreeType::LeafNodeType::NUM_VALUES;

    CPPUNIT_ASSERT_EQUAL(Index32(1), grid.tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(Index64(NUM_VALUES + 1), grid.activeVoxelCount());

    auto points = points::denseUniformPointScatter(grid, pointsPerVoxel);

    const Index64 expectedCount = Index64(pointsPerVoxel * (NUM_VALUES + 1));

#ifndef OPENVDB_2_ABI_COMPATIBLE
    CPPUNIT_ASSERT_EQUAL(Index64(0), points->tree().activeTileCount());
#endif
    CPPUNIT_ASSERT_EQUAL(Index32(2), points->tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(Index64(NUM_VALUES + 1), points->activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(expectedCount, pointCount(points->tree()));

    // Explicitly check P attribute

    const auto* attributeSet = &(points->tree().cbeginLeaf()->attributeSet());
    CPPUNIT_ASSERT_EQUAL(size_t(1), attributeSet->size());
    const auto* array = attributeSet->getConst(0);
    CPPUNIT_ASSERT(array);

    using PositionArrayT = TypedAttributeArray<Vec3f, NullCodec>;
    CPPUNIT_ASSERT(array->isType<PositionArrayT>());

    size_t size = array->size();
    CPPUNIT_ASSERT_EQUAL(size_t(pointsPerVoxel * NUM_VALUES), size);

    AttributeHandle<Vec3f, NullCodec>::Ptr pHandle =
        AttributeHandle<Vec3f, NullCodec>::create(*array);
    for (size_t i = 0; i < size; ++i) {
        const Vec3f P = pHandle->get(Index(i));
        CPPUNIT_ASSERT(P[0] >=-0.5f);
        CPPUNIT_ASSERT(P[0] <= 0.5f);
        CPPUNIT_ASSERT(P[1] >=-0.5f);
        CPPUNIT_ASSERT(P[1] <= 0.5f);
        CPPUNIT_ASSERT(P[2] >=-0.5f);
        CPPUNIT_ASSERT(P[2] <= 0.5f);
    }

    // Test the rng seed

    const Vec3f firstPosition = pHandle->get(0);
    points = points::denseUniformPointScatter(grid, pointsPerVoxel, /*seed*/1);

    attributeSet = &(points->tree().cbeginLeaf()->attributeSet());
    CPPUNIT_ASSERT_EQUAL(size_t(1), attributeSet->size());

    array = attributeSet->getConst(0);
    CPPUNIT_ASSERT(array);
    CPPUNIT_ASSERT(array->isType<PositionArrayT>());

    size = array->size();
    CPPUNIT_ASSERT_EQUAL(size_t(pointsPerVoxel * NUM_VALUES), size);
    pHandle = AttributeHandle<Vec3f, NullCodec>::create(*array);

    const Vec3f secondPosition = pHandle->get(0);
    CPPUNIT_ASSERT(!math::isExactlyEqual(firstPosition[0], secondPosition[0]));
    CPPUNIT_ASSERT(!math::isExactlyEqual(firstPosition[1], secondPosition[1]));
    CPPUNIT_ASSERT(!math::isExactlyEqual(firstPosition[2], secondPosition[2]));

    // Test spread

    points = points::denseUniformPointScatter(grid, pointsPerVoxel, /*seed*/1, /*spread*/0.2f);

    attributeSet = &(points->tree().cbeginLeaf()->attributeSet());
    CPPUNIT_ASSERT_EQUAL(size_t(1), attributeSet->size());
    array = attributeSet->getConst(0);
    CPPUNIT_ASSERT(array);
    CPPUNIT_ASSERT(array->isType<PositionArrayT>());

    size = array->size();
    CPPUNIT_ASSERT_EQUAL(size_t(pointsPerVoxel * NUM_VALUES), size);

    pHandle = AttributeHandle<Vec3f, NullCodec>::create(*array);
    for (size_t i = 0; i < size; ++i) {
        const Vec3f P = pHandle->get(Index(i));
        CPPUNIT_ASSERT(P[0] >=-0.2f);
        CPPUNIT_ASSERT(P[0] <= 0.2f);
        CPPUNIT_ASSERT(P[1] >=-0.2f);
        CPPUNIT_ASSERT(P[1] <= 0.2f);
        CPPUNIT_ASSERT(P[2] >=-0.2f);
        CPPUNIT_ASSERT(P[2] <= 0.2f);
    }

    // Test mt11213b

    using mt11213b = std::mersenne_twister_engine<uint32_t, 32, 351, 175, 19,
        0xccab8ee7, 11, 0xffffffff, 7, 0x31b6ab00, 15, 0xffe50000, 17, 1812433253>;

    points = points::denseUniformPointScatter<BoolGrid, mt11213b>(grid, pointsPerVoxel);

    CPPUNIT_ASSERT_EQUAL(Index32(2), points->tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(Index64(NUM_VALUES + 1), points->activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(expectedCount, pointCount(points->tree()));
}

void
TestPointScatter::testNonUniformPointScatter()
{
    const Index32 pointsPerVoxel = 8;
    const math::CoordBBox totalBoxBounds(math::Coord(-2), math::Coord(2)); // 125 voxels across 8 leaves
    const math::CoordBBox activeBoxBounds(math::Coord(-1), math::Coord(1)); // 27 voxels across 8 leaves

    // Test the free function for all default scalar grid types

    {
        BoolGrid grid;
        grid.sparseFill(totalBoxBounds, false, /*active*/true);
        grid.sparseFill(activeBoxBounds, true, /*active*/true);
        auto points = points::nonUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        DoubleGrid grid;
        grid.sparseFill(totalBoxBounds, 0.0, /*active*/true);
        grid.sparseFill(activeBoxBounds, 1.0, /*active*/true);
        auto points = points::nonUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        FloatGrid grid;
        grid.sparseFill(totalBoxBounds, 0.0f, /*active*/true);
        grid.sparseFill(activeBoxBounds, 1.0f, /*active*/true);
        auto points = points::nonUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        Int32Grid grid;
        grid.sparseFill(totalBoxBounds, 0, /*active*/true);
        grid.sparseFill(activeBoxBounds, 1, /*active*/true);
        auto points = points::nonUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        Int64Grid grid;
        grid.sparseFill(totalBoxBounds, 0, /*active*/true);
        grid.sparseFill(activeBoxBounds, 1, /*active*/true);
        auto points = points::nonUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }
    {
        MaskGrid grid;
        grid.sparseFill(totalBoxBounds, /*maskBuffer*/0);
        grid.sparseFill(activeBoxBounds, /*maskBuffer*/1);
        auto points = points::nonUniformPointScatter(grid, pointsPerVoxel);
        CPPUNIT_ASSERT_EQUAL(Index32(8), points->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index64(27), points->activeVoxelCount());
        CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 27), pointCount(points->tree()));
    }

    BoolGrid grid;

    // Test below 0 throws

    CPPUNIT_ASSERT_THROW(points::nonUniformPointScatter(grid, -0.1f), openvdb::ValueError);

    // Test a grid containing tiles scatters correctly

    grid.tree().addTile(/*level*/1, math::Coord(0), /*value*/true, /*active*/true);
    grid.tree().setValueOn(math::Coord(8,0,0), true); // add another leaf

    const Index32 NUM_VALUES = BoolGrid::TreeType::LeafNodeType::NUM_VALUES;

    CPPUNIT_ASSERT_EQUAL(Index32(1), grid.tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(Index64(NUM_VALUES + 1), grid.activeVoxelCount());

    auto points = points::nonUniformPointScatter(grid, pointsPerVoxel);

    const Index64 expectedCount = Index64(pointsPerVoxel * (NUM_VALUES + 1));

#ifndef OPENVDB_2_ABI_COMPATIBLE
    CPPUNIT_ASSERT_EQUAL(Index64(0), points->tree().activeTileCount());
#endif
    CPPUNIT_ASSERT_EQUAL(Index32(2), points->tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(Index64(NUM_VALUES + 1), points->activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(expectedCount, pointCount(points->tree()));

    // Explicitly check P attribute

    const auto* attributeSet = &(points->tree().cbeginLeaf()->attributeSet());
    CPPUNIT_ASSERT_EQUAL(size_t(1), attributeSet->size());
    const auto* array = attributeSet->getConst(0);
    CPPUNIT_ASSERT(array);

    using PositionArrayT = TypedAttributeArray<Vec3f, NullCodec>;
    CPPUNIT_ASSERT(array->isType<PositionArrayT>());

    size_t size = array->size();
    CPPUNIT_ASSERT_EQUAL(size_t(pointsPerVoxel * NUM_VALUES), size);

    AttributeHandle<Vec3f, NullCodec>::Ptr pHandle =
        AttributeHandle<Vec3f, NullCodec>::create(*array);
    for (size_t i = 0; i < size; ++i) {
        const Vec3f P = pHandle->get(Index(i));
        CPPUNIT_ASSERT(P[0] >=-0.5f);
        CPPUNIT_ASSERT(P[0] <= 0.5f);
        CPPUNIT_ASSERT(P[1] >=-0.5f);
        CPPUNIT_ASSERT(P[1] <= 0.5f);
        CPPUNIT_ASSERT(P[2] >=-0.5f);
        CPPUNIT_ASSERT(P[2] <= 0.5f);
    }

    // Test the rng seed

    const Vec3f firstPosition = pHandle->get(0);
    points = points::nonUniformPointScatter(grid, pointsPerVoxel, /*seed*/1);

    attributeSet = &(points->tree().cbeginLeaf()->attributeSet());
    CPPUNIT_ASSERT_EQUAL(size_t(1), attributeSet->size());

    array = attributeSet->getConst(0);
    CPPUNIT_ASSERT(array);
    CPPUNIT_ASSERT(array->isType<PositionArrayT>());

    size = array->size();
    CPPUNIT_ASSERT_EQUAL(size_t(pointsPerVoxel * NUM_VALUES), size);
    pHandle = AttributeHandle<Vec3f, NullCodec>::create(*array);

    const Vec3f secondPosition = pHandle->get(0);
    CPPUNIT_ASSERT(!math::isExactlyEqual(firstPosition[0], secondPosition[0]));
    CPPUNIT_ASSERT(!math::isExactlyEqual(firstPosition[1], secondPosition[1]));
    CPPUNIT_ASSERT(!math::isExactlyEqual(firstPosition[2], secondPosition[2]));

    // Test spread

    points = points::nonUniformPointScatter(grid, pointsPerVoxel, /*seed*/1, /*spread*/0.2f);

    attributeSet = &(points->tree().cbeginLeaf()->attributeSet());
    CPPUNIT_ASSERT_EQUAL(size_t(1), attributeSet->size());
    array = attributeSet->getConst(0);
    CPPUNIT_ASSERT(array);
    CPPUNIT_ASSERT(array->isType<PositionArrayT>());

    size = array->size();
    CPPUNIT_ASSERT_EQUAL(size_t(pointsPerVoxel * NUM_VALUES), size);

    pHandle = AttributeHandle<Vec3f, NullCodec>::create(*array);
    for (size_t i = 0; i < size; ++i) {
        const Vec3f P = pHandle->get(Index(i));
        CPPUNIT_ASSERT(P[0] >=-0.2f);
        CPPUNIT_ASSERT(P[0] <= 0.2f);
        CPPUNIT_ASSERT(P[1] >=-0.2f);
        CPPUNIT_ASSERT(P[1] <= 0.2f);
        CPPUNIT_ASSERT(P[2] >=-0.2f);
        CPPUNIT_ASSERT(P[2] <= 0.2f);
    }

    // Test varying counts

    Int32Grid countGrid;

    // tets negative values equate to 0
    countGrid.tree().setValueOn(Coord(0), -1);
    for (int i = 1; i < 8; ++i) {
        countGrid.tree().setValueOn(Coord(i), i);
    }

    points = points::nonUniformPointScatter(countGrid, pointsPerVoxel);

    CPPUNIT_ASSERT_EQUAL(Index32(1), points->tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(Index64(7), points->activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(Index64(pointsPerVoxel * 28), pointCount(points->tree()));

    for (int i = 1; i < 8; ++i) {
        CPPUNIT_ASSERT(points->tree().isValueOn(Coord(i)));
        auto& value = points->tree().getValue(Coord(i));
        Index32 expected(0);
        for (Index32 j = i; j > 0; --j) expected += j;
        CPPUNIT_ASSERT_EQUAL(Index32(expected * pointsPerVoxel), Index32(value));
    }

    // Test mt11213b

    using mt11213b = std::mersenne_twister_engine<uint32_t, 32, 351, 175, 19,
        0xccab8ee7, 11, 0xffffffff, 7, 0x31b6ab00, 15, 0xffe50000, 17, 1812433253>;

    points = points::nonUniformPointScatter<BoolGrid, mt11213b>(grid, pointsPerVoxel);

    CPPUNIT_ASSERT_EQUAL(Index32(2), points->tree().leafCount());
    CPPUNIT_ASSERT_EQUAL(Index64(NUM_VALUES + 1), points->activeVoxelCount());
    CPPUNIT_ASSERT_EQUAL(expectedCount, pointCount(points->tree()));
}


CPPUNIT_TEST_SUITE_REGISTRATION(TestPointScatter);

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
