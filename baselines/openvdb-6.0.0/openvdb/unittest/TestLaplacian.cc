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

#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include "util.h" // for unittest_util::makeSphere()
#include <cppunit/extensions/HelperMacros.h>
#include <sstream>


class TestLaplacian: public CppUnit::TestFixture
{
public:
    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestLaplacian);
    CPPUNIT_TEST(testISLaplacian);                    // Laplacian in Index Space
    CPPUNIT_TEST(testISLaplacianStencil);
    CPPUNIT_TEST(testWSLaplacian);                    // Laplacian in World Space
    CPPUNIT_TEST(testWSLaplacianFrustum);             // Laplacian in World Space
    CPPUNIT_TEST(testWSLaplacianStencil);
    CPPUNIT_TEST(testLaplacianTool);                  // Laplacian tool
    CPPUNIT_TEST(testLaplacianMaskedTool);                  // Laplacian tool
    CPPUNIT_TEST(testOldStyleStencils);               // old stencil impl
    CPPUNIT_TEST_SUITE_END();

    void testISLaplacian();
    void testISLaplacianStencil();
    void testWSLaplacian();
    void testWSLaplacianFrustum();
    void testWSLaplacianStencil();
    void testLaplacianTool();
    void testLaplacianMaskedTool();
    void testOldStyleStencils();
};


CPPUNIT_TEST_SUITE_REGISTRATION(TestLaplacian);

void
TestLaplacian::testISLaplacian()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
    FloatTree& tree = grid->tree();
    CPPUNIT_ASSERT(tree.empty());

    const Coord dim(64,64,64);
    const Coord c(35,30,40);
    const openvdb::Vec3f
        center(static_cast<float>(c[0]), static_cast<float>(c[1]), static_cast<float>(c[2]));
    const float radius=0.0f;//point at {35,30,40}
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);
    CPPUNIT_ASSERT(!tree.empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));

    Coord xyz(35,10,40);

    // Index Space Laplacian random access
    FloatGrid::ConstAccessor inAccessor = grid->getConstAccessor();
    FloatGrid::ValueType result;
    result = math::ISLaplacian<math::CD_SECOND>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20.0, result, /*tolerance=*/0.01);

    result = math::ISLaplacian<math::CD_FOURTH>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20.0, result, /*tolerance=*/0.01);

    result = math::ISLaplacian<math::CD_SIXTH>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20.0, result, /*tolerance=*/0.01);
}


void
TestLaplacian::testISLaplacianStencil()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
    FloatTree& tree = grid->tree();
    CPPUNIT_ASSERT(tree.empty());

    const Coord dim(64,64,64);
    const Coord c(35,30,40);
    const openvdb::Vec3f
        center(static_cast<float>(c[0]), static_cast<float>(c[1]), static_cast<float>(c[2]));
    const float radius=0;//point at {35,30,40}
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);
    CPPUNIT_ASSERT(!tree.empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));

    Coord xyz(35,10,40);

    // Index Space Laplacian stencil access
    FloatGrid::ValueType result;

    math::SevenPointStencil<FloatGrid> sevenpt(*grid);
    sevenpt.moveTo(xyz);
    result = math::ISLaplacian<math::CD_SECOND>::result(sevenpt);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20.0, result, /*tolerance=*/0.01);

    math::ThirteenPointStencil<FloatGrid> thirteenpt(*grid);
    thirteenpt.moveTo(xyz);
    result = math::ISLaplacian<math::CD_FOURTH>::result(thirteenpt);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20.0, result, /*tolerance=*/0.01);

    math::NineteenPointStencil<FloatGrid> nineteenpt(*grid);
    nineteenpt.moveTo(xyz);
    result = math::ISLaplacian<math::CD_SIXTH>::result(nineteenpt);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20.0, result, /*tolerance=*/0.01);
}


void
TestLaplacian::testWSLaplacian()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
    FloatTree& tree = grid->tree();
    CPPUNIT_ASSERT(tree.empty());

    const Coord dim(64,64,64);
    const Coord c(35,30,40);
    const openvdb::Vec3f
        center(static_cast<float>(c[0]), static_cast<float>(c[1]), static_cast<float>(c[2]));
    const float radius=0.0f;//point at {35,30,40}
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!tree.empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));

    Coord xyz(35,10,40);

    FloatGrid::ValueType result;
    FloatGrid::ConstAccessor inAccessor = grid->getConstAccessor();

    // try with a map
    math::UniformScaleMap map;
    math::MapBase::Ptr rotated_map = map.preRotate(1.5, math::X_AXIS);
    // verify the new map is an affine map
    CPPUNIT_ASSERT(rotated_map->type() == math::AffineMap::mapType());
    math::AffineMap::Ptr affine_map = StaticPtrCast<math::AffineMap, math::MapBase>(rotated_map);

    // the laplacian is invariant to rotation
    result = math::Laplacian<math::AffineMap, math::CD_SECOND>::result(
        *affine_map, inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);
    result = math::Laplacian<math::AffineMap, math::CD_FOURTH>::result(
        *affine_map, inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);
    result = math::Laplacian<math::AffineMap, math::CD_SIXTH>::result(
        *affine_map, inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);

    // test uniform map
    math::UniformScaleMap uniform;

    result = math::Laplacian<math::UniformScaleMap, math::CD_SECOND>::result(
        uniform, inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);
    result = math::Laplacian<math::UniformScaleMap, math::CD_FOURTH>::result(
        uniform, inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);
    result = math::Laplacian<math::UniformScaleMap, math::CD_SIXTH>::result(
        uniform, inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);

    // test the GenericMap Grid interface
    {
        math::GenericMap generic_map(*grid);
        result = math::Laplacian<math::GenericMap, math::CD_SECOND>::result(
            generic_map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);

        result = math::Laplacian<math::GenericMap, math::CD_FOURTH>::result(
            generic_map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);
    }
    {
        // test the GenericMap Transform interface
        math::GenericMap generic_map(grid->transform());
        result = math::Laplacian<math::GenericMap, math::CD_SECOND>::result(
            generic_map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);

    }
    {
        // test the GenericMap Map interface
        math::GenericMap generic_map(rotated_map);
        result = math::Laplacian<math::GenericMap, math::CD_SECOND>::result(
            generic_map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);
    }

}


void
TestLaplacian::testWSLaplacianFrustum()
{
    using namespace openvdb;

    // Create a Frustum Map:

    openvdb::BBoxd bbox(Vec3d(0), Vec3d(100));
    math::NonlinearFrustumMap frustum(bbox, 1./6., 5);
    /// frustum will have depth, far plane - near plane = 5
    /// the frustum has width 1 in the front and 6 in the back

    math::Vec3d trans(2,2,2);
    math::NonlinearFrustumMap::Ptr map =
        StaticPtrCast<math::NonlinearFrustumMap, math::MapBase>(
            frustum.preScale(Vec3d(10,10,10))->postTranslate(trans));

    CPPUNIT_ASSERT(!map->hasUniformScale());

    math::Vec3d result;
    result = map->voxelSize();

    CPPUNIT_ASSERT( math::isApproxEqual(result.x(), 0.1));
    CPPUNIT_ASSERT( math::isApproxEqual(result.y(), 0.1));
    CPPUNIT_ASSERT( math::isApproxEqual(result.z(), 0.5, 0.0001));


    // Create a tree
    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/0.0);
    FloatTree& tree = grid->tree();
    CPPUNIT_ASSERT(tree.empty());

    // Load cos(x)sin(y)cos(z)
    Coord ijk(10,10,10);
    for (Int32& i=ijk.x(); i < 20; ++i) {
        for (Int32& j=ijk.y(); j < 20; ++j) {
            for (Int32& k=ijk.z(); k < 20; ++k) {
                // world space image of the ijk coord
                const Vec3d ws = map->applyMap(ijk.asVec3d());
                const float value = float(cos(ws.x() ) * sin( ws.y()) * cos(ws.z()));
                tree.setValue(ijk, value);
            }
        }
    }

    const Coord testloc(16,16,16);
    float test_result = math::Laplacian<math::NonlinearFrustumMap, math::CD_SECOND>::result(
        *map, tree, testloc);
    float expected_result =  -3.f * tree.getValue(testloc);

    // The exact solution of Laplacian( cos(x)sin(y)cos(z) ) = -3 cos(x) sin(y) cos(z)

    CPPUNIT_ASSERT( math::isApproxEqual(test_result, expected_result, /*tolerance=*/0.02f) );
}


void
TestLaplacian::testWSLaplacianStencil()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
    FloatTree& tree = grid->tree();
    CPPUNIT_ASSERT(tree.empty());

    const Coord dim(64,64,64);
    const Coord c(35,30,40);
    const openvdb::Vec3f
        center(static_cast<float>(c[0]), static_cast<float>(c[1]), static_cast<float>(c[2]));
    const float radius=0.0f;//point at {35,30,40}
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!tree.empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));

    Coord xyz(35,10,40);

    FloatGrid::ValueType result;

    // try with a map
    math::UniformScaleMap map;
    math::MapBase::Ptr rotated_map = map.preRotate(1.5, math::X_AXIS);
    // verify the new map is an affine map
    CPPUNIT_ASSERT(rotated_map->type() == math::AffineMap::mapType());
    math::AffineMap::Ptr affine_map = StaticPtrCast<math::AffineMap, math::MapBase>(rotated_map);

    // the laplacian is invariant to rotation
    math::SevenPointStencil<FloatGrid> sevenpt(*grid);
    math::ThirteenPointStencil<FloatGrid> thirteenpt(*grid);
    math::NineteenPointStencil<FloatGrid> nineteenpt(*grid);
    math::SecondOrderDenseStencil<FloatGrid> dense_2nd(*grid);
    math::FourthOrderDenseStencil<FloatGrid> dense_4th(*grid);
    math::SixthOrderDenseStencil<FloatGrid> dense_6th(*grid);
    sevenpt.moveTo(xyz);
    thirteenpt.moveTo(xyz);
    nineteenpt.moveTo(xyz);
    dense_2nd.moveTo(xyz);
    dense_4th.moveTo(xyz);
    dense_6th.moveTo(xyz);

    result = math::Laplacian<math::AffineMap, math::CD_SECOND>::result(*affine_map, dense_2nd);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);
    result = math::Laplacian<math::AffineMap, math::CD_FOURTH>::result(*affine_map, dense_4th);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);
    result = math::Laplacian<math::AffineMap, math::CD_SIXTH>::result(*affine_map, dense_6th);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);

    // test uniform map
    math::UniformScaleMap uniform;

    result = math::Laplacian<math::UniformScaleMap, math::CD_SECOND>::result(uniform, sevenpt);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);
    result = math::Laplacian<math::UniformScaleMap, math::CD_FOURTH>::result(uniform, thirteenpt);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);
    result = math::Laplacian<math::UniformScaleMap, math::CD_SIXTH>::result(uniform, nineteenpt);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);

    // test the GenericMap Grid interface
    {
        math::GenericMap generic_map(*grid);
        result = math::Laplacian<math::GenericMap, math::CD_SECOND>::result(generic_map, dense_2nd);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);

        result = math::Laplacian<math::GenericMap, math::CD_FOURTH>::result(generic_map, dense_4th);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);
    }
    {
        // test the GenericMap Transform interface
        math::GenericMap generic_map(grid->transform());
        result = math::Laplacian<math::GenericMap, math::CD_SECOND>::result(generic_map, dense_2nd);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);

    }
    {
        // test the GenericMap Map interface
        math::GenericMap generic_map(rotated_map);
        result = math::Laplacian<math::GenericMap, math::CD_SECOND>::result(generic_map, dense_2nd);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/20., result, /*tolerance=*/0.01);
    }
}


void
TestLaplacian::testOldStyleStencils()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = FloatGrid::create(/*backgroundValue=*/5.0);
    grid->setTransform(math::Transform::createLinearTransform(/*voxel size=*/0.5));
    CPPUNIT_ASSERT(grid->empty());

    const Coord dim(32, 32, 32);
    const Coord c(35,30,40);
    const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
    const float radius=10.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!grid->empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(grid->activeVoxelCount()));

    math::GradStencil<FloatGrid> gs(*grid);
    math::WenoStencil<FloatGrid> ws(*grid);
    math::CurvatureStencil<FloatGrid> cs(*grid);

    Coord xyz(20,16,20);//i.e. 8 voxel or 4 world units away from the center
    gs.moveTo(xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/4.0, gs.laplacian(), 0.01);// 2/distance from center

    ws.moveTo(xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/4.0, ws.laplacian(), 0.01);// 2/distance from center

    cs.moveTo(xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/4.0, cs.laplacian(), 0.01);// 2/distance from center

    xyz.reset(12,16,10);//i.e. 10 voxel or 5 world units away from the center
    gs.moveTo(xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/5.0, gs.laplacian(), 0.01);// 2/distance from center

    ws.moveTo(xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/5.0, ws.laplacian(), 0.01);// 2/distance from center

    cs.moveTo(xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0/5.0, cs.laplacian(), 0.01);// 2/distance from center
}


void
TestLaplacian::testLaplacianTool()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
    FloatTree& tree = grid->tree();
    CPPUNIT_ASSERT(tree.empty());

    const Coord dim(64, 64, 64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);
    const float radius=0.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!tree.empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));
    FloatGrid::Ptr lap = tools::laplacian(*grid);
    CPPUNIT_ASSERT_EQUAL(int(tree.activeVoxelCount()), int(lap->activeVoxelCount()));

    Coord xyz(35,30,30);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(
        2.0/10.0, lap->getConstAccessor().getValue(xyz), 0.01);// 2/distance from center

    xyz.reset(35,10,40);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(
        2.0/20.0, lap->getConstAccessor().getValue(xyz),0.01);// 2/distance from center
}

void
TestLaplacian::testLaplacianMaskedTool()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
    FloatTree& tree = grid->tree();
    CPPUNIT_ASSERT(tree.empty());

    const Coord dim(64, 64, 64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);
    const float radius=0.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!tree.empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));

    const openvdb::CoordBBox maskbbox(openvdb::Coord(35, 30, 30), openvdb::Coord(41, 41, 41));
    BoolGrid::Ptr maskGrid = BoolGrid::create(false);
    maskGrid->fill(maskbbox, true/*value*/, true/*activate*/);


    FloatGrid::Ptr lap = tools::laplacian(*grid, *maskGrid);

    {// outside the masked region
        Coord xyz(34,30,30);

        CPPUNIT_ASSERT(!maskbbox.isInside(xyz));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            0, lap->getConstAccessor().getValue(xyz), 0.01);// 2/distance from center

        xyz.reset(35,10,40);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            0, lap->getConstAccessor().getValue(xyz),0.01);// 2/distance from center
    }

    {// inside the masked region
        Coord xyz(35,30,30);

        CPPUNIT_ASSERT(maskbbox.isInside(xyz));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            2.0/10.0, lap->getConstAccessor().getValue(xyz), 0.01);// 2/distance from center

    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
