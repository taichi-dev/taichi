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


class TestGradient: public CppUnit::TestFixture
{
public:
    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestGradient);
    CPPUNIT_TEST(testISGradient);               // gradient in index space
    CPPUNIT_TEST(testISGradientStencil);
    CPPUNIT_TEST(testWSGradient);               // gradient in world space
    CPPUNIT_TEST(testWSGradientStencil);
    CPPUNIT_TEST(testWSGradientStencilFrustum);
    CPPUNIT_TEST(testWSGradientNormSqr);        // gradient norm sqr (world space only)
    CPPUNIT_TEST(testWSGradientNormSqrStencil); // gradient norm sqr (world space only)
    CPPUNIT_TEST(testGradientTool);             // gradient tool
    CPPUNIT_TEST(testGradientMaskedTool);       // gradient tool
    CPPUNIT_TEST(testIntersectsIsoValue);       // zero-crossing
    CPPUNIT_TEST(testOldStyleStencils);         // old stencil impl - deprecate

    CPPUNIT_TEST_SUITE_END();

    void testISGradient();
    void testISGradientStencil();
    void testWSGradient();
    void testWSGradientStencilFrustum();
    void testWSGradientStencil();
    void testWSGradientNormSqr();
    void testWSGradientNormSqrStencil();
    void testGradientTool();
    void testGradientMaskedTool();
    void testIntersectsIsoValue();
    void testOldStyleStencils();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestGradient);


void
TestGradient::testISGradient()
{
    using namespace openvdb;

    using AccessorType = FloatGrid::ConstAccessor;
    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
    FloatTree& tree = grid->tree();

    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f ,30.0f, 40.0f);
    const float radius=10.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!tree.empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));
    const Coord xyz(10, 20, 30);


    // Index Space Gradients: random access and stencil version
    AccessorType inAccessor = grid->getConstAccessor();
    Vec3f result;
    result = math::ISGradient<math::CD_2ND>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::CD_4TH>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::CD_6TH>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::FD_1ST>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.02);

    result = math::ISGradient<math::FD_2ND>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::FD_3RD>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::BD_1ST>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.02);

    result = math::ISGradient<math::BD_2ND>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::BD_3RD>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::FD_WENO5>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::BD_WENO5>::result(inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
}


void
TestGradient::testISGradientStencil()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
    FloatTree& tree = grid->tree();

    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f ,30.0f, 40.0f);
    const float radius = 10.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!tree.empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));
    const Coord xyz(10, 20, 30);


    // Index Space Gradients: stencil version
    Vec3f result;
    // this stencil is large enough for all thie different schemes used
    // in this test
    math::NineteenPointStencil<FloatGrid> stencil(*grid);
    stencil.moveTo(xyz);

    result = math::ISGradient<math::CD_2ND>::result(stencil);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::CD_4TH>::result(stencil);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::CD_6TH>::result(stencil);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::FD_1ST>::result(stencil);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.02);

    result = math::ISGradient<math::FD_2ND>::result(stencil);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::FD_3RD>::result(stencil);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::BD_1ST>::result(stencil);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.02);

    result = math::ISGradient<math::BD_2ND>::result(stencil);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::BD_3RD>::result(stencil);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::FD_WENO5>::result(stencil);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    result = math::ISGradient<math::BD_WENO5>::result(stencil);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
}


void
TestGradient::testWSGradient()
{
    using namespace openvdb;

    using AccessorType = FloatGrid::ConstAccessor;

    double voxel_size = 0.5;
    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
    grid->setTransform(math::Transform::createLinearTransform(voxel_size));
    CPPUNIT_ASSERT(grid->empty());

    const openvdb::Coord dim(32,32,32);
    const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
    const float radius = 10.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!grid->empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(grid->activeVoxelCount()));
    const Coord xyz(11, 17, 26);

    AccessorType inAccessor = grid->getConstAccessor();
    // try with a map

    // Index Space Gradients: stencil version
    Vec3f result;
    math::MapBase::Ptr rotated_map;
    {
        math::UniformScaleMap map(voxel_size);
        result = math::Gradient<math::UniformScaleMap, math::CD_2ND>::result(
            map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
        rotated_map = map.preRotate(1.5, math::X_AXIS);
        // verify the new map is an affine map
        CPPUNIT_ASSERT(rotated_map->type() == math::AffineMap::mapType());
        math::AffineMap::Ptr affine_map =
            StaticPtrCast<math::AffineMap, math::MapBase>(rotated_map);
        // the gradient should have the same length even after rotation
        result = math::Gradient<math::AffineMap, math::CD_2ND>::result(
            *affine_map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
        result = math::Gradient<math::AffineMap, math::CD_4TH>::result(
            *affine_map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
    {
        math::UniformScaleTranslateMap map(voxel_size, Vec3d(0,0,0));
        result = math::Gradient<math::UniformScaleTranslateMap, math::CD_2ND>::result(
            map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
    {
        math::ScaleTranslateMap map(Vec3d(voxel_size, voxel_size, voxel_size), Vec3d(0,0,0));
        result = math::Gradient<math::ScaleTranslateMap, math::CD_2ND>::result(
            map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }

    {
        // this map has no scale, expect result/voxel_spaceing = 1
        math::TranslationMap map;
        result = math::Gradient<math::TranslationMap, math::CD_2ND>::result(map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(voxel_size, result.length(), /*tolerance=*/0.01);
    }

    {
        // test the GenericMap Grid interface
        math::GenericMap generic_map(*grid);
        result = math::Gradient<math::GenericMap, math::CD_2ND>::result(
            generic_map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
    {
        // test the GenericMap Transform interface
        math::GenericMap generic_map(grid->transform());
        result = math::Gradient<math::GenericMap, math::CD_2ND>::result(
            generic_map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
    {
        // test the GenericMap Map interface
        math::GenericMap generic_map(rotated_map);
        result = math::Gradient<math::GenericMap, math::CD_2ND>::result(
            generic_map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
    {
        // test a map with non-uniform SCALING AND ROTATION
        Vec3d voxel_sizes(0.25, 0.45, 0.75);
        math::MapBase::Ptr base_map( new math::ScaleMap(voxel_sizes));
        // apply rotation
        rotated_map = base_map->preRotate(1.5, math::X_AXIS);
        grid->setTransform(math::Transform::Ptr(new math::Transform(rotated_map)));
        // remake the sphere
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        math::AffineMap::Ptr affine_map =
            StaticPtrCast<math::AffineMap, math::MapBase>(rotated_map);

        // math::ScaleMap map(voxel_sizes);
        result = math::Gradient<math::AffineMap, math::CD_2ND>::result(
            *affine_map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
    {
        // test a map with non-uniform SCALING
        Vec3d voxel_sizes(0.25, 0.45, 0.75);
        math::MapBase::Ptr base_map( new math::ScaleMap(voxel_sizes));
        grid->setTransform(math::Transform::Ptr(new math::Transform(base_map)));
        // remake the sphere
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);
        math::ScaleMap::Ptr scale_map = StaticPtrCast<math::ScaleMap, math::MapBase>(base_map);

        // math::ScaleMap map(voxel_sizes);
        result = math::Gradient<math::ScaleMap, math::CD_2ND>::result(*scale_map, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
}

void
TestGradient::testWSGradientStencilFrustum()
{
    using namespace openvdb;

    // Construct a frustum that matches the one in TestMaps::testFrustum()

    openvdb::BBoxd bbox(Vec3d(0), Vec3d(100));
    math::NonlinearFrustumMap frustum(bbox, 1./6., 5);
    /// frustum will have depth, far plane - near plane = 5
    /// the frustum has width 1 in the front and 6 in the back

    Vec3d trans(2,2,2);
    math::NonlinearFrustumMap::Ptr map =
        StaticPtrCast<math::NonlinearFrustumMap, math::MapBase>(
            frustum.preScale(Vec3d(10,10,10))->postTranslate(trans));


    // Create a grid with this frustum

    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/0.f);
    math::Transform::Ptr transform = math::Transform::Ptr( new math::Transform(map));
    grid->setTransform(transform);

    FloatGrid::Accessor acc = grid->getAccessor();
    // Totally fill the interior of the frustum with word space distances
    // from its center.


    math::Vec3d isCenter(.5 * 101, .5 * 101, .5 * 101);
    math::Vec3d wsCenter = map->applyMap(isCenter);

    math::Coord ijk;

    // convert to IntType
    Vec3i min(bbox.min());
    Vec3i max = Vec3i(bbox.max()) + Vec3i(1, 1, 1);

    for (ijk[0] = min.x(); ijk[0] < max.x(); ++ijk[0]) {
        for (ijk[1] = min.y(); ijk[1] < max.y(); ++ijk[1]) {
            for (ijk[2] = min.z(); ijk[2] < max.z(); ++ijk[2]) {
                const math::Vec3d wsLocation = transform->indexToWorld(ijk);
                const float dis = float((wsLocation - wsCenter).length());

                acc.setValue(ijk, dis);
            }
        }
    }


    {
    // test at location 10, 10, 10 in index space
    math::Coord xyz(10, 10, 10);

    math::Vec3s result =
          math::Gradient<math::NonlinearFrustumMap, math::CD_2ND>::result(*map, acc, xyz);

    // The Gradient should be unit lenght for this case
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    math::Vec3d wsVec = transform->indexToWorld(xyz);
    math::Vec3d direction = (wsVec - wsCenter);
    direction.normalize();

    // test the actual direction of the gradient
    CPPUNIT_ASSERT(direction.eq(result, 0.01 /*tolerance*/));
    }

    {
    // test at location 30, 30, 60 in index space
    math::Coord xyz(30, 30, 60);

    math::Vec3s result =
          math::Gradient<math::NonlinearFrustumMap, math::CD_2ND>::result(*map, acc, xyz);

    // The Gradient should be unit lenght for this case
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

    math::Vec3d wsVec = transform->indexToWorld(xyz);
    math::Vec3d direction = (wsVec - wsCenter);
    direction.normalize();

    // test the actual direction of the gradient
    CPPUNIT_ASSERT(direction.eq(result, 0.01 /*tolerance*/));
    }
}



void
TestGradient::testWSGradientStencil()
{
    using namespace openvdb;

    double voxel_size = 0.5;
    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
    grid->setTransform(math::Transform::createLinearTransform(voxel_size));
    CPPUNIT_ASSERT(grid->empty());

    const openvdb::Coord dim(32,32,32);
    const openvdb::Vec3f center(6.0f, 8.0f ,10.0f);//i.e. (12,16,20) in index space
    const float radius = 10;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!grid->empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(grid->activeVoxelCount()));
    const Coord xyz(11, 17, 26);

    // try with a map
    math::SevenPointStencil<FloatGrid> stencil(*grid);
    stencil.moveTo(xyz);

    math::SecondOrderDenseStencil<FloatGrid> dense_2ndOrder(*grid);
    dense_2ndOrder.moveTo(xyz);

    math::FourthOrderDenseStencil<FloatGrid> dense_4thOrder(*grid);
    dense_4thOrder.moveTo(xyz);

    Vec3f result;
    math::MapBase::Ptr rotated_map;
    {
        math::UniformScaleMap map(voxel_size);
        result = math::Gradient<math::UniformScaleMap, math::CD_2ND>::result(
            map, stencil);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
        rotated_map = map.preRotate(1.5, math::X_AXIS);
        // verify the new map is an affine map
        CPPUNIT_ASSERT(rotated_map->type() == math::AffineMap::mapType());
        math::AffineMap::Ptr affine_map =
            StaticPtrCast<math::AffineMap, math::MapBase>(rotated_map);
        // the gradient should have the same length even after rotation

        result = math::Gradient<math::AffineMap, math::CD_2ND>::result(
            *affine_map, dense_2ndOrder);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);

        result = math::Gradient<math::AffineMap, math::CD_4TH>::result(
            *affine_map, dense_4thOrder);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
    {
        math::UniformScaleTranslateMap map(voxel_size, Vec3d(0,0,0));

        result = math::Gradient<math::UniformScaleTranslateMap, math::CD_2ND>::result(map, stencil);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
    {
        math::ScaleTranslateMap map(Vec3d(voxel_size, voxel_size, voxel_size), Vec3d(0,0,0));
        result = math::Gradient<math::ScaleTranslateMap, math::CD_2ND>::result(map, stencil);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
    {
        math::TranslationMap map;
        result = math::Gradient<math::TranslationMap, math::CD_2ND>::result(map, stencil);
        // value = 1 because the translation map assumes uniform spacing
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, result.length(), /*tolerance=*/0.01);
    }
    {
        // test the GenericMap Grid interface
        math::GenericMap generic_map(*grid);
        result = math::Gradient<math::GenericMap, math::CD_2ND>::result(
            generic_map, dense_2ndOrder);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
    {
        // test the GenericMap Transform interface
        math::GenericMap generic_map(grid->transform());
        result = math::Gradient<math::GenericMap, math::CD_2ND>::result(
            generic_map, dense_2ndOrder);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
    {
        // test the GenericMap Map interface
        math::GenericMap generic_map(rotated_map);
        result = math::Gradient<math::GenericMap, math::CD_2ND>::result(
            generic_map, dense_2ndOrder);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
    {
        // test a map with non-uniform SCALING AND ROTATION
        Vec3d voxel_sizes(0.25, 0.45, 0.75);
        math::MapBase::Ptr base_map( new math::ScaleMap(voxel_sizes));
        // apply rotation
        rotated_map = base_map->preRotate(1.5, math::X_AXIS);
        grid->setTransform(math::Transform::Ptr(new math::Transform(rotated_map)));
        // remake the sphere
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);
        math::AffineMap::Ptr affine_map =
            StaticPtrCast<math::AffineMap, math::MapBase>(rotated_map);

        stencil.moveTo(xyz);
        result = math::Gradient<math::AffineMap, math::CD_2ND>::result(*affine_map, stencil);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
    {
        // test a map with NON-UNIFORM SCALING
        Vec3d voxel_sizes(0.5, 1.0, 0.75);
        math::MapBase::Ptr base_map( new math::ScaleMap(voxel_sizes));
        grid->setTransform(math::Transform::Ptr(new math::Transform(base_map)));
        // remake the sphere
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        math::ScaleMap map(voxel_sizes);
        dense_2ndOrder.moveTo(xyz);

        result = math::Gradient<math::ScaleMap, math::CD_2ND>::result(map, dense_2ndOrder);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.length(), /*tolerance=*/0.01);
    }
}


void
TestGradient::testWSGradientNormSqr()
{
    using namespace openvdb;

    using AccessorType = FloatGrid::ConstAccessor;
    double voxel_size = 0.5;
    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
    grid->setTransform(math::Transform::createLinearTransform(voxel_size));
    CPPUNIT_ASSERT(grid->empty());

    const openvdb::Coord dim(32,32,32);
    const openvdb::Vec3f center(6.0f,8.0f,10.0f);//i.e. (12,16,20) in index space
    const float radius = 10.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!grid->empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(grid->activeVoxelCount()));
    const Coord xyz(11, 17, 26);

    AccessorType inAccessor = grid->getConstAccessor();

    // test gradient in index and world space using the 7-pt stencil
    math::UniformScaleMap uniform_scale(voxel_size);
    FloatTree::ValueType normsqrd;
    normsqrd = math::GradientNormSqrd<math::UniformScaleMap, math::FIRST_BIAS>::result(
        uniform_scale, inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, normsqrd, /*tolerance=*/0.07);

    // test world space using the 13pt stencil
    normsqrd = math::GradientNormSqrd<math::UniformScaleMap, math::SECOND_BIAS>::result(
        uniform_scale, inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, normsqrd, /*tolerance=*/0.05);

    math::AffineMap affine(voxel_size*math::Mat3d::identity());
    normsqrd = math::GradientNormSqrd<math::AffineMap, math::FIRST_BIAS>::result(
        affine, inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, normsqrd, /*tolerance=*/0.07);

    normsqrd = math::GradientNormSqrd<math::UniformScaleMap, math::THIRD_BIAS>::result(
        uniform_scale, inAccessor, xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, normsqrd, /*tolerance=*/0.05);
}


void
TestGradient::testWSGradientNormSqrStencil()
{
    using namespace openvdb;

    double voxel_size = 0.5;
    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
    grid->setTransform(math::Transform::createLinearTransform(voxel_size));
    CPPUNIT_ASSERT(grid->empty());

    const openvdb::Coord dim(32,32,32);
    const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
    const float radius = 10.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!grid->empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(grid->activeVoxelCount()));
    const Coord xyz(11, 17, 26);

    math::SevenPointStencil<FloatGrid> sevenpt(*grid);
    sevenpt.moveTo(xyz);

    math::ThirteenPointStencil<FloatGrid> thirteenpt(*grid);
    thirteenpt.moveTo(xyz);

    math::SecondOrderDenseStencil<FloatGrid> dense_2ndOrder(*grid);
    dense_2ndOrder.moveTo(xyz);

    math::NineteenPointStencil<FloatGrid> nineteenpt(*grid);
    nineteenpt.moveTo(xyz);

    // test gradient in index and world space using the 7-pt stencil
    math::UniformScaleMap uniform_scale(voxel_size);
    FloatTree::ValueType normsqrd;
    normsqrd = math::GradientNormSqrd<math::UniformScaleMap, math::FIRST_BIAS>::result(
        uniform_scale, sevenpt);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, normsqrd, /*tolerance=*/0.07);


    // test gradient in index and world space using the 13pt stencil
    normsqrd = math::GradientNormSqrd<math::UniformScaleMap, math::SECOND_BIAS>::result(
        uniform_scale, thirteenpt);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, normsqrd, /*tolerance=*/0.05);

    math::AffineMap affine(voxel_size*math::Mat3d::identity());
    normsqrd = math::GradientNormSqrd<math::AffineMap, math::FIRST_BIAS>::result(
        affine, dense_2ndOrder);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, normsqrd, /*tolerance=*/0.07);

    normsqrd = math::GradientNormSqrd<math::UniformScaleMap, math::THIRD_BIAS>::result(
        uniform_scale, nineteenpt);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, normsqrd, /*tolerance=*/0.05);
}


void
TestGradient::testGradientTool()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
    FloatTree& tree = grid->tree();

    const openvdb::Coord dim(64, 64, 64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);
    const float radius = 10.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!tree.empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));
    const Coord xyz(10, 20, 30);

    Vec3SGrid::Ptr grad = tools::gradient(*grid);
    CPPUNIT_ASSERT_EQUAL(int(tree.activeVoxelCount()), int(grad->activeVoxelCount()));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, grad->getConstAccessor().getValue(xyz).length(),
        /*tolerance=*/0.01);
}


void
TestGradient::testGradientMaskedTool()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
    FloatTree& tree = grid->tree();

    const openvdb::Coord dim(64, 64, 64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);
    const float radius = 10.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!tree.empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));

    const openvdb::CoordBBox maskbbox(openvdb::Coord(35, 30, 30), openvdb::Coord(41, 41, 41));
    BoolGrid::Ptr maskGrid = BoolGrid::create(false);
    maskGrid->fill(maskbbox, true/*value*/, true/*activate*/);

    Vec3SGrid::Ptr grad = tools::gradient(*grid, *maskGrid);
    {// outside the masked region
        const Coord xyz(10, 20, 30);
        CPPUNIT_ASSERT(!maskbbox.isInside(xyz));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, grad->getConstAccessor().getValue(xyz).length(),
                                     /*tolerance=*/0.01);
    }
    {// inside the masked region
        const Coord xyz(38, 35, 33);
        CPPUNIT_ASSERT(maskbbox.isInside(xyz));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, grad->getConstAccessor().getValue(xyz).length(),
                                     /*tolerance=*/0.01);
    }
}


void
TestGradient::testIntersectsIsoValue()
{
    using namespace openvdb;

    {// test zero crossing in -x
        FloatGrid  grid(/*backgroundValue=*/5.0);
        FloatTree& tree = grid.tree();
        Coord xyz(2,-5,60);
        tree.setValue(xyz, 1.3f);
        tree.setValue(xyz.offsetBy(-1,0,0), -2.0f);
        math::SevenPointStencil<FloatGrid>  stencil(grid);
        stencil.moveTo(xyz);
        CPPUNIT_ASSERT( stencil.intersects(     ));
        CPPUNIT_ASSERT( stencil.intersects( 0.0f));
        CPPUNIT_ASSERT( stencil.intersects( 2.0f));
        CPPUNIT_ASSERT(!stencil.intersects( 5.5f));
        CPPUNIT_ASSERT(!stencil.intersects(-2.5f));
    }
    {// test zero crossing in +x
        FloatGrid  grid(/*backgroundValue=*/5.0);
        FloatTree& tree = grid.tree();
        Coord xyz(2,-5,60);
        tree.setValue(xyz, 1.3f);
        tree.setValue(xyz.offsetBy(1,0,0), -2.0f);
        math::SevenPointStencil<FloatGrid>  stencil(grid);
        stencil.moveTo(xyz);
        CPPUNIT_ASSERT(stencil.intersects());
    }
    {// test zero crossing in -y
        FloatGrid  grid(/*backgroundValue=*/5.0);
        FloatTree& tree = grid.tree();
        Coord xyz(2,-5,60);
        tree.setValue(xyz, 1.3f);
        tree.setValue(xyz.offsetBy(0,-1,0), -2.0f);
        math::SevenPointStencil<FloatGrid>  stencil(grid);
        stencil.moveTo(xyz);
        CPPUNIT_ASSERT(stencil.intersects());
    }
    {// test zero crossing in y
        FloatGrid  grid(/*backgroundValue=*/5.0);
        FloatTree& tree = grid.tree();
        Coord xyz(2,-5,60);
        tree.setValue(xyz, 1.3f);
        tree.setValue(xyz.offsetBy(0,1,0), -2.0f);
        math::SevenPointStencil<FloatGrid>  stencil(grid);
        stencil.moveTo(xyz);
        CPPUNIT_ASSERT(stencil.intersects());
    }
    {// test zero crossing in -z
        FloatGrid  grid(/*backgroundValue=*/5.0);
        FloatTree& tree = grid.tree();
        Coord xyz(2,-5,60);
        tree.setValue(xyz, 1.3f);
        tree.setValue(xyz.offsetBy(0,0,-1), -2.0f);
        math::SevenPointStencil<FloatGrid>  stencil(grid);
        stencil.moveTo(xyz);
        CPPUNIT_ASSERT(stencil.intersects());
    }
    {// test zero crossing in z
        FloatGrid  grid(/*backgroundValue=*/5.0);
        FloatTree& tree = grid.tree();
        Coord xyz(2,-5,60);
        tree.setValue(xyz, 1.3f);
        tree.setValue(xyz.offsetBy(0,0,1), -2.0f);
        math::SevenPointStencil<FloatGrid>  stencil(grid);
        stencil.moveTo(xyz);
        CPPUNIT_ASSERT(stencil.intersects());
    }
    {// test zero crossing in -x & z
        FloatGrid  grid(/*backgroundValue=*/5.0);
        FloatTree& tree = grid.tree();
        Coord xyz(2,-5,60);
        tree.setValue(xyz, 1.3f);
        tree.setValue(xyz.offsetBy(-1,0,1), -2.0f);
        math::SevenPointStencil<FloatGrid>  stencil(grid);
        stencil.moveTo(xyz);
        CPPUNIT_ASSERT(!stencil.intersects());
    }
    {// test zero multiple crossings
        FloatGrid  grid(/*backgroundValue=*/5.0);
        FloatTree& tree = grid.tree();
        Coord xyz(2,-5,60);
        tree.setValue(xyz, 1.3f);
        tree.setValue(xyz.offsetBy(-1, 0, 1), -1.0f);
        tree.setValue(xyz.offsetBy( 0, 0, 1), -2.0f);
        tree.setValue(xyz.offsetBy( 0, 1, 0), -3.0f);
        tree.setValue(xyz.offsetBy( 0, 0,-1), -2.0f);
        math::SevenPointStencil<FloatGrid>  stencil(grid);
        stencil.moveTo(xyz);
        CPPUNIT_ASSERT(stencil.intersects());
    }
}


void
TestGradient::testOldStyleStencils()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = FloatGrid::create(/*backgroundValue=*/5.0);
    grid->setTransform(math::Transform::createLinearTransform(/*voxel size=*/0.5));
    CPPUNIT_ASSERT(grid->empty());

    const openvdb::Coord dim(32,32,32);
    const openvdb::Vec3f center(6.0f,8.0f,10.0f);//i.e. (12,16,20) in index space
    const float radius=10.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!grid->empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(grid->activeVoxelCount()));
    const Coord xyz(11, 17, 26);

    math::GradStencil<FloatGrid> gs(*grid);
    gs.moveTo(xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, gs.gradient().length(), /*tolerance=*/0.01);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, gs.normSqGrad(),        /*tolerance=*/0.10);

    math::WenoStencil<FloatGrid> ws(*grid);
    ws.moveTo(xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ws.gradient().length(), /*tolerance=*/0.01);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ws.normSqGrad(),        /*tolerance=*/0.01);

    math::CurvatureStencil<FloatGrid> cs(*grid);
    cs.moveTo(xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, cs.gradient().length(), /*tolerance=*/0.01);
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
