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
#include <openvdb/math/Stencils.h>
#include <openvdb/math/Operators.h>
#include <openvdb/tools/GridOperators.h>
#include "util.h" // for unittest_util::makeSphere()
#include <cppunit/extensions/HelperMacros.h>


class TestMeanCurvature: public CppUnit::TestFixture
{
public:
    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestMeanCurvature);
    CPPUNIT_TEST(testISMeanCurvature);                    // MeanCurvature in Index Space
    CPPUNIT_TEST(testISMeanCurvatureStencil);
    CPPUNIT_TEST(testWSMeanCurvature);                    // MeanCurvature in World Space
    CPPUNIT_TEST(testWSMeanCurvatureStencil);
    CPPUNIT_TEST(testMeanCurvatureTool);                  // MeanCurvature tool
    CPPUNIT_TEST(testMeanCurvatureMaskedTool);                  // MeanCurvature tool
    CPPUNIT_TEST(testOldStyleStencils);                   // old stencil impl - deprecate

    CPPUNIT_TEST_SUITE_END();

    void testISMeanCurvature();
    void testISMeanCurvatureStencil();
    void testWSMeanCurvature();
    void testWSMeanCurvatureStencil();
    void testMeanCurvatureTool();
    void testMeanCurvatureMaskedTool();
    void testOldStyleStencils();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMeanCurvature);


void
TestMeanCurvature::testISMeanCurvature()
{
    using namespace openvdb;

    typedef FloatGrid::ConstAccessor  AccessorType;

    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
    FloatTree& tree = grid->tree();
    AccessorType inAccessor = grid->getConstAccessor();
    AccessorType::ValueType alpha, beta, meancurv, normGrad;
    Coord xyz(35,30,30);

    // First test an empty grid
    CPPUNIT_ASSERT(tree.empty());
    typedef math::ISMeanCurvature<math::CD_SECOND, math::CD_2ND> SecondOrder;
    CPPUNIT_ASSERT(!SecondOrder::result(inAccessor, xyz, alpha, beta));

    typedef math::ISMeanCurvature<math::CD_FOURTH, math::CD_4TH> FourthOrder;
    CPPUNIT_ASSERT(!FourthOrder::result(inAccessor, xyz, alpha, beta));

    typedef math::ISMeanCurvature<math::CD_SIXTH, math::CD_6TH> SixthOrder;
    CPPUNIT_ASSERT(!SixthOrder::result(inAccessor, xyz, alpha, beta));

    // Next test a level set sphere
    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f ,30.0f, 40.0f);
    const float radius=0.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!tree.empty());

    SecondOrder::result(inAccessor, xyz, alpha, beta);

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, meancurv, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, normGrad, 0.001);

    FourthOrder::result(inAccessor, xyz, alpha, beta);

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, meancurv, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, normGrad, 0.001);

    SixthOrder::result(inAccessor, xyz, alpha, beta);

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, meancurv, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, normGrad, 0.001);

    xyz.reset(35,10,40);

    SecondOrder::result(inAccessor, xyz, alpha, beta);

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/20.0, meancurv, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/20.0, normGrad, 0.001);
}


void
TestMeanCurvature::testISMeanCurvatureStencil()
{
    using namespace openvdb;

    typedef FloatGrid::ConstAccessor  AccessorType;

    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
    FloatTree& tree = grid->tree();
    math::SecondOrderDenseStencil<FloatGrid> dense_2nd(*grid);
    math::FourthOrderDenseStencil<FloatGrid> dense_4th(*grid);
    math::SixthOrderDenseStencil<FloatGrid> dense_6th(*grid);
    AccessorType::ValueType alpha, beta;
    Coord xyz(35,30,30);
    dense_2nd.moveTo(xyz);
    dense_4th.moveTo(xyz);
    dense_6th.moveTo(xyz);

    // First test on an empty grid
    CPPUNIT_ASSERT(tree.empty());

    typedef math::ISMeanCurvature<math::CD_SECOND, math::CD_2ND> SecondOrder;
    CPPUNIT_ASSERT(!SecondOrder::result(dense_2nd, alpha, beta));

    typedef math::ISMeanCurvature<math::CD_FOURTH, math::CD_4TH> FourthOrder;
    CPPUNIT_ASSERT(!FourthOrder::result(dense_4th, alpha, beta));

    typedef math::ISMeanCurvature<math::CD_SIXTH, math::CD_6TH> SixthOrder;
    CPPUNIT_ASSERT(!SixthOrder::result(dense_6th, alpha, beta));

    // Next test on a level set sphere
    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f ,30.0f, 40.0f);
    const float radius=0.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);
    dense_2nd.moveTo(xyz);
    dense_4th.moveTo(xyz);
    dense_6th.moveTo(xyz);

    CPPUNIT_ASSERT(!tree.empty());

    CPPUNIT_ASSERT(SecondOrder::result(dense_2nd, alpha, beta));

    AccessorType::ValueType meancurv = alpha/(2*math::Pow3(beta) );
    AccessorType::ValueType normGrad = alpha/(2*math::Pow2(beta) );

    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, meancurv, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, normGrad, 0.001);

    CPPUNIT_ASSERT(FourthOrder::result(dense_4th, alpha, beta));

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, meancurv, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, normGrad, 0.001);

    CPPUNIT_ASSERT(SixthOrder::result(dense_6th, alpha, beta));

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, meancurv, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, normGrad, 0.001);

    xyz.reset(35,10,40);
    dense_2nd.moveTo(xyz);
    CPPUNIT_ASSERT(SecondOrder::result(dense_2nd, alpha, beta));

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/20.0, meancurv, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/20.0, normGrad, 0.001);
}


void
TestMeanCurvature::testWSMeanCurvature()
{
    using namespace openvdb;
    using math::AffineMap;
    using math::TranslationMap;
    using math::UniformScaleMap;

    typedef FloatGrid::ConstAccessor  AccessorType;

    {// Empty grid test
        FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
        FloatTree& tree = grid->tree();
        AccessorType inAccessor = grid->getConstAccessor();
        Coord xyz(35,30,30);
        CPPUNIT_ASSERT(tree.empty());

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        AffineMap affine;
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            affine, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meancurv, 0.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, normGrad, 0.0);

        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            affine, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meancurv, 0.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, normGrad, 0.0);

        UniformScaleMap uniform;
        meancurv = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::result(
            uniform, inAccessor, xyz);
        normGrad = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            uniform, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meancurv, 0.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, normGrad, 0.0);

        xyz.reset(35,10,40);

        TranslationMap trans;
        meancurv = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::result(
            trans, inAccessor, xyz);
        normGrad = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            trans, inAccessor, xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meancurv, 0.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, normGrad, 0.0);
    }

    { // unit size voxel test
        FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
        FloatTree& tree = grid->tree();

        const openvdb::Coord dim(64,64,64);
        const openvdb::Vec3f center(35.0f ,30.0f, 40.0f);
        const float radius=0.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        CPPUNIT_ASSERT(!tree.empty());
        Coord xyz(35,30,30);

        AccessorType inAccessor = grid->getConstAccessor();

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        AffineMap affine;
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            affine, inAccessor, xyz);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, normGrad, 0.001);
        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            affine, inAccessor, xyz);


        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, normGrad, 0.001);

        UniformScaleMap uniform;
        meancurv = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::result(
            uniform, inAccessor, xyz);
        normGrad = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            uniform, inAccessor, xyz);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, normGrad, 0.001);

        xyz.reset(35,10,40);

        TranslationMap trans;
        meancurv = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::result(
            trans, inAccessor, xyz);
        normGrad = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            trans, inAccessor, xyz);


        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/20.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/20.0, normGrad, 0.001);
    }
    { // non-unit sized voxel

        double voxel_size = 0.5;
        FloatGrid::Ptr grid = FloatGrid::create(/*backgroundValue=*/5.0);
        grid->setTransform(math::Transform::createLinearTransform(voxel_size));
        CPPUNIT_ASSERT(grid->empty());

        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        AccessorType inAccessor = grid->getConstAccessor();

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        Coord xyz(20,16,20);
        AffineMap affine(voxel_size*math::Mat3d::identity());
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            affine, inAccessor, xyz);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/4.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/4.0, normGrad, 0.001);
        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            affine, inAccessor, xyz);


        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/4.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/4.0, normGrad, 0.001);

        UniformScaleMap uniform(voxel_size);
        meancurv = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::result(
            uniform, inAccessor, xyz);
        normGrad = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            uniform, inAccessor, xyz);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/4.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/4.0, normGrad, 0.001);

    }
    { // NON-UNIFORM SCALING AND ROTATION

        Vec3d voxel_sizes(0.25, 0.45, 0.75);
        FloatGrid::Ptr grid = FloatGrid::create();
        math::MapBase::Ptr base_map( new math::ScaleMap(voxel_sizes));
        // apply rotation
        math::MapBase::Ptr rotated_map = base_map->preRotate(1.5, math::X_AXIS);
        grid->setTransform(math::Transform::Ptr(new math::Transform(rotated_map)));
        CPPUNIT_ASSERT(grid->empty());

        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        AccessorType inAccessor = grid->getConstAccessor();

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        Coord xyz(20,16,20);
        Vec3d location = grid->indexToWorld(xyz);
        double dist = (center - location).length();
        AffineMap::ConstPtr affine = grid->transform().map<AffineMap>();
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            *affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            *affine, inAccessor, xyz);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/dist, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/dist, normGrad, 0.001);
        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            *affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            *affine, inAccessor, xyz);


        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/dist, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/dist, normGrad, 0.001);
    }
}


void
TestMeanCurvature::testWSMeanCurvatureStencil()
{
    using namespace openvdb;
    using math::AffineMap;
    using math::TranslationMap;
    using math::UniformScaleMap;

    typedef FloatGrid::ConstAccessor AccessorType;

    {// empty grid test
        FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
        FloatTree& tree = grid->tree();
        CPPUNIT_ASSERT(tree.empty());
        Coord xyz(35,30,30);

        math::SecondOrderDenseStencil<FloatGrid> dense_2nd(*grid);
        math::FourthOrderDenseStencil<FloatGrid> dense_4th(*grid);
        math::SixthOrderDenseStencil<FloatGrid> dense_6th(*grid);
        dense_2nd.moveTo(xyz);
        dense_4th.moveTo(xyz);
        dense_6th.moveTo(xyz);

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        AffineMap affine;
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            affine, dense_2nd);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            affine, dense_2nd);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meancurv, 0.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, normGrad, 0.00);

        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            affine, dense_4th);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            affine, dense_4th);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meancurv, 0.00);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, normGrad, 0.00);

        UniformScaleMap uniform;
        meancurv = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::result(
            uniform, dense_6th);
        normGrad = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            uniform, dense_6th);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meancurv, 0.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, normGrad, 0.0);

        xyz.reset(35,10,40);
        dense_6th.moveTo(xyz);

        TranslationMap trans;
        meancurv = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::result(
            trans, dense_6th);
        normGrad = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            trans, dense_6th);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, meancurv, 0.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, normGrad, 0.0);
    }

    { // unit-sized voxels

        FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
        FloatTree& tree = grid->tree();

        const openvdb::Coord dim(64,64,64);
        const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);//i.e. (35,30,40) in index space
        const float radius=0.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        CPPUNIT_ASSERT(!tree.empty());
        Coord xyz(35,30,30);
        math::SecondOrderDenseStencil<FloatGrid> dense_2nd(*grid);
        math::FourthOrderDenseStencil<FloatGrid> dense_4th(*grid);
        math::SixthOrderDenseStencil<FloatGrid> dense_6th(*grid);
        dense_2nd.moveTo(xyz);
        dense_4th.moveTo(xyz);
        dense_6th.moveTo(xyz);

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        AffineMap affine;
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            affine, dense_2nd);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            affine, dense_2nd);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, normGrad, 0.001);
        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            affine, dense_4th);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            affine, dense_4th);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, normGrad, 0.001);

        UniformScaleMap uniform;
        meancurv = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::result(
            uniform, dense_6th);
        normGrad = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            uniform, dense_6th);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, normGrad, 0.001);

        xyz.reset(35,10,40);
        dense_6th.moveTo(xyz);

        TranslationMap trans;
        meancurv = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::result(
            trans, dense_6th);
        normGrad = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            trans, dense_6th);


        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/20.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/20.0, normGrad, 0.001);
    }
    { // non-unit sized voxel

        double voxel_size = 0.5;
        FloatGrid::Ptr grid = FloatGrid::create(/*backgroundValue=*/5.0);
        grid->setTransform(math::Transform::createLinearTransform(voxel_size));
        CPPUNIT_ASSERT(grid->empty());

        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);


        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        Coord xyz(20,16,20);
        math::SecondOrderDenseStencil<FloatGrid> dense_2nd(*grid);
        math::FourthOrderDenseStencil<FloatGrid> dense_4th(*grid);
        math::SixthOrderDenseStencil<FloatGrid> dense_6th(*grid);
        dense_2nd.moveTo(xyz);
        dense_4th.moveTo(xyz);
        dense_6th.moveTo(xyz);

        AffineMap affine(voxel_size*math::Mat3d::identity());
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            affine, dense_2nd);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            affine, dense_2nd);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/4.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/4.0, normGrad, 0.001);
        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            affine, dense_4th);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            affine, dense_4th);


        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/4.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/4.0, normGrad, 0.001);

        UniformScaleMap uniform(voxel_size);
        meancurv = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::result(
            uniform, dense_6th);
        normGrad = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            uniform, dense_6th);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/4.0, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/4.0, normGrad, 0.001);
    }
    { // NON-UNIFORM SCALING AND ROTATION

        Vec3d voxel_sizes(0.25, 0.45, 0.75);
        FloatGrid::Ptr grid = FloatGrid::create();
        math::MapBase::Ptr base_map( new math::ScaleMap(voxel_sizes));
        // apply rotation
        math::MapBase::Ptr rotated_map = base_map->preRotate(1.5, math::X_AXIS);
        grid->setTransform(math::Transform::Ptr(new math::Transform(rotated_map)));
        CPPUNIT_ASSERT(grid->empty());

        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        Coord xyz(20,16,20);
        math::SecondOrderDenseStencil<FloatGrid> dense_2nd(*grid);
        math::FourthOrderDenseStencil<FloatGrid> dense_4th(*grid);
        dense_2nd.moveTo(xyz);
        dense_4th.moveTo(xyz);


        Vec3d location = grid->indexToWorld(xyz);
        double dist = (center - location).length();
        AffineMap::ConstPtr affine = grid->transform().map<AffineMap>();
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            *affine, dense_2nd);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            *affine, dense_2nd);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/dist, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/dist, normGrad, 0.001);
        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            *affine, dense_4th);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            *affine, dense_4th);


        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/dist, meancurv, 0.001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/dist, normGrad, 0.001);
    }
}


void
TestMeanCurvature::testMeanCurvatureTool()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
    FloatTree& tree = grid->tree();

    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);//i.e. (35,30,40) in index space
    const float radius=0.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!tree.empty());
    FloatGrid::Ptr curv = tools::meanCurvature(*grid);
    FloatGrid::ConstAccessor accessor = curv->getConstAccessor();

    Coord xyz(35,30,30);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, accessor.getValue(xyz), 0.001);

    xyz.reset(35,10,40);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/20.0, accessor.getValue(xyz), 0.001);
}


void
TestMeanCurvature::testMeanCurvatureMaskedTool()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
    FloatTree& tree = grid->tree();

    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);//i.e. (35,30,40) in index space
    const float radius=0.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    CPPUNIT_ASSERT(!tree.empty());


    const openvdb::CoordBBox maskbbox(openvdb::Coord(35, 30, 30), openvdb::Coord(41, 41, 41));
    BoolGrid::Ptr maskGrid = BoolGrid::create(false);
    maskGrid->fill(maskbbox, true/*value*/, true/*activate*/);


    FloatGrid::Ptr curv = tools::meanCurvature(*grid, *maskGrid);
    FloatGrid::ConstAccessor accessor = curv->getConstAccessor();

    // test inside
    Coord xyz(35,30,30);
    CPPUNIT_ASSERT(maskbbox.isInside(xyz));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/10.0, accessor.getValue(xyz), 0.001);

    // test outside
    xyz.reset(35,10,40);
    CPPUNIT_ASSERT(!maskbbox.isInside(xyz));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, accessor.getValue(xyz), 0.001);
}


void
TestMeanCurvature::testOldStyleStencils()
{
    using namespace openvdb;

    {// test of level set to sphere at (6,8,10) with R=10 and dx=0.5

        FloatGrid::Ptr grid = FloatGrid::create(/*backgroundValue=*/5.0);
        grid->setTransform(math::Transform::createLinearTransform(/*voxel size=*/0.5));
        CPPUNIT_ASSERT(grid->empty());
        math::CurvatureStencil<FloatGrid> cs(*grid);
        Coord xyz(20,16,20);//i.e. 8 voxel or 4 world units away from the center
        cs.moveTo(xyz);

        // First test on an empty grid
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, cs.meanCurvature(), 0.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, cs.meanCurvatureNormGrad(), 0.0);

        // Next test on a level set sphere
        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        CPPUNIT_ASSERT(!grid->empty());
        CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(grid->activeVoxelCount()));
        cs.moveTo(xyz);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/4.0, cs.meanCurvature(), 0.01);// 1/distance from center
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            1.0/4.0, cs.meanCurvatureNormGrad(), 0.01);// 1/distance from center

        xyz.reset(12,16,10);//i.e. 10 voxel or 5 world units away from the center
        cs.moveTo(xyz);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0/5.0, cs.meanCurvature(), 0.01);// 1/distance from center
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            1.0/5.0, cs.meanCurvatureNormGrad(), 0.01);// 1/distance from center
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
