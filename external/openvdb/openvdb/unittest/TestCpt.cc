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

#include <sstream>
#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/math/Stencils.h> // for old GradientStencil
#include "util.h" // for unittest_util::makeSphere()

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/0.0);

class TestCpt: public CppUnit::TestFixture
{
public:
    virtual void setUp() { openvdb::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestCpt);
    CPPUNIT_TEST(testCpt);                      // Cpt in World Space
    CPPUNIT_TEST(testCptStencil);
    CPPUNIT_TEST(testCptTool);                  // Cpt tool
    CPPUNIT_TEST(testCptMaskedTool);
    CPPUNIT_TEST(testOldStyleStencils);         // old stencil impl
    CPPUNIT_TEST_SUITE_END();

    void testCpt();
    void testCptStencil();
    void testCptTool();
    void testCptMaskedTool();
    void testOldStyleStencils();
};


CPPUNIT_TEST_SUITE_REGISTRATION(TestCpt);

void
TestCpt::testCpt()
{
    using namespace openvdb;

    typedef FloatGrid::ConstAccessor AccessorType;

    { // unit voxel size tests

        FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
        const FloatTree& tree = grid->tree();
        CPPUNIT_ASSERT(tree.empty());

        const Coord dim(64,64,64);
        const Vec3f center(35.0, 30.0f, 40.0f);
        const float radius=0;//point at {35,30,40}
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);
        CPPUNIT_ASSERT(!tree.empty());
        CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));


        AccessorType inAccessor = grid->getConstAccessor();
        // this uses the gradient.  Only test for a few maps, since the gradient is
        // tested elsewhere

        Coord xyz(35,30,30);

        math::TranslationMap translate;
        // Note the CPT::result is in continuous index space
        Vec3f P = math::CPT<math::TranslationMap, math::CD_2ND>::result(translate, inAccessor, xyz);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);

        // CPT_RANGE::result is in the range of the map
        // CPT_RANGE::result = map.applyMap(CPT::result())
        // for our tests, the map is an identity so in this special case
        // the two versions of the Cpt should exactly agree
        P = math::CPT_RANGE<math::TranslationMap, math::CD_2ND>::result(translate, inAccessor, xyz);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);

        xyz.reset(35,30,35);

        P = math::CPT<math::TranslationMap, math::CD_2ND>::result(translate, inAccessor, xyz);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);


        P = math::CPT_RANGE<math::TranslationMap, math::CD_2ND>::result(translate, inAccessor, xyz);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);
    }
    {
        // NON-UNIT VOXEL SIZE

        double voxel_size = 0.5;
        FloatGrid::Ptr grid = FloatGrid::create(/*backgroundValue=*/5.0);
        grid->setTransform(math::Transform::createLinearTransform(voxel_size));
        CPPUNIT_ASSERT(grid->empty());
        AccessorType inAccessor = grid->getConstAccessor();

        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10;//i.e. (16,8,10) and (6,8,0) are on the sphere
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        CPPUNIT_ASSERT(!grid->empty());
        CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(grid->activeVoxelCount()));

        Coord xyz(20,16,20);//i.e. (10,8,10) in world space or 6 world units inside the sphere
        math::AffineMap affine(voxel_size*math::Mat3d::identity());

        Vec3f P = math::CPT<math::AffineMap, math::CD_2ND>::result(affine, inAccessor, xyz);
        ASSERT_DOUBLES_EXACTLY_EQUAL(32,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(16,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(20,P[2]);


        P = math::CPT_RANGE<math::AffineMap, math::CD_2ND>::result(affine, inAccessor, xyz);
        ASSERT_DOUBLES_EXACTLY_EQUAL(16,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(8,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(10,P[2]);

        xyz.reset(12,16,10);

        P = math::CPT<math::AffineMap, math::CD_2ND>::result(affine, inAccessor, xyz);
        ASSERT_DOUBLES_EXACTLY_EQUAL(12,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(16,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(0,P[2]);


        P = math::CPT_RANGE<math::AffineMap, math::CD_2ND>::result(affine, inAccessor, xyz);
        ASSERT_DOUBLES_EXACTLY_EQUAL(6,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(8,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(0,P[2]);

    }
    {
        // NON-UNIFORM SCALING
        Vec3d voxel_sizes(0.5, 1, 0.5);
        math::MapBase::Ptr base_map( new math::ScaleMap(voxel_sizes));
        FloatGrid::Ptr grid = FloatGrid::create(/*backgroundValue=*/5.0);
        grid->setTransform(math::Transform::Ptr(new math::Transform(base_map)));

        CPPUNIT_ASSERT(grid->empty());
        AccessorType inAccessor = grid->getConstAccessor();


        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10;//i.e. (16,8,10) and (6,8,0) are on the sphere
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        CPPUNIT_ASSERT(!grid->empty());
        CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(grid->activeVoxelCount()));

        Coord ijk = grid->transform().worldToIndexNodeCentered(Vec3d(10,8,10));

        //Coord xyz(20,16,20);//i.e. (10,8,10) in world space or 6 world units inside the sphere
        math::ScaleMap scale(voxel_sizes);
        Vec3f P;
        P = math::CPT<math::ScaleMap, math::CD_2ND>::result(scale, inAccessor, ijk);
        ASSERT_DOUBLES_EXACTLY_EQUAL(32,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(8,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(20,P[2]);


        // world space result
        P = math::CPT_RANGE<math::ScaleMap, math::CD_2ND>::result(scale, inAccessor, ijk);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(16,P[0], 0.02 );
        CPPUNIT_ASSERT_DOUBLES_EQUAL(8, P[1], 0.02);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(10,P[2], 0.02);

        //xyz.reset(12,16,10);
        ijk = grid->transform().worldToIndexNodeCentered(Vec3d(6,8,5));

        P = math::CPT<math::ScaleMap, math::CD_2ND>::result(scale, inAccessor, ijk);
        ASSERT_DOUBLES_EXACTLY_EQUAL(12,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(8,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(0,P[2]);


        P = math::CPT_RANGE<math::ScaleMap, math::CD_2ND>::result(scale, inAccessor, ijk);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(6,P[0], 0.02);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(8,P[1], 0.02);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0,P[2], 0.02);

    }

}


void
TestCpt::testCptStencil()
{
    using namespace openvdb;

    { // UNIT VOXEL TEST

        FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
        const FloatTree& tree = grid->tree();
        CPPUNIT_ASSERT(tree.empty());

        const openvdb::Coord dim(64,64,64);
        const openvdb::Vec3f center(35.0f ,30.0f, 40.0f);
        const float radius=0.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);
        CPPUNIT_ASSERT(!tree.empty());
        CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));

        // this uses the gradient.  Only test for a few maps, since the gradient is
        // tested elsewhere

        math::SevenPointStencil<FloatGrid> sevenpt(*grid);
        math::SecondOrderDenseStencil<FloatGrid> dense_2nd(*grid);


        Coord xyz(35,30,30);
        CPPUNIT_ASSERT(tree.isValueOn(xyz));

        sevenpt.moveTo(xyz);
        dense_2nd.moveTo(xyz);

        math::TranslationMap translate;
        // Note the CPT::result is in continuous index space
        Vec3f P = math::CPT<math::TranslationMap, math::CD_2ND>::result(translate, sevenpt);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);

        // CPT_RANGE::result_stencil is in the range of the map
        // CPT_RANGE::result_stencil = map.applyMap(CPT::result_stencil())
        // for our tests, the map is an identity so in this special case
        // the two versions of the Cpt should exactly agree
        P = math::CPT_RANGE<math::TranslationMap, math::CD_2ND>::result(translate, sevenpt);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);

        xyz.reset(35,30,35);

        sevenpt.moveTo(xyz);
        dense_2nd.moveTo(xyz);

        CPPUNIT_ASSERT(tree.isValueOn(xyz));

        P = math::CPT<math::TranslationMap, math::CD_2ND>::result(translate, sevenpt);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);


        P = math::CPT_RANGE<math::TranslationMap, math::CD_2ND>::result(translate, sevenpt);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);


        xyz.reset(35,30,30);

        sevenpt.moveTo(xyz);
        dense_2nd.moveTo(xyz);

        math::AffineMap affine;

        P = math::CPT<math::AffineMap, math::CD_2ND>::result(affine, dense_2nd);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);


        P = math::CPT_RANGE<math::AffineMap, math::CD_2ND>::result(affine, dense_2nd);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);

        xyz.reset(35,30,35);

        sevenpt.moveTo(xyz);
        dense_2nd.moveTo(xyz);

        CPPUNIT_ASSERT(tree.isValueOn(xyz));

        P = math::CPT<math::AffineMap, math::CD_2ND>::result(affine, dense_2nd);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);

        CPPUNIT_ASSERT(tree.isValueOn(xyz));

        P = math::CPT_RANGE<math::AffineMap, math::CD_2ND>::result(affine, dense_2nd);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);

    }
     {
        // NON-UNIT VOXEL SIZE

        double voxel_size = 0.5;
        FloatGrid::Ptr grid = FloatGrid::create(/*backgroundValue=*/5.0);
        grid->setTransform(math::Transform::createLinearTransform(voxel_size));
        CPPUNIT_ASSERT(grid->empty());

        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10;//i.e. (16,8,10) and (6,8,0) are on the sphere
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        CPPUNIT_ASSERT(!grid->empty());
        CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(grid->activeVoxelCount()));


        math::SecondOrderDenseStencil<FloatGrid> dense_2nd(*grid);


        Coord xyz(20,16,20);//i.e. (10,8,10) in world space or 6 world units inside the sphere
        math::AffineMap affine(voxel_size*math::Mat3d::identity());
        dense_2nd.moveTo(xyz);

        Vec3f P = math::CPT<math::AffineMap, math::CD_2ND>::result(affine, dense_2nd);
        ASSERT_DOUBLES_EXACTLY_EQUAL(32,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(16,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(20,P[2]);


        P = math::CPT_RANGE<math::AffineMap, math::CD_2ND>::result(affine, dense_2nd);
        ASSERT_DOUBLES_EXACTLY_EQUAL(16,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(8,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(10,P[2]);

        xyz.reset(12,16,10);
        dense_2nd.moveTo(xyz);

        P = math::CPT<math::AffineMap, math::CD_2ND>::result(affine, dense_2nd);
        ASSERT_DOUBLES_EXACTLY_EQUAL(12,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(16,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(0,P[2]);


        P = math::CPT_RANGE<math::AffineMap, math::CD_2ND>::result(affine, dense_2nd);
        ASSERT_DOUBLES_EXACTLY_EQUAL(6,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(8,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(0,P[2]);

    }
    {
        // NON-UNIFORM SCALING
        Vec3d voxel_sizes(0.5, 1, 0.5);
        math::MapBase::Ptr base_map( new math::ScaleMap(voxel_sizes));
        FloatGrid::Ptr grid = FloatGrid::create(/*backgroundValue=*/5.0);
        grid->setTransform(math::Transform::Ptr(new math::Transform(base_map)));

        CPPUNIT_ASSERT(grid->empty());


        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10;//i.e. (16,8,10) and (6,8,0) are on the sphere
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        CPPUNIT_ASSERT(!grid->empty());
        CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(grid->activeVoxelCount()));

        Coord ijk = grid->transform().worldToIndexNodeCentered(Vec3d(10,8,10));
        math::SevenPointStencil<FloatGrid> sevenpt(*grid);

        sevenpt.moveTo(ijk);

        //Coord xyz(20,16,20);//i.e. (10,8,10) in world space or 6 world units inside the sphere
        math::ScaleMap scale(voxel_sizes);
        Vec3f P;
        P = math::CPT<math::ScaleMap, math::CD_2ND>::result(scale, sevenpt);
        ASSERT_DOUBLES_EXACTLY_EQUAL(32,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(8,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(20,P[2]);


        // world space result
        P = math::CPT_RANGE<math::ScaleMap, math::CD_2ND>::result(scale, sevenpt);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(16,P[0], 0.02 );
        CPPUNIT_ASSERT_DOUBLES_EQUAL(8, P[1], 0.02);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(10,P[2], 0.02);

        //xyz.reset(12,16,10);
        ijk = grid->transform().worldToIndexNodeCentered(Vec3d(6,8,5));
        sevenpt.moveTo(ijk);
        P = math::CPT<math::ScaleMap, math::CD_2ND>::result(scale, sevenpt);
        ASSERT_DOUBLES_EXACTLY_EQUAL(12,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(8,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(0,P[2]);


        P = math::CPT_RANGE<math::ScaleMap, math::CD_2ND>::result(scale, sevenpt);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(6,P[0], 0.02);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(8,P[1], 0.02);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0,P[2], 0.02);

    }


}

void
TestCpt::testCptTool()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
    const FloatTree& tree = grid->tree();
    CPPUNIT_ASSERT(tree.empty());

    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);
    const float radius=0;//point at {35,30,40}
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);
    CPPUNIT_ASSERT(!tree.empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));

    // run the tool
    typedef openvdb::tools::Cpt<FloatGrid> FloatCpt;
    FloatCpt cpt(*grid);
    FloatCpt::OutGridType::Ptr cptGrid =
        cpt.process(true/*threaded*/, false/*use world transform*/);

    FloatCpt::OutGridType::ConstAccessor cptAccessor = cptGrid->getConstAccessor();

    Coord xyz(35,30,30);
    CPPUNIT_ASSERT(tree.isValueOn(xyz));

    Vec3f P = cptAccessor.getValue(xyz);
    ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);

    xyz.reset(35,30,35);
    CPPUNIT_ASSERT(tree.isValueOn(xyz));

    P = cptAccessor.getValue(xyz);
    ASSERT_DOUBLES_EXACTLY_EQUAL(center[0],P[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(center[1],P[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(center[2],P[2]);
}

void
TestCpt::testCptMaskedTool()
{
    using namespace openvdb;

    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/5.0);
    const FloatTree& tree = grid->tree();
    CPPUNIT_ASSERT(tree.empty());

    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);
    const float radius=0;//point at {35,30,40}
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);
    CPPUNIT_ASSERT(!tree.empty());
    CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(tree.activeVoxelCount()));

    const openvdb::CoordBBox maskbbox(openvdb::Coord(35, 30, 30), openvdb::Coord(41, 41, 41));
    BoolGrid::Ptr maskGrid = BoolGrid::create(false);
    maskGrid->fill(maskbbox, true/*value*/, true/*activate*/);
    
    // run the tool
    //typedef openvdb::tools::Cpt<FloatGrid> FloatCpt;//fails because MaskT defaults to MaskGrid
    typedef openvdb::tools::Cpt<FloatGrid, BoolGrid> FloatCpt;
    FloatCpt cpt(*grid, *maskGrid);
    FloatCpt::OutGridType::Ptr cptGrid =
        cpt.process(true/*threaded*/, false/*use world transform*/);

    FloatCpt::OutGridType::ConstAccessor cptAccessor = cptGrid->getConstAccessor();

    // inside the masked region
    Coord xyz(35,30,30);
    CPPUNIT_ASSERT(tree.isValueOn(xyz));

    Vec3f P = cptAccessor.getValue(xyz);
    ASSERT_DOUBLES_EXACTLY_EQUAL(center[0], P[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(center[1], P[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(center[2], P[2]);

    // outside the masked region
    xyz.reset(42,42,42);
    CPPUNIT_ASSERT(!cptAccessor.isValueOn(xyz));
}

void
TestCpt::testOldStyleStencils()
{
    using namespace openvdb;

    {// test of level set to sphere at (6,8,10) with R=10 and dx=0.5

        FloatGrid::Ptr grid = FloatGrid::create(/*backgroundValue=*/5.0);
        grid->setTransform(math::Transform::createLinearTransform(/*voxel size=*/0.5));
        CPPUNIT_ASSERT(grid->empty());

        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f,8.0f,10.0f);//i.e. (12,16,20) in index space
        const float radius=10;//i.e. (16,8,10) and (6,8,0) are on the sphere
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        CPPUNIT_ASSERT(!grid->empty());
        CPPUNIT_ASSERT_EQUAL(dim[0]*dim[1]*dim[2], int(grid->activeVoxelCount()));
        math::GradStencil<FloatGrid> gs(*grid);

        Coord xyz(20,16,20);//i.e. (10,8,10) in world space or 6 world units inside the sphere
        gs.moveTo(xyz);
        float dist = gs.getValue();//signed closest distance to sphere in world coordinates
        Vec3f P = gs.cpt();//closes point to sphere in index space
        ASSERT_DOUBLES_EXACTLY_EQUAL(dist,-6);
        ASSERT_DOUBLES_EXACTLY_EQUAL(32,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(16,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(20,P[2]);

        xyz.reset(12,16,10);//i.e. (6,8,5) in world space or 15 world units inside the sphere
        gs.moveTo(xyz);
        dist = gs.getValue();//signed closest distance to sphere in world coordinates
        P = gs.cpt();//closes point to sphere in index space
        ASSERT_DOUBLES_EXACTLY_EQUAL(-5,dist);
        ASSERT_DOUBLES_EXACTLY_EQUAL(12,P[0]);
        ASSERT_DOUBLES_EXACTLY_EQUAL(16,P[1]);
        ASSERT_DOUBLES_EXACTLY_EQUAL( 0,P[2]);
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
