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
#include <openvdb/math/Transform.h>
#include <sstream>


class TestTransform: public CppUnit::TestCase
{
public:
    virtual void setUp();
    virtual void tearDown();

    CPPUNIT_TEST_SUITE(TestTransform);
    CPPUNIT_TEST(testLinearTransform);
    CPPUNIT_TEST(testTransformEquality);
    CPPUNIT_TEST(testBackwardCompatibility);
    CPPUNIT_TEST(testIsIdentity);
    CPPUNIT_TEST(testBoundingBoxes);
    CPPUNIT_TEST_SUITE_END();

    void testLinearTransform();
    void testTransformEquality();
    void testBackwardCompatibility();
    void testIsIdentity();
    void testBoundingBoxes();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestTransform);


////////////////////////////////////////


void
TestTransform::setUp()
{
    openvdb::math::MapRegistry::clear();
    openvdb::math::AffineMap::registerMap();
    openvdb::math::ScaleMap::registerMap();
    openvdb::math::UniformScaleMap::registerMap();
    openvdb::math::TranslationMap::registerMap();
    openvdb::math::ScaleTranslateMap::registerMap();
    openvdb::math::UniformScaleTranslateMap::registerMap();
}


void
TestTransform::tearDown()
{
    openvdb::math::MapRegistry::clear();
}


////openvdb::////////////////////////////////////


void
TestTransform::testLinearTransform()
{
    using namespace openvdb;
    double TOL = 1e-7;

    // Test: Scaling
    math::Transform::Ptr t = math::Transform::createLinearTransform(0.5);

    Vec3R voxelSize = t->voxelSize();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, voxelSize[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, voxelSize[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, voxelSize[2], TOL);

    CPPUNIT_ASSERT(t->hasUniformScale());

    // world to index space
    Vec3R xyz(-1.0, 2.0, 4.0);
    xyz = t->worldToIndex(xyz);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-2.0, xyz[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 4.0, xyz[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 8.0, xyz[2], TOL);

    xyz = Vec3R(-0.7, 2.4, 4.7);

    // cell centered conversion
    Coord ijk = t->worldToIndexCellCentered(xyz);
    CPPUNIT_ASSERT_EQUAL(Coord(-1, 5, 9), ijk);

    // node centrered conversion
    ijk = t->worldToIndexNodeCentered(xyz);
    CPPUNIT_ASSERT_EQUAL(Coord(-2, 4, 9), ijk);

    // index to world space
    ijk = Coord(4, 2, -8);
    xyz = t->indexToWorld(ijk);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 2.0, xyz[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0, xyz[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-4.0, xyz[2], TOL);

    // I/O test
    {
        std::stringstream
            ss(std::stringstream::in | std::stringstream::out | std::stringstream::binary);

        t->write(ss);

        t = math::Transform::createLinearTransform();

        // Since we wrote only a fragment of a VDB file (in particular, we didn't
        // write the header), set the file format version number explicitly.
        io::setCurrentVersion(ss);

        t->read(ss);
    }

    // check map type
    CPPUNIT_ASSERT_EQUAL(math::UniformScaleMap::mapType(), t->baseMap()->type());

    voxelSize = t->voxelSize();

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, voxelSize[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, voxelSize[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, voxelSize[2], TOL);

    //////////

    // Test: Scale, translation & rotation
    t = math::Transform::createLinearTransform(2.0);

    // rotate, 180 deg, (produces a diagonal matrix that can be simplified into a scale map)
    // with diagonal -2, 2, -2
    const double PI = std::atan(1.0)*4;
    t->preRotate(PI, math::Y_AXIS);

    // this is just a rotation so it will have uniform scale
    CPPUNIT_ASSERT(t->hasUniformScale());

    CPPUNIT_ASSERT_EQUAL(math::ScaleMap::mapType(), t->baseMap()->type());

    voxelSize = t->voxelSize();
    xyz = t->worldToIndex(Vec3R(-2.0, -2.0, -2.0));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, voxelSize[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, voxelSize[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, voxelSize[2], TOL);

    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0, xyz[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, xyz[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0, xyz[2], TOL);

    // translate
    t->postTranslate(Vec3d(1.0, 0.0, 1.0));

    CPPUNIT_ASSERT_EQUAL(math::ScaleTranslateMap::mapType(), t->baseMap()->type());

    voxelSize = t->voxelSize();
    xyz = t->worldToIndex(Vec3R(-2.0, -2.0, -2.0));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, voxelSize[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, voxelSize[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, voxelSize[2], TOL);

    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.5, xyz[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, xyz[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.5, xyz[2], TOL);


    // I/O test
    {
        std::stringstream
            ss(std::stringstream::in | std::stringstream::out | std::stringstream::binary);

        t->write(ss);

        t = math::Transform::createLinearTransform();

        // Since we wrote only a fragment of a VDB file (in particular, we didn't
        // write the header), set the file format version number explicitly.
        io::setCurrentVersion(ss);

        t->read(ss);
    }

    // check map type
    CPPUNIT_ASSERT_EQUAL(math::ScaleTranslateMap::mapType(), t->baseMap()->type());

    voxelSize = t->voxelSize();

    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, voxelSize[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, voxelSize[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, voxelSize[2], TOL);

    xyz = t->worldToIndex(Vec3R(-2.0, -2.0, -2.0));

    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.5, xyz[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, xyz[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.5, xyz[2], TOL);

    // new transform
    t = math::Transform::createLinearTransform(1.0);

    // rotate 90 deg
    t->preRotate( std::atan(1.0) * 2 , math::Y_AXIS);

    // check map type
    CPPUNIT_ASSERT_EQUAL(math::AffineMap::mapType(), t->baseMap()->type());

    xyz = t->worldToIndex(Vec3R(1.0, 1.0, 1.0));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, xyz[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0, xyz[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0, xyz[2], TOL);

    // I/O test
    {
        std::stringstream
            ss(std::stringstream::in | std::stringstream::out | std::stringstream::binary);

        t->write(ss);

        t = math::Transform::createLinearTransform();

        CPPUNIT_ASSERT_EQUAL(math::UniformScaleMap::mapType(), t->baseMap()->type());

        xyz = t->worldToIndex(Vec3R(1.0, 1.0, 1.0));

        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, xyz[0], TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, xyz[1], TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, xyz[2], TOL);

        // Since we wrote only a fragment of a VDB file (in particular, we didn't
        // write the header), set the file format version number explicitly.
        io::setCurrentVersion(ss);

        t->read(ss);
    }

    // check map type
    CPPUNIT_ASSERT_EQUAL(math::AffineMap::mapType(), t->baseMap()->type());

    xyz = t->worldToIndex(Vec3R(1.0, 1.0, 1.0));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, xyz[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0, xyz[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0, xyz[2], TOL);
}


////////////////////////////////////////

void
TestTransform::testTransformEquality()
{
    using namespace openvdb;

    // maps created in different ways may be equivalent
    math::Transform::Ptr t1 = math::Transform::createLinearTransform(0.5);
    math::Mat4d mat = math::Mat4d::identity();
    mat.preScale(math::Vec3d(0.5, 0.5, 0.5));
    math::Transform::Ptr t2 = math::Transform::createLinearTransform(mat);

    CPPUNIT_ASSERT( *t1 == *t2);

    // test that the auto-convert to the simplest form worked
    CPPUNIT_ASSERT( t1->mapType() == t2->mapType());


    mat.preScale(math::Vec3d(1., 1., .4));
    math::Transform::Ptr t3 = math::Transform::createLinearTransform(mat);

    CPPUNIT_ASSERT( *t1 != *t3);

    // test equality between different but equivalent maps
    math::UniformScaleTranslateMap::Ptr ustmap(
        new math::UniformScaleTranslateMap(1.0, math::Vec3d(0,0,0)));
    math::Transform::Ptr t4( new math::Transform( ustmap) );
    CPPUNIT_ASSERT( t4->baseMap()->isType<math::UniformScaleMap>() );
    math::Transform::Ptr t5( new math::Transform);  // constructs with a scale map
    CPPUNIT_ASSERT( t5->baseMap()->isType<math::ScaleMap>() );

    CPPUNIT_ASSERT( *t5 == *t4);

    CPPUNIT_ASSERT( t5->mapType() != t4->mapType() );

    // test inequatlity of two maps of the same type
    math::UniformScaleTranslateMap::Ptr ustmap2(
        new math::UniformScaleTranslateMap(1.0, math::Vec3d(1,0,0)));
    math::Transform::Ptr t6( new math::Transform( ustmap2) );
    CPPUNIT_ASSERT( t6->baseMap()->isType<math::UniformScaleTranslateMap>() );
    CPPUNIT_ASSERT( *t6 != *t4);

    // test comparison of linear to nonlinear map
    openvdb::BBoxd bbox(math::Vec3d(0), math::Vec3d(100));
    math::Transform::Ptr frustum = math::Transform::createFrustumTransform(bbox, 0.25, 10);

    CPPUNIT_ASSERT( *frustum != *t1 );


}
////////////////////////////////////////

void
TestTransform::testBackwardCompatibility()
{
    using namespace openvdb;
    double TOL = 1e-7;

    // Register maps
    math::MapRegistry::clear();
    math::AffineMap::registerMap();
    math::ScaleMap::registerMap();
    math::TranslationMap::registerMap();
    math::ScaleTranslateMap::registerMap();

    std::stringstream
        ss(std::stringstream::in | std::stringstream::out | std::stringstream::binary);


    //////////

    // Construct and write out an old transform that gets converted
    // into a ScaleMap on read.

    // First write the old transform type name
    writeString(ss, Name("LinearTransform"));

    // Second write the old transform's base class membes.
    Coord tmpMin(0), tmpMax(1);
    ss.write(reinterpret_cast<char*>(&tmpMin), sizeof(Coord::ValueType) * 3);
    ss.write(reinterpret_cast<char*>(&tmpMax), sizeof(Coord::ValueType) * 3);

    // Last write out the old linear transform's members
    math::Mat4d tmpLocalToWorld = math::Mat4d::identity(),
                tmpWorldToLocal = math::Mat4d::identity(),
                tmpVoxelToLocal = math::Mat4d::identity(),
                tmpLocalToVoxel = math::Mat4d::identity();

    tmpVoxelToLocal.preScale(math::Vec3d(0.5, 0.5, 0.5));

    tmpLocalToWorld.write(ss);
    tmpWorldToLocal.write(ss);
    tmpVoxelToLocal.write(ss);
    tmpLocalToVoxel.write(ss);

    // Read in the old transform and converting it to the new map based implementation.

    math::Transform::Ptr t = math::Transform::createLinearTransform(1.0);

    t->read(ss);

    // check map type
    CPPUNIT_ASSERT_EQUAL(math::UniformScaleMap::mapType(), t->baseMap()->type());

    Vec3d voxelSize = t->voxelSize();

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, voxelSize[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, voxelSize[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, voxelSize[2], TOL);

    Vec3d xyz = t->worldToIndex(Vec3d(-1.0, 2.0, 4.0));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-2.0, xyz[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 4.0, xyz[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 8.0, xyz[2], TOL);


    //////////

    // Construct and write out an old transform that gets converted
    // into a ScaleTranslateMap on read.

    ss.clear();
    writeString(ss, Name("LinearTransform"));
    ss.write(reinterpret_cast<char*>(&tmpMin), sizeof(Coord::ValueType) * 3);
    ss.write(reinterpret_cast<char*>(&tmpMax), sizeof(Coord::ValueType) * 3);
    tmpLocalToWorld = math::Mat4d::identity(),
    tmpWorldToLocal = math::Mat4d::identity(),
    tmpVoxelToLocal = math::Mat4d::identity(),
    tmpLocalToVoxel = math::Mat4d::identity();

    tmpVoxelToLocal.preScale(math::Vec3d(2.0, 2.0, 2.0));
    tmpLocalToWorld.setTranslation(math::Vec3d(1.0, 0.0, 1.0));

    tmpLocalToWorld.write(ss);
    tmpWorldToLocal.write(ss);
    tmpVoxelToLocal.write(ss);
    tmpLocalToVoxel.write(ss);

    // Read in the old transform and converting it to the new map based implementation.

    t = math::Transform::createLinearTransform(); // rest transform
    t->read(ss);

    CPPUNIT_ASSERT_EQUAL(math::UniformScaleTranslateMap::mapType(), t->baseMap()->type());

    voxelSize = t->voxelSize();

    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, voxelSize[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, voxelSize[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, voxelSize[2], TOL);

    xyz = t->worldToIndex(Vec3d(1.0, 1.0, 1.0));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, xyz[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, xyz[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, xyz[2], TOL);


    //////////

    // Construct and write out an old transform that gets converted
    // into a AffineMap on read.

    ss.clear();
    writeString(ss, Name("LinearTransform"));
    ss.write(reinterpret_cast<char*>(&tmpMin), sizeof(Coord::ValueType) * 3);
    ss.write(reinterpret_cast<char*>(&tmpMax), sizeof(Coord::ValueType) * 3);
    tmpLocalToWorld = math::Mat4d::identity(),
    tmpWorldToLocal = math::Mat4d::identity(),
    tmpVoxelToLocal = math::Mat4d::identity(),
    tmpLocalToVoxel = math::Mat4d::identity();

    tmpVoxelToLocal.preScale(math::Vec3d(1.0, 1.0, 1.0));
    tmpLocalToWorld.preRotate( math::Y_AXIS, std::atan(1.0) * 2);

    tmpLocalToWorld.write(ss);
    tmpWorldToLocal.write(ss);
    tmpVoxelToLocal.write(ss);
    tmpLocalToVoxel.write(ss);

    // Read in the old transform and converting it to the new map based implementation.

    t = math::Transform::createLinearTransform(); // rest transform
    t->read(ss);

    CPPUNIT_ASSERT_EQUAL(math::AffineMap::mapType(), t->baseMap()->type());

    voxelSize = t->voxelSize();

    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, voxelSize[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, voxelSize[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, voxelSize[2], TOL);

    xyz = t->worldToIndex(Vec3d(1.0, 1.0, 1.0));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, xyz[0], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0, xyz[1], TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0, xyz[2], TOL);
}


void
TestTransform::testIsIdentity()
{
    using namespace openvdb;
    math::Transform::Ptr t = math::Transform::createLinearTransform(1.0);

    CPPUNIT_ASSERT(t->isIdentity());

    t->preScale(Vec3d(2,2,2));

    CPPUNIT_ASSERT(!t->isIdentity());

    t->preScale(Vec3d(0.5,0.5,0.5));
    CPPUNIT_ASSERT(t->isIdentity());

    BBoxd bbox(math::Vec3d(-5,-5,0), Vec3d(5,5,10));
    math::Transform::Ptr f = math::Transform::createFrustumTransform(bbox,
                                                                     /*taper*/ 1,
                                                                     /*depth*/ 1,
                                                                     /*voxel size*/ 1);
    f->preScale(Vec3d(10,10,10));

    CPPUNIT_ASSERT(f->isIdentity());

    // rotate by PI/2
    f->postRotate(std::atan(1.0)*2, math::Y_AXIS);
    CPPUNIT_ASSERT(!f->isIdentity());

    f->postRotate(std::atan(1.0)*6, math::Y_AXIS);
    CPPUNIT_ASSERT(f->isIdentity());
}


void
TestTransform::testBoundingBoxes()
{
    using namespace openvdb;

    {
        math::Transform::ConstPtr t = math::Transform::createLinearTransform(0.5);

        const BBoxd bbox(Vec3d(-8.0), Vec3d(16.0));

        BBoxd xBBox = t->indexToWorld(bbox);
        CPPUNIT_ASSERT_EQUAL(Vec3d(-4.0), xBBox.min());
        CPPUNIT_ASSERT_EQUAL(Vec3d(8.0), xBBox.max());

        xBBox = t->worldToIndex(xBBox);
        CPPUNIT_ASSERT_EQUAL(bbox.min(), xBBox.min());
        CPPUNIT_ASSERT_EQUAL(bbox.max(), xBBox.max());
    }
    {
        const double PI = std::atan(1.0) * 4.0, SQRT2 = std::sqrt(2.0);

        math::Transform::Ptr t = math::Transform::createLinearTransform(1.0);
        t->preRotate(PI / 4.0, math::Z_AXIS);

        const BBoxd bbox(Vec3d(-10.0), Vec3d(10.0));

        BBoxd xBBox = t->indexToWorld(bbox); // expand in x and y by sqrt(2)
        CPPUNIT_ASSERT(Vec3d(-10.0 * SQRT2, -10.0 * SQRT2, -10.0).eq(xBBox.min()));
        CPPUNIT_ASSERT(Vec3d(10.0 * SQRT2, 10.0 * SQRT2, 10.0).eq(xBBox.max()));

        xBBox = t->worldToIndex(xBBox); // expand again in x and y by sqrt(2)
        CPPUNIT_ASSERT(Vec3d(-20.0, -20.0, -10.0).eq(xBBox.min()));
        CPPUNIT_ASSERT(Vec3d(20.0, 20.0, 10.0).eq(xBBox.max()));
    }

    /// @todo frustum transform
}


////////////////////////////////////////


/// @todo Test the new frustum transform.
/*
void
TestTransform::testNonlinearTransform()
{
    using namespace openvdb;
    double TOL = 1e-7;
}
*/

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
