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

#include <openvdb/Exceptions.h>
#include <openvdb/math/Maps.h>
#include <openvdb/util/MapsUtil.h>
#include <cppunit/extensions/HelperMacros.h>


class TestMaps: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestMaps);
    CPPUNIT_TEST(testTranslation);
    CPPUNIT_TEST(testRotation);
    CPPUNIT_TEST(testScaleDefault);
    CPPUNIT_TEST(testScaleTranslate);
    CPPUNIT_TEST(testUniformScaleTranslate);
    CPPUNIT_TEST(testDecomposition);
    CPPUNIT_TEST(testFrustum);
    CPPUNIT_TEST(testCalcBoundingBox);
    CPPUNIT_TEST(testApproxInverse);
    CPPUNIT_TEST(testUniformScale);
    CPPUNIT_TEST(testJacobians);
    CPPUNIT_TEST_SUITE_END();

    void testTranslation();
    void testRotation();
    void testScaleDefault();
    void testScaleTranslate();
    void testUniformScaleTranslate();
    void testDecomposition();
    void testFrustum();
    void testCalcBoundingBox();
    void testApproxInverse();
    void testUniformScale();
    void testJacobians();
    //void testIsType();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMaps);

void
TestMaps::testApproxInverse()
{
    using namespace openvdb::math;

    Mat4d singular = Mat4d::identity();
    singular[1][1] = 0.f;
    {
        Mat4d singularInv = approxInverse(singular);

        CPPUNIT_ASSERT( singular == singularInv );
    }
    {
        Mat4d rot = Mat4d::identity();
        rot.setToRotation(X_AXIS, M_PI/4.);

        Mat4d rotInv = rot.inverse();
        Mat4d mat = rotInv * singular * rot;

        Mat4d singularInv = approxInverse(mat);

        // this matrix is equal to its own singular inverse
        CPPUNIT_ASSERT( mat.eq(singularInv) );

    }
    {
        Mat4d m = Mat4d::identity();
        m[0][1] = 1;

        // should give true inverse, since this matrix has det=1
        Mat4d minv = approxInverse(m);

        Mat4d prod = m * minv;
        CPPUNIT_ASSERT( prod.eq( Mat4d::identity() ) );
    }
    {
        Mat4d m = Mat4d::identity();
        m[0][1] = 1;
        m[1][1] = 0;
        // should give true inverse, since this matrix has det=1
        Mat4d minv = approxInverse(m);

        Mat4d expected = Mat4d::zero();
        expected[3][3] = 1;
        CPPUNIT_ASSERT( minv.eq(expected ) );
    }


}


void
TestMaps::testUniformScale()
{
    using namespace openvdb::math;

    AffineMap map;

    CPPUNIT_ASSERT(map.hasUniformScale());

    // Apply uniform scale: should still have square voxels
    map.accumPreScale(Vec3d(2, 2, 2));

    CPPUNIT_ASSERT(map.hasUniformScale());

    // Apply a rotation, should still have squaure voxels.
    map.accumPostRotation(X_AXIS, 2.5);

    CPPUNIT_ASSERT(map.hasUniformScale());

    // non uniform scaling will stretch the voxels
    map.accumPostScale(Vec3d(1, 3, 1) );

    CPPUNIT_ASSERT(!map.hasUniformScale());
}

void
TestMaps::testTranslation()
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    TranslationMap::Ptr translation(new TranslationMap(Vec3d(1,1,1)));
    CPPUNIT_ASSERT(is_linear<TranslationMap>::value);

    TranslationMap another_translation(Vec3d(1,1,1));
    CPPUNIT_ASSERT(another_translation == *translation);

    TranslationMap::Ptr translate_by_two(new TranslationMap(Vec3d(2,2,2)));

    CPPUNIT_ASSERT(*translate_by_two != *translation);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(translate_by_two->determinant(), 1, TOL);

    CPPUNIT_ASSERT(translate_by_two->hasUniformScale());

    /// apply the map forward
    Vec3d unit(1,0,0);
    Vec3d result = translate_by_two->applyMap(unit);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), 3, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), 2, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), 2, TOL);

    /// invert the map
    result = translate_by_two->applyInverseMap(result);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), 1, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), 0, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), 0, TOL);

    /// Inverse Jacobian Transpose
    result = translate_by_two->applyIJT(result);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), 1, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), 0, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), 0, TOL);

    /// Jacobian Transpose
    result = translate_by_two->applyJT(translate_by_two->applyIJT(unit));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), unit(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), unit(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), unit(2), TOL);


    MapBase::Ptr inverse = translation->inverseMap();
    CPPUNIT_ASSERT(inverse->type() == TranslationMap::mapType());
    // apply the map forward and the inverse map back
    result = inverse->applyMap(translation->applyMap(unit));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), 1, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), 0, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), 0, TOL);


}

void
TestMaps::testScaleDefault()
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    // testing default constructor
    // should be the identity
    ScaleMap::Ptr  scale(new ScaleMap());
    Vec3d unit(1, 1, 1);

    Vec3d result = scale->applyMap(unit);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(unit(0), result(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(unit(1), result(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(unit(2), result(2), TOL);

    result = scale->applyInverseMap(unit);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(unit(0), result(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(unit(1), result(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(unit(2), result(2), TOL);


    MapBase::Ptr inverse = scale->inverseMap();
    CPPUNIT_ASSERT(inverse->type() == ScaleMap::mapType());
    // apply the map forward and the inverse map back
    result = inverse->applyMap(scale->applyMap(unit));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), unit(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), unit(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), unit(2), TOL);


}

void
TestMaps::testRotation()
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    double pi = 4.*atan(1.);
    UnitaryMap::Ptr  rotation(new UnitaryMap(Vec3d(1,0,0), pi/2));

    CPPUNIT_ASSERT(is_linear<UnitaryMap>::value);

    UnitaryMap another_rotation(Vec3d(1,0,0), pi/2.);
    CPPUNIT_ASSERT(another_rotation == *rotation);

    UnitaryMap::Ptr rotation_two(new UnitaryMap(Vec3d(1,0,0), pi/4.));

    CPPUNIT_ASSERT(*rotation_two != *rotation);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(rotation->determinant(), 1, TOL);

    CPPUNIT_ASSERT(rotation_two->hasUniformScale());

    /// apply the map forward
    Vec3d unit(0,1,0);
    Vec3d result = rotation->applyMap(unit);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, result(2), TOL);

    /// invert the map
    result = rotation->applyInverseMap(result);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, result(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result(2), TOL);

    /// Inverse Jacobian Transpose
    result = rotation_two->applyIJT(result); // rotate backwards
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(sqrt(2.)/2, result(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(sqrt(2.)/2, result(2), TOL);

    /// Jacobian Transpose
    result = rotation_two->applyJT(rotation_two->applyIJT(unit));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), unit(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), unit(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), unit(2), TOL);


    // Test inverse map
    MapBase::Ptr inverse = rotation->inverseMap();
    CPPUNIT_ASSERT(inverse->type() == UnitaryMap::mapType());
    // apply the map forward and the inverse map back
    result = inverse->applyMap(rotation->applyMap(unit));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), unit(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), unit(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), unit(2), TOL);
}


void
TestMaps::testScaleTranslate()
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    CPPUNIT_ASSERT(is_linear<ScaleTranslateMap>::value);

    TranslationMap::Ptr  translation(new TranslationMap(Vec3d(1,1,1)));
    ScaleMap::Ptr  scale(new ScaleMap(Vec3d(1,2,3)));

    ScaleTranslateMap::Ptr scaleAndTranslate(
        new ScaleTranslateMap(*scale, *translation));

    TranslationMap translate_by_two(Vec3d(2,2,2));
    ScaleTranslateMap another_scaleAndTranslate(*scale, translate_by_two);

    CPPUNIT_ASSERT(another_scaleAndTranslate != *scaleAndTranslate);

    CPPUNIT_ASSERT(!scaleAndTranslate->hasUniformScale());
    //CPPUNIT_ASSERT_DOUBLES_EQUAL(scaleAndTranslate->determinant(), 6, TOL);

    /// apply the map forward
    Vec3d unit(1,0,0);
    Vec3d result = scaleAndTranslate->applyMap(unit);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2, result(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, result(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, result(2), TOL);

    /// invert the map
    result = scaleAndTranslate->applyInverseMap(result);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, result(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result(2), TOL);

    /// Inverse Jacobian Transpose
    result = Vec3d(0,2,0);
    result = scaleAndTranslate->applyIJT(result );
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, result(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result(2), TOL);


    /// Jacobian Transpose
    result = scaleAndTranslate->applyJT(scaleAndTranslate->applyIJT(unit));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), unit(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), unit(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), unit(2), TOL);


    // Test inverse map
    MapBase::Ptr inverse = scaleAndTranslate->inverseMap();
    CPPUNIT_ASSERT(inverse->type() == ScaleTranslateMap::mapType());
    // apply the map forward and the inverse map back
    result = inverse->applyMap(scaleAndTranslate->applyMap(unit));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), unit(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), unit(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), unit(2), TOL);

}


void
TestMaps::testUniformScaleTranslate()
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    CPPUNIT_ASSERT(is_linear<UniformScaleMap>::value);
    CPPUNIT_ASSERT(is_linear<UniformScaleTranslateMap>::value);

    TranslationMap::Ptr  translation(new TranslationMap(Vec3d(1,1,1)));
    UniformScaleMap::Ptr  scale(new UniformScaleMap(2));

    UniformScaleTranslateMap::Ptr scaleAndTranslate(
        new UniformScaleTranslateMap(*scale, *translation));

    TranslationMap translate_by_two(Vec3d(2,2,2));
    UniformScaleTranslateMap another_scaleAndTranslate(*scale, translate_by_two);

    CPPUNIT_ASSERT(another_scaleAndTranslate != *scaleAndTranslate);
    CPPUNIT_ASSERT(scaleAndTranslate->hasUniformScale());
    //CPPUNIT_ASSERT_DOUBLES_EQUAL(scaleAndTranslate->determinant(), 6, TOL);

    /// apply the map forward
    Vec3d unit(1,0,0);
    Vec3d result = scaleAndTranslate->applyMap(unit);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3, result(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, result(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, result(2), TOL);

    /// invert the map
    result = scaleAndTranslate->applyInverseMap(result);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, result(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result(2), TOL);

    /// Inverse Jacobian Transpose
    result = Vec3d(0,2,0);
    result = scaleAndTranslate->applyIJT(result );
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, result(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, result(2), TOL);


    /// Jacobian Transpose
    result = scaleAndTranslate->applyJT(scaleAndTranslate->applyIJT(unit));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), unit(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), unit(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), unit(2), TOL);



    // Test inverse map
    MapBase::Ptr inverse = scaleAndTranslate->inverseMap();
    CPPUNIT_ASSERT(inverse->type() == UniformScaleTranslateMap::mapType());
    // apply the map forward and the inverse map back
    result = inverse->applyMap(scaleAndTranslate->applyMap(unit));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), unit(0), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), unit(1), TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), unit(2), TOL);

}


void
TestMaps::testDecomposition()
{
    using namespace openvdb::math;

    //double TOL = 1e-7;

    CPPUNIT_ASSERT(is_linear<UnitaryMap>::value);
    CPPUNIT_ASSERT(is_linear<SymmetricMap>::value);
    CPPUNIT_ASSERT(is_linear<PolarDecomposedMap>::value);
    CPPUNIT_ASSERT(is_linear<FullyDecomposedMap>::value);

    Mat4d matrix(Mat4d::identity());
    Vec3d input_translation(0,0,1);
    matrix.setTranslation(input_translation);


    matrix(0,0) =  1.8930039;
    matrix(1,0) = -0.120080537;
    matrix(2,0) = -0.497615212;

    matrix(0,1) = -0.120080537;
    matrix(1,1) =  2.643265436;
    matrix(2,1) = 0.6176957495;

    matrix(0,2) = -0.497615212;
    matrix(1,2) =  0.6176957495;
    matrix(2,2) = 1.4637305884;

    FullyDecomposedMap::Ptr decomp = createFullyDecomposedMap(matrix);

    /// the singular values
    const Vec3<double>& singular_values =
        decomp->firstMap().firstMap().secondMap().getScale();
    /// expected values
    Vec3d expected_values(2, 3, 1);

    CPPUNIT_ASSERT( isApproxEqual(singular_values, expected_values) );

    const Vec3<double>& the_translation = decomp->secondMap().secondMap().getTranslation();
    CPPUNIT_ASSERT( isApproxEqual(the_translation, input_translation));
}


void
TestMaps::testFrustum()
{
    using namespace openvdb::math;

    openvdb::BBoxd bbox(Vec3d(0), Vec3d(100));
    NonlinearFrustumMap frustum(bbox, 1./6., 5);
    /// frustum will have depth, far plane - near plane = 5
    /// the frustum has width 1 in the front and 6 in the back

    Vec3d trans(2,2,2);
    NonlinearFrustumMap::Ptr map =
        openvdb::StaticPtrCast<NonlinearFrustumMap, MapBase>(
            frustum.preScale(Vec3d(10,10,10))->postTranslate(trans));

    CPPUNIT_ASSERT(!map->hasUniformScale());

    Vec3d result;
    result = map->voxelSize();

    CPPUNIT_ASSERT( isApproxEqual(result.x(), 0.1));
    CPPUNIT_ASSERT( isApproxEqual(result.y(), 0.1));
    CPPUNIT_ASSERT( isApproxEqual(result.z(), 0.5, 0.0001));
    //--------- Front face
    Vec3d corner(0,0,0);
    result  = map->applyMap(corner);
    CPPUNIT_ASSERT(isApproxEqual(result, Vec3d(-5, -5, 0) + trans));

    corner = Vec3d(100,0,0);
    result  = map->applyMap(corner);
    CPPUNIT_ASSERT( isApproxEqual(result, Vec3d(5, -5, 0) + trans));

    corner = Vec3d(0,100,0);
    result  = map->applyMap(corner);
    CPPUNIT_ASSERT( isApproxEqual(result, Vec3d(-5, 5, 0) + trans));

    corner = Vec3d(100,100,0);
    result  = map->applyMap(corner);
    CPPUNIT_ASSERT( isApproxEqual(result, Vec3d(5, 5, 0) + trans));

    //--------- Back face
    corner = Vec3d(0,0,100);
    result  = map->applyMap(corner);
    CPPUNIT_ASSERT( isApproxEqual(result, Vec3d(-30, -30, 50) + trans)); // 10*(5/2 + 1/2) = 30

    corner = Vec3d(100,0,100);
    result  = map->applyMap(corner);
    CPPUNIT_ASSERT( isApproxEqual(result, Vec3d(30, -30, 50) + trans));

    corner = Vec3d(0,100,100);
    result  = map->applyMap(corner);
    CPPUNIT_ASSERT( isApproxEqual(result, Vec3d(-30, 30, 50) + trans));

    corner = Vec3d(100,100,100);
    result  = map->applyMap(corner);
    CPPUNIT_ASSERT( isApproxEqual(result, Vec3d(30, 30, 50) + trans));


    // invert a single corner
    result = map->applyInverseMap(Vec3d(30,30,50) + trans);
    CPPUNIT_ASSERT( isApproxEqual(result, Vec3d(100, 100, 100)));

    CPPUNIT_ASSERT(map->hasSimpleAffine());

    /// create a frustum from from camera type information

    // the location of the camera
    Vec3d position(100,10,1);
    // the direction the camera is pointing
    Vec3d direction(0,1,1);
    direction.normalize();

    // the up-direction for the camera
    Vec3d up(10,3,-3);

    // distance from camera to near-plane measured in the direction 'direction'
    double z_near = 100.;
    // depth of frustum to far-plane to near-plane
    double depth = 500.;
    //aspect ratio of frustum: width/height
    double aspect = 2;

    // voxel count in frustum.  the y_count = x_count / aspect
    Coord::ValueType x_count = 500;
    Coord::ValueType z_count = 5000;


    NonlinearFrustumMap frustumMap_from_camera(
        position, direction, up, aspect, z_near, depth, x_count, z_count);
    Vec3d center;
    // find the center of the near plane and make sure it is in the correct place
    center = Vec3d(0,0,0);
    center += frustumMap_from_camera.applyMap(Vec3d(0,0,0));
    center += frustumMap_from_camera.applyMap(Vec3d(500,0,0));
    center += frustumMap_from_camera.applyMap(Vec3d(0,250,0));
    center +=  frustumMap_from_camera.applyMap(Vec3d(500,250,0));
    center = center /4.;
    CPPUNIT_ASSERT( isApproxEqual(center, position + z_near * direction));
    // find the center of the far plane and make sure it is in the correct place
    center = Vec3d(0,0,0);
    center += frustumMap_from_camera.applyMap(Vec3d(  0,  0,5000));
    center += frustumMap_from_camera.applyMap(Vec3d(500,  0,5000));
    center += frustumMap_from_camera.applyMap(Vec3d(  0,250,5000));
    center += frustumMap_from_camera.applyMap(Vec3d(500,250,5000));
    center = center /4.;
    CPPUNIT_ASSERT( isApproxEqual(center, position + (z_near+depth) * direction));
    // check that the frustum has the correct heigh on the near plane
    Vec3d corner1  = frustumMap_from_camera.applyMap(Vec3d(0,0,0));
    Vec3d corner2  = frustumMap_from_camera.applyMap(Vec3d(0,250,0));
    Vec3d side = corner2-corner1;
    CPPUNIT_ASSERT( isApproxEqual( side.length(), 2 * up.length()));
    // check that the frustum is correctly oriented w.r.t up
    side.normalize();
    CPPUNIT_ASSERT( isApproxEqual( side * (up.length()), up));
    // check that the linear map inside the frustum is a simple affine map (i.e. has no shear)
    CPPUNIT_ASSERT(frustumMap_from_camera.hasSimpleAffine());
}


void
TestMaps::testCalcBoundingBox()
{
    using namespace openvdb::math;

    openvdb::BBoxd world_bbox(Vec3d(0,0,0), Vec3d(1,1,1));
    openvdb::BBoxd voxel_bbox;
    openvdb::BBoxd expected;
    {
        AffineMap affine;
        affine.accumPreScale(Vec3d(2,2,2));

        openvdb::util::calculateBounds<AffineMap>(affine, world_bbox, voxel_bbox);

        expected = openvdb::BBoxd(Vec3d(0,0,0), Vec3d(0.5, 0.5, 0.5));
        CPPUNIT_ASSERT(isApproxEqual(voxel_bbox.min(), expected.min()));
        CPPUNIT_ASSERT(isApproxEqual(voxel_bbox.max(), expected.max()));

        affine.accumPostTranslation(Vec3d(1,1,1));
        openvdb::util::calculateBounds<AffineMap>(affine, world_bbox, voxel_bbox);
        expected = openvdb::BBoxd(Vec3d(-0.5,-0.5,-0.5), Vec3d(0, 0, 0));
        CPPUNIT_ASSERT(isApproxEqual(voxel_bbox.min(), expected.min()));
        CPPUNIT_ASSERT(isApproxEqual(voxel_bbox.max(), expected.max()));
    }
    {
        AffineMap affine;
        affine.accumPreScale(Vec3d(2,2,2));
        affine.accumPostTranslation(Vec3d(1,1,1));
        // test a sphere:
        Vec3d center(0,0,0);
        double radius = 10;

        openvdb::util::calculateBounds<AffineMap>(affine, center, radius, voxel_bbox);
        expected = openvdb::BBoxd(Vec3d(-5.5,-5.5,-5.5), Vec3d(4.5, 4.5, 4.5));
        CPPUNIT_ASSERT(isApproxEqual(voxel_bbox.min(), expected.min()));
        CPPUNIT_ASSERT(isApproxEqual(voxel_bbox.max(), expected.max()));
    }
    {
        AffineMap affine;
        affine.accumPreScale(Vec3d(2,2,2));
        double pi = 4.*atan(1.);
        affine.accumPreRotation(X_AXIS, pi/4.);
        Vec3d center(0,0,0);
        double radius = 10;

        openvdb::util::calculateBounds<AffineMap>(affine, center, radius, voxel_bbox);
        expected = openvdb::BBoxd(Vec3d(-5,-5,-5), Vec3d(5, 5, 5));
        CPPUNIT_ASSERT(isApproxEqual(voxel_bbox.min(), expected.min()));
        CPPUNIT_ASSERT(isApproxEqual(voxel_bbox.max(), expected.max()));
    }
    {
        AffineMap affine;
        affine.accumPreScale(Vec3d(2,1,1));
        double pi = 4.*atan(1.);
        affine.accumPreRotation(X_AXIS, pi/4.);
        Vec3d center(0,0,0);
        double radius = 10;

        openvdb::util::calculateBounds<AffineMap>(affine, center, radius, voxel_bbox);
        expected = openvdb::BBoxd(Vec3d(-5,-10,-10), Vec3d(5, 10, 10));
        CPPUNIT_ASSERT(isApproxEqual(voxel_bbox.min(), expected.min()));
        CPPUNIT_ASSERT(isApproxEqual(voxel_bbox.max(), expected.max()));
     }
     {
        AffineMap affine;
        affine.accumPreScale(Vec3d(2,1,1));
        double pi = 4.*atan(1.);
        affine.accumPreRotation(X_AXIS, pi/4.);
        affine.accumPostTranslation(Vec3d(1,1,1));
        Vec3d center(1,1,1);
        double radius = 10;

        openvdb::util::calculateBounds<AffineMap>(affine, center, radius, voxel_bbox);
        expected = openvdb::BBoxd(Vec3d(-5,-10,-10), Vec3d(5, 10, 10));
        CPPUNIT_ASSERT(isApproxEqual(voxel_bbox.min(), expected.min()));
        CPPUNIT_ASSERT(isApproxEqual(voxel_bbox.max(), expected.max()));
     }
     {
         openvdb::BBoxd bbox(Vec3d(0), Vec3d(100));
         NonlinearFrustumMap frustum(bbox, 2, 5);
         NonlinearFrustumMap::Ptr map =
             openvdb::StaticPtrCast<NonlinearFrustumMap, MapBase>(
                 frustum.preScale(Vec3d(2,2,2)));
         Vec3d center(20,20,10);
         double radius(1);

         openvdb::util::calculateBounds<NonlinearFrustumMap>(*map, center, radius, voxel_bbox);
     }
}
void
TestMaps::testJacobians()
{
    using namespace openvdb::math;
    const double TOL = 1e-7;
    {
        AffineMap affine;

        const int n = 10;
        const double dtheta = M_PI / n;

        const Vec3d test(1,2,3);
        const Vec3d origin(0,0,0);

        for (int i = 0; i < n; ++i) {
            double theta = i * dtheta;

            affine.accumPostRotation(X_AXIS, theta);

            Vec3d result = affine.applyJacobian(test);
            Vec3d expected = affine.applyMap(test) - affine.applyMap(origin);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), expected(0), TOL);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), expected(1), TOL);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), expected(2), TOL);

            Vec3d tmp = affine.applyInverseJacobian(result);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(0), test(0), TOL);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(1), test(1), TOL);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(2), test(2), TOL);
        }
    }

    {
        UniformScaleMap scale(3);
        const Vec3d test(1,2,3);
        const Vec3d origin(0,0,0);


        Vec3d result = scale.applyJacobian(test);
        Vec3d expected = scale.applyMap(test) - scale.applyMap(origin);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), expected(0), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), expected(1), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), expected(2), TOL);

        Vec3d tmp = scale.applyInverseJacobian(result);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(0), test(0), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(1), test(1), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(2), test(2), TOL);
    }

    {
        ScaleMap scale(Vec3d(1,2,3));
        const Vec3d test(1,2,3);
        const Vec3d origin(0,0,0);


        Vec3d result = scale.applyJacobian(test);
        Vec3d expected = scale.applyMap(test) - scale.applyMap(origin);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), expected(0), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), expected(1), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), expected(2), TOL);

        Vec3d tmp = scale.applyInverseJacobian(result);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(0), test(0), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(1), test(1), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(2), test(2), TOL);
    }
    {
        TranslationMap map(Vec3d(1,2,3));
        const Vec3d test(1,2,3);
        const Vec3d origin(0,0,0);


        Vec3d result = map.applyJacobian(test);
        Vec3d expected = map.applyMap(test) - map.applyMap(origin);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), expected(0), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), expected(1), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), expected(2), TOL);

        Vec3d tmp = map.applyInverseJacobian(result);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(0), test(0), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(1), test(1), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(2), test(2), TOL);
    }
    {
        ScaleTranslateMap map(Vec3d(1,2,3), Vec3d(3,5,4));
        const Vec3d test(1,2,3);
        const Vec3d origin(0,0,0);


        Vec3d result = map.applyJacobian(test);
        Vec3d expected = map.applyMap(test) - map.applyMap(origin);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(result(0), expected(0), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(result(1), expected(1), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(result(2), expected(2), TOL);

        Vec3d tmp = map.applyInverseJacobian(result);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(0), test(0), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(1), test(1), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(2), test(2), TOL);
    }
    {
        openvdb::BBoxd bbox(Vec3d(0), Vec3d(100));
        NonlinearFrustumMap frustum(bbox, 1./6., 5);
        /// frustum will have depth, far plane - near plane = 5
        /// the frustum has width 1 in the front and 6 in the back

        Vec3d trans(2,2,2);
        NonlinearFrustumMap::Ptr map =
            openvdb::StaticPtrCast<NonlinearFrustumMap, MapBase>(
                frustum.preScale(Vec3d(10,10,10))->postTranslate(trans));

        const Vec3d test(1,2,3);
        const Vec3d origin(0, 0, 0);

        // these two drop down to just the linear part
        Vec3d lresult = map->applyJacobian(test);
        Vec3d ltmp = map->applyInverseJacobian(lresult);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(ltmp(0), test(0), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(ltmp(1), test(1), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(ltmp(2), test(2), TOL);

        Vec3d isloc(4,5,6);
        // these two drop down to just the linear part
        Vec3d result = map->applyJacobian(test, isloc);
        Vec3d tmp = map->applyInverseJacobian(result, isloc);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(0), test(0), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(1), test(1), TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(tmp(2), test(2), TOL);



    }


}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
