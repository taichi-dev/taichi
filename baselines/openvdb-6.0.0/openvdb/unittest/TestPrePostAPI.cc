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
#include <openvdb/math/Mat4.h>
#include <openvdb/math/Maps.h>
#include <openvdb/math/Transform.h>
#include <openvdb/util/MapsUtil.h>


class TestPrePostAPI: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestPrePostAPI);

    CPPUNIT_TEST(testMat4);
    CPPUNIT_TEST(testMat4Rotate);
    CPPUNIT_TEST(testMat4Scale);
    CPPUNIT_TEST(testMat4Shear);
    CPPUNIT_TEST(testMaps);
    CPPUNIT_TEST(testLinearTransform);
    CPPUNIT_TEST(testFrustumTransform);

    CPPUNIT_TEST_SUITE_END();

    void testMat4();
    void testMat4Rotate();
    void testMat4Scale();
    void testMat4Shear();
    void testMaps();
    void testLinearTransform();
    void testFrustumTransform();
    //void testIsType();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestPrePostAPI);


void
TestPrePostAPI::testMat4()
{
    using namespace openvdb::math;

    double TOL = 1e-7;


    Mat4d m = Mat4d::identity();
    Mat4d minv = Mat4d::identity();

    // create matrix with pre-API
    // Translate Shear Rotate Translate Scale matrix
    m.preScale(Vec3d(1, 2, 3));
    m.preTranslate(Vec3d(2, 3, 4));
    m.preRotate(X_AXIS, 20);
    m.preShear(X_AXIS, Y_AXIS, 2);
    m.preTranslate(Vec3d(2, 2, 2));

    // create inverse using the post-API
    minv.postScale(Vec3d(1.f, 1.f/2.f, 1.f/3.f));
    minv.postTranslate(-Vec3d(2, 3, 4));
    minv.postRotate(X_AXIS,-20);
    minv.postShear(X_AXIS, Y_AXIS, -2);
    minv.postTranslate(-Vec3d(2, 2, 2));

    Mat4d mtest = minv * m;

    // verify that the results is an identity
    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][0], 1, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][1], 1, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][2], 1, TOL);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][1], 0, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][2], 0, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][3], 0, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][0], 0, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][2], 0, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][3], 0, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][0], 0, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][1], 0, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][3], 0, TOL);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][0], 0, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][1], 0, TOL);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][2], 0, TOL);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][3], 1, TOL);
}


void
TestPrePostAPI::testMat4Rotate()
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    Mat4d rx, ry, rz;
    const double angle1 = 20. * M_PI / 180.;
    const double angle2 = 64. * M_PI / 180.;
    const double angle3 = 125. *M_PI / 180.;
    rx.setToRotation(Vec3d(1,0,0), angle1);
    ry.setToRotation(Vec3d(0,1,0), angle2);
    rz.setToRotation(Vec3d(0,0,1), angle3);

    Mat4d shear = Mat4d::identity();
    shear.setToShear(X_AXIS, Z_AXIS, 2.0);
    shear.preShear(Y_AXIS, X_AXIS, 3.0);
    shear.preTranslate(Vec3d(2,4,1));

    const Mat4d preResult = rz*ry*rx*shear;
    Mat4d mpre = shear;
    mpre.preRotate(X_AXIS, angle1);
    mpre.preRotate(Y_AXIS, angle2);
    mpre.preRotate(Z_AXIS, angle3);

    CPPUNIT_ASSERT( mpre.eq(preResult, TOL) );

    const Mat4d postResult = shear*rx*ry*rz;
    Mat4d mpost = shear;
    mpost.postRotate(X_AXIS, angle1);
    mpost.postRotate(Y_AXIS, angle2);
    mpost.postRotate(Z_AXIS, angle3);

    CPPUNIT_ASSERT( mpost.eq(postResult, TOL) );

    CPPUNIT_ASSERT( !mpost.eq(mpre, TOL));

}


void
TestPrePostAPI::testMat4Scale()
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    Mat4d mpre, mpost;
    double* pre  = mpre.asPointer();
    double* post = mpost.asPointer();
    for (int i = 0; i < 16; ++i) {
        pre[i] = double(i);
        post[i] = double(i);
    }

    Mat4d scale = Mat4d::identity();
    scale.setToScale(Vec3d(2, 3, 5.5));
    Mat4d preResult = scale * mpre;
    Mat4d postResult = mpost * scale;

    mpre.preScale(Vec3d(2, 3, 5.5));
    mpost.postScale(Vec3d(2, 3, 5.5));

    CPPUNIT_ASSERT( mpre.eq(preResult, TOL) );
    CPPUNIT_ASSERT( mpost.eq(postResult, TOL) );
}


void
TestPrePostAPI::testMat4Shear()
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    Mat4d mpre, mpost;
    double* pre  = mpre.asPointer();
    double* post = mpost.asPointer();
    for (int i = 0; i < 16; ++i) {
        pre[i] = double(i);
        post[i] = double(i);
    }

    Mat4d shear = Mat4d::identity();
    shear.setToShear(X_AXIS, Z_AXIS, 13.);
    Mat4d preResult = shear * mpre;
    Mat4d postResult = mpost * shear;

    mpre.preShear(X_AXIS, Z_AXIS, 13.);
    mpost.postShear(X_AXIS, Z_AXIS, 13.);

    CPPUNIT_ASSERT( mpre.eq(preResult, TOL) );
    CPPUNIT_ASSERT( mpost.eq(postResult, TOL) );
}


void
TestPrePostAPI::testMaps()
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    { // pre translate
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        const Vec3d trans(1,2,3);
        Mat4d correct = Mat4d::identity();
        correct.preTranslate(trans);
        {
            MapBase::Ptr base = usm.preTranslate(trans);
            Mat4d result = (base->getAffineMap())->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.preTranslate(trans)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.preTranslate(trans)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.preTranslate(trans)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
        const Mat4d result = am.preTranslate(trans)->getAffineMap()->getConstMat4();
        CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
    }
    { // post translate
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        const Vec3d trans(1,2,3);
        Mat4d correct = Mat4d::identity();
        correct.postTranslate(trans);
        {
            const Mat4d result = usm.postTranslate(trans)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.postTranslate(trans)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.postTranslate(trans)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.postTranslate(trans)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
        const Mat4d result = am.postTranslate(trans)->getAffineMap()->getConstMat4();
        CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
    }
    { // pre scale
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        const Vec3d scale(1,2,3);
        Mat4d correct = Mat4d::identity();
        correct.preScale(scale);
        {
            const Mat4d result = usm.preScale(scale)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.preScale(scale)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.preScale(scale)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.preScale(scale)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
        const Mat4d result = am.preScale(scale)->getAffineMap()->getConstMat4();
        CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
    }
    { // post scale
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        const Vec3d scale(1,2,3);
        Mat4d correct = Mat4d::identity();
        correct.postScale(scale);
        {
            const Mat4d result = usm.postScale(scale)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.postScale(scale)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.postScale(scale)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.postScale(scale)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
        const Mat4d result = am.postScale(scale)->getAffineMap()->getConstMat4();
        CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
    }
    { // pre shear
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        Mat4d correct = Mat4d::identity();
        correct.preShear(X_AXIS, Z_AXIS, 13.);
        {
            const Mat4d result = usm.preShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.preShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.preShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.preShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
        const Mat4d result = am.preShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
        CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
    }
    { // post shear
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        Mat4d correct = Mat4d::identity();
        correct.postShear(X_AXIS, Z_AXIS, 13.);
        {
            const Mat4d result = usm.postShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result =
                ustm.postShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.postShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.postShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = am.postShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
    }
    { // pre rotate
        const double angle1 = 20. * M_PI / 180.;
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        Mat4d correct = Mat4d::identity();
        correct.preRotate(X_AXIS, angle1);
        {
            const Mat4d result = usm.preRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.preRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.preRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.preRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = am.preRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
    }
    { // post rotate
        const double angle1 = 20. * M_PI / 180.;
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        Mat4d correct = Mat4d::identity();
        correct.postRotate(X_AXIS, angle1);
        {
            const Mat4d result = usm.postRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.postRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.postRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.postRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
        {
            const Mat4d result = am.postRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            CPPUNIT_ASSERT( correct.eq(result, TOL));
        }
    }
}


void
TestPrePostAPI::testLinearTransform()
{
    using namespace openvdb::math;

    double TOL = 1e-7;
    {
        Transform::Ptr t = Transform::createLinearTransform(1.f);
        Transform::Ptr tinv = Transform::createLinearTransform(1.f);

        // create matrix with pre-API
        // Translate Shear Rotate Translate Scale matrix
        t->preScale(Vec3d(1, 2, 3));
        t->preTranslate(Vec3d(2, 3, 4));
        t->preRotate(20);
        t->preShear(2, X_AXIS, Y_AXIS);
        t->preTranslate(Vec3d(2, 2, 2));

        // create inverse using the post-API
        tinv->postScale(Vec3d(1.f, 1.f/2.f, 1.f/3.f));
        tinv->postTranslate(-Vec3d(2, 3, 4));
        tinv->postRotate(-20);
        tinv->postShear(-2, X_AXIS, Y_AXIS);
        tinv->postTranslate(-Vec3d(2, 2, 2));


        // test this by verifying that equvilent interal matrix
        // represenations are inverses
        Mat4d m = t->baseMap()->getAffineMap()->getMat4();
        Mat4d minv = tinv->baseMap()->getAffineMap()->getMat4();

        Mat4d mtest = minv * m;

        // verify that the results is an identity
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][0], 1, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][1], 1, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][2], 1, TOL);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][1], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][2], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][3], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][0], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][2], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][3], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][0], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][1], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][3], 0, TOL);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][0], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][1], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][2], 0, TOL);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][3], 1, TOL);
    }

    {
        Transform::Ptr t = Transform::createLinearTransform(1.f);

        Mat4d m = Mat4d::identity();

        // create matrix with pre-API
        // Translate Shear Rotate Translate Scale matrix
        m.preScale(Vec3d(1, 2, 3));
        m.preTranslate(Vec3d(2, 3, 4));
        m.preRotate(X_AXIS, 20);
        m.preShear(X_AXIS, Y_AXIS, 2);
        m.preTranslate(Vec3d(2, 2, 2));

        t->preScale(Vec3d(1,2,3));
        t->preMult(m);
        t->postMult(m);

        Mat4d minv = Mat4d::identity();

        // create inverse using the post-API
        minv.postScale(Vec3d(1.f, 1.f/2.f, 1.f/3.f));
        minv.postTranslate(-Vec3d(2, 3, 4));
        minv.postRotate(X_AXIS,-20);
        minv.postShear(X_AXIS, Y_AXIS, -2);
        minv.postTranslate(-Vec3d(2, 2, 2));

        t->preMult(minv);
        t->postMult(minv);

        Mat4d mtest = t->baseMap()->getAffineMap()->getMat4();


        // verify that the results is the scale
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][0], 1, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][1], 2, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][2], 3, 1e-6);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][1], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][2], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][3], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][0], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][2], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][3], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][0], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][1], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][3], 0, TOL);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][0], 0, 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][1], 0, 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][2], 0, TOL);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][3], 1, TOL);
    }


}


void
TestPrePostAPI::testFrustumTransform()
{
    using namespace openvdb::math;

    using BBoxd = BBox<Vec3d>;

    double TOL = 1e-7;
    {

        BBoxd bbox(Vec3d(-5,-5,0), Vec3d(5,5,10));
        Transform::Ptr t = Transform::createFrustumTransform(
            bbox, /* taper*/ 1, /*depth*/10, /* voxel size */1.f);
        Transform::Ptr tinv = Transform::createFrustumTransform(
            bbox, /* taper*/ 1, /*depth*/10, /* voxel size */1.f);


        // create matrix with pre-API
        // Translate Shear Rotate Translate Scale matrix
        t->preScale(Vec3d(1, 2, 3));
        t->preTranslate(Vec3d(2, 3, 4));
        t->preRotate(20);
        t->preShear(2, X_AXIS, Y_AXIS);
        t->preTranslate(Vec3d(2, 2, 2));

        // create inverse using the post-API
        tinv->postScale(Vec3d(1.f, 1.f/2.f, 1.f/3.f));
        tinv->postTranslate(-Vec3d(2, 3, 4));
        tinv->postRotate(-20);
        tinv->postShear(-2, X_AXIS, Y_AXIS);
        tinv->postTranslate(-Vec3d(2, 2, 2));


        // test this by verifying that equvilent interal matrix
        // represenations are inverses
        NonlinearFrustumMap::Ptr frustum =
            openvdb::StaticPtrCast<NonlinearFrustumMap, MapBase>(t->baseMap());
        NonlinearFrustumMap::Ptr frustuminv =
            openvdb::StaticPtrCast<NonlinearFrustumMap, MapBase>(tinv->baseMap());

        Mat4d m = frustum->secondMap().getMat4();
        Mat4d minv = frustuminv->secondMap().getMat4();

        Mat4d mtest = minv * m;

        // verify that the results is an identity
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][0], 1, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][1], 1, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][2], 1, TOL);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][1], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][2], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][3], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][0], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][2], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][3], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][0], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][1], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][3], 0, TOL);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][0], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][1], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][2], 0, TOL);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][3], 1, TOL);
    }

    {

        BBoxd bbox(Vec3d(-5,-5,0), Vec3d(5,5,10));
        Transform::Ptr t = Transform::createFrustumTransform(
            bbox, /* taper*/ 1, /*depth*/10, /* voxel size */1.f);


        Mat4d m = Mat4d::identity();

        // create matrix with pre-API
        // Translate Shear Rotate Translate Scale matrix
        m.preScale(Vec3d(1, 2, 3));
        m.preTranslate(Vec3d(2, 3, 4));
        m.preRotate(X_AXIS, 20);
        m.preShear(X_AXIS, Y_AXIS, 2);
        m.preTranslate(Vec3d(2, 2, 2));

        t->preScale(Vec3d(1,2,3));
        t->preMult(m);
        t->postMult(m);

        Mat4d minv = Mat4d::identity();

        // create inverse using the post-API
        minv.postScale(Vec3d(1.f, 1.f/2.f, 1.f/3.f));
        minv.postTranslate(-Vec3d(2, 3, 4));
        minv.postRotate(X_AXIS,-20);
        minv.postShear(X_AXIS, Y_AXIS, -2);
        minv.postTranslate(-Vec3d(2, 2, 2));

        t->preMult(minv);
        t->postMult(minv);

        NonlinearFrustumMap::Ptr frustum =
            openvdb::StaticPtrCast<NonlinearFrustumMap, MapBase>(t->baseMap());
        Mat4d mtest = frustum->secondMap().getMat4();

        // verify that the results is the scale
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][0], 1, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][1], 2, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][2], 3, 1e-6);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][1], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][2], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[0][3], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][0], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][2], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[1][3], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][0], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][1], 0, TOL);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[2][3], 0, TOL);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][0], 0, 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][1], 0, 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][2], 0, TOL);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(mtest[3][3], 1, TOL);
    }


}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
