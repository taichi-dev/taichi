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
#include <openvdb/math/Math.h>
#include <openvdb/math/Quat.h>
#include <openvdb/math/Mat4.h>

using namespace openvdb::math;

class TestQuat: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE( TestQuat );
    CPPUNIT_TEST( testConstructor );
    CPPUNIT_TEST( testAxisAngle );
    CPPUNIT_TEST( testOpPlus );
    CPPUNIT_TEST( testOpMinus );
    CPPUNIT_TEST( testOpMultiply );
    CPPUNIT_TEST( testInvert );
    CPPUNIT_TEST( testEulerAngles );
    CPPUNIT_TEST_SUITE_END();

    void testConstructor();
    void testAxisAngle();
    void testOpPlus();
    void testOpMinus();
    void testOpMultiply();
    void testInvert();
    void testEulerAngles();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestQuat);


void
TestQuat::testConstructor()
{
    {
        Quat<float> qq(1.23f, 2.34f, 3.45f, 4.56f);
        CPPUNIT_ASSERT( isExactlyEqual(qq.x(), 1.23f) );
        CPPUNIT_ASSERT( isExactlyEqual(qq.y(), 2.34f) );
        CPPUNIT_ASSERT( isExactlyEqual(qq.z(), 3.45f) );
        CPPUNIT_ASSERT( isExactlyEqual(qq.w(), 4.56f) );
    }

    {
        float a[] = { 1.23f, 2.34f, 3.45f, 4.56f };
        Quat<float> qq(a);
        CPPUNIT_ASSERT( isExactlyEqual(qq.x(), 1.23f) );
        CPPUNIT_ASSERT( isExactlyEqual(qq.y(), 2.34f) );
        CPPUNIT_ASSERT( isExactlyEqual(qq.z(), 3.45f) );
        CPPUNIT_ASSERT( isExactlyEqual(qq.w(), 4.56f) );
    }
}


void
TestQuat::testAxisAngle()
{
    float TOL = 1e-6f;

    Quat<float> q1(1.0f, 2.0f, 3.0f, 4.0f);
    Quat<float> q2(1.2f, 2.3f, 3.4f, 4.5f);

    Vec3s v(1, 2, 3);
    v.normalize();
    float a = float(M_PI / 4.f);

    Quat<float> q(v,a);
    float b = q.angle();
    Vec3s vv = q.axis();

    CPPUNIT_ASSERT( isApproxEqual(a, b, TOL) );
    CPPUNIT_ASSERT( v.eq(vv, TOL) );

    q1.setAxisAngle(v,a);
    b = q1.angle();
    vv = q1.axis();
    CPPUNIT_ASSERT( isApproxEqual(a, b, TOL) );
    CPPUNIT_ASSERT( v.eq(vv, TOL) );
}


void
TestQuat::testOpPlus()
{
    Quat<float> q1(1.0f, 2.0f, 3.0f, 4.0f);
    Quat<float> q2(1.2f, 2.3f, 3.4f, 4.5f);

    Quat<float> q = q1 + q2;

    float
        x=q1.x()+q2.x(), y=q1.y()+q2.y(), z=q1.z()+q2.z(), w=q1.w()+q2.w();
    CPPUNIT_ASSERT( isExactlyEqual(q.x(), x) );
    CPPUNIT_ASSERT( isExactlyEqual(q.y(), y) );
    CPPUNIT_ASSERT( isExactlyEqual(q.z(), z) );
    CPPUNIT_ASSERT( isExactlyEqual(q.w(), w) );

    q = q1;
    q += q2;
    CPPUNIT_ASSERT( isExactlyEqual(q.x(), x) );
    CPPUNIT_ASSERT( isExactlyEqual(q.y(), y) );
    CPPUNIT_ASSERT( isExactlyEqual(q.z(), z) );
    CPPUNIT_ASSERT( isExactlyEqual(q.w(), w) );

    q.add(q1,q2);
    CPPUNIT_ASSERT( isExactlyEqual(q.x(), x) );
    CPPUNIT_ASSERT( isExactlyEqual(q.y(), y) );
    CPPUNIT_ASSERT( isExactlyEqual(q.z(), z) );
    CPPUNIT_ASSERT( isExactlyEqual(q.w(), w) );
}


void
TestQuat::testOpMinus()
{
    Quat<float> q1(1.0f, 2.0f, 3.0f, 4.0f);
    Quat<float> q2(1.2f, 2.3f, 3.4f, 4.5f);

    Quat<float> q = q1 - q2;

    float
        x=q1.x()-q2.x(), y=q1.y()-q2.y(), z=q1.z()-q2.z(), w=q1.w()-q2.w();
    CPPUNIT_ASSERT( isExactlyEqual(q.x(), x) );
    CPPUNIT_ASSERT( isExactlyEqual(q.y(), y) );
    CPPUNIT_ASSERT( isExactlyEqual(q.z(), z) );
    CPPUNIT_ASSERT( isExactlyEqual(q.w(), w) );

    q = q1;
    q -= q2;
    CPPUNIT_ASSERT( isExactlyEqual(q.x(), x) );
    CPPUNIT_ASSERT( isExactlyEqual(q.y(), y) );
    CPPUNIT_ASSERT( isExactlyEqual(q.z(), z) );
    CPPUNIT_ASSERT( isExactlyEqual(q.w(), w) );

    q.sub(q1,q2);
    CPPUNIT_ASSERT( isExactlyEqual(q.x(), x) );
    CPPUNIT_ASSERT( isExactlyEqual(q.y(), y) );
    CPPUNIT_ASSERT( isExactlyEqual(q.z(), z) );
    CPPUNIT_ASSERT( isExactlyEqual(q.w(), w) );
}


void
TestQuat::testOpMultiply()
{
    Quat<float> q1(1.0f, 2.0f, 3.0f, 4.0f);
    Quat<float> q2(1.2f, 2.3f, 3.4f, 4.5f);

    Quat<float> q = q1 * 1.5f;

    CPPUNIT_ASSERT( isExactlyEqual(q.x(), float(1.5f)*q1.x()) );
    CPPUNIT_ASSERT( isExactlyEqual(q.y(), float(1.5f)*q1.y()) );
    CPPUNIT_ASSERT( isExactlyEqual(q.z(), float(1.5f)*q1.z()) );
    CPPUNIT_ASSERT( isExactlyEqual(q.w(), float(1.5f)*q1.w()) );

    q = q1;
    q *= 1.5f;
    CPPUNIT_ASSERT( isExactlyEqual(q.x(), float(1.5f)*q1.x()) );
    CPPUNIT_ASSERT( isExactlyEqual(q.y(), float(1.5f)*q1.y()) );
    CPPUNIT_ASSERT( isExactlyEqual(q.z(), float(1.5f)*q1.z()) );
    CPPUNIT_ASSERT( isExactlyEqual(q.w(), float(1.5f)*q1.w()) );

    q.scale(1.5f, q1);
    CPPUNIT_ASSERT( isExactlyEqual(q.x(), float(1.5f)*q1.x()) );
    CPPUNIT_ASSERT( isExactlyEqual(q.y(), float(1.5f)*q1.y()) );
    CPPUNIT_ASSERT( isExactlyEqual(q.z(), float(1.5f)*q1.z()) );
    CPPUNIT_ASSERT( isExactlyEqual(q.w(), float(1.5f)*q1.w()) );
}


void
TestQuat::testInvert()
{
    float TOL = 1e-6f;

    Quat<float> q1(1.0f, 2.0f, 3.0f, 4.0f);
    Quat<float> q2(1.2f, 2.3f, 3.4f, 4.5f);


    q1 = q2;
    q2 = q2.inverse();

    Quat<float> q = q1*q2;

    CPPUNIT_ASSERT( q.eq( Quat<float>(0,0,0,1), TOL ) );

    q1.normalize();
    q2 = q1.conjugate();
    q = q1*q2;
    CPPUNIT_ASSERT( q.eq( Quat<float>(0,0,0,1), TOL ) );
}


void
TestQuat::testEulerAngles()
{

    {
        double TOL = 1e-7;

        Mat4d rx, ry, rz;
        const double angle1 = 20. * M_PI / 180.;
        const double angle2 = 64. * M_PI / 180.;
        const double angle3 = 125. *M_PI / 180.;
        rx.setToRotation(Vec3d(1,0,0), angle1);
        ry.setToRotation(Vec3d(0,1,0), angle2);
        rz.setToRotation(Vec3d(0,0,1), angle3);

        Mat4d r = rx * ry * rz;

        const Quat<double> rot(r.getMat3());
        Vec3d result = rot.eulerAngles(ZYX_ROTATION);

        rx.setToRotation(Vec3d(1,0,0), result[0]);
        ry.setToRotation(Vec3d(0,1,0), result[1]);
        rz.setToRotation(Vec3d(0,0,1), result[2]);

        Mat4d rtest = rx * ry * rz;

        CPPUNIT_ASSERT(r.eq(rtest, TOL));
    }

    {
        double TOL = 1e-7;

        Mat4d rx, ry, rz;
        const double angle1 = 20. * M_PI / 180.;
        const double angle2 = 64. * M_PI / 180.;
        const double angle3 = 125. *M_PI / 180.;
        rx.setToRotation(Vec3d(1,0,0), angle1);
        ry.setToRotation(Vec3d(0,1,0), angle2);
        rz.setToRotation(Vec3d(0,0,1), angle3);

        Mat4d r = rz * ry * rx;

        const Quat<double> rot(r.getMat3());
        Vec3d result = rot.eulerAngles(XYZ_ROTATION);

        rx.setToRotation(Vec3d(1,0,0), result[0]);
        ry.setToRotation(Vec3d(0,1,0), result[1]);
        rz.setToRotation(Vec3d(0,0,1), result[2]);

        Mat4d rtest = rz * ry * rx;

        CPPUNIT_ASSERT(r.eq(rtest, TOL));
    }

    {
        double TOL = 1e-7;

        Mat4d rx, ry, rz;
        const double angle1 = 20. * M_PI / 180.;
        const double angle2 = 64. * M_PI / 180.;
        const double angle3 = 125. *M_PI / 180.;
        rx.setToRotation(Vec3d(1,0,0), angle1);
        ry.setToRotation(Vec3d(0,1,0), angle2);
        rz.setToRotation(Vec3d(0,0,1), angle3);

        Mat4d r = rz * rx * ry;

        const Quat<double> rot(r.getMat3());
        Vec3d result = rot.eulerAngles(YXZ_ROTATION);

        rx.setToRotation(Vec3d(1,0,0), result[0]);
        ry.setToRotation(Vec3d(0,1,0), result[1]);
        rz.setToRotation(Vec3d(0,0,1), result[2]);

        Mat4d rtest = rz * rx * ry;

        CPPUNIT_ASSERT(r.eq(rtest, TOL));
    }

    {
        const Quat<float> rot(X_AXIS, 1.0);
        Vec3s result = rot.eulerAngles(XZY_ROTATION);
        CPPUNIT_ASSERT_EQUAL(result, Vec3s(1,0,0));
    }

}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
