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

/// @file unittest/TestLevelSetRayIntersector.cc
/// @author Ken Museth

// Uncomment to enable statistics of ray-intersections
//#define STATS_TEST

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/math/Ray.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/RayTracer.h>// for Film
#ifdef STATS_TEST
//only needed for statistics
#include <openvdb/math/Stats.h>
#include <openvdb/util/CpuTimer.h>
#include <iostream>
#endif


#define ASSERT_DOUBLES_APPROX_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/1.e-6);


class TestLevelSetRayIntersector : public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestLevelSetRayIntersector);
    CPPUNIT_TEST(tests);

#ifdef STATS_TEST
    CPPUNIT_TEST(stats);
#endif

    CPPUNIT_TEST_SUITE_END();

    void tests();
#ifdef STATS_TEST
    void stats();
#endif
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLevelSetRayIntersector);

void
TestLevelSetRayIntersector::tests()
{
    using namespace openvdb;
    typedef math::Ray<double>  RayT;
    typedef RayT::Vec3Type     Vec3T;

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(20.0f, 0.0f, 0.0f);
        const float s = 0.5f, w = 2.0f;

        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);

        tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);

        const Vec3T dir(1.0, 0.0, 0.0);
        const Vec3T eye(2.0, 0.0, 0.0);
        const RayT ray(eye, dir);
        //std::cerr << ray << std::endl;
        Vec3T xyz(0);
        Real time = 0;
        CPPUNIT_ASSERT(lsri.intersectsWS(ray, xyz, time));
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[2]);
        ASSERT_DOUBLES_APPROX_EQUAL(13.0, time);
        double t0=0, t1=0;
        CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(t0, time);
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << " time = " << time << std::endl;
        CPPUNIT_ASSERT(ray(t0) == xyz);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(20.0f, 0.0f, 0.0f);
        const float s = 0.5f, w = 2.0f;

        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);

        tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);

        const Vec3T dir(1.0,-0.0,-0.0);
        const Vec3T eye(2.0, 0.0, 0.0);
        const RayT ray(eye, dir);
        //std::cerr << ray << std::endl;
        Vec3T xyz(0);
        Real time = 0;
        CPPUNIT_ASSERT(lsri.intersectsWS(ray, xyz, time));
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[2]);
        ASSERT_DOUBLES_APPROX_EQUAL(13.0, time);
        double t0=0, t1=0;
        CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(t0, time);
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        CPPUNIT_ASSERT(ray(t0) == xyz);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(0.0f, 20.0f, 0.0f);
        const float s = 1.5f, w = 2.0f;

        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);

        tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);

        const Vec3T dir(0.0, 1.0, 0.0);
        const Vec3T eye(0.0,-2.0, 0.0);
        RayT ray(eye, dir);
        Vec3T xyz(0);
        Real time = 0;
        CPPUNIT_ASSERT(lsri.intersectsWS(ray, xyz, time));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[2]);
        ASSERT_DOUBLES_APPROX_EQUAL(17.0, time);
        double t0=0, t1=0;
        CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(t0, time);
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[0]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, ray(t0)[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[2]);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(0.0f, 20.0f, 0.0f);
        const float s = 1.5f, w = 2.0f;

        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);

        tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);

        const Vec3T dir(-0.0, 1.0,-0.0);
        const Vec3T eye( 0.0,-2.0, 0.0);
        RayT ray(eye, dir);
        Vec3T xyz(0);
        Real time = 0;
        CPPUNIT_ASSERT(lsri.intersectsWS(ray, xyz, time));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[2]);
        ASSERT_DOUBLES_APPROX_EQUAL(17.0, time);
        double t0=0, t1=0;
        CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(t0, time);
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[0]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, ray(t0)[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[2]);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(0.0f, 0.0f, 20.0f);
        const float s = 1.0f, w = 3.0f;

        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);

        typedef tools::LinearSearchImpl<FloatGrid> SearchImplT;
        tools::LevelSetRayIntersector<FloatGrid, SearchImplT, -1> lsri(*ls);

        const Vec3T dir(0.0, 0.0, 1.0);
        const Vec3T eye(0.0, 0.0, 4.0);
        RayT ray(eye, dir);
        Vec3T xyz(0);
        Real time = 0;
        CPPUNIT_ASSERT(lsri.intersectsWS(ray, xyz, time));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, xyz[2]);
        ASSERT_DOUBLES_APPROX_EQUAL(11.0, time);
        double t0=0, t1=0;
        CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(t0, time);
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, ray(t0)[2]);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(0.0f, 0.0f, 20.0f);
        const float s = 1.0f, w = 3.0f;

        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);

        typedef tools::LinearSearchImpl<FloatGrid> SearchImplT;
        tools::LevelSetRayIntersector<FloatGrid, SearchImplT, -1> lsri(*ls);

        const Vec3T dir(-0.0,-0.0, 1.0);
        const Vec3T eye( 0.0, 0.0, 4.0);
        RayT ray(eye, dir);
        Vec3T xyz(0);
        Real time = 0;
        CPPUNIT_ASSERT(lsri.intersectsWS(ray, xyz, time));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, xyz[2]);
        ASSERT_DOUBLES_APPROX_EQUAL(11.0, time);
        double t0=0, t1=0;
        CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(t0, time);
        //std::cerr << "t0 = " << t0 << " t1 = " << t1 << std::endl;
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, ray(t0)[2]);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(0.0f, 0.0f, 20.0f);
        const float s = 1.0f, w = 3.0f;

        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);

        typedef tools::LinearSearchImpl<FloatGrid> SearchImplT;
        tools::LevelSetRayIntersector<FloatGrid, SearchImplT, -1> lsri(*ls);

        const Vec3T dir(-0.0,-0.0, 1.0);
        const Vec3T eye( 0.0, 0.0, 4.0);
        RayT ray(eye, dir, 16.0);
        Vec3T xyz(0);
        Real time = 0;
        CPPUNIT_ASSERT(lsri.intersectsWS(ray, xyz, time));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(25.0, xyz[2]);
        ASSERT_DOUBLES_APPROX_EQUAL(21.0, time);
        double t0=0, t1=0;
        CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
        //std::cerr << "t0 = " << t0 << " t1 = " << t1 << std::endl;
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        ASSERT_DOUBLES_APPROX_EQUAL(t1, time);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(25.0, ray(t1)[2]);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(10.0f, 10.0f, 10.0f);
        const float s = 1.0f, w = 3.0f;

        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);

        tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);

        Vec3T dir(1.0, 1.0, 1.0); dir.normalize();
        const Vec3T eye(0.0, 0.0, 0.0);
        RayT ray(eye, dir);
        //std::cerr << "ray: " << ray << std::endl;
        Vec3T xyz(0);
        Real time = 0;
        CPPUNIT_ASSERT(lsri.intersectsWS(ray, xyz, time));
        //std::cerr << "\nIntersection at  xyz = " << xyz << std::endl;
        //analytical intersection test
        double t0=0, t1=0;
        CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(t0, time);
        ASSERT_DOUBLES_APPROX_EQUAL((ray(t0)-c).length()-r, 0);
        ASSERT_DOUBLES_APPROX_EQUAL((ray(t1)-c).length()-r, 0);
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "\nray("<<t1<<")="<<ray(t1)<<std::endl;
        const Vec3T delta = xyz - ray(t0);
        //std::cerr << "delta = " << delta << std::endl;
        //std::cerr << "|delta|/dx=" << (delta.length()/ls->voxelSize()[0]) << std::endl;
        ASSERT_DOUBLES_APPROX_EQUAL(0, delta.length());
    }

    {// test intersections against a high-resolution level set sphere @1024^3
        const float r = 5.0f;
        const Vec3f c(10.0f, 10.0f, 20.0f);
        const float s = 0.01f, w = 2.0f;
        double t0=0, t1=0;
        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);

        typedef tools::LinearSearchImpl<FloatGrid, /*iterations=*/2> SearchImplT;
        tools::LevelSetRayIntersector<FloatGrid, SearchImplT> lsri(*ls);

        Vec3T xyz(0);
        Real time = 0;
        const size_t width = 1024;
        const double dx = 20.0/width;
        const Vec3T dir(0.0, 0.0, 1.0);

        for (size_t i=0; i<width; ++i) {
            for (size_t j=0; j<width; ++j) {
                const Vec3T eye(dx*double(i), dx*double(j), 0.0);
                const RayT ray(eye, dir);
                if (lsri.intersectsWS(ray, xyz, time)){
                    CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, 100*(t0-time)/t0, /*tolerance=*/0.1);//percent
                    double delta = (ray(t0)-xyz).length()/s;//in voxel units
                    CPPUNIT_ASSERT(delta < 0.06);
                }
            }
        }
    }
}

#ifdef STATS_TEST
void
TestLevelSetRayIntersector::stats()
{
    using namespace openvdb;
    typedef math::Ray<double>  RayT;
    typedef RayT::Vec3Type     Vec3T;
    util::CpuTimer timer;

    {// generate an image, benchmarks and statistics

        // Generate a high-resolution level set sphere @1024^3
        const float r = 5.0f;
        const Vec3f c(10.0f, 10.0f, 20.0f);
        const float s = 0.01f, w = 2.0f;
        double t0=0, t1=0;
        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);

        typedef tools::LinearSearchImpl<FloatGrid, /*iterations=*/2> SearchImplT;
        tools::LevelSetRayIntersector<FloatGrid, SearchImplT> lsri(*ls);

        Vec3T xyz(0);
        const size_t width = 1024;
        const double dx = 20.0/width;
        const Vec3T dir(0.0, 0.0, 1.0);

        tools::Film film(width, width);
        math::Stats stats;
        math::Histogram hist(0.0, 0.1, 20);

        timer.start("\nSerial ray-intersections of sphere");
        for (size_t i=0; i<width; ++i) {
            for (size_t j=0; j<width; ++j) {
                const Vec3T eye(dx*i, dx*j, 0.0);
                const RayT ray(eye, dir);
                if (lsri.intersectsWS(ray, xyz)){
                    CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
                    double delta = (ray(t0)-xyz).length()/s;//in voxel units
                    stats.add(delta);
                    hist.add(delta);
                    if (delta > 0.01) {
                        film.pixel(i, j) = tools::Film::RGBA(1.0f, 0.0f, 0.0f);
                    } else {
                        film.pixel(i, j) = tools::Film::RGBA(0.0f, 1.0f, 0.0f);
                    }
                }
            }
        }
        timer.stop();

        film.savePPM("sphere_serial");
        stats.print("First hit");
        hist.print("First hit");
    }
}
#endif // STATS_TEST

#undef STATS_TEST

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
