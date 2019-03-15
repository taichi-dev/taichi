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

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h> // for createLevelSetSphere
#include <openvdb/tools/LevelSetUtil.h> // for sdfToFogVolume
#include <openvdb/tools/VolumeToSpheres.h> // for fillWithSpheres

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>


class TestVolumeToSpheres: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestVolumeToSpheres);
    CPPUNIT_TEST(testFromLevelSet);
    CPPUNIT_TEST(testFromFog);
    CPPUNIT_TEST(testMinimumSphereCount);
    CPPUNIT_TEST(testClosestSurfacePoint);
    CPPUNIT_TEST_SUITE_END();

    void testFromLevelSet();
    void testFromFog();
    void testMinimumSphereCount();
    void testClosestSurfacePoint();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestVolumeToSpheres);


////////////////////////////////////////


void
TestVolumeToSpheres::testFromLevelSet()
{
    const float
        radius = 20.0f,
        voxelSize = 1.0f,
        halfWidth = 3.0f;
    const openvdb::Vec3f center(15.0f, 13.0f, 16.0f);

    openvdb::FloatGrid::ConstPtr grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
        radius, center, voxelSize, halfWidth);

    const bool overlapping = false;
    const int instanceCount = 10000;
    const float
        isovalue = 0.0f,
        minRadius = 5.0f,
        maxRadius = std::numeric_limits<float>::max();
    const openvdb::Vec2i sphereCount(1, 100);

    {
        std::vector<openvdb::Vec4s> spheres;

        openvdb::tools::fillWithSpheres(*grid, spheres, sphereCount, overlapping,
            minRadius, maxRadius, isovalue, instanceCount);

        CPPUNIT_ASSERT_EQUAL(1, int(spheres.size()));

        //for (size_t i=0; i< spheres.size(); ++i) {
        //    std::cout << "\nSphere #" << i << ": " << spheres[i] << std::endl;
        //}

        const auto tolerance = 2.0 * voxelSize;
        CPPUNIT_ASSERT_DOUBLES_EQUAL(center[0], spheres[0][0], tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(center[1], spheres[0][1], tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(center[2], spheres[0][2], tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(radius,    spheres[0][3], tolerance);
    }
    {
        // Verify that an isovalue outside the narrow band still produces a valid sphere.
        std::vector<openvdb::Vec4s> spheres;
        openvdb::tools::fillWithSpheres(*grid, spheres, sphereCount,
            overlapping, minRadius, maxRadius, 1.5f * halfWidth, instanceCount);
        CPPUNIT_ASSERT_EQUAL(1, int(spheres.size()));
    }
    {
        // Verify that an isovalue inside the narrow band produces no spheres.
        std::vector<openvdb::Vec4s> spheres;
        openvdb::tools::fillWithSpheres(*grid, spheres, sphereCount,
            overlapping, minRadius, maxRadius, -1.5f * halfWidth, instanceCount);
        CPPUNIT_ASSERT_EQUAL(0, int(spheres.size()));
    }
}


void
TestVolumeToSpheres::testFromFog()
{
    const float
        radius = 20.0f,
        voxelSize = 1.0f,
        halfWidth = 3.0f;
    const openvdb::Vec3f center(15.0f, 13.0f, 16.0f);

    auto grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
        radius, center, voxelSize, halfWidth);
    openvdb::tools::sdfToFogVolume(*grid);

    const bool overlapping = false;
    const int instanceCount = 10000;
    const float
        isovalue = 0.01f,
        minRadius = 5.0f,
        maxRadius = std::numeric_limits<float>::max();
    const openvdb::Vec2i sphereCount(1, 100);

    {
        std::vector<openvdb::Vec4s> spheres;
        openvdb::tools::fillWithSpheres(*grid, spheres, sphereCount, overlapping,
            minRadius, maxRadius, isovalue, instanceCount);

        //for (size_t i=0; i< spheres.size(); ++i) {
        //    std::cout << "\nSphere #" << i << ": " << spheres[i] << std::endl;
        //}

        CPPUNIT_ASSERT_EQUAL(1, int(spheres.size()));

        const auto tolerance = 2.0 * voxelSize;
        CPPUNIT_ASSERT_DOUBLES_EQUAL(center[0], spheres[0][0], tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(center[1], spheres[0][1], tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(center[2], spheres[0][2], tolerance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(radius,    spheres[0][3], tolerance);
    }
    {
        // Verify that an isovalue outside the narrow band still produces valid spheres.
        std::vector<openvdb::Vec4s> spheres;
        openvdb::tools::fillWithSpheres(*grid, spheres, sphereCount, overlapping,
            minRadius, maxRadius, 10.0f, instanceCount);
        CPPUNIT_ASSERT(!spheres.empty());
    }
}


void
TestVolumeToSpheres::testMinimumSphereCount()
{
    using namespace openvdb;
    {
        auto grid = tools::createLevelSetSphere<FloatGrid>(/*radius=*/5.0f,
            /*center=*/Vec3f(15.0f, 13.0f, 16.0f), /*voxelSize=*/1.0f, /*halfWidth=*/3.0f);

        // Verify that the requested minimum number of spheres is generated, for various minima.
        const int maxSphereCount = 100;
        for (int minSphereCount = 1; minSphereCount < 20; minSphereCount += 5) {
            std::vector<Vec4s> spheres;
            tools::fillWithSpheres(*grid, spheres, Vec2i(minSphereCount, maxSphereCount),
                /*overlapping=*/true, /*minRadius=*/2.0f);

            // Given the relatively large minimum radius, the actual sphere count
            // should be no larger than the requested mimimum count.
            CPPUNIT_ASSERT_EQUAL(minSphereCount, int(spheres.size()));
            //CPPUNIT_ASSERT(int(spheres.size()) >= minSphereCount);
            CPPUNIT_ASSERT(int(spheres.size()) <= maxSphereCount);
        }
    }
    {
        // One step in the sphere packing algorithm is to erode the active voxel mask
        // of the input grid.  Previously, for very small grids this sometimes resulted in
        // an empty mask and therefore no spheres.  Verify that that no longer happens
        // (as long as the minimum sphere count is nonzero).

        FloatGrid grid;
        CoordBBox bbox(Coord(1), Coord(2));
        grid.fill(bbox, 1.0f);

        const float minRadius = 1.0f;
        const Vec2i sphereCount(5, 100);

        std::vector<Vec4s> spheres;
        tools::fillWithSpheres(grid, spheres, sphereCount, /*overlapping=*/true, minRadius);

        CPPUNIT_ASSERT(int(spheres.size()) >= sphereCount[0]);
    }
}


void
TestVolumeToSpheres::testClosestSurfacePoint()
{
    using namespace openvdb;

    const float voxelSize = 1.0f;
    const Vec3f center{0.0f}; // ensure multiple internal nodes

    for (const float radius: { 8.0f, 50.0f }) {
        // Construct a spherical level set.
        const auto sphere = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize);
        CPPUNIT_ASSERT(sphere);

        // Construct the corners of a cube that exactly encloses the sphere.
        const std::vector<Vec3R> corners{
            { -radius, -radius, -radius },
            { -radius, -radius,  radius },
            { -radius,  radius, -radius },
            { -radius,  radius,  radius },
            {  radius, -radius, -radius },
            {  radius, -radius,  radius },
            {  radius,  radius, -radius },
            {  radius,  radius,  radius },
        };
        // Compute the distance from a corner of the cube to the surface of the sphere.
        const auto distToSurface = Vec3d{radius}.length() - radius;

        auto csp = tools::ClosestSurfacePoint<FloatGrid>::create(*sphere);
        CPPUNIT_ASSERT(csp);

        // Move each corner point to the closest surface point.
        auto points = corners;
        std::vector<float> distances;
        bool ok = csp->searchAndReplace(points, distances);
        CPPUNIT_ASSERT(ok);
        CPPUNIT_ASSERT_EQUAL(8, int(points.size()));
        CPPUNIT_ASSERT_EQUAL(8, int(distances.size()));

        for (auto d: distances) {
            CPPUNIT_ASSERT((std::abs(d - distToSurface) / distToSurface) < 0.01); // rel err < 1%
        }
        for (int i = 0; i < 8; ++i) {
            const auto intersection = corners[i] + distToSurface * (center - corners[i]).unit();
            CPPUNIT_ASSERT(points[i].eq(intersection, /*tolerance=*/0.1));
        }

        // Place a point inside the sphere.
        points.clear();
        distances.clear();
        points.emplace_back(1, 0, 0);
        ok = csp->searchAndReplace(points, distances);
        CPPUNIT_ASSERT(ok);
        CPPUNIT_ASSERT_EQUAL(1, int(points.size()));
        CPPUNIT_ASSERT_EQUAL(1, int(distances.size()));
        CPPUNIT_ASSERT((std::abs(radius - 1 - distances[0]) / (radius - 1)) < 0.01);
        CPPUNIT_ASSERT(points[0].eq(Vec3R{radius, 0, 0}, /*tolerance=*/0.5));
            ///< @todo off by half a voxel in y and z
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
