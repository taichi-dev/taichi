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

#include <openvdb/openvdb.h>
#include <openvdb/math/Math.h> // for math::Random01
#include <openvdb/tools/PointsToMask.h>
#include <openvdb/util/CpuTimer.h>
#include <cppunit/extensions/HelperMacros.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "util.h" // for genPoints


struct TestPointsToMask: public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(TestPointsToMask);
    CPPUNIT_TEST(testPointsToMask);
    CPPUNIT_TEST_SUITE_END();

    void testPointsToMask();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointsToMask);

////////////////////////////////////////

namespace {

class PointList
{
public:
    PointList(const std::vector<openvdb::Vec3R>& points) : mPoints(&points) {}

    size_t size() const { return mPoints->size(); }

    void getPos(size_t n, openvdb::Vec3R& xyz) const { xyz = (*mPoints)[n]; }
protected:
    std::vector<openvdb::Vec3R> const * const mPoints;
}; // PointList

} // namespace



////////////////////////////////////////


void
TestPointsToMask::testPointsToMask()
{
    {// BoolGrid
        // generate one point
        std::vector<openvdb::Vec3R> points;
        points.push_back( openvdb::Vec3R(-19.999, 4.50001, 6.71) );
        //points.push_back( openvdb::Vec3R( 20,-4.5,-5.2) );
        PointList pointList(points);

        // construct an empty mask grid
        openvdb::BoolGrid grid( false );
        const float voxelSize = 0.1f;
        grid.setTransform( openvdb::math::Transform::createLinearTransform(voxelSize) );
        CPPUNIT_ASSERT( grid.empty() );

        // generate mask from points
        openvdb::tools::PointsToMask<openvdb::BoolGrid> mask( grid );
        mask.addPoints( pointList );
        CPPUNIT_ASSERT(!grid.empty() );
        CPPUNIT_ASSERT_EQUAL( 1, int(grid.activeVoxelCount()) );
        openvdb::BoolGrid::ValueOnCIter iter = grid.cbeginValueOn();
        //std::cerr << "Coord = " << iter.getCoord() << std::endl;
        const openvdb::Coord p(-200, 45, 67);
        CPPUNIT_ASSERT( iter.getCoord() == p );
        CPPUNIT_ASSERT(grid.tree().isValueOn( p ) );
    }

    {// MaskGrid
        // generate one point
        std::vector<openvdb::Vec3R> points;
        points.push_back( openvdb::Vec3R(-19.999, 4.50001, 6.71) );
        //points.push_back( openvdb::Vec3R( 20,-4.5,-5.2) );
        PointList pointList(points);

        // construct an empty mask grid
        openvdb::MaskGrid grid( false );
        const float voxelSize = 0.1f;
        grid.setTransform( openvdb::math::Transform::createLinearTransform(voxelSize) );
        CPPUNIT_ASSERT( grid.empty() );

        // generate mask from points
        openvdb::tools::PointsToMask<> mask( grid );
        mask.addPoints( pointList );
        CPPUNIT_ASSERT(!grid.empty() );
        CPPUNIT_ASSERT_EQUAL( 1, int(grid.activeVoxelCount()) );
        openvdb::TopologyGrid::ValueOnCIter iter = grid.cbeginValueOn();
        //std::cerr << "Coord = " << iter.getCoord() << std::endl;
        const openvdb::Coord p(-200, 45, 67);
        CPPUNIT_ASSERT( iter.getCoord() == p );
        CPPUNIT_ASSERT(grid.tree().isValueOn( p ) );
    }


    // generate shared transformation
    openvdb::Index64 voxelCount = 0;
    const float voxelSize = 0.001f;
    const openvdb::math::Transform::Ptr xform =
        openvdb::math::Transform::createLinearTransform(voxelSize);

    // generate lots of points
    std::vector<openvdb::Vec3R> points;
    unittest_util::genPoints(15000000, points);
    PointList pointList(points);

    //openvdb::util::CpuTimer timer;
    {// serial BoolGrid
        // construct an empty mask grid
        openvdb::BoolGrid grid( false );
        grid.setTransform( xform );
        CPPUNIT_ASSERT( grid.empty() );

        // generate mask from points
        openvdb::tools::PointsToMask<openvdb::BoolGrid> mask( grid );
        //timer.start("\nSerial BoolGrid");
        mask.addPoints( pointList, 0 );
        //timer.stop();

        CPPUNIT_ASSERT(!grid.empty() );
        //grid.print(std::cerr, 3);
        voxelCount = grid.activeVoxelCount();
    }
    {// parallel BoolGrid
        // construct an empty mask grid
        openvdb::BoolGrid grid( false );
        grid.setTransform( xform );
        CPPUNIT_ASSERT( grid.empty() );

        // generate mask from points
        openvdb::tools::PointsToMask<openvdb::BoolGrid> mask( grid );
        //timer.start("\nParallel BoolGrid");
        mask.addPoints( pointList );
        //timer.stop();

        CPPUNIT_ASSERT(!grid.empty() );
        //grid.print(std::cerr, 3);
        CPPUNIT_ASSERT_EQUAL( voxelCount, grid.activeVoxelCount() );
    }
    {// parallel MaskGrid
        // construct an empty mask grid
        openvdb::MaskGrid grid( false );
        grid.setTransform( xform );
        CPPUNIT_ASSERT( grid.empty() );

        // generate mask from points
        openvdb::tools::PointsToMask<> mask( grid );
        //timer.start("\nParallel MaskGrid");
        mask.addPoints( pointList );
        //timer.stop();

        CPPUNIT_ASSERT(!grid.empty() );
        //grid.print(std::cerr, 3);
        CPPUNIT_ASSERT_EQUAL( voxelCount, grid.activeVoxelCount() );
    }
    {// parallel create TopologyGrid
        //timer.start("\nParallel Create MaskGrid");
        openvdb::MaskGrid::Ptr grid = openvdb::tools::createPointMask(pointList, *xform);
        //timer.stop();

        CPPUNIT_ASSERT(!grid->empty() );
        //grid->print(std::cerr, 3);
        CPPUNIT_ASSERT_EQUAL( voxelCount, grid->activeVoxelCount() );
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
